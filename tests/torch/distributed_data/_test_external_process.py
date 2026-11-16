# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Public external builder coverage with DP constructors and CP consumers."""

import os
from collections.abc import Iterator
from contextlib import closing
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from hyper_parallel.distributed_data import (
    DistributedDatasetConfig, SampleMetadata, WorkloadCost, build_distributed_dataloader,
)


class Source:
    """Finite replayable bins; only even ranks may read them."""

    def __init__(self, rank: int) -> None:
        """Keep rank ownership and an initially empty consumed cursor."""
        self.rank, self.position = rank, 0

    def __iter__(self) -> Iterator[list[list[dict[str, int]]]]:
        """Yield four steps only from constructor ranks."""
        if self.rank % 2:
            raise RuntimeError("A CP peer must never iterate the source")
        while self.position < 4:
            step = self.position
            self.position += 1
            yield [[{"id": self.rank * 100 + step * 2 + index,
                     "tokens": 4 if self.rank == 0 else 1} for index in range(2)]]

    def state_dict(self) -> dict[str, int]:
        """Capture the next unread source position."""
        return {"position": self.position}

    def load_state_dict(self, state: dict[str, int]) -> None:
        """Restore source progress.

        Args:
            state: Previously captured source position.
        """
        self.position = state["position"]

    def set_epoch(self, epoch: int) -> None:
        """Replay the same fixture for every epoch.

        Args:
            epoch: Epoch selected by the test.
        """
        del epoch
        self.position = 0


def _metadata(sample: dict[str, int]) -> SampleMetadata:
    """Estimate attention work while retaining the packed footprint."""
    return SampleMetadata(sample["tokens"], cost=WorkloadCost(llm=sample["tokens"] ** 2))


def _pack(samples: list[dict[str, int]], seq_len: int) -> dict[str, Any]:
    """Produce CPU tensors with source identities and producer ownership."""
    del seq_len
    return {"ids": torch.tensor([sample["id"] for sample in samples]),
            "cu_seqlens": torch.arange(len(samples) + 1), "producer_pid": os.getpid()}


def _move(batch: list[dict[str, Any]], device: torch.device) -> list[dict[str, Any]]:
    """Move model inputs while retaining CPU sequence boundaries."""
    return [{**item, "ids": item["ids"].to(device, non_blocking=True)} for item in batch]


def _exercise(backend: str) -> None:
    """Exercise the public builder, exact membership and pending recovery."""
    device = torch.device("npu", int(os.environ["LOCAL_RANK"])) if backend == "hccl" else torch.device("cpu")
    if device.type == "npu":
        torch.npu.set_device(device)
    dist.init_process_group(backend, timeout=timedelta(seconds=120))
    mesh = init_device_mesh(device.type, (2, 2), mesh_dim_names=("dp", "cp"))
    try:
        for broadcast_plan in (True, False):
            config = DistributedDatasetConfig(seq_len=32, local_batch_size=1, communication_backend=backend,
                                              broadcast_plan=broadcast_plan, balance_group_size=2)
            with closing(build_distributed_dataloader(
                None, mesh, config, external_step_source=Source(dist.get_rank()),
                metadata_fn=_metadata, pack_fn=_pack, cost_model=lambda item: item.cost,
                device=device, batch_to_device_fn=_move,
            )) as loader:
                first = next(loader)[0]
                assert first["producer_pid"] != os.getpid(), "CPU work ran in the training process"
                assert first["cu_seqlens"].device.type == "cpu", "CPU metadata placement changed"
                assert first["ids"].device == device, "Model inputs were not transferred"
                if dist.get_rank() % 2 == 0:
                    assert loader.last_balance_stats["moved_samples"] > 0, "Expected a beneficial redistribution"
                checkpoint = loader.state_dict()
                expected = [batch[0]["ids"].cpu().tolist() for batch in loader]
                loader.load_state_dict(checkpoint)
                loader.set_epoch(0)
                actual = [batch[0]["ids"].cpu().tolist() for batch in loader]
                assert actual == expected, f"Resume lost or duplicated samples: {actual} != {expected}"
                all_ids = [None] * 4
                # The test uses a separate CPU group for assertions only.
                validation = dist.new_group(backend="gloo")
                dist.all_gather_object(all_ids, [first["ids"].cpu().tolist(), *actual], group=validation)
                assert all_ids[0] == all_ids[1] and all_ids[2] == all_ids[3], "CP peers differ"
                for step, (left, right) in enumerate(zip(all_ids[0], all_ids[2])):
                    assert sorted(left + right) == [step * 2, step * 2 + 1, 200 + step * 2, 201 + step * 2]
                dist.destroy_process_group(validation)
                loader.set_epoch(1)
                torch.testing.assert_close(next(loader)[0]["ids"], first["ids"], rtol=0, atol=0)
        if device.type == "npu":
            torch.npu.synchronize()
    finally:
        dist.destroy_process_group()


def test_external_process_gloo():
    """Verify source ownership, balance, CP placement, pending resume and epoch."""
    _exercise("gloo")


def test_external_process_hccl():
    """Run the same public builder and recovery checks over real NPU collectives."""
    _exercise("hccl")
