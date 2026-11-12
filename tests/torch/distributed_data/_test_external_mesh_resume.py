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
"""Mesh input replication and exact external-source replay on CPU and NPU."""

import io
import os
from collections.abc import Iterator
from contextlib import closing
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh as TorchDeviceMesh

from hyper_parallel import DeviceMesh
from hyper_parallel.distributed_data import (
    DistributedDatasetConfig,
    SampleMetadata,
    WorkloadCost,
    build_distributed_dataloader,
)
from hyper_parallel.distributed_data.transport import all_gather_control_object


class _Source:
    """Stateful producer whose samples expose both routing and variable tensor layouts."""

    def __init__(self, rank: int) -> None:
        """Create a rank-local synthetic source."""
        self.rank = rank
        self.position = 0
        self.reads = 0

    def __len__(self) -> int:
        """Return the agreed number of optimizer steps."""
        return 4

    def __iter__(self) -> Iterator[list[list[dict]]]:
        """Read samples on constructors and reject reads on model peers."""
        if self.rank % 2:
            raise AssertionError("CP peers must never iterate the external source")
        while self.position < len(self):
            step = self.position
            self.position += 1
            self.reads += 1
            yield [[{
                "id": step * 4 + self.rank + offset,
                "cost": 9 if self.rank == 0 else 1,
                "pixels": torch.full((step + 1, offset + 1), step * 4 + self.rank + offset, dtype=torch.bfloat16),
            } for offset in range(2)]]

    def state_dict(self) -> dict:
        """Snapshot the source cursor and its owned random state."""
        return {"position": self.position}

    def load_state_dict(self, state: dict) -> None:
        """Restore source progress before constructing an iterator."""
        self.position = state["position"]

    def set_epoch(self, epoch: int) -> None:
        """Reset source progress for another epoch."""
        del epoch
        self.position = 0


def _pack(samples, seq_len):
    del seq_len
    return {
        "ids": torch.tensor([sample["id"] for sample in samples]),
        "pixels": [sample["pixels"] for sample in samples],
        "empty": torch.empty((0, 3)),
        "optional": None,
        "tag": ("images", True, len(samples)),
    }


def _host_values(batch):
    if isinstance(batch, torch.Tensor):
        return (str(batch.dtype), tuple(batch.shape), batch.cpu().tolist())
    if isinstance(batch, dict):
        return {key: _host_values(value) for key, value in batch.items()}
    if isinstance(batch, (tuple, list)):
        return type(batch)(_host_values(value) for value in batch)
    return batch


def _build(mesh, backend, device):
    source = _Source(dist.get_rank())
    loader = build_distributed_dataloader(
        None, mesh, DistributedDatasetConfig(
            seq_len=8, local_batch_size=1, dp_dim_names=("dp",), balance_group_size=2,
            communication_backend=backend,
        ), external_step_source=source,
        metadata_fn=lambda sample: SampleMetadata(2, cost=WorkloadCost(llm=sample["cost"])),
        cost_model=lambda metadata: metadata.cost, pack_fn=_pack, collate_fn=list, device=device,
    )
    return loader, source


def _consume(loader, device, backend, start, end):
    outputs = []
    for step in range(start, end):
        batch = next(loader)
        assert batch[0]["ids"].device == device
        values = _host_values(batch)
        gathered = all_gather_control_object(values, group=None, device=device, backend=backend)
        assert gathered[0] == gathered[1] and gathered[2] == gathered[3], gathered
        ids = sorted(gathered[0][0]["ids"][2] + gathered[2][0]["ids"][2])
        assert ids == list(range(step * 4, (step + 1) * 4)), ids
        if dist.get_rank() % 2 == 0:
            stats = loader.last_balance_stats
            assert stats["moved_samples"] > 0, stats
            assert stats["max_cost_after"] < stats["max_cost_before"], stats
        assert loader.step == step + 1
        outputs.append(values)
        loader.prefetch_plan()
        loader.prefetch()
    return outputs


def _run_case(mesh_type, backend, device):
    # Non-monotonic mesh coordinates catch confusion with process-group rank order.
    mesh = mesh_type(device.type, [[2, 3], [0, 1]], mesh_dim_names=("dp", "cp"), _init_backend=False)
    loader, source = _build(mesh, backend, device)
    with closing(loader):
        _consume(loader, device, backend, 0, 1)
        iter(loader)
        assert loader.step == 1
        state = loader.state_dict()
        assert source.reads == (2 if dist.get_rank() % 2 == 0 else 0)
        checkpoint = io.BytesIO()
        torch.save(state, checkpoint)
        expected = _consume(loader, device, backend, 1, 4)
        assert not list(loader)
        final_state = loader.state_dict()
    restored, restored_source = _build(mesh, backend, device)
    with closing(restored):
        checkpoint.seek(0)
        restored.load_state_dict(torch.load(checkpoint, weights_only=False))
        restored.set_epoch(0)
        iter(restored)
        restored.set_epoch(0)
        assert restored.step == 1
        actual = _consume(restored, device, backend, 1, 4)
        assert actual == expected, (actual, expected)
        assert not list(restored)
        assert restored_source.reads == (2 if dist.get_rank() % 2 == 0 else 0)
        restored.load_state_dict(final_state)
        assert not list(restored)
        assert restored.step == 4
        restored.set_epoch(1)
        _consume(restored, device, backend, 0, 1)


def test_mesh_resume_gloo() -> None:
    """Verify real Torch and HP DP2/CP2 meshes, sample routing and exact checkpoint replay."""
    dist.init_process_group("gloo", timeout=timedelta(seconds=90))
    try:
        for mesh_type in (TorchDeviceMesh, DeviceMesh):
            _run_case(mesh_type, "gloo", torch.device("cpu"))
    finally:
        dist.destroy_process_group()


def test_mesh_resume_hccl() -> None:
    """Verify NPU tensor delivery and foreground HCCL prefetch/checkpoint phases."""
    device = torch.device("npu", int(os.environ["LOCAL_RANK"]))
    torch.npu.set_device(device)
    dist.init_process_group("hccl", timeout=timedelta(seconds=120))
    try:
        for mesh_type in (TorchDeviceMesh, DeviceMesh):
            _run_case(mesh_type, "hccl", device)
    finally:
        dist.destroy_process_group()
