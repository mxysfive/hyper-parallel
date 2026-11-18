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
"""Workers checking coalesced model broadcasts over Gloo and mixed Gloo/HCCL."""

import os
from datetime import timedelta
from unittest.mock import patch

import torch
import torch.distributed as dist

from hyper_parallel.distributed_data.model_batch import ModelBatchLayout, TensorFieldSpec
from hyper_parallel.distributed_data.topology import DataTopology
from hyper_parallel.distributed_data.transport import DataGroups, ModelParallelTransport


def _batch(device, step):
    boundaries = torch.tensor([0, 7, 12 + step])
    return {
        "tokens": torch.arange(12 + step, device=device, dtype=torch.int64),
        "labels": torch.arange(12 + step, device=device, dtype=torch.int64) + 1,
        "images": [torch.full((3, 256, 256), float(index + step), device=device) for index in range(19 - step)],
        "mask": torch.ones(12 + step, dtype=torch.bool, device=device),
        "features": torch.arange(12, dtype=torch.bfloat16, device=device).reshape(3, 4).t(),
        "empty": torch.empty((step, 0), device=device),
        "scale": torch.tensor(0.25, device=device),
        "cu_seqlens": boundaries,
        "_hp_cpu": {"cu_seqlens": boundaries, "ds_idx": torch.tensor([0, 1])},
        "label": ("train", step, None),
    }


def _check_tree(expected, received):
    layout = ModelBatchLayout(ModelBatchLayout.describe(expected))
    for reference, actual in zip(layout.flatten(expected), layout.flatten(received)):
        assert actual.device == reference.device, f"device={actual.device}, expected={reference.device}"
        assert actual.dtype == reference.dtype, f"dtype={actual.dtype}, expected={reference.dtype}"
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)


def _exercise(device, accelerator_group=None):
    rank, size = dist.get_rank(), dist.get_world_size()
    topology = DataTopology.from_layout((1, size), ("dp", "cp"), tuple(range(size)), rank)
    groups = DataGroups((0,), None, None, dist.group.WORLD, 0, True, accelerator_group)
    transport = ModelParallelTransport(topology, groups, device=device, dynamic_layout=True)
    for step in range(3):
        expected = _batch(device, step)
        with patch("hyper_parallel.distributed_data.transport.dist.broadcast", wraps=dist.broadcast) as broadcast:
            received = transport.broadcast(expected if rank == 0 else None)
        # The dynamic object descriptor uses Gloo's two internal broadcasts;
        # this wrapper counts only direct tensor payload calls from the transport.
        expected_calls = 5 if device.type != "cpu" else 4
        assert broadcast.call_count == expected_calls, (
            f"payload broadcasts={broadcast.call_count}, expected={expected_calls}"
        )
        if accelerator_group is not None:
            device_calls = sum(call.kwargs["group"] is accelerator_group for call in broadcast.call_args_list)
            assert device_calls == 4, f"device broadcasts={device_calls}, expected={4}"
        _check_tree(expected, received)
        assert received["label"] == expected["label"], f"label={received['label']}, expected={expected['label']}"
    assert transport.broadcast(None) is None, f"rank={rank} did not receive EOF"
    with patch("hyper_parallel.distributed_data.transport.dist.broadcast", wraps=dist.broadcast) as broadcast:
        empty = transport.broadcast({"empty": torch.empty(0, device=device)} if rank == 0 else None)
    assert broadcast.call_count == 0, f"empty payload broadcasts={broadcast.call_count}, expected={0}"
    assert empty["empty"].numel() == 0, f"empty shape={empty['empty'].shape}, expected={(0,)}"

    spec = {
        "tokens": TensorFieldSpec((None,), torch.int64),
        "labels": TensorFieldSpec((None,), torch.int64),
        "host": TensorFieldSpec((None,), torch.int64, "cpu"),
    }
    fixed = ModelParallelTransport(topology, groups, device=device, model_batch_spec=spec)
    for length in (0, 13):
        expected = {"tokens": torch.arange(length, device=device),
                    "labels": torch.arange(length, device=device) + 1, "host": torch.arange(length)}
        with patch("hyper_parallel.distributed_data.transport.dist.broadcast", wraps=dist.broadcast) as broadcast:
            received = fixed.broadcast(expected if rank == 0 else None)
        expected_calls = 3 if length else 1
        assert broadcast.call_count == expected_calls, (
            f"fixed broadcasts={broadcast.call_count}, expected={expected_calls}"
        )
        _check_tree(expected, received)


def test_model_broadcast_gloo():
    """Coalesced CPU payloads retain values, structure, dynamic sizes and EOF across processes."""
    dist.init_process_group("gloo", timeout=timedelta(seconds=90))
    try:
        _exercise(torch.device("cpu"))
    finally:
        dist.destroy_process_group()


def test_model_broadcast_hccl():
    """Coalesced NPU payloads and CPU fields use their respective real process groups."""
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group("gloo", timeout=timedelta(seconds=90))
    accelerator_group = dist.new_group(backend="hccl", timeout=timedelta(seconds=90))
    try:
        _exercise(torch.device("npu", local_rank), accelerator_group)
        torch.npu.synchronize()
    finally:
        dist.destroy_process_group(accelerator_group)
        dist.destroy_process_group()
