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
"""Distributed CPU test for applying non-fused FSDP gradients before the root hook."""

import copy
import os
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.checkpoint import checkpoint

from hyper_parallel import init_device_mesh
from hyper_parallel.core.fully_shard.api import fully_shard
from hyper_parallel.core.fully_shard.hsdp_param import HSDPParamV2
from hyper_parallel.core.fully_shard.hsdp_scheduler import HSDPSchedulerV2


class _Network(nn.Module):
    """Four nested units with the activation-checkpointing option used by VeOmni."""

    def __init__(self, use_checkpoint: bool) -> None:
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(8, 8, bias=False) for _ in range(4))
        self.use_checkpoint = use_checkpoint

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            inputs = checkpoint(layer, inputs, use_reentrant=False) if self.use_checkpoint else layer(inputs)
        return inputs


def _check_early_grad_accumulation(device_type: str, use_checkpoint: bool) -> None:
    """Compare four synchronized microsteps with a distributed eager reference."""
    if device_type == "cuda":
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    if not dist.is_initialized():
        dist.init_process_group("nccl" if device_type == "cuda" else "gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    mesh = init_device_mesh(device_type, (world_size,), mesh_dim_names=("fsdp",))
    torch.manual_seed(1234)
    model = _Network(use_checkpoint).to(device)
    reference = copy.deepcopy(model)
    root_active = False
    early_applies = []
    original_root = HSDPSchedulerV2._root_backward_hook
    original_apply = HSDPParamV2.apply_reduced_grad

    def traced_root(scheduler):
        nonlocal root_active
        root_active = True
        try:
            return original_root(scheduler)
        finally:
            root_active = False

    def traced_apply(param, grad):
        if not root_active:
            early_applies.append(param)
        return original_apply(param, grad)

    # Public fully_shard accepts accelerator meshes only; supply the requested
    # device to exercise the same scheduler with Gloo in the CPU variant.
    with patch("hyper_parallel.core.fully_shard.api._get_device_from_mesh", return_value=device), \
         patch.object(HSDPSchedulerV2, "_root_backward_hook", traced_root), \
         patch.object(HSDPParamV2, "apply_reduced_grad", traced_apply):
        for layer in model.layers:
            fully_shard(layer, mesh=mesh, comm_fusion=False)
        fully_shard(model, mesh=mesh, comm_fusion=False)
        model.set_reduce_op_type("sum")
        model.set_reshard_after_backward(True)

        for iteration in range(2):
            model.zero_grad()
            reference.zero_grad(set_to_none=True)
            for micro in range(4):
                inputs = torch.arange(16, dtype=torch.float32, device=device).view(2, 8) / 64
                inputs = inputs + (iteration + micro + rank) / 16
                model(inputs).square().sum().backward()
                reference(inputs).square().sum().backward()
                for (_, sharded), (_, eager) in zip(model.named_parameters(), reference.named_parameters()):
                    expected = eager.grad.detach().clone()
                    dist.all_reduce(expected, op=dist.ReduceOp.SUM)
                    expected_shard = torch.chunk(expected, world_size, dim=0)[rank]
                    torch.testing.assert_close(sharded.grad._local_tensor, expected_shard, rtol=1e-5, atol=1e-6)

    assert len(early_applies) >= 2 * 4, "Expected at least one gradient per microstep before the root hook"


def test_early_grad_accumulation_matches_distributed_eager() -> None:
    """Check two-rank CPU/Gloo accumulation with and without checkpointing."""
    for use_checkpoint in (False, True):
        _check_early_grad_accumulation("cpu", use_checkpoint)


def test_early_grad_accumulation_matches_distributed_eager_cuda() -> None:
    """Check CUDA/NCCL accumulation under an allocated GPU job."""
    for use_checkpoint in (False, True):
        _check_early_grad_accumulation("cuda", use_checkpoint)
