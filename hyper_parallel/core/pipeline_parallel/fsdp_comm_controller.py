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
"""Own pipeline synchronization boundaries without moving HSDP communication queues."""
from collections.abc import Sequence

from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.fully_shard.hsdp_state import HSDPStateV2
from hyper_parallel.core.fully_shard.param_group import AllReduceParamGroup
from hyper_parallel.core.pipeline_parallel._stage import PipelineStageBase


class FSDPCommController:
    """Coordinate stage-local HSDP gradients on the caller's current device stream."""

    def __init__(self, stages: Sequence[PipelineStageBase], micro_batch_num: int) -> None:
        """Record the stage-owned HSDP units and their settings."""
        if micro_batch_num <= 0:
            raise ValueError("micro_batch_num must be positive")
        self.micro_batch_num = micro_batch_num
        self.stage_schedulers = {}
        self.stage_contexts = {}
        self.user_settings = {}
        self.current_settings = {}
        self.context_settings = {}
        self.completed_microbatches = {stage.stage_index: set() for stage in stages}
        if len(self.completed_microbatches) != len(stages):
            raise ValueError("Local pipeline stages must have distinct stage indices")
        self.dx_microbatches = {stage.stage_index: set() for stage in stages}
        self.reduction_tail_stage = None
        self.retained_backward_units = set()
        self._validate_stage_ownership(stages)
        for stage in stages:
            schedulers = stage.all_hsdp_schedulers
            if not schedulers:
                continue
            context = schedulers[0].scheduler_ctx
            self.stage_schedulers[stage.stage_index] = schedulers
            self.stage_contexts[stage.stage_index] = context
            self.context_settings[stage.stage_index] = context.defer_grad_finalize
            for scheduler in schedulers:
                settings = {
                    "reshard_after_forward": scheduler.reshard_after_forward,
                    "reshard_after_backward": scheduler.hsdp_state.reshard_after_backward,
                    "requires_gradient_sync": scheduler.hsdp_state.reduce_grads,
                    "requires_all_reduce": scheduler.hsdp_state.requires_all_reduce,
                }
                self.user_settings[scheduler] = settings
                self.current_settings[scheduler] = settings.copy()

    @staticmethod
    def _validate_stage_ownership(stages: Sequence[PipelineStageBase]) -> None:
        """Ensure a scheduler or mutable parameter is owned by one stage."""
        scheduler_owners = set()
        wrapper_owners = set()
        parameter_owners = {}
        storage_owners = {}
        for stage in stages:
            for scheduler in stage.all_hsdp_schedulers:
                if scheduler in scheduler_owners:
                    raise ValueError("Pipeline stages must not share an HSDP scheduler")
                scheduler_owners.add(scheduler)
                for param in scheduler.hsdp_state.hsdp_params:
                    if param in wrapper_owners:
                        raise ValueError("Pipeline stages must not share HSDP parameter wrappers")
                    wrapper_owners.add(param)

            # Storage identity also catches distinct Parameters created as views.
            for param in stage.submodule.parameters():
                if param in parameter_owners and parameter_owners[param] != stage.stage_index:
                    raise ValueError("Pipeline stages must not share parameter objects")
                parameter_owners[param] = stage.stage_index
                local_param = param.to_local() if isinstance(param, DTensor) else param
                storage = local_param.untyped_storage()
                if storage in storage_owners and storage_owners[storage] != stage.stage_index:
                    raise ValueError("Pipeline stages must not share mutable parameter storage")
                storage_owners[storage] = stage.stage_index

    def begin(self) -> None:
        """Take ownership of module gradient processing until successful finalization.

        An interrupted run keeps its handler and communication state installed;
        starting another run must not overwrite that outstanding work.
        """
        if (
            any(self.completed_microbatches.values())
            or any(self.dx_microbatches.values())
            or self.reduction_tail_stage is not None
            or any(context.post_backward_handler is not None
                   or context.defer_grad_finalize != self.context_settings[stage_index]
                   for stage_index, context in self.stage_contexts.items())
        ):
            raise RuntimeError("The previous pipeline run has unfinished HSDP work")
        for context in self.stage_contexts.values():
            context.defer_grad_finalize = True
            context.post_backward_handler = self.post_backward

    @staticmethod
    def accumulate_grad_for_all_params_if_needed(hsdp_state: HSDPStateV2) -> None:
        """Accumulate full gradients and retain them locally during no-sync steps."""
        for param in hsdp_state.hsdp_params:
            param.accumulate_unsharded_grad_if_needed()
        if not hsdp_state.reduce_grads:
            for param in hsdp_state.hsdp_params:
                param.to_accumulated_grad_if_needed()

    @staticmethod
    def _wait_reduce_scatter(hsdp_state: HSDPStateV2, keep_units: int = 0) -> list[AllReduceParamGroup]:
        """Consume old ReduceScatter slots and return their ready replica groups."""
        groups = []
        while len(hsdp_state.scheduler_ctx.pre_reduce_scatter_params) > keep_units:
            groups.extend(hsdp_state._wait_prev_reduce_scatter())
            hsdp_state._wait_prev_reduce_scatter_without_all_reduce()
        return groups

    def post_backward(self, hsdp_state: HSDPStateV2) -> None:
        """Advance gradient communication when a parameter-owning unit finishes."""
        self.accumulate_grad_for_all_params_if_needed(hsdp_state)
        if hsdp_state.reshard_after_backward:
            hsdp_state.shard()
        if not hsdp_state.reduce_grads or not hsdp_state.hsdp_params:
            return

        # A later microbatch can reach the same unit before the interval expires.
        # Its parameters need their previous slots released before another submission.
        keep_units = hsdp_state.scheduler_ctx.reduce_interval - 1
        if any(param.reduce_scatter_comm_ctx.reduce_scatter_output is not None for param in hsdp_state.hsdp_params):
            keep_units = 0
        previous_groups = self._wait_reduce_scatter(hsdp_state, keep_units)
        hsdp_state._issue_reduce_scatter_for_current_module()
        hsdp_state._issue_prev_fused_all_reduce(previous_groups)

    def reduce_dw_gradients(self, stage_index: int) -> None:
        """Consume old work, then submit every ready unit without waiting on this batch."""
        schedulers = self.stage_schedulers.get(stage_index)
        if not schedulers:
            return

        old_groups = []
        for old_stage_index in dict.fromkeys((self.reduction_tail_stage, stage_index)):
            old_schedulers = self.stage_schedulers.get(old_stage_index)
            if old_schedulers:
                state = old_schedulers[0].hsdp_state
                old_groups.append((state, self._wait_reduce_scatter(state)))
        for scheduler in schedulers:
            state = scheduler.hsdp_state
            self.accumulate_grad_for_all_params_if_needed(state)
            if state.reshard_after_backward:
                state.shard()
        for scheduler in schedulers:
            state = scheduler.hsdp_state
            if state.reduce_grads and state.hsdp_params:
                state._issue_reduce_scatter_for_current_module()
        for state, groups in old_groups:
            state._issue_prev_fused_all_reduce(groups)
        self.reduction_tail_stage = stage_index

    def flush_reduce_scatter(self, stage_index: int) -> None:
        """Consume only this stage's original queue and submit its ready AllReduce work."""
        schedulers = self.stage_schedulers.get(stage_index)
        if schedulers:
            schedulers[0].flush_reduce_scatter()

    def wait_all_reduce(self, stage_index: int) -> None:
        """Wait already-submitted AllReduce work without launching ReduceScatter."""
        schedulers = self.stage_schedulers.get(stage_index)
        if schedulers:
            schedulers[0].wait_all_reduce()

    def finalize(self) -> None:
        """Finish a successful run, restore user settings, and preserve optimizer gradients.

        Failures leave outstanding state visible and propagate to the caller.
        Callbacks must have joined their compute threads.
        """
        if self.retained_backward_units:
            raise RuntimeError("Overlap callbacks must release retained units")
        for stage_index in self.stage_schedulers:
            if self.dx_microbatches[stage_index]:
                raise RuntimeError(f"Stage {stage_index} has unfinished weight backwards")
            if self.completed_microbatches[stage_index] != set(range(self.micro_batch_num)):
                raise RuntimeError(f"Stage {stage_index} did not complete every backward microbatch")

        # Releasing old inputs precedes any attempt to reuse their slots. This
        # also exposes historical shards for parameters unused in the last graph.
        for stage_index in self.stage_contexts:
            self.flush_reduce_scatter(stage_index)
        for schedulers in self.stage_schedulers.values():
            for scheduler in schedulers:
                state = scheduler.hsdp_state
                self.accumulate_grad_for_all_params_if_needed(state)
                state._issue_reduce_scatter_for_current_module()
        for stage_index in self.stage_contexts:
            self.flush_reduce_scatter(stage_index)
        for stage_index in self.stage_contexts:
            self.wait_all_reduce(stage_index)
        for stage_index, schedulers in self.stage_schedulers.items():
            schedulers[0]._apply_reduced_gradients()
            for scheduler in schedulers:
                scheduler.hsdp_state.shard()
                scheduler.reset_iter_state()
                settings = self.user_settings[scheduler]
                scheduler.set_reshard_after_forward(settings["reshard_after_forward"])
                scheduler.set_reshard_after_backward(settings["reshard_after_backward"])
                scheduler.set_requires_grad_sync(settings["requires_gradient_sync"])
                scheduler.set_requires_all_reduce(settings["requires_all_reduce"])
                self.current_settings[scheduler] = settings.copy()
            self.stage_contexts[stage_index].defer_grad_finalize = self.context_settings[stage_index]
            self.stage_contexts[stage_index].post_backward_handler = None
        for completed in self.completed_microbatches.values():
            completed.clear()
        self.reduction_tail_stage = None
