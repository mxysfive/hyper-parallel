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
"""Deliver one external reader's batches to its model-parallel mesh peers."""

from __future__ import annotations

from typing import Any

from hyper_parallel.distributed_data.topology import DataTopology
from hyper_parallel.distributed_data.transport import ModelParallelTransport


class MeshBalancingDataLoader:
    """Keep source iteration and prefetch on constructors while every mesh rank consumes.

    Checkpoints are rank-local and require the same named root mesh on restore.
    All ranks call iteration, checkpoint and epoch methods at matching boundaries.
    """

    def __init__(
            self, source: Any, reader: Any, topology: DataTopology,
            transport: ModelParallelTransport,
    ) -> None:
        """Bind the optional constructor loader and the common model transport."""
        self._source = source
        self._reader = reader
        self._topology = topology
        self._transport = transport
        self._step = 0
        self._finished = False
        self._resuming = False
        self._epoch = 0
        self.last_balance_stats = None
        self.last_host_batch = None

    @property
    def step(self) -> int:
        """Return the number of complete local steps delivered to training."""
        return self._step

    def __len__(self) -> int:
        """Return the source's declared number of complete local steps."""
        return len(self._source)

    def __iter__(self) -> MeshBalancingDataLoader:
        """Start or resume collective consumption without advancing model peers' sources."""
        if self._reader is not None:
            iter(self._reader)
        return self

    def __next__(self) -> Any:
        """Broadcast a constructed batch, EOF, or source failure to model peers."""
        if self._finished:
            raise StopIteration
        batch, error = None, None
        if self._reader is not None:
            try:
                batch = next(self._reader)
            except StopIteration:
                pass
            except BaseException as exc:  # pylint: disable=broad-exception-caught
                error = exc
        batch = self._transport.broadcast(batch, error=error)
        if batch is None:
            self._finished = True
            raise StopIteration
        self._step += 1
        self._resuming = False
        if self._reader is not None:
            self.last_balance_stats = self._reader.last_balance_stats
            self.last_host_batch = self._reader.last_host_batch
        return batch

    def prefetch_plan(self) -> None:
        """Launch reader metadata collectives at a shared training boundary."""
        if self._reader is not None:
            self._reader.prefetch_plan()

    def prefetch(self) -> None:
        """Launch reader payload exchange while current-step computation is queued."""
        if self._reader is not None:
            self._reader.prefetch()

    def wait_for_prefetch(self) -> None:
        """Wait for outstanding reader work without consuming the pending step."""
        if self._reader is not None:
            self._reader.wait_for_prefetch()

    def set_epoch(self, epoch: int) -> None:
        """Forward epoch transitions only to the source-owning constructor.

        Args:
            epoch: Source epoch to start, preserving a just-restored same epoch.
        """
        if self._resuming and epoch == self._epoch:
            return
        if self._reader is not None:
            self._reader.set_epoch(epoch)
        self._epoch = epoch
        self._resuming = False
        self._step = 0
        self._finished = False
        self.last_balance_stats = None
        self.last_host_batch = None

    def _checkpoint_topology(self) -> tuple:
        topology = self._topology
        return (topology.mesh_shape, topology.mesh_dim_names, topology.rank_list,
                topology.dp_dim_names, topology.global_rank)

    def state_dict(self) -> dict[str, Any]:
        """Save each rank's consumption state and each constructor's pending input."""
        return {
            "topology": self._checkpoint_topology(), "step": self._step, "finished": self._finished,
            "epoch": self._epoch,
            "reader": self._reader.state_dict() if self._reader is not None else None,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore a rank-local checkpoint under the original mesh.

        Args:
            state: This rank's loader state saved at a completed optimizer step.
        """
        if state["topology"] != self._checkpoint_topology():
            raise ValueError("External loader checkpoint requires the same named mesh and rank placement.")
        if self._reader is not None:
            self._reader.load_state_dict(state["reader"])
        elif state["reader"] is not None:
            raise ValueError("A model peer cannot restore a constructor's source state.")
        self._step, self._finished = state["step"], state["finished"]
        self._epoch = state["epoch"]
        self._resuming = True
        self.last_balance_stats = None
        self.last_host_batch = None

    def close(self) -> None:
        """Drain reader prefetch before releasing the iterator."""
        if self._reader is not None:
            self._reader.close()
        self._finished = True
        self.last_host_batch = None


__all__ = ["MeshBalancingDataLoader"]
