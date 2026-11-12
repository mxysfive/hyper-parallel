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
"""Checkpoint coverage for externally owned sources and speculative packing."""

import copy
import random
import unittest
from collections.abc import Iterator
from contextlib import closing
from types import SimpleNamespace
from unittest.mock import PropertyMock, patch

from hyper_parallel.distributed_data import (
    DistributedDatasetConfig,
    SampleMetadata,
    build_distributed_dataloader,
)
from hyper_parallel.distributed_data.transport import DataPlaneTransport


class _Mesh:
    mesh_shape = (1,)
    mesh_dim_names = ("dp",)
    rank_list = (0,)


class _StatefulSource:
    def __init__(self) -> None:
        """Create an independent source RNG and cursor."""
        self.position = 0
        self.rng = random.Random(24)

    def __iter__(self) -> Iterator[list[list[dict]]]:
        """Yield bounded complete steps without resetting restored progress."""
        while self.position < 5:
            sample = {"id": self.position, "tokens": self.rng.randint(1, 8)}
            self.position += 1
            yield [[sample]]

    def state_dict(self) -> dict:
        """Snapshot the source cursor and its owned random state."""
        return {"position": self.position, "rng": self.rng.getstate()}

    def load_state_dict(self, state: dict) -> None:
        """Restore source progress before constructing an iterator."""
        self.position = state["position"]
        self.rng.setstate(state["rng"])

    def set_epoch(self, epoch: int) -> None:
        """Reset source progress for another epoch."""
        self.position = 0
        self.rng.seed(24 + epoch)


def _build(source, mesh=None, **kwargs):
    return build_distributed_dataloader(
        None, _Mesh() if mesh is None else mesh, DistributedDatasetConfig(
            seq_len=16, local_batch_size=1, communication_backend="gloo", **kwargs,
        ), external_step_source=source,
        metadata_fn=lambda sample: SampleMetadata(sample["tokens"]),
        cost_model=lambda metadata: metadata.cost,
        device="cpu",
    )


class TestExternalResume(unittest.TestCase):
    """Verify delivered cursors rather than speculative source cursors."""

    def test_resume_every_pipeline_phase(self):
        """Feature: External checkpoint replay.
        Description: Save before iteration, at every prefetch phase, and at exhaustion.
        Expectation: Remaining batches, RNG draws and delivered steps match a continuous run.
        """
        with closing(_build(_StatefulSource())) as baseline:
            expected = list(baseline)
        for backend in ("gloo", "hccl"):
            for consumed in (0, 1, 4, 5):
                for phase in ("source", "plan", "batch"):
                    with self.subTest(backend=backend, consumed=consumed, phase=phase), patch.object(
                            DataPlaneTransport, "communication_backend", new_callable=PropertyMock,
                            return_value=backend,
                    ), closing(_build(_StatefulSource())) as loader:
                        for _ in range(consumed):
                            next(loader)
                        if phase == "plan":
                            loader.prefetch_plan()
                        elif phase == "batch":
                            loader.prefetch()
                        state = loader.state_dict()
                        saved = copy.deepcopy(state)
                        self.assertEqual(list(loader), expected[consumed:])
                        self.assertEqual(loader.step, len(expected))
                        self.assertEqual(state, saved)
                        with closing(_build(_StatefulSource())) as restored:
                            restored.load_state_dict(state)
                            self.assertEqual(restored.step, consumed)
                            restored.set_epoch(0)
                            iter(restored)
                            restored.set_epoch(0)
                            self.assertEqual(list(restored), expected[consumed:])
                            self.assertEqual(restored.step, len(expected))
                            self.assertEqual(list(restored), [])
                            restored.set_epoch(1)
                            self.assertEqual(len(list(restored)), len(expected))

    def test_checkpoint_is_independent_and_validates_config(self):
        """Feature: Checkpoint isolation.
        Description: Mutate a saved pending batch and restore with a different packing limit.
        Expectation: Live batches stay intact and incompatible configuration is rejected.
        """
        with closing(_build(_StatefulSource())) as loader:
            next(loader)
            state = loader.state_dict()
            pending = state["iterator"]["pending"][0]
            pending[0][0]["id"] = -1
            self.assertEqual(next(loader)[0][0]["id"], 1)
        with closing(_build(_StatefulSource(), seed=123)) as restored:
            with self.assertRaisesRegex(ValueError, "configuration"):
                restored.load_state_dict(state)

    def test_reenter_preserves_pending_batch(self):
        """Feature: Iterator reentry.
        Description: Reenter iteration after a batch has triggered speculative input loading.
        Expectation: No skipped batch and no reset of the delivered cursor.
        """
        with closing(_build(_StatefulSource())) as loader:
            next(loader)
            iter(loader)
            iter(loader)
            self.assertEqual(loader.step, 1)
            self.assertEqual([batch[0][0]["id"] for batch in loader], [1, 2, 3, 4])

    def test_rejects_changed_mesh_with_same_reader_group(self) -> None:
        """Feature: External checkpoint topology validation.
        Description: Restore under a differently named root mesh with the same reader ranks.
        Expectation: Restoring fails before applying source state.
        """
        with closing(_build(_StatefulSource())) as loader:
            state = loader.state_dict()
        other = SimpleNamespace(mesh_shape=(1, 1), mesh_dim_names=("dp", "cp"), rank_list=(0,))
        with closing(_build(_StatefulSource(), mesh=other)) as restored:
            with self.assertRaisesRegex(ValueError, "topology"):
                restored.load_state_dict(state)
