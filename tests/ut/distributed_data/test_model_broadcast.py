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
"""Coalesced model-input broadcasts with simulated constructor/consumer communication."""

import unittest
from unittest.mock import patch

import torch

from hyper_parallel.distributed_data.model_batch import ModelBatchLayout, TensorFieldSpec
from hyper_parallel.distributed_data.topology import DataTopology
from hyper_parallel.distributed_data.transport import DataGroups, ModelParallelTransport
from tests.common.mark_utils import arg_mark


def _transport(rank, spec=None, size=2):
    topology = DataTopology.from_layout((1, size), ("dp", "cp"), tuple(range(size)), rank)
    groups = DataGroups((0,), "readers", "readers", "model", 0, True)
    return ModelParallelTransport(topology, groups, device="cpu", model_batch_spec=spec, dynamic_layout=spec is None)


class TestModelBroadcast(unittest.TestCase):
    """Check collective counts, exact values, placement declarations and independent storage."""

    def _roundtrip(self, batch, spec=None):
        payloads = []
        with (
            patch("hyper_parallel.distributed_data.transport.dist.get_backend", return_value="gloo"),
            patch("hyper_parallel.distributed_data.transport.broadcast_control_object",
                  side_effect=lambda value, **kwargs: value) as control,
            patch("hyper_parallel.distributed_data.transport.dist.broadcast",
                  side_effect=lambda tensor, **kwargs: payloads.append(tensor.clone())),
        ):
            sent = _transport(0, spec).broadcast(batch)
            descriptor = control.call_args.args[0] if control.called else None
        pending = iter(payloads)
        with (
            patch("hyper_parallel.distributed_data.transport.dist.get_backend", return_value="gloo"),
            patch("hyper_parallel.distributed_data.transport.broadcast_control_object", return_value=descriptor),
            patch("hyper_parallel.distributed_data.transport.dist.broadcast",
                  side_effect=lambda tensor, **kwargs: tensor.copy_(next(pending))) as receive,
        ):
            received = _transport(1, spec).broadcast(None)
        self.assertEqual(receive.call_count, len(payloads))
        if batch is not None:
            layout = ModelBatchLayout(spec if spec is not None else ModelBatchLayout.describe(batch))
            for original, result in zip(layout.flatten(batch), layout.flatten(received)):
                self.assertEqual(result.device.type, "cpu")
                self.assertTrue(torch.equal(
                    original.contiguous().reshape(-1).view(torch.uint8),
                    result.contiguous().reshape(-1).view(torch.uint8),
                ))
        return sent, received, payloads

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_nested_multimodal_batch_uses_one_payload_per_dtype(self):
        """Feature: Coalescing.
        Description: Mixed nested fields.
        Expectation: Four exact typed payloads.
        """
        batch = {
            "tokens": torch.tensor([2**60 + 3, -7]),
            "images": [torch.arange(12, dtype=torch.float32).reshape(3, 4).t() for _ in range(19)],
            "special": torch.tensor([float("nan"), float("inf"), -0.0]),
            "extra": (torch.tensor(True), torch.tensor([1.5, -2.0], dtype=torch.bfloat16)),
            "empty": torch.empty((2, 0, 3)),
            "label": "image",
            "optional": None,
        }
        _, received, payloads = self._roundtrip(batch)
        self.assertEqual(len(payloads), 4)
        self.assertIsInstance(received["extra"], tuple)
        self.assertEqual(received["label"], "image")
        self.assertIsNone(received["optional"])
        self.assertEqual(received["empty"].shape, (2, 0, 3))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_repeated_cpu_tensor_is_transmitted_once(self):
        """Feature: Deduplication.
        Description: Repeated cu_seqlens.
        Expectation: One copy on the wire.
        """
        boundaries = torch.tensor([0, 7, 12])
        batch = {"cu_seqlens": boundaries, "_hp_cpu": {"cu_seqlens": boundaries, "ds_idx": torch.tensor([1, 2])}}
        sent, received, payloads = self._roundtrip(batch)
        self.assertEqual(len(payloads), 1)
        self.assertEqual(payloads[0].numel(), 5)
        self.assertIs(sent["cu_seqlens"], boundaries)
        received["cu_seqlens"].fill_(99)
        self.assertTrue(torch.equal(received["_hp_cpu"]["cu_seqlens"], boundaries))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_storage_views_are_not_mistaken_for_repeated_tensors(self):
        """Feature: Deduplication.
        Description: Distinct views share storage.
        Expectation: Both values survive.
        """
        storage = torch.arange(8)
        _, received, payloads = self._roundtrip([storage[:4], storage[4:]])
        self.assertEqual(payloads[0].numel(), 8)
        self.assertNotEqual(received[0].untyped_storage().data_ptr(), received[1].untyped_storage().data_ptr())

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_empty_batches_and_eof_do_not_send_tensor_payloads(self):
        """Feature: Empty inputs.
        Description: Empty leaves, scalars and EOF.
        Expectation: No tensor collective.
        """
        for batch in ([], {"empty": torch.empty((0, 4)), "other": torch.empty(0)}, {"step": 3}, None):
            with self.subTest(batch=batch):
                _, received, payloads = self._roundtrip(batch)
                self.assertEqual(payloads, [])
                if batch is None:
                    self.assertIsNone(received)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_fixed_layout_coalesces_and_resolves_dynamic_shapes(self):
        """Feature: Fixed layouts.
        Description: Variable sizes.
        Expectation: Header plus two typed payloads.
        """
        spec = {
            "tokens": TensorFieldSpec((None,), torch.int64, "cpu"),
            "labels": TensorFieldSpec((None,), torch.int64, "cpu"),
            "weight": TensorFieldSpec((), torch.float32, "cpu"),
            "mode": "train",
        }
        for size in (0, 1, 17):
            with self.subTest(size=size):
                _, _, payloads = self._roundtrip({
                    "tokens": torch.arange(size), "labels": torch.arange(size) + 1,
                    "weight": torch.tensor(0.25), "mode": "train",
                }, spec)
                self.assertEqual(len(payloads), 3 if size else 2)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_cpu_and_device_declarations_stay_in_separate_buckets(self):
        """Feature: Placement.
        Description: Same dtype at different destinations.
        Expectation: Separate payloads.
        """
        spec = {
            "host": TensorFieldSpec((2,), torch.int64, "cpu"),
            "device": TensorFieldSpec((2,), torch.int64, "device"),
        }
        _, _, payloads = self._roundtrip({"host": torch.arange(2), "device": torch.arange(2)}, spec)
        self.assertEqual(len(payloads), 3)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_large_tensor_coalesces_with_other_fields_without_size_limit(self):
        """Feature: Coalescing.
        Description: A leaf exceeds 64 MiB.
        Expectation: One combined payload.
        """
        large_size = 64 * 1024 * 1024 + 1
        batch = [torch.full((large_size,), 7, dtype=torch.uint8), torch.tensor([1, 2, 3], dtype=torch.uint8)]
        _, _, payloads = self._roundtrip(batch)
        self.assertEqual([tensor.numel() for tensor in payloads], [large_size + 3])

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_fixed_and_dynamic_eof_only_send_control(self):
        """Feature: EOF.
        Description: Both layout modes end iteration.
        Expectation: No data payload.
        """
        spec = {"tokens": TensorFieldSpec((None,), torch.int64, "cpu")}
        _, received, payloads = self._roundtrip(None, spec)
        self.assertIsNone(received)
        self.assertEqual(len(payloads), 1)
        self.assertEqual(payloads[0].tolist(), [0, 0])

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_preparation_error_reaches_peer_before_payload(self):
        """Feature: Error propagation.
        Description: Source preparation fails.
        Expectation: Both peers raise.
        """
        with (
            patch("hyper_parallel.distributed_data.transport.dist.get_backend", return_value="gloo"),
            patch("hyper_parallel.distributed_data.transport.broadcast_control_object",
                  side_effect=lambda value, **kwargs: value) as control,
            patch("hyper_parallel.distributed_data.transport.dist.broadcast") as payload,
        ):
            with self.assertRaisesRegex(ValueError, "preparation failed"):
                _transport(0).broadcast(None, error=ValueError("preparation failed"))
            descriptor = control.call_args.args[0]
            control.side_effect = None
            control.return_value = descriptor
            with self.assertRaisesRegex(RuntimeError, "preparation failed"):
                _transport(1).broadcast(None)
            payload.assert_not_called()

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_single_rank_is_identity(self):
        """Feature: Single rank.
        Description: No peers.
        Expectation: Original batch without communication.
        """
        batch = {"tokens": torch.arange(3)}
        with patch("hyper_parallel.distributed_data.transport.dist.broadcast") as payload:
            self.assertIs(_transport(0, size=1).broadcast(batch), batch)
            payload.assert_not_called()
