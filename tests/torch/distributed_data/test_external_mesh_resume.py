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
"""Thin launchers for external-source Mesh and checkpoint coverage."""

from pathlib import Path

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark

_WORKER = str(Path(__file__).with_name("_test_external_mesh_resume.py"))


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_external_mesh_resume_gloo():
    """Feature: Mesh external loading.
    Description: Exercise DP2/CP2 input replication and source resume over Gloo.
    Expectation: Identical CP batches and exact uninterrupted/resumed sequences.
    """
    torchrun_case(file_name=_WORKER, case_name="test_mesh_resume_gloo", num_proc=4)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_external_mesh_resume_hccl():
    """Feature: Mesh external loading on Ascend.
    Description: Exercise foreground HCCL prefetch and checkpoint with real device tensors.
    Expectation: Identical CP batches and exact uninterrupted/resumed sequences.
    """
    torchrun_case(file_name=_WORKER, case_name="test_mesh_resume_hccl", num_proc=4)
