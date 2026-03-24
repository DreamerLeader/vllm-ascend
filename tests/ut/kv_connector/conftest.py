# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""conftest for kv_connector tests - mocks torch.npu and torch_npu
so tests can run without NPU hardware."""

import importlib
import sys
import types
from unittest.mock import MagicMock

# Mock torch_npu as a proper module with __spec__ that allows attribute imports
if "torch_npu" not in sys.modules:

    class _TorchNpuModule(types.ModuleType):
        """A module that returns MagicMock for any attribute access."""

        def __getattr__(self, name):
            if name in ("__spec__", "__name__", "__loader__", "__path__", "__file__", "__package__"):
                return super().__getattribute__(name)
            return MagicMock()

    torch_npu_mock = _TorchNpuModule("torch_npu")
    torch_npu_mock.__spec__ = importlib.machinery.ModuleSpec("torch_npu", None)
    torch_npu_mock.__package__ = "torch_npu"
    sys.modules["torch_npu"] = torch_npu_mock

# Mock torch.npu before any imports
import torch

if not hasattr(torch, "npu"):
    torch.npu = MagicMock()
    torch.npu.Event = MagicMock
