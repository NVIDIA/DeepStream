# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test-only import shims for unit tests outside a generated app container."""

import sys
import types
from types import SimpleNamespace


if "config" not in sys.modules:
    config_module = types.ModuleType("config")
    config_module.global_config = SimpleNamespace()
    sys.modules["config"] = config_module

if "custom" not in sys.modules:
    sys.modules["custom"] = types.ModuleType("custom")

try:
    import pyservicemaker  # noqa: F401
except ImportError:
    sys.modules.pop("pyservicemaker", None)

    pyservicemaker_module = types.ModuleType("pyservicemaker")

    class _BufferProvider:
        pass

    class _BufferRetriever:
        pass

    class _BufferOperator:
        pass

    class _BatchMetadataOperator:
        pass

    class _Buffer:
        pass

    class _Pipeline:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

        def stop(self):
            pass

        def wait(self):
            pass

    class _Flow:
        def __init__(self, *args, **kwargs):
            pass

        def inject(self, *args, **kwargs):
            return self

        def decode(self, *args, **kwargs):
            return self

        def encode(self, *args, **kwargs):
            return self

        def retrieve(self, *args, **kwargs):
            return self

    class _ColorFormat:
        RGB = "RGB"
        RGBA = "RGBA"
        I420 = "I420"
        NV12 = "NV12"

    def _as_tensor(tensor, *args, **kwargs):
        return tensor

    pyservicemaker_module.BufferProvider = _BufferProvider
    pyservicemaker_module.Buffer = _Buffer
    pyservicemaker_module.Pipeline = _Pipeline
    pyservicemaker_module.Flow = _Flow
    pyservicemaker_module.BufferRetriever = _BufferRetriever
    pyservicemaker_module.BufferOperator = _BufferOperator
    pyservicemaker_module.BatchMetadataOperator = _BatchMetadataOperator
    pyservicemaker_module.ColorFormat = _ColorFormat
    pyservicemaker_module.as_tensor = _as_tensor
    pyservicemaker_module.EOSMessage = object
    pyservicemaker_module.osd = SimpleNamespace()
    pyservicemaker_module.utils = SimpleNamespace()
    pyservicemaker_module.signal = SimpleNamespace()
    pyservicemaker_module.CommonFactory = SimpleNamespace()

    utils_module = types.ModuleType("pyservicemaker.utils")

    class _MediaExtractor:
        pass

    class _MediaChunk:
        pass

    class _MediaInfo:
        @staticmethod
        def discover(*args, **kwargs):
            return SimpleNamespace(duration=0)

    utils_module.MediaExtractor = _MediaExtractor
    utils_module.MediaChunk = _MediaChunk
    utils_module.MediaInfo = _MediaInfo

    sys.modules["pyservicemaker"] = pyservicemaker_module
    sys.modules["pyservicemaker.utils"] = utils_module
