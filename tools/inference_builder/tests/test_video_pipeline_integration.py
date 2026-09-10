# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

"""Integration tests for the end-to-end video output asset pipeline (T4).

Tests the full path:
    Synthetic image tensors
    -> VideoOutputDataFlow._process_custom_data()
    -> VideoEncoder.encode()
    -> AssetManager.create_from_path()
    -> asset_id string

GPU-requiring tests are marked @pytest.mark.requires_gpu.
All other tests use mocks so they can run without a GPU.

To run mocked tests:
    pytest tests/test_video_pipeline_integration.py -v -m "not requires_gpu"

To run full GPU tests (inside DeepStream SDK container):
    pytest tests/test_video_pipeline_integration.py -v -m requires_gpu
"""

import os
from unittest.mock import MagicMock, patch

import pytest
import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_frames(n=10, height=480, width=640):
    return [torch.randint(0, 256, (height, width, 3), dtype=torch.uint8) for _ in range(n)]


def make_video_output_config(name="frames_out"):
    return {"name": name, "data_type": "TYPE_CUSTOM_VIDEO_OUTPUT_ASSET", "dims": [-1, -1, 3]}


# ---------------------------------------------------------------------------
# Mocked integration tests (Tier 2 — no GPU)
# ---------------------------------------------------------------------------


class TestVideoOutputPipelineMocked:
    """End-to-end integration tests with VideoEncoder and AssetManager mocked."""

    @pytest.fixture
    def isolated_asset_manager(self, tmp_path, monkeypatch):
        """Fresh AssetManager backed by tmp_path."""
        import lib.asset_manager as am_module

        am_module.AssetManager._instance = None
        monkeypatch.setattr(am_module, "DEFAULT_ASSET_DIR", str(tmp_path / "assets"))
        return am_module.AssetManager()

    def test_full_pipeline_returns_asset_id(self, isolated_asset_manager, tmp_path):
        """Full mocked pipeline: frames -> VideoOutputDataFlow -> asset_id string."""
        from lib.inference import VideoOutputDataFlow

        configs = [make_video_output_config()]
        tensor_names = [("frames_out", "frames_out")]
        flow = VideoOutputDataFlow(configs, tensor_names, "TYPE_CUSTOM_VIDEO_OUTPUT_ASSET")

        frames = make_frames(n=10)

        # Create a real temp MP4 stub that will be "moved" by create_from_path
        tmp_mp4 = str(tmp_path / "encoded.mp4")
        with open(tmp_mp4, "wb") as f:
            f.write(b"\x00\x00\x00\x1cftypisom" + b"\x00" * 50)

        with patch.object(flow._encoder, "encode", return_value=tmp_mp4) as mock_encode, patch(
            "lib.inference.AssetManager"
        ) as MockAM:
            mock_asset = MagicMock()
            mock_asset.id = "integration-asset-001"
            MockAM.return_value.create_from_path.return_value = mock_asset

            result = flow._process_custom_data(frames, "TYPE_CUSTOM_VIDEO_OUTPUT_ASSET")

        assert isinstance(result, str), f"Expected str asset_id, got {type(result)}"
        assert result == "integration-asset-001"

        # Verify VideoEncoder was called with correct arguments
        mock_encode.assert_called_once()
        encode_args = mock_encode.call_args[0]
        assert len(encode_args[0]) == 10  # 10 frames
        assert encode_args[2] == 480  # height
        assert encode_args[3] == 640  # width

        # Verify AssetManager was called with video/mp4
        call = MockAM.return_value.create_from_path.call_args
        assert "video/mp4" in str(call)

    def test_encoding_error_produces_enhanced_error_not_exception(self, isolated_asset_manager):
        """A VideoEncodingError must not propagate as a Python exception."""
        from lib.codec import VideoEncodingError
        from lib.errors import EnhancedError
        from lib.inference import VideoOutputDataFlow

        configs = [make_video_output_config()]
        tensor_names = [("frames_out", "frames_out")]
        flow = VideoOutputDataFlow(configs, tensor_names, "TYPE_CUSTOM_VIDEO_OUTPUT_ASSET")
        frames = make_frames(n=5)

        with patch.object(
            flow._encoder, "encode", side_effect=VideoEncodingError("simulated nvenc failure")
        ):
            result = flow._process_custom_data(frames, "TYPE_CUSTOM_VIDEO_OUTPUT_ASSET")

        assert isinstance(
            result, EnhancedError
        ), f"Expected EnhancedError on encoding failure, got {type(result)}"

    def test_asset_creation_failure_produces_enhanced_error(self, isolated_asset_manager):
        """AssetManager returning None must produce EnhancedError."""
        from lib.errors import EnhancedError
        from lib.inference import VideoOutputDataFlow

        configs = [make_video_output_config()]
        tensor_names = [("frames_out", "frames_out")]
        flow = VideoOutputDataFlow(configs, tensor_names, "TYPE_CUSTOM_VIDEO_OUTPUT_ASSET")
        frames = make_frames(n=3)

        with patch.object(flow._encoder, "encode", return_value="/tmp/out.mp4"), patch(
            "lib.inference.AssetManager"
        ) as MockAM:
            MockAM.return_value.create_from_path.return_value = None

            result = flow._process_custom_data(frames, "TYPE_CUSTOM_VIDEO_OUTPUT_ASSET")

        assert isinstance(result, EnhancedError)

    def test_asset_id_is_string_type(self, isolated_asset_manager):
        """Returned asset_id must be a Python str, not bytes or other type."""
        from lib.inference import VideoOutputDataFlow

        configs = [make_video_output_config()]
        tensor_names = [("frames_out", "frames_out")]
        flow = VideoOutputDataFlow(configs, tensor_names, "TYPE_CUSTOM_VIDEO_OUTPUT_ASSET")
        frames = make_frames(n=2)

        with patch.object(flow._encoder, "encode", return_value="/tmp/out.mp4"), patch(
            "lib.inference.AssetManager"
        ) as MockAM:
            mock_asset = MagicMock()
            mock_asset.id = "string-asset-id"
            MockAM.return_value.create_from_path.return_value = mock_asset

            result = flow._process_custom_data(frames, "TYPE_CUSTOM_VIDEO_OUTPUT_ASSET")

        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# GPU integration test (requires DeepStream SDK container)
# ---------------------------------------------------------------------------


@pytest.mark.requires_gpu
class TestVideoOutputPipelineGPU:
    """Full end-to-end integration test with real GPU encoding.

    Requires a DeepStream SDK container with pyservicemaker and nvenc support.
    """

    @pytest.fixture
    def isolated_asset_manager(self, tmp_path, monkeypatch):
        import lib.asset_manager as am_module

        am_module.AssetManager._instance = None
        monkeypatch.setattr(am_module, "DEFAULT_ASSET_DIR", str(tmp_path / "assets"))
        return am_module.AssetManager()

    def test_end_to_end_produces_valid_asset(self, isolated_asset_manager):
        """Feed 10 synthetic tensors through the full pipeline; verify asset."""
        try:
            from pyservicemaker.utils import MediaInfo
        except ImportError:
            pytest.skip("pyservicemaker not available")

        from lib.inference import VideoOutputDataFlow

        configs = [make_video_output_config()]
        tensor_names = [("frames_out", "frames_out")]
        flow = VideoOutputDataFlow(configs, tensor_names, "TYPE_CUSTOM_VIDEO_OUTPUT_ASSET")

        frames = make_frames(n=10, height=480, width=640)
        result = flow._process_custom_data(frames, "TYPE_CUSTOM_VIDEO_OUTPUT_ASSET")

        # Result must be a valid asset_id string
        assert isinstance(result, str), f"Expected asset_id str, got {type(result)}: {result}"

        # Verify the asset exists and has correct metadata
        asset = isolated_asset_manager.get_asset(result)
        assert asset is not None, f"Asset {result} not found in AssetManager"
        assert (
            asset.mime_type == "video/mp4"
        ), f"Expected mime_type video/mp4, got {asset.mime_type}"
        assert asset.duration > 0, f"Asset duration should be > 0, got {asset.duration}"

        # Verify the MP4 file is valid
        assert os.path.exists(asset.path), f"Asset file not found at {asset.path}"
        assert os.path.getsize(asset.path) > 0

        info = MediaInfo.discover(asset.path)
        assert info.duration > 0, "MP4 MediaInfo duration should be > 0"
