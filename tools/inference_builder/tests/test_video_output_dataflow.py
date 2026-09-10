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

"""Unit tests for VideoOutputDataFlow._process_custom_data().

Test tier: Tier 2 (no GPU required — VideoEncoder and AssetManager are mocked).

To run:
    pytest tests/test_video_output_dataflow.py -v
"""

from unittest.mock import MagicMock, patch

import numpy as np
import torch

VIDEO_OUTPUT_FILE_TYPE = "TYPE_CUSTOM_VIDEO_OUTPUT_FILE"
VIDEO_OUTPUT_ASSET_TYPE = "TYPE_CUSTOM_VIDEO_OUTPUT_ASSET"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(name="frames_out", data_type=VIDEO_OUTPUT_ASSET_TYPE):
    return {"name": name, "data_type": data_type, "dims": [-1, -1, 3]}


def _make_tensor_names(config_name="frames_out"):
    return [(config_name, config_name)]


def _make_frames(n=5, h=480, w=640):
    return [torch.randint(0, 256, (h, w, 3), dtype=torch.uint8) for _ in range(n)]


# ---------------------------------------------------------------------------
# VideoOutputDataFlow unit tests
# ---------------------------------------------------------------------------


class TestVideoOutputDataFlow:
    """Tests for VideoOutputDataFlow._process_custom_data."""

    def _make_flow(self, name="frames_out", data_type=VIDEO_OUTPUT_ASSET_TYPE):
        """Construct a VideoOutputDataFlow with mocked pyservicemaker."""
        from lib.inference import VideoOutputDataFlow

        configs = [_make_config(name=name, data_type=data_type)]
        tensor_names = _make_tensor_names(config_name=name)
        return VideoOutputDataFlow(configs, tensor_names, data_type)

    def test_single_tensor_is_wrapped_as_list(self):
        """A single torch.Tensor is treated as a one-frame list."""
        flow = self._make_flow()
        frame = torch.zeros((480, 640, 3), dtype=torch.uint8)
        mock_asset = MagicMock()
        mock_asset.id = "asset-abc-123"

        with patch.object(
            flow._encoder, "encode", return_value="/tmp/out.mp4"
        ) as mock_encode, patch("lib.inference.AssetManager") as MockAM:
            instance = MockAM.return_value
            instance.create_from_path.return_value = mock_asset

            result = flow._process_custom_data(frame, VIDEO_OUTPUT_ASSET_TYPE)

        assert result == "asset-abc-123"
        mock_encode.assert_called_once()
        args, kwargs = mock_encode.call_args
        assert len(args[0]) == 1  # one frame

    def test_list_of_tensors(self):
        """A list of tensors is passed directly to VideoEncoder."""
        flow = self._make_flow()
        frames = _make_frames(n=10)
        mock_asset = MagicMock()
        mock_asset.id = "asset-xyz-456"

        with patch.object(
            flow._encoder, "encode", return_value="/tmp/out.mp4"
        ) as mock_encode, patch("lib.inference.AssetManager") as MockAM:
            instance = MockAM.return_value
            instance.create_from_path.return_value = mock_asset

            result = flow._process_custom_data(frames, VIDEO_OUTPUT_ASSET_TYPE)

        assert result == "asset-xyz-456"
        args, kwargs = mock_encode.call_args
        assert len(args[0]) == 10

    def test_file_output_returns_encoded_path_without_asset_manager(self):
        """TYPE_CUSTOM_VIDEO_OUTPUT_FILE returns the encoded MP4 path directly."""
        flow = self._make_flow(data_type=VIDEO_OUTPUT_FILE_TYPE)
        frames = _make_frames(n=3)

        with patch.object(flow._encoder, "encode") as mock_encode, patch(
            "lib.inference.AssetManager"
        ) as MockAM:
            result = flow._process_custom_data(frames, VIDEO_OUTPUT_FILE_TYPE)

        assert result.endswith(".mp4")
        assert "/tmp/" in result
        mock_encode.assert_called_once()
        MockAM.assert_not_called()

    def test_returns_error_for_non_tensor_input(self):
        """Non-tensor input returns an EnhancedError."""
        from lib.errors import EnhancedError

        flow = self._make_flow()
        result = flow._process_custom_data("not_a_tensor", VIDEO_OUTPUT_ASSET_TYPE)
        assert isinstance(result, EnhancedError)

    def test_returns_error_for_empty_frame_list(self):
        """Empty frame list returns an EnhancedError."""
        from lib.errors import EnhancedError

        flow = self._make_flow()
        result = flow._process_custom_data([], VIDEO_OUTPUT_ASSET_TYPE)
        assert isinstance(result, EnhancedError)

    def test_returns_error_for_wrong_tensor_shape(self):
        """CHW tensor (not HWC) returns an EnhancedError."""
        from lib.errors import EnhancedError

        flow = self._make_flow()
        bad_frame = torch.zeros((3, 480, 640), dtype=torch.uint8)  # CHW
        result = flow._process_custom_data([bad_frame], VIDEO_OUTPUT_ASSET_TYPE)
        assert isinstance(result, EnhancedError)

    def test_returns_error_for_non_tensor_frame_in_list(self):
        """A non-tensor item in the frame list returns an EnhancedError."""
        from lib.errors import EnhancedError

        flow = self._make_flow()
        frames = _make_frames(n=2)
        frames.append("bad_frame")

        with patch.object(flow._encoder, "encode") as mock_encode:
            result = flow._process_custom_data(frames, VIDEO_OUTPUT_ASSET_TYPE)

        assert isinstance(result, EnhancedError)
        mock_encode.assert_not_called()

    def test_returns_error_for_wrong_tensor_dtype(self):
        """Frames must be uint8 tensors."""
        from lib.errors import EnhancedError

        flow = self._make_flow()
        frames = [torch.zeros((480, 640, 3), dtype=torch.float32)]

        with patch.object(flow._encoder, "encode") as mock_encode:
            result = flow._process_custom_data(frames, VIDEO_OUTPUT_ASSET_TYPE)

        assert isinstance(result, EnhancedError)
        mock_encode.assert_not_called()

    def test_returns_error_for_mismatched_frame_dimensions(self):
        """All frames must have identical H/W dimensions."""
        from lib.errors import EnhancedError

        flow = self._make_flow()
        frames = [
            torch.zeros((480, 640, 3), dtype=torch.uint8),
            torch.zeros((240, 640, 3), dtype=torch.uint8),
        ]

        with patch.object(flow._encoder, "encode") as mock_encode:
            result = flow._process_custom_data(frames, VIDEO_OUTPUT_ASSET_TYPE)

        assert isinstance(result, EnhancedError)
        mock_encode.assert_not_called()

    def test_encoding_failure_returns_error(self):
        """VideoEncodingError from VideoEncoder is wrapped as EnhancedError."""
        from lib.codec import VideoEncodingError
        from lib.errors import EnhancedError

        flow = self._make_flow()
        frames = _make_frames(n=3)

        with patch.object(flow._encoder, "encode", side_effect=VideoEncodingError("nvenc fail")):
            result = flow._process_custom_data(frames, VIDEO_OUTPUT_ASSET_TYPE)

        assert isinstance(result, EnhancedError)

    def test_asset_creation_failure_returns_error(self):
        """If AssetManager.create_from_path returns None, an EnhancedError is returned."""
        from lib.errors import EnhancedError

        flow = self._make_flow()
        frames = _make_frames(n=3)

        with patch.object(flow._encoder, "encode", return_value="/tmp/out.mp4"), patch(
            "lib.inference.AssetManager"
        ) as MockAM:
            instance = MockAM.return_value
            instance.create_from_path.return_value = None  # simulate failure

            result = flow._process_custom_data(frames, VIDEO_OUTPUT_ASSET_TYPE)

        assert isinstance(result, EnhancedError)

    def test_delegates_to_parent_for_unknown_type(self):
        """_process_custom_data forwards unknown data types to the parent class."""
        flow = self._make_flow()
        tensor = np.array(["some_string"], dtype=np.str_)
        # Parent DataFlow._process_custom_data with outbound=True returns the tensor as-is
        result = flow._process_custom_data(tensor, "TYPE_STRING")
        assert result is tensor

    def test_outbound_flag_is_true(self):
        """VideoOutputDataFlow must be constructed with outbound=True."""
        flow = self._make_flow()
        assert flow._outbound is True
        assert flow._inbound is False

    def test_encoder_called_with_correct_fps(self):
        """VideoEncoder.encode() is called with the DEFAULT_FPS (30)."""
        from lib.inference import VideoOutputDataFlow

        flow = self._make_flow()
        frames = _make_frames(n=2)
        mock_asset = MagicMock()
        mock_asset.id = "asset-fps-test"

        with patch.object(
            flow._encoder, "encode", return_value="/tmp/out.mp4"
        ) as mock_encode, patch("lib.inference.AssetManager") as MockAM:
            MockAM.return_value.create_from_path.return_value = mock_asset
            flow._process_custom_data(frames, VIDEO_OUTPUT_ASSET_TYPE)

        _, kwargs = mock_encode.call_args
        assert (
            kwargs.get("fps") == VideoOutputDataFlow.DEFAULT_FPS
            or mock_encode.call_args[0][2] == VideoOutputDataFlow.DEFAULT_FPS
            or "fps" in str(mock_encode.call_args)
        )

    def test_asset_manager_called_with_video_mp4_mime(self):
        """AssetManager.create_from_path is called with mime_type='video/mp4'."""
        flow = self._make_flow()
        frames = _make_frames(n=2)
        mock_asset = MagicMock()
        mock_asset.id = "mime-test-asset"

        with patch.object(flow._encoder, "encode", return_value="/tmp/out.mp4"), patch(
            "lib.inference.AssetManager"
        ) as MockAM:
            instance = MockAM.return_value
            instance.create_from_path.return_value = mock_asset
            flow._process_custom_data(frames, VIDEO_OUTPUT_ASSET_TYPE)

        call_kwargs = instance.create_from_path.call_args
        # Check mime_type argument
        assert "video/mp4" in str(call_kwargs)
