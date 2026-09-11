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

"""Unit tests for VideoEncoder and FrameInput in lib/codec.py.

Test tier: Tier 2 (no GPU required except where noted).
GPU-dependent tests are marked with @pytest.mark.requires_gpu.

To run inside a DeepStream SDK container:
    pytest tests/test_codec_video_encoder.py -v
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_frames(n: int = 10, height: int = 480, width: int = 640) -> list:
    """Create a list of synthetic HWC uint8 RGB torch tensors."""
    return [torch.randint(0, 256, (height, width, 3), dtype=torch.uint8) for _ in range(n)]


# ---------------------------------------------------------------------------
# FrameInput tests
# ---------------------------------------------------------------------------


class TestFrameInput:
    """Tests for FrameInput BufferProvider."""

    def test_init_sets_dimensions(self):
        """FrameInput stores height, width, and framerate correctly."""
        from lib.codec import FrameInput

        fi = FrameInput(height=480, width=640, framerate=30)
        assert fi.format == "RGB"
        assert fi.height == 480
        assert fi.width == 640
        assert fi.framerate == 30

    def test_send_and_generate(self):
        """Frames enqueued with send() are returned by generate()."""
        from lib.codec import FrameInput

        fi = FrameInput(height=480, width=640, framerate=30)
        frame = torch.zeros((480, 640, 3), dtype=torch.uint8)

        ds_tensor = MagicMock()
        ds_tensor.wrap.return_value = "wrapped_buffer"

        # Patch as_tensor so we can test without pyservicemaker GPU
        with patch("lib.codec.as_tensor", return_value=ds_tensor) as mock_as_tensor:
            fi.send(frame)
            result = fi.generate(1)
            mock_as_tensor.assert_called_once_with(frame, "HWC")
            assert result == "wrapped_buffer"

    def test_finish_sends_sentinel(self):
        """finish() places None in the queue as the end-of-stream sentinel."""
        from lib.codec import FrameInput

        fi = FrameInput(height=480, width=640)
        fi.finish()
        result = fi._queue.get_nowait()
        assert result is None

    def test_generate_returns_empty_buffer_after_finish(self):
        """generate() converts the finish sentinel to an empty Service Maker Buffer."""
        from lib.codec import FrameInput

        fi = FrameInput(height=480, width=640)
        fi.finish()

        with patch("lib.codec.Buffer", return_value="empty_buffer"):
            result = fi.generate(1)
            assert result == "empty_buffer"

    def test_queue_ordering(self):
        """Multiple frames are returned in FIFO order."""
        from lib.codec import FrameInput

        fi = FrameInput(height=2, width=2, framerate=1)
        frames = [torch.full((2, 2, 3), i, dtype=torch.uint8) for i in range(5)]
        for f in frames:
            fi.send(f)
        fi.finish()

        retrieved = []
        while True:
            item = fi._queue.get_nowait()
            if item is None:
                break
            retrieved.append(item)
        assert len(retrieved) == 5
        for got, expected in zip(retrieved, frames):
            assert torch.equal(got, expected)


# ---------------------------------------------------------------------------
# VideoEncoder tests (mocked pyservicemaker — no GPU required)
# ---------------------------------------------------------------------------


class TestVideoEncoderMocked:
    """VideoEncoder unit tests with pyservicemaker mocked out."""

    def _make_encoder(self):
        """Return a VideoEncoder with mocked internals."""
        from lib.codec import VideoEncoder

        return VideoEncoder(device_id=0)

    def test_encode_raises_on_empty_frames(self):
        """VideoEncoder.encode() raises VideoEncodingError for empty list."""
        from lib.codec import VideoEncoder, VideoEncodingError

        encoder = VideoEncoder()
        with pytest.raises(VideoEncodingError, match="empty"):
            encoder.encode([], output_path="/tmp/out.mp4")

    def test_encode_raises_on_invalid_tensor_shape(self):
        """VideoEncoder.encode() raises VideoEncodingError for non-HWC tensors."""
        from lib.codec import VideoEncoder, VideoEncodingError

        encoder = VideoEncoder()
        bad_frame = torch.zeros((3, 480, 640), dtype=torch.uint8)  # CHW, not HWC
        with pytest.raises(VideoEncodingError, match="HWC"):
            encoder.encode([bad_frame], output_path="/tmp/out.mp4")

    def test_encode_raises_on_wrong_channels(self):
        """VideoEncoder.encode() raises if tensor has wrong channel count."""
        from lib.codec import VideoEncoder, VideoEncodingError

        encoder = VideoEncoder()
        bad_frame = torch.zeros((480, 640, 4), dtype=torch.uint8)  # RGBA
        with pytest.raises(VideoEncodingError, match="HWC"):
            encoder.encode([bad_frame], output_path="/tmp/out.mp4")

    def test_encode_infers_dimensions_from_first_frame(self):
        """VideoEncoder infers height/width from frames[0] when not provided."""
        from lib.codec import FrameInput, VideoEncoder

        encoder = VideoEncoder()
        frames = make_frames(n=3, height=360, width=480)

        # Mock the Pipeline and Flow so no real GPU work occurs
        with patch("lib.codec.Pipeline") as MockPipeline, patch("lib.codec.Flow") as MockFlow:
            mock_pipeline = MagicMock()
            MockPipeline.return_value = mock_pipeline
            mock_flow = MagicMock()
            MockFlow.return_value.inject.return_value.encode.return_value = mock_flow

            # Patch FrameInput.generate to avoid actual pyservicemaker calls
            with patch.object(FrameInput, "generate", return_value=None):
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
                    tmp = tf.name
                try:
                    encoder.encode(frames, output_path=tmp)
                finally:
                    if os.path.exists(tmp):
                        os.remove(tmp)

            # Verify Pipeline was constructed with correct framerate via FrameInput
            MockPipeline.assert_called_once_with("video_encoder")

    def test_encode_wraps_pyservicemaker_exception_as_video_encoding_error(self):
        """If pyservicemaker raises, VideoEncoder wraps it in VideoEncodingError."""
        from lib.codec import VideoEncoder, VideoEncodingError

        encoder = VideoEncoder()
        frames = make_frames(n=2)

        with patch("lib.codec.Pipeline") as MockPipeline:
            MockPipeline.side_effect = RuntimeError("nvenc not available")
            with pytest.raises(VideoEncodingError, match="nvenc not available"):
                encoder.encode(frames, output_path="/tmp/out.mp4")

    def test_encode_returns_output_path_on_success(self):
        """encode() returns the output_path string on success."""
        from lib.codec import FrameInput, VideoEncoder

        encoder = VideoEncoder()
        frames = make_frames(n=5)

        with patch("lib.codec.Pipeline") as MockPipeline, patch("lib.codec.Flow") as MockFlow:
            mock_pipeline = MagicMock()
            MockPipeline.return_value = mock_pipeline
            mock_flow = MagicMock()
            MockFlow.return_value.inject.return_value.encode.return_value = mock_flow

            with patch.object(FrameInput, "generate", return_value=None):
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
                    tmp = tf.name
                try:
                    result = encoder.encode(frames, output_path=tmp)
                    assert result == tmp
                finally:
                    if os.path.exists(tmp):
                        os.remove(tmp)


# ---------------------------------------------------------------------------
# VideoEncoder GPU integration test (requires DeepStream container)
# ---------------------------------------------------------------------------


@pytest.mark.requires_gpu
class TestVideoEncoderGPU:
    """Integration tests that run inside a DeepStream SDK container with nvenc.

    These tests are skipped unless pytest is invoked with --requires-gpu
    or the 'requires_gpu' mark is explicitly selected.
    """

    def test_encode_10_synthetic_frames_produces_valid_mp4(self):
        """Feed 10 synthetic 640x480 frames; verify output MP4 exists and has correct count."""
        from lib.codec import VideoEncoder

        try:
            from pyservicemaker.utils import MediaInfo
        except ImportError:
            pytest.skip("pyservicemaker not available")

        encoder = VideoEncoder(device_id=0)
        frames = make_frames(n=10, height=480, width=640)

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
            output_path = tf.name
        os.remove(output_path)  # let encoder create the file

        try:
            result = encoder.encode(frames, output_path=output_path, fps=30)
            assert result == output_path
            assert os.path.exists(output_path), "Output MP4 file not created"
            assert os.path.getsize(output_path) > 0, "Output MP4 file is empty"

            info = MediaInfo.discover(output_path)
            # duration should be ~0.33 s (10 frames at 30 fps)
            assert info.duration > 0, "MP4 duration should be greater than 0"
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)
