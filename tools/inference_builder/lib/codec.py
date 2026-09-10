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


from pyservicemaker import BufferProvider, Buffer, Pipeline, Flow, BufferRetriever, as_tensor, ColorFormat
from queue import Queue, Empty, Full
import numpy as np
from .utils import get_logger
from typing import List
import base64
import torch

logger = get_logger(__name__)

png_data = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAAEElEQVR4nGK6HcwNCAAA//8DTgE8HuxwEQAAAABJRU5ErkJggg==")
jpg_data = base64.b64decode("/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAIBAQEBAQIBAQECAgICAgQDAgICAgUEBAMEBgUGBgYFBgYGBwkIBgcJBwYGCAsICQoKCgoKBggLDAsKDAkKCgr/2wBDAQICAgICAgUDAwUKBwYHCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgr/wAARCAAgACADASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD+f+iiigAooooAKKKKACiiigD/2Q==")


class ImageInput(BufferProvider):
    DEFAULT_HEIGHT = 1080
    DEFAULT_WIDTH = 1920

    def __init__(self, format):
        super().__init__()
        self.format = format
        self.height = ImageInput.DEFAULT_HEIGHT
        self.width = ImageInput.DEFAULT_WIDTH
        self.framerate = 1
        self.device = 'cpu'
        self.queue = Queue(maxsize=1)

    def generate(self, size):
        tensor = self.queue.get()
        return tensor.wrap(ColorFormat.I420)

    def send(self, data):
        self.queue.put(data)

class ImageOutput(BufferRetriever):
    def __init__(self, max_queue_size: int=1):
        super().__init__()
        self._timeout = 10
        self._output = Queue(maxsize=max_queue_size)

    def consume(self, buffer):
        try:
            tensor = buffer.extract(0).clone()
            torch_tensor = torch.utils.dlpack.from_dlpack(tensor)
            self._output.put(torch_tensor)
        except Full:
            logger.error(f"ImageOutput queue is full, buffer dropped")
        return 1

    def get(self):
        try:
            data = self._output.get(timeout=self._timeout)
        except Empty:
            logger.error("ImageOutput timeout, failed to decode the image, input data may be corrupted")
            return None
        return data


class ImageDecoder:
    def __init__(self, formats: List[str], device_id: int=0):
        self._piplines = [Pipeline(f"image_decoder_{format}") for format in formats]
        self._image_inputs = {format: ImageInput(format) for format in formats}
        self._image_output = ImageOutput()
        self._flows = []
        for pipline, format in zip(self._piplines, formats):
            image_input = self._image_inputs[format]
            flow = Flow(pipline).inject([image_input]).decode().retrieve(self._image_output, gpu_id=device_id)
            self._flows.append(flow)
            pipline.start()

        if "PNG" in formats:
            logger.info("PNG decoder warmup:")
            warmup_data = np.frombuffer(png_data, dtype=np.uint8)
            self.decode(as_tensor(warmup_data.copy(), "PNG"), "PNG")
        if "JPEG" in formats:
            logger.info("JPEG decoder warmup:")
            warmup_data = np.frombuffer(jpg_data, dtype=np.uint8)
            self.decode(as_tensor(warmup_data.copy(), "JPEG"), "JPEG")
        logger.info("Image decoder initialized")


    def decode(self, tensor, format: str):
        self._image_inputs[format].send(tensor)
        return self._image_output.get()


class VideoEncodingError(Exception):
    """Exception raised when video encoding via pyservicemaker fails.

    Attributes:
        message: Human-readable description of the encoding failure.
        cause: Optional underlying exception that triggered this error.
    """

    def __init__(self, message: str, cause: Exception = None):
        super().__init__(message)
        self.cause = cause


class FrameInput(BufferProvider):
    """BufferProvider subclass that feeds image tensors into a pyservicemaker encoding pipeline.

    Each call to generate() returns the next frame from an internal queue.
    Frames are expected to be HWC uint8 RGB torch.Tensor objects (GPU or CPU).
    The provider wraps each tensor as a raw RGB Buffer for the Service Maker
    appsrc path; the downstream encode flow converts it to I420/NVMM as needed.

    Attributes:
        format (str): Raw frame format advertised to pyservicemaker.
        height (int): Frame height in pixels.
        width (int): Frame width in pixels.
        framerate (int): Frames per second for the output video.
        device (str): Device where frames reside ('cpu' or 'gpu').
    """

    def __init__(self, height: int, width: int, framerate: int = 30, device: str = "gpu"):
        super().__init__()
        self.format = "RGB"
        self.height = height
        self.width = width
        self.framerate = framerate
        self.device = device
        self._queue: Queue = Queue()

    def generate(self, size: int):
        """Called by pyservicemaker to retrieve the next frame buffer.

        Blocks until a frame is available or the sentinel None is received.

        Args:
            size: Number of buffers requested (unused; one frame per call).

        Returns:
            Buffer wrapped in RGB color format, or an empty Buffer when stream ends.
        """
        tensor = self._queue.get()
        if tensor is None:
            # Sentinel: Service Maker providers signal EOS with an empty Buffer.
            return Buffer()
        # Flow.inject() sees format=RGB and inserts the converter required by
        # Flow.encode(), so the buffer format must match the advertised caps.
        return as_tensor(tensor, "HWC").wrap(ColorFormat.RGB)

    def send(self, frame: torch.Tensor) -> None:
        """Enqueue a frame tensor for encoding.

        Args:
            frame: HWC uint8 RGB torch.Tensor to encode.
        """
        self._queue.put(frame)

    def finish(self) -> None:
        """Signal end-of-stream to the encoding pipeline."""
        self._queue.put(None)


class VideoEncoder:
    """Hardware-accelerated H.264 MP4 video encoder using pyservicemaker nvenc.

    Uses pyservicemaker's Flow.encode() API (symmetric to Flow.decode()) to
    encode a list of image tensors into an H.264 MP4 file via NVIDIA's nvenc
    hardware encoder.

    Example usage::

        encoder = VideoEncoder(device_id=0)
        output_path = encoder.encode(frames, "/tmp/output.mp4", fps=30)

    Args:
        device_id (int): GPU device ID to use for encoding (default: 0).
    """

    def __init__(self, device_id: int = 0):
        self._device_id = device_id

    def encode(
        self,
        frames: List[torch.Tensor],
        output_path: str,
        fps: int = 30,
        width: int = None,
        height: int = None,
    ) -> str:
        """Encode a list of image tensors into an H.264 MP4 file.

        Derives frame resolution from the first tensor when width/height are not
        specified. All frames must have identical spatial dimensions.

        Args:
            frames: Ordered list of HWC uint8 RGB torch.Tensor objects.
            output_path: Filesystem path for the encoded MP4 output file.
            fps: Frames per second (default: 30).
            width: Frame width in pixels. Inferred from frames[0] when None.
            height: Frame height in pixels. Inferred from frames[0] when None.

        Returns:
            output_path: The path to the encoded MP4 file.

        Raises:
            VideoEncodingError: If frames is empty, dimensions are invalid,
                or the pyservicemaker encoding pipeline fails.
        """
        if not frames:
            raise VideoEncodingError("Cannot encode an empty frame list")

        first = frames[0]
        if height is None or width is None:
            if first.ndim != 3 or first.shape[2] != 3:
                raise VideoEncodingError(
                    f"Expected HWC uint8 RGB tensor with shape [H, W, 3], "
                    f"got shape {list(first.shape)}"
                )
            height = first.shape[0]
            width = first.shape[1]

        logger.info(
            "VideoEncoder: encoding %d frames at %dx%d @ %d fps -> %s",
            len(frames),
            width,
            height,
            fps,
            output_path,
        )

        frame_device = "gpu" if torch.cuda.is_available() else "cpu"
        frame_input = FrameInput(
            height=height,
            width=width,
            framerate=fps,
            device=frame_device,
        )
        pipeline = None

        try:
            pipeline = Pipeline("video_encoder")
            # Mirror of the decode API:
            # Flow(pipeline).inject([frame_input]).decode().retrieve(...)
            # Encoding API:
            # Flow(pipeline).inject([frame_input]).encode(output_path, ...)
            flow = Flow(pipeline).inject([frame_input]).encode(
                output_path,
                codec="h264",
                bitrate=0,  # auto bitrate
                framerate=fps,
            )
            pipeline.start()

            # Feed all frames to the encoding pipeline
            for i, frame in enumerate(frames):
                # Service Maker wraps RGB image tensors as buffers only from
                # CUDA memory. Keep CPU tensors for mocked/no-GPU unit tests.
                if torch.cuda.is_available():
                    frame = frame.to(
                        device=torch.device("cuda", self._device_id),
                        non_blocking=True,
                    )
                frame = frame.contiguous()
                frame_input.send(frame)
                logger.debug("VideoEncoder: sent frame %d/%d", i + 1, len(frames))

            # Signal end-of-stream and wait for pipeline to flush
            frame_input.finish()

            # Wait for the flow to complete flushing to disk
            if hasattr(flow, "wait"):
                flow.wait()
            elif hasattr(pipeline, "wait"):
                pipeline.wait()

        except VideoEncodingError:
            raise
        except Exception as exc:
            raise VideoEncodingError(
                f"pyservicemaker encoding pipeline failed: {exc}", cause=exc
            ) from exc
        finally:
            if pipeline is not None:
                try:
                    pipeline.stop()
                except Exception:
                    pass

        logger.info("VideoEncoder: encoding complete -> %s", output_path)
        return output_path
