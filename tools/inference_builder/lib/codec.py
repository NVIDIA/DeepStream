# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
