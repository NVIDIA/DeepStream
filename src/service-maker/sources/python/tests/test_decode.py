# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from pyservicemaker import Pipeline, Flow, BufferProvider, Buffer, BufferRetriever, as_tensor, ColorFormat, RenderMode
import numpy as np
import os

RENDER_MODE = RenderMode.DISPLAY if os.environ.get("DISPLAY") else RenderMode.DISCARD
FILE_PATH = "/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.jpg"

class JpegBufferProvider(BufferProvider):

    def __init__(self, file_path:str):
        super().__init__()
        self._file_path = file_path
        self.format = "JPEG"
        self.width = 1280
        self.height = 720
        self.framerate = 0
        self.count = 0
        self.expected = 255

    def generate(self, size):
        bytes = []
        with open(self._file_path, "rb") as f:
            bytes = list(f.read())
        if self.count < self.expected:
            self.count += 1
        return Buffer() if self.count == self.expected else Buffer(bytes)

class NumpyBufferProvider(BufferProvider):
    def __init__(self, file_path:str):
        super().__init__()
        self._file_path = file_path
        self.format = "JPEG"
        self.width = 1280
        self.height = 720
        self.framerate = 0
        self.count = 0
        self.expected = 255

    def generate(self, size):
        bytes = np.fromfile(self._file_path, np.uint8)
        if self.count < self.expected:
            self.count += 1
        bytes = as_tensor(bytes, 'JPEG')
        return Buffer() if self.count == self.expected else bytes.wrap(ColorFormat.I420)

class AudioBufferProvider(BufferProvider):
    def __init__(self):
        super().__init__()
        self.media_type = "audio"
        self.format = "S16LE"
        self.sample_rate = 16000
        self.channels = 1
        self.layout = "interleaved"
        self.count = 0
        self.expected = 8

    def generate(self, size):
        if self.count >= self.expected:
            return Buffer()
        self.count += 1
        # 20 ms of S16LE mono silence.
        return Buffer([0] * (self.sample_rate // 50 * self.channels * 2))

class AudioBufferRetriever(BufferRetriever):
    def __init__(self):
        super().__init__()
        self.count = 0

    def consume(self, buffer):
        self.count += 1
        return 1


def test_decoder():
    r = JpegBufferProvider(FILE_PATH)
    Flow(Pipeline("test")).inject([r]).decode().batch().render(RENDER_MODE)()
    assert r.count == r.expected
    r = NumpyBufferProvider(FILE_PATH)
    Flow(Pipeline("test")).inject([r]).decode().batch().render(RENDER_MODE)()
    assert r.count == r.expected
    r = JpegBufferProvider(FILE_PATH)
    Flow(Pipeline("test")).inject([r]).decode().render(RENDER_MODE)()
    assert r.count == r.expected
    r = NumpyBufferProvider(FILE_PATH)
    Flow(Pipeline("test")).inject([r]).decode().render(RENDER_MODE)()
    assert r.count == r.expected

def test_audio_decoder():
    provider = AudioBufferProvider()
    retriever = AudioBufferRetriever()
    Flow(Pipeline("test-audio-decode")).inject([provider]).decode().retrieve(
        retriever,
        audio_caps="audio/x-raw,format=S16LE,rate=16000,channels=1,layout=interleaved"
    )()
    assert provider.count == provider.expected
    assert retriever.count > 0

if __name__ == '__main__':
    test_decoder()
    test_audio_decoder()