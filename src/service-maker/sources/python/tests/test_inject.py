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

from pyservicemaker import Pipeline, Flow, BufferProvider, Buffer, RenderMode
import threading, time
import os

RENDER_MODE = RenderMode.DISPLAY if os.environ.get("DISPLAY") else RenderMode.DISCARD

class MyBufferProvider(BufferProvider):

    def __init__(self, width, height, device='cpu', framerate=30, format="RGB"):
        super().__init__()
        self.width = width
        self.height = height
        self.format = format
        self.framerate = framerate
        self.device = device
        self.count = 0
        self.expected = 255

    def generate(self, size):
        data = [self.count]*(self.width*self.height*3)
        if self.count < self.expected:
            self.count += 1
        return Buffer() if self.count == self.expected else Buffer(data)

def test_feeder():
    r = MyBufferProvider(320, 240)
    Flow(Pipeline("test")).inject([r]).render(RENDER_MODE)()
    assert r.count == r.expected

if __name__ == '__main__':
    test_feeder()