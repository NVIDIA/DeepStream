# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from pyservicemaker import Pipeline, Flow, BufferRetriever
import threading, time

def stop_pipeline(pipeline, timeout=1):
    time.sleep(timeout)
    print("Stop")
    pipeline.stop()

samples = [
    "file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.h264",
    "file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4",
    "file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h265.mp4",
    "file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4"
]

class MyBufferRetriever(BufferRetriever):
    def __init__(self):
        super().__init__()
        self.frames = 0

    def consume(self, buffer):
        tensor = buffer.extract(0)
        assert len(tensor.shape) == 3
        self.frames += 1
        return 1

def test_retrieve_tensor():
    for sample in samples:
        pipeline = Pipeline("test")
        thread = threading.Thread(target=stop_pipeline, args=(pipeline, 2))
        thread.start()
        r = MyBufferRetriever()
        Flow(pipeline).capture([sample]).retrieve(r)()
        assert r.frames > 0
        thread.join()

if __name__ == '__main__':
    test_retrieve_tensor()