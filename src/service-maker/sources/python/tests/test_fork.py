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

from pyservicemaker import Pipeline, Flow, RenderMode
import threading, time

def stop_pipeline(pipeline, timeout=1):
    time.sleep(timeout)
    print("Stop")
    pipeline.stop()

samples_1080p = [
    "/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h265.mp4",
    "file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h265.mp4",
    "/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4",
    "file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4"
]

def test_fork():
    pipeline = Pipeline("test")
    thread = threading.Thread(target=stop_pipeline, args=(pipeline, 2))
    thread.start()
    flow = Flow(pipeline).capture(samples_1080p).batch().fork()
    for i in range(10):
        flow.render(mode=RenderMode.DISCARD)
    flow()
    thread.join()

if __name__ == '__main__':
    test_fork()