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

from pyservicemaker import Pipeline, Flow, MediaType, RenderMode
import threading, time
import os

RENDER_MODE = RenderMode.DISPLAY if os.environ.get("DISPLAY") else RenderMode.DISCARD

def stop_pipeline(pipeline, timeout=1):
    time.sleep(timeout)
    print("Stop")
    pipeline.stop()

samples_720p = [
    "/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.h264",
    "file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.h264",
    "/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4",
    "file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4"
]
samples_1080p = [
    "/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h265.mp4",
    "file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h265.mp4",
    "/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4",
    "file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4"
]
AUDIO_FILE = "/opt/nvidia/deepstream/deepstream/samples/streams/sonyc_mixed_audio.wav"

source_list = [
    "/opt/nvidia/deepstream/deepstream/service-maker/sources/apps/cpp/deepstream_test5_app/source_list_dynamic.yaml",
    "/opt/nvidia/deepstream/deepstream/service-maker/sources/apps/cpp/deepstream_test5_app/source_list_static.yaml"
]

def test_batch_capture_video():
    for source in source_list:
        pipeline = Pipeline("test")
        thread = threading.Thread(target=stop_pipeline, args=(pipeline, 5))
        thread.start()
        Flow(pipeline).batch_capture(source, file_loop=False).render(RENDER_MODE)()
        thread.join()

    for input, d in [(samples_1080p, (1920, 1080)), (samples_720p, (1280, 720))]:
        pipeline = Pipeline("test")
        thread = threading.Thread(target=stop_pipeline, args=(pipeline, 5))
        thread.start()
        Flow(pipeline).batch_capture(input, width=d[0], height=d[1]).render(RENDER_MODE)()
        thread.join()

def test_capture_video():
    for sample in samples_720p + samples_1080p:
        pipeline = Pipeline("test")
        thread = threading.Thread(target=stop_pipeline, args=(pipeline,))
        thread.start()
        Flow(pipeline).capture([sample]).render(RENDER_MODE)()
        thread.join()

def test_capture_audio():
    pipeline = Pipeline("test-audio")
    thread = threading.Thread(target=stop_pipeline, args=(pipeline,))
    thread.start()
    Flow(pipeline).capture([AUDIO_FILE], media_types=MediaType.AUDIO).render(RenderMode.DISCARD)()
    thread.join()

def test_capture_audio_stream_builds():
    Flow(Pipeline("test-audio-stream")).capture(
        [AUDIO_FILE],
        media_types=MediaType.AUDIO
    ).render(RenderMode.STREAM, rtsp_port=8556, sync=False)

if __name__ == '__main__':
    test_capture_video()
    test_batch_capture_video()
    test_capture_audio()
