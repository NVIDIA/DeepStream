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

from pyservicemaker import Pipeline, Flow, RenderMode, BatchMetadataOperator, Probe, osd
import os

RENDER_MODE = RenderMode.DISPLAY if os.environ.get("DISPLAY") else RenderMode.DISCARD

samples_1080p = ["/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h265.mp4"]

class TranscodeOsdMarker(BatchMetadataOperator):
    def handle_metadata(self, batch_meta):
        for frame_meta in batch_meta.frame_items:
            text = f"Frame={frame_meta.frame_number}"
            display_meta = batch_meta.acquire_display_meta()
            label = osd.Text()
            label.display_text = text.encode('ascii')
            label.x_offset = 10
            label.y_offset = 12
            label.font.name = osd.FontFamily.Serif
            label.font.size = 12
            label.font.color = osd.Color(1.0, 1.0, 1.0, 1.0)
            label.set_bg_color = True
            label.bg_color = osd.Color(0.0, 0.0, 0.0, 1.0)
            display_meta.add_text(label)
            frame_meta.append(display_meta)

def test_transcode():
    pipeline = Pipeline("test")
    dest = "udp://localhost:5400"
    flow = Flow(pipeline).batch_capture(samples_1080p)
    flow = flow.attach(what=Probe("osd_marker", TranscodeOsdMarker())).fork()
    flow.encode(dest, sync=True)
    flow.encode("/tmp/sample.mp4")
    flow.encode("/tmp/sample.mjpg")
    flow.render(RENDER_MODE, sync=True)
    flow()

if __name__ == '__main__':
    test_transcode()
