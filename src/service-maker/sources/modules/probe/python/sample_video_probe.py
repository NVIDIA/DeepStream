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

"""
Python equivalent of the C++ sample_video_probe plugin.

Importing this module registers "sample_video_probe" in CommonFactory so it
can be attached to pipeline nodes by name string, identically to a compiled C++
plugin:

    pipeline.attach("pgie", "sample_video_probe", "my_probe")
    pipeline["my_probe"].set({"font-size": 24})
"""

from pyservicemaker import probe, BatchMetadataOperator, osd

PGIE_CLASS_ID_VEHICLE = 0
PGIE_CLASS_ID_PERSON  = 2


@probe(
    name="sample_video_probe",
    params={"font-size": ("integer", 12, "size of the font to show the counter")}
)
class CountMarker(BatchMetadataOperator):
    """Count vehicles and persons per frame and draw the result on the OSD overlay."""

    def __init__(self):
        super().__init__()

    def handle_metadata(self, batch_meta):
        font_size = int(self.get_property("font-size") or 12)
        for frame_meta in batch_meta.frame_items:
            vehicle_count = 0
            person_count  = 0

            for obj in frame_meta.object_items:
                if obj.class_id == PGIE_CLASS_ID_VEHICLE:
                    vehicle_count += 1
                elif obj.class_id == PGIE_CLASS_ID_PERSON:
                    person_count += 1

            display_meta = batch_meta.acquire_display_meta()
            text = osd.Text()
            text.display_text = f"Person={person_count},Vehicle={vehicle_count}"
            text.x_offset = 10
            text.y_offset = 12
            text.font.name = osd.FontFamily.Serif
            text.font.size = font_size
            text.font.color = osd.Color(1.0, 1.0, 1.0, 1.0)
            text.set_bg_color = 1
            text.bg_color = osd.Color(0.0, 0.0, 0.0, 1.0)
            display_meta.add_text(text)
            frame_meta.append(display_meta)
