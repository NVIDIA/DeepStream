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
Python equivalent of the C++ kitti_dump_probe plugin.

Importing this module registers "kitti_dump_probe" in CommonFactory:

    pipeline.attach("pgie", "kitti_dump_probe", "kitti")

Each frame produces a text file:
    <kitti-dir>/<pad_index:03d>_<frame_num:06d>.txt

Supported properties:
  kitti-dir            : output directory (default /tmp/kitti/)
  tracker-kitti-output : dump tracker bbox/confidence instead of detector
                         bbox/confidence; also appends past-frame objects
                         (default false)
"""

import os

from pyservicemaker import probe, BatchMetadataOperator
from pyservicemaker._pydeepstream import (
    ObjectVisibilityUserMetadata,
    ObjectImageFootLocationUserMetadata,
    TrackerPastFrameUserMetadata,
)

UNTRACKED_OBJECT_ID = 0xFFFFFFFFFFFFFFFF
DEFAULT_KITTI_DIR   = "/tmp/kitti/"

# Meta type constants (mirror the C++ #defines in kitti_dump_probe.hpp)
NVDS_OBJ_VISIBILITY          = 20
NVDS_OBJ_IMAGE_FOOT_LOCATION = 21
NVDS_TRACKER_PAST_FRAME_META = 15


@probe(
    name="kitti_dump_probe",
    params={
        "kitti-dir":            ("string",  DEFAULT_KITTI_DIR, "output directory for KITTI label files"),
        "tracker-kitti-output": ("boolean", False,             "use tracker bbox/confidence instead of detector"),
    },
)
class NvDsKittiDump(BatchMetadataOperator):

    def __init__(self):
        super().__init__()
        self._kitti_dir    = DEFAULT_KITTI_DIR
        self._tracker_mode = False
        self._initialized  = False

    def _ensure_dir(self, path):
        os.makedirs(path, exist_ok=True)

    def _generate_inference_kitti_dump(self, batch_meta):
        for frame_meta in batch_meta.frame_items:
            path = os.path.join(
                self._kitti_dir,
                f"{frame_meta.pad_index:03d}_{frame_meta.frame_number:06d}.txt",
            )
            with open(path, "w") as f:
                for obj in frame_meta.object_items:
                    label  = obj.label or "unknown"
                    r      = obj.rect_params
                    left   = r.left
                    top    = r.top
                    right  = left + r.width
                    bottom = top  + r.height
                    conf   = obj.confidence
                    obj_id = obj.object_id

                    if obj_id == UNTRACKED_OBJECT_ID:
                        f.write(
                            f"{label} 0.0 0 0.0 "
                            f"{left:.6f} {top:.6f} {right:.6f} {bottom:.6f} "
                            f"0.0 0.0 0.0 0.0 0.0 0.0 0.0 {conf:.6f}\n"
                        )
                    else:
                        f.write(
                            f"{label} {obj_id} 0.0 0 0.0 "
                            f"{left:.6f} {top:.6f} {right:.6f} {bottom:.6f} "
                            f"0.0 0.0 0.0 0.0 0.0 0.0 0.0 {conf:.6f}\n"
                        )

    def _generate_tracker_kitti_dump(self, batch_meta):
        for frame_meta in batch_meta.frame_items:
            path = os.path.join(
                self._kitti_dir,
                f"{frame_meta.pad_index:03d}_{frame_meta.frame_number:06d}.txt",
            )
            with open(path, "w") as f:
                for obj in frame_meta.object_items:
                    label  = obj.label or "unknown"
                    bbox   = obj.nv_bbox_info          # tracker output bbox
                    left   = bbox["left"]
                    top    = bbox["top"]
                    right  = left + bbox["width"]
                    bottom = top  + bbox["height"]
                    conf   = obj.tracker_confidence
                    obj_id = obj.object_id

                    # Optional visibility and foot-location from user meta
                    visibility  = -1.0
                    x_img_foot  = -1.0
                    y_img_foot  = -1.0
                    write_proj  = False

                    for user_meta in obj.user_meta_items(NVDS_OBJ_VISIBILITY):
                        vis_meta   = ObjectVisibilityUserMetadata(user_meta)
                        visibility = vis_meta.visibility
                        write_proj = True

                    for user_meta in obj.user_meta_items(NVDS_OBJ_IMAGE_FOOT_LOCATION):
                        foot_meta  = ObjectImageFootLocationUserMetadata(user_meta)
                        loc        = foot_meta.image_foot_location
                        x_img_foot = loc["x"]
                        y_img_foot = loc["y"]
                        write_proj = True

                    if write_proj:
                        f.write(
                            f"{label} {obj_id} 0.0 0 0.0 "
                            f"{left:.6f} {top:.6f} {right:.6f} {bottom:.6f} "
                            f"0.0 0.0 0.0 0.0 0.0 0.0 0.0 {conf:.6f} "
                            f"{visibility:.6f} {x_img_foot:.6f} {y_img_foot:.6f}\n"
                        )
                    else:
                        f.write(
                            f"{label} {obj_id} 0.0 0 0.0 "
                            f"{left:.6f} {top:.6f} {right:.6f} {bottom:.6f} "
                            f"0.0 0.0 0.0 0.0 0.0 0.0 0.0 {conf:.6f}\n"
                        )

    def _generate_tracker_past_kitti_dump(self, batch_meta):
        for user_meta in batch_meta.user_meta_items(NVDS_TRACKER_PAST_FRAME_META):
            past_meta = TrackerPastFrameUserMetadata(user_meta)
            for stream in past_meta.past_frame_data:
                stream_id = stream["stream_id"]
                for obj in stream["objects"]:
                    obj_id = obj["unique_id"]
                    label  = obj["label"] or "unknown"
                    for frame in obj["frames"]:
                        path = os.path.join(
                            self._kitti_dir,
                            f"{stream_id:03d}_{frame['frame_num']:06d}.txt",
                        )
                        b      = frame["bbox"]
                        left   = b["left"]
                        top    = b["top"]
                        right  = left + b["width"]
                        bottom = top  + b["height"]
                        conf   = frame["confidence"]
                        with open(path, "a") as f:
                            f.write(
                                f"{label} {obj_id} 0.0 0 0.0 "
                                f"{left:.6f} {top:.6f} {right:.6f} {bottom:.6f} "
                                f"0.0 0.0 0.0 0.0 0.0 0.0 0.0 {conf:.6f}\n"
                            )

    def handle_metadata(self, batch_meta):
        if not self._initialized:
            kitti_dir = str(self.get_property("kitti-dir") or "")
            if kitti_dir:
                self._kitti_dir = kitti_dir
            self._tracker_mode = bool(self.get_property("tracker-kitti-output") or False)
            self._ensure_dir(self._kitti_dir)
            self._initialized = True

        if self._tracker_mode:
            self._generate_tracker_kitti_dump(batch_meta)
            self._generate_tracker_past_kitti_dump(batch_meta)
        else:
            self._generate_inference_kitti_dump(batch_meta)
