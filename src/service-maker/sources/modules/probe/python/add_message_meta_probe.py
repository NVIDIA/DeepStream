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
Python equivalent of the C++ add_message_meta_probe plugin.

Importing this module registers "add_message_meta_probe" in CommonFactory:

    pipeline.attach("osd", "add_message_meta_probe", "msg_meta")

Supported properties (set via YAML or probe params):
  frame-interval  : generate event metadata every N frames (default 1)
  source-config   : path to a source-config YAML file with sensor info
  label-file      : path to a text file with one class label per line
"""

from pyservicemaker import probe, BatchMetadataOperator, SourceConfig


@probe(
    name="add_message_meta_probe",
    params={
        "frame-interval": ("integer", 1,   "generate event metadata every N frames"),
        "source-config":  ("string",  "",  "path to source config YAML"),
        "label-file":     ("string",  "",  "path to label file (one label per line)"),
    },
)
class MsgMetaGenerator(BatchMetadataOperator):

    def __init__(self):
        super().__init__()
        self._initialized = False
        # Per source: a single shared counter would advance once per frame in
        # the batch, so frame-interval would apply to the batch rather than to
        # each source. With N sources and an interval that divides N, a source
        # would then emit either on every frame or never.
        self._frames = {}       # source_id -> frames seen from that source
        self._frame_interval = 1
        self._sensor_map = {}   # source_id -> (sensor_id, uri)
        self._labels = []

    def _init_once(self):
        # frames % interval below is undefined for 0 and matches every frame
        # for -1, so a bad config would either raise or silently emit on every
        # frame. Read and convert once, then fall back to the default.
        frame_interval = int(self.get_property("frame-interval") or 1)
        if frame_interval <= 0:
            print(f"add_message_meta_probe: frame-interval must be > 0, got "
                  f"{frame_interval}; using 1", flush=True)
            frame_interval = 1
        self._frame_interval = frame_interval

        source_config_path = str(self.get_property("source-config") or "")
        if source_config_path:
            sc = SourceConfig()
            sc.load(source_config_path)
            for i, info in enumerate(sc.sensor_list):
                self._sensor_map[i] = (info.sensor_id, info.uri)

        label_file = str(self.get_property("label-file") or "")
        if label_file:
            with open(label_file) as f:
                self._labels = [line.strip().lower() for line in f if line.strip()]

        self._initialized = True

    def handle_metadata(self, batch_meta):
        if not self._initialized:
            self._init_once()

        for frame_meta in batch_meta.frame_items:
            source_id = frame_meta.source_id
            frames = self._frames.get(source_id, 0)
            for obj_meta in frame_meta.object_items:
                if frames % self._frame_interval == 0:
                    event_meta = batch_meta.acquire_event_message_meta()
                    sensor_id, uri = self._sensor_map.get(source_id, ("N/A", "N/A"))
                    event_meta.generate(obj_meta, frame_meta, sensor_id, uri, self._labels)
                    frame_meta.append(event_meta)
            self._frames[source_id] = frames + 1
