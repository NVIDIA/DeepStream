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

import yaml
from collections import namedtuple
from dataclasses import dataclass
from typing import Optional

SensorInfo = namedtuple("SensorInfo", ["sensor_id", "sensor_name", "uri"])
CameraInfo = namedtuple("CameraInfo", ["camera_type", "camera_video_format", "camera_width", "camera_height", "camera_fps_n", "camera_fps_d",
                                       "camera_csi_sensor_id", "camera_v4l2_dev_node", "gpu_id", "nvbuf_mem_type",
                                       "nvvideoconvert_copy_hw"])

class SourceConfig:
    """Class for parsing source configuration"""
    def __init__(self):
        self._sensor_list = []
        self._camera_list = []
        self._source_type = None
        self._source_properties = dict()

    @property
    def sensor_list(self):
        return self._sensor_list

    @property
    def camera_list(self):
        return self._camera_list

    @property
    def source_type(self):
        return self._source_type

    @property
    def source_properties(self):
        return self._source_properties

    def load(self, config_file: str):
        """Load source configurations from a YAML file"""
        with open(config_file, 'r') as file:
            doc = yaml.safe_load(file)
            if doc:
                source_list = doc.get("source-list")
                camera_list = doc.get("camera-list")
                source_config = doc.get("source-config")
                if source_list is not None:
                    for source in source_list:
                        uri = source.get("uri")
                        sensor_id = source.get("sensor-id")
                        sensor_name = source.get("sensor-name")
                        self._sensor_list.append(SensorInfo(sensor_id, sensor_name, uri))
                if camera_list is not None:
                    for camera in camera_list:
                        camera_type = camera.get("camera-type")
                        camera_video_format = camera.get("camera-video-format")
                        camera_width = camera.get("camera-width")
                        camera_height = camera.get("camera-height")
                        camera_fps_n = camera.get("camera-fps-n")
                        camera_fps_d = camera.get("camera-fps-d")
                        camera_csi_sensor_id = camera.get("camera-csi-sensor-id")
                        camera_v4l2_dev_node = camera.get("camera-v4l2-dev-node")
                        gpu_id = camera.get("gpu-id")
                        nvbuf_mem_type = camera.get("nvbuf-mem-type")
                        nvvideoconvert_copy_hw = camera.get("nvvideoconvert-copy-hw")
                        self._camera_list.append(CameraInfo(camera_type, camera_video_format, camera_width, camera_height, camera_fps_n, camera_fps_d,
                                                            camera_csi_sensor_id, camera_v4l2_dev_node, gpu_id, nvbuf_mem_type,
                                                            nvvideoconvert_copy_hw))
                if source_config:
                    self._source_type = source_config.get("source-bin")
                    self._source_properties = source_config.get("properties")
                if camera_list is not None and len(camera_list):
                    self._source_type = "camerabin"


@dataclass
class RecordConfig:
    """Unified recording configuration for DeepStream pipeline.

    Set ``recording_type`` to either "local" or "cloud".

    - When "cloud": provide broker/protocol settings (Kafka, etc.).
    - When "local": only local recording settings are needed.

    Cloud parameters (only used when recording_type == "cloud"):
        - proto_lib: Path to protocol library (e.g., Kafka proto lib)
        - conn_str: Connection string (e.g., "localhost;9092")
        - msgconv_config_file: Message converter config file path
        - proto_config_file: Protocol adaptor config file path
        - topic_list: Comma-separated list of topics

    Common recording parameters (applied to smart-rec properties):
        - rec_cache: Cache size in seconds (default 20)
        - rec_container: Container format (0: MP4, 1: MKV)
        - rec_dir_path: Output directory path
        - rec_mode: 0 both, 1 video-only, 2 audio-only
    """
    recording_type: str = "local"  # "local" or "cloud"

    # Cloud-only fields
    proto_lib: Optional[str] = None
    conn_str: Optional[str] = None
    msgconv_config_file: Optional[str] = None
    proto_config_file: Optional[str] = None
    topic_list: Optional[str] = None

    # Common recording fields
    rec_cache: int = 20
    rec_container: int = 0
    rec_dir_path: str = "."
    rec_mode: int = 0

    def __post_init__(self):
        # Enforce mandatory cloud properties when recording_type is 'cloud'
        if isinstance(self.recording_type, str) and self.recording_type.lower() == "cloud":
            required = [
                "proto_lib",
                "conn_str",
                "msgconv_config_file",
                "proto_config_file",
                "topic_list",
            ]
            missing = [name for name in required if not getattr(self, name)]
            if missing:
                raise ValueError(f"Missing required cloud recording fields: {', '.join(missing)}")