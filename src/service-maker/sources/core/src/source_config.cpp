/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "source_config.hpp"

using namespace deepstream;

#define KEY_SOURCE_LIST    "source-list"
#define KEY_SOURCE_CONFIG  "source-config"
#define KEY_SOURCE_CONFIG_BIN "source-bin"
#define KEY_SOURCE_CONFIG_PROPERTIES "properties"

#define KEY_CAMERA_LIST "camera-list"

SourceConfig::SourceConfig(const std::string& config_file) {
  YAML::Node root = YAML::LoadFile(config_file);
  for (YAML::const_iterator it1 = root.begin(); it1 != root.end(); ++it1) {
    std::string key = it1->first.as<std::string>();
    if (key == KEY_SOURCE_LIST) {
      YAML::Node list_node = it1->second;
      for (YAML::const_iterator it2 = list_node.begin(); it2 != list_node.end(); ++it2) {
        SensorInfo sensor_info = {"N/A", "N/A", "N/A"};
        const YAML::Node& sensor_node = *it2;
        for (YAML::const_iterator it3 = sensor_node.begin(); it3 != sensor_node.end(); ++it3) {
          std::string name = it3->first.as<std::string>();
          std::string value = it3->second.as<std::string>();
          if (name == "uri") {
            sensor_info.uri = value;
          } else if (name == "sensor-id") {
            sensor_info.sensor_id = value;
          } else if (name == "sensor-name") {
            sensor_info.sensor_name = value;
          }
          else
          {
            std::string what = "Invalid key in source config: ";
            throw std::runtime_error(what + name);
          }
        }
        sensor_info_.push_back(sensor_info);
      }
    }
    else if (key == KEY_CAMERA_LIST)
    {
      use_camerabin_ = true;
      YAML::Node list_node = it1->second;
      for (YAML::const_iterator it2 = list_node.begin(); it2 != list_node.end(); ++it2)
      {
        CameraInfo camera_info;
        const YAML::Node& sensor_node = *it2;
        for (YAML::const_iterator it3 = sensor_node.begin(); it3 != sensor_node.end(); ++it3)
        {
          std::string name = it3->first.as<std::string>();
          std::string value = it3->second.as<std::string>();
          if (name == "camera-type")
          {
            camera_info.camera_type = value;
          }
          else if (name == "camera-v4l2-dev-node")
          {
            camera_info.camera_v4l2_dev_node = value;
          }
          else if (name == "camera-csi-sensor-id")
          {
            camera_info.camera_csi_sensor_id = std::stoi(value);
          }
          else if (name == "camera-width")
          {
            camera_info.camera_width = value;
          }
          else if (name == "camera-height")
          {
            camera_info.camera_height = value;
          }
          else if (name == "camera-fps-n")
          {
            camera_info.camera_fps_n = value;
          }
          else if (name == "camera-fps-d")
          {
            camera_info.camera_fps_d = value;
          }
          else if (name == "gpu-id")
          {
            camera_info.gpu_id = std::stoi(value);
          }
          else if (name == "nvbuf-mem-type")
          {
            camera_info.nvbuf_mem_type = std::stoi(value);
          }
          else if (name == "nvvideoconvert-copy-hw")
          {
            camera_info.nvvideoconvert_copy_hw = std::stoi(value);
          }
          else
          {
            std::string what = "Invalid key in source config: ";
            throw std::runtime_error(what + name);
          }
        }
        camera_info_.push_back(camera_info);
      }
    }
    else if (key == KEY_SOURCE_CONFIG)
    {
      YAML::Node config_node = it1->second;
      for (YAML::const_iterator it2 = config_node.begin(); it2 != config_node.end(); ++it2)
      {
        std::string name = it2->first.as<std::string>();
        if (name == KEY_SOURCE_CONFIG_BIN)
        {
          use_nvmultiurisrcbin_ = it2->second.as<std::string>() == "nvmultiurisrcbin";
          use_nvurisrcbin_ = it2->second.as<std::string>() == "nvurisrcbin";
        }
        else if (name == KEY_SOURCE_CONFIG_PROPERTIES)
        {
          properties_ = it2->second;
        }
        else
        {
          std::string what = "Invalid key in source config: ";
          throw std::runtime_error(what + name);
        }
      }
    }
  }
}

uint32_t SourceConfig::nSources() const {
  return sensor_info_.size();
}

uint32_t SourceConfig::nCameraSources() const
{
  return camera_info_.size();
}

std::string SourceConfig::listSensorIds() const {
  std::string str = "";
  size_t n = sensor_info_.size();
  for (size_t i = 0; i < n; i++) {
    str += sensor_info_[i].sensor_id;
    if (i < n - 1) {
        str += ",";
    }
  }
  return str;
}

std::string SourceConfig::listSensorNames() const {
  std::string str = "";
  size_t n = sensor_info_.size();
  for (size_t i = 0; i < n; i++) {
    str += sensor_info_[i].sensor_name;
    if (i < n - 1) {
        str += ",";
    }
  }
  return str;
}

std::string SourceConfig::listUris() const {
  std::string str = "";
  size_t n = sensor_info_.size();
  for (size_t i = 0; i < n; i++) {
    str += sensor_info_[i].uri;
    if (i < n - 1) {
        str += ",";
    }
  }
  return str;
}

SensorInfo SourceConfig::getSensorInfo(uint32_t index) const {
  return sensor_info_[index];
}

CameraInfo SourceConfig::getCameraInfo(uint32_t index) const
{
  return camera_info_[index];
}

const YAML::Node& SourceConfig::getProperties() const {
  return properties_;
}

bool SourceConfig::useMultiUriSrcBin() const {
  return use_nvmultiurisrcbin_;
}

bool SourceConfig::useUriSrcBin() const
{
  return use_nvurisrcbin_;
}

bool SourceConfig::useCameraBin() const
{
  return use_camerabin_;
}
