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

#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <fstream>
#include <algorithm> // for std::transform
#include <cctype> // for std::tolower

#include "buffer_probe.hpp"
#include "source_config.hpp"

namespace deepstream {

  static std::vector<std::string> parse_labels(const std::string& filename) {
    std::ifstream file(filename); // Open the file
    std::vector<std::string> lines; // Vector to store lines
  
    if (file.is_open()) { // Check if the file is open
      std::string line;
      while (std::getline(file, line)) { // Read each line
        std::string result=line;
        std::transform(line.begin(), line.end(), result.begin(), [](unsigned char c){ return std::tolower(c); });
        lines.push_back(result); // Store the line in the vector
      }
      file.close(); // Close the file
    } else {
        std::cerr << "Unable to open file: " << filename << std::endl;
    }
  
    return lines;
  }

class MsgMetaGenerator : public BufferProbe::IBatchMetadataOperator {
public:
  MsgMetaGenerator() : initialized_(false) {}
  virtual ~MsgMetaGenerator() {}

  virtual probeReturn handleData(BufferProbe& probe, BatchMetadata& data) {
    // Initialize only once
    if (!initialized_) {
      probe.getProperty("frame-interval", frame_interval);
      // frames % frame_interval below is undefined for 0 (SIGFPE) and matches
      // every frame for -1, so a bad config would either kill the pipeline or
      // silently emit on every frame. Fall back to the default instead.
      if (frame_interval <= 0) {
        std::cerr << "add_message_meta_probe: frame-interval must be > 0, got "
                  << frame_interval << "; using 1" << std::endl;
        frame_interval = 1;
      }
      std::string source_config;
      probe.getProperty("source-config", source_config);
      std::string label_file;
      probe.getProperty("label-file", label_file);

      if (!source_config.empty()) {
        SourceConfig source_config_obj(source_config);
        for (size_t i = 0; i < source_config_obj.nSources(); i++) {
          // multiple sources added
          std::string src_name = "src_";
          src_name += std::to_string(i);
          std::string uri = source_config_obj.getSensorInfo(i).uri;
          std::string sensor_id = source_config_obj.getSensorInfo(i).sensor_id;
          sensor_map_[i] = SensorInfo{uri, sensor_id, src_name};
        }
      }

      if (!label_file.empty()) {
        labels_ = parse_labels(label_file);
      }
      
      initialized_ = true;
    }

    FrameMetadata::Iterator frame_itr;
    for (data.initiateIterator(frame_itr); !frame_itr->done(); frame_itr->next())
    {
      auto source_id = (*frame_itr)->sourceId();
      int frames = frames_[source_id];
      ObjectMetadata::Iterator obj_itr;
      for ((*frame_itr)->initiateIterator(obj_itr); !obj_itr->done(); obj_itr->next()) {
        if (frames % frame_interval == 0)
        {
          EventMessageUserMetadata event_user_meta;
          if (data.acquire(event_user_meta)) {
            if(sensor_map_.empty()) {
              event_user_meta.generate(**obj_itr, **frame_itr, "N/A", "N/A", labels_);
              (*frame_itr)->append(event_user_meta);
            } else {
            auto itr = sensor_map_.find(source_id);
            if (itr != sensor_map_.end()) {
              const std::string sensor = itr->second.sensor_id;
              const std::string uri = itr->second.uri;
              event_user_meta.generate(**obj_itr, **frame_itr, sensor, uri, labels_);
              (*frame_itr)->append(event_user_meta);
            }
            }
          }
        }
      }
      frames_[source_id] = frames + 1;
    }

    return probeReturn::Probe_Ok;
  }

 protected:
  // Per source: a single shared counter would advance once per frame in the
  // batch, so frame_interval would apply to the batch rather than to each
  // source. With N sources and an interval that divides N, a source would then
  // emit either on every frame or never.
  std::map<uint32_t, int> frames_;   // source id -> frames seen from that source
  int frame_interval = 1;
  std::map<uint32_t, SensorInfo> sensor_map_;
  std::vector<std::string> labels_;
  bool initialized_;
};

}
