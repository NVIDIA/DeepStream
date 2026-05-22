/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <string>
#include <unordered_map>
#include <vector>
#include <chrono>
#include <mutex>
#include <condition_variable>

#include "signal_emitter.hpp"
#include "lib/msgbroker_c2d_receiver.hpp"

// static bool flag = false;
static deepstream::Cloud2DeviceReceiver receiver;

namespace deepstream {

class SmartRecordingAction : 
 public SignalEmitter::IActionOwner,
 public Cloud2DeviceReceiver::ISmartRecordingController {

 public:
  SmartRecordingAction () {}

  virtual ~SmartRecordingAction () {}
  
  virtual std::vector<std::string> list() {
    std::vector<std::string> actions;
    actions.push_back("start-sr");
    actions.push_back("stop-sr");
    return actions;
  }

  virtual void onAttached(SignalEmitter* emitter,
                          const std::string& action,
                          const std::string& object_name
  ) {
    emitter_ = emitter;
  
    if (!receiver.hasHandler(this)) {
      receiver.addHandler(this);
    }
    if (!receiver.isConnected()) {
      Cloud2DeviceReceiver::Config config;
      emitter->getProperty("proto-lib", config.proto_lib);
      emitter->getProperty("conn-str", config.conn_str);
      emitter->getProperty("msgconv-config-file", config.sensor_list_file);
      emitter->getProperty("proto-config-file", config.config_file_path);
      emitter->getProperty("topic-list", config.topicList);
      receiver.connect(config);
    }
    // to maintain a map from the source id to object, we assume the object name
    // be formatted as something like "name_id"
    auto pos = object_name.rfind('_');
    int64_t source_id = std::stoll(object_name.substr(pos+1));
    if (object_map_.find(source_id) == object_map_.end()) {
      object_map_.insert({source_id, object_name});
    }
  }

  virtual void startSmartRecord(int64_t camera_id, uint32_t *sessionId,
                      unsigned int startTime, unsigned int duration,
                      void *userData) {
    if (!emitter_) {
      throw std::runtime_error("Signal Emitter empty");
    }
    if (object_map_.find(camera_id) == object_map_.end()) {
      // error
      return;
    }
    emitter_->emit("start-sr", object_map_[camera_id],
                    sessionId, startTime, duration, userData, nullptr);
  }

  virtual void stopSmartRecord(int64_t camera_id, uint32_t sessionId) {
    if (!emitter_) {
      throw std::runtime_error("Signal Emitter empty");
    }
    if (object_map_.find(camera_id) == object_map_.end()) {
      // error
      return;
    }
    emitter_->emit("stop-sr", object_map_[camera_id],
                   sessionId, nullptr);
  }

 private:
  SignalEmitter* emitter_ = nullptr;
  std::map<int64_t, std::string> object_map_;
};

}