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

#include <iostream>
#include <ds3d/gst/nvds3d_gst_plugin.h>
#include <ds3d/common/defines.h>
#include <ds3d/common/hpp/datamap.hpp>

#include "data_receiver.hpp"


namespace deepstream {

class EnsembleRender : public DataReceiver::IDataConsumer {
public:
  void initialize(DataReceiver& receiver) {
    using namespace ds3d;
    ErrCode c;
    if (!_initialized) {
      _initialized = true;
      std::string configPath;
      Ptr<CustomLibFactory> customlib;
      std::string configContent;
      receiver.getProperty("config-path", configPath);
      std::cout << "ensemble_render plugin config-path: " << configPath << std::endl;
      readFile(configPath, configContent);
      config::parseComponentConfig(configContent, configPath, config);
      c = gst::loadCustomProcessor(config, render, customlib);
      DS_ASSERT(c == ErrCode::kGood);
      c = render.start(config.rawContent, config.filePath);
      DS_ASSERT(c == ErrCode::kGood);
    }
  }

  virtual int consume(DataReceiver& receiver, Buffer buffer) {
    if (!_initialized) initialize(receiver);
    DS_ASSERT(render.state() == ds3d::State::kRunning);
    ds3d::ErrCode c;
    const ds3d::abiRefDataMap* datamap = nullptr;
    GstBuffer* buf = buffer.give();
    gst_buffer_unref(buf);
    c = NvDs3D_Find1stDataMap(buf, datamap);
    DS_ASSERT(c == ds3d::ErrCode::kGood);
    DS_ASSERT(datamap);
    ds3d::GuardDataMap guardData(*datamap);
    c = render.render(guardData, [](ds3d::ErrCode err, const ds3d::abiRefDataMap*) {});
    DS_ASSERT(c >= ds3d::ErrCode::kGood);
    return 1;
  }

private:
  OpaqueBuffer *buffer_ = nullptr;
  ds3d::GuardDataRender render;
  ds3d::config::ComponentConfig config;
  bool _initialized = false;
};
}
