/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
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
