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

// #include <stdio.h>
// #include <cuda_runtime_api.h>
// #include <cuda.h>

#include "data_feeder.hpp"
#include <ds3d/gst/nvds3d_gst_plugin.h>
#include <ds3d/common/defines.h>
#include <ds3d/common/hpp/datamap.hpp>
#include "buffer.hpp"
#include "iostream"
// #include "gst-nvevent.h"

namespace deepstream
{

  class DS3DBuffer : public Buffer
  {
  public:
    DS3DBuffer() : Buffer() {}
    DS3DBuffer(OpaqueBuffer *buffer) : Buffer(buffer)
    {
      gst_buffer_unref(buffer_);
    }
  };

  class LiDARDataSource : public DataFeeder::IDataProvider
  {
  public:
    void initialize(DataFeeder &feeder)
    {
      using namespace ds3d;
      _initialized = true;
      ErrCode c;
      std::string configPath;
      Ptr<CustomLibFactory> customlib;
      std::string configContent;
      feeder.getProperty("config-path", configPath);
      std::cout << "lidar_feeder plugin config-path: " << configPath << std::endl;
      readFile(configPath, configContent);
      config::parseComponentConfig(configContent, configPath, config);
      c = gst::loadCustomProcessor(config, loader, customlib);
      DS_ASSERT(c == ErrCode::kGood);
    }

    Buffer read(DataFeeder &feeder, unsigned int size, bool &eos)
    {
      using namespace ds3d;
      ErrCode c;
      if (!_initialized)
      {
        initialize(feeder);
      }

      if (loader.state() == State::kNone)
      {
        DataProcessUserData *uData = (DataProcessUserData *)loader.getUserData();
        c = loader.start(uData->configContent, uData->configPath);
        DS_ASSERT(c == ErrCode::kGood);
      }
      c = loader.readData(datamap);
      DS_ASSERT(c >= ErrCode::kGood);
      eos = (c == ErrCode::KEndOfStream);
      if (!eos)
      {
        DS_ASSERT(datamap);

        c = NvDs3D_CreateGstBuf(buffer_, datamap.abiRef(), false);
        DS_ASSERT(c == ErrCode::kGood);
        return DS3DBuffer(buffer_);
      }
      else
      {
        std::cout << "EOS reached" << std::endl;
        return DS3DBuffer();
      }
    }


  private:
    OpaqueBuffer *buffer_ = nullptr;
    bool _initialized = false;
    ds3d::config::ComponentConfig config;
    ds3d::GuardDataMap datamap;
    ds3d::GuardDataLoader loader;
  };
}
