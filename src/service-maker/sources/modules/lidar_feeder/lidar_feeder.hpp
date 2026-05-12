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
