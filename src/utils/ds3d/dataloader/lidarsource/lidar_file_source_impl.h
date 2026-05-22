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

#ifndef _DS3D_DATALOADER_LIDARSOURCE_LIDAR_FILE_SOURCE_IMPL_H
#define _DS3D_DATALOADER_LIDARSOURCE_LIDAR_FILE_SOURCE_IMPL_H

#include "lidar_file_source.h"
#include "lidar_file_config.h"
#include "ds3d/common/hpp/profiling.hpp"
#include "ds3d/common/impl/impl_frames.h"
#include "ds3d/common/helper/memdata.h"
#include "ds3d/common/helper/cuda_utils.h"

namespace ds3d { namespace impl { namespace lidarsource {

using MemPtr = std::unique_ptr<MemData>;

/**
 * @brief Class for lidar data file reader,
 *
 *  Each synced dataloader must derive from SyncImplDataLoader and implement
 *  C++ interface:
 *    startImpl(...)
 *    readDataImpl(...)
 *    flushImpl(...)
 *    stopImpl(...)
 */
class LidarFileSourceImpl : public SyncImplDataLoader {
public:
    LidarFileSourceImpl() = default;
    ~LidarFileSourceImpl() override = default;

protected:
    /**
     * @brief Parse yaml config content and prepare all of the resource ready
     * to fill into each frame datamap.
     * @param[in] content  yaml config content.
     * @param[in] path the file location where the content is from.
     * @return, return ErrCode::kGood if successed.
     */
    ErrCode startImpl(const std::string& content, const std::string& path) override;
    /**
     * @brief Read a frame, create new output datamap and fill the frame into it.
     * @param[out] datamap  New allocated datamap with parsed frame buffer.
     * @return, return ErrCode::kGood if successed,
     *          return ErrCode::KEndOfStream if reach to end of file list
     *          return ErrCode::kByPass if some frame need to skip
     *          otherwise, return error.
     */
    ErrCode readDataImpl(GuardDataMap& datamap) override;
    /**
     * @brief flush all frames, Implementation of this function could be skipped.
     * @return, return ErrCode::kGood if successed,
     */
    ErrCode flushImpl() final { return ErrCode::kGood; }
    /**
     * @brief Stop and close all resources.
     *  Note: if there is some custom-lib handles still in use, close them in destructor
     * @return, return ErrCode::kGood if successed,
     */
    ErrCode stopImpl() override;

private:
    /**
     * @brief reserve CPU/GPU memory for buffer pool.
     */
    ErrCode reserveMem(
        Ptr<BufferPool<MemPtr>>& pool, size_t memSize, uint32_t count, const std::string& name);

    Ptr<BufferPool<MemPtr>> _bufMem;
    std::vector<uint8_t> _tmpStrideBuf;
    Ptr<MemData> _cpuSwapBuf;
    Config _config;
    profiling::FileReader _dataReader;
    uint32_t _bytesPerFrame = 0;
    uint32_t _bytesStrideFrame = 0;
    uint32_t _totalNumFrames = 0;
    uint32_t _totalFrameDuration = 0;
    uint32_t _readFrameCount = 0;
    bool _isFirstFrame = true;
};

}}}

#endif
