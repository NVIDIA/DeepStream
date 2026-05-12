/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
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
