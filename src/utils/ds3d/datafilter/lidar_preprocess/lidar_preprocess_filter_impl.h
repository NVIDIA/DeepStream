/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef DS3D_DATAFILTER_LIDAR_PREPROCESS_FILTER_IMPL_H
#define DS3D_DATAFILTER_LIDAR_PREPROCESS_FILTER_IMPL_H

#include "lidar_preprocess_filter.h"

#include "ds3d/common/helper/safe_queue.h"
#include "ds3d/common/helper/cuda_utils.h"
#include "ds3d/common/hpp/datafilter.hpp"
#include "ds3d/common/impl/impl_datafilter.h"

#include "voxelization.hpp"
#include "lidar_preprocess_config.h"

namespace ds3d { namespace impl { namespace filter {

class LidarPreprocessFilter : public BaseImplDataFilter {
public:
    LidarPreprocessFilter() = default;
    ~LidarPreprocessFilter() override;

protected:
    ErrCode processImpl(
        GuardDataMap datamap, OnGuardDataCBImpl outputDataCb,
        OnGuardDataCBImpl inputConsumedCb) override;
    ErrCode stopImpl() override;
    ErrCode flushImpl() override;
    ErrCode startImpl(const std::string& content, const std::string& path);

private:
    ErrCode reserveInputMem(uint& devId, uint32_t count, int& batchSize);
    ErrCode doLidarPreProcess(GuardDataMap &dataMap, SharedBatchArray& batchArray);

    Config _config;
    std::unordered_map<std::string, std::shared_ptr<BufferPool<UniqCudaTensorBuf>>> _inputBuferPoolMap;
    volatile bool _inProcess = false;
    Ptr<CudaStream> _cudaStream;
    std::unique_ptr<bevfusion::pointpillars::Voxelization> _voxelization;
};

}}}  // namespace ds3d::impl::filter


#endif  // DS3D_DATAFILTER_LIDAR_PREPROCESS_FILTER_IMPL_H
