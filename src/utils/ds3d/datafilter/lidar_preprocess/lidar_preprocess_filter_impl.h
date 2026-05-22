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
