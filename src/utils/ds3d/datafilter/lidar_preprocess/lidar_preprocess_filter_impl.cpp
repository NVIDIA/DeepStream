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

#include "lidar_preprocess_filter_impl.h"
#include "ds3d/common/common.h"
#include "ds3d/common/helper/check.hpp"
#include "ds3d/common/impl/impl_frames.h"
#include "infer_cuda_utils.h"

#include <unistd.h>
namespace ds3d { namespace impl { namespace filter {

using namespace nvdsinferserver;

ErrCode
LidarPreprocessFilter::reserveInputMem(uint& devId, uint32_t count, int& batchSize)
{
    for (auto iter : _config.inputLayersDes) {
        const auto& des = iter.second;
        auto pool = std::make_shared<BufferPool<UniqCudaTensorBuf>>(des.name);
        DS_ASSERT(pool);
        InferMemType memType = _config.inputTensorMemType;
        LOG_INFO("reserveInputMem mt:%d\n", (int)memType);
        for (uint32_t i = 0; i < count; ++i) {
            //createTensorBuf only support gpucuda  and cpucuda.
            UniqCudaTensorBuf ptr =
                createTensorBuf(des.dims, des.dataType, batchSize, des.name, memType, devId, false);
            pool->setBuffer(std::move(ptr));
        }
        DS3D_FAILED_RETURN(
            pool->size() == count, ErrCode::kMem, "allocate memory failed on %s", des.name.c_str());
        _inputBuferPoolMap[des.name] = pool;
    }
    return ErrCode::kGood;
}

LidarPreprocessFilter::~LidarPreprocessFilter()
{
    stop_i();
}

ErrCode
LidarPreprocessFilter::processImpl(
    GuardDataMap datamap, OnGuardDataCBImpl outputDataCb, OnGuardDataCBImpl inputConsumedCb)
{
    LOG_DEBUG("lidar preprocess datafilter is starting to filter datamap");
    std::unique_lock<ImplMutex> locker(mutex());
    Frame2DGuard lidarFrame;

    // get data from datamap
    const std::string& filterInputDatamapKey = _config.filterInputDatamapKey;

    if (!filterInputDatamapKey.empty()) {
        datamap.printDebug();
        DS3D_FAILED_RETURN(
            datamap.hasData(filterInputDatamapKey), ErrCode::kConfig,
            "lidarpreprocess configured datamap key but no lidar frame found in datamap");

        DS3D_ERROR_RETURN(
            datamap.getGuardData(filterInputDatamapKey, lidarFrame),
            "No lidar data found in datamap from lidarpreprocess filter");
    }

    // prepare buffer for lidar process
    SharedBatchArray batchArray = std::make_shared<BaseBatchArray>();
    for (auto iter : _config.inputLayersDes) {
        const auto& des = iter.second;
        auto bufferPool = _inputBuferPoolMap[des.name];
        auto buffer = bufferPool->acquireBuffer();
        LOG_DEBUG("add buffer name: %s", des.name.c_str());
        batchArray->addBuf(std::move(buffer));
    }

    DS3D_ERROR_RETURN(doLidarPreProcess(datamap, batchArray),
        "ds3d lidar preprocess failed");

    inputConsumedCb(ErrCode::kGood, datamap);
    outputDataCb(ErrCode::kGood, datamap);

    return ErrCode::kGood;
}

ErrCode LidarPreprocessFilter::doLidarPreProcess(GuardDataMap &dataMap, SharedBatchArray& batchArray) {
    if (_voxelization.get() == nullptr) {
        bevfusion::pointpillars::VoxelizationParameter voxelization_param;
        voxelization_param.min_range  = ds3d::Float3(0, -51.2, -5);
        voxelization_param.max_range  = ds3d::Float3(102.4, 51.2, 3);
        voxelization_param.voxel_size = ds3d::Float3(0.8, 0.8, 8);
        voxelization_param.grid_size  = voxelization_param.compute_grid_size(
            voxelization_param.max_range, voxelization_param.min_range,
            voxelization_param.voxel_size);
        voxelization_param.max_points_per_voxel = 10;
        voxelization_param.input_feature = 4;
        voxelization_param.max_batch  = 4;
        voxelization_param.max_voxels = 8000;
        voxelization_param.max_points = 100000;
        _voxelization = bevfusion::pointpillars::create_voxelization(voxelization_param);
    }

    Shape shape = {0};
    std::vector<float *> radars; // K x C(x, y, z, intensity)
    std::vector<int> Ks;
    ErrCode code;

    for (auto key : _config.lidarDataFrom) {
        FrameGuard lidarFrame;
        code = dataMap.getGuardData(key, lidarFrame);
        if (!isGood(code)) {
          LOG_ERROR("dataMap getGuardData %s kLidarFrame failed\n", key.c_str());
          return code;
        }
        radars.emplace_back((float *)lidarFrame->base());
        shape = lidarFrame->shape();
        Ks.emplace_back(shape.d[0]);
    }

    _voxelization->forward(radars.data(), Ks.data(), radars.size(), _cudaStream->get());

    // copy tensor to frame, and add to datamap
    for (uint32_t layerIdx = 0; layerIdx < batchArray->getSize(); layerIdx++) {
        auto buf = batchArray->getSafeBuf(layerIdx);
        auto des = buf->getBufDesc();
        auto buffer_size = des.dims.numElements * des.elementSize;
        shape.numDims = des.dims.numDims;
        for (uint32_t i = 0; i < des.dims.numDims; i++) {
            shape.d[i] = des.dims.d[i];
        }

        const void *src = nullptr;
        auto key = "";
        if (des.name == "N") {
            // kLidarPointNumTensor
            src = _voxelization->num_voxels_device();
            key = kLidarPointNumTensor;
        } else if (des.name == "coords") {
            // kLidarCoordTensor
            src = _voxelization->indices();
            key = kLidarCoordTensor;
        } else if (des.name == "feats") {
            // kLidarFeatureTensor
            src = _voxelization->features();
            key = kLidarFeatureTensor;
        } else {
            LOG_WARNING("unknown layer name %s !!! ", des.name.c_str());
        }

        checkCudaErrors(
            cudaMemcpy(buf->getBufPtr(0), src, buffer_size, cudaMemcpyDefault));
        auto tensorFrame = impl::WrapFrame<uint8_t, FrameType::kCustom>(
            buf->getBufPtr(0), buffer_size, shape, MemType::kGpuCuda, 0, [buf](void *){});
        code = dataMap.setGuardData(key, tensorFrame);
        if (!isGood(code)) {
            LOG_ERROR("dataMap setGuardData %s failed \n", des.name.c_str());
            return ErrCode::kUnknown;
        }
    }
    return ErrCode::kGood;
}

ErrCode
LidarPreprocessFilter::stopImpl()
{
    LOG_INFO("lidarinference datafilter is closing");
    std::unique_lock<ImplMutex> locker(mutex());
    if (_cudaStream) {
        DS3D_CHECK_CUDA_ERROR(
            cudaStreamSynchronize(_cudaStream->get()), return ErrCode::kCuda,
            "Failed to synchronize cuda stream on ds3d inference filter");
    }

    for (auto buffer:_inputBuferPoolMap){
        (buffer.second).reset();
    }
    _inputBuferPoolMap.clear();
    _cudaStream.reset();

    return ErrCode::kGood;
}

ErrCode
LidarPreprocessFilter::flushImpl()
{
    LOG_INFO("lidar preprocess datafilter is flushing");

    std::unique_lock<ImplMutex> locker(mutex());

    return ErrCode::kGood;
}

ErrCode
LidarPreprocessFilter::startImpl(const std::string& content, const std::string& path)
{
    LOG_INFO("lidar preprocess datafilter is starting");
    std::unique_lock<ImplMutex> locker(mutex());
    ErrCode code =
        config::CatchYamlCall([&, this]() { return parseConfig(content, path, _config); });
    DS3D_ERROR_RETURN(code, "parse datafilter lidar preprocess config: %s failed", path.c_str());
    DS3D_FAILED_RETURN(
        _config.check(), ErrCode::kConfig, "check datafilter lidar preprocess config: %s failed",
        path.c_str());

    _cudaStream.reset(new CudaStream(cudaStreamDefault, 0));
    DS_ASSERT(_cudaStream);

    //prepare bufferpool for model input data.
    int batchSize = 0;  //0 means no batching
    reserveInputMem(_config.gpuid, _config.memPoolSize, batchSize);

    setInputCaps(_config.compConfig.gstInCaps);
    setOutputCaps(_config.compConfig.gstOutCaps);

    return ErrCode::kGood;
}

}}}  // namespace ds3d::impl::filter

using namespace ds3d;

DS3D_EXTERN_C_BEGIN
DS3D_EXPORT_API abiRefDataFilter*
createLidarPreprocessFilter()
{
    return NewAbiRef<abiDataFilter>(new impl::filter::LidarPreprocessFilter);
}

DS3D_EXTERN_C_END
