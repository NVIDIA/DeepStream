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

#include "lidar_file_source_impl.h"
#include <sys/stat.h>

namespace ds3d { namespace impl { namespace lidarsource {

constexpr uint64_t kMsToNas = 1000000ULL;
constexpr uint64_t kUsToNas = 1000ULL;

ErrCode
LidarFileSourceImpl::reserveMem(
    Ptr<BufferPool<MemPtr>>& pool, size_t memSize, uint32_t count, const std::string& name)
{
    pool = std::make_shared<BufferPool<MemPtr>>(name);
    DS_ASSERT(pool);
    for (uint32_t i = 0; i < count; ++i) {
        MemPtr ptr;
        if (isCpuMem(_config.memType)) {
            ptr = CpuMemBuf::CreateBuf(memSize);
        } else {
            ptr = GpuCudaMemBuf::CreateBuf(memSize, _config.gpuId);
        }
        pool->setBuffer(std::move(ptr));
    }
    DS3D_FAILED_RETURN(
        pool->size() == count, ErrCode::kMem, "allocate CPU memory failed on %s", name.c_str());
    return ErrCode::kGood;
}


ErrCode
LidarFileSourceImpl::startImpl(const std::string& content, const std::string& path)
{
    LOG_INFO("LidarFileSource dataloader is starting");
    ErrCode code =
        config::CatchYamlCall([&, this]() { return parseConfig(content, path, _config); });
    DS3D_ERROR_RETURN(code, "parse config: %s failed", path.c_str());

    setOutputCaps(_config.compConfig.gstOutCaps);
    LOG_DEBUG("LidarFileSource dataloader is started");

    // elementSize could be 3/4
    _bytesPerFrame = _config.pointNums * dataTypeBytes(_config.dataType) * _config.elementSize;
    // elementStride could be 3/4/5, nuscene has 5 elements
    _bytesStrideFrame = _config.pointNums * dataTypeBytes(_config.dataType) * _config.elementStride;
    assert(_bytesStrideFrame >= _bytesPerFrame);
    if (_bytesStrideFrame > _bytesPerFrame) {
        _tmpStrideBuf.resize(_bytesStrideFrame);
    }
    _totalNumFrames = _config.dataParas[0].size();
    _totalFrameDuration = _config.lastFileTimestamp + _config.frameDuration;

    // allocate CPU/GPU memory for output buffer pool
    DS3D_ERROR_RETURN(
        reserveMem(_bufMem, _bytesPerFrame, _config.memPoolSize, "bufMem"),
        "lidarfilesource loader reserve points memory failed");
    if (isGpuMem(_config.memType)) {
        _cpuSwapBuf = CpuMemBuf::CreateBuf(_bytesPerFrame);
        DS_ASSERT(_cpuSwapBuf);
    }

    return ErrCode::kGood;
}

ErrCode
LidarFileSourceImpl::readDataImpl(GuardDataMap& outData)
{
    LOG_DEBUG("LidarFileSource dataloader is reading data %d",(int)_config.dataParas.size());

    GuardDataMap datamap(NvDs3d_CreateDataHashMap(), true);
    uint64_t timestamp = 0;

    for(size_t i = 0; i < _config.dataParas.size(); i++) {
        std::deque<std::map<uint64_t, std::string>> dataQueue = _config.dataParas[i];
        if (dataQueue.size() == 0) {
            LOG_INFO("Lidar source read completely, return EOS");
            return ErrCode::KEndOfStream;
        }

        std::map<uint64_t, std::string> dataFileParas;
        std::string filepath = "";

        if(!_config.fileLoop) {
            dataFileParas = dataQueue.front();
        } else {
            dataFileParas = dataQueue[_readFrameCount%_totalNumFrames];
        }
        for(auto i = dataFileParas.begin(); i != dataFileParas.end(); ++i) {
            timestamp = i->first;
            filepath = i->second;
        }

        if (_config.fileLoop) {
            timestamp = timestamp + (((uint64_t)_readFrameCount / _totalNumFrames) * (_totalFrameDuration));
        }

        LOG_DEBUG("lidar data file name %s", filepath.c_str());
        DS3D_FAILED_RETURN(
                _dataReader.open(filepath), ErrCode::kConfig, "open lidar source file: %s failed.",
                filepath.c_str());
        struct stat statBuf;
        stat(filepath.c_str(), &statBuf);
        uint32_t fileSize = statBuf.st_size;
        LOG_DEBUG("lidar data file size %d", fileSize);

        // elementStride could be 3/4/5, nuscene has 5 elements
        // elementStride >= elementSize
        uint32_t pointStrideBytes = dataTypeBytes(_config.dataType) * _config.elementStride;
        assert(pointStrideBytes);
        uint32_t pointsToRead = fileSize / pointStrideBytes;
        if (pointsToRead > _config.pointNums) {
            pointsToRead = _config.pointNums;
            LOG_DEBUG(
                    "Lidar data file: %s size is larger than expected size! fileSize %d expected frame "
                    "size %d",
                    filepath.c_str(), fileSize, _bytesStrideFrame);
        } else if (pointsToRead < _config.pointNums) {
            LOG_DEBUG(
                    "Lidar data file: %s size is less than expected size! fileSize %d expected frame size "
                    "%d",
                    filepath.c_str(), fileSize, _bytesStrideFrame);
        }

        DS_ASSERT(_bufMem);
        Ptr<MemData> dstBuf = _bufMem->acquireBuffer();
        Ptr<MemData> cpuBuf;

        DS_ASSERT(
                (isCpuMem(_config.memType) && isCpuMem(dstBuf->type)) ||
                (isGpuMem(_config.memType) && isGpuMem(dstBuf->type)));

        if (isGpuMem(_config.memType)) {
            cpuBuf = _cpuSwapBuf;
        } else {
            cpuBuf = dstBuf;
        }
        memset(cpuBuf->data, 0, _bytesPerFrame);
        uint32_t bytesToRead = pointsToRead * pointStrideBytes;
        void* dataPtr =
            (_config.elementSize == _config.elementStride ? cpuBuf->data : (void*)_tmpStrideBuf.data());
        DS3D_FAILED_RETURN(
                _dataReader.read(dataPtr, bytesToRead) == (int32_t)bytesToRead, ErrCode::kOutOfRange,
                "read file: %s with bytes: %d failed", filepath.c_str(), bytesToRead);

        if (_config.elementSize != _config.elementStride) {
            uint32_t pointBytes = dataTypeBytes(_config.dataType) * _config.elementSize;
            uint8_t* fromPtr = _tmpStrideBuf.data();
            uint8_t* toPtr = (uint8_t*)(cpuBuf->data);
            for (uint32_t iN = 0; iN < pointsToRead;
                    ++iN, fromPtr += pointStrideBytes, toPtr += pointBytes) {
                memcpy(toPtr, fromPtr, pointBytes);
            }
        }

        if (isGpuMem(_config.memType)) {
            DS3D_CHECK_CUDA_ERROR(
                    cudaSetDevice(_config.gpuId), return ErrCode::kCuda,
                    "cudaSetDevice: %d failed when copy file into cuda mem", _config.gpuId);
            DS3D_CHECK_CUDA_ERROR(
                    cudaMemcpy(dstBuf->data, cpuBuf->data, _bytesPerFrame, cudaMemcpyHostToDevice),
                    return ErrCode::kCuda, "copy lidar file into GPU buffer failed");
        }

        void* pointPtr = dstBuf->data;
        FrameGuard lidarFrame;
        uint32_t pointsNum = (_config.fixedPointsNum ? _config.pointNums : pointsToRead);
        if (_config.elementSize == 4) {
            lidarFrame = wrapLidarXYZIFrame<float>(
                    (void*)pointPtr, pointsNum, _config.memType, 0, [dstBuf = std::move(dstBuf)](void*) {});
        } else if (_config.elementSize == 3) {
            lidarFrame = wrapPointXYZFrame<float>(
                    (void*)pointPtr, pointsNum, _config.memType, 0, [dstBuf = std::move(dstBuf)](void*) {});
        }

        if (!lidarFrame) {
            LOG_ERROR("generate Lidar lidarFrame failed.");
            _dataReader.close();
            return ErrCode::kUnknown;
        }
        if (datamap.setGuardData(_config.datamapKey[i], lidarFrame) != ErrCode::kGood) {
            LOG_ERROR("datamap set Lidar lidarFrame failed.");
        }

        _dataReader.close();

        if(!_config.fileLoop) {
            _config.dataParas[i].pop_front();
        }
    }

    datamap.setData(kSourceId, _config.sourceId);
    if(_isFirstFrame) {
        datamap.setData(kFirstSourceFrame, _isFirstFrame);
        _isFirstFrame = false;
    }

    TimeStamp ts{0};
    ts.t0 = timestamp * kMsToNas;
    LOG_DEBUG("set pts %lld", (long long int)ts.t0);
    if(datamap.setData(kTimeStamp, ts) != ErrCode::kGood) {
        LOG_ERROR("set timestamp failed");
        return ErrCode::kUnknown;
    }

    _readFrameCount++;
    outData = std::move(datamap);
    LOG_DEBUG("LidarFileSource dataloader read data successfully.");
    return ErrCode::kGood;
}

ErrCode
LidarFileSourceImpl::stopImpl()
{
    LOG_INFO("LidarFileSource dataloader is closing");
    _bufMem.reset();
    LOG_DEBUG("LidarFileSource dataloader is closed");
    return ErrCode::kGood;
}

}}}  // namespace ds3d::impl::lidarsource

using namespace ds3d;

DS3D_EXTERN_C_BEGIN
DS3D_EXPORT_API abiRefDataLoader*
createLidarFileLoader()
{
    return NewAbiRef<abiDataLoader>(new impl::lidarsource::LidarFileSourceImpl);
}
DS3D_EXTERN_C_END
