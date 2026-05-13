/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */


#ifndef DS3D_COMMON_HELPER_CUDA_UTILS_H
#define DS3D_COMMON_HELPER_CUDA_UTILS_H

#include "ds3d/common/common.h"
#include "ds3d/common/helper/memdata.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

namespace ds3d {

class CudaStream {
    cudaStream_t _stream = nullptr;
    int _gpuId = 0;
    DS3D_DISABLE_CLASS_COPY(CudaStream);

public:
    explicit CudaStream(uint flag = cudaStreamDefault, int gpuId = 0, int priority = 0)
    {
        cudaSetDevice(_gpuId);
        DS3D_CHECK_CUDA_ERROR(
            cudaStreamCreateWithPriority(&_stream, flag, priority), , "cudaStreamCreate failed");
    }
    ~CudaStream()
    {
        if (_stream != nullptr) {
            DS3D_CHECK_CUDA_ERROR(
                cudaSetDevice(_gpuId), , "cudaStreamDestroy failed to set gpu-id:%d", _gpuId);
            DS3D_CHECK_CUDA_ERROR(cudaStreamDestroy(_stream), , "cudaStreamDestroy failed");
        }
    }
    ErrCode sync()
    {
        if (!_stream) {
            return ErrCode::kGood;
        }
        DS3D_CHECK_CUDA_ERROR(
            cudaSetDevice(_gpuId), return ErrCode::kCuda,
            "cudaStreamSynchronize failed to set gpu-id:%d", _gpuId);
        DS3D_CHECK_CUDA_ERROR(
            cudaStreamSynchronize(_stream), return ErrCode::kCuda, "cudaStreamSynchronize failed");
        return ErrCode::kGood;
    }
    int gpuId() const { return _gpuId; }
    cudaStream_t& get() { return _stream; }
};

class GpuCudaMemBuf : public MemData {
public:
    GpuCudaMemBuf() {}
    static std::unique_ptr<GpuCudaMemBuf> CreateBuf(size_t size, int gpuId)
    {
        auto mem = std::make_unique<GpuCudaMemBuf>();

        DS3D_CHECK_CUDA_ERROR(
            cudaSetDevice(gpuId), return nullptr,
            "cudaSetDevice: %d failed when allocate cuda memory", gpuId);
        size = DS3D_ROUND_UP(size, 256);
        void* data = nullptr;
        DS3D_CHECK_CUDA_ERROR(
            cudaMalloc(&data, size), return nullptr, "cudaMalloc size: %d failed", (int)size);
        DS_ASSERT(data);

        mem->data = data;
        mem->byteSize = size;
        mem->devId = 0;
        mem->type = MemType::kGpuCuda;
        return mem;
    }
    ~GpuCudaMemBuf()
    {
        if (data) {
            cudaSetDevice(devId);
            cudaFree(data);
            data = nullptr;
            byteSize = 0;
        }
    }
};

};  // namespace ds3d
#endif
