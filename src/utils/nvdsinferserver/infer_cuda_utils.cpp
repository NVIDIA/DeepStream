/*
 * SPDX-FileCopyrightText: Copyright (c) 2019 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <dlfcn.h>
#include <unistd.h>
#include <array>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>

#include "infer_cuda_utils.h"
#include "infer_utils.h"

namespace nvdsinferserver {

CudaStream::CudaStream(uint flag, int gpuId, int priority) : m_GpuId(gpuId) {
    CONTINUE_CUDA_ERR(cudaStreamCreateWithPriority(&m_Stream, flag, priority),
        "cudaStreamCreateWithPriority failed");
}

CudaStream::~CudaStream() {
    if (m_Stream != nullptr) {
        CONTINUE_CUDA_ERR(cudaSetDevice(m_GpuId),
            "cudaStreamDestroy failed to set dev-id:%d", m_GpuId);
        CONTINUE_CUDA_ERR(
            cudaStreamDestroy(m_Stream), "cudaStreamDestroy failed");
    }
}

CudaEvent::CudaEvent(uint flag, int gpuId) : m_GpuId(gpuId) {
    CONTINUE_CUDA_ERR(cudaEventCreateWithFlags(&m_Event, flag),
        "cudaEventCreateWithFlags failed");
}

CudaEvent::~CudaEvent() {
    if (m_Event != nullptr) {
        CONTINUE_CUDA_ERR(cudaSetDevice(m_GpuId),
            "CudaEventDestroy failed to set dev-id:%d", m_GpuId);
        CONTINUE_CUDA_ERR(cudaEventDestroy(m_Event), "cudaEventDestroy failed");
    }
}

CudaDeviceMem::CudaDeviceMem(size_t size, int gpuId) : SysMem(size, gpuId) {
    _allocate(size);
}

CudaDeviceMem::~CudaDeviceMem()
{
    CONTINUE_CUDA_ERR(cudaSetDevice(m_DevId), "CudaDeviceMem failed to set dev-id:%d", m_DevId);
    if (m_Buf != nullptr) {
        CONTINUE_CUDA_ERR(cudaFree(m_Buf), "cudaFree failed");
    }
}

void
CudaDeviceMem::_allocate(size_t size)
{
    assert(size);
    m_Buf = nullptr;
    size = INFER_ROUND_UP(size, INFER_MEM_ALIGNMENT);
    CONTINUE_CUDA_ERR(cudaMalloc(&m_Buf, size), "cudaMalloc failed");
    m_Size = size;
    m_Type = InferMemType::kGpuCuda;
}

void
CudaDeviceMem::grow(size_t bytes)
{
    if (bytes <= m_Size) {
        return;
    }
    CONTINUE_CUDA_ERR(
        cudaSetDevice(m_DevId), "CudaDeviceMem grow failed to set dev-id:%d", m_DevId);
    if (m_Buf != nullptr) {
        CONTINUE_CUDA_ERR(cudaFree(m_Buf), "cudaFree failed");
    }
    _allocate(bytes);
}

CudaHostMem::CudaHostMem(size_t size, int gpuId) : SysMem(size, gpuId) {
    _allocate(size);
}

CudaHostMem::~CudaHostMem()
{
    if (m_Buf != nullptr) {
        CONTINUE_CUDA_ERR(cudaFreeHost(m_Buf), "cudaFreeHost failed");
    }
}

void
CudaHostMem::_allocate(size_t size)
{
    assert(size);
    m_Buf = nullptr;
    size = INFER_ROUND_UP(size, INFER_MEM_ALIGNMENT);
    CONTINUE_CUDA_ERR(
        cudaHostAlloc(&m_Buf, size, cudaHostAllocPortable),
        "cudaMallocHost failed");
    m_Size = size;
    m_Type = InferMemType::kCpuCuda;
}

void
CudaHostMem::grow(size_t bytes)
{
    if (bytes <= m_Size) {
        return;
    }
    if (m_Buf != nullptr) {
        CONTINUE_CUDA_ERR(cudaFreeHost(m_Buf), "cudaFreeHost failed");
    }
    _allocate(bytes);
}

CpuMem::CpuMem(size_t size) : SysMem(size, 0), m_Data(size, 0)
{
    m_Buf = (void*)m_Data.data();
    m_Size = size;
    m_Type = InferMemType::kCpu;
}

CpuMem::~CpuMem()
{
    m_Data.clear();
}

void
CpuMem::grow(size_t bytes)
{
    if (bytes <= m_Size) {
        return;
    }
    m_Data.resize(bytes);
    m_Buf = (void*)m_Data.data();
    m_Size = m_Data.size();
    m_Type = InferMemType::kCpu;
}

CudaTensorBuf::CudaTensorBuf(const InferDims& dims, InferDataType dt,
    int batchSize, const std::string& name, InferMemType mt, int devId,
    bool initCuEvent)
    : BaseBatchBuffer(batchSize) {
    assert(InferMemType::kGpuCuda == mt || InferMemType::kCpuCuda == mt);
    assert(batchSize >= 0);
    assert(dt != InferDataType::kString);
    assert(getElementSize(dt) > 0);
    assert(!hasWildcard(dims));
    m_MaxBatchSize = batchSize;

    size_t bufBytes =
        (int)getElementSize(dt) * dimsSize(fullDims(batchSize, dims));
    assert(bufBytes > 0);
    switch (mt) {
    case InferMemType::kCpuCuda:
        m_CudaMem = std::make_unique<CudaHostMem>(bufBytes, devId);
        break;
    case InferMemType::kGpuCuda:
        m_CudaMem = std::make_unique<CudaDeviceMem>(bufBytes, devId);
        break;
    default:
        InferError("failed to create cuda tensor with unsupported memtype:%d",
            static_cast<int>(mt));
        return;
    }
    assert(m_CudaMem && m_CudaMem->ptr());
    InferBufferDescription bufDesc{
        memType : mt,
        devId : devId,
        dataType : dt,
        dims : dims,
        elementSize : getElementSize(dt),
        name : name,
        isInput : false,
    };
    normalizeDims(bufDesc.dims);
    setBufDesc(bufDesc);

    if (initCuEvent) {
        auto event = std::make_shared<CudaEvent>(
            cudaEventBlockingSync | cudaEventDisableTiming, devId);
        assert(event && event->ptr());
        setCuEvent(std::move(event));
    }
}

CudaTensorBuf::~CudaTensorBuf() {
    m_CudaMem.reset();
}

void* CudaTensorBuf::getBufPtr(uint32_t batchIdx) const {
    if (!m_CudaMem) {
        InferError("cuda tensor is empty");
        return nullptr;
    }
    uint32_t batchSize = getBatchSize();
    RETURN_IF_FAILED(
        batchIdx < batchSize || (isNonBatch(batchSize) && batchIdx == 0),
        nullptr,
        "failed to get bufptr since requested batch idx: %d is larger than "
        "batch-size: %d",
        batchIdx, (int)getBatchSize());

    const InferBufferDescription& desc = getBufDesc();
    assert(desc.elementSize * desc.dims.numElements * getBatchSize() <=
           m_CudaMem->bytes());
    assert(batchIdx == 0 || desc.dataType != InferDataType::kString);
    return (void*)(m_CudaMem->ptr<uint8_t>() +
                   desc.elementSize * desc.dims.numElements * batchIdx);
}

UniqCudaTensorBuf createTensorBuf(const InferDims& dims, InferDataType dt,
    int batchSize, const std::string& name, InferMemType mt, int devId,
    bool initCuEvent) {
    if (dt == InferDataType::kString) {
        InferError(
            "create cuda tensor buf fail since kString is not supported.");
        return nullptr;
    }
    UniqCudaTensorBuf buf = std::make_unique<CudaTensorBuf>(
        dims, dt, batchSize, name, mt, devId, initCuEvent);
    if (!buf || !buf->getBufPtr(0)) {
        InferError("create cuda tensor buf failed, dt:%s, dims:%s, name:%s",
            safeStr(dataType2Str(dt)), safeStr(dims2Str(dims)), safeStr(name));
        return nullptr;
    }
    return buf;
}

UniqCudaTensorBuf createGpuTensorBuf(const InferDims& dims,
    InferDataType dt, int batchSize, const std::string& name, int devId,
    bool initCuEvent) {
    return createTensorBuf(
        dims, dt, batchSize, name, InferMemType::kGpuCuda, devId, initCuEvent);
}

UniqCudaTensorBuf createCpuTensorBuf(const InferDims& dims,
    InferDataType dt, int batchSize, const std::string& name, int devId,
    bool initCuEvent) {
    return createTensorBuf(
        dims, dt, batchSize, name, InferMemType::kCpuCuda, devId, initCuEvent);
}

NvDsInferStatus syncAllCudaEvents(const SharedBatchArray &bufList) {
    assert(bufList);
    if (bufList->cuEvent()) {
        RETURN_CUDA_ERR(cudaEventSynchronize(*bufList->cuEvent()),
            "Failed to synchronize cuda events on bufArray");
    }

    for (auto const &buf : bufList->bufs()) {
        if (!buf->cuEvent()) {
            continue;
        }
        RETURN_CUDA_ERR(
            cudaEventSynchronize(*buf->cuEvent()),
            "Failed to synchronize cuda events on each buffer");
    }
    return NVDSINFER_SUCCESS;
}

} // namespace nvdsinferserver
