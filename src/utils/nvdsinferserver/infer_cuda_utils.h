/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

/**
 * @file infer_cuda_utils.h
 *
 * @brief Header file declaring utility classes for CUDA memory management,
 * CIDA streams and events.
 */

#ifndef __NVDSINFER_CUDA_UTILS_H__
#define __NVDSINFER_CUDA_UTILS_H__

#include <stdarg.h>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include "infer_batch_buffer.h"
#include "infer_common.h"

namespace nvdsinferserver {

/**
 * @brief Wrapper class for CUDA streams.
 */
class CudaStream {
public:
    explicit CudaStream(
        uint flag = cudaStreamDefault, int gpuId = 0, int priority = 0);
    ~CudaStream();
    operator cudaStream_t() { return m_Stream; }
    int devId() const { return m_GpuId; }
    cudaStream_t& ptr() { return m_Stream; }
    SIMPLE_MOVE_COPY(CudaStream)

private:
    void move_copy(CudaStream&& o) {
        m_Stream = o.m_Stream;
        o.m_Stream = nullptr;
    }
    DISABLE_CLASS_COPY(CudaStream);

    cudaStream_t m_Stream = nullptr;
    int m_GpuId = 0;
};

/**
 * @brief Wrapper class for CUDA events.
 */
class CudaEvent {
public:
    explicit CudaEvent(uint flag = cudaEventDefault, int gpuId = 0);
    virtual ~CudaEvent();
    operator cudaEvent_t() { return m_Event; }
    int devId() const { return m_GpuId; }
    cudaEvent_t& ptr() { return m_Event; }
    SIMPLE_MOVE_COPY(CudaEvent)

private:
    void move_copy(CudaEvent&& o) {
        m_Event = o.m_Event;
        o.m_Event = nullptr;
    }

    DISABLE_CLASS_COPY(CudaEvent);

    cudaEvent_t m_Event = nullptr;
    int m_GpuId = 0;
};

/**
 * @brief Base class for managing memory allocation.
 */
class SysMem {
public:
    virtual ~SysMem() = default;
    size_t bytes() const { return m_Size; }

    template <typename T>
    T* ptr() const {
        return (T*)m_Buf;
    }

    void* ptr() const { return m_Buf; }
    int devId() const { return m_DevId; }
    InferMemType type() const { return m_Type; }
    void reuse() {}
    virtual void grow(size_t bytes) = 0;

    SIMPLE_MOVE_COPY(SysMem)

protected:
    SysMem(size_t s, int devId) : m_Size(s), m_DevId(devId) {}
    void move_copy(SysMem&& o) {
        m_Buf = o.m_Buf;
        o.m_Buf = nullptr;
        m_Size = o.m_Size;
        o.m_Size = 0;
        m_DevId = o.m_DevId;
        o.m_DevId = 0;
        m_Type = o.m_Type;
        o.m_Type = InferMemType::kNone;
    }

    DISABLE_CLASS_COPY(SysMem);

    void* m_Buf = nullptr;
    size_t m_Size = 0;
    int m_DevId = 0;
    InferMemType m_Type = InferMemType::kNone;
};

/**
 * @brief Allocates and manages CUDA device memory.
 */
class CudaDeviceMem : public SysMem {
public:
    CudaDeviceMem(size_t size, int gpuId = 0);
    ~CudaDeviceMem() override;
    void grow(size_t bytes) override;

private:
    void _allocate(size_t bytes);
};

/**
 * @brief Allocates and manages CUDA pinned memory.
 */
class CudaHostMem : public SysMem {
public:
    CudaHostMem(size_t size, int gpuId = 0);
    ~CudaHostMem() override;
    void grow(size_t bytes) override;

private:
    void _allocate(size_t bytes);
};

/**
 * @brief Allocates and manages host memory.
 */
class CpuMem : public SysMem {
public:
    CpuMem(size_t size);
    ~CpuMem() override;
    void grow(size_t bytes) override;

private:
    std::vector<uint8_t> m_Data;
};

/**
 * @brief A batch buffer with CUDA memory allocation.
 */
class CudaTensorBuf : public BaseBatchBuffer {
public:
    CudaTensorBuf(const InferDims& dims, InferDataType dt, int batchSize,
        const std::string& name, InferMemType mt, int devId, bool initCuEvent);

    ~CudaTensorBuf() override;

    void setBatchSize(uint32_t size) override {
        assert(size <= m_MaxBatchSize);
        BaseBatchBuffer::setBatchSize(size);
    }

    void setName(const std::string& name) { mutableBufDesc().name = name; }
    void* getBufPtr(uint32_t batchIdx) const final;
    void reuse()
    {
        setBatchSize(m_MaxBatchSize);
        detach();
    }

private:
    UniqSysMem m_CudaMem;
    uint32_t m_MaxBatchSize = 0;
};

/**
 * @brief Create a tensor buffer of the specified memory type, dimensions
 * on the given device.
 *
 * @param[in] dims         Dimensions of the tensor.
 * @param[in] dt           Datatype.
 * @param[in] batchSize    Batch size.
 * @param[in] name         Name of the buffer.
 * @param[in] mt           Memory type.
 * @param[in] devId        Device ID for the memory allocation.
 * @param[in] initCuEvent  Flag to create an associated CUDA event.
 * @return Pointer to the newly created tensor buffer, null if the buffer
 *         could not be created.
 */
UniqCudaTensorBuf createTensorBuf(const InferDims& dims, InferDataType dt,
    int batchSize, const std::string& name, InferMemType mt, int devId,
    bool initCuEvent);

/**
 * @brief Create a CUDA device memory tensor buffer of specified dimensions
 * on the given device.
 *
 * @param[in] dims         Dimensions of the tensor.
 * @param[in] dt           Datatype.
 * @param[in] batchSize    Batch size.
 * @param[in] name         Name of the buffer.
 * @param[in] devId        Device ID for the memory allocation.
 * @param[in] initCuEvent  Flag to create an associated CUDA event.
 * @return Pointer to the newly created tensor buffer, null if the buffer
 *         could not be created.
 */
UniqCudaTensorBuf createGpuTensorBuf(const InferDims& dims,
    InferDataType dt, int batchSize, const std::string& name = "",
    int devId = 0, bool initCuEvent = false);

/**
 * @brief Create a CUDA pinned memory tensor buffer of specified dimensions
 * on the given device.
 *
 * @param[in] dims         Dimensions of the tensor.
 * @param[in] dt           Datatype.
 * @param[in] batchSize    Batch size.
 * @param[in] name         Name of the buffer.
 * @param[in] devId        Device ID for the memory allocation.
 * @param[in] initCuEvent  Flag to create an associated CUDA event.
 * @return Pointer to the newly created tensor buffer, null if the buffer
 *         could not be created.
 */
UniqCudaTensorBuf createCpuTensorBuf(const InferDims& dims,
    InferDataType dt, int batchSize, const std::string& name = "",
    int devId = 0, bool initCuEvent = false);

/**
 * @brief Synchronize on all events associated with the batch buffer array.
 * @param[in] bufList Array of the batch buffers.
 * @return NVDSINFER_SUCCESS or NVDSINFER_CUDA_ERROR.
 */
NvDsInferStatus syncAllCudaEvents(const SharedBatchArray &bufList);

} // namespace nvdsinferserver

#endif
