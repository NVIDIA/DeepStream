/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef __NVDSINFER_BACKEND_H__
#define __NVDSINFER_BACKEND_H__

#include <stdarg.h>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>

#include <cuda_runtime_api.h>

#include <NvInfer.h>
#include <NvInferRuntime.h>

#include "nvdsinfer_func_utils.h"

/* This file provides backend inference interface for abstracting implementation
 * details in various cases like inferencing on implicit batch dims/full dims
 * network, inferencing on DLA etc. This file also provides helper classes for
 * managing the lifecycle of CUDA resources like streams, buffers, events. */

namespace nvdsinfer {

/**
 * Helper class for managing Cuda Streams.
 */
class CudaStream
{
public:
    explicit CudaStream(uint flag = cudaStreamDefault, int priority = 0);
    ~CudaStream();
    operator cudaStream_t() { return m_Stream; }
    cudaStream_t& ptr() { return m_Stream; }
    SIMPLE_MOVE_COPY(CudaStream)

private:
    void move_copy(CudaStream&& o)
    {
        m_Stream = o.m_Stream;
        o.m_Stream = nullptr;
    }
    DISABLE_CLASS_COPY(CudaStream);

    cudaStream_t m_Stream = nullptr;
};

/**
 * Helper class for managing Cuda events.
 */
class CudaEvent
{
public:
    explicit CudaEvent(uint flag = cudaEventDefault);
    ~CudaEvent();
    operator cudaEvent_t() { return m_Event; }
    cudaEvent_t& ptr() { return m_Event; }
    SIMPLE_MOVE_COPY(CudaEvent)

private:
    void move_copy(CudaEvent&& o)
    {
        m_Event = o.m_Event;
        o.m_Event = nullptr;
    }
    DISABLE_CLASS_COPY(CudaEvent);

    cudaEvent_t m_Event = nullptr;
};

/**
 * Helper base class for managing Cuda allocated buffers.
 */
class CudaBuffer
{
public:
    virtual ~CudaBuffer() = default;
    size_t bytes() const { return m_Size; }

    template <typename T>
    T* ptr()
    {
        return (T*)m_Buf;
    }

    void* ptr() { return m_Buf; }
    SIMPLE_MOVE_COPY(CudaBuffer)

protected:
    explicit CudaBuffer(size_t s) : m_Size(s) {}
    void move_copy(CudaBuffer&& o)
    {
        m_Buf = o.m_Buf;
        o.m_Buf = nullptr;
        m_Size = o.m_Size;
        o.m_Size = 0;
    }
    DISABLE_CLASS_COPY(CudaBuffer);
    void* m_Buf = nullptr;
    size_t m_Size = 0;
};

/**
 * CUDA device buffers.
 */
class CudaDeviceBuffer : public CudaBuffer
{
public:
    explicit CudaDeviceBuffer(size_t size);
    ~CudaDeviceBuffer();
};

/**
 * CUDA host buffers.
 */
class CudaHostBuffer : public CudaBuffer
{
public:
    explicit CudaHostBuffer(size_t size);
    ~CudaHostBuffer();
};

/**
 * Abstract interface to manage a batched buffer for inference.
 */
class InferBatchBuffer
{
public:
    InferBatchBuffer() = default;
    virtual ~InferBatchBuffer() = default;

    /* Get device buffer pointers for bound layers associated with this batch. */
    virtual std::vector<void*>& getDeviceBuffers() = 0;
    /* Get the data type of the buffer(layer) for a bound layer having index
     * `bindingIndex`. */
    virtual NvDsInferDataType getDataType(int bindingIndex = 0) const = 0;
    /* Get the batch dimensions for the buffer allocated for a bound layer having
     * index `bindingIndex. */
    virtual NvDsInferBatchDims getBatchDims(int bindingIndex = 0) const = 0;

private:
    DISABLE_CLASS_COPY(InferBatchBuffer);
};

/**
 * Abstract interface for managing the actual inferencing implementation. This
 * interface abstracts away the low-level implementation details required for
 * inferencing with implicit batch dimensions network/full dimensions network on
 * GPU and inferencing on DLA.
 *
 * Actual instance of a BackendContext can be created using `createBackendContext`
 * function. This function will create the appropriate BackendContext
 * (ImplicitTrtBackendContext/FullDimTrtBackendContext/DlaTrtBackendContext)
 * based on the parameters used to build the network/engine.
 */
class BackendContext
{
public:
    BackendContext() = default;
    virtual ~BackendContext() = default;

    /* Initialize the backend context. */
    virtual NvDsInferStatus initialize() = 0;
    /* Get the number of bound layers for the engine. */
    virtual int getNumBoundLayers() = 0;

    /* Get information for a bound layer with index `bindingIdx`. */
    virtual const NvDsInferBatchDimsLayerInfo& getLayerInfo(int bindingIdx) = 0;
    /* Get binding index for a bound layer with name `bindingName`. */
    virtual bool isOutputLayer(const std::string& bindingName) = 0;

    /* Returns if the bound layer at index `bindingIdx` can support the
     * provided batch dimensions. */
    virtual bool canSupportBatchDims(
        int bindingIdx, const NvDsInferBatchDims& batchDims) = 0;

    /* Get the min/max/optimal batch dimensions for a bound layer. */
    virtual NvDsInferBatchDims getMaxBatchDims(int bindingIdx) = 0;
    virtual NvDsInferBatchDims getMinBatchDims(int bindingIdx) = 0;
    virtual NvDsInferBatchDims getOptBatchDims(int bindingIdx) = 0;

    /* Enqueue a batched buffer for inference. */
    virtual NvDsInferStatus enqueueBuffer(
        const std::shared_ptr<InferBatchBuffer>& buffer, CudaStream& stream,
        CudaEvent* consumeEvent) = 0;

private:
    DISABLE_CLASS_COPY(BackendContext);
};

class TrtEngine;

/**
 * Base class for implementations of the BackendContext interface. Implements
 * functionality common to all backends.
 */
class TrtBackendContext : public BackendContext
{
public:
    ~TrtBackendContext();

protected:
    TrtBackendContext(std::unique_ptr<nvinfer1::IExecutionContext>&& ctx,
        std::shared_ptr<TrtEngine> engine);

    bool isOutputLayer(const std::string& bindingName) override;
    int getNumBoundLayers() override;

    const NvDsInferBatchDimsLayerInfo& getLayerInfo(int bindingIdx) override
    {
        assert(bindingIdx < (int)m_AllLayers.size());
        return m_AllLayers[bindingIdx];
    }

    bool canSupportBatchDims(
        int bindingIdx, const NvDsInferBatchDims& batchDims) override;

    virtual NvDsInferBatchDims getMaxBatchDims(int bindingIdx) override
    {
        assert(bindingIdx < (int)m_AllLayers.size());
        return m_AllLayers[bindingIdx].profileDims[kSELECTOR_MAX];
    }
    virtual NvDsInferBatchDims getMinBatchDims(int bindingIdx) override
    {
        assert(bindingIdx < (int)m_AllLayers.size());
        return m_AllLayers[bindingIdx].profileDims[kSELECTOR_MIN];
    }
    virtual NvDsInferBatchDims getOptBatchDims(int bindingIdx) override
    {
        assert(bindingIdx < (int)m_AllLayers.size());
        return m_AllLayers[bindingIdx].profileDims[kSELECTOR_OPT];
    }

protected:
    std::unique_ptr<nvinfer1::IExecutionContext> m_Context;
    std::shared_ptr<TrtEngine> m_CudaEngine;
    std::vector<NvDsInferBatchDimsLayerInfo> m_AllLayers;

    int m_GpuId = -1;

    static std::mutex sDLAExecutionMutex;
};

/**
 * Backend context for full dimensions network.
 */
class FullDimTrtBackendContext : public TrtBackendContext
{
public:
    FullDimTrtBackendContext(
        std::unique_ptr<nvinfer1::IExecutionContext>&& ctx,
        std::shared_ptr<TrtEngine> engine, int profile = 0);

private:
    NvDsInferStatus initialize() override;

    NvDsInferStatus enqueueBuffer(
        const std::shared_ptr<InferBatchBuffer>& buffer, CudaStream& stream,
        CudaEvent* consumeEvent) override;

protected:
    // Only idx 0 profile supported.
    const int m_ProfileIndex = 0;
};

/**
 * Backend context for implicit batch dimension network inferencing on DLA.
 */
class DlaFullDimTrtBackendContext : public FullDimTrtBackendContext
{
public:
    DlaFullDimTrtBackendContext(std::unique_ptr<nvinfer1::IExecutionContext>&& ctx,
        std::shared_ptr<TrtEngine> engine, int profile = 0);

    NvDsInferStatus enqueueBuffer(
        const std::shared_ptr<InferBatchBuffer>& buffer, CudaStream& stream,
        CudaEvent* consumeEvent) override;

private:
    static std::mutex sExecutionMutex;
};

/**
 * Create an instance of a BackendContext.
 *
 * ImplicitTrtBackendContext - created when TRT CudaEngine/network is built with
 *                             implicit batch dimensions
 * FullDimTrtBackendContext - created when TRT CudaEngine/network is built with
 *                            full dimensions support
 * DlaTrtBackendContext - created when TRT CudaEngine is built for DLA
 */
std::unique_ptr<TrtBackendContext> createBackendContext(
    const std::shared_ptr<TrtEngine>& engine);

} // end of namespace nvdsinfer

#endif
