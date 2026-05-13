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

#include <dlfcn.h>
#include <unistd.h>
#include <array>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>

#include <NvInfer.h>
#include <NvInferRuntime.h>

#include "nvdsinfer_backend.h"
#include "nvdsinfer_custom_impl.h"
#include "nvdsinfer_model_builder.h"

#define DEFAULT_CONTEXT_PROFILE_IDX 0

/* This file contains implementation of the various backends and wrappers for
 * managing CUDA resources. */

namespace nvdsinfer {

static const int INPUT_LAYER_INDEX = 0;

CudaStream::CudaStream(uint flag, int priority)
{
    CHECK_CUDA_ERR_NO_ACTION(
        cudaStreamCreateWithPriority(&m_Stream, flag, priority),
        "cudaStreamCreateWithPriority failed");
}

CudaStream::~CudaStream()
{
    if (m_Stream != nullptr)
    {
        CHECK_CUDA_ERR_NO_ACTION(
            cudaStreamDestroy(m_Stream), "cudaStreamDestroy failed");
    }
}

CudaEvent::CudaEvent(uint flag)
{
    CHECK_CUDA_ERR_NO_ACTION(cudaEventCreateWithFlags(&m_Event, flag),
        "cudaEventCreateWithFlags failed");
}

CudaEvent::~CudaEvent()
{
    if (m_Event != nullptr)
    {
        CHECK_CUDA_ERR_NO_ACTION(
            cudaEventDestroy(m_Event), "cudaEventDestroy failed");
    }
}

CudaDeviceBuffer::CudaDeviceBuffer(size_t size) : CudaBuffer(size)
{
    CHECK_CUDA_ERR_NO_ACTION(cudaMalloc(&m_Buf, size), "cudaMalloc failed");
    m_Size = size;
}

CudaDeviceBuffer::~CudaDeviceBuffer()
{
    if (m_Buf != nullptr)
    {
        CHECK_CUDA_ERR_NO_ACTION(cudaFree(m_Buf), "cudaFree failed");
    }
}

CudaHostBuffer::CudaHostBuffer(size_t size) : CudaBuffer(size)
{
    CHECK_CUDA_ERR_NO_ACTION(
        cudaMallocHost(&m_Buf, size), "cudaMallocHost failed");
    m_Size = size;
}

CudaHostBuffer::~CudaHostBuffer()
{
    if (m_Buf != nullptr)
    {
        CHECK_CUDA_ERR_NO_ACTION(cudaFreeHost(m_Buf), "cudaFreeHost failed");
    }
}

TrtBackendContext::TrtBackendContext(
    std::unique_ptr<nvinfer1::IExecutionContext>&& ctx,
    std::shared_ptr<TrtEngine> engine)
    : m_Context(std::move(ctx)), m_CudaEngine(engine)
{
    assert(m_CudaEngine);
    assert(m_Context);
}

TrtBackendContext::~TrtBackendContext()
{
    m_Context.reset();
    m_CudaEngine.reset();
}

bool
TrtBackendContext::isOutputLayer(const std::string& bindingName)
{
    assert(m_CudaEngine);
    assert(!bindingName.empty());
    return (*m_CudaEngine)->getTensorIOMode(bindingName.c_str()) == nvinfer1::TensorIOMode::kOUTPUT;
}

int
TrtBackendContext::getNumBoundLayers()
{
    assert(m_CudaEngine);
    assert(!m_AllLayers.empty());
    assert((int)m_AllLayers.size() == (*m_CudaEngine)->getNbIOTensors());
    return m_AllLayers.size();
}

bool
TrtBackendContext::canSupportBatchDims(int bindingIdx,
        const NvDsInferBatchDims& batchDims)
{
    assert((int)m_AllLayers.size() > bindingIdx);
    assert((int)m_AllLayers[bindingIdx].isInput == 1);
    /* Number of dimensions should match. */
    assert(m_AllLayers[bindingIdx].inferDims.numDims == batchDims.dims.numDims);

    const NvDsInferBatchDims& minBatchDims =
        m_AllLayers[bindingIdx].profileDims[kSELECTOR_MIN];
    const NvDsInferBatchDims& maxBatchDims =
        m_AllLayers[bindingIdx].profileDims[kSELECTOR_MAX];

    /* Number of dimensions should match. */
    assert(minBatchDims.dims.numDims == maxBatchDims.dims.numDims);
    if (batchDims.dims.numDims != minBatchDims.dims.numDims ||
        batchDims.dims.numDims != maxBatchDims.dims.numDims)
    {
        dsInferWarning("Backend context num dims doesn't match input dims");
        return false;
    }

    /* Check if provided dims are within range of [min,max] dims */
    if (minBatchDims.dims > batchDims.dims ||
        minBatchDims.batchSize > batchDims.batchSize ||
        batchDims.dims > maxBatchDims.dims ||
        batchDims.batchSize > maxBatchDims.batchSize)
    {
        dsInferWarning(
            "Backend context bufferIdx(%d) request dims:%s is out of range"
            ", [min: %s, max: %s]",
            bindingIdx, safeStr(batchDims2Str(batchDims)),
            safeStr(batchDims2Str(minBatchDims)),
            safeStr(batchDims2Str(maxBatchDims)));
        return false;
    }

    return true;
}

std::unique_ptr<TrtBackendContext>
createBackendContext(const std::shared_ptr<TrtEngine>& engine)
{
    if (!engine)
    {
        dsInferError("create backend context failed since TrtEngine is empty");
        return nullptr;
    }

    std::unique_ptr<nvinfer1::IExecutionContext> cudaCtx(
        (*engine)->createExecutionContext());

    if (!cudaCtx)
    {
        dsInferError("create TRT cuda executionContext failed");
        return nullptr;
    }

    std::unique_ptr<TrtBackendContext> backend;

    /* Engine built with fulldims support */
    assert((*engine)->getNbOptimizationProfiles() > 0);

    if (engine->hasDla())
    {
        backend = std::make_unique<DlaFullDimTrtBackendContext>(
            std::move(cudaCtx), engine, DEFAULT_CONTEXT_PROFILE_IDX);
    }
    else
    {
        backend = std::make_unique<FullDimTrtBackendContext>(
            std::move(cudaCtx), engine, DEFAULT_CONTEXT_PROFILE_IDX);
    }

    if (!backend)
    {
        dsInferError("create TRT backend context failed");
        return nullptr;
    }

    CHECK_NVINFER_ERROR(backend->initialize(), return nullptr,
        "Failed to initialize TRT backend");

    return backend;
}

std::mutex TrtBackendContext::sDLAExecutionMutex;

FullDimTrtBackendContext::FullDimTrtBackendContext(
    std::unique_ptr<nvinfer1::IExecutionContext>&& ctx,
    std::shared_ptr<TrtEngine> engine, int profile)
    : TrtBackendContext(std::move(ctx), engine), m_ProfileIndex(profile) {}

NvDsInferStatus
FullDimTrtBackendContext::initialize()
{
    assert(m_Context);
    if (m_Context->getOptimizationProfile() != m_ProfileIndex)
    {
        if (!m_Context->setOptimizationProfileAsync(m_ProfileIndex, 0))
        {
            dsInferError("Failed to setOptimizationProfile with idx:%d ",
                m_ProfileIndex);
            return NVDSINFER_INVALID_PARAMS;
        }
    }

    RETURN_NVINFER_ERROR(
        m_CudaEngine->getFullDimsLayersInfo(m_ProfileIndex, m_AllLayers),
        "Failed to get fullDimLayersInfo of profile idx:%d ", m_ProfileIndex);

    /* Set optimal dims as binding dims for all input layers. */
    for (int i = 0; i < (int)(*m_CudaEngine)->getNbIOTensors(); i++)
    {
        auto binding_name = (*m_CudaEngine)->getIOTensorName(i);
        if ((*m_CudaEngine)->getTensorIOMode(binding_name) == nvinfer1::TensorIOMode::kOUTPUT)
            continue;

        nvinfer1::Dims optDims = (*m_CudaEngine)
                                     ->getProfileShape(binding_name, m_ProfileIndex,
                                         nvinfer1::OptProfileSelector::kOPT);
        if (!m_Context->setInputShape(binding_name, optDims))
        {
            dsInferError(
                "Failed to initialize fulldim backend when seting "
                "bindings idx:%s with opt dims:%s",
                binding_name, safeStr(dims2Str(optDims)));
            return NVDSINFER_INVALID_PARAMS;
        }
    }

    if (!m_Context->allInputDimensionsSpecified())
    {
        dsInferError(
            "Failed to initialize fulldims context, check any input dims is "
            "not specified");
        return NVDSINFER_TENSORRT_ERROR;
    }

    for (int i = 0; i < (int)m_AllLayers.size(); ++i)
    {
        NvDsInferBatchDimsLayerInfo& layerInfo = m_AllLayers.at(i);

        nvinfer1::Dims fullInferDims = m_Context->getTensorShape(layerInfo.layerName);
        assert(!hasWildcard(fullInferDims));

        NvDsInferDims inferDims = {0};
        int batchSize = 0;

        /* Split the full dims recieved from getBindingDimensions into batchSize
         * and rest of the dims. */
        SplitFullDims(fullInferDims, inferDims, batchSize);
        layerInfo.inferDims = inferDims;
    }

    return NVDSINFER_SUCCESS;
}

NvDsInferStatus
FullDimTrtBackendContext::enqueueBuffer(
    const std::shared_ptr<InferBatchBuffer>& buffer, CudaStream& stream,
    CudaEvent* consumeEvent)
{
    assert(m_Context);
    assert(stream.ptr());

    std::vector<void*> bindingBuffers = buffer->getDeviceBuffers();

    for (int iL = 0; iL < (int)m_AllLayers.size(); ++iL)
    {
        m_Context->setTensorAddress(m_AllLayers[iL].layerName, bindingBuffers.data()[iL]);
        if (!m_AllLayers[iL].isInput)
            continue;

        NvDsInferBatchDims batchDims = buffer->getBatchDims(iL);
        assert(batchDims.batchSize == buffer->getBatchDims(0).batchSize);

        if(batchDims.batchSize < m_AllLayers[iL].profileDims[kSELECTOR_MIN].batchSize)
            batchDims.batchSize = m_AllLayers[iL].profileDims[kSELECTOR_MIN].batchSize;

        if (!canSupportBatchDims(iL, batchDims))
        {
            dsInferError(
                "Failed to enqueue buffer in fulldims mode because "
                "binding idx: %d with batchDims: %s is not supported ",
                iL, safeStr(batchDims2Str(batchDims)));
            return NVDSINFER_INVALID_PARAMS;
        }

        nvinfer1::Dims dimsWBatch =
            CombineDimsBatch(batchDims.dims, batchDims.batchSize);
        nvinfer1::Dims lastDimsBatch = m_Context->getTensorShape(m_AllLayers[iL].layerName);
        if (dimsWBatch != lastDimsBatch)
        {
            if (!m_Context->setInputShape(m_AllLayers[iL].layerName, dimsWBatch))
            {
                dsInferError(
                    "Failed to enqueue buffer when setting bindings idx:%d with "
                    "dims:%s",
                    iL, safeStr(dims2Str(dimsWBatch)));
                return NVDSINFER_INVALID_PARAMS;
            }
        }
    }

    if (!m_Context->allInputDimensionsSpecified())
    {
        dsInferError(
            "Failed to enqueue buffer because context dims are not specified "
            "in dynamic mode");
        return NVDSINFER_TENSORRT_ERROR;
    }

    if (!m_Context->enqueueV3(stream))
    {
        dsInferError("Failed to enqueue trt inference batch");
        return NVDSINFER_TENSORRT_ERROR;
    }
    cudaEventRecord(consumeEvent->ptr(), stream);

    return NVDSINFER_SUCCESS;
}

DlaFullDimTrtBackendContext::DlaFullDimTrtBackendContext(
    std::unique_ptr<nvinfer1::IExecutionContext>&& ctx,
    std::shared_ptr<TrtEngine> engine,
    int profile)
    : FullDimTrtBackendContext(std::move(ctx), engine, profile) {}

NvDsInferStatus
DlaFullDimTrtBackendContext::enqueueBuffer(
    const std::shared_ptr<InferBatchBuffer>& buffer, CudaStream& stream,
    CudaEvent* consumeEvent)
{
    assert(m_Context);
    assert(stream.ptr());

    std::vector<void*> bindingBuffers = buffer->getDeviceBuffers();

    for (int iL = 0; iL < (int)m_AllLayers.size(); ++iL)
    {
        m_Context->setTensorAddress(m_AllLayers[iL].layerName, bindingBuffers.data()[iL]);
        if (!m_AllLayers[iL].isInput)
            continue;

        NvDsInferBatchDims batchDims = buffer->getBatchDims(iL);
        NvDsInferBatchDims maxDims = getMaxBatchDims(iL);
        if (batchDims.batchSize > maxDims.batchSize)
        {
            dsInferError(
                "Failed to enqueue buffer in fulldims mode because "
                "binding idx: %d with batchDims: %s is not supported ",
                iL, safeStr(batchDims2Str(batchDims)));
            return NVDSINFER_INVALID_PARAMS;
        }
        batchDims.batchSize = maxDims.batchSize;
        if (!canSupportBatchDims(iL, batchDims))
        {
            dsInferError(
                "Failed to enqueue buffer in fulldims mode because "
                "binding idx: %d with batchDims: %s is not supported ",
                iL, safeStr(batchDims2Str(batchDims)));
            return NVDSINFER_INVALID_PARAMS;
        }
    }

    if (!m_Context->allInputDimensionsSpecified())
    {
        dsInferError(
            "Failed to enqueue buffer because context dims are not specified "
            "in dynamic mode");
        return NVDSINFER_TENSORRT_ERROR;
    }

    /* Parallel enqueue for multiple DLA engines fails. Serialize enqueue
     * calls for DLA engines with a static mutex. */
    std::unique_lock<std::mutex> locker(sDLAExecutionMutex);

    if (!m_Context->enqueueV3(stream))
    {
        dsInferError("Failed to enqueue trt inference batch");
        return NVDSINFER_TENSORRT_ERROR;
    }
    cudaEventRecord(consumeEvent->ptr(), stream);

    return NVDSINFER_SUCCESS;
}

} // namespace nvdsinfer
