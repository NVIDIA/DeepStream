/*
 * SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "infer_extra_processor.h"

#include <chrono>

#include "infer_cuda_utils.h"
#include "infer_stream_manager.h"

namespace nvdsinferserver {

namespace {
std::vector<uint64_t>
getStreamIds(const SharedIOptions& options)
{
    if (!options || !options->hasValue(OPTION_NVDS_SREAM_IDS)) {
        return {};
    }
    std::vector<uint64_t> streamIds;
    RETURN_IF_FAILED(
        options->getValueArray(OPTION_NVDS_SREAM_IDS, streamIds) == NVDSINFER_SUCCESS, {},
        "failed to get %s from shared options", OPTION_NVDS_SREAM_IDS);
    return streamIds;
}

template <typename UniqBuffer>
std::unique_ptr<MapBufferPool<std::string, UniqBuffer>>
CreateMapBufferPool(
    const std::vector<InferBufferDescription>& descList, uint32_t maxBatchSize, uint32_t poolSize,
    std::function<UniqBuffer(const InferBufferDescription&, uint32_t)> createBufFunc,
    const std::string& mapName = "mapPool")
{
    auto mapPool = std::make_unique<MapBufferPool<std::string, UniqBuffer>>(mapName);
    assert(descList.size() > 0);
    std::set<std::string> exists;
    for (const auto& eachDesc : descList) {
        RETURN_IF_FAILED(
            !exists.count(eachDesc.name), nullptr, "tensor: %s already exist in map pool: %s",
            safeStr(eachDesc.name), safeStr(mapName));
        for (uint32_t i = 0; i < poolSize; ++i) {
            auto buf = createBufFunc(eachDesc, maxBatchSize);
            RETURN_IF_FAILED(
                buf, nullptr, "create tensor: %s failed for map pool: %s", safeStr(eachDesc.name),
                safeStr(mapName));
            mapPool->setBuffer(eachDesc.name, std::move(buf));
        }
        exists.insert(eachDesc.name);
    }
    assert(mapPool);
    return mapPool;
}
}  // namespace

InferExtraProcessor::InferExtraProcessor() {}

InferExtraProcessor::~InferExtraProcessor()
{
    m_CustomProcessor.reset();
}

NvDsInferStatus
InferExtraProcessor::initCustomProcessor(
    SharedDllHandle dlHandle, const std::string& funcName, const std::string& config)
{
    assert(!funcName.empty());
    if (!dlHandle) {
        dlHandle = std::make_shared<DlLibHandle>("");
    }
    RETURN_IF_FAILED(
        dlHandle && dlHandle->isValid(), NVDSINFER_CUSTOM_LIB_FAILED, "dlopen lib failed.");
    auto funcPtr = dlHandle->symbol<CreateCustomProcessorFunc>(funcName);
    RETURN_IF_FAILED(
        funcPtr, NVDSINFER_CUSTOM_LIB_FAILED, "dlsym %s failed. dlerror: %s", safeStr(funcName),
        dlerror());
    m_CustomProcessor.reset(
        funcPtr(config.c_str(), (uint32_t)config.length()),
        [dlHandle](IInferCustomProcessor* p) { delete p; });
    m_RequireInferLoop = m_CustomProcessor->requireInferLoop();
    if (m_RequireInferLoop) {
        m_StreamManager = std::make_unique<StreamManager>();
    }
    return NVDSINFER_SUCCESS;
}

NvDsInferStatus
InferExtraProcessor::allocateExtraInputs(
    BaseBackend& backend, const std::set<std::string>& excludes, int32_t poolSize, int gpuId)
{
    uint32_t maxBatch = (uint32_t)backend.maxBatchSize();  // 0 means non batching
    m_maxBatch = maxBatch;
    m_firstDimDynamicBatch = backend.isFirstDimBatch();
    const LayerDescription* layers = nullptr;
    int layerSize = 0;
    std::tie(layers, layerSize) = backend.getInputLayers();
    assert(layerSize > 0);
    std::vector<InferBufferDescription> descList;
    for (int i = 0; i < layerSize; ++i) {
        m_FullInputLayers.push_back(layers[i]);
        if (excludes.count(layers[i].name)) {
            continue;
        }
        m_ExtraInputLayers.push_back(layers[i]);
        InferBufferDescription desc{
            memType : InferMemType::kCpu,
            devId : 0,
            dataType : layers[i].dataType,
            dims : layers[i].inferDims,
            elementSize : getElementSize(layers[i].dataType),
            name : layers[i].name,
            isInput : true,
        };
        descList.emplace_back(desc);
    }
    if (descList.empty()) {
        return NVDSINFER_SUCCESS;
    }

    auto hostPool = CreateMapBufferPool<UniqCudaTensorBuf>(
        descList, maxBatch, poolSize,
        [](const InferBufferDescription& desc, uint32_t batch) {
            auto buf = createCpuTensorBuf(desc.dims, desc.dataType, batch, desc.name, 0);
            if (buf) {
                buf->mutableBufDesc().isInput = true;
            }
            return buf;
        },
        "extra_input_host_tensors");
    RETURN_IF_FAILED(
        hostPool, NVDSINFER_RESOURCE_ERROR, "create extra_input_host_tensors pool failed.");

    auto gpuPool = CreateMapBufferPool<UniqCudaTensorBuf>(
        descList, maxBatch, poolSize,
        [gpuId](const InferBufferDescription& desc, uint32_t batch) {
            auto buf = createGpuTensorBuf(desc.dims, desc.dataType, batch, desc.name, gpuId);
            if (buf) {
                buf->mutableBufDesc().isInput = true;
            }
            return buf;
        },
        "extra_input_gpu_tensors");
    RETURN_IF_FAILED(
        gpuPool, NVDSINFER_RESOURCE_ERROR, "create extra_input_gpu_tensors pool failed.");

    m_ExtInputHostPool = std::move(hostPool);
    m_ExtInputGpuPool = std::move(gpuPool);
    m_Host2GpuStream = std::make_shared<CudaStream>(cudaStreamDefault, gpuId);
    RETURN_IF_FAILED(
        m_Host2GpuStream && m_Host2GpuStream->ptr(), NVDSINFER_CUDA_ERROR,
        "create host2gpu cuda stream failed.");
    return NVDSINFER_SUCCESS;
}

NvDsInferStatus
InferExtraProcessor::processExtraInputs(SharedBatchArray& inputs)
{
    assert(m_CustomProcessor);
    InferMemType inputType = InferMemType::kCpu;
    m_CustomProcessor->supportInputMemType(inputType);
    if (inputType != InferMemType::kCpu && inputType != InferMemType::kGpuCuda) {
        InferError(
            "Customprocessor returned Input memtype:%s is not supported.",
            safeStr(memType2Str(inputType)));
        return NVDSINFER_INVALID_PARAMS;
    }

    assert(inputs && inputs->getSize() > 0);
    SharedBatchBuf primaryBuf = inputs->buf(0);
    assert(primaryBuf);
    uint32_t primaryBatch = primaryBuf->getBatchSize();

    std::vector<SharedBatchBuf> extraInputs;
    if (!m_ExtraInputLayers.empty() && inputs->getSize() < m_FullInputLayers.size()) {
        TensorMapPool* pool = m_ExtInputGpuPool.get();
        if (isCpuMem(inputType)) {
            pool = m_ExtInputHostPool.get();
        }
        for (auto const& layer : m_ExtraInputLayers) {
            SharedBatchBuf buf = pool->acquireBuffer(layer.name);
            RETURN_IF_FAILED(
                buf, NVDSINFER_RESOURCE_ERROR, "extrac input tensor: %s acquire buffer failed.",
                safeStr(layer.name));
            assert(buf);
            InferDims bufDims = buf->getBufDesc().dims;
            uint32_t batch = buf->getBatchSize();
            if (!isNonBatch(m_maxBatch)) {
                batch = primaryBatch;
            } else if (m_firstDimDynamicBatch) {  // for non batching only
                // bufDims.d[0] should be equal to equal to max batch size.
                // buf is generated from allocated input pool which was set as maximized shape
                // update the value to primary batch size. e.g.
                // original input is [-1, 16, 16], max shape of the layer is [maxBatch, 16, 16],
                // update the buf desc to [primaryBatch, 16, 16]
                RETURN_IF_FAILED(
                    bufDims.numDims > 0 && bufDims.d[0] >= (int32_t)primaryBatch,
                    NVDSINFER_UNKNOWN_ERROR, "update extra input dims: %s failed.",
                    safeStr(dims2Str(bufDims)));
                bufDims.d[0] = primaryBatch;
            }
            bufDims = fullDims(batch, bufDims);
            // use full dims for all custom processing
            buf = ReshapeBuf(buf, 0, bufDims, true);
            assert(buf);
            extraInputs.push_back(buf);
        }
    }

    auto fullDimPrimary = reshapeToFullDimsBuf(primaryBuf, false);
    assert(fullDimPrimary);
    auto options = inputs->getSafeOptions();
    std::vector<IBatchBuffer*> iBuffers;
    for (auto& each : extraInputs) {
        assert(each);
        iBuffers.push_back(each.get());
    }

    // get timestamp
    uint64_t timestamp = UINT64_MAX;
    if (options && options->hasValue(OPTION_TIMESTAMP)) {
        RETURN_NVINFER_ERROR(
            options->getUInt(OPTION_TIMESTAMP, timestamp), "get %s from Input options failed",
            OPTION_TIMESTAMP);
    }
    if (timestamp == UINT64_MAX) {
        auto now = std::chrono::time_point_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now());
        timestamp = now.time_since_epoch().count();
        timestamp *= 1000000;  // ms to ns
    }
    // get all stream ids
    std::vector<uint64_t> streamIds;
    if (requireLoop()) {
        streamIds = getStreamIds(options);
        assert(m_StreamManager);
        for (int64_t eachStream : streamIds) {
            RETURN_NVINFER_ERROR(
                m_StreamManager->waitStream(eachStream), "failed to wait stream:%" PRId64,
                eachStream);
        }
    }

    RETURN_NVINFER_ERROR(
        m_CustomProcessor->extraInputProcess({fullDimPrimary.get()}, iBuffers, options.get()),
        "custom extra input processing failed.");

    if (streamIds.size() && requireLoop()) {
        assert(m_StreamManager);
        for (uint64_t eachStream : streamIds) {
            RETURN_NVINFER_ERROR(
                m_StreamManager->startStream(eachStream, timestamp, nullptr),
                "failed to wait stream:%" PRId64, eachStream);
        }
    }

    if (isCpuMem(inputType) && !extraInputs.empty()) {
        std::vector<SharedBatchBuf> gpuBufs;
        for (auto& eachCpuBuf : extraInputs) {
            uint32_t batch = eachCpuBuf->getBatchSize();
            InferDims dims = eachCpuBuf->getBufDesc().dims;
            std::string name = eachCpuBuf->getBufDesc().name;
            SharedBatchBuf gpuBuf = m_ExtInputGpuPool->acquireBuffer(name);
            // recalculate bytes
            gpuBuf = ReshapeBuf(gpuBuf, batch, dims, true);
            assert(m_Host2GpuStream);
            RETURN_NVINFER_ERROR(
                tensorBufferCopy(eachCpuBuf, gpuBuf, m_Host2GpuStream),
                "copy extra processed input tensor: %s to gpu failed.", safeStr(name));
            gpuBufs.emplace_back(std::move(gpuBuf));
        }
        assert(m_Host2GpuStream);
        RETURN_CUDA_ERR(
            cudaStreamSynchronize(*m_Host2GpuStream), "Failed to synch lstm cuda stream");
        extraInputs.swap(gpuBufs);
    }

    for (auto& eachBuf : extraInputs) {
        inputs->addBuf(eachBuf);
    }
    assert(inputs->getSize() >= m_ExtraInputLayers.size());
    return NVDSINFER_SUCCESS;
}

NvDsInferStatus
InferExtraProcessor::destroy()
{
    if (m_ExtInputGpuPool) {
        m_ExtInputGpuPool->clear();
        m_ExtInputGpuPool.reset();
    }
    if (m_ExtInputHostPool) {
        m_ExtInputHostPool->clear();
        m_ExtInputHostPool.reset();
    }
    m_CustomProcessor.reset();
    m_Host2GpuStream.reset();
    m_StreamManager.reset();
    return NVDSINFER_SUCCESS;
}

NvDsInferStatus
InferExtraProcessor::checkInferOutputs(SharedBatchArray& outputs, SharedOptions inOptions)
{
    if (m_CustomProcessor) {
        const auto& bufs = outputs->bufs();
        auto fullDimArray = std::make_shared<BaseBatchArray>();
        assert(fullDimArray);
        fullDimArray->setOptions(outputs->getSafeOptions());
        for (const auto& each : bufs) {
            SharedBatchBuf reshaped = each;
            if (!isPrivateTensor(each->getBufDesc().name)) {
                reshaped = reshapeToFullDimsBuf(each, false);
            }
            fullDimArray->addBuf(reshaped);
        }
        RETURN_NVINFER_ERROR(
            m_CustomProcessor->inferenceDone(fullDimArray.get(), inOptions.get()),
            "Failed in custom processor in inferenceDone callbacks.");
    }

    if (requireLoop()) {
        auto streamIds = getStreamIds(inOptions);
        assert(m_StreamManager);
        for (const auto& stream : streamIds) {
            RETURN_NVINFER_ERROR(
                m_StreamManager->streamInferDone(stream, outputs),
                "failed to notity streaminfer done, stream_id: %" PRId64, stream);
        }
    }
    return NVDSINFER_SUCCESS;
}

void
InferExtraProcessor::notifyError(NvDsInferStatus status)
{
    if (m_CustomProcessor) {
        m_CustomProcessor->notifyError(status);
    }
    if (m_StreamManager) {
        m_StreamManager->notifyError(status);
    }
}

}  // namespace nvdsinferserver