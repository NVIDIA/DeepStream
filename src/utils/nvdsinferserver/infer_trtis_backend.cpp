/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @file infer_trtis_backend.cpp
 *
 * @brief Source file of Triton Inference Server inference backend.
 *
 * This file defines the inference backend for the Triton Inference Server.
 */

#include "infer_trtis_backend.h"

#include "infer_batch_buffer.h"
#include "infer_cuda_utils.h"
#include "infer_datatypes.h"
#include "infer_post_datatypes.h"
#include "infer_trtis_server.h"
#include "infer_utils.h"


#define INFER_DEFAULT_MIN_DIM 1
#define INFER_DEFAULT_MAX_DIM 4096 * 4096
#define INFER_MAX_BATCH_SIZE 4096

namespace nvdsinferserver {

TrtISBackend::TrtISBackend(
    const std::string& name, int64_t version, TrtServerPtr ptr)
    : m_Model(name), m_ModelVersion(version), m_Server(std::move(ptr))
{
}

TrtISBackend::~TrtISBackend() {
    if (m_NeedUnload) {
        CONTINUE_NVINFER_ERROR(m_Server->unloadModel(m_Model),
            "failed to unload model: %s but continue", safeStr(m_Model));
    }
}

NvDsInferStatus TrtISBackend::ensureModelReady() {
    InferDebug("TrtISBackend id:%d ensure model: %s", uniqueId(), safeStr(model()));

    if (!m_Server->isModelReady(model(), m_ModelVersion)) {
        RETURN_NVINFER_ERROR(m_Server->loadModel(model()),
            "failed to load model: %s", safeStr(model()));
        m_NeedUnload = true;
    }
    return NVDSINFER_SUCCESS;
}

NvDsInferStatus
TrtISBackend::setupLayersInfo()
{
    ni::ModelConfig modelConfig;
    RETURN_NVINFER_ERROR(
        m_Server->getModelConfig(model(), m_ModelVersion, modelConfig),
        "failed to get model: %s config", safeStr(model()));

    InferDebug(
        "Triton backend ensure model:%s is ready, model-config\n%s", safeStr(model()),
        safeStr(modelConfig.ShortDebugString()));

    if (!modelConfig.input_size() || !modelConfig.output_size()) {
        InferError(
            "gie id: %d model: %s input or output layers are empty, "
            "check triton model config settings",
            uniqueId(), safeStr(model()));
        return NVDSINFER_TRITON_ERROR;
    }

    setMaxBatchSize((uint32_t)modelConfig.max_batch_size());

    LayerDescriptionList layers(
        modelConfig.input_size() + modelConfig.output_size());

    int bindingIdx = 0;
    InferTensorOrder tensorOrder = InferTensorOrder::kNone;
    for (const auto& in : modelConfig.input()) {
        LayerDescription& desc = layers[bindingIdx];
        desc.bindingIndex = bindingIdx;
        desc.isInput = true;
        desc.name = in.name();
        desc.dataType = DataTypeFromTritonPb(in.data_type());
        desc.inferDims = DimsFromTriton(in.dims());
        InferTensorOrder order = TensorOrderFromTritonPb(in.format());
        if (desc.inferDims.numDims <= 0) {
            InferError(
                "Triton failed to ensure model:%s since input:%s dims not "
                "configured",
                safeStr(m_Model), safeStr(desc.name));
            return NVDSINFER_CONFIG_FAILED;
        }
        if (tensorOrder == InferTensorOrder::kNone) {
            tensorOrder = order;
        } else if (order != InferTensorOrder::kNone && order != tensorOrder) {
            InferError(
                "Triton failed to ensure model:%s since input:%s format "
                "disordered",
                safeStr(m_Model), safeStr(desc.name));
            return NVDSINFER_CONFIG_FAILED;
        }
        ++bindingIdx;
    }

    setInputTensorOrder(tensorOrder);

    for (const auto& out : modelConfig.output()) {
        LayerDescription& desc = layers[bindingIdx];
        desc.bindingIndex = bindingIdx;
        desc.isInput = false;
        desc.name = out.name();
        desc.dataType = DataTypeFromTritonPb(out.data_type());
        desc.inferDims = DimsFromTriton(out.dims());
        ++bindingIdx;
    }

    resetLayers(std::move(layers), modelConfig.input_size());

    if (m_ClassifyParams.size() == 1 &&
        m_ClassifyParams[0].tensorName.empty()) {
        m_ClassifyParams[0].tensorName = modelConfig.output(0).name();
    }

    for (const auto& classP : m_ClassifyParams) {
        RETURN_IF_FAILED(
            !classP.tensorName.empty(), NVDSINFER_CONFIG_FAILED,
            "Triton model:%s configured empty tensor_name in "
            "triton_classification",
            safeStr(m_Model));
        const LayerDescription* classLayer = getLayerInfo(classP.tensorName);
        RETURN_IF_FAILED(
            classLayer && !classLayer->isInput, NVDSINFER_CONFIG_FAILED,
            "Triton can NOT find classification.tensor_name: %s"
            "in model:%s output tensors, please update config file",
            safeStr(classP.tensorName), safeStr(m_Model));
    }

    return NVDSINFER_SUCCESS;
}

NvDsInferStatus
TrtISBackend::ensureServerReady()
{
    if (!m_Server) {
        m_Server = TrtISServer::getInstance(nullptr);
        RETURN_IF_FAILED(
            m_Server, NVDSINFER_CONFIG_FAILED,
            "failed to get triton server instance while initializing model:%s",
            safeStr(model()));
    }

    RETURN_IF_FAILED(
        m_Server->isServerLive(), NVDSINFER_TRITON_ERROR,
        "failed to check server live state for model:%s", safeStr(model()));

#if 0
    RETURN_IF_FAILED(
        m_Server->isServerReady(), NVDSINFER_TRITON_ERROR,
        "failed to check triton server ready", safeStr(model()));
#endif
    return NVDSINFER_SUCCESS;
}

NvDsInferStatus
TrtISBackend::setupReorderThread()
{
    m_ReorderThread = std::make_unique<ReorderThread>(
        [this](ReorderItemPtr item) {
            return this->inferenceDoneReorderLoop(item);
        },
        "TrtISBeReorder");
    assert(m_ReorderThread);
    return NVDSINFER_SUCCESS;
}

NvDsInferStatus
TrtISBackend::initialize()
{
    RETURN_NVINFER_ERROR(
        ensureServerReady(),
        "failed to initialize backend while ensuring server ready");

    RETURN_NVINFER_ERROR(
        ensureModelReady(),
        "failed to initialize backend while ensuring model:%s ready",
        safeStr(model()));

    RETURN_NVINFER_ERROR(
        setupLayersInfo(),
        "failed to initialize backend while setup model:%s layer info",
        safeStr(model()));

    RETURN_NVINFER_ERROR(
        setupReorderThread(),
        "failed to initialize backend(%s) while setup reorder thread",
        safeStr(model()));
    assert(m_ReorderThread);

    setAllocator(std::make_unique<TrtServerAllocator>(
        std::bind(
            &TrtISBackend::allocateResponseBuf, this, std::placeholders::_1,
            std::placeholders::_2, std::placeholders::_3,
            std::placeholders::_4),
        std::bind(
            &TrtISBackend::releaseResponseBuf, this, std::placeholders::_1,
            std::placeholders::_2)));

    InferInfo("TrtISBackend id:%d initialized model: %s", uniqueId(), safeStr(model()));

    return NVDSINFER_SUCCESS;
}

SharedSysMem
TrtISBackend::allocateResponseBuf(
    const std::string& tensor, size_t bytes, InferMemType memType, int64_t devId)
{
    // Default CPU memType has higher priority, otherwise overwrite it to user settings.
    if (!isCpuMem(memType) && outputMemType() != InferMemType::kNone) {
        assert(memType == InferMemType::kGpuCuda || memType == InferMemType::kCpuCuda);
        memType = outputMemType();
    }

    if (outputDevId() >= 0) {
        devId = outputDevId();
    }
    if (isCpuMem(memType)) {
        devId = 0;
        memType = InferMemType::kCpuCuda;
    }
    PoolKey key = std::make_tuple(tensor, devId, memType);

    PoolValue pool = findResponsePool(key);
    if (!pool) {
        pool = createResponsePool(key, bytes);
    }
    if (!pool) {
        InferError(
            "Triton backend failed to acquire buffer since there is no buffer "
            "pool created");
        return nullptr;
    }
    SharedSysMem buf = pool->acquireBuffer();
    if (!buf) {
        InferError("Triton backend failed to acquire buffer, internal error");
        return nullptr;
    }
    if (buf->bytes() < bytes) {
        buf->grow(bytes);
    }
    return buf;
}

void
TrtISBackend::releaseResponseBuf(const std::string& tensor, SharedSysMem mem)
{
    assert(!tensor.empty());
    assert(getLayerInfo(tensor));
    assert(mem);
    mem.reset();
}

TrtISBackend::PoolValue TrtISBackend::findResponsePool(PoolKey& key) {
    std::shared_lock<SharedMutex> readLock(m_ResponseMutex);
    auto iter = m_ResponsePool.find(key);
    if (iter != m_ResponsePool.end()) {
        assert(iter->second);
        return iter->second;
    }
    return nullptr;
}

TrtISBackend::PoolValue TrtISBackend::createResponsePool(
    PoolKey& key, size_t bytes) {
    std::unique_lock<SharedMutex> writeLock(m_ResponseMutex);
    PoolValue& pool = m_ResponsePool[key];
    if (pool)
        return pool;

    // prepare memory size > 0
    if (!bytes) {
        bytes = 1;
    }
    std::string tensorName = std::get<kName>(key);
    pool = std::make_shared<BufferPool<UniqSysMem>>("IsTensorPool-" + tensorName);
    assert(m_ResponsePool[key]);
    size_t maxBytes = m_TensorMaxBytes[tensorName];
    if (maxBytes < bytes) {
        setTensorMaxBytes(tensorName, bytes);
        maxBytes = m_TensorMaxBytes[tensorName];
    }
    bytes = std::max<size_t>(maxBytes, bytes);

    int gpuId = std::get<kGpuId>(key);
    InferMemType memType = std::get<kMemType>(key);
    if (memType == InferMemType::kGpuCuda) {
        CONTINUE_CUDA_ERR(
            cudaSetDevice(gpuId),
            "failed to set gpu device id:%d during creating Triton buffer pool",
            gpuId);
    }

    for (int i = 0; i < m_PerPoolSize; ++i) {
        UniqSysMem newMem;
        if (memType == InferMemType::kGpuCuda) {
            newMem = std::make_unique<CudaDeviceMem>(bytes, gpuId);
        } else {
            newMem = std::make_unique<CudaHostMem>(bytes, gpuId);
        }
        pool->setBuffer(std::move(newMem));
    }
    assert(pool->size() == m_PerPoolSize);

    return pool;
}

NvDsInferStatus TrtISBackend::specifyInputDims(const InputShapes& shapes) {
    InferDebug("TrtISBackend id:%d specify input-dims for model: %s",
        uniqueId(), safeStr(model()));

    RETURN_IF_FAILED(
        shapes.size() == getInputLayerSize(), NVDSINFER_CONFIG_FAILED,
        "failed to specify input_dims for model:%s since inputs num :%d"
        "doesn't match input layers size:%d",
        safeStr(model()), (int)shapes.size(), getInputLayerSize());

    RETURN_IF_FAILED(
        checkInputDims(shapes), NVDSINFER_CONFIG_FAILED,
        "failed to specify input_dims for model:%s while checking input dims",
        safeStr(model()));

    int gpuId = 0;
    CONTINUE_CUDA_ERR(
        cudaGetDevice(&gpuId), "CudaDeviceMem failed to get dev-id:%d", gpuId);

    SharedBatchArray allInputs = std::make_shared<BaseBatchArray>();
    for (const auto& in : shapes) {
        const std::string& name = std::get<kInShapeName>(in);
        const InferBatchDims& dims = std::get<kInShapeDims>(in);
        const LayerDescription* layer = getLayerInfo(name);
        assert(layer);
        UniqCudaTensorBuf tensor = createGpuTensorBuf(
            dims.dims, layer->dataType, dims.batchSize, name, gpuId, false);
        RETURN_IF_FAILED(
            tensor, NVDSINFER_CUDA_ERROR, "failed to create GPU tensor buffer.");

        tensor->mutableBufDesc().isInput = true;
        allInputs->mutableBufs().emplace_back(std::move(tensor));
    }

    using RunResult = std::tuple<NvDsInferStatus, SharedBatchArray>;
    std::promise<RunResult> p;
    std::future<RunResult> f = p.get_future();
    RETURN_NVINFER_ERROR(
        Run(allInputs, nullptr,
            [&p](NvDsInferStatus s, SharedBatchArray out) {
                p.set_value(std::make_tuple(s, std::move(out)));
            }),
        "failed to specify dims when running inference on model:%s",
        safeStr(model()));

    RunResult result = f.get();
    RETURN_NVINFER_ERROR(std::get<0>(result),
        "failed to specify dims after running inference failed on model:%s",
        safeStr(model()));

    SharedBatchArray& allOutputs = std::get<1>(result);
    assert(allOutputs);
    auto& outBufList = allOutputs->mutableBufs();
    outBufList.erase(
        std::remove_if(
            outBufList.begin(), outBufList.end(),
            [](const SharedBatchBuf& buf) {
                assert(buf);
                return isPrivateTensor(buf->getBufDesc().name);
            }),
        outBufList.end());
    RETURN_NVINFER_ERROR(
        fixateDims(allOutputs),
        "failed to specify dims during fixate output dims for model:%s",
        safeStr(model()));
    RETURN_NVINFER_ERROR(
        fixateDims(allInputs),
        "failed to specify dims during fixate output dims for model:%s",
        safeStr(model()));

    return NVDSINFER_SUCCESS;
}

NvDsInferStatus
TrtISBackend::fixateDims(const SharedBatchArray& bufs)
{
    assert(bufs);
    for (const auto& buf : bufs->bufs()) {
        assert(buf);
        const InferBufferDescription& bufDesc = buf->getBufDesc();

        LayerDescription* layer = mutableLayerInfo(bufDesc.name);
        if (!layer) {
            InferError("failed to fixate tensor: %s dims for model:%s",
                safeStr(bufDesc.name), safeStr(model()));
            return NVDSINFER_INVALID_PARAMS;
        }
        layer->inferDims =
            isNonBatching() ? fullDims((int)buf->getBatchSize(), bufDesc.dims)
                            : bufDesc.dims;
    }
    return NVDSINFER_SUCCESS;
}

bool
TrtISBackend::debatchingOutput(
    SharedBatchArray& outputs, SharedBatchArray& inputs)
{
    assert(inputs && inputs->getSize() > 0);
    assert(outputs);
    assert(isNonBatching());

    uint32_t inBatch = inputs->buf(0)->getBatchSize();
    if (isNonBatch(inBatch)) {
        return true;
    }

    auto& bufList = outputs->mutableBufs();
    for (SharedBatchBuf& buf : bufList) {
        InferBufferDescription desc = buf->getBufDesc();
        if (isPrivateTensor(desc.name) || !isNonBatch(buf->getBatchSize())) {
            continue;
        }
        // Only de-batch non-batched buffers according to input batch size
        InferDims debatchDims;
        uint32_t outBatch = 0;
        RETURN_IF_FAILED(
            debatchFullDims(desc.dims, debatchDims, outBatch), false,
            "De-batching full dims on tensor:%s failed. dims:%s",
            safeStr(desc.name), safeStr(dims2Str(desc.dims)));

        if (outBatch != inBatch) {
            continue;
        }
        desc.dims = debatchDims;

        size_t bufOffset = buf->getBufOffset(0);
        if (bufOffset == (size_t) -1) {
            InferError(
                    "Debatching output failed, invalid buffer offset.");
            return false;
        }
        SharedRefBatchBuf refBuf(
            new RefBatchBuffer(
                (void*)buf->getBufPtr(0), bufOffset,
                buf->getTotalBytes(), desc, inBatch),
            [priv = buf](RefBatchBuffer* buf) mutable {
                delete buf;
                priv.reset();
            });
        buf = refBuf;
    }
    return true;
}

bool
TrtISBackend::inferenceDoneReorderLoop(ReorderItemPtr item)
{
    if (!item) {
        InferError(
            "TrtISBackend reorder received empty item on model :%s",
            safeStr(model()));
        return false;
    }
    item->future.wait();
    if (needKeepInputs() && item->status == NVDSINFER_SUCCESS) {
        assert(item->outputs);
        const auto& inBufs = item->inputs->bufs();
        auto& outBufs = item->outputs->mutableBufs();
        outBufs.insert(outBufs.begin(), inBufs.begin(), inBufs.end());
    };

    NvDsInferStatus status = item->status;
    if (isNonBatching() && NVDSINFER_SUCCESS == status) {
        debatchingOutput(item->outputs, item->inputs);
    }

    item->inputs.reset();  // keep input release in order
    assert(item->inferenceDone);
    item->inferenceDone(status, std::move(item->outputs));
    return true;
}

NvDsInferStatus TrtISBackend::enqueue(SharedBatchArray inputs,
    SharedCuStream stream, InputsConsumed bufConsumed,
    InferenceDone inferenceDone) {
    INFER_UNUSED(stream);
    assert(inputs && inputs->getSize() > 0);

    int batchSize = inputs->buf(0)->getBatchSize();
    RETURN_NVINFER_ERROR(
        syncAllCudaEvents(inputs),
        "Triton Failed to synchronize model:%s inputs events buffer inference", safeStr(model()));

    InferDebug("TrtISBackend id:%d enqueue batch:%d for model: %s to inference",
        uniqueId(), batchSize, safeStr(model()));

    RETURN_NVINFER_ERROR(
        ensureInputs(inputs), "Failed to ensure input tensors for model: %s",
        safeStr(model()));

    ReorderItemPtr item = std::make_shared<ReorderItem>();
    item->inputs = inputs;
    item->future = item->promise.get_future();
    item->inferenceDone = std::move(inferenceDone);
    RETURN_NVINFER_ERROR(
        Run(std::move(inputs), std::move(bufConsumed),
            [this, item](NvDsInferStatus status, SharedBatchArray outs) {
                assert(item);
                item->status = status;
                item->outputs = outs;
                item->promise.set_value();
            }),
        "Triton failed to run inference on model %s", safeStr(model()));

    m_ReorderThread->queueItem(std::move(item));

    return NVDSINFER_SUCCESS;
}

NvDsInferStatus
TrtISBackend::ensureInputs(SharedBatchArray& inputs)
{
    assert(inputs);
    for (auto& buf : inputs->mutableBufs()) {
        const InferBufferDescription& bufDesc = buf->getBufDesc();
        const LayerDescription* layerDesc = getLayerInfo(bufDesc.name);

        RETURN_IF_FAILED(
            layerDesc && layerDesc->isInput, NVDSINFER_INVALID_PARAMS,
            "Input buffer: %s is not found in model input_tensors.",
            safeStr(bufDesc.name));

        InferDims bufDim = fullDims(buf->getBatchSize(), bufDesc.dims);
        InferDims modelDim = fullDims(maxBatchSize(), layerDesc->inferDims);
        if (bufDim.numDims == modelDim.numDims) {
            // Do not check shape value, backend might be specified as max
            continue;
        }

        // Squeeze reshape only works for cases like this:
        // e.g. model requires [3, 255, 255] but buf is [1, 3, 255, 255]
        if (squeezeMatch(bufDim, modelDim)) {
            buf = ReshapeBuf(buf, maxBatchSize(), layerDesc->inferDims);
        }
    }
    return NVDSINFER_SUCCESS;
}

void
TrtISBackend::requestTritonOutputNames(std::set<std::string>& outNames)
{
    const LayerDescription* outLayers = nullptr;
    int outLayerSize = 0;
    std::tie(outLayers, outLayerSize) = getOutputLayers();
    assert(outLayers && outLayerSize);
    for (int i = 0; i < outLayerSize; ++i) {
        outNames.emplace(outLayers[i].name);
    }
}

NvDsInferStatus
TrtISBackend::Run(
    SharedBatchArray inputs, InputsConsumed bufConsumed, AsyncDone asyncDone)
{
    assert(m_Server);
    std::set<std::string> outNames;
    requestTritonOutputNames(outNames);  // derived interface

    SharedRequest request = m_Server->createRequest(
        model(), version(), inputs, {outNames.begin(), outNames.end()},
        m_ClassifyParams);
    if (!request) {
        InferError("Triton run failed to create request for model: %s",
            safeStr(model()));
        return NVDSINFER_TRITON_ERROR;
    }

    RETURN_NVINFER_ERROR(
        m_Server->inferAsync(
            std::move(request), m_ResponseAllocator,
            [this, bufConsumed, asyncDone](
                SharedRequest req, UniqResponse res) {
                this->serverInferCompleted(
                    std::move(req), std::move(res), std::move(bufConsumed),
                    asyncDone);
            }),
        "TRT-IS async inference failed.");

    return NVDSINFER_SUCCESS;
}

void
TrtISBackend::serverInferCompleted(
    SharedRequest request, UniqResponse uniqResponse,
    InputsConsumed inputsConsumed, AsyncDone asyncDone)
{
    InferDebug("TrtISBackend id:%d inference batch done on model: %s",
        uniqueId(), safeStr(model()));

    SharedBatchArray inputs = request->releaseInputs();
    if (inputsConsumed) {
        inputsConsumed(std::move(inputs));
    } else {
        inputs.reset();
    }

    if (!uniqResponse) {
        asyncDone(NVDSINFER_TRITON_ERROR, nullptr);
        return;
    }

    if (uniqResponse->parse(request.get()) != NVDSINFER_SUCCESS) {
        InferError(
            "Triton server failed to parse response with "
            "request-id:%" PRIu64 " model:%s",
            request->id(), safeStr(uniqResponse->model()));
        asyncDone(NVDSINFER_TRITON_ERROR, nullptr);
        return;
    }

    UniqResponse response(std::move(uniqResponse));

    const LayerDescription* outLayers = nullptr;
    int outlayerNum = 0;
    std::tie(outLayers, outlayerNum) = getOutputLayers();

    NvDsInferStatus status = response->getStatus();
    assert(NVDSINFER_SUCCESS == status);
    assert(request->model() == response->model());

    SharedBatchArray outputs = std::make_shared<BaseBatchArray>();
    outputs->setBufId(request->bufId());
    SharedOptions options = response->takeoverOptions();
    if (options) {
        outputs->setOptions(options);
    }

    for (SharedBatchBuf& t : response->mutableOutputs()) {
        InferBufferDescription& desc = t->mutableBufDesc();
        if (isPrivateTensor(desc.name)) {
            outputs->mutableBufs().emplace_back(t);
            continue;
        }
        const LayerDescription* layer = getLayerInfo(desc.name);
        if (!layer) {
            InferWarning(
                "Triton: backend failed to understand output tensor name:%s"
                " from model:%s, skip this tensor",
                safeStr(desc.name), safeStr(response->model()));
            continue;
        }
        desc.dataType = layer->dataType;
        desc.elementSize = getElementSize(layer->dataType);
        if (hasWildcard(desc.dims)) {
            desc.dims = layer->inferDims;
        }
        assert(!hasWildcard(desc.dims));
        assert(desc.dataType != InferDataType::kNone);
        assert(desc.elementSize >= 0);

        // must combine RefBatchBuffer reference with response for safe
        outputs->mutableBufs().emplace_back(t);
    }
    request.reset();
    response.reset();
    assert(asyncDone);
    asyncDone(status, outputs->getSize() ? std::move(outputs) : nullptr);
}

} // namespace nvdsinferserver
