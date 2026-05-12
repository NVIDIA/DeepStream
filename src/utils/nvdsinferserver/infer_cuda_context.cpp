/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @file infer_cuda_context.cpp
 *
 * @brief Source file for the CUDA inference context class.
 *
 * This file defines the CUDA processing class for handling the inference
 * context for the nvinferserver low level library.
 */

#include "infer_cuda_context.h"

#include <dlfcn.h>
#include <unistd.h>

#include <array>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>

#include "infer_base_backend.h"
#include "infer_cuda_utils.h"
#include "infer_extra_processor.h"
#include "infer_lstm.h"
#include "infer_postprocess.h"
#include "infer_preprocess.h"
#include "infer_proto_utils.h"
#include "infer_stream_manager.h"
#include "infer_utils.h"
#include "nvdsinfer_custom_impl.h"

namespace nvdsinferserver {

namespace {
const static int kDefaultSurfacePoolSize = 2;
const static uint32_t kDefaultNetworkPreprocessPoolSize = 3;

const static int kDefaultGpuId = 0;
const static int kMinHostTensorPoolSize = 2;
const static int kMaxHostTensorPoolSize = 64;

} // namespace

class CudaEventInPool : public CudaEvent {
public:
    CudaEventInPool(uint flag, int gpuId) : CudaEvent(flag, gpuId){}
    void reuse() {
        assert(ptr());
    }
};

using UniqEventPtrInPool = std::unique_ptr<CudaEventInPool>;

InferCudaContext::InferCudaContext()
    : m_HostTensorPool("HostPostMapPool")
{}

InferCudaContext::~InferCudaContext()
{
    m_ExtraInputs.clear();
    m_LstmController.reset();
    m_HostTensorPool.clear();
}

int InferCudaContext::tensorPoolSize() const {
    if (!config().has_extra())
        return kMinHostTensorPoolSize;

    int poolSize = std::max<int>(kMinHostTensorPoolSize,
        config().extra().output_buffer_pool_size());

    return poolSize;
}

NvDsInferStatus InferCudaContext::deinit() {
    m_HostTensorPool.clear();
    m_HostTensorEvents.reset();
    m_ExtraInputs.clear();
    if (m_LstmController) {
        m_LstmController->destroy();
        m_LstmController.reset();
    }
    if (m_ExtraProcessor) {
        m_ExtraProcessor->destroy();
        m_ExtraProcessor.reset();
    }
    return InferBaseContext::deinit();
}

NvDsInferStatus InferCudaContext::fixateInferenceInfo(
    const ic::InferenceConfig& config, BaseBackend& backend) {
    static const int kDefaultInputIndex = 0;

    m_InputTensorOrder = backend.getInputTensorOrder();
    const LayerDescription* inputLayers = nullptr;
    int inputSize = 0;
    std::tie(inputLayers, inputSize) = backend.getInputLayers();
    RETURN_IF_FAILED(
        inputSize > 0, NVDSINFER_CONFIG_FAILED, "No input layers are found in bakcend model");

    const LayerDescription* inputInfo = &inputLayers[kDefaultInputIndex];
    if (config.has_preprocess()) {
        const ic::PreProcessParams& preprocess = config.preprocess();
        m_NetworkImageFormat =
            mediaFormatFromDsProto(preprocess.network_format());
        InferTensorOrder configOrder =
            tensorOrderFromDsProto(preprocess.tensor_order());
        if (configOrder != InferTensorOrder::kNone) {
            if (m_InputTensorOrder == InferTensorOrder::kNone) {
                m_InputTensorOrder = configOrder;
            } else if (m_InputTensorOrder != configOrder) {
                printError(
                    "preprocess.tensor_order:%s is not backend model expected "
                    "format:%s",
                    safeStr(tensorOrder2Str(configOrder)),
                    safeStr(tensorOrder2Str(m_InputTensorOrder)));
                return NVDSINFER_CONFIG_FAILED;
            }
        }
        if (m_InputTensorOrder == InferTensorOrder::kNone) {
            printError(
                "InferContext(uid:%d) cannot figure out input tensor order, "
                "please specify in config file(preprocess.)",
                uniqueId());
            return NVDSINFER_CONFIG_FAILED;
        }

        if (!preprocess.tensor_name().empty()) {
            inputInfo = backend.getLayerInfo(preprocess.tensor_name());
        }
        RETURN_IF_FAILED(
            inputInfo, NVDSINFER_CONFIG_FAILED,
            "preprocess.tensor_name is not found in model's input tensors");

        m_NetworkImageName = inputInfo->name;
        m_InputDataType = inputInfo->dataType;
        if (hasWildcard(inputInfo->inferDims)) {
            printError(
                "failed to fixate inference info since input has wildcard "
                "dims.");
            return NVDSINFER_CONFIG_FAILED;
        }
        m_NetworkImageInfo =
            dims2ImageInfo(inputInfo->inferDims, m_InputTensorOrder);
        assert(m_InputTensorOrder != InferTensorOrder::kNone);
    } else if (!hasWildcard(inputInfo->inferDims)) {
        m_NetworkImageInfo =
            dims2ImageInfo(inputInfo->inferDims, m_InputTensorOrder);
    }

    if (config.has_lstm()) {
        m_LstmController = std::make_unique<LstmController>(
            config.lstm(), config.gpu_ids(0), config.max_batch_size());
        assert(m_LstmController);
        CTX_RETURN_NVINFER_ERROR(
            m_LstmController->initInputState(backend),
            "LSTM controller failed to init input state");
    }
    if (config.has_extra() && !config.extra().custom_process_funcion().empty()) {
        CTX_RETURN_NVINFER_ERROR(
            loadExtraProcessor(config, backend, inputInfo->name),
            "Load extra processing functions failed.");
    }
    return NVDSINFER_SUCCESS;
}

NvDsInferStatus
InferCudaContext::loadExtraProcessor(
    const ic::InferenceConfig& config, BaseBackend& backend, const std::string& primaryTensor)
{
    assert(config.has_extra() && !config.extra().custom_process_funcion().empty());
    assert(!m_ExtraProcessor);
    const std::string customProcessor{config.extra().custom_process_funcion()};
    SharedDllHandle dlhanle = customLib();
    auto extraProcessor = std::make_unique<InferExtraProcessor>();
    CTX_RETURN_NVINFER_ERROR(
        extraProcessor->initCustomProcessor(dlhanle, customProcessor, config.DebugString()),
        "loading custom process function: %s failed.", safeStr(customProcessor));

    assert(extraProcessor);
    CTX_RETURN_NVINFER_ERROR(
        extraProcessor->allocateExtraInputs(
            backend, {primaryTensor}, kDefaultNetworkPreprocessPoolSize, config.gpu_ids(0)),
        "extra processor allocating inputs failed.");

    m_ExtraProcessor = std::move(extraProcessor);
    return NVDSINFER_SUCCESS;
}

NvDsInferStatus InferCudaContext::createPreprocessor(
    const ic::PreProcessParams& params,
    std::vector<UniqPreprocessor>& processors) {
    printDebug("InferCudaContxt create preprocessors");
    processors.clear();
    // set nv surface crop converter
    auto surfaceConverter =
        std::make_unique<ThreadPreprocessor<CropSurfaceConverter>>(
            kDefaultSurfacePoolSize);
    assert(surfaceConverter);
    surfaceConverter->setThreadName("SurfConv-" + std::to_string(uniqueId()));
    surfaceConverter->setUniqueId(uniqueId());
    surfaceConverter->setParams(m_NetworkImageInfo.width,
        m_NetworkImageInfo.height, m_NetworkImageFormat, maxBatchSize());
    surfaceConverter->setMaintainAspectRatio(params.maintain_aspect_ratio());
    surfaceConverter->setSymmetricPadding(params.symmetric_padding());
    surfaceConverter->setScalingHW(computeHWFromDsProto(params.frame_scaling_hw()));
    surfaceConverter->setScalingFilter(
        scalingFilterFromDsProto(params.frame_scaling_filter()));
    m_SurfaceConverter = surfaceConverter.get();
    processors.emplace_back(std::move(surfaceConverter));

    // set network preprocessor
    auto networkPreprocessor =
        std::make_unique<ThreadPreprocessor<NetworkPreprocessor>>(
            m_NetworkImageInfo, m_NetworkImageFormat, m_InputDataType,
            maxBatchSize());
    networkPreprocessor->setThreadName("NetPreproc" + std::to_string(uniqueId()));
    assert(networkPreprocessor);
    networkPreprocessor->setUniqueId(uniqueId());
    networkPreprocessor->setPoolSize(kDefaultNetworkPreprocessPoolSize);
    const auto& normalizeParams = params.normalize();
    assert(!fEqual(normalizeParams.scale_factor(), 0));
    std::vector<float> offsets(normalizeParams.channel_offsets().begin(),
        normalizeParams.channel_offsets().end());
    if (!networkPreprocessor->setScaleOffsets(
            normalizeParams.scale_factor(), offsets)) {
        InferError("failed to set preprocess scalor and offsets.");
        return NVDSINFER_CONFIG_FAILED;
    }
    if (!normalizeParams.mean_file().empty()) {
        if (!networkPreprocessor->setMeanFile(normalizeParams.mean_file())) {
            InferError("failed to set preprocess mean file: %s.",
                safeStr(normalizeParams.mean_file()));
            return NVDSINFER_CONFIG_FAILED;
        }
    }
    networkPreprocessor->setNetworkTensorOrder(m_InputTensorOrder);
    networkPreprocessor->setNetworkTensorName(m_NetworkImageName);

    m_NetworkPreprocessor = networkPreprocessor.get();
    processors.emplace_back(std::move(networkPreprocessor));
    return NVDSINFER_SUCCESS;
}

NvDsInferStatus InferCudaContext::createPostprocessor(
    const ic::PostProcessParams& params, UniqPostprocessor& processor) {
    printDebug("InferCudaContxt create postprocessor");

    std::unique_ptr<Postprocessor> p;
    if (params.has_detection()) {
        p = std::make_unique<ThreadCudaPostprocessor<DetectPostprocessor>>(
            uniqueId(), params.detection());
    } else if (params.has_classification()) {
        p = std::make_unique<ThreadCudaPostprocessor<ClassifyPostprocessor>>(
            uniqueId(), params.classification());
    } else if (params.has_segmentation()) {
        p = std::make_unique<ThreadHostPostprocessor<
            ThreadCudaPostprocessor<SegmentPostprocessor>>>(
            uniqueId(), params.segmentation());
    } else if (params.has_trtis_classification()) {
        InferError(
            "INTERNAL: trtis_classification[deprecated] should"
            " be replaced to triton_classification");
        assert(false);
        return NVDSINFER_CONFIG_FAILED;
    } else if (params.has_triton_classification()) {
        p = std::make_unique<ThreadCudaPostprocessor<TrtIsClassifier>>(
            uniqueId(), params.triton_classification());
    } else if (params.has_other()) {
        p = std::make_unique<ThreadHostPostprocessor<
            ThreadCudaPostprocessor<OtherPostprocessor>>>(
            uniqueId(), params.other());
    } else {
        printError("no postprocessing params found from config files");
        return NVDSINFER_CONFIG_FAILED;
    }
    assert(p);
    p->setUniqueId(uniqueId());
    p->setDllHandle(customLib());
    p->setLabelPath(params.labelfile_path());
    p->setNetworkInfo(m_NetworkImageInfo);
    p->setOutputLayerCount(
        std::get<IBackend::kTpLayerNum>(backend()->getOutputLayers()));
    p->setInputCopy(needCopyInputToHost());
    p->setAllocator(
        std::bind(
            &InferCudaContext::acquireTensorHostBuf, this, std::placeholders::_1,
            std::placeholders::_2),
        std::bind(&InferCudaContext::acquireTensorHostEvent, this));
    m_FinalProcessor = p.get();
    processor = std::move(p);

    return NVDSINFER_SUCCESS;
}

NvDsInferStatus
InferCudaContext::preInference(SharedBatchArray& inputs, const ic::InferenceConfig& config)
{
    assert(config.has_backend());
    if (config.has_lstm()) {
        return ensureLstmInputReady(inputs, config);
    }
    if (hasExtraProcess()) {
        return m_ExtraProcessor->processExtraInputs(inputs);
    }

    assert(backend());
    if (inputs->getSize() >= backend()->getInputLayerSize()) {
        return NVDSINFER_SUCCESS;
    }

    if (m_ExtraInputs.empty()) {
        RETURN_NVINFER_ERROR(
            initFixedExtraInputLayers(inputs, config),
            "Init fixed extra input tensors failed.");
    }
    for (auto const& each : m_ExtraInputs) {
        inputs->addBuf(each);
    }
    assert(inputs->getSize() == backend()->getInputLayerSize());
    return NVDSINFER_SUCCESS;
}

NvDsInferStatus
InferCudaContext::initFixedExtraInputLayers(
    SharedBatchArray& inputs, const ic::InferenceConfig& config)
{
    printDebug("InferCudaContxt initFixedExtraInputLayers");

    // the 1st is image input
    auto imageTensor = inputs->getSafeBuf(0);
    assert(imageTensor && imageTensor->getBufDesc().isInput);
    std::string imageName = imageTensor->getBufDesc().name;

    /* Needs the custom library to be specified. */
    if (!customLib()) {
        printWarning(
            "More than one input layers but custom initialization "
            "function not implemented");
        return NVDSINFER_CUSTOM_LIB_FAILED;
    }

    /* Check if the interface to initialize the layers has been implemented. */
    auto initInputFcn =
        READ_SYMBOL(customLib(), NvDsInferInitializeInputLayers);
    if (initInputFcn == nullptr) {
        printWarning(
            "More than one input layers but custom initialization "
            "function not implemented");
        return NVDSINFER_CUSTOM_LIB_FAILED;
    }

    /* Interface implemented.  */
    /* Vector of NvDsInferLayerInfo for non-image input layers. */
    LayerDescriptionList inputLayers;
    const LayerDescription* originalInLayers = nullptr;
    int inputSize = 0;
    std::tie(originalInLayers, inputSize) = backend()->getInputLayers();
    for (int iL = 0; iL < inputSize; ++iL) {
        assert(originalInLayers[iL].isInput);
        if (originalInLayers[iL].name != imageName) {
            inputLayers.push_back(originalInLayers[iL]);
        }
    }

    /* Vector of host memories that can be initialized using CPUs. */
    m_ExtraInputs.resize(inputLayers.size());
    std::vector<SharedCudaTensorBuf> cudaCpuBufs(inputLayers.size());
    int gpuId = config.gpu_ids(0);
    std::vector<NvDsInferLayerInfo> capiLayersInfo(inputLayers.size());
    for (size_t i = 0; i < inputLayers.size(); i++) {
        /* For each layer calculate the size required for the layer, allocate
         * the host memory and assign the pointer to layer info structure. */
        assert(inputLayers[i].inferDims.numElements > 0);
        assert(!hasWildcard(inputLayers[i].inferDims));
        m_ExtraInputs[i] = createGpuTensorBuf(inputLayers[i].inferDims,
            inputLayers[i].dataType, maxBatchSize(), inputLayers[i].name,
            gpuId);
        RETURN_IF_FAILED(
            m_ExtraInputs[i], NVDSINFER_RESOURCE_ERROR,
            "Create extra GPU input tensor: %s buffer failed.",
            safeStr(inputLayers[i].name));
        m_ExtraInputs[i]->mutableBufDesc().isInput = true;
        cudaCpuBufs[i] = createCpuTensorBuf(inputLayers[i].inferDims,
            inputLayers[i].dataType, maxBatchSize(), inputLayers[i].name, gpuId);
        RETURN_IF_FAILED(
            cudaCpuBufs[i], NVDSINFER_RESOURCE_ERROR,
            "Create extra CPU input tensor: %s buffer failed.",
            safeStr(inputLayers[i].name));
        capiLayersInfo[i] =
            toCapi(inputLayers[i], cudaCpuBufs[i]->getBufPtr(0));
    }

    /* Call the input layer initialization function. */
    if (!initInputFcn(capiLayersInfo, m_NetworkImageInfo, maxBatchSize())) {
        printError(
            "Failed to initialize input layers using "
            "NvDsInferInitializeInputLayers() in custom lib");
        return NVDSINFER_CUSTOM_LIB_FAILED;
    }

    /* Memcpy the initialized contents from the host memory to device memory for
     * layer binding buffers. */
    for (size_t i = 0; i < inputLayers.size(); i++) {
        size_t byteSize = inputLayers[i].inferDims.numElements *
                      getElementSize(inputLayers[i].dataType) * maxBatchSize();
        CTX_RETURN_CUDA_ERR(
            cudaMemcpyAsync(
                m_ExtraInputs[i]->getBufPtr(0), cudaCpuBufs[i]->getBufPtr(0),
                byteSize, cudaMemcpyHostToDevice, *mainStream()),
            "Failed to copy from host to device memory");

        if (needCopyInputToHost()) {
            //m_ExtraInputs[i]->attach(cudaCpuBufs[i]);
        }
    }
    CTX_RETURN_CUDA_ERR(
        cudaStreamSynchronize(*mainStream()),
        "Failed to synchronize cuda stream");

    return NVDSINFER_SUCCESS;
}

NvDsInferStatus
InferCudaContext::ensureLstmInputReady(
    SharedBatchArray& inputs, const ic::InferenceConfig& config)
{
    assert(m_LstmController);
    CTX_RETURN_NVINFER_ERROR(
        m_LstmController->waitAndGetInputs(inputs),
        "LSTM failed to wait inputs ready");

    return NVDSINFER_SUCCESS;
}

NvDsInferStatus
InferCudaContext::extraOutputTensorCheck(SharedBatchArray& outputs, SharedOptions inOptions)
{
    if (hasExtraProcess()) {
        CTX_RETURN_NVINFER_ERROR(
            m_ExtraProcessor->checkInferOutputs(outputs, inOptions),
            "Extrac process failed to check inference outputs");
    }

    if (m_LstmController) {
        CTX_RETURN_NVINFER_ERROR(
            m_LstmController->feedbackInputs(outputs),
            "LSTM controller feedback output into inputs failed");
    }
    return NVDSINFER_SUCCESS;
}

void
InferCudaContext::notifyError(NvDsInferStatus status)
{
    if (hasExtraProcess()) {
        m_ExtraProcessor->notifyError(status);
    }
    if (m_LstmController) {
        m_LstmController->notifyError(status);
    }
}

SharedSysMem
InferCudaContext::acquireTensorHostBuf(const std::string& name, size_t bytes)
{
    SharedSysMem mem = std::move(m_HostTensorPool.acquireBuffer(name));
    if (mem->bytes() < bytes) {
        mem->grow(bytes);
    }
    assert(mem->ptr());
    return mem;
}

SharedCuEvent InferCudaContext::acquireTensorHostEvent() {
    return std::move(m_HostTensorEvents->acquireBuffer());
}

NvDsInferStatus InferCudaContext::addHostTensorPool(
    const LayerDescription& layer, int poolSize, int gpuId) {
    for (int iB = 0; iB < poolSize; ++iB) {
        uint32_t eleByte = getElementSize(layer.dataType);
        if (!eleByte) {
            eleByte = 4;
        }
        size_t bufBytes = eleByte * dimsSize(fullDims(maxBatchSize(), layer.inferDims));
        bufBytes = INFER_ROUND_UP(bufBytes, INFER_MEM_ALIGNMENT);
        if (bufBytes < INFER_MEM_ALIGNMENT) {
            bufBytes = INFER_MEM_ALIGNMENT;
        }

        UniqSysMem hostTensor = std::make_unique<CudaHostMem>(bufBytes);
        if (!hostTensor) {
            printError("failed to create cpu tensor:%s while adding tensor pool",
                safeStr(layer.name));
            return NVDSINFER_RESOURCE_ERROR;
        }
        if (!m_HostTensorPool.setBuffer(layer.name, std::move(hostTensor))) {
            printError("failed to add new tensor:%s buffer into map pool",
                safeStr(layer.name));
            return NVDSINFER_UNKNOWN_ERROR;
        }
    }
    assert(m_HostTensorPool.getPoolSize(layer.name) == (uint32_t)poolSize);
    return NVDSINFER_SUCCESS;
}

NvDsInferStatus InferCudaContext::allocateResource(
    const ic::InferenceConfig& config) {
    printDebug("InferCudaContxt allocateResource");

    std::vector<int> gpuIds(config.gpu_ids().begin(), config.gpu_ids().end());
    assert(gpuIds.size());
    CTX_RETURN_CUDA_ERR(
        cudaSetDevice(gpuIds[0]),
        "infer cuda context failed to set cuda device(%d) during allocating "
        "resource",
        gpuIds[0]);

    if (m_SurfaceConverter) {
        CTX_RETURN_NVINFER_ERROR(
            static_cast<BasePreprocessor*>(m_SurfaceConverter)
                ->allocateResource(gpuIds),
            "failed to allocate resource for surface converter.");
    }
    if (m_NetworkPreprocessor) {
        CTX_RETURN_NVINFER_ERROR(
            static_cast<BasePreprocessor*>(m_NetworkPreprocessor)
                ->allocateResource(gpuIds),
            "failed to allocate resource for network preprocessor.");
    }
    if (m_FinalProcessor) {
        CTX_RETURN_NVINFER_ERROR(
            m_FinalProcessor->allocateResource(gpuIds),
            "failed to allocate resource for postprocessor.");
    }

    int poolSize = tensorPoolSize();
    if (poolSize > kMaxHostTensorPoolSize)
        printWarning(
            "Attention !! Tensor pool size larger than max host tensor pool size: %d "
            "Continuing with user settings",
            kMaxHostTensorPoolSize);

    bool needCopyInput = needCopyInputToHost();
    for (const LayerDescription& layer : backend()->allLayers()) {
        if (layer.isInput && !needCopyInput) {
            continue;
        }
        CTX_RETURN_NVINFER_ERROR(
            addHostTensorPool(layer, poolSize, gpuIds[0]),
            "failed to allocate resource for postprocessor.");
    }
    m_HostTensorEvents = std::make_shared<BufferPool<UniqEventPtrInPool>>(
        "PostProcHostBufPool");
    for (int i = 0; i < poolSize; ++i) {
        UniqEventPtrInPool ev = std::make_unique<CudaEventInPool>(
            cudaEventDisableTiming|cudaEventBlockingSync, gpuIds[0]);
        assert(ev->ptr());
        m_HostTensorEvents->setBuffer(std::move(ev));
        assert(m_HostTensorEvents);
    }

    return NVDSINFER_SUCCESS;
}

} // namespace nvdsinferserver
