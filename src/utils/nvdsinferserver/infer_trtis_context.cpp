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
 * @file infer_trtis_context.cpp
 *
 * @brief Source file for the Triton C-API mode inference context class.
 *
 * This file defines the inference context class that uses the C_API
 * mode of the Triton Inference Server.
 */

#pragma GCC diagnostic ignored "-Wstringop-overflow="

#include <set>

#include "infer_base_backend.h"
#include "infer_cuda_utils.h"
#include "infer_preprocess.h"
#include "infer_proto_utils.h"
#include "infer_trtis_backend.h"
#include "infer_trtis_context.h"
#include "infer_trtis_server.h"
#include "infer_trtis_utils.h"

namespace nvdsinferserver {

/**
 * @brief Maximum preprocessor pool sizes.
 */
/**@{*/
const static int32_t kNetworkPoolSize = 3;
const static int32_t kNonPreprocessNetworkPoolSize = 5;
/**@}*/

InferTrtISContext::InferTrtISContext() {}

InferTrtISContext::~InferTrtISContext() {
}

NvDsInferStatus InferTrtISContext::deinit() {
    if (m_MainStream) {
        int gpuId = m_MainStream->devId();
        CTX_RETURN_CUDA_ERR(
            cudaSetDevice(gpuId),
            "InferTrtISContext failed to set cuda device(%d) during deinit",
            gpuId);
        cudaStreamSynchronize(*m_MainStream);
        m_MainStream.reset();
    }

    return InferCudaContext::deinit();
}

NvDsInferStatus
InferTrtISContext::getConfigInOutMap(
    const ic::BackendParams& params,
    std::unordered_map<std::string, InferDims>& inputLayers,
    std::set<std::string>& outputLayers)
{
    for (const auto& in : params.inputs()) {
        std::string inName = in.name();
        InferDims confShape{0, {0}};
        confShape.numDims = in.dims_size();
        for (int i = 0; i < (int)in.dims_size(); ++i) {
            confShape.d[i] = (int)in.dims(i);
        }
        InferDims fullShape;
        if (in.dims_size()) {
            fullShape = fullDims(maxBatchSize(), confShape);
        }
        if (inputLayers.count(inName)) {
            InferError("config file has duplicate BackendParams.inputs.name:%s",
                safeStr(inName));
            return NVDSINFER_CONFIG_FAILED;
        }
        inputLayers[inName] = fullShape;
    }

    for (const auto& out : params.outputs()) {
        const std::string outName = out.name();
        if (outName.empty()) {
            InferError("config file doesn't have correct "
                "BackendParams.outputs[i].name");
            return NVDSINFER_CONFIG_FAILED;
        }
        if (outputLayers.count(outName)) {
            InferError("config file has duplicate "
              "BackendParams.outputs.name:%s", safeStr(outName));
            return NVDSINFER_CONFIG_FAILED;
        }
        outputLayers.insert(outName);
    }
    return NVDSINFER_SUCCESS;
}

NvDsInferStatus
InferTrtISContext::specifyBackendDims(
    BaseBackend* be, const std::string& model, const ic::BackendParams& params)
{
    // Map of input layer names and full dimensions.
    std::unordered_map<std::string, InferDims> configInputs;
    std::set<std::string> configOutputs;
    CTX_RETURN_NVINFER_ERROR(
        getConfigInOutMap(params, configInputs, configOutputs),
        "failed to create triton backend on model:%s when getting "
        "inputs/outputs",
        safeStr(model));

    const LayerDescription* beInputs = nullptr;
    int beInputSize = 0;
    std::tie(beInputs, beInputSize) = be->getInputLayers();
    bool nonBatching = be->isNonBatching();
    int32_t beBatch = nonBatching ? 0 : maxBatchSize();
    assert(maxBatchSize() > 0);  // context must have reasonable batchSize
    bool firstDimIsBatch = false;  // for non-batching test

    // Full dimensions for specifying to the backend.
    IBackend::InputShapes warmupInputs;
    for (int i = 0; i < beInputSize; ++i) {
        assert(&beInputs[i]);
        InferDims modelShape = fullDims(beBatch, beInputs[i].inferDims);
        // Check for non-batching NCHW or NHWC
        static const uint32_t kImageNumFullDims = 4;
        if (modelShape.numDims >= kImageNumFullDims &&
            modelShape.d[0] <= INFER_WILDCARD_DIM_VALUE) {
            modelShape.d[0] = maxBatchSize();
            firstDimIsBatch = true;
        }
        auto iConf = configInputs.find(beInputs[i].name);
        if (iConf != configInputs.end()) {
            if (iConf->second.numDims) {  // intersect dims if configured
                InferDims fixed;
                if (!intersectDims(modelShape, iConf->second, fixed)) {
                    printError(
                        "failed to create triton backend on model:%s "
                        "because tensor:%s plugin dims: %s is not matched with "
                        "model dims: %s",
                        safeStr(model), safeStr(beInputs[i].name), safeStr(dims2Str(iConf->second)),
                        safeStr(dims2Str(modelShape)));
                    return NVDSINFER_CONFIG_FAILED;
                }
                modelShape = fixed;
            }
            configInputs.erase(iConf);
        }
        if (modelShape.d[0] <= INFER_WILDCARD_DIM_VALUE && maxBatchSize()) {
            modelShape.d[0] = maxBatchSize();
        }
        if (!modelShape.numDims || hasWildcard(modelShape)) {
            printError(
                "failed to figure out on model :%s, tensor:%s, dims: %s, "
                "please specify fixed dims without batchsize in "
                "\ninfer_config{ backend{ inputs [{\n\tname:%s \n\tdims[...] \n}] } }\n",
                safeStr(model), safeStr(beInputs[i].name), safeStr(dims2Str(modelShape)),
                safeStr(beInputs[i].name));
            return NVDSINFER_CONFIG_FAILED;
        }
        InferBatchDims batchDim{0, modelShape};
        if (!nonBatching) {
            uint32_t b = 0;
            debatchFullDims(modelShape, batchDim.dims, b);
            batchDim.batchSize = b;
        }
        warmupInputs.emplace_back(std::make_tuple(beInputs[i].name, batchDim));
    }

    // If any layer has a fixed dims in non-batching mode, then it means
    // some input tensors in mixed-batching, cannot set to first dim batching.
    if (nonBatching) {
        for (int i = 0; i < beInputSize; ++i) {
            assert(&beInputs[i]);
            if (!hasWildcard(beInputs[i].inferDims)) {
                firstDimIsBatch = false;
            }
        }
    }

    if (!configInputs.empty()) {
        for (auto& inC : configInputs) {
            printError(
                "input tensor:%s is not found in model:%s from config file",
                safeStr(inC.first), safeStr(model));
        }
    }

    for (auto& outC : configOutputs) {
        const LayerDescription* outLayer = be->getLayerInfo(outC);
        RETURN_IF_FAILED(
            outLayer && !outLayer->isInput, NVDSINFER_CONFIG_FAILED,
            "configured output tensor:%s is not found in triton model:%s",
            safeStr(outC), safeStr(model));
    }

    if (!params.disable_warmup()) {
        CTX_RETURN_NVINFER_ERROR(
            be->specifyInputDims(warmupInputs),
            "failed to specify input dims triton backend for model:%s", safeStr(model));
    }

    be->setFirstDimBatch(firstDimIsBatch);

    return NVDSINFER_SUCCESS;
}

NvDsInferStatus InferTrtISContext::createNNBackend(
    const ic::BackendParams& params, int maxBatchSize, UniqBackend& backend) {
    assert(config().has_backend());
    if (!hasTriton(params)) {
        printError("config file doesn't have triton parameters");
        return NVDSINFER_CONFIG_FAILED;
    }

    assert(config().gpu_ids_size());
    int32_t defaultGpu = config().gpu_ids(0);
    CTX_RETURN_CUDA_ERR(
        cudaSetDevice(defaultGpu),
        "InferTrtISContext failed to set cuda device(%d) during creating"
        "NN backend",
        defaultGpu);

    const ic::TritonParams& tritonParams = getTritonParam(params);
    const std::string model = tritonParams.model_name();

    if (tritonParams.model_name().empty()) {
        printError("config file doesn't have model_name");
        return NVDSINFER_CONFIG_FAILED;
    }

    TrtServerPtr server;
    std::string debugStr{"null"};
    if (tritonParams.has_model_repo()) {
        const auto& modelRepo = tritonParams.model_repo();
        const auto& gpus = config().gpu_ids();
        triton::RepoSettings repoSettings;
        if (!repoSettings.initFrom(modelRepo, {gpus.begin(), gpus.end()})) {
            printError(
                "model:%s repo settings failed. info:%s", safeStr(model),
                safeStr(modelRepo.DebugString()));
            return NVDSINFER_CONFIG_FAILED;
        }
        server = TrtISServer::getInstance(&repoSettings);
        debugStr = repoSettings.debugStr;
    } else {
        server = TrtISServer::getInstance(nullptr);
    }

    if (!server) {
        printError(
            "model:%s get triton server instance failed. repo:%s",
            safeStr(model), safeStr(debugStr));
        return NVDSINFER_TRITON_ERROR;
    }

    UniqTrtISBackend be =
        std::make_unique<TrtISBackend>(model, tritonParams.version(), std::move(server));
    assert(be);
    be->setUniqueId(uniqueId());
    be->setKeepInputs(needCopyInputToHost());
    int poolSize = tensorPoolSize();
    if (!needPreprocess()) {
        poolSize = std::max<int>(poolSize, kNetworkPoolSize);
    } else {
        poolSize = std::max<int>(poolSize, kNonPreprocessNetworkPoolSize);
    }
    be->setOutputPoolSize(poolSize);
    be->setOutputDevId(defaultGpu);
    for (const auto& bc : params.outputs()) {
        be->setTensorMaxBytes(bc.name(), bc.max_buffer_bytes());
    }

    InferMemType outputMemType = memTypeFromDsProto(params.output_mem_type());
    be->setOutputMemType(outputMemType);

    if (config().has_postprocess() && config().postprocess().has_triton_classification()) {
        auto const& tritonClass = config().postprocess().triton_classification();
        uint32_t topK = std::max(tritonClass.topk(), 1U);
        TritonClassParams c{topK, tritonClass.threshold(), tritonClass.tensor_name()};
        be->addClassifyParams(c);
    }

    CTX_RETURN_NVINFER_ERROR(
        be->initialize(), "failed to initialize triton backend for model:%s", safeStr(model));

    CTX_RETURN_NVINFER_ERROR(
        specifyBackendDims(be.get(), model, params),
        "failed to specify triton backend input dims for model:%s", safeStr(model));

    m_Backend = be.get();
    backend = std::move(be);
    return NVDSINFER_SUCCESS;
}

NvDsInferStatus InferTrtISContext::allocateResource(
    const ic::InferenceConfig& config) {
    auto gpuIds = config.gpu_ids();
    assert (!gpuIds.empty());

    CTX_RETURN_CUDA_ERR(
        cudaSetDevice(*gpuIds.begin()),
        "InferTrtISContext failed to set cuda device(%d) during allocating "
        "resource",
        *gpuIds.begin());
    m_MainStream =
        std::make_shared<CudaStream>(cudaStreamNonBlocking, *gpuIds.begin());
    assert(m_MainStream && m_MainStream->ptr());

    // Limit preprocesss pool size for async mode
    int networkPoolSize = m_Backend->outputPoolSize();
    if (m_NetworkPreprocessor &&
        m_NetworkPreprocessor->poolSize() > networkPoolSize) {
        m_NetworkPreprocessor->setPoolSize(networkPoolSize);
    }

    return InferCudaContext::allocateResource(config);
}

} // namespace nvdsinferserver

nvdsinferserver::IInferContext*
createInferTrtISContext(const char* configStr, uint32_t configStrLen)
{
    return new nvdsinferserver::InferTrtISContext;
}
