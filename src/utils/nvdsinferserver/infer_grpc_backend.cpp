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

#include "infer_grpc_backend.h"
#include "infer_batch_buffer.h"
#include "infer_trtis_utils.h"
#include "infer_trtis_server.h"

namespace nvdsinferserver {

TritonGrpcBackend::TritonGrpcBackend(std::string model, int64_t version)
    : TrtISBackend(model, version)
{}

TritonGrpcBackend::~TritonGrpcBackend()
{}

NvDsInferStatus TritonGrpcBackend::ensureModelReady() {
    InferDebug("TritonGrpcBackend id:%d ensure model: %s", uniqueId(), safeStr(model()));

    RETURN_IF_FAILED(
        m_InferGrpcClient->isModelReady(model()),
        NVDSINFER_TRITON_ERROR, "model:%s is not ready", safeStr(model()));

    return NVDSINFER_SUCCESS;
}

NvDsInferStatus
TritonGrpcBackend::setupLayersInfo()
{
    inference::ModelConfigResponse modelConfigRespose;
    inference::ModelConfig modelConfig;

    RETURN_NVINFER_ERROR(
        m_InferGrpcClient->getModelConfig(&modelConfigRespose, model()),
        "failed to get model: %s config", safeStr(model()));

    modelConfig = modelConfigRespose.config();

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
                safeStr(model()), safeStr(desc.name));
            return NVDSINFER_CONFIG_FAILED;
        }
        if (tensorOrder == InferTensorOrder::kNone) {
            tensorOrder = order;
        } else if (order != InferTensorOrder::kNone && order != tensorOrder) {
            InferError(
                "Triton failed to ensure model:%s since input:%s format "
                "disordered",
                safeStr(model()), safeStr(desc.name));
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

    return NVDSINFER_SUCCESS;
}

NvDsInferStatus
TritonGrpcBackend::ensureServerReady()
{
    RETURN_IF_FAILED(
        m_InferGrpcClient->isServerLive(), NVDSINFER_TRITON_ERROR,
        "failed to check server live state for model:%s", safeStr(model()));

    RETURN_IF_FAILED(
        m_InferGrpcClient->isServerReady(), NVDSINFER_TRITON_ERROR,
        "failed to check triton server ready", safeStr(model()));

    return NVDSINFER_SUCCESS;
}

NvDsInferStatus
TritonGrpcBackend::Run(SharedBatchArray inputs, InputsConsumed bufConsumed,
    AsyncDone asyncDone)
{
    assert(m_InferGrpcClient);

    std::set<std::string> outNames;
    requestTritonOutputNames(outNames);

    std::string versionStr;
    if (version() > 0)
        versionStr = std::to_string (version());

    SharedGrpcRequest request = m_InferGrpcClient->createRequest(model(), versionStr, inputs,
        {outNames.begin(), outNames.end()}, getClassifyParams());

    if (!request) {
        InferError("gRPC backend run failed to create request for model: %s",
            safeStr(model()));
        return NVDSINFER_TRITON_ERROR;
    }

    RETURN_NVINFER_ERROR(m_InferGrpcClient->inferAsync(request, asyncDone),
                         "gRPC async inference failed");

    return NVDSINFER_SUCCESS;
}

NvDsInferStatus
TritonGrpcBackend::initialize()
{
    m_InferGrpcClient = std::make_shared<InferGrpcClient>(m_Url,
            m_EnableCudaBufferSharing);

    RETURN_NVINFER_ERROR(m_InferGrpcClient->Initialize(),
        "failed to initialize backend while connecting GRPC endpoint: %s",
         safeStr(m_Url));
    RETURN_NVINFER_ERROR(ensureServerReady(),
        "failed to initialize backend while ensuring server ready");
    RETURN_NVINFER_ERROR(ensureModelReady(),
        "failed to initialize backend while ensuring model:%s ready",
        safeStr(model()));
    RETURN_NVINFER_ERROR(setupLayersInfo(),
        "failed to initialize backend while setup model:%s layer info",
        safeStr(model()));

    InferInfo("TritonGrpcBackend id:%d initialized for model: %s", uniqueId(), safeStr(model()));

    return NVDSINFER_SUCCESS;
}

void
TritonGrpcBackend::requestTritonOutputNames(std::set<std::string>& names)
{
    if (m_RequestOutputs.empty()) {
        TrtISBackend::requestTritonOutputNames(names);
        return;
    }
    names = m_RequestOutputs;
}

NvDsInferStatus
TritonGrpcBackend::enqueue(
    SharedBatchArray inputs, SharedCuStream stream, InputsConsumed bufConsumed,
    InferenceDone inferDone)
{
    assert(inputs && inputs->getSize() > 0);
    int batchSize = inputs->buf(0)->getBatchSize();

    InferDebug("TritonGrpcBackend id:%d enqueue batch:%d for model: %s to inference",
        uniqueId(), batchSize, safeStr(model()));

    RETURN_NVINFER_ERROR(
        Run(std::move(inputs), std::move(bufConsumed), std::move(inferDone)),
        "TritonSimple failed to run inference on model %s", safeStr(model()));
    return NVDSINFER_SUCCESS;
}

} // namespace nvdsinferserver
