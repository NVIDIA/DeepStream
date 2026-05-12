/*
 * SPDX-FileCopyrightText: Copyright (c) 2020 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "infer_simple_context.h"
#include "infer_batch_buffer.h"
#include "infer_cuda_utils.h"
#include "infer_postprocess.h"
#include "infer_preprocess.h"
#include "infer_proto_utils.h"
#include "infer_simple_runtime.h"
#include "infer_grpc_backend.h"
#include "infer_trtis_server.h"
#include "infer_utils.h"

namespace nvdsinferserver {

InferSimpleContext::InferSimpleContext() {}

InferSimpleContext::~InferSimpleContext() {}

NvDsInferStatus
InferSimpleContext::createNNBackend(
    const ic::BackendParams& params, int maxBatchSize, UniqBackend& backend)
{
    const ic::TritonParams& tritonParams = getTritonParam(params);
    const std::string model = tritonParams.model_name();

    if (tritonParams.model_name().empty()) {
        printError("config file doesn't have model_name");
        return NVDSINFER_CONFIG_FAILED;
    }
    std::set<std::string> outputs;
    for (const auto& out : params.outputs()) {
        const std::string outName = out.name();
        if (outName.empty()) {
            InferError("config file outputs.name has empty string");
            return NVDSINFER_CONFIG_FAILED;
        }
        outputs.emplace(outName);
    }

    std::unique_ptr<nvdsinferserver::TrtISBackend> be;
    if (tritonParams.has_grpc()) {
        be = std::make_unique<TritonGrpcBackend>(model, tritonParams.version());
    } else {
        be = std::make_unique<TritonSimpleRuntime>(model, tritonParams.version());
    }
    assert(be);

    be->setUniqueId(uniqueId());
    for (const auto& cls : tritonParams.class_params()) {
        uint32_t topK = std::max(cls.topk(), 1U);
        TritonClassParams c{topK, cls.threshold(), cls.tensor_name()};
        be->addClassifyParams(c);
        if (!c.tensorName.empty()) {
            outputs.emplace(c.tensorName);
        }
    }
    if (tritonParams.has_grpc()) {
        auto beGrpc = dynamic_cast<TritonGrpcBackend*>(be.get());
        beGrpc->setUrl (tritonParams.grpc().url());
        beGrpc->setOutputs(outputs);
    } else {
        auto beSR = dynamic_cast<TritonSimpleRuntime*>(be.get());
        beSR->setOutputs(outputs);
    }

    CTX_RETURN_NVINFER_ERROR(
        be->initialize(),
        "failed to initialize triton simple runtime for model:%s",
        safeStr(model));

    backend = std::move(be);
    return NVDSINFER_SUCCESS;
}

NvDsInferStatus
InferSimpleContext::fixateInferenceInfo(
    const ic::InferenceConfig& config, BaseBackend& backend)
{
    return NVDSINFER_SUCCESS;
}
NvDsInferStatus
InferSimpleContext::deinit()
{
    return NVDSINFER_SUCCESS;
}

NvDsInferStatus
InferSimpleContext::createPreprocessor(
    const ic::PreProcessParams& params,
    std::vector<UniqPreprocessor>& processors)
{
    printError("InferSimpleContext does not support <preprocess> from config");
    return NVDSINFER_CONFIG_FAILED;
}

NvDsInferStatus
InferSimpleContext::createPostprocessor(
    const ic::PostProcessParams& params, UniqPostprocessor& processor)
{
    printError("InferSimpleContext does not support <postprocess> from config");
    return NVDSINFER_CONFIG_FAILED;
}

NvDsInferStatus
InferSimpleContext::allocateResource(const ic::InferenceConfig& config)
{
    return NVDSINFER_SUCCESS;
}

}  // namespace nvdsinferserver

nvdsinferserver::IInferContext*
createInferTritonSimpleContext()
{
    return new nvdsinferserver::InferSimpleContext();
}
