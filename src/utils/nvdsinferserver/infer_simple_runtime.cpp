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

#include "infer_simple_runtime.h"
#include "infer_batch_buffer.h"
#include "infer_trtis_server.h"
#include "infer_cuda_utils.h"

namespace nvdsinferserver {

TritonSimpleRuntime::TritonSimpleRuntime(std::string model, int64_t version)
    : TrtISBackend(model, version)
{}

TritonSimpleRuntime::~TritonSimpleRuntime()
{}

NvDsInferStatus
TritonSimpleRuntime::initialize()
{
    RETURN_NVINFER_ERROR(ensureServerReady(),
        "failed to initialize backend while ensuring server ready");
    assert(server());

    RETURN_NVINFER_ERROR(ensureModelReady(),
        "failed to initialize backend while ensuring model:%s ready",
        safeStr(model()));

    const static uint32_t sMaxTryTimes = 6;
    const static uint32_t sEachTrySecs = 5;
    uint32_t iTry = 0;
    for (; iTry < sMaxTryTimes; ++iTry) {
        if (server()->isModelReady(model(), version())) {
            break;
        }
        InferWarning(
            "model:%s is not ready, retry %d times", safeStr(model()),
            iTry + 1);
        sleep(sEachTrySecs);  // sleep 5 seconds
    }
    RETURN_IF_FAILED(
        iTry < sMaxTryTimes, NVDSINFER_TRITON_ERROR, "model:%s is not ready, check triton backends",
        safeStr(model()));

    RETURN_NVINFER_ERROR(setupLayersInfo(),
        "failed to initialize backend while setup model:%s layer info",
        safeStr(model()));
#if 0
    RETURN_NVINFER_ERROR(setupReorderThread(),
        "failed to initialize backend(%s) while setup reorder thread",
        safeStr(model()));
    assert(m_ReorderThread);
#endif

    setAllocator(std::make_unique<TrtServerAllocator>(
        std::bind(
            &TritonSimpleRuntime::allocateSimpleRes, this,
            std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
            std::placeholders::_4),
        &TritonSimpleRuntime::releaseSimpleRes));

    InferInfo("TrtISBackend id:%d initialized model: %s", uniqueId(), safeStr(model()));

    return NVDSINFER_SUCCESS;
}

NvDsInferStatus
TritonSimpleRuntime::specifyInputDims(const InputShapes& shapes)
{
    InferError("TritonSimpleRuntime should never call specifyInputDims");
    return NVDSINFER_UNKNOWN_ERROR;
}

void
TritonSimpleRuntime::requestTritonOutputNames(std::set<std::string>& names)
{
    if (m_RequestOutputs.empty()) {
        TrtISBackend::requestTritonOutputNames(names);
        return;
    }
    names = m_RequestOutputs;
}

NvDsInferStatus
TritonSimpleRuntime::enqueue(
    SharedBatchArray inputs, SharedCuStream stream, InputsConsumed bufConsumed,
    InferenceDone inferDone)
{
    assert(!stream); // stream should be nullptr
    assert(inputs && inputs->getSize() > 0);

    //TODO check dims and batchSize.
    int batchSize = inputs->buf(0)->getBatchSize();
    // getFull-dims;
    //if (nonBatching()) {
    //    getfullDims;
    //    checkInputDims();
    //}

    InferDebug("TritonSimpleRuntime id:%d enqueue batch:%d for model: %s to inference",
        uniqueId(), batchSize, safeStr(model()));

    RETURN_NVINFER_ERROR(
        Run(std::move(inputs), std::move(bufConsumed), std::move(inferDone)),
        "TritonSimple failed to run inference on model %s", safeStr(model()));
    return NVDSINFER_SUCCESS;
}

SharedSysMem
TritonSimpleRuntime::allocateSimpleRes(
    const std::string& tensor, size_t bytes, InferMemType memType, int64_t devId)
{
    InferDebug(
        "TritonSimpleRuntime id:%d allocated response mem:%uB for"
        "model: %s on tensor:%s ",
        uniqueId(), (uint32_t)bytes, safeStr(model()), safeStr(tensor));

    if (!bytes) {
        bytes = 1;
    }
    bytes = INFER_ROUND_UP(bytes, INFER_MEM_ALIGNMENT);
    auto ret = std::make_shared<CpuMem>(bytes);
    return ret;
}

void
TritonSimpleRuntime::releaseSimpleRes(const std::string& tensor, SharedSysMem buf)
{
    //TODO nothing check
    InferDebug("TritonSimpleRuntime released tensor:%s mem", safeStr(tensor));
    buf.reset();
}

}
