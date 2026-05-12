/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "infer_lstm.h"

#include "infer_cuda_utils.h"

namespace nvdsinferserver {

NvDsInferStatus
LstmController::checkTensorInfo(BaseBackend& backend)
{
    assert(m_Params.loops_size());
    // check all tensors match backend input/output tensors
    for (const auto& perLoop : m_Params.loops()) {
        const LayerDescription* inInfo = backend.getLayerInfo(perLoop.input());
        const LayerDescription* outInfo =
            backend.getLayerInfo(perLoop.output());
        if (!inInfo || !inInfo->isInput) {
            InferError(
                "config lstm input:%s not found or not an input tensor",
                safeStr(perLoop.input()));
            return NVDSINFER_CONFIG_FAILED;
        }
        if (!outInfo || outInfo->isInput) {
            InferError(
                "config lstm output:%s not found or not an output tensor",
                safeStr(perLoop.output()));
            return NVDSINFER_CONFIG_FAILED;
        }
    }

    return NVDSINFER_SUCCESS;
}

namespace {
template <typename T>
void
fillBuf(void* ptr, uint32_t size, float value)
{
    for (uint32_t i = 0; i < size; ++i) {
        ((T*)ptr)[i] = (T)(value);
    }
}
}  // namespace

NvDsInferStatus
LstmController::initInputState(BaseBackend& backend)
{
    RETURN_NVINFER_ERROR(
        checkTensorInfo(backend), "failed to check lstm tensors");

    RETURN_CUDA_ERR(
        cudaSetDevice(m_DevId), "initInputState failed to set devId:%d",
        m_DevId);
    m_LstmStream = std::make_shared<CudaStream>(cudaStreamNonBlocking, m_DevId);
    assert(m_LstmStream && m_LstmStream->ptr());
    m_InputReadyEvent =
        std::make_shared<CudaEvent>(cudaStreamNonBlocking, m_DevId);

    uint32_t initBatch = backend.isNonBatching() ? 0 : m_MaxBatchSize;
    std::vector<SharedCudaTensorBuf> hostBufs;
    for (const auto& perLoop : m_Params.loops()) {
        const LayerDescription* inInfo = backend.getLayerInfo(perLoop.input());
        assert(inInfo);
        if (!perLoop.has_init_const()) {
            InferError("LSTM does not find init_const.");
            return NVDSINFER_INVALID_PARAMS;
        }
        SharedCudaTensorBuf hostBuf = createCpuTensorBuf(
            inInfo->inferDims, inInfo->dataType, initBatch, inInfo->name,
            m_DevId);
        if (!hostBuf) {
            InferError("LSTM create host buf failed. maybe Out-Of-Memory");
            return NVDSINFER_RESOURCE_ERROR;
        }
        void* hostPtr = hostBuf->getBufPtr(0);
        uint32_t perBatchSize = inInfo->inferDims.numElements;
        float value = perLoop.init_const().value();
        switch (inInfo->dataType) {
        case InferDataType::kFp32:
            fillBuf<float>(hostPtr, perBatchSize, value);
            break;
        case InferDataType::kInt8:
            fillBuf<int8_t>(hostPtr, perBatchSize, value);
            break;
        case InferDataType::kUint8:
            fillBuf<uint8_t>(hostPtr, perBatchSize, value);
            break;
        case InferDataType::kInt16:
            fillBuf<int16_t>(hostPtr, perBatchSize, value);
            break;
        case InferDataType::kUint16:
            fillBuf<uint16_t>(hostPtr, perBatchSize, value);
            break;
        case InferDataType::kInt32:
            fillBuf<int32_t>(hostPtr, perBatchSize, value);
            break;
        case InferDataType::kUint32:
            fillBuf<uint32_t>(hostPtr, perBatchSize, value);
            break;
        case InferDataType::kFp16:
        default:
            InferError(
                "LSTM init_state unsupport datatype:%s",
                safeStr(dataType2Str(inInfo->dataType)));
            return NVDSINFER_INVALID_PARAMS;
        }

        // create loopState
        SharedCudaTensorBuf inputTensor = createGpuTensorBuf(
            inInfo->inferDims, inInfo->dataType, initBatch, inInfo->name,
            m_DevId);
        if (!inputTensor) {
            InferError("LSTM create gput tensor failed. maybe Out-Of-Memory");
            return NVDSINFER_RESOURCE_ERROR;
        }
        inputTensor->mutableBufDesc().isInput = true;
        void* devPtr = inputTensor->getBufPtr(0);
        uint32_t perBatchBytes =
            perBatchSize * getElementSize(inInfo->dataType);

        for (uint32_t i = 0; i < (initBatch ? initBatch : 1); ++i) {
            RETURN_CUDA_ERR(
                cudaMemcpyAsync(
                    (void*)(((uint8_t*)devPtr) + perBatchBytes * i), hostPtr,
                    perBatchBytes, cudaMemcpyHostToDevice, *m_LstmStream),
                "Failed to copy LSTM init_state to loop_input");
        }
        hostBufs.emplace_back(hostBuf);
        m_LoopStateMap.emplace(
            perLoop.output(),
            LoopState{
                inInfo->name, inputTensor, nullptr, perLoop.keep_output()});
        m_LstmInputs.emplace_back(inputTensor);
    }
    RETURN_CUDA_ERR(
        cudaStreamSynchronize(*m_LstmStream),
        "Failed to synch lstm cuda stream");
    return NVDSINFER_SUCCESS;
}

NvDsInferStatus
LstmController::waitAndGetInputs(SharedBatchArray& inputs)
{
    auto appendInputsFcn = [this, &inputs]() {
        for (auto& eachIn : m_LstmInputs) {
            inputs->addBuf(eachIn);
        }
        m_InProgress = 1;
    };

    UniqLock locker(m_Mutex);
    if (m_InProgress == 0) {
        appendInputsFcn();
        return NVDSINFER_SUCCESS;
    } else if (m_InProgress < 0) {
        return NVDSINFER_UNKNOWN_ERROR;
    }
    m_Cond.wait(locker, [this]() { return m_InProgress <= 0; });
    assert(m_InProgress <= 0);
    locker.unlock();
    RETURN_CUDA_ERR(
        cudaEventSynchronize(*m_InputReadyEvent),
        "Failed to synchronize cuda events on each buffer");

    locker.lock();
    // release output buffers
    for (auto& iter : m_LoopStateMap) {
        LoopState& state = iter.second;
        state.outputTensor.reset();
    }
    if (m_InProgress < 0) {
        return NVDSINFER_UNKNOWN_ERROR;
    }

    appendInputsFcn();
    return NVDSINFER_SUCCESS;
}

NvDsInferStatus
LstmController::feedbackInputs(SharedBatchArray& outputs)
{
    // release and update
    struct Updater {
        LstmController& lstm;
        int ret = -1;
        Updater(LstmController& thisLstm) : lstm(thisLstm) {}
        ~Updater()
        {
            UniqLock locker(lstm.m_Mutex);
            lstm.m_InProgress = ret;
            lstm.m_Cond.notify_all();
        }
    };

    Updater updater(*this);

    auto& outBufs = outputs->mutableBufs();

    RETURN_CUDA_ERR(
        cudaSetDevice(m_DevId), "feedbackInput failed to set devId:%d",
        m_DevId);
    for (const SharedBatchBuf& buf : outBufs) {
        assert(buf);
        const InferBufferDescription& outDesc = buf->getBufDesc();
        auto iter = m_LoopStateMap.find(outDesc.name);
        if (iter == m_LoopStateMap.end()) {
            continue;
        }
        LoopState& state = iter->second;
        const InferBufferDescription& inDesc = state.inputTensor->getBufDesc();
        if (inDesc.dataType != outDesc.dataType ||
            inDesc.dims != outDesc.dims ||
            state.inputTensor->getBatchSize() != buf->getBatchSize()) {
            InferError(
                "LSTM output tensor:%s dtype/dims/batchSize does not match "
                "input tensor:%s",
                safeStr(outDesc.name), safeStr(inDesc.name));
            return NVDSINFER_UNKNOWN_ERROR;
        }
        // update input buf
        size_t byteSize = inDesc.dims.numElements *
                          getElementSize(inDesc.dataType) * buf->getBatchSize();
        enum cudaMemcpyKind kind = cudaMemcpyHostToDevice;
        if (outDesc.memType == InferMemType::kGpuCuda) {
            kind = cudaMemcpyDeviceToDevice;
        }
        RETURN_CUDA_ERR(
            cudaMemcpyAsync(
                state.inputTensor->getBufPtr(0), buf->getBufPtr(0), byteSize,
                kind, *m_LstmStream),
            "Failed to copy LSTM output to input loop");
        state.outputTensor = buf;
    }
    RETURN_CUDA_ERR(
        cudaEventRecord(*m_InputReadyEvent, *m_LstmStream),
        "Failed to record cuda event for LSTM feedback");

    outBufs.erase(
        std::remove_if(
            outBufs.begin(), outBufs.end(),
            [this](const SharedBatchBuf& buf) -> bool {
                auto& desc = buf->getBufDesc();
                const auto iState = m_LoopStateMap.find(desc.name);
                if (iState != m_LoopStateMap.end() &&
                    !(iState->second.keepOutputParsing)) {
                    return true;
                }
                return false;
            }),
        outBufs.end());
    // update condition
    updater.ret = 0;
    return NVDSINFER_SUCCESS;
}

void
LstmController::notifyError(NvDsInferStatus status)
{
    if (status == NVDSINFER_SUCCESS)
        return;
    UniqLock locker(m_Mutex);
    m_InProgress = -1;
    m_Cond.notify_all();
}

}  // namespace nvdsinferserver
