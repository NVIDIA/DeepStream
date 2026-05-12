/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "infer_grpc_client.h"
#include "infer_batch_buffer.h"
#include "infer_options.h"
#include "infer_utils.h"
#include "infer_trtis_utils.h"
#include "infer_cuda_utils.h"

/* Maximum size of buffer name string */
#define MAX_STR_LEN 64

namespace nvdsinferserver {

TritonGrpcRequest::~TritonGrpcRequest()
{
    for (auto &data: m_CpuData)
        free(data);
}

NvDsInferStatus
TritonGrpcRequest::appendInput(const std::shared_ptr<tc::InferInput> &input)
{
    m_InferInputs.push_back(input);
    return NVDSINFER_SUCCESS;
}

NvDsInferStatus
TritonGrpcRequest::setOutput(const std::vector<std::shared_ptr<tc::InferRequestedOutput>> &output)
{
    m_RequestOutputs = output;
    return NVDSINFER_SUCCESS;
}

NvDsInferStatus
TritonGrpcRequest::setOption(std::shared_ptr<tc::InferOptions> &option)
{
    m_InferOptions = option;
    return NVDSINFER_SUCCESS;
}

InferGrpcClient::InferGrpcClient (std::string url, bool enableCudaBufferSharing)
{
    m_Url = url;
    m_EnableCudaBufferSharing = enableCudaBufferSharing;
}

NvDsInferStatus
InferGrpcClient::Initialize()
{
    tc::Error err;
    err = tc::InferenceServerGrpcClient::Create(
            &m_GrpcClient, m_Url, false);

    if (!err.IsOk()) {
        InferError("failed to create gRPC client: %s", safeStr(err.Message()));
        return NVDSINFER_TRITON_ERROR;
    }
    return NVDSINFER_SUCCESS;
}

InferGrpcClient::~InferGrpcClient() {

}

NvDsInferStatus
InferGrpcClient::getModelMetadata (inference::ModelMetadataResponse *model_metadata,
    std::string &model_name, std::string &model_version)
{
    tc::Error err;

    if (!model_metadata || model_name.empty()) {
        InferError ("Invalid arguments to get model metadata");
        return NVDSINFER_INVALID_PARAMS;
    }

    err = m_GrpcClient->ModelMetadata(
        model_metadata, model_name, model_version);
    if (!err.IsOk()) {
        InferError("failed to get model(%s) metadata: %s", safeStr(model_name),
            safeStr(err.Message()));
        return NVDSINFER_TRITON_ERROR;
    }

    return NVDSINFER_SUCCESS;
}

NvDsInferStatus
InferGrpcClient::getModelConfig (inference::ModelConfigResponse *config, const std::string &name,
        const std::string &version, const tc::Headers &headers)
{
    tc::Error err;

    if (!config || name.empty()) {
        InferError ("Invalid arguments to get model config");
        return NVDSINFER_INVALID_PARAMS;
    }

    err = m_GrpcClient->ModelConfig(config, name, version, headers);
    if (!err.IsOk()) {
        InferError("failed to get model(%s) config: %s", safeStr(name),
            safeStr(err.Message()));
        return NVDSINFER_TRITON_ERROR;
    }
    return NVDSINFER_SUCCESS;
}

bool InferGrpcClient::isServerLive()
{
    tc::Error err;
    bool live;

    err = m_GrpcClient->IsServerLive (&live);
    if (!err.IsOk()) {
        InferError("error: server is not live: %s", safeStr(err.Message()));
        return false;
    }
    return true;
}

bool InferGrpcClient::isServerReady()
{
    tc::Error err;
    bool ready;

    err = m_GrpcClient->IsServerReady (&ready);
    if (!err.IsOk()) {
        InferError("error: server is not ready: %s", safeStr(err.Message()));
        return false;
    }
    return true;
}

bool InferGrpcClient::isModelReady (const std::string &model, const std::string version)
{
    bool ready;
    tc::Error err;

    err = m_GrpcClient->IsModelReady (&ready, model, version);
    if (!err.IsOk()) {
        InferError("error: model(%s) is not ready: %s", safeStr(model),
            safeStr(err.Message()));
        return false;
    }
    return true;
}

NvDsInferStatus
InferGrpcClient::LoadModel(const std::string& model_name, const Headers& headers)
{
  tc::Error err;

  err = m_GrpcClient->LoadModel(model_name, headers);
  if (!err.IsOk()) {
      InferError ("Triton failed to load the model %s, err: %s",
          safeStr(model_name), safeStr(err.Message()));
      return NVDSINFER_TRITON_ERROR;
  }
  return NVDSINFER_SUCCESS;
}

NvDsInferStatus
InferGrpcClient::UnloadModel(const std::string& model_name, const Headers& headers)
{
    tc::Error err;

    err = m_GrpcClient->UnloadModel(model_name, headers);
    if (!err.IsOk()) {
        InferError ("Triton failed to unload the model %s, err: %s",
            safeStr(model_name), safeStr(err.Message()));
        return NVDSINFER_TRITON_ERROR;
    }
    return NVDSINFER_SUCCESS;
}

#if !defined(__aarch64__) || defined(AARCH64_IS_SBSA)
static cudaError_t
CreateCUDAIPCHandle(cudaIpcMemHandle_t* cudaHandle, void* deviceMemPtr,
        int deviceId)
{
  /* Set the GPU device to the desired GPU */
  cudaError_t result = cudaSetDevice(deviceId);
  if (result != cudaSuccess) {
      return result;
  }
  /* Create IPC handle for data on the GPU */
  return cudaIpcGetMemHandle(cudaHandle, deviceMemPtr);
}

tc::Error
InferGrpcClient::SetInputCudaSharedMemory(tc::InferInput *inferInput,
        const SharedBatchBuf& inbuf, SharedGrpcRequest request, uint64_t bufId)
{
    tc::Error err;
    cudaIpcMemHandle_t inputCudaHandle;
    char bufferName[MAX_STR_LEN];
    const InferBufferDescription& inDesc = inbuf->getBufDesc();
    size_t bytes = inbuf->getTotalBytes();
    size_t bufOffset = inbuf->getBufOffset(0);
    void* bufBase = (void*)((char*)inbuf->getBufPtr(0) - bufOffset);
    size_t bufSize = bufOffset + bytes;

    snprintf(bufferName, MAX_STR_LEN, "inbuf_%p_%lu", inbuf.get(), bufId);
    std::string inputCudaBufName(bufferName);

    cudaError_t cuErr = CreateCUDAIPCHandle(&inputCudaHandle,
            bufBase, inDesc.devId);
    if (cuErr != cudaSuccess) {
        InferError("Failed to create IPC handle, err :%d, err_str:%s",
                (int)cuErr, cudaGetErrorName(cuErr));
        return tc::Error("CUDA IPC handle error");
    }

    err = m_GrpcClient->RegisterCudaSharedMemory(
            inputCudaBufName, inputCudaHandle, inDesc.devId, bufSize);
    if (!err.IsOk()) {
        InferError("Failed to register CUDA shared memory.");
        return err;
    }

    if (bufOffset == (size_t) -1) {
        return tc::Error("Invalid CUDA buffer offset.");
    }
    err = inferInput->SetSharedMemory(inputCudaBufName, bytes,
            bufOffset /* Offset of the buffer from the start of allocation */
            );
    if (!err.IsOk()) {
        InferError("Unable to set shared memory for input.");
        return err;
    }

    /* Attach the CUDA buffer to the request, for unregistering after
     * completing inference */
    request->attachInputCudaBuffer(inputCudaBufName);
    return err;
}
#endif /* __aarch64__ */

SharedGrpcRequest
InferGrpcClient::createRequest(const std::string &model, const std::string &version,
    SharedIBatchArray input, const std::vector<std::string> &outputs,
    const std::vector<TritonClassParams> &classList)
{
    tc::Error err;
    SharedGrpcRequest request;

    SharedBatchArray inputs = std::dynamic_pointer_cast<BaseBatchArray>(input);
    if (!inputs) {
        InferError (
            "InferGrpcClient input buffer is invalid to cast to "
            "BaseBatchArray");
        return nullptr;
    }

    request = std::make_shared<TritonGrpcRequest>();
    const std::vector<SharedBatchBuf>& inBufs = inputs->bufs();
    void *hostMem = NULL;

    for (auto &inbuf: inBufs) {
        tc::InferInput *inferInput;
        const InferBufferDescription& inDesc = inbuf->getBufDesc();

        InferDims fullShape = fullDims(inbuf->getBatchSize(), inDesc.dims);
        std::vector<int64_t> inDims(fullShape.d, fullShape.d + fullShape.numDims);

        std::string dataType = dataType2GrpcStr(inDesc.dataType);
        if (dataType.empty()) {
            InferError ("unable to get triton data type for %d", inDesc.dataType);
            return nullptr;
        }

        size_t bytes = inbuf->getTotalBytes();
        err = tc::InferInput::Create(&inferInput, inDesc.name, inDims, dataType);
        if (!err.IsOk()) {
            InferError("unable to create infer input:%s", safeStr(err.Message()));
            return nullptr;
        }

        err = inferInput->Reset();
        if (!err.IsOk()) {
            InferError("failed to reset infer input: %s", safeStr(err.Message()));
            delete inferInput;
            return nullptr;
        }

        if (inDesc.memType == InferMemType::kGpuCuda) {
            if (inbuf->cuEvent()) {
                cudaError_t err = cudaEventSynchronize(*inbuf->cuEvent());
                if (err != cudaSuccess) {
                    InferError("Failed to synchronize cuda events on buffer, err :%d, err_str:%s",
                        (int) err, cudaGetErrorName (err));
                }
            }
#if !defined(__aarch64__) || defined(AARCH64_IS_SBSA)
            if (m_EnableCudaBufferSharing) {
                err = SetInputCudaSharedMemory(inferInput, inbuf, request,
                        inputs->bufId());
            } else
#endif /* __aarch64__ */
            {
                hostMem = malloc(bytes);
                cudaMemcpy (hostMem, inbuf->getBufPtr(0), bytes, cudaMemcpyDeviceToHost);
                err = inferInput->AppendRaw((const uint8_t*)hostMem, bytes);
            }
        } else {
            err = inferInput->AppendRaw((const uint8_t*)inbuf->getBufPtr(0), bytes);
        }

        if (!err.IsOk()) {
            InferError("Failed to set inference input: %s", safeStr(err.Message()));
            inferInput->Reset();
            delete inferInput;
            return nullptr;
        }

        if (hostMem)
            request->attachData(hostMem);
        request->appendInput(std::shared_ptr<tc::InferInput>(inferInput));
    }

    std::vector<std::shared_ptr<tc::InferRequestedOutput>> reqOutputs;

    for (auto outName: outputs) {
        tc::InferRequestedOutput* reqOutput;
        err = tc::InferRequestedOutput::Create(&reqOutput, outName);
        if (!err.IsOk()) {
            InferError("unable to create request output: %s", safeStr(err.Message()));
            return nullptr;
        }
        reqOutputs.push_back(std::shared_ptr<tc::InferRequestedOutput>(reqOutput));
    }

    request->setOutput(reqOutputs);

    std::shared_ptr<tc::InferOptions> options = std::make_shared<tc::InferOptions>(model);
    options->model_version_ = version;
    options->request_id_ = std::to_string(m_LastRequestId++);

    const auto option = inputs->getOptions();
    if (option)
        parseOptions(options.get(), option);

    request->setOption(options);
    request->setInputBatchArray(input);
    request->setOutNames(outputs);

    return request;
}

NvDsInferStatus
InferGrpcClient::parseOptions(tc::InferOptions *outOpt, const IOptions *inOpt)
{
    uint64_t seqId = 0;
    if (inOpt->hasValue(OPTION_SEQUENCE_ID)) {
        RETURN_NVINFER_ERROR(
            inOpt->getUInt(OPTION_SEQUENCE_ID, seqId),
            "InferGrpcClient failed to get option " OPTION_SEQUENCE_ID);

        outOpt->sequence_id_ = seqId;
    }

    if (inOpt->hasValue(OPTION_SEQUENCE_START)) {
        bool f = 0;
        RETURN_NVINFER_ERROR(
            inOpt->getBool(OPTION_SEQUENCE_START, f),
            "InferGrpcClient failed to get option" OPTION_SEQUENCE_START);

        outOpt->sequence_start_ = f;
    }

    if (inOpt->hasValue(OPTION_SEQUENCE_END)) {
        bool f = 0;
        RETURN_NVINFER_ERROR(
            inOpt->getBool(OPTION_SEQUENCE_END, f),
            "InferGrpcClient failed to get option" OPTION_SEQUENCE_END);

        outOpt->sequence_end_ = f;
    }

    if (inOpt->hasValue(OPTION_PRIORITY)) {
        uint64_t priority = 0;
        RETURN_NVINFER_ERROR(
            inOpt->getUInt(OPTION_PRIORITY, priority),
            "InferGrpcClient failed to get option " OPTION_PRIORITY);

        outOpt->priority_ = priority;
    }

    if (inOpt->hasValue(OPTION_TIMEOUT)) {
        uint64_t timeout = 0;
        RETURN_NVINFER_ERROR(
            inOpt->getUInt(OPTION_TIMEOUT, timeout),
            "InferGrpcClient failed to get option " OPTION_TIMEOUT);

        outOpt->server_timeout_ = timeout;
    }

    return NVDSINFER_SUCCESS;
}

void InferGrpcClient::InferComplete (tc::InferResult *result,
    SharedGrpcRequest request, TritonGrpcAsyncDone asyncDone)
{
    tc::Error err;
    SharedBatchArray inputs;
    std::vector<int64_t> shape;
    std::vector<std::string> outNames;
    std::string datatype;
    const uint8_t* outBuf = nullptr;
    size_t byte_size = 0;
    InferDims dsDims{0, {0}};
    uint32_t batchSize = 0;

#if !defined(__aarch64__) || defined(AARCH64_IS_SBSA)
    for (auto &cudaBufName: request->getInputCudaBufNames()) {
        err = m_GrpcClient->UnregisterCudaSharedMemory(cudaBufName);
        if (!err.IsOk()) {
            InferError("failed to unregister input CUDA buffer");
            asyncDone(NVDSINFER_TRITON_ERROR, nullptr);
            return;
        }
    }
#endif /* __aarch64__ */

    if (!result->RequestStatus().IsOk()) {
        InferError("inference failed with error: %s",
           safeStr(result->RequestStatus().Message()));

        asyncDone(NVDSINFER_TRITON_ERROR, nullptr);
        return;
    }

    inputs = std::dynamic_pointer_cast<BaseBatchArray>(request->inputBatchArray());
    if (!inputs) {
        InferError("failed to retrieve input");
        asyncDone(NVDSINFER_TRITON_ERROR, nullptr);
        return;
    }

    assert(inputs->getSize());
    auto& inBuf0 = inputs->buf(0);
    uint32_t inBatchSize = inBuf0->getBatchSize();

    std::shared_ptr<tc::InferResult> result_ptr;
    result_ptr.reset(result);

    SharedBatchArray outputsArr = std::make_shared<BaseBatchArray>();
    outputsArr->setBufId(inputs->bufId());

    outNames = request->getOutNames();
    for (auto &outname: outNames) {
        err = result->Shape(outname, &shape);
        if (!err.IsOk()) {
            InferError("unable to get shape for %s", safeStr(outname));
            asyncDone(NVDSINFER_TRITON_ERROR, nullptr);
        }

        err = result->Datatype(outname, &datatype);
        if (!err.IsOk()) {
            InferError("unable to get output data type for %s", safeStr(outname));
            asyncDone(NVDSINFER_TRITON_ERROR, nullptr);
        }

        err = result->RawData(outname, &outBuf, &byte_size);
        if (!err.IsOk()) {
            InferError("unable to get tensor data for %s", safeStr(outname));
            asyncDone(NVDSINFER_TRITON_ERROR, nullptr);
        }

        bool maybeBatch = (!isNonBatch(inBatchSize) && inBatchSize == shape[0]);
        dsDims.numDims = (maybeBatch ? (shape.size() - 1) : shape.size());
        batchSize = (maybeBatch ? shape[0] : 0);
        for (uint32_t iD = 0; iD < dsDims.numDims; iD++) {
            dsDims.d[iD] = (int)shape[maybeBatch ? (iD + 1) : iD];
        }

        normalizeDims(dsDims);
        InferBufferDescription bufDesc{
            memType : InferMemType::kCpu,
            devId : 0,
            dataType : grpcStr2DataType(datatype),
            dims : dsDims,
            elementSize : getElementSize(grpcStr2DataType(datatype)),
            name : outname,
            isInput : false
        };

        SharedBatchBuf outBatchBuf;
        outBatchBuf.reset (new RefBatchBuffer((void *)outBuf, 0, byte_size, bufDesc,
                    batchSize), [result_ptr] (RefBatchBuffer *batchBuf) {
            delete batchBuf;
        });
        outputsArr->addBuf(outBatchBuf);
    }
    asyncDone(NVDSINFER_SUCCESS, outputsArr);
}

NvDsInferStatus
InferGrpcClient::inferAsync (SharedGrpcRequest request, TritonGrpcAsyncDone done)
{
    tc::Error err;
    tc::InferOptions options = *(request->getOption().get());

    std::vector<tc::InferInput*> inputs;
    std::vector<const tc::InferRequestedOutput*> outputs;

    for(auto input: request->inputs()) {
        inputs.push_back(input.get());
    }

    for(auto output: request->outputs()) {
        outputs.push_back(output.get());
    }

    err = m_GrpcClient->AsyncInfer(std::bind(&InferGrpcClient::InferComplete,
            this, std::placeholders::_1, request, done),
            options, inputs, outputs);

    if (!err.IsOk()) {
        InferError("failed to send synchronous infer request: %s", safeStr(err.Message()));
        return NVDSINFER_TRITON_ERROR;
    }
    return NVDSINFER_SUCCESS;
}

} // namespace nvdsinferserver
