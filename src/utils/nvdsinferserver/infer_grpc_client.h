/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @file infer_grpc_client.h
 *
 * @brief Header file for the gRPC client and request class for inference
 * using the Triton Inference Server. Uses the Triton client library for
 * inference using gRPC.
 */

#ifndef __INFER_GRPC_CLIENT_H__
#define __INFER_GRPC_CLIENT_H__

#include <stdarg.h>
#include <condition_variable>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <queue>

#include "infer_icontext.h"
#include "infer_post_datatypes.h"
#include "infer_common.h"
#include "grpc_client.h"


namespace tc = triton::client;

namespace nvdsinferserver {

typedef std::map<std::string, std::string> Headers;

class TritonGrpcRequest;

using SharedGrpcRequest = std::shared_ptr<TritonGrpcRequest>;

using TritonGrpcAsyncDone = std::function<void(NvDsInferStatus, SharedBatchArray)>;


/**
 * @brief Triton gRPC inference request class holding data associated with one
 * inference request.
 */
class TritonGrpcRequest {
public:
    /**
     * @brief Destructor, free the host memory allocated for the request.
     */
    ~TritonGrpcRequest();
    /**
     * @brief Append the array of Triton client library inference input objects.
     */
    NvDsInferStatus appendInput(const std::shared_ptr<tc::InferInput> &input);
    /**
     * @brief Helper functions to access the member variables.
     */
    /**@{*/
    NvDsInferStatus setOutput(const std::vector<std::shared_ptr<tc::InferRequestedOutput>> &output);
    NvDsInferStatus setOption(std::shared_ptr<tc::InferOptions> &option);
    std::vector<std::shared_ptr<tc::InferInput>> inputs() { return m_InferInputs; }
    std::vector<std::shared_ptr<tc::InferRequestedOutput>> outputs() { return m_RequestOutputs; }
    std::shared_ptr<tc::InferOptions> getOption() {return m_InferOptions;}
    SharedIBatchArray inputBatchArray() { return m_InputBatchArray;}
    void setInputBatchArray(SharedIBatchArray inputBatch) { m_InputBatchArray = inputBatch;}
    std::vector<std::string> getOutNames() { return m_OutputNames;}
    std::vector<std::string> getInputCudaBufNames() { return m_InputCudaBufNames;}
    void setOutNames(std::vector<std::string> outnames) {m_OutputNames = outnames;}
    /**@}*/

    /**
     * @brief Append the array of host memory allocations.
     */
    void attachData(void *data) {
        m_CpuData.push_back(data);
    }

    /**
     * @brief Append the list of shared CUDA input buffers.
     */
    void attachInputCudaBuffer(std::string bufName) {
        m_InputCudaBufNames.push_back(bufName);
    }

private:
    /**
     * @brief Array of Triton client library input objects for the request.
     */
    std::vector<std::shared_ptr<tc::InferInput>> m_InferInputs;
    /**
     * @brief Array of Triton client library output objects for the request.
     */
    std::vector<std::shared_ptr<tc::InferRequestedOutput>> m_RequestOutputs;
    /**
     * @brief Pointer to the inference options message for the request.
     */
    std::shared_ptr<tc::InferOptions> m_InferOptions;
    /**
     * @brief Array of names of requested output tensors.
     */
    std::vector<std::string> m_OutputNames;
    /**
     * @brief Pointer to the input batch buffer array of the request.
     */
    SharedIBatchArray m_InputBatchArray;
    /**
     * @brief Array of pointers to host memory used for input buffers.
     */
    std::vector<void*> m_CpuData;
    /**
     * @brief Array of names of input CUDA buffers shared with Triton.
     */
    std::vector<std::string> m_InputCudaBufNames;
};

/**
 * @brief Wrapper class for the gRPC client of the Triton Inference Server,
 * interfaces with the Triton client library.
 */
class InferGrpcClient {
public:
    /**
     * @brief Constructor, save the server server URL and CUDA sharing flag.
     * @param[in] url                      The Triton server address.
     * @param[in] enableCudaBufferSharing  Flag to enable CUDA buffer sharing.
     */
    InferGrpcClient (std::string url, bool enableCudaBufferSharing);
    /**
     * @brief Destructor, default.
     */
    ~InferGrpcClient();
    /**
     * @brief Create the gRPC client instance of the Triton Client library.
     * @return Error status.
     */
    NvDsInferStatus Initialize();
    /**
     * @brief Get the model metadata from the Triton Inference server.
     * @param[out] model_metadata The model metadata protobuf message.
     * @param[in]  model_name     Model name.
     * @param[in]  model_version  Model version.
     * @return Error status.
     */
    NvDsInferStatus getModelMetadata(inference::ModelMetadataResponse *model_metadata,
        std::string &model_name, std::string &model_version);
    /**
     * @brief Get the model configuration from the Triton Inference Server.
     * @param[out] config The model configuration protobuf message.
     * @param[in] name    Model name.
     * @param[in] version Model version.
     * @param[in] headers Optional HTTP headers to be included in the gRPC request.
     * @return Error status.
     */
    NvDsInferStatus getModelConfig(inference::ModelConfigResponse *config, const std::string &name,
        const std::string &version = "", const Headers &headers = Headers());
    /**
     * @brief Check if the Triton Inference Server is live.
     */
    bool isServerLive();
    /**
     * @brief Check if the Triton Inference Server is ready.
     */
    bool isServerReady();
    /**
     * @brief Check if the specified model is ready for inference.
     */
    bool isModelReady(const std::string &model, const std::string version = "");
    /**
     * @brief Request to load the given model using the Triton client library.
     */
    NvDsInferStatus LoadModel(const std::string& model_name, const Headers& headers = Headers());
    /**
     * @brief Request to unload the given model using the Triton client library.
     */
    NvDsInferStatus UnloadModel(const std::string& model_name, const Headers& headers = Headers());
    /**
     * @brief Create a new gRPC inference request. Create the Triton client
     * library InferInput objects from the input and copy/register the
     * input buffers. Create InferRequestedOutput objects for the output
     * layers.
     * @param[in] model     Model name.
     * @param[in] version   Model version.
     * @param[in] input     Array of input batch buffers.
     * @param[in] outputs   List of output layer names.
     * @param[in] classList List of configured Triton classification parameters.
     * @return Pointer to the gRPC inference request object created.
     */
    SharedGrpcRequest createRequest(const std::string& model, const std::string &version, SharedIBatchArray input,
        const std::vector<std::string> &outputs,
        const std::vector<TritonClassParams>& classList = std::vector<TritonClassParams>());

    /**
     * @brief Get the inference input and output list from the request and
     * trigger the asynchronous inference request using the Triton client
     * library.
     * @param[in] request The inference request object.
     * @param[in] done    The inference complete callback.
     * @return Error status.
     */
    NvDsInferStatus inferAsync(SharedGrpcRequest request, TritonGrpcAsyncDone done);

private:
    /**
     * @brief Callback function for the gRPC inference request. De-register the
     * shared CUDA buffers, populate output batch buffer array and call the
     * inference done callback function from the calling backend.
     * @param[in] result   The inference result from the Triton client library.
     * @param[in] request  Pointer to the inference request object.
     * @param[in] done     The callback function provided with inferAsync().
     */
    void InferComplete (tc::InferResult *result, SharedGrpcRequest request,
        TritonGrpcAsyncDone done);
    /**
     * @brief Parse the input nvinferserver inference options and populate the
     * Triton client library inference options.
     * @param[out] outOpt  Triton client library options object to be populated.
     * @param[in]  inOpt   Input nvinfereserver options.
     * @return Error status.
     */
    NvDsInferStatus parseOptions(tc::InferOptions *outOpt, const IOptions *inOpt);
    /**
     * @brief Register the input buffer as a CUDA buffer shared with Triton.
     * @param[out] inferInput The Triton client library inference input object
     *                        to attach the buffer.
     * @param[in]  inbuf      The input buffer for the layer.
     * @param[out] request    The gRPC request instance to register the buffer.
     * @return Error status.
     */
    tc::Error SetInputCudaSharedMemory(tc::InferInput *inferInput,
        const SharedBatchBuf& inbuf, SharedGrpcRequest request, uint64_t bufId);

private:
    /**
     * @brief Network address of the Triton Inference Server.
     */
    std::string m_Url;
    /**
     * @brief Flag to enable CUDA buffer sharing with Triton Inference Server.
     */
    bool m_EnableCudaBufferSharing;
    /**
     * @brief Handle to the gRPC client object of the Triton client library.
     */
    std::unique_ptr<tc::InferenceServerGrpcClient> m_GrpcClient;
    /**
     * @brief Counter to track the gRPC inference requests.
     */
    std::atomic<uint64_t> m_LastRequestId{UINT64_C(0)};
};

}  // namespace nvdsinferserver

#endif
