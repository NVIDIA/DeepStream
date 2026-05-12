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
 * @file infer_trtis_server.h
 *
 * @brief Header file of wrapper classes for Triton Inference Server
 * server instance, inference request, response.
 *
 * This file declares the wrapper classes used for inference processing
 * using the Triton Inference Server C-API mode.
 *
 */

#ifndef __NVDSINFER_TRTIS_SERVER_H__
#define __NVDSINFER_TRTIS_SERVER_H__

#include <string>
#include <thread>
#include <vector>
#include <memory>

#include "infer_batch_buffer.h"
#include "infer_post_datatypes.h"
#include "infer_proto_utils.h"
#include "infer_trtis_utils.h"

#include "nvds_version.h"

#ifdef IS_TEGRA
#define TRITON_DEFAULT_MINIMUM_COMPUTE_CAPABILITY 5.3
#define TRITON_DEFAULT_PINNED_MEMORY_BYTES (1 << 26)
#else
#define TRITON_DEFAULT_MINIMUM_COMPUTE_CAPABILITY 6.0
#define TRITON_DEFAULT_PINNED_MEMORY_BYTES (1 << 28)
#endif
#define TRITON_DEFAULT_BACKEND_DIR "/opt/tritonserver/backends"

struct TRITONSERVER_Server;

namespace nvdsinferserver {

namespace ni = inference;

class TrtISServer;
class TrtServerRequest;
class TrtServerResponse;
class TrtServerAllocator;

using SharedRequest = std::shared_ptr<TrtServerRequest>;
using UniqResponse = std::unique_ptr<TrtServerResponse>;
using SharedResponse = std::shared_ptr<TrtServerResponse>;

using TritonInferAsyncDone = std::function<void(SharedRequest, UniqResponse)>;

/**
 * @brief Wrapper class for Triton inference request.
 */
class TrtServerRequest {
protected:
    friend class TrtISServer;
    /**
     * @brief Constructor. Save the server instance pointer and
     * register the Triton request deletion function.
     * @param server
     */
    TrtServerRequest(TrtServerPtr server);

    /**
     * @brief Create a new Triton inference request with the specified
     * inputs and parameters.
     * @param[in] model    Model name.
     * @param[in] version  Model version.
     * @param[in] inputs   Array of input batch buffers.
     * @param[in] outputs  List of names of required output tensors.
     * @param[in] reqId    ID of this request.
     * @param[in] clasList Triton classification parameters, if any.
     * @return NVDSINFER_SUCCESS or NVDSINFER_TRITON_ERROR.
     */
    NvDsInferStatus init(
        const std::string& model, int64_t version, SharedBatchArray& inputs,
        const std::vector<std::string>& outputs, uint64_t reqId,
        const std::vector<TritonClassParams>& clasList);

    /**
     * @brief Set the release callback function for the request.
     * @param[in] requestCompleteCb The request release callback function.
     * @param[in] userPtr           The user data pointer for the callback.
     * @return NVDSINFER_SUCCESS or NVDSINFER_TRITON_ERROR.
     */
    NvDsInferStatus setRequestComplete(
        TRITONSERVER_InferenceRequestReleaseFn_t requestCompleteCb,
        void* userPtr);

    /**
     * @brief Set the allocator and response callback for the request.
     * @param[in] allocator Pointer to the output allocator instance.
     * @param[in] responseCompleteCb The response callback function.
     * @param[in] responseUserPtr    The user data pointer.
     * @return NVDSINFER_SUCCESS or NVDSINFER_TRITON_ERROR.
     */
    NvDsInferStatus setResponseComplete(
        ShrTritonAllocator& allocator,
        TRITONSERVER_InferenceResponseCompleteFn_t responseCompleteCb,
        void* responseUserPtr);

    /**
     * @brief The callback function to release the request instance.
     * @param[in] request Pointer to the request.
     * @param[in] flags   Flags associated with the callback.
     * @param[in] userp   User data pointer.
     */
    static void RequestOnRelease(
        TRITONSERVER_InferenceRequest* request, const uint32_t flags,
        void* userp);

public:
    /**
     * @brief Destructor. Releases the Triton inference request instance.
     */
    ~TrtServerRequest();

    /**
     * @brief Get the pointer to the Triton inference request object.
     */
    TRITONSERVER_InferenceRequest* ptr() { return m_ReqPtr.get(); }
    /**
     * @brief Get the model name.
     */
    const std::string& model() const { return m_Model; }
    /**
     * @brief Get the request ID.
     */
    uint64_t id() const { return m_ReqId; }
    /**
     * @brief Get the input buffer ID associated with the request.
     */
    uint64_t bufId() const { return m_BufId; }
    /**
     * @brief Release the ownership of input batch buffer array.
     * @return Shared pointer to the input array.
     */
    SharedBatchArray releaseInputs() { return std::move(m_Inputs); }
    /**
     * @brief Get the list of requested output layer names.
     */
    const std::vector<std::string>& outputs() const { return m_Outputs; }
    /**
     * @brief Get the Triton classification parameters list
     * (tensor name : classification parameters).
     */
    const std::map<std::string, TritonClassParams>& classParams() const {
        return m_ClasList; }

private:
    /**
     * @brief Add the inputs to the Triton inference request and assign buffer
     * data.
     * @param inputs Array of input batch buffers.
     * @return NVDSINFER_SUCCESS or NVDSINFER_TRITON_ERROR.
     */
    NvDsInferStatus setInputs(SharedBatchArray& inputs);
    /**
     * @brief Set the Triton options for the inference request.
     * @return NVDSINFER_SUCCESS or NVDSINFER_TRITON_ERROR.
     */
    NvDsInferStatus setOption(const IOptions* option);

    DISABLE_CLASS_COPY(TrtServerRequest);

private:
    /**
     * @brief Pointer to Triton inference request instance.
     */
    UniqTritonT<TRITONSERVER_InferenceRequest> m_ReqPtr;
    /**
     * @brief Pointer to the Triton server instance.
     */
    std::weak_ptr<TrtISServer> m_Server;
    /**
     * @brief Array of batch inputs for the request.
     */
    SharedBatchArray m_Inputs;
    /**
     * @brief Model name.
     */
    std::string m_Model;
    /**
     * @brief Request ID.
     */
    uint64_t m_ReqId = UINT64_C(0);
    /**
     * @brief Input buffer ID.
     */
    uint64_t m_BufId = UINT64_C(0);

    /**
     * @brief List of output layer names expected in response.
     */
    std::vector<std::string> m_Outputs;
    /**
     * @brief Triton classification parameters map (tensor name : parameters).
     */
    std::map<std::string, TritonClassParams> m_ClasList;
};

/**
 * @brief Wrapper class for Triton output parsing.
 */
class TrtServerResponse {
    friend class TrtISServer;

protected:
    /**
     * @brief Constructor.
     * @param server Handle to the server class instance.
     * @param data   Pointer to the inference result.
     * @param id     The corresponding request ID.
     */
    TrtServerResponse(
        TrtServerPtr server, UniqTritonT<TRITONSERVER_InferenceResponse> data,
        uint64_t id);

public:
    /**
     * @brief Check for error and parse the inference output.
     *
     * This functions checks the error status of the response and gets the
     * model name and version from the response. It then call the parseParams()
     * and parseOutputData() functions.
     *
     * @param req Pointer to the corresponding Triton request.
     * @return NVDSINFER_SUCCESS or NVDSINFER_TRITON_ERROR.
     */
    NvDsInferStatus parse(const TrtServerRequest* req);
    /**
     * @brief Get the model name parsed from the Triton response.
     */
    const std::string& model() const { return m_Model; }
    /**
     * @brief Get the list of output batch buffers.
     */
    std::vector<SharedBatchBuf>& mutableOutputs() { return m_BufOutputs; }
    /**
     * @brief Check if the response could be parsed correctly.
     */
    NvDsInferStatus getStatus() const { return m_Status; }
    /**
     * @brief Get and own the options list.
     */
    SharedOptions takeoverOptions() { return std::move(m_Options); }

private:
    /**
     * @brief Parse the inference parameters in the response.
     */
    NvDsInferStatus parseParams();

    /**
     * @brief Parse the inference output from the response.
     *
     * This function fetches the output tensors information the Triton server.
     * If the Triton Classification post processing option is set, the
     * addClass() function is called for corresponding tensors. For rest of
     * the outputs a batch buffer is formed and added to the output buffers
     * list.
     *
     * @param req Pointer to the corresponding Triton request.
     * @return NVDSINFER_SUCCESS or NVDSINFER_TRITON_ERROR.
     */
    NvDsInferStatus parseOutputData(const TrtServerRequest* req);

    /**
     * @brief Perform TopK classification for each output in the batch and add
     * the classification output buffer to the output buffer list.
     * @param[in] classP    The classification parameters for the output tensor.
     * @param[in] desc      Buffer descriptor formed from the inference response.
     * @param[in] batchSize Batch size of the output.
     * @param[in] tensorIdx Index of the output tensor in the inference response.
     * @param[in] base      Pointer to the tensor buffer.
     * @return NVDSINFER_SUCCESS or NVDSINFER_TRITON_ERROR.
     */
    NvDsInferStatus addClass(
        const TritonClassParams& classP, const InferBufferDescription& desc, uint32_t batchSize,
        uint32_t tensorIdx, const void* base);

    /**
     * @brief Parse the Triton Classification output tensor to get top K results
     * and save the output in the output buffer.
     * @param[out] ret       Reference to the classification output buffer.
     * @param[in]  classP    The classification parameters for the output tensor.
     * @param[in]  desc      Buffer descriptor formed from the inference response.
     * @param[in]  tensorIdx Index of the output tensor in the inference response.
     * @param[in]  base      Pointer to the tensor buffer.
     * @return NVDSINFER_SUCCESS,  NVDSINFER_TRITON_ERROR, NVDSINFER_INVALID_PARAMS.
     */
    NvDsInferStatus topKClass(
        InferClassificationOutput& ret, const TritonClassParams& classP,
        const InferBufferDescription& desc, uint32_t tensorIdx, const void* base);

    DISABLE_CLASS_COPY(TrtServerResponse);

    /**
     * @brief The response ID returned by the inference response.
     */
    uint64_t m_ResponseId = UINT64_C(0);
    /**
     * @brief Pointer to the raw Triton inference response.
     */
    ShrTritonT<TRITONSERVER_InferenceResponse> m_Data;
    /**
     * @brief Handle to the Triton server class.
     */
    std::weak_ptr<TrtISServer> m_Server;
    /**
     * @brief Model name from the inference response.
     */
    std::string m_Model;
    /**
     * @brief Model version from the inference response.
     */
    int64_t m_ModelVersion = UINT64_C(1);
    /**
     * @brief Error status of the inference response.
     */
    NvDsInferStatus m_Status = NVDSINFER_SUCCESS;

    /**
     * @brief List of output batch buffers. Mix of SharedRefBatchBuf
     * and SharedClassOutput.
     */
    std::vector<SharedBatchBuf> m_BufOutputs;
    /**
     * @brief Output parameters for the inference response.
     */
    SharedOptions m_Options;
};

/**
 * @brief Wrapper class for Triton server output memory allocator.
 */
class TrtServerAllocator
    : public std::enable_shared_from_this<TrtServerAllocator> {
public:
    using AllocFn = std::function<SharedSysMem(const std::string&, size_t, InferMemType, int64_t)>;
    using FreeFn = std::function<void(const std::string&, SharedSysMem)>;

    /**
     * @brief Constructor, create an instance of the
     * type TRITONSERVER_ResponseAllocator which calls provided allocator
     * and release functions.
     * @param alloc   Allocation function for the output tensors.
     * @param release Release function for the output tensors.
     */
    TrtServerAllocator(AllocFn alloc, FreeFn release);

    /**
     * @brief Destructor. Default.
     */
    virtual ~TrtServerAllocator() = default;

    /**
     * @brief Get the pointer to the TRITONSERVER_ResponseAllocator instance.
     */
    TRITONSERVER_ResponseAllocator* ptr() { return m_Allocator.get(); }

private:
    /**
     * @brief The allocator function registered with the Triton server.
     * @param[in] allocator         Handle of the allocator instance.
     * @param[in] tensorName        Name of the output tensor.
     * @param[in] bytes             Size of the requested allocation.
     * @param[in] preferredMemType  Preferred memory type for the allocation.
     * @param[in] preferredDevId    Preferred device ID.
     * @param[in] userP             User data pointer.
     * @param[out] buffer           Pointer to the allocated memory.
     * @param[out] bufferUserP      User pointer to be provided with release
     *                              function.
     * @param[out] actualMemType    Actual memory where the buffer is allocated.
     * @param[out] actualMemTypeId  Actual device ID.
     * @return Triton server error object if error, otherwise null.
     */
    static TRITONSERVER_Error* ResponseAlloc(
        TRITONSERVER_ResponseAllocator* allocator, const char* tensorName,
        size_t bytes, TRITONSERVER_MemoryType preferredMemType,
        int64_t preferredDevId, void* userP, void** buffer, void** bufferUserP,
        TRITONSERVER_MemoryType* actualMemType, int64_t* actualMemTypeId);

    /**
     * @brief The release function registered with Triton server.
     * @param[in] allocator   Handle of the allocator instance.
     * @param[in] buffer      Pointer to the buffer to be released.
     * @param[in] bufferUserP User pointer registered with the buffer.
     * @param[in] bytes       Size of the buffer.
     * @param[in] memType     Memory type of the buffer.
     * @param[in] devId       Device ID of the allocation.
     * @return Triton server error object if error, otherwise null.
     */
    static TRITONSERVER_Error* ResponseRelease(
        TRITONSERVER_ResponseAllocator* allocator, void* buffer,
        void* bufferUserP, size_t bytes, TRITONSERVER_MemoryType memType,
        int64_t devId);

private:
    DISABLE_CLASS_COPY(TrtServerAllocator);

    /**
     * @brief The Triton allocator instance.
     */
    UniqTritonT<TRITONSERVER_ResponseAllocator> m_Allocator;
    /**
     * @brief Function to allocate the buffers.
     */
    AllocFn m_allocFn;
    /**
     * @brief Function to release the buffers.
     */
    FreeFn m_releaseFn;
};

namespace triton {

/**
 * @brief The backend configuration settings.
 */
struct BackendConfig {
    /**
     * @brief Name of the backend.
     */
    std::string backend;
    /**
     * @brief Name of the setting.
     */
    std::string key;
    /**
     * @brief Value of the setting.
     */
    std::string value;
};

/**
 * @brief Model repository settings for the Triton Inference Server.
 */
struct RepoSettings {
    /**
     * @brief Set of model repository directories.
     */
    std::set<std::string> roots;
    /**
     * @brief Level of the Triton log output.
     */
    uint32_t logLevel = 0;
    /**
     * @brief Flag to enable/disable soft placement of TF operators.
     */
    bool tfAllowSoftPlacement = true;
    /**
     * @brief TensorFlow GPU memory fraction per process.
     */
    float tfGpuMemoryFraction = 0;
    /**
     * @brief Flag to enable/disable Triton strict model configuration.
     */
    bool strictModelConfig = true;
    /**
     * @brief The minimun supported compute compability for Triton server.
     */
    double minComputeCapacity = TRITON_DEFAULT_MINIMUM_COMPUTE_CAPABILITY;
    /**
     * @brief Pre-allocated pinned memory on host for Triton runtime.
     */
    uint64_t pinnedMemBytes = TRITON_DEFAULT_PINNED_MEMORY_BYTES;
    /**
     * @brief The path to the Triton backends directory.
     */
    std::string backendDirectory{TRITON_DEFAULT_BACKEND_DIR};
    /**
     * @brief Triton model control mode.
     */
    int32_t controlMode = (int32_t)TRITONSERVER_MODEL_CONTROL_EXPLICIT;
    /**
     * @brief Map of the device IDs and corresponding size of CUDA memory pool
     * to be allocated.
     */
    std::map<uint32_t, uint64_t> cudaDevMemMap;
    /**
     * @brief Array of backend configurations settings.
     */
    std::vector<BackendConfig> backendConfigs;

    /**
     * @brief Debug string of the TritonModelRepo protobuf message.
     */
    std::string debugStr;

    /**
     * @brief Populate the RepoSettings instance with the values from
     * the TritonModelRepo protobuf message.
     * @param[in] repo The model repository configuration proto message.
     * @param[in] devIds Not used.
     * @return Success or failure status.
     */
    bool initFrom(
        const ic::TritonModelRepo& repo, const std::vector<int>& devIds);

    /**
     * @brief Comparison operators. Check that the two repository settings are
     * same/different. Different control modes are reported as warning.
     * CudaDeviceMem is not checked.
     */
    /**@{*/
    bool operator==(const RepoSettings& other) const;
    bool operator!=(const RepoSettings& other) const
    {
        return !this->operator==(other);
    }
    /**@}*/
};
}  // namespace triton

/**
 * @brief Wrapper class for creating Triton Inference Server instance.
 */
class TrtISServer : public std::enable_shared_from_this<TrtISServer> {
    friend class TrtServerRequest;
    friend class TrtServerResponse;

protected:
    /**
     * @brief Constructor. Save the model repository configuration settings.
     * @param repo
     */
    TrtISServer(const triton::RepoSettings& repo);

    /**
     * @brief Create a new instance of the Triton Inference Server.
     *
     * This functions creates a TRITONSERVER_ServerOptions instance and updates
     * it as per the repository configuration. These options are then used to
     * create a new Triton instance and save the handle in m_Impl.
     *
     * @return Error status.
     */
    NvDsInferStatus initialize();

    /**
     * @brief Get the model repository settings.
     */
    const triton::RepoSettings& getRepoSettings() { return m_RepoSettings; }

public:
    /**
     * @brief Destructor. Stops the Triton server if the server handle is valid.
     */
    ~TrtISServer();

    /**
     * @brief Get a new or existing instance of the Triton Inference Server.
     *
     * This function checks if an instance of Triton is present. If it doesn't
     * exist, a new one is instantiated using the provided model repository
     * configuration. If a Triton instance is already running and repository
     * configuration is provided, the existing and new configuration is
     * checked to be same otherwise a null pointer is returned.
     *
     * @param[in] repo Model repository configuration, can be null when the
     *                 Triton server is already instantiated.
     *
     * @return Pointer to the Triton server instance.
     */
    static TrtServerPtr getInstance(const triton::RepoSettings* repo);

    /**
     * @brief Check if the server is ready.
     */
    bool isServerReady();

    /**
     * @brief Check if the server is live.
     */
    bool isServerLive();

    /**
     * @brief Check if the server is ready for inference using specified model.
     * @param[in] model  Name of the model.
     * @param[in] version Version of the model.
     * @return Boolean indicating readiness.
     */
    bool isModelReady(const std::string& model, int64_t version);

    /**
     * @brief Load or reload the specified model.
     * @param[in] modelName Name of the model
     * @return NVDSINFER_SUCCESS or NVDSINFER_TRITON_ERROR.
     */
    NvDsInferStatus loadModel(const std::string& modelName);

    /**
     * @brief Unload the specified model.
     * @param[in] modelName Name of the model.
     * @return NVDSINFER_SUCCESS or NVDSINFER_TRITON_ERROR.
     */
    NvDsInferStatus unloadModel(const std::string& modelName);

    /**
     * @brief Get the model configuration for the specified model.
     * @param[in] model   Name of the model.
     * @param[in] version Version of the model.
     * @param[out] config ModelConfig protobuf message to be populated.
     * @return NVDSINFER_SUCCESS or NVDSINFER_TRITON_ERROR.
     */
    NvDsInferStatus getModelConfig(
        const std::string& model, int64_t version, ni::ModelConfig& config);

    /**
     * @brief Create and initializes a new inference request.
     * @param[in] model    Name of the model.
     * @param[in] version  Version of the model.
     * @param[in] inputs   Array of input batch buffers.
     * @param[in] outputs  List of requested output names.
     * @param[in] clasList Triton Classification parameters, if any.
     * @return Pointer to the request instance upon success,
     *         null pointer on failure.
     */
    SharedRequest createRequest(
        const std::string& model, int64_t version, SharedBatchArray& inputs,
        const std::vector<std::string>& outputs, const std::vector<TritonClassParams>& clasList);

    /**
     * @brief Submit a request for asynchronous inference.
     *
     * This functions sets the release and response callback functions
     * for the request and then triggers the asynchronous inference.
     *
     * @param[in] request   Pointer to the request object.
     * @param[in] allocator Pointer to the response allocator.
     * @param[in] done      Pointer to the function to be called after
     *                      inference is done.
     * @return NVDSINFER_SUCCESS or NVDSINFER_TRITON_ERROR.
     */
    NvDsInferStatus inferAsync(
        SharedRequest request, WeakTritonAllocator allocator,
        TritonInferAsyncDone done);

private:
    /**
     * @brief Create a inference response instance.
     *
     * This function takes in the response provided by the Triton server
     * and creates a  TrtServerResponse instance from it.
     *
     * @param[in] data The inference response from Triton.
     * @param[in] id The request ID corresponding to the response.
     * @return Pointer to the TrtServerResponse created from input.
     */
    UniqResponse createResponse(
        UniqTritonT<TRITONSERVER_InferenceResponse>&& data, uint64_t id);

    using InferUserData =
        std::tuple<SharedRequest, TritonInferAsyncDone, TrtISServer*>;
    /**
     * @brief Response callback function, pass the response to
     * the provided inference done routine.
     *
     * @param[in] response The response from Triton server.
     * @param[in] flags    Flags returned by the server along with the response.
     * @param[in] userp    The user data pointer passed to set callback in
     *                     inferAsync().
     */
    static void InferComplete(
        TRITONSERVER_InferenceResponse* response, const uint32_t flags,
        void* userp);

    /**
     * @brief Returns the handle to the underlying Triton Inference Server.
     */
    TRITONSERVER_Server* serverPtr() const { return m_Impl.get(); }

    DISABLE_CLASS_COPY(TrtISServer);

    /**
     * @brief Static pointer to the server instance.
     */
    static std::weak_ptr<TrtISServer> sTrtServerInstance;
    /**
     * @brief Mutex to guard access to the server instance.
     */
    static std::mutex sTrtServerMutex;

private:
    /**
     * @brief Handle of the Triton Inference Server.
     */
    UniqTritonT<TRITONSERVER_Server> m_Impl;
    /**
     * @brief Counter for the number of requests created.
     */
    std::atomic<uint64_t> m_LastRequestId{UINT64_C(0)};
    /**
     * @brief Model repository settings for the server.
     */
    triton::RepoSettings m_RepoSettings;
};

} // namespace nvdsinferserver

#endif
