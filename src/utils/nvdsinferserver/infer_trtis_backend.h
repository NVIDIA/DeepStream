/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @file infer_trtis_backend.h
 *
 * @brief Header file of Triton Inference Server inference backend.
 *
 * This file declares the inference backend for the Triton Inference Server.
 */

#ifndef __NVDSINFER_TRTIS_BACKEND_H__
#define __NVDSINFER_TRTIS_BACKEND_H__

#include "infer_base_backend.h"
#include "infer_common.h"
#include "infer_post_datatypes.h"
#include "infer_utils.h"

namespace nvdsinferserver {

class TrtServerAllocator;
class TrtISServer;
class TrtServerRequest;
class TrtServerResponse;

/**
 * @brief Triton backend processing class.
 */
class TrtISBackend : public BaseBackend {
public:
    /**
     * @brief Constructor. Save the model name, version and server handle.
     * @param[in] name    Model name.
     * @param[in] version Model version.
     * @param[in] ptr     Handle to Triton server class instance.
     */
    TrtISBackend(
        const std::string& name, int64_t version, TrtServerPtr ptr = nullptr);

    /**
     * @brief Destructor. Unload the model if needed.
     */
    ~TrtISBackend() override;

    /**
     * @brief Add Triton Classification parameters to the list.
     */
    void addClassifyParams(const TritonClassParams& c) {
        m_ClassifyParams.emplace_back(c); }

    /**
     * @brief Helper function to access the member variables.
     */
    /**@{*/
    void setOutputPoolSize(int size) { m_PerPoolSize = size; }
    int outputPoolSize() const { return m_PerPoolSize; }
    void setOutputMemType(InferMemType memType) { m_OutputMemType = memType; }
    InferMemType outputMemType() const { return m_OutputMemType; }
    void setOutputDevId(int64_t devId) { m_OutputDevId = devId; }
    int64_t outputDevId() const { return m_OutputDevId; }
    std::vector<TritonClassParams> getClassifyParams() {
        return m_ClassifyParams;}
    const std::string& model() const { return m_Model; }
    int64_t version() const { return m_ModelVersion; }
    /**@}*/

    /**
     * @brief Check that the server and model is ready, get the information
     * of layers, setup reorder thread and output tensor allocator.
     * @return NVDSINFER_SUCCESS or NVDSINFER_TRITON_ERROR.
     */
    NvDsInferStatus initialize() override;

    /**
     * @brief Specify the input layers for the backend.
     * @param shapes List of name and shapes of the input layers.
     * @return Status code of the type NvDsInferStatus.
     */
    NvDsInferStatus specifyInputDims(const InputShapes& shapes) override;

    /**
     * @brief Enqueue an input for inference request by calling Run() and
     * adding corresponding task to the reorder thread queue.
     * @param[in] inputs        The array of input batch buffers.
     * @param[in] stream        The CUDA stream to be used.
     * @param[in] bufConsumed   Callback function for releasing input buffer.
     * @param[in] inferenceDone Callback function for processing result.
     * @return
     */
    NvDsInferStatus enqueue(SharedBatchArray inputs, SharedCuStream stream,
        InputsConsumed bufConsumed, InferenceDone inferenceDone) override;

    /**
     * @brief Set the maximum size for the tensor, the maximum of the existing
     * size and new input size is used. The size is rounded up to
     * INFER_MEM_ALIGNMENT bytes.
     * @param name     Name of the tensor.
     * @param maxBytes New maximum number of bytes for the buffer.
     */
    void setTensorMaxBytes(const std::string& name, size_t maxBytes)
    {
        size_t& bytes = m_TensorMaxBytes[name];
        bytes = std::max<size_t>(maxBytes, bytes);
        bytes = INFER_ROUND_UP(bytes, INFER_MEM_ALIGNMENT);
    }

protected:
    // interface for derived class

    /**
     * @brief Get the list of output tensor names.
     * @param[out] outNames The set of strings to which the names are added.
     */
    virtual void requestTritonOutputNames(std::set<std::string>& outNames);

    /**
     * @brief Check that the Triton inference server is live.
     * @return NVDSINFER_SUCCESS or NVDSINFER_TRITON_ERROR.
     */
    virtual NvDsInferStatus ensureServerReady();

    /**
     * @brief Check that the model is ready, load the model if it is not.
     * @return NVDSINFER_SUCCESS or NVDSINFER_TRITON_ERROR.
     */
    virtual NvDsInferStatus ensureModelReady();

    /**
     * @brief Create a loop thread that calls inferenceDoneReorderLoop
     * on the queued items.
     * @return NVDSINFER_SUCCESS or NVDSINFER_TRITON_ERROR.
     */
    NvDsInferStatus setupReorderThread();

    /**
     * @brief Set the output tensor allocator.
     */
    void setAllocator(UniqTritonAllocator allocator)
    {
        m_ResponseAllocator = std::move(allocator);
    }

    /**
     * @brief Get the model configuration from the server and populate
     * layer information. Set maximum batch size as specified in configuration
     * settings.
     * @return NVDSINFER_SUCCESS or NVDSINFER_TRITON_ERROR.
     */
    virtual NvDsInferStatus setupLayersInfo();

    /**
     * @brief Get the Triton server handle.
     */
    TrtServerPtr& server() { return m_Server; }

    /**
     * @brief Asynchronous inference done function: AsyncDone(Status, outputs).
     */
    using AsyncDone = std::function<void(NvDsInferStatus, SharedBatchArray)>;
    /**
     * @brief Create an inference request and trigger asynchronous inference.
     * serverInferCompleted() is set as callback function that in turn calls
     * asyncDone.
     *
     * @param[in] inputs      Array of input batch buffers.
     * @param[in] bufConsumed Callback function for releasing input buffer.
     * @param[in] asyncDone   Callback function for processing response .
     * @return
     */
    virtual NvDsInferStatus Run(
        SharedBatchArray inputs, InputsConsumed bufConsumed,
        AsyncDone asyncDone);

    /**
     * @brief Extend the dimensions to include batch size for the buffers in
     * input array. Do nothing if batch input is not required.
     */
    NvDsInferStatus fixateDims(const SharedBatchArray& bufs);

    /**
     * @brief Acquire a buffer from the output buffer pool associated with the
     * device ID and memory type. Create the pool if it doesn't exist.
     * @param[in] tensor   Name of the output tensor.
     * @param[in] bytes    Buffer size.
     * @param[in] memType  Requested memory type.
     * @param[in] devId    Device ID for the allocation.
     * @return Pointer to the allocated buffer.
     */
    SharedSysMem allocateResponseBuf(
        const std::string& tensor, size_t bytes, InferMemType memType, int64_t devId);

    /**
     * @brief Release the output tensor buffer.
     * @param[in] tensor Name of the output tensor.
     * @param[in] mem    Pointer to the memory buffer.
     */
    void releaseResponseBuf(const std::string& tensor, SharedSysMem mem);

    /**
     * @brief Ensure that the array of input buffers are expected by the model
     * and reshape the input buffers if required.
     * @param inputs Array of input batch buffers.
     * @return NVDSINFER_SUCCESS or NVDSINFER_TRITON_ERROR.
     */
    NvDsInferStatus ensureInputs(SharedBatchArray& inputs);

    /**
     * Tuple keys as <tensor-name, gpu-id, memType>
     */
    enum { kName, kGpuId, kMemType };
    /**
     * @brief Tuple holding tensor name, GPU ID, memory type.
     */
    using PoolKey = std::tuple<std::string, int64_t, InferMemType>;
    /**
     * @brief The buffer pool for the specified tensor, GPU and
     * memory type combination.
     */
    using PoolValue = SharedBufPool<UniqSysMem>;

    /**
     * @brief Find the buffer pool for the given key.
     */
    PoolValue findResponsePool(PoolKey& key);

    /**
     * @brief Create a new buffer pool for the key.
     * @param[in] key   The pool key combination.
     * @param[in] bytes Size of the requested buffer.
     * @return
     */
    PoolValue createResponsePool(PoolKey& key, size_t bytes);

    /**
     * @brief Call the inputs consumed function and parse the inference
     * response to form the array of output batch buffers and call
     * asyncDone on it.
     * @param[in] request        Pointer to the inference request.
     * @param[in] uniqResponse   Pointer to the inference response from the
     *                           server.
     * @param[in] inputsConsumed Callback function for releasing input buffer.
     * @param[in] asyncDone      Callback function for processing response .
     */
    void serverInferCompleted(
        std::shared_ptr<TrtServerRequest> request,
        std::unique_ptr<TrtServerResponse> uniqResponse,
        InputsConsumed inputsConsumed, AsyncDone asyncDone);

    /**
     * @brief Reorder thread task.
     */
    struct ReorderItem {
        /**
         * @brief Status of processing
         */
        NvDsInferStatus status = NVDSINFER_SUCCESS;
        /**
         * @brief Array of input batch buffers. Order preserved and held until
         * output reordered.
         */
        SharedBatchArray inputs;
        /**
         * @brief Array of output batch buffers.
         */
        SharedBatchArray outputs;

        /**
         * @brief Synchronization objects.
         */
        /**@{*/
        std::promise<void> promise;
        std::future<void> future;
        /**@}*/

        /**
         * @brief Inference done callback function.
         */
        InferenceDone inferenceDone;
    };
    /*
     * @brief Pointer to the reorder thread task.
     */
    using ReorderItemPtr = std::shared_ptr<ReorderItem>;
    /**
     * @brief Add input buffers to the output buffer list if required. De-batch
     * and run inference done callback.
     * @param[in] item The reorder task.
     * @return Boolean indicating success or failure.
     */
    bool inferenceDoneReorderLoop(ReorderItemPtr item);

    /**
     * @brief Separate the batch dimension from the output buffer descriptors.
     * @param[in] outputs Array of output batch buffers.
     * @param[in] inputs  Array of input batch buffers.
     * @return Boolean indicating success or failure.
     */
    bool debatchingOutput(SharedBatchArray& outputs, SharedBatchArray& inputs);

private:
    /**
     * @brief Name of the model.
     */
    std::string m_Model;
    /**
     * @brief Version of the model.
     */
    int64_t m_ModelVersion = -1;
    /**
     * @brief Handle of the Triton server instance.
     */
    TrtServerPtr m_Server;
    /**
     * @brief Flag to indicate if the model should be unloaded when the
     * instance is being deleted.
     */
    bool m_NeedUnload = false;
    /**
     * @brief List of Triton Classification parameters.
     */
    std::vector<TritonClassParams> m_ClassifyParams;
    /**
     * @brief Pointer to the inference output buffer allocator.
     */
    ShrTritonAllocator m_ResponseAllocator;
    /**
     * @brief The preferred output memory type.
     */
    InferMemType m_OutputMemType = InferMemType::kNone;
    /**
     * @brief The device ID for output tensor memory allocation.
     */
    int64_t m_OutputDevId = -1;
    /**
     * @brief Number of buffers in the pool.
     */
    int m_PerPoolSize = 2;
    /**
     * @brief Map of pool keys and output tensor pools.
     */
    std::map<PoolKey, PoolValue> m_ResponsePool;
    /**
     * @brief A shared timed mutex.
     */
    using SharedMutex = std::shared_timed_mutex;
    /**
     * @brief Shared timed mutex to protect access to the output tensor pool.
     */
    SharedMutex m_ResponseMutex;
    /**
     * @brief Map of output tensor names and configured maximum size in bytes.
     */
    std::unordered_map<std::string, size_t> m_TensorMaxBytes;

    /**
     * @brief A queue thread for the reordering.
     */
    using ReorderThread = QueueThread<std::vector<ReorderItemPtr>>;
    /**
     * @brief Pointer to the reorder queue thread.
     */
    std::unique_ptr<ReorderThread> m_ReorderThread;
};

} // namespace nvdsinferserver

#endif
