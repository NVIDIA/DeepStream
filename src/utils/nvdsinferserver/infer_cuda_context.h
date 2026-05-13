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
 * @file infer_cuda_context.h
 *
 * @brief Header file for the CUDA inference context class.
 *
 * This file declares the CUDA processing class for handling the inference
 * context for the nvinferserver low level library.
 *
 */

#ifndef __INFER_CUDA_CONTEXT_H__
#define __INFER_CUDA_CONTEXT_H__

#include <shared_mutex>

#include "infer_base_context.h"
#include "infer_common.h"
#include "infer_datatypes.h"
#include "infer_utils.h"

namespace nvdsinferserver {

class CropSurfaceConverter;
class NetworkPreprocessor;
class Postprocessor;
class CudaEventInPool;

/**
 * @brief Class for inference context CUDA processing. Handles preprocessing,
 * post-processing, extra input processing and LSTM related processing.
 */
class InferCudaContext : public InferBaseContext {
public:
    /**
     * @brief Constructor. Instantiate the host tensor pool.
     */
    InferCudaContext();
    /**
     * @brief Destructor. Clear the extra input list and host tensor pool.
     */
    ~InferCudaContext() override;

    /**
     * @brief Allocator. Acquire a host buffer for the inference output.
     * @param[in] name  Name of the output layer.
     * @param[in] bytes Size of the buffer.
     * @return Pointer to the memory buffer.
     */
    SharedSysMem acquireTensorHostBuf(const std::string& name, size_t bytes);

    /**
     * @brief Acquire a CUDA event from the events pool.
     */
    SharedCuEvent acquireTensorHostEvent();

protected:
    /**
     * @brief Check the tensor order, media format, and datatype for the input
     * tensor. Initiate the extra processor and lstm controller if configured.
     * @param[in] config The inference configuration protobuf message.
     * @param[in] backend The inference backend instance.
     * @return Status code.
     */
    NvDsInferStatus fixateInferenceInfo(
        const ic::InferenceConfig& config, BaseBackend& backend) override;
    /**
     * @brief Create the surface converter and network preprocessor.
     * @param params      The preprocessor configuration.
     * @param processors  List of the created preprocessor handles.
     * @return Status code.
     */
    NvDsInferStatus createPreprocessor(const ic::PreProcessParams& params,
        std::vector<UniqPreprocessor>& processors) override;
    /**
     * @brief Create the post-processor as per the network output type.
     * @param[in] params     The post processing configuration parameters.
     * @param[out] processor The handle to the created post processor.
     * @return Status code.
     */
    NvDsInferStatus createPostprocessor(const ic::PostProcessParams& params,
        UniqPostprocessor& processor) override;
    /**
     * @brief Allocate resources for the preprocessors and post-processor.
     * Allocate the host tensor pool buffers.
     * @param[in]    config The inference configuration settings.
     * @return Status code.
     */
    NvDsInferStatus allocateResource(
        const ic::InferenceConfig& config) override;
    /**
     * @brief Initialize non-image input layers if the custom library has
     * implemented the interface.
     * @param[inout] inputs Array of the input batch buffers.
     * @param[in]    config The inference configuration settings.
     * @return Status code.
     */
    NvDsInferStatus preInference(
        SharedBatchArray& inputs, const ic::InferenceConfig& config) override;
    /**
     * @brief Post inference steps for the custom processor and LSTM controller.
     * @param[inout] outputs   The output batch buffers array.
     * @param[in]    inOptions The configuration options for the buffers.
     * @return Status code.
     */
    NvDsInferStatus extraOutputTensorCheck(
        SharedBatchArray& outputs, SharedOptions inOptions) override;
    /**
     * @brief In case of error, notify the waiting threads.
     */
    void notifyError(NvDsInferStatus status) override;
    /**
     * @brief Release the host tensor pool buffers, extra input buffers, LSTM
     * controller, extra input processor. De-initialize the context.
     * @return Status code.
     */
    NvDsInferStatus deinit() override;
    /**
     * @brief Get the network input layer information.
     */
    void getNetworkInputInfo(NvDsInferNetworkInfo &networkInfo) override {
        networkInfo = m_NetworkImageInfo;
    }
    /**
     * @brief Get the size of the tensor pool.
     */
    int tensorPoolSize() const;

private:
    /**
     * @brief Create and add the buffers for the host tensor pool for a layer.
     * @param[in] layer    The layer description.
     * @param[in] poolSize The number of buffers in the pool.
     * @param[in] gpuId    The GPU ID.
     * @return Status code.
     */
    NvDsInferStatus addHostTensorPool(const LayerDescription& layer,
        int poolSize, int gpuId);

    /**
     * @brief Update the input buffers for the LSTM input from the previous output.
     * @param[in] inputs Array of the input batch buffers.
     * @param[in] config The inference configuration settings.
     * @return Status code.
     */
    NvDsInferStatus ensureLstmInputReady(
        SharedBatchArray& inputs, const ic::InferenceConfig& config);

    /**
     * @brief Allocate and initialize CPU, GPU buffers for the additional input
     * layers using the custom initialization function. The data initialized in
     * the CPU buffers is copied into GPU buffers.
     * @param[in] inputs Array of the input batch buffers.
     * @param[in] config The inference configuration settings.
     * @return Status code.
     */
    NvDsInferStatus initFixedExtraInputLayers(
        SharedBatchArray& inputs, const ic::InferenceConfig& config);

    /**
     * @brief Check if the extra processor handle is valid.
     */
    bool hasExtraProcess() const { return m_ExtraProcessor.get(); }

    /**
     * @brief Create and initialize the custom processor for processing additional
     * inputs to the model apart from the primary input. Allocate the buffers for
     * the additional inputs.
     * @param[in] config        The inference configuration, specifies the extra
     *                          processing library.
     * @param[in] backend       The backend object to get model and batch dimensions.
     * @param[in] primaryTensor Name of the input tensor to be excluded from allocation.
     * @return Status code.
     */
    NvDsInferStatus loadExtraProcessor(
        const ic::InferenceConfig& config, BaseBackend& backend, const std::string& primaryTensor);

protected:

    /** Network input height, width, channels for preprocessing. */
    NvDsInferNetworkInfo m_NetworkImageInfo{0, 0, 0};
    /**
     * @brief The input layer media format.
     */
    InferMediaFormat m_NetworkImageFormat = InferMediaFormat::kRGB;
    /**
     * @brief The input layer name.
     */
    std::string m_NetworkImageName;
    /**
     * @brief The input layer tensor order.
     */
    InferTensorOrder m_InputTensorOrder = InferTensorOrder::kNone;
    /**
     * @brief The input layer datatype.
     */
    InferDataType m_InputDataType = InferDataType::kFp32;

    /**
     * @brief Array of buffers of the additional inputs.
     */
    std::vector<SharedCudaTensorBuf> m_ExtraInputs;
    /**
     * @brief Map of pools for the output tensors.
     */
    MapBufferPool<std::string, UniqSysMem> m_HostTensorPool;
    /**
     * @brief Pool of CUDA events for host tensor copy.
     */
    SharedBufPool<std::unique_ptr<CudaEventInPool>> m_HostTensorEvents;

    /** LSTM controller. */
    UniqLstmController m_LstmController;
    /** stream-id based management. */
    UniqStreamManager m_MultiStreamManager;
    /** Extra and custom processing pre/post inference. */
    UniqInferExtraProcessor m_ExtraProcessor;

    /** Preprocessor and post-processor handles */
    /**@{*/
    CropSurfaceConverter* m_SurfaceConverter = nullptr;
    NetworkPreprocessor* m_NetworkPreprocessor = nullptr;
    Postprocessor* m_FinalProcessor = nullptr;
    /**@}*/
};

} // namespace nvdsinferserver

#endif
