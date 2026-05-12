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
 * @file infer_trtis_context.h
 *
 * @brief Header file for the Triton C-API mode inference context class.
 *
 * This file declares the inference context class that uses the C_API
 * mode of the Triton Inference Server.
 */

#ifndef __INFER_TRTIS_CONTEXT_H__
#define __INFER_TRTIS_CONTEXT_H__

#include "infer_common.h"
#include "infer_cuda_context.h"

namespace nvdsinferserver {

/**
 * @brief Inference context accessing Triton Inference Server in C-API mode.
 */
class InferTrtISContext : public InferCudaContext {
public:
    /**
     * @brief Constructor, default.
     */
    InferTrtISContext();

    /**
     * @brief Destructor, default.
     */
    ~InferTrtISContext() override;

    /**
     * @brief Synchronize on the CUDA stream and call InferCudaContext::deinit().
     */
    NvDsInferStatus deinit() override;

    /**
     * @brief Get the main processing CUDA event
     */
    SharedCuStream& mainStream() override
    {
        assert(m_MainStream);
        return m_MainStream;
    }

private:
    /**
     * @brief Create the Triton C-API mode inference processing backend.
     * @param[in]  params       The backend configuration parameters.
     * @param[in]  maxBatchSize The maximum batch size configuration.
     * @param[out] backend      Pointer for the backend handle.
     * @return Status code.
     */
    NvDsInferStatus createNNBackend(const ic::BackendParams& params,
        int maxBatchSize, UniqBackend& backend);

    /**
     * @brief Allocate resources.
     *
     * This function creates the CUDA stream object for the inference context
     * and calls InferCudaContext::allocateResource().
     *
     * @param[in] config The inference configuration setting protobuf message.
     * @return Status code.
     */
    NvDsInferStatus allocateResource(
        const ic::InferenceConfig& config) override;

    /**
     * @brief Finalize the input output tensor dimensions considering maximum
     * batch size.
     * @param[inout] be     Handle to the backend instance.
     * @param[in]    model  Name of the model.
     * @param[in]    params The backend configuration settings.
     * @return Status code.
     */
    NvDsInferStatus specifyBackendDims(
        BaseBackend* be, const std::string& model, const ic::BackendParams& params);

    /**
     * @brief Create maps of input layer names, dimensions and get the
     * list of output layer names from the configuration setting.
     * @param[in]  params  The backend configuration parameter protobuf message.
     * @param[out] inputs  The map of input layer dimensions, name.
     * @param[out] outputs The set of output layer names.
     * @return Status code.
     */
    NvDsInferStatus getConfigInOutMap(
        const ic::BackendParams& params,
        std::unordered_map<std::string, InferDims>& inputs,
        std::set<std::string>& outputs);

private:
    /**
     * @brief Pointer to CUDA stream instance of the inference context.
     */
    SharedCuStream m_MainStream;
    /**
     * @brief Handle of the Triton inference backend.
     */
    TrtISBackend* m_Backend{nullptr};
};

} // namespace nvdsinferserver

#endif
