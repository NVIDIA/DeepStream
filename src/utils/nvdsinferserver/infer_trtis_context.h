/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
