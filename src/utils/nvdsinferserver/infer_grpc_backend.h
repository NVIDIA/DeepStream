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
 * @file infer_grpc_backend.h
 *
 * @brief Header file of Triton Inference Server inference backend using gRPC.
 *
 * This file declares the inference backend for the Triton Inference Server
 * accessed using gRPC.
 */

#ifndef __NVDSINFERSERVER_GRPC_BACKEND_H__
#define __NVDSINFERSERVER_GRPC_BACKEND_H__

#include "infer_common.h"
#include "infer_trtis_backend.h"
#include "infer_grpc_client.h"

namespace nvdsinferserver {

/**
 * @brief Triton gRPC mode backend processing class.
 */
class TritonGrpcBackend : public TrtISBackend {
public:
    TritonGrpcBackend(std::string model, int64_t version);
    ~TritonGrpcBackend() override;

    void setOutputs(const std::set<std::string>& names) {
        m_RequestOutputs = names;
    }
    void setUrl (const std::string &url) { m_Url = url;}
    void setEnableCudaBufferSharing (const bool enableSharing) {
        m_EnableCudaBufferSharing = enableSharing;
    }
    NvDsInferStatus initialize() override;

protected:
    NvDsInferStatus enqueue(SharedBatchArray inputs, SharedCuStream stream,
        InputsConsumed bufConsumed, InferenceDone inferenceDone) override;
    void requestTritonOutputNames(std::set<std::string>& names) override;

    NvDsInferStatus ensureServerReady() override;
    NvDsInferStatus ensureModelReady() override;
    NvDsInferStatus setupLayersInfo() override;
    NvDsInferStatus Run(
        SharedBatchArray inputs, InputsConsumed bufConsumed,
        AsyncDone asyncDone) override;

private:
    std::string m_Url;
    std::set<std::string> m_RequestOutputs;
    std::shared_ptr<nvdsinferserver::InferGrpcClient> m_InferGrpcClient;
    bool m_EnableCudaBufferSharing = false;
};

}  // namespace nvdsinferserver

#endif
