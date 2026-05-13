/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
