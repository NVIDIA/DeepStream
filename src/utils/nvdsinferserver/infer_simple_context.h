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

#ifndef __NVDSINFERSERVER_SIMPLE_CONTEXT_H__
#define __NVDSINFERSERVER_SIMPLE_CONTEXT_H__

#include "infer_base_backend.h"
#include "infer_base_context.h"
#include "infer_cuda_utils.h"
#include "infer_datatypes.h"
#include "infer_utils.h"

namespace nvdsinferserver {


class InferSimpleContext : public InferBaseContext {
public:
    InferSimpleContext();
    ~InferSimpleContext() override;

protected:
    NvDsInferStatus createNNBackend(
        const ic::BackendParams& params, int maxBatchSize,
        UniqBackend& backend);
    NvDsInferStatus fixateInferenceInfo(
        const ic::InferenceConfig& config, BaseBackend& backend) override;
    NvDsInferStatus deinit() override;

private:
    // should not called
    NvDsInferStatus createPreprocessor(
        const ic::PreProcessParams& params,
        std::vector<UniqPreprocessor>& processors) override;
    NvDsInferStatus createPostprocessor(
        const ic::PostProcessParams& params,
        UniqPostprocessor& processor) override;
    NvDsInferStatus allocateResource(
        const ic::InferenceConfig& config) override;

    void getNetworkInputInfo(NvDsInferNetworkInfo& networkInfo) override
    {
        networkInfo = m_NetworkImageInfo;
    }
    void notifyError(NvDsInferStatus status) override {}
    SharedCuStream& mainStream() override { return m_Stream; }

private:
    // Optional, not needed
    NvDsInferNetworkInfo m_NetworkImageInfo{0, 0, 0};

    SharedCuStream m_Stream{nullptr};
};

}  // namespace nvdsinferserver

#endif  //__NVDSINFERSERVER_SIMPLE_CONTEXT_H__