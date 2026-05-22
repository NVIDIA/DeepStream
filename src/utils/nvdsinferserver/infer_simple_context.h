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