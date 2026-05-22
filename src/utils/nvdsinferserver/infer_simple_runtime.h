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

#ifndef __NVDSINFERSERVER_SIMPLE_RUNTIME_H__
#define __NVDSINFERSERVER_SIMPLE_RUNTIME_H__

#include "infer_common.h"
#include "infer_trtis_backend.h"

namespace nvdsinferserver {

class TritonSimpleRuntime : public TrtISBackend {
public:
    TritonSimpleRuntime(std::string model, int64_t version);
    ~TritonSimpleRuntime() override;

    void setOutputs(const std::set<std::string>& names)
    {
        m_RequestOutputs = names;
    }

    // derived functions
    NvDsInferStatus initialize() override;

protected:
    NvDsInferStatus specifyInputDims(const InputShapes& shapes) override;
    NvDsInferStatus enqueue(SharedBatchArray inputs, SharedCuStream stream,
        InputsConsumed bufConsumed, InferenceDone inferenceDone) override;
    void requestTritonOutputNames(std::set<std::string>& names) override;

private:
    SharedSysMem allocateSimpleRes(
        const std::string& tensor, size_t bytes, InferMemType memType, int64_t devId);
    // Mark releaseCallback as static in case
    static void releaseSimpleRes(const std::string& tensor, SharedSysMem mem);

private:
    std::set<std::string> m_RequestOutputs;
};

}

#endif
