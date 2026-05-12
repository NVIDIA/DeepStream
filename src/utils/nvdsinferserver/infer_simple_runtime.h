/*
 * SPDX-FileCopyrightText: Copyright (c) 2020 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
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
