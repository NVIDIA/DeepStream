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

#ifndef __INFER_LSTM_CONTROL_H__
#define __INFER_LSTM_CONTROL_H__

#include "infer_base_context.h"
#include "infer_common.h"
#include "infer_datatypes.h"
#include "infer_proto_utils.h"
#include "infer_utils.h"

namespace nvdsinferserver {

class LstmController {
public:
    LstmController(const ic::LstmParams& params, int devId, int maxBatchSize)
    {
        m_Params.CopyFrom(params);
        m_DevId = devId;
        m_MaxBatchSize = maxBatchSize;
    }
    ~LstmController() = default;

    NvDsInferStatus initInputState(BaseBackend& backend);
    NvDsInferStatus feedbackInputs(SharedBatchArray& outTensors);
    NvDsInferStatus waitAndGetInputs(SharedBatchArray& inputs);
    void notifyError(NvDsInferStatus status);
    void destroy()
    {
        UniqLock locker(m_Mutex);
        m_InProgress = 0;
        m_Cond.notify_all();
        locker.unlock();
        m_LoopStateMap.clear();
        m_LstmInputs.clear();
    }

private:
    // check input/output tensor names/dims/datatype must be same
    NvDsInferStatus checkTensorInfo(BaseBackend& backend);
    struct LoopState {
        std::string inputName;
        SharedCudaTensorBuf inputTensor;
        SharedBatchBuf outputTensor;
        bool keepOutputParsing = false;
    };

private:
    ic::LstmParams m_Params;
    int m_DevId = 0;
    int m_MaxBatchSize = 1;
    // map<outputName, loopState>
    std::unordered_map<std::string, LoopState> m_LoopStateMap;
    std::vector<SharedCudaTensorBuf> m_LstmInputs;
    std::atomic<int32_t> m_InProgress{0};
    std::mutex m_Mutex;
    std::condition_variable m_Cond;
    SharedCuEvent m_InputReadyEvent;
    SharedCuStream m_LstmStream;
};

}  // namespace nvdsinferserver

#endif
