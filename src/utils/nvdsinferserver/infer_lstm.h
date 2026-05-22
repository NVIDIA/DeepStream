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
