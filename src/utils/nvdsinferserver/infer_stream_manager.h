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

#ifndef __INFER_STREAM_MANAGER_H__
#define __INFER_STREAM_MANAGER_H__

#include <functional>
#include <queue>

#include "infer_common.h"
#include "infer_utils.h"

namespace nvdsinferserver {

// When loop is enabled, for full frame processing only
//
class StreamManager {
public:
    using StreamId = uint64_t;

    StreamManager() {}
    ~StreamManager() = default;
    // wait until the streaming update from running(kRunning) to other state
    NvDsInferStatus waitStream(StreamId id);
    // start a stream into waiting list, update status into kRunning
    NvDsInferStatus startStream(StreamId id, int64_t timestamp, void* userptr);
    // stop a stream, set it to kStopped but keep in the list
    NvDsInferStatus stopStream(StreamId id);
    // update stream state into kReady
    NvDsInferStatus streamInferDone(StreamId id, SharedBatchArray& outTensors);
    void notifyError(NvDsInferStatus status);

    // Information
    // max_size; timeout_list;
private:
    enum class ProgressType : int {
        kStopped = 0,
        kReady = 1,
        kRunning = 2,
    };

    struct StreamState {
        int64_t timestamp = 0;
        ProgressType progress = ProgressType::kReady;
        void* reserved = nullptr;
    };
    using StreamList = std::unordered_map<StreamId, StreamState>;

    bool isRunning() const { return !m_Stopping; }
    bool popDeprecatedStream();

    StreamList m_StreamList;
    uint32_t m_MaxStreamSize = 256;
    bool m_Stopping = false;
    std::mutex m_Mutex;
    std::condition_variable m_Cond;
};

}  // namespace nvdsinferserver

#endif
