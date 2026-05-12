/*
 * SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
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
