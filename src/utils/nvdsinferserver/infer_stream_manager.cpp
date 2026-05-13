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

#include "infer_stream_manager.h"

#include "infer_cuda_utils.h"

namespace nvdsinferserver {

NvDsInferStatus
StreamManager::waitStream(StreamId id)
{
    UniqLock locker(m_Mutex);
    m_Cond.wait(locker, [this, id]() {
        if (!isRunning()) {
            return true;
        }
        auto iter = m_StreamList.find(id);
        if (iter != m_StreamList.end() && iter->second.progress == ProgressType::kRunning) {
            return false;
        }
        return true;
    });
    if (!isRunning()) {
        InferDebug("waitstream returned due to all streams are stopping.");
    }
    return NVDSINFER_SUCCESS;
}

NvDsInferStatus
StreamManager::startStream(StreamId id, int64_t timestamp, void* userptr)
{
    UniqLock locker(m_Mutex);
    auto iter = m_StreamList.find(id);
    if (iter == m_StreamList.end()) {
        while (!m_StreamList.empty() && m_StreamList.size() >= m_MaxStreamSize) {
            if (!popDeprecatedStream()) {
                InferDebug(
                    "pop deprecated stream failed,all streams(size: %d) are running.",
                    (int32_t)m_StreamList.size());
                break;
            }
        }
        auto ret = m_StreamList.insert({id, StreamState{timestamp, ProgressType::kRunning}});
        assert(ret.second);
        iter = ret.first;
    }
    StreamState& state = iter->second;
    state.timestamp = timestamp;
    state.progress = ProgressType::kRunning;
    return NVDSINFER_SUCCESS;
}

NvDsInferStatus
StreamManager::stopStream(StreamId id)
{
    UniqLock locker(m_Mutex);
    auto iter = m_StreamList.find(id);
    if (iter != m_StreamList.end()) {
        StreamState& state = iter->second;
        state.progress = ProgressType::kStopped;
    } else {
        InferDebug("there is no inference stream: %d", (int32_t)id);
    }
    return NVDSINFER_SUCCESS;
}

bool
StreamManager::popDeprecatedStream()
{
    int64_t oldest = INT64_MAX;
    auto oldestIter = m_StreamList.end();
    for (auto i = m_StreamList.begin(); i != m_StreamList.end(); ++i) {
        if (i->second.timestamp < oldest && i->second.progress != ProgressType::kRunning) {
            oldest = i->second.timestamp;
            oldestIter = i;
        }
    }
    if (oldestIter == m_StreamList.end()) {
        return false;  // pop failed
    }
    m_StreamList.erase(oldestIter);
    m_Cond.notify_all();
    return true;
}

NvDsInferStatus
StreamManager::streamInferDone(StreamId id, SharedBatchArray& outTensors)
{
    UniqLock locker(m_Mutex);
    auto iter = m_StreamList.find(id);
    RETURN_IF_FAILED(
        iter != m_StreamList.end(), NVDSINFER_UNKNOWN_ERROR,
        "Internal error, stream: %d has been removed before inference done.", (int32_t)id);
    assert(iter->second.progress == ProgressType::kRunning);
    iter->second.progress = ProgressType::kReady;
    m_Cond.notify_all();
    return NVDSINFER_SUCCESS;
}

void
StreamManager::notifyError(NvDsInferStatus status)
{
    UniqLock locker(m_Mutex);
    m_Stopping = true;
    m_Cond.notify_all();
}

}  // namespace nvdsinferserver
