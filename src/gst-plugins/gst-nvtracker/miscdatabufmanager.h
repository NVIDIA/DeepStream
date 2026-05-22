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

#ifndef _MISCDATABUFMANAGER_H
#define _MISCDATABUFMANAGER_H

#include <queue>
#include <mutex>
#include <vector>
#include <condition_variable>
#include <chrono>

#include <gst/gst.h>

#include "nvdstracker.h"

/** Tracker misc data buffer for a batch. */
struct NvTrackerMiscDataBuffer
{
  /** Past frame objects. */
  NvDsTargetMiscDataBatch pastFrameObjBatch;
  /** Target trajectories. */
  NvDsTrajectoryBatch trajectoryBatch;
  /** ReID tensor. */
  NvDsReidTensorBatch reidTensorBatch;
  /** Terminated Objects . */
  NvDsTargetMiscDataBatch terminatedTrackBatch;
  /** Shadow Objects . */
  NvDsTargetMiscDataBatch shadowTracksBatch;

};

/** Tracker misc data memory pool. */
class TrackerMiscDataManager
{
public:
  TrackerMiscDataManager();
  ~TrackerMiscDataManager();
  /** Initialize buffer pool based on tracker size information. */
  bool init(uint32_t batchSize, uint32_t gpuId,
    uint32_t maxTargetsPerStream, uint32_t maxShadowTrackingAge,
    uint32_t reidFeatureSize, uint32_t maxBufferPoolSize, bool pastFrame,
    bool outputReidTensor,
    bool outputTerminatedTracks,
    bool outputShadowTracks,
    uint32_t maxTerminatedFrameHistory);
  /** Return buffer to pool. */
  void returnBuffer(NvTrackerMiscDataBuffer *data);
  /** Pop a buffer from pool. */
  NvTrackerMiscDataBuffer *pop();
  /** Release buffer pool. */
  void deInit();


private:
  /** Buffer pool is empty when there is no misc data.*/
  bool m_IntentionallyEmpty;
  bool m_PastFrame;
  bool m_OutputReidTensor;
  bool m_OutputTerminatedTracks;
  bool m_OutputShadowTracks;
  /** Lock to write the free queue. */
  std::mutex m_Mutex;
  std::condition_variable m_Cond;
  /** Unused buffer queue. */
  std::queue<NvTrackerMiscDataBuffer *> m_FreeQueue;
  /** All buffers in the pool. */
  std::vector<NvTrackerMiscDataBuffer *> m_BufferSet;
  /** Getter */
  bool isQueueFull() { return m_BufferSet.size() == m_FreeQueue.size(); }
  /** Allocate memory for past frame objects. */
  void allocatePastFrame(NvTrackerMiscDataBuffer *pNewBuf,
    uint32_t batchSize, uint32_t maxTargetsPerStream, uint32_t maxShadowTrackingAge);
  /** Release memory for past frame objects. */
  void releasePastFrame(NvTrackerMiscDataBuffer *pBuffer);
  /** Clear objects in memory filled previously. */
  void resetPastFrame(NvTrackerMiscDataBuffer *pBuffer);
  /** Allocate memory for reid tensors. */
  void allocateReid(NvTrackerMiscDataBuffer *pNewBuf,
    uint32_t batchSize, uint32_t maxTargetsPerStream,
    uint32_t reidFeatureSize);
  /** Release memory for reid tensors. */
  void releaseReid(NvTrackerMiscDataBuffer *pBuffer);
  /** Clear reid tensors in memory filled previously. */
  void resetReid(NvTrackerMiscDataBuffer *pBuffer);
    /** Allocate memory for Terminated Tracks */
  void allocateTerminatedTracks(NvTrackerMiscDataBuffer *pNewBuf,
    uint32_t batchSize, uint32_t maxTargetsPerStream, uint32_t maxTermTrackFrameHistory);
  /** Release memory for Terminated Tracks */
  void releaseTerminatedTracks(NvTrackerMiscDataBuffer *pBuffer);
  /** Clear memory for Terminated Tracks. */
  void resetTerminatedTracks(NvTrackerMiscDataBuffer *pBuffer);
  /** Allocate memory for Shadow Tracks. */
  void allocateShadowTracks(NvTrackerMiscDataBuffer *pNewBuf,
                            uint32_t batchSize, uint32_t maxTargetsPerStream );
  /** Clear memory for Shadow Tracks. */
  void releaseShadowTracks(NvTrackerMiscDataBuffer *pBuffer);
  /** Release memory for Shadow Tracks. */
  void resetShadowTracks(NvTrackerMiscDataBuffer *pBuffer);
};

/** GStreamer mini object for GStreamer pipeline to control tracker user meta. */
struct GstNvTrackerMiscDataObject
{
  /** GStreamer mini object for refcount. */
  GstMiniObject mini_object;
  /** Pointer to user misc data buffer pool. */
  TrackerMiscDataManager* misc_data_manager;
  /** Pointer to current buffer. */
  NvTrackerMiscDataBuffer* misc_data_buffer;
};

#endif