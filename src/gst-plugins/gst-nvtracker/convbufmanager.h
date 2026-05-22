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

#ifndef _CONVBUFMANAGER_H
#define _CONVBUFMANAGER_H

#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <cuda.h>
#include <cuda_runtime.h>

#include "nvbufsurface.h"
#include "nvbufsurftransform.h"

#define MAX_BUFFER_POOL_SIZE 2 // Set surface buffer pool size as a fixed number.

class ConvBufManager
{
public:
  ConvBufManager();
  ~ConvBufManager();

  bool init(uint32_t batchSize, int32_t gpuId, int32_t compute_hw,
        NvBufSurfaceCreateParams& bufferParam, const bool empty);
  void deInit();

  /** Perform surface transform for a batch asynchronously. */
  NvBufSurface *convertBatchAsync(NvBufSurface *pBatchIn, NvBufSurfTransformSyncObj_t *bufSetSyncObjs);
  /** Return unused buffer. */
  void returnBuffer(NvBufSurface *pBufferSet);
  /** Wait until surface transform is finished. */
  void syncBuffer(NvBufSurface *pBufferSet, NvBufSurfTransformSyncObj_t *bufSetSyncObjs);
  /** Getter */
  bool isQueueFull() {
    return (m_FreeQueue.size() == MAX_BUFFER_POOL_SIZE);
  }
  bool isQueueEmpty() {
    return (m_FreeQueue.empty());
  }
  int getMaxPoolSize() {
    /** Total pool size = num of buffer sets */
    return MAX_BUFFER_POOL_SIZE;
  }

private:
  /** All buffers in the pool. */
  std::vector<NvBufSurface *> m_BufferSet;

  /** Buffer pool is in running state. */
  bool m_Running;
  /** Buffer pool is initialized. */
  bool m_Initialized;
  /** Buffer pool is empty when surface transform is not required such as IOU tracker. */
  bool m_IntentionallyEmpty;
  /** Parameters for a surface transform call. */
  NvBufSurfTransformParams m_TransformParams;
  /** Config for a transform/composite session. */
  NvBufSurfTransformConfigParams m_TransformConfigParams;
  /** Unused buffer in the pool. */
  std::queue<NvBufSurface *> m_FreeQueue;
};

#endif
