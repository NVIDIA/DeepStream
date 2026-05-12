/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
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
