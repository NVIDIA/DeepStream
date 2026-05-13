/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <unistd.h>

#include "convbufmanager.h"
#include "logging.h"

using namespace std;

ConvBufManager::ConvBufManager()
{
  m_Running = false;
  m_Initialized = false;
  m_IntentionallyEmpty = false;
  m_TransformParams.src_rect = NULL;
  m_TransformParams.dst_rect = NULL;
  m_TransformConfigParams.cuda_stream = 0;
}

ConvBufManager::~ConvBufManager()
{
}

bool ConvBufManager::init(uint32_t batchSize, int32_t gpuId,
              int32_t compute_hw,
              NvBufSurfaceCreateParams &bufferParam,
              const bool empty)
{
  /** If no buffers are actually requested, then just open for business. */
  if (empty)
  {
    m_IntentionallyEmpty = true;
    m_Running = true;
    return true;
  }

  /** Setup transform session configs */
  m_TransformConfigParams.compute_mode = static_cast<NvBufSurfTransform_Compute>(compute_hw);

  m_TransformConfigParams.gpu_id = gpuId;
  m_TransformConfigParams.cuda_stream = 0;
  cudaError_t cudaReturn = cudaSetDevice(gpuId);
  if (cudaReturn != cudaSuccess)
  {
    LOG_ERROR("gstnvtracker: Failed to set gpu-id with error: %s\n",
          cudaGetErrorName(cudaReturn));
    return false;
  }
  cudaReturn =
    cudaStreamCreateWithFlags(&m_TransformConfigParams.cuda_stream,
                  cudaStreamNonBlocking);
  if (cudaReturn != cudaSuccess)
  {
    LOG_ERROR("gstnvtracker: Failed to create cuda stream for buffer conversion: %s\n",
          cudaGetErrorName(cudaReturn));
    return false;
  }

  /** Set up transform params for color conversion and scaling. */
  m_TransformParams.transform_flag = NVBUFSURF_TRANSFORM_FILTER;
  m_TransformParams.transform_flip = NvBufSurfTransform_None;
  m_TransformParams.transform_filter = NvBufSurfTransformInter_Bilinear;
  /** Src/dst_rect will be filled in later when doing the conversion */
  m_TransformParams.src_rect = (NvBufSurfTransformRect *)calloc(batchSize, sizeof(NvBufSurfTransformRect));
  m_TransformParams.dst_rect = (NvBufSurfTransformRect *)calloc(batchSize, sizeof(NvBufSurfTransformRect));

  /** Create the buffers. The proper way is to set number of buffer sets as a fixed number. */
  for (uint32_t setInd = 0; setInd < MAX_BUFFER_POOL_SIZE; setInd++)
  {

    NvBufSurface *pNewBuf = nullptr;
    int ret = NvBufSurfaceCreate(&pNewBuf, batchSize, &bufferParam);
    if (ret < 0)
    {
      LOG_ERROR("gstnvtracker: Got %d creating nvbufsurface\n", ret);
      deInit();
      return false;
    }

    m_BufferSet.push_back(pNewBuf);

    /** Map all the buffers */
    switch (pNewBuf->memType)
    {
#ifdef __aarch64__
    case NVBUF_MEM_DEFAULT:
    case NVBUF_MEM_SURFACE_ARRAY:
    case NVBUF_MEM_HANDLE:
#else
    case NVBUF_MEM_CUDA_UNIFIED:
#endif
      /** Need to loop for now because the batch mapping only happens for filled buffers */
      for (uint32_t k = 0; k < pNewBuf->batchSize; k++)
      {
        ret = NvBufSurfaceMap(pNewBuf, k, -1, NVBUF_MAP_READ);
        if (ret < 0)
        {
          LOG_ERROR("gstnvtracker: Got %d mapping nvbufsurface\n", ret);
          deInit();
          return false;
        }
      }
      break;
    default:
      /** Mapping is not needed or supported for other types. */
      break;
    }

    if (pNewBuf->memType == NVBUF_MEM_SURFACE_ARRAY)
    {
      /** Map all the buffers to EglImage (this works for unfilled buffers) */
      ret = NvBufSurfaceMapEglImage(pNewBuf, -1);
      if (ret < 0)
      {
        LOG_ERROR("gstnvtracker: Got %d mapping nvbufsurface\n", ret);
        deInit();
        return false;
      }
    }
    returnBuffer(pNewBuf);
  }

  m_Running = true;
  m_Initialized = true;

  return true;
}

void ConvBufManager::deInit()
{
  m_Running = false;

  if (m_IntentionallyEmpty)
  {
    return;
  }

  /** Clear the queue */
  while (!m_FreeQueue.empty())
  {
    int ret = -1;
    NvBufSurface *pBuffer = m_FreeQueue.front();
    m_FreeQueue.pop();

    if (pBuffer->memType == NVBUF_MEM_SURFACE_ARRAY)
    {
      ret = NvBufSurfaceUnMap(pBuffer, -1, -1);
      if (ret < 0)
      {
        LOG_WARNING("gstnvtracker: Got %d unmapping nvbufsurface %p\n", ret, pBuffer);
      }
      ret = NvBufSurfaceUnMapEglImage(pBuffer, -1);
      if (ret < 0)
      {
        LOG_WARNING("gstnvtracker: Got %d unmapping egl image for nvbufsurface %p\n", ret, pBuffer);
      }
    }
    ret = NvBufSurfaceDestroy(pBuffer);
    if (ret < 0)
    {
      LOG_WARNING("gstnvtracker: Got %d destroying nvbufsurface\n", ret);
    }
    pBuffer = nullptr;

  }

  m_BufferSet.clear();

  if (m_TransformParams.src_rect)
  {
    free(m_TransformParams.src_rect);
    m_TransformParams.src_rect = nullptr;
  }

  if (m_TransformParams.dst_rect)
  {
    free(m_TransformParams.dst_rect);
    m_TransformParams.dst_rect = nullptr;
  }

  if (m_TransformConfigParams.cuda_stream != 0)
  {
    cudaStreamDestroy(m_TransformConfigParams.cuda_stream);
    m_TransformConfigParams.cuda_stream = 0;
  }
}

NvBufSurface *ConvBufManager::convertBatchAsync(NvBufSurface *pBatchIn, NvBufSurfTransformSyncObj_t *bufSetSyncObjs)
{
  if (m_IntentionallyEmpty)
  {
    return nullptr;
  }

  /** Sometimes there may be spurious wakeups on the
   * condition variable. So keep waiting until we
   * actually get a buffer set or buffer pool destruction. */
  NvBufSurface *pBuffer = m_FreeQueue.front();
  m_FreeQueue.pop();

  if (nullptr == pBuffer)
  {
    return nullptr;
  }

  NvBufSurfTransform_Error err;

  err = NvBufSurfTransformSetSessionParams(&m_TransformConfigParams);
  if (err != NvBufSurfTransformError_Success)
  {
    LOG_ERROR("gstnvtracker: NvBufSurfTransformSetSessionParams failed with error %d", err);
    return nullptr;
  }

  uint32_t batchSize = pBatchIn->numFilled;

  /** Set each src surface ROI to the entire surface */
  for (uint32_t i = 0; i < batchSize; i++)
  {
    m_TransformParams.src_rect[i].top = 0;
    m_TransformParams.src_rect[i].left = 0;
    m_TransformParams.src_rect[i].width = pBatchIn->surfaceList[i].width;
    m_TransformParams.src_rect[i].height = pBatchIn->surfaceList[i].height;
  }

  /** Set each dst surface ROI to the entire surface */
  for (uint32_t frameInd = 0; frameInd < batchSize; frameInd++)
  {
    m_TransformParams.dst_rect[frameInd].top = 0;
    m_TransformParams.dst_rect[frameInd].left = 0;
    m_TransformParams.dst_rect[frameInd].width = pBuffer->surfaceList[frameInd].width;
    m_TransformParams.dst_rect[frameInd].height = pBuffer->surfaceList[frameInd].height;
  }
  *bufSetSyncObjs = nullptr;
  err = NvBufSurfTransformAsync(pBatchIn, pBuffer, &m_TransformParams, bufSetSyncObjs);
  if (err != NvBufSurfTransformError_Success)
  {
    LOG_ERROR("gstnvtracker: NvBufSurfTransform failed with error %d while converting buffer", err);
    returnBuffer(pBuffer);
    return nullptr;
  }
  return pBuffer;
}

void ConvBufManager::returnBuffer(NvBufSurface *pBuffer)
{
  if (m_IntentionallyEmpty)
  {
    return;
  }

  m_FreeQueue.push(pBuffer);
}

void ConvBufManager::syncBuffer(NvBufSurface *pBuffer, NvBufSurfTransformSyncObj_t *bufSetSyncObjs)
{
  if (m_IntentionallyEmpty)
  {
    return;
  }

  /** Sync surface transform */
  if (bufSetSyncObjs && (*bufSetSyncObjs))
  {
    NvBufSurfTransformSyncObjWait(*bufSetSyncObjs, -1);
    NvBufSurfTransformSyncObjDestroy(bufSetSyncObjs);
  }

  /** Sync on CPU for NVBUF_MEM_SURFACE_ARRAY memory type */
  if (pBuffer && pBuffer->memType == NVBUF_MEM_SURFACE_ARRAY)
  {
    for (uint32_t i = 0; i < pBuffer->numFilled; i++)
      NvBufSurfaceSyncForCpu(pBuffer, i, -1);
  }
}
