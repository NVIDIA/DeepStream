/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "cuda_runtime.h"

#include "miscdatabufmanager.h"
#include "logging.h"

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/** CUDA: various checks for different function calls. */
#define CUDA_CHECK(condition)	gpuAssert(condition, __FILE__, __LINE__);

#define MAX_BUFFER_WAIT_TIMEOUT std::chrono::milliseconds(50)

TrackerMiscDataManager::TrackerMiscDataManager()
{
  m_IntentionallyEmpty = false;
}

TrackerMiscDataManager::~TrackerMiscDataManager()
{
}

bool TrackerMiscDataManager::init(uint32_t batchSize, uint32_t gpuId,
  uint32_t maxTargetsPerStream, uint32_t maxShadowTrackingAge,
  uint32_t reidFeatureSize, uint32_t maxBufferPoolSize, bool pastFrame, bool outputReidTensor,
  bool outputTerminatedTracks, bool outputShadowTracks, uint32_t maxTerminatedFrameHistory )
{
  if ( (!pastFrame) && (!outputReidTensor) && (!outputTerminatedTracks) )
  {
    m_IntentionallyEmpty = true;
  }
  m_PastFrame = pastFrame;
  m_OutputReidTensor = outputReidTensor;
  m_OutputTerminatedTracks = outputTerminatedTracks;
  m_OutputShadowTracks     = outputShadowTracks;

  if (m_IntentionallyEmpty)
  {
    return true;
  }

  if (m_OutputReidTensor && reidFeatureSize == 0)
  {
    LOG_ERROR("reidFeatureSize must be a positive integer\n");
    return false;
  }

  std::unique_lock<std::mutex> lock(m_Mutex);
  /** Create the buffers. The proper way is to set number of buffer sets as a fixed number. */
  for (uint32_t setInd = 0; setInd < maxBufferPoolSize; setInd++)
  {
    NvTrackerMiscDataBuffer *pNewBuf = new NvTrackerMiscDataBuffer;
    allocatePastFrame(pNewBuf, batchSize, maxTargetsPerStream, maxShadowTrackingAge);
    allocateReid(pNewBuf, batchSize, maxTargetsPerStream, reidFeatureSize);
    allocateTerminatedTracks(pNewBuf, batchSize, maxTargetsPerStream, maxTerminatedFrameHistory);
    allocateShadowTracks(pNewBuf, batchSize, maxTargetsPerStream);

    m_BufferSet.push_back(pNewBuf);
    m_FreeQueue.push(pNewBuf);
  }
  return true;
}

void TrackerMiscDataManager::allocatePastFrame(NvTrackerMiscDataBuffer *pNewBuf,
    uint32_t batchSize, uint32_t maxTargetsPerStream, uint32_t maxShadowTrackingAge)
{
  if (!m_PastFrame)
  {
    return;
  }
  /** Past frame memory */
  NvDsTargetMiscDataBatch& objBatch = pNewBuf->pastFrameObjBatch;
  objBatch.list = new NvDsTargetMiscDataStream [batchSize];
  objBatch.numAllocated = batchSize;
  objBatch.numFilled = 0;
  objBatch.priv_data = nullptr;
  for(uint32_t streamInd=0; streamInd < objBatch.numAllocated; streamInd++)
  {
    NvDsTargetMiscDataStream& objStream =  objBatch.list[streamInd];
    objStream.list = new NvDsTargetMiscDataObject [maxTargetsPerStream];
    objStream.numAllocated = maxTargetsPerStream;
    objStream.numFilled = 0;
    for(uint32_t targetInd=0; targetInd < maxTargetsPerStream; targetInd++)
    {
      NvDsTargetMiscDataObject& objList = objStream.list[targetInd];
      objList.list = new NvDsTargetMiscDataFrame [maxShadowTrackingAge];
      objList.numObj = 0;
    }
  }
}

void TrackerMiscDataManager::releasePastFrame(NvTrackerMiscDataBuffer *pBuffer)
{
  if (!m_PastFrame)
  {
    return;
  }
  /** Past frame memory */
  NvDsTargetMiscDataBatch& objBatch = pBuffer->pastFrameObjBatch;
  for(uint32_t streamInd=0; streamInd < objBatch.numAllocated; streamInd++)
  {
    NvDsTargetMiscDataStream& objStream =  objBatch.list[streamInd];
    for(uint32_t targetInd=0; targetInd < objStream.numAllocated; targetInd++)
    {
      NvDsTargetMiscDataObject& objList = objStream.list[targetInd];
      delete [] objList.list;
    }
    delete [] objStream.list;
  }
  delete [] objBatch.list;
}

void TrackerMiscDataManager::resetPastFrame(NvTrackerMiscDataBuffer *pBuffer)
{
  if (!m_PastFrame)
  {
    return;
  }
  NvDsTargetMiscDataBatch& objBatch = pBuffer->pastFrameObjBatch;
  objBatch.numFilled = 0;
  for(uint32_t streamInd=0; streamInd < objBatch.numAllocated; streamInd++)
  {
    NvDsTargetMiscDataStream& objStream =  objBatch.list[streamInd];
    objStream.numFilled = 0;
    for(uint32_t targetInd=0; targetInd < objStream.numAllocated; targetInd++)
    {
      NvDsTargetMiscDataObject& objList = objStream.list[targetInd];
      objList.numObj = 0;
    }
  }
}

void TrackerMiscDataManager::allocateReid(NvTrackerMiscDataBuffer *pNewBuf,
    uint32_t batchSize, uint32_t maxTargetsPerStream,
    uint32_t reidFeatureSize)
{
  if (!m_OutputReidTensor)
  {
    return;
  }
  NvDsReidTensorBatch& reidTensor = pNewBuf->reidTensorBatch;
  CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &reidTensor.ptr_dev ),
      batchSize * maxTargetsPerStream * reidFeatureSize * sizeof(float)) );
  CUDA_CHECK( cudaMallocHost( reinterpret_cast<void**>( &reidTensor.ptr_host ),
      batchSize * maxTargetsPerStream * reidFeatureSize  * sizeof(float)) );
  reidTensor.numFilled = 0;
  reidTensor.featureSize = reidFeatureSize;
  reidTensor.priv_data = nullptr;
}

void TrackerMiscDataManager::releaseReid(NvTrackerMiscDataBuffer *pBuffer)
{
  if (!m_OutputReidTensor)
  {
    return;
  }
  CUDA_CHECK( cudaFree( pBuffer->reidTensorBatch.ptr_dev ) );
  CUDA_CHECK( cudaFreeHost( pBuffer->reidTensorBatch.ptr_host ) );
}

void TrackerMiscDataManager::resetReid(NvTrackerMiscDataBuffer *pBuffer)
{
}

void TrackerMiscDataManager::allocateTerminatedTracks(NvTrackerMiscDataBuffer *pNewBuf,
    uint32_t batchSize, uint32_t maxTargetsPerStream, uint32_t maxTermTrackFrameHistory)
{
  if (!m_OutputTerminatedTracks)
  {
    return;
  }

  /** allocate Batch memory covering all streams */
  NvDsTargetMiscDataBatch& objBatch = pNewBuf->terminatedTrackBatch;

  objBatch.list = new NvDsTargetMiscDataStream [batchSize];
  objBatch.numAllocated = batchSize;
  objBatch.numFilled = 0;
  objBatch.priv_data = nullptr;

  /** For all Streams, allocate for max obj per stream */
  for(uint32_t streamInd=0; streamInd < objBatch.numAllocated; streamInd++)
  {
    NvDsTargetMiscDataStream& objStream =  objBatch.list[streamInd];
    objStream.list = new NvDsTargetMiscDataObject [maxTargetsPerStream];
    objStream.numAllocated = maxTargetsPerStream;
    objStream.numFilled = 0;

    /** For each Target, allocate max frames of history */
    for(uint32_t targetInd=0; targetInd < maxTargetsPerStream; targetInd++)
    {
      NvDsTargetMiscDataObject& objList = objStream.list[targetInd];
      objList.list = new NvDsTargetMiscDataFrame [maxTermTrackFrameHistory];
      objList.numAllocated = maxTermTrackFrameHistory;
      objList.numObj = 0;
    }
  }

}

void TrackerMiscDataManager::releaseTerminatedTracks(NvTrackerMiscDataBuffer *pBuffer)
{
  if (!m_OutputTerminatedTracks)
  {
    return;
  }

  /** Past frame memory */
  NvDsTargetMiscDataBatch& objBatch = pBuffer->terminatedTrackBatch;
  for(uint32_t streamInd=0; streamInd < objBatch.numAllocated; streamInd++)
  {
    NvDsTargetMiscDataStream& objStream =  objBatch.list[streamInd];
    for(uint32_t targetInd=0; targetInd < objStream.numAllocated; targetInd++)
    {
      NvDsTargetMiscDataObject& objList = objStream.list[targetInd];
      delete [] objList.list;
    }
    delete [] objStream.list;
  }
  delete [] objBatch.list;

}

void TrackerMiscDataManager::resetTerminatedTracks(NvTrackerMiscDataBuffer *pBuffer)
{
  if (!m_OutputTerminatedTracks)
  {
    return;
  }

  NvDsTargetMiscDataBatch& objBatch = pBuffer->terminatedTrackBatch;
  objBatch.numFilled = 0;
  for(uint32_t streamInd=0; streamInd < objBatch.numAllocated; streamInd++)
  {
    NvDsTargetMiscDataStream& objStream =  objBatch.list[streamInd];
    objStream.numFilled = 0;
    for(uint32_t targetInd=0; targetInd < objStream.numAllocated; targetInd++)
    {
      NvDsTargetMiscDataObject& objList = objStream.list[targetInd];
      objList.numObj = 0;
    }
  }
}


void TrackerMiscDataManager::allocateShadowTracks(NvTrackerMiscDataBuffer *pNewBuf,
    uint32_t batchSize, uint32_t maxTargetsPerStream )
{
  if (!m_OutputShadowTracks)
  {
    return;
  }

  /** allocate Batch memory covering all streams */
  NvDsTargetMiscDataBatch& objBatch = pNewBuf->shadowTracksBatch;

  objBatch.list = new NvDsTargetMiscDataStream [batchSize];
  objBatch.numAllocated = batchSize;
  objBatch.numFilled = 0;
  objBatch.priv_data = nullptr;

  /** For all Streams, allocate for max obj per stream */
  for(uint32_t streamInd=0; streamInd < objBatch.numAllocated; streamInd++)
  {
    NvDsTargetMiscDataStream& objStream =  objBatch.list[streamInd];
    objStream.list = new NvDsTargetMiscDataObject [maxTargetsPerStream];
    objStream.numAllocated = maxTargetsPerStream;
    objStream.numFilled = 0;

    /** For each Target, allocate max frames of history */
    for(uint32_t targetInd=0; targetInd < maxTargetsPerStream; targetInd++)
    {
      NvDsTargetMiscDataObject& objList = objStream.list[targetInd];
      objList.list = new NvDsTargetMiscDataFrame [1];
      objList.numObj = 0;
    }
  }

}

void TrackerMiscDataManager::releaseShadowTracks(NvTrackerMiscDataBuffer *pBuffer)
{
  if (!m_OutputShadowTracks)
  {
    return;
  }

  /** Past frame memory */
  NvDsTargetMiscDataBatch& objBatch = pBuffer->shadowTracksBatch;
  for(uint32_t streamInd=0; streamInd < objBatch.numAllocated; streamInd++)
  {
    NvDsTargetMiscDataStream& objStream =  objBatch.list[streamInd];
    for(uint32_t targetInd=0; targetInd < objStream.numAllocated; targetInd++)
    {
      NvDsTargetMiscDataObject& objList = objStream.list[targetInd];
      delete [] objList.list;
    }
    delete [] objStream.list;
  }
  delete [] objBatch.list;

}

void TrackerMiscDataManager::resetShadowTracks(NvTrackerMiscDataBuffer *pBuffer)
{
  if (!m_OutputShadowTracks)
  {
    return;
  }

  NvDsTargetMiscDataBatch& objBatch = pBuffer->shadowTracksBatch;
  objBatch.numFilled = 0;
  for(uint32_t streamInd=0; streamInd < objBatch.numAllocated; streamInd++)
  {
    NvDsTargetMiscDataStream& objStream =  objBatch.list[streamInd];
    objStream.numFilled = 0;
    for(uint32_t targetInd=0; targetInd < objStream.numAllocated; targetInd++)
    {
      NvDsTargetMiscDataObject& objList = objStream.list[targetInd];
      objList.numObj = 0;
    }
  }
}

void TrackerMiscDataManager::returnBuffer(NvTrackerMiscDataBuffer *data)
{
  if (m_IntentionallyEmpty)
  {
    return;
  }

  resetPastFrame(data);
  resetReid(data);
  resetTerminatedTracks(data);
  resetShadowTracks(data);

  std::unique_lock<std::mutex> lock(m_Mutex);
  m_FreeQueue.push(data);
  m_Cond.notify_one();
}

NvTrackerMiscDataBuffer *TrackerMiscDataManager::pop()
{
  if (m_IntentionallyEmpty)
  {
    return nullptr;
  }

  std::unique_lock<std::mutex> lock(m_Mutex);
  NvTrackerMiscDataBuffer *ret = nullptr;

  if (m_Cond.wait_for(lock, MAX_BUFFER_WAIT_TIMEOUT, [this]() { return !m_FreeQueue.empty(); }) )
  {
    if (!m_FreeQueue.empty())
    {
      ret = m_FreeQueue.front();
      m_FreeQueue.pop();
    }
  }
  return ret;
}

void TrackerMiscDataManager::deInit()
{
  if (m_IntentionallyEmpty)
  {
    return;
  }

  std::unique_lock<std::mutex> lock(m_Mutex);
  while (!isQueueFull())
  {
    m_Cond.wait(lock);
  }
  while (!m_FreeQueue.empty())
  {
    NvTrackerMiscDataBuffer *pBuffer = m_FreeQueue.front();
    m_FreeQueue.pop();

    releasePastFrame(pBuffer);
    releaseReid(pBuffer);
    releaseTerminatedTracks(pBuffer);
    releaseShadowTracks(pBuffer);

    delete pBuffer;
  }
  m_BufferSet.clear();
}
