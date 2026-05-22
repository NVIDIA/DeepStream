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

#ifndef _NVTRACKERPROC_H
#define _NVTRACKERPROC_H

#include <queue>
#include <map>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <shared_mutex>

#include "nvdsmeta.h"
#include "invtracker_proc.h"
#include "convbufmanager.h"
#include "miscdatabufmanager.h"
#include "nvdstracker.h"

/** Tracker process class. */
class NvTrackerProc : public INvTrackerProc
{
public:
  NvTrackerProc();
  virtual ~NvTrackerProc();

  virtual bool init(const TrackerConfig& config);
  virtual void deInit();

  /** Tracker actions when a source is added to the pipeline. */
  virtual bool addSource(uint32_t sourceId);
  /** Tracker actions when a source is removed to the pipeline. */
  virtual bool removeSource(uint32_t sourceId, bool removeObjectIdMapping=true);
  /** Tracker actions when a source is reset. */
  virtual bool resetSource(uint32_t sourceId);
  /** Submit an input batch to tracker process queue. */
  virtual bool submitInput(const InputParams& inputParams);
  /** Wait until a batch's process is done.*/
  virtual CompletionStatus waitForCompletion(InputParams& inputParams);
  /** Flush the request to send the batch downstream. */
  virtual bool flushReqs();
  /** Update the low-level tracker lib's config dynamically during runtime */
  virtual bool updateDynamicConfig(const std::string& configStr, uint32_t sourceId);

protected:
  /** Object class information. */
  struct ClassInfo
  {
    NvOSD_RectParams rectParams;
    NvOSD_TextParams textParams;
    std::string displayTextString;
    std::string objLabel;
    uint uniqueComponentId;
  };

  /** Info mapped per class. The key is class ID. */
  std::map<int, ClassInfo> m_ClassInfoMap;
  std::shared_mutex m_ClassInfoMapLock;

  /** Current batch ID. */
  typedef uint32_t BatchId;

  /** Composition of unique ID for each surface from each source
   *  Some sources (e.g. 360d) produce multiple surfaces per frame due to dewarping etc.
   */
  typedef uint64_t SurfaceStreamId;
  /** Source stream ID. */
  typedef uint32_t StreamId;
  /** Surface type ID. */
  typedef uint32_t SurfaceId;
  /** sub-batch ID. */
  typedef uint32_t SubBatchId;
  /** Sequence index in a sub-batch. i.e. "seq_index" in "NvMOTFrame" */
  typedef uint32_t SeqIndex;
  /** Tracker processing config. */
  TrackerConfig m_Config;
  /** Tracker lib is running. */
  bool m_Running = false;
  /** Tracker lib encounters an error. */
  std::atomic<bool> m_TrackerLibError;

  /** Buffer manager for surface transformation. */
  ConvBufManager m_ConvBufMgr;

  /** Buffer manager for misc data (user meta). */
  TrackerMiscDataManager m_MiscDataMgr;

  /** Update dynamic config info. */
  struct UpdateDynamicConfigInfo
  {
    bool updateDynamicConfig;
    std::string configStr;
  };

  /** Source remove info. */
  struct SourceRemoveInfo
  {
    bool removeSource;
    SurfaceStreamId ssId;  /** Id of the stream to be removed */
  };

  /** Tracker data submitted for proceeding. */
  struct ProcParams
  {
    InputParams input;
    bool useConvBuf;
    NvBufSurface *pConvBuf;
    BatchId batchId;
    NvBufSurfTransformSyncObj_t bufSetSyncObjs;
    SourceRemoveInfo sourceRemoveInfo;
    UpdateDynamicConfigInfo updateDynamicConfigInfo;
  };

  /** Input queue waiting for tracker proceeding. */
  std::queue<ProcParams> m_ProcQueue;
  std::mutex m_ProcQueueLock;
  std::condition_variable m_ProcQueueCond;
  std::condition_variable m_BufQueueCond;

  /** Stores batches not finishing tracker proceeding. */
  std::map< BatchId, std::vector<SubBatchId> > m_PendingBatch;
  BatchId m_BatchId = 0;
  std::mutex m_PendingBatchLock;

  /** Stores status of pending batches for buffer conversion. */
  std::map< BatchId, bool > m_ConvBufComplete;
  std::mutex m_ConvBufLock;

  /** Variable to check if a source is removed from low level tracker lib */
  bool m_SourceRemoved = false;
  std::mutex m_SourceRemoveLock;
  std::condition_variable m_SourceRemoveCond;

  /** Batch dispatch to low-level tracker lib */
  NvMOTContextHandle m_BatchContextHandle = nullptr;
  std::thread m_ProcessBatchThread;
   /******************************************************************************************************
    * NOTE : Data structures and variables declared below are used for sub-batch processing.
    * To get more insights into sub-batches feature implementation, please refer to the
    * detailed documentation and examples elaborated at the end of nvtracker_proc.cpp
    * ******************************************************************************************************/

  /** Per-sub-batch dispatch to low-level tracker lib */
  struct DispatchReq
  {
    BatchId batchId;
    /** Map for frames present in a sub-batch */
    std::map<SurfaceStreamId, NvDsFrameMeta *> subBatchFrameMap;
    /* Tracker data for processing */
    ProcParams procParams;
  };
  /** Info corresponding to each sub-batch thread */
  struct DispatchInfo
  {
    SubBatchId sbId;
    std::thread sbThread;
    bool running;
    std::queue<DispatchReq> reqQueue;
    std::mutex reqQueueLock;
    std::condition_variable reqQueueCond;
    NvMOTContextHandle contextHandle;

    // FIXME: operator= needed but not used in meaningful way
    DispatchInfo &operator=(DispatchInfo const &rhs)
    {
      return *this;
    }
  };
  /** Mapping from sub-batch id to corresponding thread info */
  std::map<SubBatchId, DispatchInfo> m_DispatchMap;
  std::mutex m_DispatchMapLock;
  /** Mapping for SurfaceStreamId => SubBatchId */
  /** Usage : Everytime a new batch arrives, this mapping is used to find the SubBatchId */
  /** that should be used to process each frame in the batch based on the SurfaceStreamId of the frame */
  std::map<SurfaceStreamId, SubBatchId> m_SsidSubBatchMap;
  /** Pad index(Source id) mapping: {Pad index(source id): sequence index in the sub-batch} */
  /** Usage : This mapping is used for every input frame to fill "seq_index" value in "NvMOTFrame" */
  /** before being passed to the low level tracker for processing */
  std::map<uint32_t, SeqIndex> m_PadIndexSeqIndexMap;
  /** For dynamic/run-time mapping of source id(pad index) to sub-batches; maintain and update a map of allocations */
  /** map <sub-batch id, map <seq index, pair <SurfaceStreamId, pad index > >> */
  /** Usage : Keeps a track of used <SubBatchId, SeqIndex> slot */
  /** Whenever a new source is added, it is assigned to a <SubBatchId, SeqIndex> pair */
  /** and accordingly m_SubBatchAllocationMap is updated */
  /** Similarly, when a source is removed, the corresponding entry is deleted from m_SubBatchAllocationMap */
  std::map<SubBatchId, std::map<SeqIndex, std::pair<SurfaceStreamId, uint32_t>>> m_SubBatchAllocationMap;
  /** Lock for critical code for variables m_SubBatchAllocationMap, m_PadIndexSeqIndexMap, m_PadIndexSsidMap, m_SsidSubBatchMap */
  std::shared_mutex m_SubBatchMapLock;
  /** Max sub-batch size */
  uint32_t m_MaxSubBatchSize = 0;
  /** Function to pop sub-batch from queue, call low level lib to process
   *  and update meta data with tracking results for the sub-batch
   */
  void processSubBatch(DispatchInfo *pDispatchInfo);
  /** Function to create sub-batches from a batch and submit to
   * corresponding sub-batch queues */
  bool dispatchSubBatches(ProcParams procParams);
  /** Fuction to map a stream with a specific (SurfaceStreamId, pad_index) to a sub-batch */
  /** The mapping happens only the first time when a stream arrives and is stored in m_SsidSubBatchMap */
  bool mapStreamToSubBatch(SurfaceStreamId ssId, uint32_t pad_index, bool dynamicSubBatching);

  /******************************************************************************************************/

  /** Completions */
  std::queue<InputParams> m_CompletionQueue;
  std::mutex m_CompletionQueueLock;
  std::condition_variable m_CompletionQueueCond;

  /** Object id mapping: {surface stream ID: object id offset} */
  std::map<SurfaceStreamId, guint64> m_ObjectIdOffsetMap;
  std::mutex m_ObjectIdOffsetMapLock;
  /** Pad index to surface stream ID mapping. Used in source removal */
  std::map<uint32_t, SurfaceStreamId> m_PadIndexSsidMap;


  /** Low-level tracker lib API and support functions. */
  void *m_TrackerLibHandle;

  /** Internal initializing and release function. */

  /** initialize the batch context with low-level tracker lib.*/
  NvMOTContextHandle initTrackerContext(uint32_t sbId = 0);
  /** Initialize surface transform buffer pool. */
  bool initConvBufPool();
  /** Deinitialize surface transform buffer pool. */
  void deInitConvBufPool();
  /** Initialize misc data (user meta) buffer pool. */
  bool initMiscDataPool();
  /** Deinitialize misc data (user meta) buffer pool. */
  void deInitMiscDataPool();
  /** Initialize low-level tracker lib. */
  bool initTrackerLib();
  /** Deinitialize low-level tracker lib. */
  void deInitTrackerLib();
  /** Initialize parameter query from low-level tracker lib. */
  bool initTrackerQuery(const std::vector<NvMOTContextHandle>& contextHandleList);
  /** Initialize parameter query from low-level tracker lib for a single sub-batch */
  /** Required when reinitializing a sub-batch */
  bool initTrackerQuery(const NvMOTContextHandle& contextHandleList, uint32_t sbId);
  /** Allocate memory for low level library input*/
  void allocateProcessMemory(NvMOTProcessParams &procInput,
                             NvMOTTrackedObjBatch &procResult);
  void allocateProcessMemory(NvMOTProcessParams &procInput,
                             NvMOTTrackedObjBatch &procResult, uint32_t batchSize);
  /** Release input memory */
  void releaseProcessMemory(NvMOTProcessParams &procInput,
                            NvMOTTrackedObjBatch &procResult);
  void releaseProcessMemory(NvMOTProcessParams &procInput,
                            NvMOTTrackedObjBatch &procResult, uint32_t batchSize);

  void allocateConvexHullMemory(NvMOTTrackedObj *list, const uint32_t numAllocated);
  void releaseConvexHullMemory(NvMOTTrackedObj *list, const uint32_t numAllocated);

  void allocateMaskMemory(NvMOTTrackedObj *list, const uint32_t numAllocated);
  void releaseMaskMemory(NvMOTTrackedObj *list, const uint32_t numAllocated);

  /** Functions to process a batch. */

  /** In case the batch contains two frames with the same ssId,
   * sort the frames based on frameNum in ascending order.*/
  void queueFrames(const NvDsBatchMeta& batchMeta,
                   std::vector<std::map<SurfaceStreamId, NvDsFrameMeta *>>& batchList);
  /** Fill input to low-level tracker lib. */
  void fillMOTFrame(SurfaceStreamId ssId,
          const ProcParams& procParams,
          const NvDsFrameMeta& frameMeta,
          NvMOTFrame& motFrame,
          NvMOTTrackedObjList& trackedObjList);
  /** Update batch meta with tracking results. */
  void updateBatchMeta(const NvMOTTrackedObjBatch& procResult,
           const ProcParams& procParams,
           std::map<SurfaceStreamId, NvDsFrameMeta *>& frameMap);
  /** Update each stream's frame meta with tracking results. */
  void updateFrameMeta(NvDsFrameMeta* pFrameMeta,
           const NvMOTTrackedObjList& objList,
           const ProcParams& procParams);
  /** Update past frame data. */
  void updatePastFrameMeta(
           const std::vector<std::map<SurfaceStreamId, NvDsFrameMeta *>>& batchList,
           GstNvTrackerMiscDataObject *pGstObj,
           ProcParams& procParams);
  /** Update tracker user meta. */
  void updateUserMeta(
           const std::vector<std::map<SurfaceStreamId, NvDsFrameMeta *>>& batchList,
           ProcParams& procParams, NvTrackerMiscDataBuffer *pMiscDataBuf);
  void updateUserMeta(
      const std::vector<std::map<SurfaceStreamId, NvDsFrameMeta *>> &batchList,
      ProcParams &procParams, NvTrackerMiscDataBuffer *pMiscDataBuf, TrackerMiscDataManager &miscDataMgr);
  /** Update segmentation mask meta. */
  void updateObjMaskMeta(NvDsObjectMeta *pObjectMeta, NvMOTTrackedObj *pTrackedObj);
  /** Update each object's reid meta. */
  void updateObjectReidMeta(NvDsObjectMeta *pObjectMeta, NvMOTTrackedObj *pTrackedObj,
            NvDsBatchMeta *pBatchMeta);
  /** Update batch reid meta. */
  void updateBatchReidMeta(GstNvTrackerMiscDataObject *pGstObj,
                    ProcParams& procParams);
  /** Update each object's model projection meta. */
  void updateObjectProjectionMeta(NvDsObjectMeta *pObjectMeta, NvMOTTrackedObj *pTrackedObj,
            NvDsBatchMeta *pBatchMeta, float scaleWidth, float scaleHeight, NvDsFrameMeta* pFrameMeta);
  /** Add display metadata for foot location on image. */
  void addDisplayMetaFootLocation(NvMOTTrackedObj *pTrackedObj, NvDsBatchMeta *pBatchMeta,
            float scaleWidth, float scaleHeight, NvDsFrameMeta* pFrameMeta);
  /** Add display metadata for 3D bounding box on image. */
  void addDisplayMetaBbox3D(NvMOTTrackedObj *pTrackedObj, NvDsBatchMeta *pBatchMeta,
            float scaleWidth, float scaleHeight, NvDsFrameMeta* pFrameMeta);
  /** Add display metadata for convex hull on image. */
  void addDisplayMetaConvexHull(NvMOTTrackedObj *pTrackedObj, NvDsBatchMeta *pBatchMeta,
            float scaleWidth, float scaleHeight, NvDsFrameMeta* pFrameMeta);

  void updateTerminatedTrackMeta(GstNvTrackerMiscDataObject *pGstObj,
                    ProcParams& procParams);

  void updateShadowTrackMeta(GstNvTrackerMiscDataObject *pGstObj,
                     ProcParams& procParams);

  /** Function to pop batch from input queue, call low level lib to process
   *  and update meta data with tracking results.
   */
  void processBatch();

  /** Getter */
  SurfaceStreamId getSurfaceStreamId(StreamId streamId, SurfaceId surfaceId)
  { return ( ((uint64_t)streamId << 32) | surfaceId ); };
  StreamId getStreamId(SurfaceStreamId ssId) { return (StreamId)( ssId >> 32 ); };
  SurfaceId getSurfaceId(SurfaceStreamId ssId) { return (SurfaceId)(ssId); };

  /** Object ID mapper to ensure ID starts from 0 after reset. */
  guint64 objectIdMapping(const guint64& objectId, const SurfaceStreamId& ssId);
  /** Remove objects that don't need an ID. */
  void removeUntrackedObjects(NvDsFrameMeta* pFrameMeta);
  /**
   * Clip the bbox to be within the frame. If the bbox is entirely
   * outside, or shrinks to 0 in either dimension, then return false
   * so it can be removed from the object list.
   */
  bool clipBBox(uint32_t frameWidth, uint32_t frameHeight, float &left, float &top, float &width,  float &height);
  /** Tracker low level library functions. */
  /** Low level lib init. */
  NvMOTStatus (*m_TrackerLibInit) (NvMOTConfig *pConfigIn,
           NvMOTContextHandle *pContextHandle,
           NvMOTConfigResponse *pConfigResponse);
  /** Low level lib deinit. */
  void (*m_TrackerLibDeInit) (NvMOTContextHandle contextHandle);
  /** Low level lib process. */
  NvMOTStatus (*m_TrackerLibProcess) (NvMOTContextHandle contextHandle,
            NvMOTProcessParams *pParams,
            NvMOTTrackedObjBatch *pTrackedObjectsBatch);
  /** Low level lib process past frame. */
  NvMOTStatus (*m_TrackerLibRetrieveMiscData) (NvMOTContextHandle contextHandle,
            NvMOTProcessParams *pParams,
            NvMOTTrackerMiscData *pTrackerMiscData);
  /** Low level lib query. */
  NvMOTStatus (*m_TrackerLibQuery) (uint16_t customConfigFilePathSize,
            char* pCustomConfigFilePath,
            NvMOTQuery *pQuery);
  /** Low level lib remove streams. */
  NvMOTStatus (*m_TrackerLibRemoveStreams) (NvMOTContextHandle contextHandle,
            NvMOTStreamId streamIdMask);
  /** Low level lib update parameters. */
  NvMOTStatus (*m_TrackerLibUpdateParams) (NvMOTContextHandle contextHandle, const std::string &configStr);

  NvDsTargetMiscDataBatch* allocatePastFrameMemory();
};

#endif
