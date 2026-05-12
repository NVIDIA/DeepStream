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

#ifndef _INVTRACKERPROC_H
#define _INVTRACKERPROC_H

#include "nvbufsurface.h"
#include "nvdstracker.h"

#include <vector>

/** Input data for tracker plugin. */
struct InputParams
{
  NvBufSurface *pSurfaceBatch;
  NvDsBatchMeta *pBatchMeta;
  void* pPreservedData;
  bool eventMarker;
};

/** Tracker process completion status. */
enum CompletionStatus
{
  CompletionStatus_OK,
  CompletionStatus_Error,
  CompletionStatus_Exit
};

/** Bitwise flags for tracker id reset. */
enum TrackingIdResetMode
{
  /** No id reset. */
  TrackingIdResetMode_Default = 0,
  /** Terminate existing objects and assign new ids after stream reset. */
  TrackingIdResetMode_NewIdAfterStreamReset = 1 << 0,
  /** Id starts from 0 after stream reaching EOS. */
  TrackingIdResetMode_FromZeroAfterEOS = 1 << 1,
  /** Max value of the enum. Supporting both NewIdAfterStreamReset and FromZeroAfterEOS. */
  TrackingIdResetMode_MaxValue = 3
};

/** Tracker plugin config params. */
struct TrackerConfig
{
  /** From DeepStream app config file. */
  uint32_t batchSize;
  uint32_t trackerWidth;
  uint32_t trackerHeight;
  char* trackerLibFile;
  char* trackerConfigFileList;
  std::vector<std::string> trackerConfigFilePerSubBatch;

  bool displayTrackingId;
  TrackingIdResetMode trackingIdResetMode;

  NvMOTCompute computeTarget;
  int32_t gpuId;
  int32_t compute_hw;

  uint32_t trackingSurfType;
  bool trackingSurfTypeFromConfig = false;

  bool inputTensorMeta = false;
  uint32_t tensorMetaGieId = 0;
  /** vector < sub-batch ids : vector <source ids in each sub-batch > >*/
  std::vector<std::vector<int>> subBatchesConfig = {};
  std::vector<uint32_t> subBatchSizes = {};
  /** dynamicSubBatching will be set to "true" when user specifies sub-batch sizes and */
  /** i.e. the actual mapping from source id (pad index) to sub-batch happens dynamically (run-time)*/
  bool dynamicSubBatching = false;
  int subBatchErrRecoveryTrialCnt;

  /** From low level tracker library query. */
  NvBufSurfaceColorFormat colorFormat;
  NvBufSurfaceMemType memType;
  uint32_t numTransforms;
  uint32_t maxTargetsPerStream;
  uint32_t maxShadowTrackingAge;
  bool pastFrame;
  bool outputTerminatedTracks;
  uint32_t maxTrajectoryBufferLength;

  bool outputShadowTracks;

  /** Store buffer pool size since low level tracker needs this info. */
  uint32_t maxConvBufPoolSize;
  uint32_t maxMiscDataPoolSize;
  uint32_t reidFeatureSize;
  uint32_t maxConvexHullSize;
  bool outputReidTensor;
  bool outputVisibility;
  bool outputFootLocation;
  bool outputConvexHull;

  char* gstName;

  /** Operate on class IDs filtering parameters. */
  std::vector<gboolean> operateOnClassIds;
};

/** Virtual base class for tracker plugin processing. */
class INvTrackerProc
{
public:
  virtual ~INvTrackerProc() {};

  virtual bool init(const TrackerConfig& config) = 0;
  virtual void deInit() = 0;

  /** Tracker actions when a source is added to the pipeline. */
  virtual bool addSource(uint32_t sourceId) = 0;
  /** Tracker actions when a source is removed to the pipeline. */
  virtual bool removeSource(uint32_t sourceId, bool removeObjectIdMapping=true) = 0;
  /** Tracker actions when a source is reset. */
  virtual bool resetSource(uint32_t sourceId) = 0;
  /** Submit an input batch to tracker process queue. */
  virtual bool submitInput(const InputParams& inputParams) = 0;
  /** Wait until a batch's process is done.*/
  virtual CompletionStatus waitForCompletion(InputParams& inputParams) = 0;
  /** Flush the request to send the batch downstream. */
  virtual bool flushReqs() = 0;
  /** Update the low-level tracker lib's config dynamically during runtime */
  virtual bool updateDynamicConfig(const std::string& configStr, uint32_t sourceId)=0;
};

#endif
