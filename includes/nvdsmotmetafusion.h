/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

/**
 * @file
 * <b>DeepStream tracker metadata fusion API </b>
 *
 * @b Description: This file defines the DeepStream tracker metadata fusion API.
 */

/**
 * @defgroup  ee_NvMOTMetaFuser API
 *
 * Defines the DeepStream tracker metadata fusion API.
 *
 * @ingroup NvDsMetaFuserApi
 * @{
 */

#ifndef _NVMOTRACKERMETAFUSER_H_
#define _NVMOTRACKERMETAFUSER_H_

#include <stdint.h>
#include <time.h>

#include "nvbufsurface.h"
#include "nvds_tracker_meta.h"
#include <vector>

#ifdef __cplusplus
extern "C"
{
#endif


typedef uint64_t NvMOTObjectId;
typedef uint32_t NvMOTFrameNum;
// <object id, <frame_num, offset in reid array>>
typedef std::map<NvMOTObjectId, std::map<NvMOTFrameNum, uint32_t>> NvMOTReidEmbeddingsList;
typedef std::map<NvMOTObjectId, NvMOTObjectId> NvMOTMFObjectIdMap;

typedef struct _NvMOTMFConfig
{
    /** Width and height of the video frame. */
    uint32_t frameWidth;
    uint32_t frameHeight;
    /** Holds the maximum possible length of a video chunk  */
    uint32_t maxVideoChunkLength;
    /** Holds the length of @a customConfigFilePath. */
    uint16_t customConfigFilePathSize;
    /** A pointer to the pathname of the tracker's custom configuration file.
     A null-terminated string. */
    char* customConfigFilePath;
} NvMOTMFConfig;

/**
 * @brief Defines configuration request return codes.
 */
typedef enum
{
    NvMOTMFConfigStatus_OK,
    NvMOTMFConfigStatus_Error,
    NvMOTMFConfigStatus_Invalid,
    NvMOTMFConfigStatus_Unsupported
} NvMOTMFConfigStatus;

/**
 * @brief Holds a tracker's configuration status.
 *
 * Holds the status of a configuration request; includes both summary and
 * per-configuration status.
 */
typedef struct _NvMOTMFConfigResponse
{
    /** Holds the summary status of the entire configuration request. */
    NvMOTMFConfigStatus summaryStatus;
    /** Holds the compute target request status. */
    NvMOTMFConfigStatus computeStatus;
    /** Holds the transform batch configuration request status:
     summary status for all transforms. */
    NvMOTMFConfigStatus transformBatchStatus;
    /** Holds the status of the miscellaneous configurations. */
    NvMOTMFConfigStatus miscConfigStatus;
    /** Holds the status of the custom configurations. */
    NvMOTMFConfigStatus customConfigStatus;
} NvMOTMFConfigResponse;


typedef enum
{
    NvMOTMFStatus_OK,
    NvMOTMFStatus_Error,
    NvMOTMFStatus_Invalid_Path
} NvMOTMFStatus;

/**
 * @brief Tracker misc data.
 */
typedef struct _NvMOTMFTrackerFusedData
{
    /** Holds the history of terminated tracks*/
    NvDsTargetMiscDataStream *pFusedTracksStream;
} NvMOTMFTrackerFusedData;

/**
 * @brief Holds all the input metadata of a video chunk to be processed
 */
typedef struct _NvMOTMFTrackedObjMeta
{
    /** Holds the relative frame number of the first frame in this video chunk w.r.t. complete video */
    uint32_t startFrameNumber;
    /** Holds the duration of the chunk in number of frames */
    uint32_t chunkDuration;
    /** Tracklets per object */
    NvDsTargetMiscDataStream *tracklets;
    /** The whole chunk's reid embeddings in a sigle vector pointed to by "ptr_host"*/
    NvDsReidTensorBatch reidTensorChunk;
    /** Offset of Reid Embeddings per object */
    NvMOTReidEmbeddingsList reidEmbeddingsList;
} NvMOTMFTrackedObjMeta;


/**
 * @brief Holds an opaque context handle.
 */
struct NvMOTMFContext;
typedef struct NvMOTMFContext* NvMOTMFContextHandle;

typedef struct _NvMOTMFQuery
{
    /** Holds maximum number of targets per stream. */
    uint32_t maxTargetsPerStream;
    /** Reid feature size. */
    uint32_t reidFeatureSize;
    /** Reid history size. */
    uint32_t reidHistorySize;
    /** Hold the context handle. */
    NvMOTMFContextHandle contextHandle;
} NvMOTMFQuery;

/**
 * @brief Initializes the fusion context 
 *
 * If successful, the context is configured as specified by @a pConfigIn.
 *
 * @param [in]  pConfigIn       A pointer to to a structure specifying
 *                              the configuration.
 * @param [out] pContextHandle  A pointer to a handle for the fusion context.
 *                              The fusion context is created and owned
 *                              by the fusion module. The returned context handle
 *                              must be included in
 *                              all subsequent calls for the specified stream.
 * @param [out] pConfigResponse A pointer to a structure that describes the
 *                              operation's status.
 * @return  The outcome of the initialization attempt.
 */

NvMOTMFStatus NvMOTMF_Init(NvMOTMFConfig *pConfigIn,
                       NvMOTMFContextHandle *pContextHandle,
                       NvMOTMFConfigResponse *pConfigResponse);

/**
 * @brief Deinitializes fusion context
 *
 * The specified context is retired and may not be used again.
 *
 * @param contextHandle     The handle for the fusion context to be retired.
 */
void NvMOTMF_DeInit(NvMOTMFContextHandle contextHandle);

/**
 * @brief Processes a video chunk.
 *
 * Given a context and a video chunk metadata, processes the metadata chunk.
 * Once processed, each chunk becomes part of the history 
 * and will be used for processing all the future video chunks
 *
 * @param [in]  contextHandle  A context handle obtained from NvMOTInit().
 * @param [in] pVideoChunk     A pointer to the metadata of the chunk to be processed.
 * @param [out] ipOpObjectIdMap Reference to a map to return input object id to output object id mapping
 * @return  The status of chunk processing.
 */
NvMOTMFStatus NvMOTMF_Process(NvMOTMFContextHandle contextHandle,
                          NvMOTMFTrackedObjMeta *pVideoChunk,
                          NvMOTMFObjectIdMap &ipOpObjectIdMap);

/**
 * @brief Retrieve the fused data 
 *
 * Given a context retrieve the fused tracklets 
 * The fused tracklets are returned in "pFusedTracksStream"
 * If eos = false : only the terminated tracklets are returned
 * If eos = true, all the active tracklets are terminated and returned
 * 
 * @param [in] contextHandle The context handle obtained from NvMOTInit()
 * @param [out] pTrackerMiscData Misc data from low level tracker that contains the fused tracklet data in pFusedTracksStream
 * @return Status of the call
 */
NvMOTMFStatus NvMOTMF_RetrieveFusedData(NvMOTMFContextHandle contextHandle,
                          NvMOTMFTrackerFusedData *pTrackerMiscData, bool eos);

/**
 * @brief Query metadata fusion lib capabilities and requirements.
 *
 * Answer query for this metadata fusion lib's capabilities and requirements.
 *
 * @param [out] pQuery                  A pointer to a query structure to be
 *                                      filled by the library.
 * @return  Status of the query.
 */
NvMOTMFStatus NvMOTMF_Query(NvMOTMFQuery *pQuery);


/** @} */ // end of API group

#ifdef __cplusplus
}
#endif

#endif
