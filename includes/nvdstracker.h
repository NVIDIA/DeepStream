/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * <b>DeepStream object tracker API </b>
 *
 * @b Description: This file defines the DeepStream object tracker API.
 */

/**
 * @defgroup  ee_NvMOTracker  Object Tracker API
 *
 * Defines the DeepStream object tracker API.
 *
 * @ingroup NvDsTrackerApi
 * @{
 */

#ifndef _NVMOTRACKER_H_
#define _NVMOTRACKER_H_

#include <stdint.h>
#include <string>
#include <time.h>
#include <vector>

#include "nvbufsurface.h"
#include "nvds_tracker_meta.h"

#ifdef __cplusplus
extern "C"
{
#endif

#define NVMOT_MAX_TRANSFORMS 4

typedef uint64_t NvMOTStreamId;
typedef uint32_t NvMOTFrameNum;
typedef uint16_t NvMOTClassId;

/**
 * @brief Compute target flags.
 *
 * You can select multiple targets; the tracker will optimize across them.
 * Combinations are allowed, e.g. NVTCOMP_GPU or NVTCOMP_PVA
 */
#define NVMOTCOMP_GPU     0x01  /**< Defines the "GPU" compute target flag. */
#define NVMOTCOMP_CPU     0x02  /**< Defines the "CPU" compute target flag. */
#define NVMOTCOMP_PVA     0x04  /**< Defines the "PVA" compute target flag. */
#define NVMOTCOMP_ANY     0xff  /**< Defines a compute target flag for
                                 "any target." */
#define NVMOTCOMP_DEFAULT NVMOTCOMP_ANY
                                /**< Defines the compute target flag for the
                                 default target. */
/** @} */

typedef uint8_t NvMOTCompute;

/**
 * @brief Holds a configuration for batches for an input transform
 *  (a scaling/color conversion).
 */
typedef struct _NvMOTPerTransformBatchConfig
{
    /** Holds the type of buffer. */
    NvBufSurfaceMemType bufferType;
    /** Holds the maximum width of each frame. */
    uint32_t maxWidth;
    /** Holds the maximum height of each frame. */
    uint32_t maxHeight;
    /** Holds the maximum pitch of each buffer. */
    uint32_t maxPitch;
    /** Holds the maximum size of the buffer in bytes. */
    uint32_t maxSize;
} NvMOTPerTransformBatchConfig;


/**
 * @brief Holds miscellaneous configurations.
 */
typedef struct _NvMOTMiscConfig
{
    /** Holds the ID of the GPU to be used. */
    uint32_t gpuId;
    /** Holds a pointer to a callback for logging messages. */
    typedef void (*logMsg) (int logLevel, const char * format, ...);
} NvMOTMiscConfig;

/**
 * @brief Holds a tracker configuration.
 *
 * Holds configuration options for the tracker, applied to the whole context.
 *
 * @note This structure must be deep-copied to be passed to a component that
 * is to use it persistently.
 */
typedef struct _NvMOTConfig
{
    /** Holds the compute target. @see NvMOTCompute. */
    NvMOTCompute computeConfig;
    /** Holds the maximum number of streams in a batch. */
    uint32_t maxStreams;
    /** Holds the max number of surface buffer addresses from buffer pool. */
    uint32_t maxBufSurfAddrSize;
    /** Holds the number of \ref NvMOTPerTransformBatchConfig entries in
     @a perTransformBatchConfig. */
    uint8_t numTransforms;
    /** Holds a pointer to a list of @a numTransform batch configurations,
     one per transform, including type and resolution. */
    NvMOTPerTransformBatchConfig *perTransformBatchConfig;
    /** Holds miscellaneous configurations. */
    NvMOTMiscConfig miscConfig;
    /** Holds the length of @a customConfigFilePath. */
    uint16_t customConfigFilePathSize;
    /** A pointer to the pathname of the tracker's custom configuration file.
     A null-terminated string. */
    char* customConfigFilePath;
} NvMOTConfig;

/**
 * @brief Defines configuration request return codes.
 */
typedef enum
{
    NvMOTConfigStatus_OK,
    NvMOTConfigStatus_Error,
    NvMOTConfigStatus_Invalid,
    NvMOTConfigStatus_Unsupported
} NvMOTConfigStatus;

typedef enum
{
    /** Unsupported batch mode. At least either batch or non-batch should be supported. **/
    NvMOTBatchMode_Error = 0,
    /** Batch processing mode. **/
    NvMOTBatchMode_Batch = 1 << 0,
    /** Non-batch processing mode. **/
    NvMOTBatchMode_NonBatch = 1 << 1,
    /** Max value of the enum. All bits are 1. **/
    NvMOTBatchMode_MaxValue = 3
} NvMOTBatchMode;

/**
 * @brief Holds a tracker's configuration status.
 *
 * Holds the status of a configuration request; includes both summary and
 * per-configuration status.
 */
typedef struct _NvMOTConfigResponse
{
    /** Holds the summary status of the entire configuration request. */
    NvMOTConfigStatus summaryStatus;
    /** Holds the compute target request status. */
    NvMOTConfigStatus computeStatus;
    /** Holds the transform batch configuration request status:
     summary status for all transforms. */
    NvMOTConfigStatus transformBatchStatus;
    /** Holds the status of the miscellaneous configurations. */
    NvMOTConfigStatus miscConfigStatus;
    /** Holds the status of the custom configurations. */
    NvMOTConfigStatus customConfigStatus;
} NvMOTConfigResponse;

/**
 * @brief Defines generic status codes for tracking operations.
 */
typedef enum
{
    NvMOTStatus_OK,
    NvMOTStatus_Error,
    NvMOTStatus_Invalid_Path
} NvMOTStatus;

/**
 * @brief Holds the definition of a rectangle.
 */
typedef struct _NvMOTRect
{
    /** Holds the left edge position of the object bounding box, in pixels. */
    float x;
    /** Holds the top edge position of the object bounding box, in pixels. */
    float y;
    /** Holds the width of the bounding box, in pixels. */
    float width;
    /** Holds the height of the bounding box, in pixels. */
    float height;
} NvMOTRect;

/**
 * @brief Holds the segmentaiton mask of an object.
 */
typedef struct _NvMOTMask
{
    /** Holds the segmentation mask. Size is width*height, in row major. Element range is from 0 to 1. */
    float* data;
    /** Width of the segmentation mask. */
    uint32_t width;
    /** Height of the segmentation mask. */
    uint32_t height;
    /** Allocated size. */
    uint32_t allocatedSize;
} NvMOTMask;

/**
 * @brief Holds information about an object to be tracked.
 *
 * NvMOT creates an instance of this structure for each tracked object.
 */
typedef struct _NvMOTObjToTrack
{
    /** Holds the class of the object. */
    NvMOTClassId classId;
    /** Holds the bounding box of the object. */
    NvMOTRect bbox;
    /** Holds the detection confidence of the object. */
    float confidence;
    /** Holds a Boolean which is true if NvMOT is to track this object. */
    bool doTracking;
    /** Holds a pointer to client data associated with the object. */
    void *pPreservedData;
} NvMOTObjToTrack;

/**
 * @brief Holds a list of objects.
 */
typedef struct _NvMOTObjToTrackList
{
    /** Holds a Boolean which is true if detection was done on this frame
     even if the list of objects to track is empty. */
    bool detectionDone;
    /** Holds a pointer to a list or array of object information blocks. */
    NvMOTObjToTrack* list;
    /** Holds the number of blocks allocated for the list. */
    uint32_t numAllocated;
    /** Holds the number of populated blocks in the list. */
    uint32_t numFilled;
} NvMOTObjToTrackList;

/**
 * @brief Holds a frame containing the image and objects to be tracked.
 *
 * @note @a numBuffers is supposed to be less than or equal to
 *  @a numTransforms in @ref NvMOTConfig.
 * @note The metadata in the NvBufSurfaceParams structures
 *  which @a bufferList points to must be checked with the parameters
 *  specified in @a perTransformBatchConfig in NvMOTConfig.
 */
typedef struct _NvMOTFrame
{
    /** Holds the stream ID of the stream source for this frame. */
    NvMOTStreamId streamID;
    /** Holds the index of the stream in the sequence of streams */
    /** Should be constant throughout the tenure of the stream */
    /** A number ranging from 0 to (maxStreams - 1) */
    uint32_t seq_index;
    /** Holds the sequential frame number that identifies the frame
     within the stream. */
    NvMOTFrameNum frameNum;
    /** Holds the width of the original source frame. */
    uint32_t srcFrameWidth;
    /** Holds the height of the original source frame. */
    uint32_t srcFrameHeight;
    /** Holds the timestamp of the frame at the time of capture. */
    time_t timeStamp;
    /** Holds a Boolean which is true if the timestamp value is properly
     populated. */
    bool timeStampValid;
    /** Holds a Boolean which is true if objects in this frame are to be
     tracked. */
    bool doTracking;
    /** Holds a Boolean which is true to reset tracking for the stream. */
    bool reset;
    /** Holds the number of entries in @a bufferList. */
    uint8_t numBuffers;
    /** Holds a pointer to an array of pointers to buffer parameter
     structures. */
    NvBufSurfaceParams** bufferList;
    /** Holds a list of objects in this frame which are to be tracked.
     Boundary boxes are scaled for the first buffer configuration. */
    NvMOTObjToTrackList objectsIn;
} NvMOTFrame;

/**
 * @brief Holds information about each tracked object.
 */
typedef struct _NvMOTTrackedObj
{
    /** Holds the class ID of the object to be tracked. */
    NvMOTClassId classId;
    /** Holds a unique ID for the object, assigned by the tracker. */
    uint64_t trackingId;
    /** Holds the bounding box. */
    NvMOTRect bbox;
    /** Holds the tracking confidence of the object. */
    float confidence;
    /** Holds the track length in frames. */
    uint32_t age;
    /** Holds a pointer to the associated input object, if there is one. */
    NvMOTObjToTrack *associatedObjectIn;
    /** Each target’s reid vector information.*/
    NvDsObjReid reid;
    /** Object visibility. */
    float visibility;
    /** Foot location in frame coordinates. */
    float ptImgFeet[2];
    /** Foot location in 3D coordinates. */
    float ptWorldFeet[2];
    /** Convex hull information projected on frame. */
    NvDsObjConvexHull convexHull;
    /** Each object's segmentation mask. */
    NvMOTMask mask;
    /** 3D bbox in world co-ordinates. */
    NvDsObj3DBbox bbox3DWorld;
    /** 3D bbox in frame co-ordinates. */
    /** 8 corners of bbox */
    float bbox3DImg[8][2];
    /** Reserved custom data field. */
    uint8_t reserved[128];
} NvMOTTrackedObj;

/**
 * @brief Holds a list of tracked objects.
 */
typedef struct _NvMOTTrackedObjList
{
    /** Holds the stream ID of the stream associated with objects in the list.*/
    NvMOTStreamId streamID;
    /** Holds the frame number for objects in the list. */
    NvMOTFrameNum frameNum;
    /** Holds a Boolean which is true if this entry in the batch is valid. */
    bool valid;
    /** Holds a pointer to a list or array of object information blocks. */
    NvMOTTrackedObj* list;
    /** Holds the number of blocks allocated for the list. */
    uint32_t numAllocated;
    /** Holds the number of populated blocks in the list. */
    uint32_t numFilled;
} NvMOTTrackedObjList;

/**
 * @brief Holds a batch of lists of tracked objects.
 */
typedef struct _NvMOTTrackedObjBatch
{
    /** Holds a pointer to an array of object lists. */
    NvMOTTrackedObjList *list;
    /** Holds the number of blocks allocated for the list. */
    uint32_t numAllocated;
    /** Holds the number of filled blocks in the list. */
    uint32_t numFilled;
    /** The whole batch’s reid tensor to fill by low level tracker. */
    NvDsReidTensorBatch* pReidTensorBatch;
} NvMOTTrackedObjBatch;

/**
 * @brief Tracker misc data.
 */
typedef struct _NvMOTTrackerMiscData
{
    /** Holds past frame data of current batch. */
    NvDsTargetMiscDataBatch *pPastFrameObjBatch;

    /** Holds the history of terminated tracks*/
    NvDsTargetMiscDataBatch *pTerminatedTrackBatch;

    /** Holds the frame info of shadow tracks*/
    NvDsTargetMiscDataBatch *pShadowTrackBatch;

} NvMOTTrackerMiscData;

/**
 * @brief Holds parameters for processing a batch.
 *
 * @see NvMOTProcessFrame.
 */
typedef struct _NvMOTProcessParams
{
    uint32_t numFrames;     /**< Holds the number of frames in the batch. */
    NvMOTFrame *frameList;  /**< Holds a pointer to an array of frame data. */
} NvMOTProcessParams;

/**
 * @brief Holds an opaque context handle.
 */
struct NvMOTContext;
typedef struct NvMOTContext* NvMOTContextHandle;

typedef struct _NvMOTQuery
{
    /** Holds flags for supported compute targets. @see NvMOTCompute. */
    NvMOTCompute computeConfig;
    /** Holds the number of \ref NvMOTPerTransformBatchConfig entries
     in perTransformBatchConfig. */
    uint8_t numTransforms;
    /** Holds the color formats for input buffers; a required value. */
    NvBufSurfaceColorFormat colorFormats[1];
    /** Holds the preferred memory type for input buffers. */
    NvBufSurfaceMemType memType;
    /** Holds maximum number of targets per stream. */
    uint32_t maxTargetsPerStream;
    /** Holds maximum shadow tracking age per stream. */
    uint32_t maxShadowTrackingAge;
    /** Whether to output ReID tensor in user meta. */
    bool outputReidTensor;
    /** Reid feature size. */
    uint32_t reidFeatureSize;
    /** Whether to output target trajectories in user meta. */
    bool outputTrajectory;
    /** Whether to output visibility in user meta. */
    bool outputVisibility;
    /** Whether to output foot location in user meta. */
    bool outputFootLocation;
    /** Whether to output convex hull in user meta. */
    bool outputConvexHull;
    /** Holdes maximum number of points in a convex hull. */
    uint32_t maxConvexHullSize;
    /** Holds a Boolean which is true if outputing past frame is supported. */
    bool supportPastFrame;
    /** Holds flags for whether batch or none batch mode is supported. */
    NvMOTBatchMode batchMode;
    /** Whether to output terminted Tacks info in user meta. */
    bool outputTerminatedTracks;
    /** maximum frame of history to save per terminated track. */
    uint32_t maxTrajectoryBufferLength;
    /** Whether to output Shadow Tracks info in user meta. */
    bool outputShadowTracks;
    /** Hold the context handle. */
    NvMOTContextHandle contextHandle;
} NvMOTQuery;

/**
 * @brief Initializes a tracking context for a batch of one or more image
 *  streams.
 *
 * If successful, the context is configured as specified by @a pConfigIn.
 *
 * @param [in]  pConfigIn       A pointer to to a structure specifying
 *                              the configuration.
 * @param [out] pContextHandle  A pointer to a handle for the stream context.
 *                              The stream context is created and owned
 *                              by the tracker. The returned context handle
 *                              must be included in
 *                              all subsequent calls for the specified stream.
 * @param [out] pConfigResponse A pointer to a structure that describes the
 *                              operation's status.
 * @return  The outcome of the initialization attempt.
 */
NvMOTStatus NvMOT_Init(NvMOTConfig *pConfigIn,
                       NvMOTContextHandle *pContextHandle,
                       NvMOTConfigResponse *pConfigResponse);

/**
 * @brief Deinitializes a stream context
 *
 * The specified context is retired and may not be used again.
 *
 * @param contextHandle     The handle for the stream context to be retired.
 */
void NvMOT_DeInit(NvMOTContextHandle contextHandle);

/**
 * @brief Processes a batch.
 *
 * Given a context and a batch of frames, processes the batch as the current
 * frames in their respective streams. Once processed, each frame becomes part
 * of the history and the previous frame in its stream.
 *
 * @param [in]  contextHandle  A context handle obtained from NvMOT_Init().
 * @param [in]  pParams        A pointer to parameters for the batch
 *                             to be processed.
 * @param [out] pTrackedObjectsBatch
 *                             A pointer to a batch of lists of tracked object
 *                             slots to be filled by the tracker. The batch is
 *                             allocated by the client. Bounding boxes are
 *                             scaled to the resolution of the first input
 *                             image transform buffer.
 * @return  The status of batch processing.
 */
NvMOTStatus NvMOT_Process(NvMOTContextHandle contextHandle,
                          NvMOTProcessParams *pParams,
                          NvMOTTrackedObjBatch *pTrackedObjectsBatch);

/**
 * @brief Process the past-frame data in the low-level tracker lib and retrieve
 *
 * Given a context and batch of frame(s), process the past-frame
 * data of each tracked object stored in the low-level tracker lib,
 * put it into the past-frame data strcture, and retrieve it
 *
 * @param [in] pContext The context handle obtained from NvMOTInit()
 * @param [in] pParams Pointer to parameters for the batch of frames with the available stream ID
 * @param [out] pTrackerMiscData Misc data from low level tracker to store in user meta
 * @return Status of batch processing
 */
NvMOTStatus NvMOT_RetrieveMiscData(NvMOTContextHandle contextHandle,
                          NvMOTProcessParams *pParams,
                          NvMOTTrackerMiscData *pTrackerMiscData);
/**
 * @brief Query tracker lib capabilities and requirements.
 *
 * Answer query for this tracker lib's capabilities and requirements.
 * The tracker's custom config file is provided for optional consultation.
 *
 * @param [in]  customConfigFilePathSize Length of the custom configuration
 *                                      file's pathname.
 * @param [in]  pCustomConfigFilePath   A pointer to the custom configuration
 *                                      file's pathname.
 * @param [out] pQuery                  A pointer to a query structure to be
 *                                      filled by the tracker.
 * @return  Status of the query.
 */
NvMOTStatus NvMOT_Query(uint16_t customConfigFilePathSize, char* pCustomConfigFilePath, NvMOTQuery *pQuery);


/**
 * @brief Removes streams from a batch.
 *
 * An optional function used in batch processing mode only. It notifies
 * the tracker library that a stream has been removed,
 * and will not be present in future batches. Any per-stream resources
 * associated with the removed stream may be freed at this time.
 *
 * This function may be called only when all processing is quiesced.
 *
 * @param [in] contextHandle    The context handle obtained from NvMOTInit().
 * @param [in] streamIdMask     A mask for finding streams to remove.
 *                              The function reports removal of all streams
 *                              where (streamId & streamIdMask) == streamIdMask.
 * @return  Status of stream removal.
 */
NvMOTStatus NvMOT_RemoveStreams(NvMOTContextHandle contextHandle, NvMOTStreamId streamIdMask);


/**
 * @brief Updates Dynamic Parameters for the low level tracker
 *
 * An optional function
 *
 * This function may be called only when all processing is quiesced.
 *
 * @param [in] contextHandle    The context handle obtained from NvMOTInit().
 * @param [in] configStr        Stringified config for the tracker.
 *
 * @return  Status of config update
 */
NvMOTStatus NvMOT_UpdateParams(NvMOTContextHandle contextHandle, const std::string &configStr );

/** @} */ // end of API group

#ifdef __cplusplus
}
#endif

#endif
