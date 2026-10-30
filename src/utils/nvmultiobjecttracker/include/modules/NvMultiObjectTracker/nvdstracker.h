/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <chrono>
#include <vector>

#include "nvbufsurface.h"
#include "nvds_tracker_meta.h"

#ifdef __cplusplus
#ifdef _WIN32
using uint = unsigned int;
#endif
extern "C"
{
#endif

#define NVMOT_MAX_TRANSFORMS 4

typedef uint64_t NvMOTTimeStamp;
typedef uint64_t NvMOTStreamId;
typedef uint32_t NvMOTFrameNum;
typedef uint16_t NvMOTClassId;
typedef NvMOTTimeStamp TimeStamp;

/** @brief Return wall-clock time in nanoseconds since the Unix epoch. */
static inline TimeStamp NvDsCurrentTimestampNs(void)
{
    return static_cast<TimeStamp>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()
        ).count()
    );
}

/** @brief Holds the stream timestamp, host-local timestamp, and frame index. */
typedef struct _NvMOTTimeMeta
{
    NvMOTFrameNum frameNum; //!< Holds the logical frame index from @ref NvMOTFrame::frameNum.
    NvMOTTimeStamp frameTimeStamp; //!< Holds the stream or media timestamp in nanoseconds when valid.
    TimeStamp localTimestamp; //!< Holds the host wall-clock timestamp in nanoseconds when prepared.
    bool frameTimeValid; //!< Indicates whether @a frameTimeStamp is populated.
} NvMOTTimeMeta;

/**
 * @brief Compute target flags.
 *
 * You can select multiple targets; the tracker will optimize across them.
 * Combinations are allowed, e.g. NVTCOMP_GPU or NVTCOMP_PVA
 */
#define NVMOTCOMP_GPU 0x01 //!< Defines the "GPU" compute target flag.
#define NVMOTCOMP_CPU 0x02 //!< Defines the "CPU" compute target flag.
#define NVMOTCOMP_PVA 0x04 //!< Defines the "PVA" compute target flag.
#define NVMOTCOMP_ANY 0xff //!< Defines a compute target flag for "any target."
#define NVMOTCOMP_DEFAULT NVMOTCOMP_ANY //!< Defines the compute target flag for the default target.
/** @} */

typedef uint8_t NvMOTCompute;

/**
 * @brief Holds a configuration for batches for an input transform (a scaling/color conversion).
 */
typedef struct _NvMOTPerTransformBatchConfig
{
    NvBufSurfaceMemType bufferType; //!< Holds the type of buffer.
    uint32_t maxWidth; //!< Holds the maximum width of each frame.
    uint32_t maxHeight; //!< Holds the maximum height of each frame.
    uint32_t maxPitch; //!< Holds the maximum pitch of each buffer.
    uint32_t maxSize; //!< Holds the maximum size of the buffer in bytes.
} NvMOTPerTransformBatchConfig;

/**
 * @brief Holds miscellaneous configurations.
 */
typedef struct _NvMOTMiscConfig
{
    uint32_t gpuId; //!< Holds the ID of the GPU to be used.
    typedef void (*logMsg) (int logLevel, const char * format, ...); //!< Holds a pointer to a callback for logging messages.
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
    NvMOTCompute computeConfig; //!< Holds the compute target. @see NvMOTCompute.
    uint32_t maxStreams; //!< Holds the maximum number of streams in a batch.
    uint32_t maxBufSurfAddrSize; //!< Holds the max number of surface buffer addresses from buffer pool.
    uint8_t numTransforms; //!< Holds the number of entries in @a perTransformBatchConfig.
    NvMOTPerTransformBatchConfig *perTransformBatchConfig; //!< Holds one batch configuration for each transform.
    NvMOTMiscConfig miscConfig; //!< Holds miscellaneous configurations.
    uint16_t customConfigFilePathSize; //!< Holds the length of @a customConfigFilePath.
    char* customConfigFilePath; //!< Holds the null-terminated path to the tracker's custom configuration file.
} NvMOTConfig;

/**
 * @brief Defines configuration request return codes.
 */
typedef enum
{
	NvMOTConfigStatus_OK, //!< Configuration was accepted.
	NvMOTConfigStatus_Error, //!< Configuration processing failed.
	NvMOTConfigStatus_Invalid, //!< Configuration contained an invalid value.
	NvMOTConfigStatus_Unsupported //!< Configuration requested an unsupported option.
} NvMOTConfigStatus;

/** @brief Defines supported tracker batch-processing modes. */
typedef enum
{
	NvMOTBatchMode_Error = 0, //!< Indicates that neither batch nor non-batch processing is supported.
	NvMOTBatchMode_Batch = 1 << 0, //!< Enables batch processing.
	NvMOTBatchMode_NonBatch = 1 << 1, //!< Enables non-batch processing.
	NvMOTBatchMode_MaxValue = 3 //!< Represents the maximum bit-mask value.
} NvMOTBatchMode;

/**
 * @brief Holds a tracker's configuration status.
 *
 * Holds the status of a configuration request; includes both summary and
 * per-configuration status.
 */
typedef struct _NvMOTConfigResponse
{
    NvMOTConfigStatus summaryStatus; //!< Holds the summary status of the entire configuration request.
    NvMOTConfigStatus computeStatus; //!< Holds the compute target request status.
    NvMOTConfigStatus transformBatchStatus; //!< Holds the summary status for all transform batch configurations.
    NvMOTConfigStatus miscConfigStatus; //!< Holds the status of the miscellaneous configurations.
    NvMOTConfigStatus customConfigStatus; //!< Holds the status of the custom configurations.
} NvMOTConfigResponse;

/**
 * @brief Defines generic status codes for tracking operations.
 */
typedef enum
{
	NvMOTStatus_OK, //!< Operation completed successfully.
	NvMOTStatus_Error, //!< Operation failed.
	NvMOTStatus_Invalid_Path //!< Operation received an invalid path.
} NvMOTStatus;

/**
 * @brief Holds the definition of a rectangle.
 */
typedef struct _NvMOTRect
{
    float x; //!< Holds the left edge position of the object bounding box, in pixels.
    float y; //!< Holds the top edge position of the object bounding box, in pixels.
    float width; //!< Holds the width of the bounding box, in pixels.
    float height; //!< Holds the height of the bounding box, in pixels.
} NvMOTRect;

/**
 * @brief Holds the segmentation mask of an object.
 */
typedef struct _NvMOTMask
{
    float* data; //!< Holds the row-major segmentation mask with values from zero to one.
    uint32_t width; //!< Width of the segmentation mask.
    uint32_t height; //!< Height of the segmentation mask.
    uint32_t allocatedSize; //!< Allocated size.
} NvMOTMask;

/**
 * @brief Holds information about an object to be tracked.
 *
 * NvMOT creates an instance of this structure for each tracked object.
 */
typedef struct _NvMOTObjToTrack
{
    NvMOTClassId classId; //!< Holds the class of the object.
    NvMOTRect bbox; //!< Holds the bounding box of the object.
    float confidence; //!< Holds the detection confidence of the object.
    bool doTracking; //!< Holds a Boolean which is true if NvMOT is to track this object.
    void *pPreservedData; //!< Holds a pointer to client data associated with the object.
} NvMOTObjToTrack;

/**
 * @brief Holds a list of objects.
 */
typedef struct _NvMOTObjToTrackList
{
    bool detectionDone; //!< Indicates whether detection ran on the frame, including when the object list is empty.
    NvMOTObjToTrack* list; //!< Holds a pointer to a list or array of object information blocks.
    uint32_t numAllocated; //!< Holds the number of blocks allocated for the list.
    uint32_t numFilled; //!< Holds the number of populated blocks in the list.
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
    NvMOTStreamId streamID; //!< Holds the stream ID of the stream source for this frame.
    uint32_t seq_index; //!< Holds the stable stream index from zero through maxStreams minus one.
    NvMOTFrameNum frameNum; //!< Holds the sequential frame number within the stream.
    NvMOTTimeMeta timeMeta; //!< Holds the stream timestamp, host-local receive time, and frame index.
    uint32_t srcFrameWidth; //!< Holds the width of the original source frame.
    uint32_t srcFrameHeight; //!< Holds the height of the original source frame.
    bool doTracking; //!< Holds a Boolean which is true if objects in this frame are to be tracked.
    bool reset; //!< Holds a Boolean which is true to reset tracking for the stream.
    uint8_t numBuffers; //!< Holds the number of entries in @a bufferList.
    NvBufSurfaceParams** bufferList; //!< Holds an array of pointers to buffer parameter structures.
    NvMOTObjToTrackList objectsIn; //!< Holds tracked inputs with bounding boxes scaled for the first buffer configuration.
    const char* sensorName; //!< Holds the stream source sensor name or null when unavailable.
} NvMOTFrame;

/**
 * @brief Holds information about each tracked object.
 */
typedef struct _NvMOTTrackedObj
{
    NvMOTClassId classId; //!< Holds the class ID of the object to be tracked.
    uint64_t trackingId; //!< Holds a unique ID for the object, assigned by the tracker.
    NvMOTRect bbox; //!< Holds the bounding box.
    float confidence; //!< Holds the tracking confidence of the object.
    uint32_t age; //!< Holds the track length in frames.
    NvMOTObjToTrack *associatedObjectIn; //!< Holds a pointer to the associated input object, if there is one.
    NvDsObjReid reid; //!< Holds the target's ReID vector information.
    float visibility; //!< Object visibility.
    float ptImgFeet[2]; //!< Foot location in frame coordinates.
    float ptWorldFeet[2]; //!< Foot location in 3D coordinates.
    NvDsObjConvexHull convexHull; //!< Convex hull information projected on frame.
    NvMOTMask mask; //!< Each object's segmentation mask.
    NvDsObj3DBbox bbox3DWorld; //!< 3D bbox in world co-ordinates.
    float bbox3DImg[8][2]; //!< Holds the eight corners of the 3D bounding box in frame coordinates.
    uint8_t reserved[128]; //!< Reserved custom data field.
} NvMOTTrackedObj;

/**
 * @brief Holds a list of tracked objects.
 */
typedef struct _NvMOTTrackedObjList
{
    NvMOTStreamId streamID; //!< Holds the stream ID associated with the objects in the list.
    NvMOTFrameNum frameNum; //!< Holds the frame number for objects in the list.
    bool valid; //!< Holds a Boolean which is true if this entry in the batch is valid.
    NvMOTTrackedObj* list; //!< Holds a pointer to a list or array of object information blocks.
    uint32_t numAllocated; //!< Holds the number of blocks allocated for the list.
    uint32_t numFilled; //!< Holds the number of populated blocks in the list.
} NvMOTTrackedObjList;

/**
 * @brief Holds a batch of lists of tracked objects.
 */
typedef struct _NvMOTTrackedObjBatch
{
    NvMOTTrackedObjList *list; //!< Holds a pointer to an array of object lists.
    uint32_t numAllocated; //!< Holds the number of blocks allocated for the list.
    uint32_t numFilled; //!< Holds the number of filled blocks in the list.
    NvDsReidTensorBatch* pReidTensorBatch; //!< The whole batch’s reid tensor to fill by low level tracker.
} NvMOTTrackedObjBatch;

/**
 * @brief Tracker misc data.
 */
typedef struct _NvMOTTrackerMiscData
{
    NvDsTargetMiscDataBatch *pPastFrameObjBatch; //!< Holds past frame data of current batch.
    NvDsTargetMiscDataBatch *pTerminatedTrackBatch; //!< Holds the history of terminated tracks.
    NvDsTargetMiscDataBatch *pShadowTrackBatch; //!< Holds frame information for shadow tracks.
} NvMOTTrackerMiscData;

/**
 * @brief Holds parameters for processing a batch.
 *
 * @see NvMOTProcessFrame.
 */
typedef struct _NvMOTProcessParams
{
    uint32_t numFrames; //!< Holds the number of frames in the batch.
    NvMOTFrame *frameList; //!< Holds a pointer to an array of frame data.
} NvMOTProcessParams;

/**
 * @brief Holds an opaque context handle.
 */
struct NvMOTContext;
typedef struct NvMOTContext* NvMOTContextHandle;

/** @brief Holds the capabilities and requirements reported by the tracker library. */
typedef struct _NvMOTQuery
{
    NvMOTCompute computeConfig; //!< Holds flags for supported compute targets. @see NvMOTCompute.
    uint8_t numTransforms; //!< Holds the number of entries in @a perTransformBatchConfig.
    NvBufSurfaceColorFormat colorFormats[1]; //!< Holds the color formats for input buffers; a required value.
    NvBufSurfaceMemType memType; //!< Holds the preferred memory type for input buffers.
    uint32_t maxTargetsPerStream; //!< Holds maximum number of targets per stream.
    uint32_t maxShadowTrackingAge; //!< Holds maximum shadow tracking age per stream.
    bool outputReidTensor; //!< Indicates whether to output the ReID tensor in user metadata.
    uint32_t reidFeatureSize; //!< Holds the ReID feature size.
    bool outputTrajectory; //!< Indicates whether to output target trajectories in user metadata.
    bool outputVisibility; //!< Indicates whether to output visibility in user metadata.
    bool outputFootLocation; //!< Indicates whether to output foot locations in user metadata.
    bool outputConvexHull; //!< Indicates whether to output convex hulls in user metadata.
    uint32_t maxConvexHullSize; //!< Holds the maximum number of points in a convex hull.
    bool supportPastFrame; //!< Indicates whether past-frame output is supported.
    NvMOTBatchMode batchMode; //!< Holds flags for supported batch-processing modes.
    bool outputTerminatedTracks; //!< Indicates whether to output terminated-track information in user metadata.
    uint32_t maxTrajectoryBufferLength; //!< Holds the maximum number of history frames saved per terminated track.
    bool outputShadowTracks; //!< Indicates whether to output shadow-track information in user metadata.
    NvMOTContextHandle contextHandle; //!< Holds the context handle.
} NvMOTQuery;

/**
 * @brief Initializes a tracking context for a batch of one or more image streams.
 *
 * If successful, the context is configured as specified by @a pConfigIn.
 *
 * @param[in] pConfigIn A pointer to a structure specifying the configuration.
 * @param[in,out] pContextHandle A pointer whose existing handle is retired and whose replacement handle is stored for later calls.
 * @param[out] pConfigResponse A pointer to a structure that describes the operation's status.
 * @return The outcome of the initialization attempt.
 */
NvMOTStatus NvMOT_Init(
    NvMOTConfig *pConfigIn, NvMOTContextHandle *pContextHandle, NvMOTConfigResponse *pConfigResponse
);

/**
 * @brief Deinitialize a stream context.
 * @param[in] contextHandle The handle for the stream context to retire.
 */
void NvMOT_DeInit(NvMOTContextHandle contextHandle);

/**
 * @brief Processes a batch.
 * @param[in] contextHandle A context handle obtained from NvMOT_Init().
 * @param[in] pParams A pointer to parameters for the batch to be processed.
 * @param[out] pTrackedObjectsBatch A pointer to client-allocated tracked-object slots that receive bounding boxes scaled to the first input transform.
 * @return The status of batch processing.
 */
NvMOTStatus NvMOT_Process(
    NvMOTContextHandle contextHandle, NvMOTProcessParams *pParams, NvMOTTrackedObjBatch *pTrackedObjectsBatch
);

/**
 * @brief Retrieve past-frame, terminated-track, and shadow-track data from the low-level tracker.
 * @param[in] contextHandle The context handle obtained from NvMOT_Init().
 * @param[in] pParams A pointer to parameters for the batch of frames with the available stream ID.
 * @param[out] pTrackerMiscData Miscellaneous data from the low-level tracker to store in user metadata.
 * @return The status of the retrieval.
 */
NvMOTStatus NvMOT_RetrieveMiscData(
    NvMOTContextHandle contextHandle, NvMOTProcessParams *pParams, NvMOTTrackerMiscData *pTrackerMiscData
);

/**
 * @brief Query tracker library capabilities and requirements.
 * @param[in] customConfigFilePathSize Length of the custom configuration file's pathname.
 * @param[in] pCustomConfigFilePath A pointer to the custom configuration file's pathname.
 * @param[out] pQuery A pointer to a query structure to be filled by the tracker.
 * @return Status of the query.
 */
NvMOTStatus NvMOT_Query(uint16_t customConfigFilePathSize, char* pCustomConfigFilePath, NvMOTQuery *pQuery);

/**
 * @brief Removes streams from a batch.
 * @param[in] contextHandle The context handle obtained from NvMOTInit().
 * @param[in] streamIdMask A mask that selects streams satisfying (streamId & streamIdMask) == streamIdMask.
 * @return Status of stream removal.
 */
NvMOTStatus NvMOT_RemoveStreams(NvMOTContextHandle contextHandle, NvMOTStreamId streamIdMask);

/**
 * @brief Update dynamic parameters for the low-level tracker.
 * @param[in] contextHandle The context handle obtained from NvMOTInit().
 * @param[in] configStr Stringified configuration for the tracker.
 * @return Status of the configuration update.
 */
NvMOTStatus NvMOT_UpdateParams(NvMOTContextHandle contextHandle, const std::string &configStr);

/** @} */ // end of API group

#ifdef __cplusplus
}
#endif

#endif // _NVMOTRACKER_H_
