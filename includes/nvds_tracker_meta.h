/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * <b>Defines Tracker Metadata</b>
 */
/**
 * @defgroup  ee_tracker_group Tracker Metadata
 *
 * Specifies metadata concerning tracking.
 *
 * @ingroup NvDsMetaApi
 * @{
 */
#ifndef _NVDS_TRACKER_META_H_
#define _NVDS_TRACKER_META_H_

#include <stdint.h>
#include "nvll_osd_struct.h"

#define MAX_LABEL_SIZE 128

#ifdef __cplusplus
extern "C"
{
#endif
typedef enum { EMPTY = 0,     ///\ The corresponding tracker is no longer is in use
               ACTIVE = 1,    ///\ tracking is being confirmed by detectors and actively reporting outputs
               INACTIVE = 2,  ///\ tracking is not confirmed or w/ low confidence, so not reporting the outputs, but keep tracking (i.e., Shadow Tracking)
               TENTATIVE = 3, ///\ tracking is just started and in a probational period. Waiting to become ACTIVE
               PROJECTED = 4,   ///\ tracking is completed, the tracklet is about to be archived, and some projected points are appended at the end of the traclet for re-assoc.
               QUASIACTIVE = 5  ///\ tracking is confirmed by peer cameras, but not by detectors
             }TRACKER_STATE;

/**
 * A single frame of misc data for a given Target
 */
 typedef struct _NvDsTargetMiscDataFrame
{
    /** Frame number. */
    uint32_t frameNum;
    /** Bounding box. */
    NvOSD_RectParams tBbox;
    /** Tracking confidence. */
    float confidence;
    /** Tracking age. */
    uint32_t age;
    /** Curret Tracker State */
	TRACKER_STATE trackerState;
    /**bbox visibility with respect to the image border */
    float visibility;
    /** Feet location of the object on the world ground. */
    float ptWorldFeet[2];

} NvDsTargetMiscDataFrame;


/**
 * All misc data output for a single target
 */
typedef struct _NvDsTargetMiscDataObject
{
    /** Pointer to a list per-frame information of the target. */
    NvDsTargetMiscDataFrame *list;
    /** Number of frames this target appreared in the past. */
    uint32_t numObj;
    /** Maximum number of frames allocated. */
    uint32_t numAllocated;
    /** Target tracking id. */
    uint64_t uniqueId;
    /** Target class id. */
    uint16_t classId;
    /** An array of the string describing the target class. */
    char objLabel[MAX_LABEL_SIZE];

} NvDsTargetMiscDataObject;

/**
 * All misc targets data for a given stream
 */
typedef struct _NvDsTargetMiscDataStream
{
    /** Pointer to targets inside this stream. */
    NvDsTargetMiscDataObject *list;
    /** Stream id the same as frame_meta->pad_index. */
    uint32_t streamID;
    /** Stream id used inside tracker plugin. */
    uint64_t surfaceStreamID;
    /** Maximum number of objects allocated. */
    uint32_t numAllocated;
    /** Number of objects in this frame. */
    uint32_t numFilled;
} NvDsTargetMiscDataStream;

/**
 * Batch of all streams of a given target misc output
 */
typedef struct _NvDsTargetMiscDataBatch
{
    /** Pointer to array of stream lists. */
    NvDsTargetMiscDataStream *list;
    /** Number of blocks allocated for the list. */
    uint32_t numAllocated;
    /** Number of filled blocks in the list. */
    uint32_t numFilled;
    /** Pointer to internal buffer pool needed by gst pipelines to return buffers. */
    void* priv_data;
} NvDsTargetMiscDataBatch;

/**
 * ReID tensor of the batch.
 */
typedef struct _NvDsReidTensorBatch
{
    /** Each target's ReID vector length. */
    uint32_t featureSize;
    /** Number of reid vectors in the batch. */
	uint32_t numFilled;
    /** ReID vector on CPU. */
    float* ptr_host;
    /** ReID vector on GPU. */
    float* ptr_dev;
    /** Pointer to internal buffer pool needed by gst pipelines to return buffers.*/
    void* priv_data;
} NvDsReidTensorBatch;


/**
 * Batch of trajectory data in all streams.
 */
typedef NvDsTargetMiscDataBatch NvDsTrajectoryBatch;

/**
 * @brief Holds convex hull information
 */
typedef struct _NvDsObjConvexHull
{
    /** Holds a pointer to a list or array of object information blocks. */
    int* list;
    /** Holds the number of blocks allocated for the list. */
    uint32_t numPointsAllocated;
    /** Holds the number of points in the list. */
    uint32_t numPoints;
} NvDsObjConvexHull;

/**
 * @brief Holds Reid Vector information for an object
 */
typedef struct _NvDsObjReid
{
    /** ReID vector length. */
    uint32_t featureSize;
    /** ReID vector pointer on CPU. */
    float* ptr_host;
    /** ReID vector pointer on GPU. */
    float* ptr_dev;
} NvDsObjReid;

/**
 * @brief Holds 3D bbox information for an object
 */
typedef struct _NvDsObj3DBbox
{
    /** Centroid of the 3D bbox */
    float xCentre;
    float yCentre;
    float zCentre;
    /* 3D bbox dimensions */
    float xLen;
    float yLen;
    float zLen;
    /* 3D bbox rotation */
    float xRot;
    float yRot;
    float zRot;
    /* 3D bbox velocity */
    float xVel;
    float yVel;
    float zVel;
} NvDsObj3DBbox;

#ifdef __cplusplus
}
#endif

#endif

/** @} */
