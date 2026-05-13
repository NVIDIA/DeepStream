/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2019 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @file nvdsinfer_dbscan.h
 * <b>NVIDIA DeepStream DBScan based Object Clustering API </b>
 *
 * @b Description: This file defines the API for the DBScan-based object
 * clustering algorithm.
 */

/**
 * @defgroup  ee_dbscan  DBScan Based Object Clustering API
 *
 * Defines the API for DBScan-based object clustering.
 *
 * @ingroup NvDsInferApi
 * @{
 */

#ifndef __NVDSINFER_DBSCAN_H__
#define __NVDSINFER_DBSCAN_H__

#include <stddef.h>
#include <stdint.h>

#include <nvdsinfer.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Holds an opaque structure for the DBScan object clustering context. */
struct NvDsInferDBScan;

/** Holds an opaque DBScan clustering context handle. */
typedef struct NvDsInferDBScan *NvDsInferDBScanHandle;

/** Holds object clustering parameters required by DBSCAN. */
typedef struct
{
    float eps;
    uint32_t minBoxes;
    /** Holds a Boolean; true enables the area-to-hit ratio (ATHR) filter.
     The ATHR is calculated as: ATHR = sqrt(clusterArea) / nObjectsInCluster. */
    int enableATHRFilter;
    /** Holds the area-to-hit ratio threshold. */
    float thresholdATHR;
    /** Holds the sum of neighborhood confidence thresholds. */
    float minScore;
} NvDsInferDBScanClusteringParams;

/**
 * Creates a new DBScan object clustering context.
 *
 * @return  A handle to the created context.
 */
NvDsInferDBScanHandle NvDsInferDBScanCreate();

/**
 * Destroys a DBScan object clustering context.
 *
 * @param[in] handle    The handle to the context to be destroyed.
 */
void NvDsInferDBScanDestroy(NvDsInferDBScanHandle handle);

/**
 * Clusters an array of objects in place using specified clustering parameters.
 *
 * @param[in]     handle        A handle to the context be used for clustering.
 * @param[in]     params        A pointer to a clustering parameter structure.
 * @param[in,out] objects       A pointer to an array of objects to be
 *                              clustered. The function places the clustered
 *                              objects in the same array.
 * @param[in,out] numObjects    A pointer to the number of valid objects
 *                              in the @a objects array. The function sets
 *                              this value after clustering.
 */
void NvDsInferDBScanCluster(NvDsInferDBScanHandle handle,
        NvDsInferDBScanClusteringParams *params,  NvDsInferObjectDetectionInfo *objects,
        size_t *numObjects);

/**
 * Clusters an array of objects in place using specified clustering parameters.
 * The outputs are partially only clustered i.e to merge close neighbors of
 * the same cluster together only and the mean normalization of all the
 * proposals in a cluster is not performed. The outputs from this stage are
 * later fed into another clustering algorithm like NMS to obtain the final
 * results.
 *
 * @param[in]     handle        A handle to the context be used for clustering.
 * @param[in]     params        A pointer to a clustering parameter structure.
 * @param[in,out] objects       A pointer to an array of objects to be
 *                              clustered. The function places the clustered
 *                              objects in the same array.
 * @param[in,out] numObjects    A pointer to the number of valid objects
 *                              in the @a objects array. The function sets
 *                              this value after clustering.
 */
void NvDsInferDBScanClusterHybrid(NvDsInferDBScanHandle handle,
        NvDsInferDBScanClusteringParams *params,  NvDsInferObjectDetectionInfo *objects,
        size_t *numObjects);

#ifdef __cplusplus
}
#endif

#endif

/** @} */
