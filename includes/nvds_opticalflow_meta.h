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

/**
 * @file
 * <b>NVIDIA Optical Flow Metadata </b>
 *
 * @b Description: This file defines the optical flow metadata.
 */

/**
 * @defgroup  ee_opticalflow_meta  Optical flow metadata
 *
 * Defines the optical flow metadata.
 * @ingroup NvDsMetaApi
 * @{
 */

#ifndef _NVDS_OPTICALFLOW_META_H_
#define _NVDS_OPTICALFLOW_META_H_

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * Holds motion vector information about an element.
 */
typedef struct _NvOFFlowVector
{
    /** Holds the motion vector X component. */
    short flowx;

    /** Holds the motion vector Y component. */
    short flowy;
} NvOFFlowVector;

/**
 * Holds optical flow metadata about a frame.
 */
typedef struct
{
    /** Holds the number of rows in the frame for a given block size,
     e.g. if block size is 4 and frame height is 720, then the number of
     rows is (720/4) = 180. */
    unsigned int rows;
    /** Holds the number of columns in the frame for given block size,
     e.g. if block size is 4 and frame width is 1280, then the number of
     columns is (1280/4) = 320. */
    unsigned int cols;
    /** Holds the size of the motion vector. @see NvOFFlowVector. */
    unsigned int mv_size;
    /** Holds the size of the confidence values of the motion vector. */
    unsigned int cost_size;
    /** Holds the current frame number of the source. */
    unsigned long frame_num;
    /** Holds a pointer to the motion vector. */
    void *data;
    /** Holds a pointer to the cost of the motion vector. */
    void *cost;
    /** Reserved for internal use. */
    void *priv;
    /** Reserved for internal use. */
    void *reserved;
} NvDsOpticalFlowMeta;

#ifdef __cplusplus
}
#endif

#endif

/** @} */
