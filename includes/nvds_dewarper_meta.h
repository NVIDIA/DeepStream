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
 * <b>Defines dewarping metadata </b>
 *
 * @b Description: This file defines metadata concerning dewarping.
 */

/**
 * @defgroup ee_dewarping_group Dewarping Metadata
 * Defines metadata concerning dewarping.
 * @ingroup NvDsMetaApi
 * @{
 */

#ifndef _NVDS_DEWARPER_META_H_
#define _NVDS_DEWARPER_META_H_

#include <gst/gst.h>

#ifdef __cplusplus
extern "C"
{
#endif

/*
 * Defines DeepStream Dewarper metadata.
 */

/**
 * Maximum number of dewarped surfaces per frame supported
 */
#define MAX_DEWARPED_VIEWS 4

/**
 * Defines metadata surface types.
 */
typedef enum
{
    NVDS_META_SURFACE_NONE=0,
    /** Defines the pushbroom surface type. */
    NVDS_META_SURFACE_FISH_PUSHBROOM=1,
    /** Defines the vertical radial cylindrical surface type. */
    NVDS_META_SURFACE_FISH_VERTCYL=2,
    /** Defines the perspective projection with Brown distortion model surface. */
    NVDS_META_SURFACE_PERSPECTIVE_PERSPECTIVE=3,
    /** Defines Fisheye to  perspective projection surface type. */
    NVDS_META_SURFACE_FISH_PERSPECTIVE=4,
    /** Defines Fisheye to  Fisheye projection surface type. */
    NVDS_META_SURFACE_FISH_FISH=5,
    /** Defines Fisheye to  cylindrical projection surface type. */
    NVDS_META_SURFACE_FISH_CYL=6,
    /** Defines Fisheye to  equirectangular projection surface type. */
    NVDS_META_SURFACE_FISH_EQUIRECT=7,
    /** Defines Fisheye to  panini projection surface type. */
    NVDS_META_SURFACE_FISH_PANINI=8,
    /** Defines perspective to  equirectangular projection surface type. */
    NVDS_META_SURFACE_PERSPECTIVE_EQUIRECT=9,
    /** Defines perspective to  panini projection surface type. */
    NVDS_META_SURFACE_PERSPECTIVE_PANINI=10,
    /** Defines equirectangular to  cylindrical projection surface type. */
    NVDS_META_SURFACE_EQUIRECT_CYLINDER=11,
    /** Defines equirectangular to  equirectangular projection surface type. */
    NVDS_META_SURFACE_EQUIRECT_EQUIRECT=12,
    /** Defines equirectangular to  Fisheye projection surface type. */
    NVDS_META_SURFACE_EQUIRECT_FISHEYE=13,
    /** Defines equirectangular to  panini projection surface type. */
    NVDS_META_SURFACE_EQUIRECT_PANINI=14,
    /** Defines equirectangular to  perspective projection surface type. */
    NVDS_META_SURFACE_EQUIRECT_PERSPECTIVE=15,
    /** Defines equirectangular to  pushbroom projection surface type. */
    NVDS_META_SURFACE_EQUIRECT_PUSHBROOM=16,
    /** Defines equirectangular to  stereographic projection surface type. */
    NVDS_META_SURFACE_EQUIRECT_STEREOGRAPHIC=17,
    /** Defines equirectangular to  vertical radial cylindrical projection surface type. */
    NVDS_META_SURFACE_EQUIRECT_VERTCYLINDER=18
} NvDsSurfaceType;

/**
 * Holds a set of dewarper metadata.
 */
typedef struct _NvDewarperSurfaceMeta {
  /** Holds an array of the types of dewarped surfaces. */
  guint type[MAX_DEWARPED_VIEWS];
  /** Holds an array of indices of dewarped surfaces. */
  guint index[MAX_DEWARPED_VIEWS];
  /** Holds the source ID of the frame, e.g. the camera ID. */
  guint source_id;
  /** Holds the number of filled surfaces in the frame. */
  guint num_filled_surfaces;
}NvDewarperSurfaceMeta;

#ifdef __cplusplus
}
#endif

#endif

/** @} */
