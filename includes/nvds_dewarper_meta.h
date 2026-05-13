/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
