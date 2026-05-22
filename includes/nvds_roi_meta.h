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
 * <b>NVIDIA GStreamer DeepStream: ROI Meta used in nvdspreprocess plugin </b>
 *
 * @b Description: This file defines the Metadata structure used to
 * carry DeepStream ROI metadata in GStreamer pipeline.
 *
 * @defgroup  ee_nvdspreprocess_api_group Pre-Process Metadata
 *
 * Specifies metadata concerning ROIs used in nvdspreprocess plugin.
 *
 * @ingroup NvDsPreProcessApi
 * @{
 */

#ifndef __NVDS_ROI_META_H__
#define __NVDS_ROI_META_H__

#include "nvdsmeta.h"

/** max polygon points ; currently not being used */
#define DS_MAX_POLYGON_POINTS 8

/** DS NvBufSurfaceParams */
typedef struct NvBufSurfaceParams NvBufSurfaceParams;

/** DS NvDsFrameMeta */
typedef struct _NvDsFrameMeta NvDsFrameMeta;

/** DS NvDsObjectMeta */
typedef struct _NvDsObjectMeta NvDsObjectMeta;

/** classifier meta list */
typedef GList NvDsClassifierMetaList;

/** user meta list */
typedef GList NvDsUserMetaList;

/**
 * Data type used for model in infer
 */
typedef enum
{
  /** FP32 data type */
  NvDsDataType_FP32,
  /** UINT8 data type */
  NvDsDataType_UINT8,
  /** INT8 data type */
  NvDsDataType_INT8,
  /** UINT32 data type */
  NvDsDataType_UINT32,
  /** INT32 data type */
  NvDsDataType_INT32,
  /** INT64 data type */
  NvDsDataType_INT64,
  /** UINT64 data type */
  NvDsDataType_UINT64,
  /** FP16 data type */
  NvDsDataType_FP16,
} NvDsDataType;

/**
 * Unit Type Fullframe/ROI/Crop Objects
 */
typedef enum
{
  /** Full frames */
  NvDsUnitType_FullFrame = 0,
  /** Region of Interests (ROIs) */
  NvDsUnitType_ROI,
  /** object mode */
  NvDsUnitType_Object,
} NvDsUnitType;

/**
 * Holds Information about ROI Metadata
 */
typedef struct NvDsRoiMeta
{
  /* per roi information */
  NvOSD_RectParams roi;

  /** currently not being used */
  guint roi_polygon[DS_MAX_POLYGON_POINTS][2];

  /* Scaled & converted buffer to processing width/height */
  NvBufSurfaceParams *converted_buffer;

  /* Deepstream frame meta */
  NvDsFrameMeta *frame_meta;

  /** Ratio by which the frame/ROI crop was scaled in horizontal direction
   * Required when scaling co-ordinates/sizes in metadata
   * back to input resolution. */
  gdouble scale_ratio_x;

  /** Ratio by which the frame/ROI crop was scaled in vertical direction
   * Required when scaling co-ordinates/sizes in metadata
   * back to input resolution. */
  gdouble scale_ratio_y;

  /** offsets in horizontal direction while scaling */
  gdouble offset_left;

  /** offsets in vertical direction while scaling */
  gdouble offset_top;

  /** Holds a pointer to a list of pointers of type @ref NvDsClassifierMeta. */
  NvDsClassifierMetaList *classifier_meta_list;

  /** Holds a pointer to a list of pointers of type @ref NvDsUserMeta. */
  NvDsUserMetaList *roi_user_meta_list;

  /* Deepstream object meta */
  NvDsObjectMeta *object_meta;

} NvDsRoiMeta;

#endif
