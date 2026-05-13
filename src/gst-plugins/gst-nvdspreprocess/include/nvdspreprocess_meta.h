/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @file nvdspreprocess_meta.h
 * <b>NVIDIA DeepStream GStreamer NvDsPreProcess meta Specification </b>
 *
 * @b Description: This file specifies the metadata attached by
 * the DeepStream GStreamer NvDsPreProcess Plugin.
 */

/**
 * @defgroup   gstreamer_nvdspreprocess_api  NvDsPreProcess Plugin
 * Defines an API for the GStreamer NvDsPreProcess plugin.
 * @ingroup custom_gstreamer
 * @{
 */

#ifndef __NVDSPREPROCESS_META_H__
#define __NVDSPREPROCESS_META_H__

#include <vector>
#include <string>
#include "nvbufsurface.h"
#include "nvds_roi_meta.h"

/**
 * tensor meta containing prepared tensor and related info
 * inside preprocess user meta which is attached at batch level
 */
typedef struct
{
  /** raw tensor buffer preprocessed for infer */
  void *raw_tensor_buffer;

  /** size of raw tensor buffer */
  guint64 buffer_size;

  /** raw tensor buffer shape */
  std::vector<int> tensor_shape;

  /** model datatype for which tensor prepared */
  NvDsDataType data_type;

  /** to be same as model input layer name */
  std::string tensor_name;

  /** gpu-id on which tensor prepared */
  guint gpu_id;

  /** pointer to buffer from tensor pool */
  void *private_data;

  /** meta id for differentiating between multiple tensor meta from same gst buffer,for the case when sum of roi's exceeds the batch size*/
  guint meta_id;

  /** parameter to inform whether aspect ratio is maintained in the preprocess tensor*/
  gboolean maintain_aspect_ratio;
} NvDsPreProcessTensorMeta;

/**
 * preprocess meta as a user meta which is attached at
 * batch level
 */
typedef struct
{
  /** target unique ids for which meta is prepared */
  std::vector<guint64> target_unique_ids;

  /** pointer to tensor meta */
  NvDsPreProcessTensorMeta *tensor_meta;

  /** list of roi vectors per batch */
  std::vector<NvDsRoiMeta> roi_vector;

  /** pointer to buffer from scaling pool*/
  void *private_data;

} GstNvDsPreProcessBatchMeta;

#endif /* __NVDSPREPROCESS_META_H__ */
