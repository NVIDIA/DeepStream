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
