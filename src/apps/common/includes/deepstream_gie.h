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

#ifndef __NVGSTDS_GIE_H__
#define __NVGSTDS_GIE_H__

#include <gst/gst.h>

#ifdef __cplusplus
extern "C"
{
#endif

#include "gstnvdsmeta.h"
#include "gstnvdsinfer.h"
#include "deepstream_config.h"

typedef enum
{
  NV_DS_GIE_PLUGIN_INFER = 0,
  NV_DS_GIE_PLUGIN_INFER_SERVER,
  NV_DS_GIE_PLUGIN_VIDEO_TEMPLATE,
  NV_DS_GIE_PLUGIN_VISION_ENCODER,
} NvDsGiePluginType;

typedef struct
{
  gboolean enable;

  gchar *config_file_path;

  gboolean input_tensor_meta;

  gboolean override_colors;

  gint operate_on_gie_id;
  gboolean is_operate_on_gie_id_set;
  gint operate_on_classes;

  gint num_operate_on_class_ids;
  gint *list_operate_on_class_ids;

  gboolean have_bg_color;
  NvOSD_ColorParams bbox_bg_color;
  NvOSD_ColorParams bbox_border_color;

  GHashTable *bbox_border_color_table;
  GHashTable *bbox_bg_color_table;

  guint batch_size;
  gboolean is_batch_size_set;

  guint interval;
  gboolean is_interval_set;
  guint unique_id;
  gboolean is_unique_id_set;
  guint gpu_id;
  gboolean is_gpu_id_set;
  guint nvbuf_memory_type;
  gchar *model_engine_file_path;

  gchar *audio_transform;
  guint frame_size;
  gboolean is_frame_size_set;
  guint hop_size;
  gboolean is_hop_size_set;
  guint input_audio_rate;

  gchar *label_file_path;
  guint n_labels;
  guint *n_label_outputs;
  gchar ***labels;

  gchar *raw_output_directory;
  gulong file_write_frame_num;

  gchar *tag;

  NvDsGiePluginType plugin_type;

  // Vision encoder specific fields (when plugin_type == NV_DS_GIE_PLUGIN_VISION_ENCODER)
  gchar *model_variant;
  gchar *device;
  guint min_crop_size;
  gboolean is_min_crop_size_set;
  guint verbose;
  gchar *backend;           // "transformers" or "triton"
  gchar *triton_url;        // Triton gRPC endpoint (host:port)
  gchar *triton_model;      // Model name in Triton repository
} NvDsGieConfig;

#ifdef __cplusplus
}
#endif

#endif
