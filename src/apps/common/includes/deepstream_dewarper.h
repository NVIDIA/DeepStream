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

#ifndef __NVGSTDS_DEWARPER_H__
#define __NVGSTDS_DEWARPER_H__

#include <gst/gst.h>

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct
{
  GstElement *bin;
  GstElement *queue;
  GstElement *src_queue;
  GstElement *conv_queue;
  GstElement *nvvidconv;
  GstElement *cap_filter;
  GstElement *dewarper_caps_filter;
  GstElement *nvdewarper;
} NvDsDewarperBin;

typedef struct
{
  gboolean enable;
  guint gpu_id;
  guint num_out_buffers;
  guint dewarper_dump_frames;
  gchar *config_file;
  guint nvbuf_memory_type;
  guint source_id;
  guint num_surfaces_per_frame;
  guint num_batch_buffers;
} NvDsDewarperConfig;

gboolean create_dewarper_bin (NvDsDewarperConfig * config, NvDsDewarperBin * bin);

#ifdef __cplusplus
}
#endif

#endif
