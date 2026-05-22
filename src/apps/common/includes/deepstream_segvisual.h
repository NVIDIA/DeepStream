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

#ifndef __NVGSTDS_SEGVISUAL_H__
#define __NVGSTDS_SEGVISUAL_H__

#include <gst/gst.h>

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct
{
  GstElement *bin;
  GstElement *queue;
  GstElement *nvvidconv;
  GstElement *conv_queue;
  GstElement *cap_filter;
  GstElement *nvsegvisual;
} NvDsSegVisualBin;

typedef struct
{
  gboolean enable;
  guint gpu_id;
  guint max_batch_size;
  guint width;
  guint height;
  guint nvbuf_memory_type; /* For nvvidconv */
} NvDsSegVisualConfig;

/**
 * Initialize @ref NvDsSegVisualBin. It creates and adds SegVisual and other elements
 * needed for processing to the bin. It also sets properties mentioned
 * in the configuration file under group @ref CONFIG_GROUP_SegVisual
 *
 * @param[in] config pointer to SegVisual @ref NvDsSegVisualConfig parsed from config file.
 * @param[in] bin pointer to @ref NvDsSegVisualBin to be filled.
 *
 * @return true if bin created successfully.
 */
gboolean create_segvisual_bin (NvDsSegVisualConfig *config, NvDsSegVisualBin *bin);

#ifdef __cplusplus
}
#endif

#endif
