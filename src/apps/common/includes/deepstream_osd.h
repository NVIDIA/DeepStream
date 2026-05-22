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

#ifndef __NVGSTDS_OSD_H__
#define __NVGSTDS_OSD_H__

#include <gst/gst.h>

#ifdef __cplusplus
extern "C"
{
#endif

#include "nvll_osd_struct.h"

typedef struct
{
  GstElement *bin;
  GstElement *queue;
  GstElement *nvvidconv;
  GstElement *conv_queue;
  GstElement *cap_filter;
  GstElement *nvosd;
} NvDsOSDBin;

typedef struct
{
  gboolean enable;
  gboolean text_has_bg;
  gboolean enable_clock;
  gboolean draw_text;
  gboolean draw_bbox;
  gboolean draw_mask;
  gint text_size;
  gint border_width;
  gint clock_text_size;
  gint clock_x_offset;
  gint clock_y_offset;
  guint gpu_id;
  guint nvbuf_memory_type; /* For nvvidconv */
  guint num_out_buffers;
  gchar *font;
  gchar *hw_blend_color_attr;
  NvOSD_Mode mode;
  NvOSD_ColorParams clock_color;
  NvOSD_ColorParams text_color;
  NvOSD_ColorParams text_bg_color;
} NvDsOSDConfig;

/**
 * Initialize @ref NvDsOSDBin. It creates and adds OSD and other elements
 * needed for processing to the bin. It also sets properties mentioned
 * in the configuration file under group @ref CONFIG_GROUP_OSD
 *
 * @param[in] config pointer to OSD @ref NvDsOSDConfig parsed from config file.
 * @param[in] bin pointer to @ref NvDsOSDBin to be filled.
 *
 * @return true if bin created successfully.
 */
gboolean create_osd_bin (NvDsOSDConfig *config, NvDsOSDBin *bin);

#ifdef __cplusplus
}
#endif

#endif
