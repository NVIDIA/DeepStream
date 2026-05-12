/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2019 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
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
