/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef __GST_DS_NVOSD_BIN_H__
#define __GST_DS_NVOSD_BIN_H__

#include <gst/video/video.h>
#include "nvll_osd_struct.h"

#include <unordered_map>
#include <string>

G_BEGIN_DECLS

enum
{
  PROP_OSD_0,
  PROP_OSD_FONT,
  PROP_OSD_TEXT_SIZE,
  PROP_OSD_TEXT_COLOR,
  PROP_OSD_TEXT_BG_COLOR,
  PROP_OSD_BBOX_BORDER_COLORS,
  PROP_OSD_BBOX_BG_COLORS,
  PROP_OSD_BORDER_WIDTH,
  PROP_OSD_DISPLAY_TRACKING_ID,
  PROP_OSD_REFORMAT_OBJECT_LABELS,
  PROP_OSD_LAST,
};

/* Standard GStreamer boilerplate */
typedef struct _GstDsNvOSDBin
{
  GstBin bin;

  GstElement *queue;
  GstElement *nvvidconv;
  GstElement *conv_queue;
  GstElement *nvosd;

  gboolean prop_set[PROP_OSD_LAST];

  gchar *font;
  guint text_size;
  NvOSD_ColorParams text_color;
  NvOSD_ColorParams text_bg_color;
  std::unordered_map<std::string, NvOSD_ColorParams> *class_border_color_map;
  std::unordered_map<std::string, NvOSD_ColorParams> *class_bg_color_map;

  gboolean display_tracking_id;
  gboolean reformat_object_labels;

  guint border_width;
} GstDsNvOSDBin;

typedef struct _GstDsNvOSDBinClass
{
  GstBinClass parent_class;

  void (*set_display_text_enabled) (GstDsNvOSDBin *, gboolean);
} GstDsNvOSDBinClass;

/* Standard GStreamer boilerplate */
#define GST_TYPE_DS_NVOSD_BIN (gst_ds_nvosd_bin_get_type())
#define GST_DS_NVOSD_BIN(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_DS_NVOSD_BIN,GstDsNvOSDBin))
#define GST_DS_NVOSD_BIN_CLASS(klass) (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_DS_NVOSD_BIN,GstDsNvOSDBinClass))
#define GST_DS_NVOSD_BIN_GET_CLASS(obj) (G_TYPE_INSTANCE_GET_CLASS((obj), GST_TYPE_DS_NVOSD_BIN, GstDsNvOSDBinClass))
#define GST_IS_DS_NVOSD_BIN(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_DS_NVOSD_BIN))
#define GST_IS_DS_NVOSD_BIN_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_DS_NVOSD_BIN))
#define GST_DS_NVOSD_BIN_CAST(obj)  ((GstDsNvOSDBin *)(obj))

GType gst_ds_nvosd_bin_get_type (void);

G_END_DECLS
#endif /* __GST_DS_NVOSD_BIN_H__ */
