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

#ifndef __GST_DS_NVVIDEORENDERER_BIN_H__
#define __GST_DS_NVVIDEORENDERER_BIN_H__

#include <gst/gst.h>

#include "nvbufsurface.h"

G_BEGIN_DECLS

typedef enum
{
  VIDEO_RENDERER_NONE,
  VIDEO_RENDERER_FAKE,
  VIDEO_RENDERER_EGL,
  VIDEO_RENDERER_DRM,
  VIDEO_RENDERER_3D,
} VideoRendererType;

/* Standard GStreamer boilerplate */
typedef struct _GstDsNvVideoRendererBin
{
  GstBin bin;

  guint offset_x;
  guint offset_y;
  guint width;
  guint height;
  guint sync;
  guint qos;
  guint gpu_id;
  guint color_range;
  guint conn_id;
  guint plane_id;
  gboolean set_mode;
  NvBufSurfaceMemType mem_type;
  gchar *create_from_config;
  VideoRendererType vrenderer_type;

  guintptr window_id;
  gboolean window_id_set;
  gint render_rect_x;
  gint render_rect_y;
  gint render_rect_w;
  gint render_rect_h;
  gboolean render_rect_set;

  GstElement *sink;
  GstElement *cap_filter;
  GstElement *transform;
  GstElement *queue;

  GstPad *bin_sink_pad;

} GstDsNvVideoRendererBin;

typedef struct _GstDsNvVideoRendererBinClass
{
  GstBinClass parent_class;
} GstDsNvVideoRendererBinClass;


/* Standard GStreamer boilerplate */
#define GST_TYPE_DS_NVVIDEORENDERER_BIN (gst_ds_nvvideorenderer_bin_get_type())
#define GST_DS_NVVIDEORENDERER_BIN(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_DS_NVVIDEORENDERER_BIN,GstDsNvVideoRendererBin))
#define GST_DS_NVVIDEORENDERER_BIN_CLASS(klass) (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_DS_NVVIDEORENDERER_BIN,GstDsNvVideoRendererBinClass))
#define GST_DS_NVVIDEORENDERER_BIN_GET_CLASS(obj) (G_TYPE_INSTANCE_GET_CLASS((obj), GST_TYPE_DS_NVVIDEORENDERER_BIN, GstDsNvVideoRendererBinClass))
#define GST_IS_DS_NVVIDEORENDERER_BIN(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_DS_NVVIDEORENDERER_BIN))
#define GST_IS_DS_NVVIDEORENDERER_BIN_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_DS_NVVIDEORENDERER_BIN))
#define GST_DS_NVVIDEORENDERER_BIN_CAST(obj)  ((GstDsNvVideoRendererBin *)(obj))

GType gst_ds_nvvideorenderer_bin_get_type (void);

G_END_DECLS
#endif /* __GST_INFER_H__ */
