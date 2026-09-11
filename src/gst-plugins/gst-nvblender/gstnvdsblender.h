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

#ifndef __GST_NVBLENDER_H__
#define __GST_NVBLENDER_H__

#include <gst/gst.h>
#include <gst/video/video.h>
#include <gst/video/gstvideoaggregator.h>

#include <cuda_runtime.h>
#include "gstnvdsmeta.h"
#include "gstnvdsbufferpool.h"
#include "nvbufsurface.h"
#include "nvbufsurftransform.h"

G_BEGIN_DECLS

#define GST_CAPS_FEATURE_MEMORY_NVMM      "memory:NVMM"

#define GST_TYPE_NVBLENDER (gst_nvblender_get_type())
#define GST_NVBLENDER(obj) \
        (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_NVBLENDER, GstNvblender))
#define GST_NVBLENDER_CLASS(klass) \
        (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_NVBLENDER, GstNvblenderClass))
#define GST_IS_NVBLENDER(obj) \
        (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_NVBLENDER))
#define GST_IS_NVBLENDER_CLASS(klass) \
        (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_NVBLENDER))

typedef struct _GstNvblender GstNvblender;
typedef struct _GstNvblenderClass GstNvblenderClass;

/**
 * GstcompositorBackground:
 * @COMPOSITOR_BACKGROUND_CHECKER: checker pattern background
 * @COMPOSITOR_BACKGROUND_BLACK: solid color black background
 * @COMPOSITOR_BACKGROUND_WHITE: solid color white background
 * @COMPOSITOR_BACKGROUND_TRANSPARENT: background is left transparent and layers are composited using "A OVER B" composition rules. This is only applicable to AYUV and ARGB (and variants) as it preserves the alpha channel and allows for further mixing.
 *
 * The different backgrounds compositor can blend over.
 */
typedef enum
{
  COMPOSITOR_BACKGROUND_CHECKER,
  COMPOSITOR_BACKGROUND_BLACK,
  COMPOSITOR_BACKGROUND_WHITE,
  COMPOSITOR_BACKGROUND_TRANSPARENT,
} GstCompositorBackground;

/**
 * GstNvblender:
 *
 * The opaque #GstNvblender structure.
 */
struct _GstNvblender
{
  GstVideoAggregator videoaggregator;
  GstCompositorBackground background;

  //BlendFunction blend, overlay;
  //FillCheckerFunction fill_checker;
  //FillColorFunction fill_color;
  GstBufferPool *pool;
  GMutex flow_lock;
  GstVideoInfo info;
  NvBufSurfaceColorFormat in_pix_fmt;
  gint width;
  gint height;
  gboolean alpha_surf_allocated=false;
  NvBufSurface *alpha_surf;
  guint batch_size;
  gint gpu_id;
  gboolean sinkpad0_active;
  gboolean batched_background;
  GstClockTime prev_pts;
  GstBuffer *matte_alpha_gst_buffer;
  //TODO: used Cuda stream session for composition
  cudaStream_t cu_nbstream;
  GHashTable *valid_bg_buf;
};

struct _GstNvblenderClass
{
  GstVideoAggregatorClass parent_class;
};

GType gst_nvblender_get_type (void);

G_END_DECLS
#endif /* __GST_NVBLENDER_H__ */
