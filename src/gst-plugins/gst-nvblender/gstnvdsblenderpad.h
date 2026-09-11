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

#ifndef __GST_NVBLENDER_PAD_H__
#define __GST_NVBLENDER_PAD_H__

#include <gst/gst.h>
#include <gst/video/video.h>

G_BEGIN_DECLS

#define GST_TYPE_NVBLENDER_PAD (gst_nvblender_pad_get_type())
#define GST_NVBLENDER_PAD(obj) \
        (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_NVBLENDER_PAD, GstNvblenderPad))
#define GST_NVBLENDER_PAD_CLASS(klass) \
        (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_NVBLENDER_PAD, GstNvblenderPadClass))
#define GST_IS_NVBLENDER_PAD(obj) \
        (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_NVBLENDER_PAD))
#define GST_IS_NVBLENDER_PAD_CLASS(klass) \
        (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_NVBLENDER_PAD))

typedef struct _GstNvblenderPad GstNvblenderPad;
typedef struct _GstNvblenderPadClass GstNvblenderPadClass;

/**
 * GstNvblenderPad:
 *
 * The opaque #GstNvblenderPad structure.
 */
struct _GstNvblenderPad
{
  GstVideoAggregatorPad parent;

  /* properties */
  gint xpos, ypos;
  gint width, height;
  gdouble alpha;
  gdouble crossfade;

  GstVideoConverter *convert;
  GstVideoInfo conversion_info;
  GstBuffer *converted_buffer;

  gboolean crossfaded;
};

struct _GstNvblenderPadClass
{
  GstVideoAggregatorPadClass parent_class;
};

GType gst_nvblender_pad_get_type (void);

G_END_DECLS
#endif /* __GST_NVBLENDER_PAD_H__ */
