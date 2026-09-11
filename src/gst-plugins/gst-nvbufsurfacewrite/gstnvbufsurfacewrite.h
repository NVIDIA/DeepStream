/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#ifndef __GST_NVBUFSURFACEWRITE_H__
#define __GST_NVBUFSURFACEWRITE_H__

#ifdef __cplusplus
extern "C"
{
#endif

#include <cuda.h>
#include <cuda_runtime.h>
#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>

G_BEGIN_DECLS

/* #defines don't like whitespacey bits */
#define GST_TYPE_NVBUFSURFACEWRITE \
  (gst_nvbufsurfacewrite_get_type())
#define GST_NVBUFSURFACEWRITE(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_NVBUFSURFACEWRITE,GstNvBufSurfaceWrite))
#define GST_NVBUFSURFACEWRITE_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_NVBUFSURFACEWRITE,GstNvBufSurfaceWriteClass))
#define GST_IS_NVBUFSURFACEWRITE(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_NVBUFSURFACEWRITE))
#define GST_IS_NVBUFSURFACEWRITE_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_NVBUFSURFACEWRITE))

typedef struct _GstNvBufSurfaceWrite      GstNvBufSurfaceWrite;
typedef struct _GstNvBufSurfaceWriteClass GstNvBufSurfaceWriteClass;

struct _GstNvBufSurfaceWrite
{
  GstBaseTransform element;

  GstPad *sinkpad, *srcpad;
  GstVideoInfo in_info;
  GstVideoInfo out_info;
  gboolean is_same_caps;

  /* source and sink pad caps */
  GstCaps *sinkcaps;
  GstCaps *srccaps;

  unsigned int input_width;
  unsigned int input_height;
  unsigned int output_width;
  unsigned int output_height;
  unsigned int gpu_id;

  int input_feature;
  int output_feature;
  GstVideoFormat input_fmt;
  GstVideoFormat output_fmt;

  cudaStream_t stream;

  int frame_num;

  // Used for writeing surfaces
  void *input_nvbufsurface = NULL;

  gboolean silent;
};

struct _GstNvBufSurfaceWriteClass
{
  GstBaseTransformClass parent_class;
};

GType gst_nvbufsurfacewrite_get_type (void);

G_END_DECLS

#ifdef __cplusplus
}
#endif

#endif /* __GST_NVBUFSURFACEWRITE_H__ */
