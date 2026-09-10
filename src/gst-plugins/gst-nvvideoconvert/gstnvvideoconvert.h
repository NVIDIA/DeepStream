/*
 * SPDX-FileCopyrightText: Copyright (c) 2014-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


#ifndef __GST_NVVIDEOCONVERT_H__
#define __GST_NVVIDEOCONVERT_H__

#include <gst/gst.h>
#include <gst/video/video.h>
#include <gst/base/gstbasetransform.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include "gstnvdsbufferpool.h"
#include "nvbufsurface.h"
#include "nvbufsurftransform.h"
//#ifndef __aarch64__
#include "gst-nvquery.h"
#include "gst-nvquery-internal.h"
//#endif

G_BEGIN_DECLS
#define GST_TYPE_NVVIDEOCONVERT \
  (gst_nvvideoconvert_get_type())
#define GST_NVVIDEOCONVERT(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_NVVIDEOCONVERT,Gstnvvideoconvert))
#define GST_NVVIDEOCONVERT_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_NVVIDEOCONVERT,GstnvvideoconvertClass))
#define GST_IS_NVVIDEOCONVERT(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_NVVIDEOCONVERT))
#define GST_IS_NVVIDEOCONVERT_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_NVVIDEOCONVERT))

/* Name of package */
#define PACKAGE "nvvideoconvert"
/* Define to the full name of this package. */
#define PACKAGE_NAME "GStreamer nvvideoconvert Plugin"
/* Define to the full name and version of this package. */
#define PACKAGE_STRING "GStreamer nvvideoconvert 1.2.3"
/* Information about the purpose of the plugin. */
#define PACKAGE_DESCRIPTION "video Colorspace conversion & scaler"
/* Define to the home page for this package. */
#define PACKAGE_URL "http://nvidia.com/"
/* Define to the version of this package. */
#define PACKAGE_VERSION "1.2.3"
/* Define under which licence the package has been released */
#define PACKAGE_LICENSE "Proprietary"
/* Version number of package */
#define VERSION "1.2.3"

#define NVRM_MAX_SURFACES                 3
#define NVFILTER_MAX_BUF                  4
#define GST_CAPS_FEATURE_MEMORY_NVMM      "memory:NVMM"
#define GST_NVSTREAM_MEMORY_TYPE          "nvstream"

typedef struct _Gstnvvideoconvert Gstnvvideoconvert;
typedef struct _GstnvvideoconvertClass GstnvvideoconvertClass;

typedef struct _GstNvVideoConvertBuffer GstNvVideoConvertBuffer;
typedef struct _GstNvInterBuffer GstNvInterBuffer;

/**
 * BufType:
 *
 * Buffer type enum.
 */
typedef enum
{
  BUF_TYPE_YUV,
  BUF_TYPE_GRAY,
  BUF_TYPE_RGB,
  BUF_NOT_SUPPORTED
} BufType;

/**
 * BufMemType:
 *
 * Buffer memory type enum.
 */
typedef enum
{
  BUF_MEM_SW,
  BUF_MEM_HW
} BufMemType;

/**
 * GstVideoFlipMethods:
 *
 * Video flip methods type enum.
 */
typedef enum
{
  GST_VIDEO_NVFLIP_METHOD_IDENTITY,
  GST_VIDEO_NVFLIP_METHOD_90L,
  GST_VIDEO_NVFLIP_METHOD_180,
  GST_VIDEO_NVFLIP_METHOD_90R,
  GST_VIDEO_NVFLIP_METHOD_HORIZ,
  GST_VIDEO_NVFLIP_METHOD_INVTRANS,
  GST_VIDEO_NVFLIP_METHOD_VERT,
  GST_VIDEO_NVFLIP_METHOD_TRANS
} GstVideoFlipMethods;


/**
 * GstNvVideoConvertBuffer:
 *
 * Nvfilter buffer.
 */
struct _GstNvVideoConvertBuffer
{
  gint dmabuf_fd;
  GstBuffer *gst_buf;
};

/**
 * GstNvInterBuffer:
 *
 * Intermediate transform buffer.
 */
struct _GstNvInterBuffer
{
  gint idmabuf_fd;
};

/**
 * Gstnvvideoconvert:
 *
 * Opaque object data structure.
 */
struct _Gstnvvideoconvert
{
  GstBaseTransform element;

  /* source and sink pad caps */
  GstCaps *sinkcaps;
  GstCaps *srccaps;

  gint to_width;
  gint to_height;
  gint from_width;
  gint from_height;
  gint tsurf_width;
  gint tsurf_height;
  gint compute_hw;
  gint gpu_id;
  gint nvbuf_mem_type;
  gint copy_hw;

  gint src_crop_left;
  gint src_crop_top;
  gint src_crop_width;
  gint src_crop_height;

  gint dst_crop_left;
  gint dst_crop_top;
  gint dst_crop_width;
  gint dst_crop_height;


  BufType inbuf_type;
  BufMemType inbuf_memtype;
  BufMemType outbuf_memtype;
  GstVideoInfo in_info;
  GstVideoInfo out_info;

  NvBufSurfTransformParams transform_params;
  NvBufSurfaceColorFormat in_pix_fmt;
  NvBufSurfaceColorFormat out_pix_fmt;
  NvBufSurfTransformConfigParams config_params;
  NvBufSurface *intermediate_buffer;
  NvBufSurface *intermediate_buffer_two;

  guint insurf_count;
  guint tsurf_count;
  guint isurf_count;
  guint ibuf_count;
  gint flip_method;
  guint num_output_buf;
  gint interpolation_method;

  gboolean silent;
  gboolean no_dimension;
  gboolean do_scaling;
  gboolean do_flip;
  gboolean do_src_cropping;
  gboolean do_dst_cropping;
  gboolean do_mem_type_conversion;
  gboolean do_gpu_id_conversion;
  gboolean need_intersurf;
  gboolean isurf_flag;
  gboolean negotiated;
  gboolean nvfilterpool;
  gboolean enable_blocklinear_output;
  gboolean allow_odd_crop;
  gboolean enable_contiguous_bufs;
  gboolean disable_passthrough;
  GstBufferPool *pool;
  GMutex flow_lock;

  GstNvInterBuffer interbuf;
  guint num_batch_buffers;
  guint session_created;
};

struct _GstnvvideoconvertClass
{
  GstBaseTransformClass parent_class;
};

GType gst_nvvideoconvert_get_type (void);

G_END_DECLS
#endif /* __GST_NVVIDEOCONVERT_H__ */
