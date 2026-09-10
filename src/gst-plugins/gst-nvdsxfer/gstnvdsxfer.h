/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef __GST_NVDSXFER_H__
#define __GST_NVDSXFER_H__

#include <gst/base/gstbasetransform.h>
#include <gst/video/video.h>
#include <glib-object.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "gstnvdsmeta.h"
#include "nvbufsurftransform.h"

#include "nvtx3/nvToolsExt.h"

/* Package and library details required for plugin_init */
#define PACKAGE "nvxfer"
#define LICENSE "Proprietary"
#define DESCRIPTION "NVIDIA NVDSXFER plugin"
#define BINARY_PACKAGE "NVIDIA DeepStream NVDSXFER Plugin"
#define URL "http://nvidia.com/"

G_BEGIN_DECLS
/* Standard boilerplate stuff */
typedef struct _GstNvXfer GstNvXfer;
typedef struct _GstNvXferClass GstNvXferClass;

/* Standard boilerplate stuff */
#define GST_TYPE_NVDSXFER (gst_nvdsxfer_get_type())
#define GST_NVXFER(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_NVDSXFER,GstNvXfer))
#define GST_NVXFER_CLASS(klass) (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_NVDSXFER,GstNvXferClass))
#define GST_NVXFER_GET_CLASS(obj) (G_TYPE_INSTANCE_GET_CLASS((obj), GST_TYPE_NVDSXFER, GstNvXferClass))
#define GST_IS_NVXFER(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_NVDSXFER))
#define GST_IS_NVXFER_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_NVDSXFER))
#define GST_NVXFER_CAST(obj)  ((GstNvXfer *)(obj))

struct _GstNvXfer
{
  GstBaseTransform base_trans;

  /** Boolean to signal output thread to stop. */
  gboolean stop;

  /** Input and Output video info (resolution, color format, framerate, etc) */
  GstVideoInfo in_video_info;
  GstVideoInfo out_video_info;

  /** GPU ID on which we expect to execute the task */
  guint gpu_id;

  /** Enable P2P (Peer to Peer) access between devices i.e.
   * between gpu_id and p2p_gpu_id */
  gint p2p_gpu_id;
  gint can_peer_access;

  /** Output Buffer Pool */
  GstBufferPool *pool;
  guint buffer_pool_size;

  /** NVTX Domain. */
  nvtxDomainHandle_t nvtx_domain;

  GstCaps *sinkcaps;
  GstCaps *srccaps;

  /** Maximum batch size. */
  guint batch_size;

  /** Frame number of the buffer, only used for debugging */
  guint frame_num;

  /** Type of NvBufSurface Memory to be allocated for output buffers */
  gint nvbuf_mem_type;

  /** Cuda Stream used while copying data */
  cudaStream_t cuda_xfer_stream;
};

/** GStreamer boilerplate. */
struct _GstNvXferClass
{
  GstBaseTransformClass parent_class;
};

GType gst_nvdsxfer_get_type (void);

G_END_DECLS
#endif /* __GST_NVDSXFER_H__ */
