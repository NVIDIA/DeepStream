/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef __GST_NVMULTISTREAMTILER_H__
#define __GST_NVMULTISTREAMTILER_H__

#include <gst/gst.h>
#include <gst/base/base.h>
#include <gst/video/video.h>

#include "cuda_runtime_api.h"
#include "gstnvdsmeta.h"
#include "INvTiler.h"

#ifdef __cplusplus
extern "C" {
#endif

G_BEGIN_DECLS
#define GST_TYPE_NVMULTISTREAMTILER \
  (gst_nvmultistreamtiler_get_type ())
#define GST_NVMULTISTREAMTILER(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_NVMULTISTREAMTILER,GstNvMultiStreamTiler))
#define GST_NVMULTISTREAMTILER_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_NVMULTISTREAMTILER,GstNvMultiStreamTilerClass))
#define GST_IS_NVMULTISTREAMTILER(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_NVMULTISTREAMTILER))
#define GST_IS_NVMULTISTREAMTILER_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_NVMULTISTREAMTILER))

typedef struct _GstNvMultiStreamTiler GstNvMultiStreamTiler;
typedef struct _GstNvMultiStreamTilerClass GstNvMultiStreamTilerClass;

#if 0
typedef struct _BBoxHolder
{
  int size; /* Total bboxes */
  int size_line; /* Total lines */
  NvDsFrameMeta *bboxes[100];
  NvDsLineMeta *num_line_meta[100];
}BBoxHolder;
#endif

struct _GstNvMultiStreamTiler
{
  GstBaseTransform basetrans;

  gint show_source;
  gint cuda_mem_type;
  gint compute_hw;
  gint interpolation_method;

  GstVideoInfo outvideoinfo;
  GstVideoInfo invideoinfo;
  guint gpu_id;

  GMutex tilerIfaceLock;

  guint num_surfaces_per_frame;
  guint buffer_pool_size;

  gboolean square_grid;
  std::map<uint32_t, uint32_t> *tiler_map;
  GMutex tiler_map_lock;
#if 0
  guint width_per_instance[3];
  guint height_per_instance[3];

  gdouble scale_x;
  gdouble scale_y;

  guint in_offsets[3];
  guint out_offsets[3];

  guint *out_inst_offsets[3];

  guint in_width[3];
  guint out_width[3];

  guint in_height[3];
  guint out_height[3];

  guint in_pstride[3];
  guint out_pstride[3];

  guint in_pitch;
  guint out_pitch;
  guint batchSize;

  gboolean *got_last_frame;
  gint *stream_last_batch_id_map;
  BBoxHolder *last_bboxes;
  FrameMeta_Params *last_frame_meta;
  void *last_frame;

  void *last_frame_ss;
  BBoxHolder last_bboxes_ss;
  gboolean copied_last_frame_ss;

  guint frame_num;
#endif
  guint frame_num;
  guint cur_sources;
  TilerConfig tilerConfig;
  INvTiler* tilerIface;
  GstBuffer* tilerScratchBufferGst;
  NvBufSurface* tilerScratchBuffer;
};

struct _GstNvMultiStreamTilerClass
{
  GstBaseTransformClass parent_class;
};

GType gst_nvmultistreamtiler_get_type (void);

gboolean allocate_tiler_memory (GstNvMultiStreamTiler *nvmultistreamtiler,
    guint rows, guint columns);

#ifdef ENABLE_GST_NVTILER_UNIT_TESTS
gboolean gGstNvTilerStaticInit();
#endif

G_END_DECLS

#ifdef __cplusplus
}
#endif

#endif
