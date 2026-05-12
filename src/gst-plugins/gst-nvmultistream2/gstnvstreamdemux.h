/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef __GST_NVSTREAMDEMUX_H__
#define __GST_NVSTREAMDEMUX_H__

#include <gst/gst.h>

G_BEGIN_DECLS
#define GST_TYPE_NVSTREAMDEMUX \
  (gst_nvstreamdemux_2_get_type ())
#define GST_NVSTREAMDEMUX(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_NVSTREAMDEMUX,GstNvStreamDemux))
#define GST_NVSTREAMDEMUX_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_NVSTREAMDEMUX,GstNvStreamDemuxClass))
#define GST_IS_NVSTREAMDEMUX(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_NVSTREAMDEMUX))
#define GST_IS_NVSTREAMDEMUX_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_NVSTREAMDEMUX))
typedef struct _GstNvStreamDemux GstNvStreamDemux;
typedef struct _GstNvStreamDemuxClass GstNvStreamDemuxClass;

struct _GstNvStreamDemux
{
  GstElement element;

  GstPad *sinkpad;

  GHashTable *pad_indexes;
  GHashTable *pad_framerates;
  GHashTable *pad_caps_is_raw;
  GHashTable *pad_stream_start_sent;
  GHashTable *eos_flag;

  guint num_surfaces_per_frame;

  GstCaps *sink_caps;

  GMutex ctx_lock;
  gboolean isAudio;
};

struct _GstNvStreamDemuxClass
{
  GstElementClass parent_class;
};

G_GNUC_INTERNAL GType gst_nvstreamdemux_2_get_type (void);

G_END_DECLS
#endif
