/*
 * Copyright (C) 2014 Collabora Ltd.
 *     Author: Nicolas Dufresne <nicolas.dufresne@collabora.co.uk>
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LGPL-2.0-only
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 51 Franklin St, Fifth Floor,
 * Boston, MA 02110-1301, USA.
 *
 */

#ifndef __GST_V4L2_VIDEO_DEC_H__
#define __GST_V4L2_VIDEO_DEC_H__

#include <gst/gst.h>
#include <gst/video/video.h>
#include <gst/video/gstvideodecoder.h>
#include <gst/video/gstvideometa.h>

#include <gstv4l2object.h>
#include <gstv4l2bufferpool.h>

G_BEGIN_DECLS
#define GST_TYPE_V4L2_VIDEO_DEC \
  (gst_v4l2_video_dec_get_type())
#define GST_V4L2_VIDEO_DEC(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_V4L2_VIDEO_DEC,GstV4l2VideoDec))
#define GST_V4L2_VIDEO_DEC_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_V4L2_VIDEO_DEC,GstV4l2VideoDecClass))
#define GST_IS_V4L2_VIDEO_DEC(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_V4L2_VIDEO_DEC))
#define GST_IS_V4L2_VIDEO_DEC_CLASS(obj) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_V4L2_VIDEO_DEC))

/* The structures are renamed as the name conflicts with the
 * OSS v4l2 library structures. */
#ifdef USE_V4L2_TARGET_NV
#define GstV4l2VideoDec GstNvV4l2VideoDec
#define GstV4l2VideoDecClass GstNvV4l2VideoDecClass
#define  LOOP_COUNT_TO_WAIT_FOR_DQEVENT  10
#define  WAIT_TIME_PER_LOOP_FOR_DQEVENT 100*1000

#define VP8_START_BYTE_0 0x9D
#define VP8_START_BYTE_1 0x01

#define VP9_START_BYTE_0 0x49
#define VP9_START_BYTE_1 0x83
#define VP9_START_BYTE_2 0x42
#endif

typedef struct _GstV4l2VideoDec GstV4l2VideoDec;
typedef struct _GstV4l2VideoDecClass GstV4l2VideoDecClass;

struct _GstV4l2VideoDec
{
  GstVideoDecoder parent;

  /* < private > */
  GstV4l2Object *v4l2output;
  GstV4l2Object *v4l2capture;

  /* pads */
  GstCaps *probed_srccaps;
  GstCaps *probed_sinkcaps;

  /* State */
  GstVideoCodecState *input_state;

  gboolean active;
  GstFlowReturn output_flow;
  guint64 frame_num;
#ifdef USE_V4L2_TARGET_NV
  GHashTable* hash_pts_systemtime;
  guint64 decoded_picture_cnt;
  guint32 skip_frames;
  gboolean idr_received;
  guint32 drop_frame_interval;
  guint32 num_extra_surfaces;
  gboolean is_drc;
  gboolean disable_dpb;
  gboolean enable_full_frame;
  gboolean enable_frame_type_reporting;
  gboolean enable_error_check;
  gboolean enable_max_performance;
  gboolean set_format;
  gboolean is_gdr_stream;
  guint32 cudadec_mem_type;
  guint32 cudadec_gpu_id;
  guint32 cudadec_num_surfaces;
  gboolean cudadec_low_latency;
  gboolean extract_sei_type5_data;
  gboolean attach_sei_timestamps;
  gchar *sei_uuid_string;
  gdouble rate;
  guint32 cap_buf_dynamic_allocation;
  guint32 current_width;
  guint32 current_height;
  guint32 old_width;
  guint32 old_height;
  /* DPB frames required by the latest SPS and by the previous one */
  guint32 current_dpb_size;
  guint32 old_dpb_size;
  gboolean valid_vpx;
  GMutex pts_hashtable_lock;
  guint64 ts;
#endif
};

struct _GstV4l2VideoDecClass
{
  GstVideoDecoderClass parent_class;

  gchar *default_device;
};

GType gst_v4l2_video_dec_get_type (void);

gboolean gst_v4l2_is_video_dec (GstCaps * sink_caps, GstCaps * src_caps);
#ifdef USE_V4L2_TARGET_NV
gboolean set_v4l2_controls (GstV4l2VideoDec *self);
#endif
void gst_v4l2_video_dec_register (GstPlugin * plugin,
    const gchar * basename,
    const gchar * device_path, GstCaps * sink_caps, GstCaps * src_caps);

G_END_DECLS
#endif /* __GST_V4L2_VIDEO_DEC_H__ */
