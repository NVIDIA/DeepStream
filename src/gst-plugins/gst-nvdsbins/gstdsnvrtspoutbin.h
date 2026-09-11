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

#ifndef __GST_DS_NVRTSPOUT_BIN_H__
#define __GST_DS_NVRTSPOUT_BIN_H__

#include <gst/gst.h>

#include "nvbufsurface.h"

G_BEGIN_DECLS

typedef enum
{
  RTSP_CODEC_H264 = 1,
  RTSP_CODEC_H265,
  RTSP_CODEC_MPEG4
} RTSPCodecType;

typedef enum
{
  RTSP_ENCODER_HW,
  RTSP_ENCODER_SW
} RTSPEncodeType;

typedef enum
{
  RTSP_PROFILE_BASELINE,
  RTSP_PROFILE_MAIN,
  RTSP_PROFILE_HIGH,
  RTSP_PROFILE_MAIN10
} RTSPProfileType;

typedef enum
{
  AUDIO_CODEC_OPUS = 1,
  AUDIO_CODEC_AC3
} AudioCodecType;

typedef struct _GstDsNvAudioBin
{
  GstElement *audio_sink;
  GstElement *audio_rtppay;
  GstElement *audio_encoder;
  GstElement *audio_cap_filter;
  GstElement *audio_transform;
  GstElement *audio_queue;
  GstElement *audio_parser;
  gint audio_udp_port;
  gint rate;
} GstDsNvAudioBin;

typedef struct _GstDsNvVideoBin
{
  GstElement *video_sink;
  GstElement *video_rtppay;
  GstElement *video_codecparse;
  GstElement *video_encoder;
  GstElement *video_cap_filter;
  GstElement *video_transform;
  GstElement *video_queue;
  gint video_udp_port;
} GstDsNvVideoBin;

/* Standard GStreamer boilerplate */
typedef struct _GstDsNvRtspOutBin
{
  GstBin bin;

  RTSPCodecType codec;
  AudioCodecType audio_codec;
  RTSPEncodeType enc_type;
  RTSPProfileType profile;
  guint iframeinterval;
  guint idrframeinterval;
  guint bitrate;
  gchar *file_location;
  gchar *rtsp_mount_point;
  guint rtsp_port;

  guint sync;
  guint qos;
  guint gpu_id;
  gboolean bypass;
  guint compute_hw;
  NvBufSurfaceMemType mem_type;
  gchar *create_from_config;

  GstDsNvVideoBin video_bin;
  GstDsNvAudioBin audio_bin;

  gboolean has_audio;
  gboolean has_video;

  GObject *server;

  GstPad *bin_vsink_pad;
  GstPad *bin_asink_pad;
} GstDsNvRtspOutBin;

typedef struct _GstDsNvRtspOutBinClass
{
  GstBinClass parent_class;
} GstDsNvRtspOutBinClass;


/* Standard GStreamer boilerplate */
#define GST_TYPE_DS_NVRTSPOUT_BIN (gst_ds_nvrtspout_bin_get_type())
#define GST_DS_NVRTSPOUT_BIN(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_DS_NVRTSPOUT_BIN,GstDsNvRtspOutBin))
#define GST_DS_NVRTSPOUT_BIN_CLASS(klass) (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_DS_NVRTSPOUT_BIN,GstDsNvRtspOutBinClass))
#define GST_DS_NVRTSPOUT_BIN_GET_CLASS(obj) (G_TYPE_INSTANCE_GET_CLASS((obj), GST_TYPE_DS_NVRTSPOUT_BIN, GstDsNvRtspOutBinClass))
#define GST_IS_DS_NVRTSPOUT_BIN(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_DS_NVRTSPOUT_BIN))
#define GST_IS_DS_NVRTSPOUT_BIN_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_DS_NVRTSPOUT_BIN))
#define GST_DS_NVRTSPOUT_BIN_CAST(obj)  ((GstDsNvRtspOutBin *)(obj))

GType gst_ds_nvrtspout_bin_get_type (void);

G_END_DECLS
#endif /* __GST_DS_NVRTSPOUT_BIN_H__ */
