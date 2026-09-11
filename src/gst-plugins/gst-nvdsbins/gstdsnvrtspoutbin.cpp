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

#include <gst/video/video.h>
#include <gst/audio/audio.h>
#include "gstdsnvrtspoutbin.h"
#include "gst-nvcommon.h"
#include "nvbufsurface.h"
#include "nvbufsurftransform.h"
#include "gstnvdsbinutils.h"

#include <gst/rtsp-server/rtsp-server.h>

GST_DEBUG_CATEGORY (gst_ds_nvrtspout_bin_debug);
#define GST_CAT_DEFAULT gst_ds_nvrtspout_bin_debug

/* Define our element type. Standard GObject/GStreamer boilerplate stuff */
#define gst_ds_nvrtspout_bin_parent_class parent_class
#define _do_init \
    GST_DEBUG_CATEGORY_INIT (gst_ds_nvrtspout_bin_debug, "nvrtspoutsinkbin", 0, "nvrtspoutsinkbin element");
G_DEFINE_TYPE_WITH_CODE (GstDsNvRtspOutBin, gst_ds_nvrtspout_bin,
    GST_TYPE_BIN, _do_init);

static void gst_ds_nvrtspout_bin_set_property (GObject * object,
    guint prop_id, const GValue * value, GParamSpec * spec);
static void gst_ds_nvrtspout_bin_get_property (GObject * object,
    guint prop_id, GValue * value, GParamSpec * spec);

static GstStateChangeReturn
gst_ds_nvrtspout_change_state (GstElement * element, GstStateChange transition);

static GstPad *gst_ds_nvrtspout_request_new_pad (GstElement * element,
    GstPadTemplate * templ, const gchar * name, const GstCaps * caps);
static void gst_ds_nvrtspout_release_pad (GstElement * element, GstPad * pad);

static gboolean create_video_pipeline (GstDsNvRtspOutBin * fbin);
static gboolean create_audio_pipeline (GstDsNvRtspOutBin * fbin);


#define GST_TYPE_NVDS_VIDEO_RTSP_CODEC_TYPE (gst_nvds_video_enc_rtsp_codec_get_type())
#define GST_TYPE_NVDS_AUDIO_RTSP_CODEC_TYPE (gst_nvds_audio_enc_rtsp_codec_get_type())
#define GstNvComputeHWType (gst_compute_hw_get_type())

static GType
gst_nvds_video_enc_rtsp_codec_get_type (void)
{
  static GType video_enc_codec_type = 0;
  static const GEnumValue video_enc_codecs[] = {
    {RTSP_CODEC_H264, "H.264 encoder", "h264"},
    {RTSP_CODEC_H265, "H.265 encoder", "h265"},
    {RTSP_CODEC_MPEG4, "MPEG4 encoder", "mpeg4"},
    {0, NULL, NULL},
  };

  if (!video_enc_codec_type) {
    video_enc_codec_type =
        g_enum_register_static ("GstNvDsVideoEncCodec1Type", video_enc_codecs);
  }
  return video_enc_codec_type;
}


static GType
gst_nvds_audio_enc_rtsp_codec_get_type (void)
{
  static GType audio_enc_codec_type = 0;
  static const GEnumValue audio_enc_codecs[] = {
    {AUDIO_CODEC_OPUS, "Opus codec", "opus"},
    {AUDIO_CODEC_AC3, "Ac3 codec", "ac3"},
    {0, NULL, NULL},
  };

  if (!audio_enc_codec_type) {
    audio_enc_codec_type =
        g_enum_register_static ("GstNvDsAudioEncCodec1Type", audio_enc_codecs);
  }
  return audio_enc_codec_type;
}


#define GST_TYPE_NVDS_VIDEO_RTSP_ENC_TYPE (gst_nvds_video_rtsp_enc_type_get_type())
static GType
gst_nvds_video_rtsp_enc_type_get_type (void)
{
  static GType video_enc_type_type = 0;
  static const GEnumValue video_enc_types[] = {
    {RTSP_ENCODER_HW, "Hardware accelerated encoder", "hw"},
    {RTSP_ENCODER_SW, "Software encoder", "sw"},
    {0, NULL, NULL},
  };

  if (!video_enc_type_type) {
    video_enc_type_type =
        g_enum_register_static ("GstNvDsVideoEncType1Type", video_enc_types);
  }
  return video_enc_type_type;
}

#define GST_TYPE_NVDS_VIDEO_RTSP_ENC_PROFILE (gst_nvds_video_rtsp_enc_profile_get_type())
static GType
gst_nvds_video_rtsp_enc_profile_get_type (void)
{
  static GType video_enc_profile_type = 0;
  static const GEnumValue video_enc_profiles[] = {
    {RTSP_PROFILE_BASELINE, "Baseline Profile (H.264 only)", "baseline"},
    {RTSP_PROFILE_MAIN, "Main Profile (H.264 & H.265)", "main"},
    {RTSP_PROFILE_HIGH, "High Profile (H.264 only)", "high"},
    {RTSP_PROFILE_MAIN10, "Main10 Profile (H.265 only))",
        "main10"},
    {0, NULL, NULL},
  };

  if (!video_enc_profile_type) {
    video_enc_profile_type =
        g_enum_register_static ("GstNvDsVideoEncProfile1Type",
        video_enc_profiles);
  }
  return video_enc_profile_type;
}

enum
{
  PROP_FIRST,
  PROP_CODEC,
  PROP_ENC_TYPE,
  PROP_BITRATE,
  PROP_PROFILE,
  PROP_IFRAMEINTERVAL,
  PROP_IDRFRAMEINTERVAL,
  PROP_OUTPUT_FILE,
  PROP_RTSP_PORT,
  PROP_RTSP_MOUNT_POINT,
  PROP_SYNC,
  PROP_QOS,
  PROP_GPU_DEVICE_ID,
  PROP_NVBUF_MEMORY_TYPE,
  PROP_CREATE_FROM_CONFIG,
  PROP_AUDIO_CODEC,
  PROP_COMPUTE_HW,
  PROP_BYPASS,
  PROP_LAST
};

#define DEFAULT_CODEC RTSP_CODEC_H264
#define DEFAULT_ENC_TYPE RTSP_ENCODER_HW
#define DEFAULT_PROFILE RTSP_PROFILE_BASELINE
#define DEFAULT_BITRATE 0
#define DEFAULT_IFRAMEINTERVAL 30
#define DEFAULT_IDRFRAMEINTERVAL 256
#define DEFAULT_RTSP_PORT 8554
#define DEFAULT_RTSP_MOUNT_POINT "/ds-test"
#define DEFAULT_SYNC TRUE
#define DEFAULT_QOS FALSE
#define DEFAULT_BYPASS_CODECS_MODE FALSE
#define DEFAULT_GPU_ID 0
#define DEFAULT_NVBUF_MEMORY_TYPE NVBUF_MEM_DEFAULT
#define DEFAULT_OUTPUT_FILE ""
#define DEFAULT_CREATE_FROM_CONFIG ""
#define DEFAULT_AUDIO_CLOCK_RATE 44100
#define DEFAULT_AUDIO_CODEC AUDIO_CODEC_OPUS
#define DEFAULT_COMPUTE_HW NvBufSurfTransformCompute_Default

static GstStaticPadTemplate gst_nvrtspoutbin_asink_template =
GST_STATIC_PAD_TEMPLATE ("asink",
    GST_PAD_SINK,
    GST_PAD_REQUEST,
    GST_STATIC_CAPS (GST_AUDIO_CAPS_MAKE (GST_AUDIO_FORMATS_ALL)));

static GstStaticPadTemplate gst_nvrtspoutbin_vsink_template =
    GST_STATIC_PAD_TEMPLATE ("vsink",
    GST_PAD_SINK,
    GST_PAD_REQUEST,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES ("memory:NVMM",
            "{ "
            "I420,  NV12, P010_10LE, BGRx, RGBA, GRAY8, YUY2, UYVY, YVYU, Y42B }")
        ";"  GST_VIDEO_CAPS_MAKE ("{ "
            "I420, P010_10LE, NV12, BGRx, RGBA, GRAY8, YUY2, UYVY, YVYU, Y42B }")
        ";" "video/x-h264,stream-format=(string)byte-stream,alignment=(string)au"
        ";" "video/x-h265,stream-format=(string)byte-stream,alignment=(string)au"));

static void
gst_ds_nvrtspout_bin_class_init (GstDsNvRtspOutBinClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;

  gobject_class = G_OBJECT_CLASS (klass);
  gstelement_class = GST_ELEMENT_CLASS (klass);

  gobject_class->set_property =
      GST_DEBUG_FUNCPTR (gst_ds_nvrtspout_bin_set_property);
  gobject_class->get_property =
      GST_DEBUG_FUNCPTR (gst_ds_nvrtspout_bin_get_property);

  g_object_class_install_property (gobject_class, PROP_CODEC,
      g_param_spec_enum ("codec", "Codec",
          "Type of codec to use",
          GST_TYPE_NVDS_VIDEO_RTSP_CODEC_TYPE, DEFAULT_CODEC, (GParamFlags)
          (G_PARAM_READWRITE |
              G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_AUDIO_CODEC,
      g_param_spec_enum ("audio-codec-type", "Audio Codec Type",
          "Type of audio codec to use",
          GST_TYPE_NVDS_AUDIO_RTSP_CODEC_TYPE, DEFAULT_AUDIO_CODEC, (GParamFlags)
          (G_PARAM_READWRITE |
              G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_ENC_TYPE,
      g_param_spec_enum ("enc-type",
          "Encoder Type",
          "Type of encoder to use",
          GST_TYPE_NVDS_VIDEO_RTSP_ENC_TYPE, DEFAULT_ENC_TYPE, (GParamFlags)
          (G_PARAM_READWRITE |
              G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_PROFILE,
      g_param_spec_enum ("profile", "Profile",
          "Encoder profile to use",
          GST_TYPE_NVDS_VIDEO_RTSP_ENC_PROFILE, DEFAULT_PROFILE, (GParamFlags)
          (G_PARAM_READWRITE |
              G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_BITRATE,
      g_param_spec_uint ("bitrate", "Bitrate",
          "Encoding bitrate in bits/sec",
          0, G_MAXUINT, DEFAULT_BITRATE, (GParamFlags)
          (G_PARAM_READWRITE |
              G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_IFRAMEINTERVAL,
      g_param_spec_uint ("iframeinterval",
          "I-Frame Interval",
          "Encoding Intra Frame occurance frequency (H/W encoder only)",
          0, G_MAXUINT, DEFAULT_IFRAMEINTERVAL, (GParamFlags)
          (G_PARAM_READWRITE |
              G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_IDRFRAMEINTERVAL,
        g_param_spec_uint ("idrinterval", "IDR Frame interval",
            "Encoding IDR Frame occurance frequency (H/W encoder only)",
            0, G_MAXUINT, DEFAULT_IDRFRAMEINTERVAL, (GParamFlags)
          (G_PARAM_READWRITE |
              G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_RTSP_PORT,
      g_param_spec_uint ("rtsp-port",
          "RTSP PORT",
          "Port on which the RTSP server would listen",
          0, G_MAXUINT, DEFAULT_RTSP_PORT, (GParamFlags)
          (G_PARAM_READWRITE |
              G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));


  g_object_class_install_property (gobject_class, PROP_SYNC,
      g_param_spec_boolean ("sync", "Sync",
          "Sync on the clock", DEFAULT_SYNC, (GParamFlags)
          (G_PARAM_READWRITE |
              G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_QOS,
      g_param_spec_boolean ("qos", "QoS",
          "Generate Quality - of - Service events upstream",
          DEFAULT_QOS, (GParamFlags)
          (G_PARAM_READWRITE |
              G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_BYPASS,
      g_param_spec_boolean ("bypass-codecs", "bypass-codecs",
          "Stream H264 & H265 video bitstream as received without transcoding",
          DEFAULT_BYPASS_CODECS_MODE, (GParamFlags)
          (G_PARAM_READWRITE |
              G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));

  PROP_NVDS_GPU_ID_INSTALL (gobject_class);

  PROP_NVBUF_MEMORY_TYPE_INSTALL (gobject_class);

  g_object_class_install_property (gobject_class, PROP_OUTPUT_FILE,
      g_param_spec_string ("output-file",
          "Output File",
          "Type of Video sink to use", DEFAULT_OUTPUT_FILE, (GParamFlags)
          (G_PARAM_READWRITE |
              G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_RTSP_MOUNT_POINT,
      g_param_spec_string ("rtsp-mount-point",
          "rtsp mount point",
          "name of rtsp mount point to use,path is of the form /node", DEFAULT_RTSP_MOUNT_POINT, (GParamFlags)
          (G_PARAM_READWRITE |
              G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_COMPUTE_HW,
      g_param_spec_enum ("compute-hw", "compute hw",
          "Compute Scaling HW", GstNvComputeHWType, DEFAULT_COMPUTE_HW, (GParamFlags)
          (G_PARAM_READWRITE |
              G_PARAM_STATIC_STRINGS | GST_PARAM_CONTROLLABLE)));

  gst_element_class_add_static_pad_template (gstelement_class,
      &gst_nvrtspoutbin_asink_template);
  gst_element_class_add_static_pad_template (gstelement_class,
      &gst_nvrtspoutbin_vsink_template);

  gstelement_class->change_state =
      GST_DEBUG_FUNCPTR (gst_ds_nvrtspout_change_state);

  gstelement_class->request_new_pad =
      GST_DEBUG_FUNCPTR (gst_ds_nvrtspout_request_new_pad);

  gstelement_class->release_pad =
      GST_DEBUG_FUNCPTR (gst_ds_nvrtspout_release_pad);

  /* Set metadata describing the element */
  gst_element_class_set_details_simple (gstelement_class,
      "NvRtspOut Bin", "NvRtspOut Bin",
      "Nvidia DeepStreamSDK RTSP Sink Bin. Internal Pipeline: queue->nvvideoconvert->encoder->codecparse->rtppay->udpsink.",
      "NVIDIA Corporation. Deepstream for Tesla forum: "
      "https://devtalk.nvidia.com/default/board/209");
}

static void
gst_ds_nvrtspout_bin_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstDsNvRtspOutBin *nvrtspoutbin = GST_DS_NVRTSPOUT_BIN (object);

  switch (prop_id) {
    case PROP_CODEC:
      nvrtspoutbin->codec = (RTSPCodecType) g_value_get_enum (value);
      break;
    case PROP_AUDIO_CODEC:
      nvrtspoutbin->audio_codec = (AudioCodecType) g_value_get_enum (value);
      break;
    case PROP_PROFILE:
      nvrtspoutbin->profile = (RTSPProfileType) g_value_get_enum (value);
      break;
    case PROP_BITRATE:
      nvrtspoutbin->bitrate = g_value_get_uint (value);
      break;
    case PROP_IFRAMEINTERVAL:
      nvrtspoutbin->iframeinterval = g_value_get_uint (value);
      break;
    case PROP_IDRFRAMEINTERVAL:
      nvrtspoutbin->idrframeinterval = g_value_get_uint (value);
      break;
    case PROP_RTSP_PORT:
      nvrtspoutbin->rtsp_port = g_value_get_uint (value);
      break;
    case PROP_ENC_TYPE:
      nvrtspoutbin->enc_type = (RTSPEncodeType) g_value_get_enum (value);
      break;
    case PROP_SYNC:
      nvrtspoutbin->sync = g_value_get_boolean (value);
      break;
    case PROP_QOS:
      nvrtspoutbin->qos = g_value_get_boolean (value);
      break;
    case PROP_GPU_DEVICE_ID:
      nvrtspoutbin->gpu_id = g_value_get_uint (value);
      break;
    case PROP_NVBUF_MEMORY_TYPE:
      nvrtspoutbin->mem_type = (NvBufSurfaceMemType) g_value_get_enum (value);
      break;
    case PROP_CREATE_FROM_CONFIG:
      g_free (nvrtspoutbin->create_from_config);
      nvrtspoutbin->create_from_config = g_value_dup_string (value);
      break;
    case PROP_OUTPUT_FILE:
      g_free (nvrtspoutbin->file_location);
      nvrtspoutbin->file_location = g_value_dup_string (value);
      break;
    case PROP_RTSP_MOUNT_POINT:
      g_free (nvrtspoutbin->rtsp_mount_point);
      nvrtspoutbin->rtsp_mount_point = g_value_dup_string (value);
      break;
    case PROP_COMPUTE_HW:
      nvrtspoutbin->compute_hw = (NvBufSurfTransform_Compute) g_value_get_enum (value);
      break;
    case PROP_BYPASS:
      nvrtspoutbin->bypass = g_value_get_boolean (value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}


static void
gst_ds_nvrtspout_bin_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstDsNvRtspOutBin *nvrtspoutbin = GST_DS_NVRTSPOUT_BIN (object);

  switch (prop_id) {
    case PROP_CODEC:
      g_value_set_enum (value, nvrtspoutbin->codec);
      break;
    case PROP_AUDIO_CODEC:
      g_value_set_enum (value, nvrtspoutbin->audio_codec);
      break;
    case PROP_ENC_TYPE:
      g_value_set_enum (value, nvrtspoutbin->enc_type);
      break;
    case PROP_PROFILE:
      g_value_set_enum (value, nvrtspoutbin->profile);
      break;
    case PROP_BITRATE:
      g_value_set_uint (value, nvrtspoutbin->bitrate);
      break;
    case PROP_IFRAMEINTERVAL:
      g_value_set_uint (value, nvrtspoutbin->iframeinterval);
      break;
    case PROP_IDRFRAMEINTERVAL:
      g_value_set_uint (value, nvrtspoutbin->idrframeinterval);
      break;
    case PROP_RTSP_PORT:
      g_value_set_uint (value, nvrtspoutbin->rtsp_port);
      break;
    case PROP_SYNC:
      g_value_set_boolean (value, nvrtspoutbin->sync);
      break;
    case PROP_QOS:
      g_value_set_boolean (value, nvrtspoutbin->qos);
      break;
    case PROP_GPU_DEVICE_ID:
      g_value_set_uint (value, nvrtspoutbin->gpu_id);
      break;
    case PROP_NVBUF_MEMORY_TYPE:
      g_value_set_enum (value, nvrtspoutbin->mem_type);
      break;
    case PROP_OUTPUT_FILE:
      g_value_set_string (value, nvrtspoutbin->file_location);
      break;
    case PROP_RTSP_MOUNT_POINT:
      g_value_set_string (value, nvrtspoutbin->rtsp_mount_point);
      break;
    case PROP_COMPUTE_HW:
      g_value_set_enum (value, nvrtspoutbin->compute_hw);
      break;
    case PROP_BYPASS:
      g_value_set_boolean (value, nvrtspoutbin->bypass);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}


static gboolean
start_rtsp_server (GstDsNvRtspOutBin * rbin)
{
  GstRTSPMountPoints *mounts;
  GstRTSPMediaFactory *factory;
  GstRTSPServer *server;
  gchar *udpsrc_pipeline = NULL;

  gchar port_num_Str[64];
  gchar encoder_name[32];

  guint64 udp_buffer_size = 512 * 1024;

  if (rbin->codec == RTSP_CODEC_H264) {
    g_snprintf (encoder_name, sizeof (encoder_name), "H264");
  } else if (rbin->codec == RTSP_CODEC_H265) {
    g_snprintf (encoder_name, sizeof (encoder_name), "H265");
  } else {
    g_snprintf (encoder_name, sizeof (encoder_name), "MPEG4");
  }


  if (rbin->has_video && rbin->has_audio) {
    GST_INFO_OBJECT(rbin, "Streaming audio+video");

    if (rbin->audio_codec == AUDIO_CODEC_AC3) {
      udpsrc_pipeline =
          g_strdup_printf
          ("( udpsrc name=pay0 port=%d buffer-size=%lu caps=\"application/x-rtp, media=video, "
          "clock-rate=90000, encoding-name=%s, payload=96 \" "
          "udpsrc name=pay1 port=%d buffer-size=%lu caps=\"application/x-rtp, media=audio, "
          "clock-rate=%d, encoding-name=AC3, payload=96 \" )",
          rbin->video_bin.video_udp_port, udp_buffer_size, encoder_name,
          rbin->audio_bin.audio_udp_port, udp_buffer_size, rbin->audio_bin.rate);
    }

    if (rbin->audio_codec == AUDIO_CODEC_OPUS) {
      udpsrc_pipeline =
          g_strdup_printf
          ("( udpsrc name=pay0 port=%d buffer-size=%lu caps=\"application/x-rtp, media=video, "
          "clock-rate=90000, encoding-name=%s, payload=96 \" "
          "udpsrc name=pay1 port=%d buffer-size=%lu caps=\"application/x-rtp, media=audio, "
          "clock-rate=%d, encoding-name=OPUS, payload=96 \" )",
          rbin->video_bin.video_udp_port, udp_buffer_size, encoder_name,
          rbin->audio_bin.audio_udp_port, udp_buffer_size, rbin->audio_bin.rate);
    }

  } else if (!rbin->has_video && rbin->has_audio) {
    GST_INFO_OBJECT(rbin, "Streaming audio only");

    if (rbin->audio_codec == AUDIO_CODEC_AC3) {

    udpsrc_pipeline =
        g_strdup_printf
        ("( udpsrc name=pay0 port=%d buffer-size=%lu caps=\"application/x-rtp, media=audio, "
        "clock-rate=%d, encoding-name=AC3 ,payload=96 \" )",
        rbin->audio_bin.audio_udp_port, udp_buffer_size, rbin->audio_bin.rate);
    }

    if (rbin->audio_codec == AUDIO_CODEC_OPUS) {

    udpsrc_pipeline =
        g_strdup_printf
        ("( udpsrc name=pay0 port=%d buffer-size=%lu caps=\"application/x-rtp, media=audio, "
        "clock-rate=%d, encoding-name=OPUS ,payload=96 \" )",
        rbin->audio_bin.audio_udp_port, udp_buffer_size, rbin->audio_bin.rate);
    }

  }
  else if (rbin->has_video && !rbin->has_audio) {
    GST_INFO_OBJECT(rbin, "Streaming video only");
    udpsrc_pipeline =
        g_strdup_printf
        ("( udpsrc name=pay0 port=%d buffer-size=%lu caps=\"application/x-rtp, media=video, "
        "clock-rate=90000, encoding-name=%s, payload=96 \" )",
        rbin->video_bin.video_udp_port, udp_buffer_size, encoder_name);
  }

  g_snprintf (port_num_Str, sizeof (port_num_Str), "%d", rbin->rtsp_port);

  if (!rbin->server) {
    server = gst_rtsp_server_new ();
    if (!server) {
      GST_ELEMENT_ERROR (rbin, RESOURCE, FAILED,
          ("Failed to create a new RTSP server"), (NULL));
      if (udpsrc_pipeline) {
        g_free(udpsrc_pipeline);
      }
      return FALSE;
    }
    g_object_set (server, "service", port_num_Str, NULL);
    g_print
        ("\n *** %s: Launched RTSP Streaming at rtsp://localhost:%d%s ***\n\n",
        GST_ELEMENT_NAME (rbin), rbin->rtsp_port, rbin->rtsp_mount_point);
  } else {
    server = (GstRTSPServer *) rbin->server;
  }

  mounts = gst_rtsp_server_get_mount_points (server);

  factory = gst_rtsp_media_factory_new ();
  gst_rtsp_media_factory_set_launch (factory, udpsrc_pipeline);

  gst_rtsp_mount_points_add_factory (mounts, rbin->rtsp_mount_point, factory);

  g_object_unref (mounts);

  gst_rtsp_server_attach (server, NULL);

  rbin->server = G_OBJECT (server);

  if (udpsrc_pipeline) {
    g_free(udpsrc_pipeline);
  }
  return TRUE;
}

static gint
get_free_port ()
{
  GstElement *udpsrc = gst_element_factory_make ("udpsrc", "testsrc");
  g_object_set (G_OBJECT (udpsrc), "port", 0, NULL);

  gst_element_set_state (udpsrc, GST_STATE_PAUSED);
  GstStateChangeReturn ret =
      gst_element_get_state (udpsrc, NULL, NULL, GST_CLOCK_TIME_NONE);

  if (ret != GST_STATE_CHANGE_SUCCESS && ret != GST_STATE_CHANGE_NO_PREROLL) {
    gst_object_unref (udpsrc);
    return -1;
  }

  gint port = 0;
  g_object_get (udpsrc, "port", &port, NULL);

  gst_element_set_state (udpsrc, GST_STATE_NULL);
  gst_element_get_state (udpsrc, NULL, NULL, GST_CLOCK_TIME_NONE);
  gst_object_unref (udpsrc);
  return port;
}

static GstPadProbeReturn
seek_query_drop_prob (GstPad * pad, GstPadProbeInfo * info, gpointer u_data)
{
  if (GST_PAD_PROBE_INFO_TYPE (info) & GST_PAD_PROBE_TYPE_QUERY_UPSTREAM) {
    GstQuery *query = GST_PAD_PROBE_INFO_QUERY (info);
    if (GST_QUERY_TYPE (query) == GST_QUERY_SEEKING) {
      return GST_PAD_PROBE_DROP;
    }
  }
  return GST_PAD_PROBE_OK;
}

static GstPadProbeReturn
audio_caps_negotiate_probe (GstPad * pad, GstPadProbeInfo * info, gpointer u_data)
{
  GstDsNvRtspOutBin *bin = (GstDsNvRtspOutBin *) u_data;
  if (info->type & GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM) {
    GstEvent *event = GST_PAD_PROBE_INFO_EVENT (info);
    if (GST_EVENT_TYPE (event) == GST_EVENT_CAPS) {
      GstCaps *caps;
      gst_event_parse_caps(event, &caps);
      GstStructure *str = gst_caps_get_structure (caps, 0);
      gint rate = 0;
      if (gst_structure_get_int (str, "clock-rate", &rate)) {
        bin->audio_bin.rate = rate;
        if (!start_rtsp_server(bin)) {
          GST_ELEMENT_ERROR (bin, RESOURCE, FAILED,
            ("Failed to start RTSP SERVER"), (NULL));
        }
      }
    }
  }
  return GST_PAD_PROBE_OK;
}

#if 0
static GstPadProbeReturn
encoder_src_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info,
    gpointer u_data)
{
    GstBuffer *buf = (GstBuffer *) info->data;
    g_print("***************** encoder src pad probe added ****************\n");
    //GST_BUFFER_PTS(buf) = buf->pts - GST_SECOND * 60 * 60 * 1000;
    g_print("********************************gstbuf output PTS = %ld\n", buf->pts);
    return GST_PAD_PROBE_OK;
}

static GstPadProbeReturn
encoder_sink_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info,
    gpointer u_data)
{
    GstBuffer *buf = (GstBuffer *) info->data;
    //GST_BUFFER_PTS(buf) = buf->pts - GST_SECOND * 60 * 60 * 1000;
    g_print("********************************************gstbuf input PTS = %ld\n", buf->pts);
    return GST_PAD_PROBE_OK;
}

#endif

static gboolean
create_video_pipeline (GstDsNvRtspOutBin * fbin)
{
  gchar elem_name[128];
  fbin->video_bin.video_udp_port = get_free_port ();

  g_snprintf (elem_name, sizeof (elem_name), "%s-video_queue",
      GST_ELEMENT_NAME (fbin));
  fbin->video_bin.video_queue = gst_element_factory_make ("queue", elem_name);
  if (!fbin->video_bin.video_queue) {
    GST_ELEMENT_ERROR (fbin, RESOURCE, NOT_FOUND,
        ("Failed to create 'video queue' element"), (NULL));
    return FALSE;
  }
  gst_bin_add (GST_BIN (fbin), fbin->video_bin.video_queue);

  g_snprintf (elem_name, sizeof (elem_name), "%s-video_transform",
      GST_ELEMENT_NAME (fbin));
  fbin->video_bin.video_transform =
      gst_element_factory_make ("nvvideoconvert", elem_name);
  if (!fbin->video_bin.video_transform) {
    GST_ELEMENT_ERROR (fbin, RESOURCE, NOT_FOUND,
        ("Failed to create 'nvvideoconvert' element"), (NULL));
    return FALSE;
  }
  gst_bin_add (GST_BIN (fbin), fbin->video_bin.video_transform);
  if (g_object_class_find_property
      (G_OBJECT_GET_CLASS (fbin->video_bin.video_transform), "gpu-id")) {
    g_object_set (G_OBJECT (fbin->video_bin.video_transform), "gpu-id",
        fbin->gpu_id, NULL);
  }
  if (g_object_class_find_property
      (G_OBJECT_GET_CLASS (fbin->video_bin.video_transform),
          "nvbuf-memory-type")) {
    g_object_set (G_OBJECT (fbin->video_bin.video_transform),
        "nvbuf-memory-type", fbin->mem_type, NULL);
  }

  g_snprintf (elem_name, sizeof (elem_name), "%s-video_capsfilter",
      GST_ELEMENT_NAME (fbin));
  fbin->video_bin.video_cap_filter =
      gst_element_factory_make ("capsfilter", elem_name);
  if (!fbin->video_bin.video_cap_filter) {
    GST_ELEMENT_ERROR (fbin, RESOURCE, NOT_FOUND,
        ("Failed to create ' video capsfilter' element"), (NULL));
    return FALSE;
  }
  gst_bin_add (GST_BIN (fbin), fbin->video_bin.video_cap_filter);

  if (fbin->codec == RTSP_CODEC_MPEG4 && fbin->enc_type == RTSP_ENCODER_HW) {
    GST_ELEMENT_WARNING (fbin, LIBRARY, SETTINGS,
        ("Overriding to software encoder."
            " MPEG4 is supported by software encoder only"), (NULL));
    fbin->enc_type = RTSP_ENCODER_SW;
  }

  g_snprintf (elem_name, sizeof (elem_name), "%s-video_encoder",
      GST_ELEMENT_NAME (fbin));
  RTSPEncodeType enc_type = fbin->enc_type;
  switch (fbin->codec) {
    case RTSP_CODEC_H264:
      if (enc_type == RTSP_ENCODER_SW) {
        fbin->video_bin.video_encoder =
          gst_element_factory_make ("x264enc", elem_name);
      } else {
        fbin->video_bin.video_encoder =
            gst_element_factory_make ("nvv4l2h264enc", elem_name);
        if (!fbin->video_bin.video_encoder) {
          GST_ELEMENT_WARNING (fbin, LIBRARY, SETTINGS,
              ("Could not create HW encoder. Falling back to SW encoder"), (NULL));
          fbin->video_bin.video_encoder =
            gst_element_factory_make ("x264enc", elem_name);
          fbin->enc_type = RTSP_ENCODER_SW;
        }
      }
      break;
    case RTSP_CODEC_H265:
      if (enc_type == RTSP_ENCODER_SW) {
        fbin->video_bin.video_encoder =
          gst_element_factory_make ("x265enc", elem_name);
      } else {
        fbin->video_bin.video_encoder =
            gst_element_factory_make ("nvv4l2h265enc", elem_name);
        if (!fbin->video_bin.video_encoder) {
          GST_ELEMENT_WARNING (fbin, LIBRARY, SETTINGS,
              ("Could not create HW encoder. Falling back to SW encoder"), (NULL));
          fbin->video_bin.video_encoder =
            gst_element_factory_make ("x265enc", elem_name);
          fbin->enc_type = RTSP_ENCODER_SW;
        }
      }
      break;
    case RTSP_CODEC_MPEG4:
      fbin->video_bin.video_encoder =
          gst_element_factory_make ("avenc_mpeg4", elem_name);
      break;
    default:
      break;
  }

  if (!fbin->video_bin.video_encoder) {
    GST_ELEMENT_ERROR (fbin, RESOURCE, NOT_FOUND,
        ("Failed to create encoder element"), (NULL));
    return FALSE;
  }
  gst_bin_add (GST_BIN (fbin), fbin->video_bin.video_encoder);

  GstCaps *caps;
  if (fbin->codec == RTSP_CODEC_MPEG4 || fbin->enc_type == RTSP_ENCODER_SW)
    caps = gst_caps_from_string ("video/x-raw, format=I420");
  else
    caps = gst_caps_from_string ("video/x-raw(memory:NVMM), format=I420");
  g_object_set (G_OBJECT (fbin->video_bin.video_cap_filter), "caps", caps,
      NULL);
  gst_caps_unref (caps);

  GstElement *swenc_caps = NULL;
  if (fbin->codec == RTSP_CODEC_H264 && fbin->enc_type == RTSP_ENCODER_SW) {
    GstCaps *enc_caps = NULL;

    swenc_caps =  gst_element_factory_make ("capsfilter", NULL);
    enc_caps = gst_caps_from_string ("video/x-h264, profile=(string)baseline");
    g_object_set (G_OBJECT (swenc_caps), "caps", enc_caps, NULL);

    gst_caps_unref (enc_caps);
  }
  else {
    swenc_caps =  gst_element_factory_make ("identity", NULL);
  }

  gst_bin_add (GST_BIN (fbin), swenc_caps);

  if (fbin->codec == RTSP_CODEC_H264 || fbin->codec == RTSP_CODEC_H265) {
    if (fbin->enc_type == RTSP_ENCODER_SW) {
    g_object_set (G_OBJECT (fbin->video_bin.video_encoder), "tune", 4, NULL);
    }
  }

  NVGSTDS_ELEM_ADD_PROBE (fbin, fbin->video_bin.video_encoder, "sink",
      seek_query_drop_prob, GST_PAD_PROBE_TYPE_QUERY_UPSTREAM, fbin);

  if (g_object_class_find_property
      (G_OBJECT_GET_CLASS (fbin->video_bin.video_encoder), "gpu-id")) {
    g_object_set (G_OBJECT (fbin->video_bin.video_encoder), "gpu-id",
        fbin->gpu_id, NULL);
  }

  if (fbin->enc_type == RTSP_ENCODER_HW) {
    if (g_object_class_find_property
        (G_OBJECT_GET_CLASS (fbin->video_bin.video_encoder), "preset-level")) {
      g_object_set (G_OBJECT (fbin->video_bin.video_encoder),
          "preset-level", 1, NULL);
    }
    if (g_object_class_find_property
        (G_OBJECT_GET_CLASS (fbin->video_bin.video_encoder),
            "insert-sps-pps")) {
      g_object_set (G_OBJECT (fbin->video_bin.video_encoder),
          "insert-sps-pps", 1, NULL);
    }
    if (g_object_class_find_property
        (G_OBJECT_GET_CLASS (fbin->video_bin.video_encoder),
            "bufapi-version")) {
      g_object_set (G_OBJECT (fbin->video_bin.video_encoder),
          "bufapi-version", 1, NULL);
    }
    switch (fbin->profile) {
      case RTSP_PROFILE_BASELINE:
        if (fbin->codec == RTSP_CODEC_H265) {
          GST_ELEMENT_WARNING (fbin, LIBRARY, SETTINGS,
              ("H.265 does not support 'Baseline' profile. Overriding to 'Main' profile."),
              (NULL));
          g_object_set (G_OBJECT (fbin->video_bin.video_encoder),
              "profile", 0, NULL);
        } else {
          g_object_set (G_OBJECT (fbin->video_bin.video_encoder),
              "profile", 0, NULL);
        }
        break;
      case RTSP_PROFILE_MAIN:
        g_object_set (G_OBJECT (fbin->video_bin.video_encoder), "profile",
            fbin->codec == RTSP_CODEC_H265 ? 0 : 2, NULL);
        break;
      case RTSP_PROFILE_HIGH:
        if (fbin->codec == RTSP_CODEC_H265) {
          GST_ELEMENT_WARNING (fbin, LIBRARY, SETTINGS,
              ("H.265 does not support 'Baseline' profile. Overriding to 'Main' profile."),
              (NULL));
          g_object_set (G_OBJECT (fbin->video_bin.video_encoder),
              "profile", 0, NULL);
        } else {
          g_object_set (G_OBJECT (fbin->video_bin.video_encoder),
              "profile", 4, NULL);
        }
        break;
      case RTSP_PROFILE_MAIN10:
        if (fbin->codec == RTSP_CODEC_H265) {
          g_object_set (G_OBJECT (fbin->video_bin.video_encoder),
              "profile", 1, NULL);
        } else {
          GST_ELEMENT_WARNING (fbin, LIBRARY, SETTINGS,
              ("H.264 does not support 'Main10' profile. Overriding to 'Baseline' profile."),
              (NULL));
          g_object_set (G_OBJECT (fbin->video_bin.video_encoder),
              "profile", 0, NULL);
        }
        break;
      default:
        break;
    }

    g_object_set (G_OBJECT (fbin->video_bin.video_encoder),
        "iframeinterval", fbin->iframeinterval, NULL);
    g_object_set (G_OBJECT (fbin->video_bin.video_encoder),
        "idrinterval", fbin->idrframeinterval, NULL);
    g_object_set (G_OBJECT (fbin->video_bin.video_encoder), "bitrate",
        fbin->bitrate, NULL);
  } else {
    if (fbin->codec == RTSP_CODEC_MPEG4)
      g_object_set (G_OBJECT (fbin->video_bin.video_encoder), "bitrate",
          fbin->bitrate, NULL);
    else {
      //bitrate is in kbits/sec for software encoder x264enc and x265enc
      g_object_set (G_OBJECT (fbin->video_bin.video_encoder), "bitrate",
          fbin->bitrate / 1000, "key-int-max", fbin->iframeinterval, NULL);
    }
  }

  g_snprintf (elem_name, sizeof (elem_name), "%s-video_codecparser",
      GST_ELEMENT_NAME (fbin));
  switch (fbin->codec) {
    case RTSP_CODEC_H264:
      fbin->video_bin.video_codecparse =
          gst_element_factory_make ("h264parse", elem_name);
      break;
    case RTSP_CODEC_H265:
      fbin->video_bin.video_codecparse =
          gst_element_factory_make ("h265parse", elem_name);
      g_object_set (G_OBJECT (fbin->video_bin.video_codecparse),
       "config-interval", -1, NULL);
      break;
    case RTSP_CODEC_MPEG4:
      fbin->video_bin.video_codecparse =
          gst_element_factory_make ("mpeg4videoparse", elem_name);
      break;
    default:
      break;
  }
  if (!fbin->video_bin.video_codecparse) {
    GST_ELEMENT_ERROR (fbin, RESOURCE, NOT_FOUND,
        ("Failed to create codecparser element"), (NULL));
    return FALSE;
  }
  gst_bin_add (GST_BIN (fbin), fbin->video_bin.video_codecparse);

  g_snprintf (elem_name, sizeof (elem_name), "%s-video_rtppay",
      GST_ELEMENT_NAME (fbin));
  switch (fbin->codec) {
    case RTSP_CODEC_H264:
      fbin->video_bin.video_rtppay =
          gst_element_factory_make ("rtph264pay", elem_name);
      break;
    case RTSP_CODEC_H265:
      fbin->video_bin.video_rtppay =
          gst_element_factory_make ("rtph265pay", elem_name);
      break;
    case RTSP_CODEC_MPEG4:
      fbin->video_bin.video_rtppay =
          gst_element_factory_make ("rtpmp4vpay", elem_name);
      break;
    default:
      break;
  }
  if (!fbin->video_bin.video_rtppay) {
    GST_ELEMENT_ERROR (fbin, RESOURCE, NOT_FOUND,
        ("Failed to create container element"), (NULL));
    return FALSE;
  }
  gst_bin_add (GST_BIN (fbin), fbin->video_bin.video_rtppay);

  g_snprintf (elem_name, sizeof (elem_name), "%s-video_sink",
      GST_ELEMENT_NAME (fbin));
  fbin->video_bin.video_sink = gst_element_factory_make ("udpsink", elem_name);
  if (!fbin->video_bin.video_sink) {
    GST_ELEMENT_ERROR (fbin, RESOURCE, NOT_FOUND,
        ("Failed to create 'udpsink' element"), (NULL));
    return FALSE;
  }
  gst_bin_add (GST_BIN (fbin), fbin->video_bin.video_sink);

  if (fbin->video_bin.video_udp_port == -1) {
    GST_ELEMENT_ERROR (fbin, RESOURCE, FAILED,
        ("Failed to find free video UDP port"), (NULL));
  }
  g_object_set (G_OBJECT (fbin->video_bin.video_sink), "host",
      "224.224.255.255", "port", fbin->video_bin.video_udp_port,
      "async", FALSE, "sync", fbin->sync, NULL);

#if 0
  GstElement *nvdslogger_video = gst_element_factory_make ("nvdslogger", "nvdslog_video");
  g_object_set(G_OBJECT(nvdslogger_video), "silent", 0, NULL);
  gst_bin_add (GST_BIN (fbin), nvdslogger_video);
#endif

  if (fbin->bypass == false) {
    NVGSTDS_LINK_ELEMENT (fbin->video_bin.video_queue,
            fbin->video_bin.video_transform, FALSE);
    NVGSTDS_LINK_ELEMENT (fbin->video_bin.video_transform,
            fbin->video_bin.video_cap_filter, FALSE);
    NVGSTDS_LINK_ELEMENT (fbin->video_bin.video_cap_filter,
            fbin->video_bin.video_encoder, FALSE);
    NVGSTDS_LINK_ELEMENT (fbin->video_bin.video_encoder,
            swenc_caps, FALSE);
    NVGSTDS_LINK_ELEMENT (swenc_caps,
            fbin->video_bin.video_codecparse, FALSE);

    NVGSTDS_LINK_ELEMENT (fbin->video_bin.video_codecparse,
#if 0
            nvdslogger_video, FALSE);
    NVGSTDS_LINK_ELEMENT (nvdslogger_video,
#endif
            fbin->video_bin.video_rtppay, FALSE);
  } else {
      NVGSTDS_LINK_ELEMENT (fbin->video_bin.video_queue,
              fbin->video_bin.video_rtppay, FALSE);
  }
  NVGSTDS_LINK_ELEMENT (fbin->video_bin.video_rtppay,
          fbin->video_bin.video_sink, FALSE);

#if 0
  if (fbin->enc_type == RTSP_ENCODER_SW) {
    GstPad *encoder_src_pad = NULL;
    encoder_src_pad = gst_element_get_static_pad (fbin->video_bin.video_encoder, "src");
    if (!encoder_src_pad)
      g_print ("Unable to get encoder src pad\n");
    else
      gst_pad_add_probe (encoder_src_pad, GST_PAD_PROBE_TYPE_BUFFER,
          encoder_src_pad_buffer_probe, NULL, NULL);
    gst_object_unref (encoder_src_pad);

    GstPad *encoder_sink_pad = NULL;
    encoder_sink_pad = gst_element_get_static_pad (fbin->video_bin.video_encoder, "sink");
    if (!encoder_sink_pad)
      g_print ("Unable to get encoder sink pad\n");
    else
      gst_pad_add_probe (encoder_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
          encoder_sink_pad_buffer_probe, NULL, NULL);
    gst_object_unref (encoder_sink_pad);


  }
#endif

  GstPad *video_queue_sink_pad =
      gst_element_get_static_pad (fbin->video_bin.video_queue, "sink");
  if (!gst_ghost_pad_set_target
      (GST_GHOST_PAD_CAST (fbin->bin_vsink_pad), video_queue_sink_pad)) {
    gst_object_unref (video_queue_sink_pad);
    GST_ELEMENT_ERROR (fbin, RESOURCE, NOT_FOUND,
        ("Failed to set '%s' as target of '%s' pad",
            GST_ELEMENT_NAME (fbin->video_bin.video_queue),
            GST_PAD_NAME (fbin->bin_vsink_pad)), (NULL));
    return FALSE;
  }
  gst_object_unref (video_queue_sink_pad);

  return TRUE;
}

static gboolean
create_audio_pipeline (GstDsNvRtspOutBin * fbin)
{
  gchar elem_name[128];
  fbin->audio_bin.audio_udp_port = get_free_port ();

  g_snprintf (elem_name, sizeof (elem_name), "%s-audio_queue",
      GST_ELEMENT_NAME (fbin));
  fbin->audio_bin.audio_queue = gst_element_factory_make ("queue", elem_name);
  if (!fbin->audio_bin.audio_queue) {
    GST_ELEMENT_ERROR (fbin, RESOURCE, NOT_FOUND,
        ("Failed to create 'queue' element"), (NULL));
    return FALSE;
  }
  gst_bin_add (GST_BIN (fbin), fbin->audio_bin.audio_queue);

  g_snprintf (elem_name, sizeof (elem_name), "%s-audio_transform",
      GST_ELEMENT_NAME (fbin));
  fbin->audio_bin.audio_transform =
      gst_element_factory_make ("audioconvert", elem_name);
  if (!fbin->audio_bin.audio_transform) {
    GST_ELEMENT_ERROR (fbin, RESOURCE, NOT_FOUND,
        ("Failed to create 'audioconvert' element"), (NULL));
    return FALSE;
  }
  gst_bin_add (GST_BIN (fbin), fbin->audio_bin.audio_transform);

  g_snprintf (elem_name, sizeof (elem_name), "%s-audio_encoder",
      GST_ELEMENT_NAME (fbin));

  if (fbin->audio_codec == AUDIO_CODEC_AC3) {
    fbin->audio_bin.audio_encoder = gst_element_factory_make ("avenc_ac3", NULL);
    if (!fbin->audio_bin.audio_encoder) {
      g_print ("Failed to create avenc_ac3 element in audio udp sink bin");
      return FALSE;
    }
  } else if (fbin->audio_codec == AUDIO_CODEC_OPUS) {
    fbin->audio_bin.audio_encoder = gst_element_factory_make ("opusenc", NULL);
    if (!fbin->audio_bin.audio_encoder) {
      g_print ("Failed to create opusenc element in audio udp sink bin");
      return FALSE;
    }
  } 
  gst_bin_add (GST_BIN (fbin), fbin->audio_bin.audio_encoder);

  g_snprintf (elem_name, sizeof (elem_name), "%s-audio_parser",
      GST_ELEMENT_NAME (fbin));

  if (fbin->audio_codec == AUDIO_CODEC_AC3) {
    fbin->audio_bin.audio_parser = gst_element_factory_make ("ac3parse", NULL);
    if (!fbin->audio_bin.audio_parser) {
      g_print ("Failed to create ac3parse element in audio udp sink bin");
      return FALSE;
    }
  } else if (fbin->audio_codec == AUDIO_CODEC_OPUS) {
    fbin->audio_bin.audio_parser = gst_element_factory_make ("opusparse", NULL);
    if (!fbin->audio_bin.audio_parser) {
      g_print ("Failed to create opusparse element in audio udp sink bin");
      return FALSE;
    }
  }

  gst_bin_add (GST_BIN (fbin), fbin->audio_bin.audio_parser);

  g_snprintf (elem_name, sizeof (elem_name), "%s-audio_rtppay",
      GST_ELEMENT_NAME (fbin));

  if (fbin->audio_codec == AUDIO_CODEC_AC3) {
    fbin->audio_bin.audio_rtppay = gst_element_factory_make ("rtpac3pay", NULL);
    if (!fbin->audio_bin.audio_rtppay) {
      g_print ("Failed to create rtpac3pay element in audio udp sink bin");
      return FALSE;
    }
  } else if (fbin->audio_codec == AUDIO_CODEC_OPUS) {
    fbin->audio_bin.audio_rtppay = gst_element_factory_make ("rtpopuspay", NULL);
    if (!fbin->audio_bin.audio_rtppay) {
      g_print ("Failed to create rtpopuspay element in audio udp sink bin");
      return FALSE;
    }
  }
  gst_bin_add (GST_BIN (fbin), fbin->audio_bin.audio_rtppay);

  NVGSTDS_ELEM_ADD_PROBE (fbin, fbin->audio_bin.audio_rtppay, "src",
      audio_caps_negotiate_probe, GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM, fbin);

  g_snprintf (elem_name, sizeof (elem_name), "%s-audio_sink",
      GST_ELEMENT_NAME (fbin));

  fbin->audio_bin.audio_sink = gst_element_factory_make ("udpsink", NULL);
  if (!fbin->audio_bin.audio_sink) {
    g_print ("Failed to create udpsink element in audio udp sink bin");
    return FALSE;
  }
  gst_bin_add (GST_BIN (fbin), fbin->audio_bin.audio_sink);

  g_object_set (G_OBJECT (fbin->audio_bin.audio_sink), "async", FALSE, NULL);

  if (fbin->audio_bin.audio_udp_port == -1) {
    GST_ELEMENT_ERROR (fbin, RESOURCE, FAILED,
        ("Failed to find free audio UDP port"), (NULL));
  }

  g_object_set (G_OBJECT (fbin->audio_bin.audio_sink), "host",
      "224.224.255.255", "port", (fbin->audio_bin.audio_udp_port),
      "async", FALSE, "sync", 1, NULL);

  g_object_set (G_OBJECT (fbin->audio_bin.audio_sink), "auto-multicast", TRUE,
      NULL);

#if 0
  GstElement *nvdslogger_audio = gst_element_factory_make ("nvdslogger", "nvdslog_audio");
  g_object_set(G_OBJECT(nvdslogger_audio), "silent", 0, NULL);
  gst_bin_add (GST_BIN (fbin), nvdslogger_audio);
#endif
  NVGSTDS_LINK_ELEMENT (fbin->audio_bin.audio_queue,
      fbin->audio_bin.audio_transform, FALSE);
  NVGSTDS_LINK_ELEMENT (fbin->audio_bin.audio_transform,
      fbin->audio_bin.audio_encoder, FALSE);
  NVGSTDS_LINK_ELEMENT (fbin->audio_bin.audio_encoder,
      fbin->audio_bin.audio_parser, FALSE);
  NVGSTDS_LINK_ELEMENT (fbin->audio_bin.audio_parser,
#if 0
      nvdslogger_audio, FALSE);
  NVGSTDS_LINK_ELEMENT (nvdslogger_audio,
#endif
      fbin->audio_bin.audio_rtppay, FALSE);
  NVGSTDS_LINK_ELEMENT (fbin->audio_bin.audio_rtppay,
      fbin->audio_bin.audio_sink, FALSE);

  GstPad *audio_queue_sink_pad =
      gst_element_get_static_pad (fbin->audio_bin.audio_queue, "sink");
  if (!gst_ghost_pad_set_target
      (GST_GHOST_PAD_CAST (fbin->bin_asink_pad), audio_queue_sink_pad)) {
    gst_object_unref (audio_queue_sink_pad);
    GST_ELEMENT_ERROR (fbin, RESOURCE, NOT_FOUND,
        ("Failed to set '%s' as target of '%s' pad",
            GST_ELEMENT_NAME (fbin->audio_bin.audio_queue),
            GST_PAD_NAME (fbin->bin_asink_pad)), (NULL));
    return FALSE;
  }
  gst_object_unref (audio_queue_sink_pad);

  return TRUE;
}

static void
stop_rtsp_server (GstDsNvRtspOutBin * nvrtspoutbin)
{
  GstRTSPServer *server = GST_RTSP_SERVER (nvrtspoutbin->server);
  GstRTSPMountPoints *mounts = gst_rtsp_server_get_mount_points (server);
  gst_rtsp_mount_points_remove_factory (mounts, nvrtspoutbin->rtsp_mount_point);
  g_object_unref (mounts);
  gst_rtsp_server_client_filter (server,[](GstRTSPServer *, GstRTSPClient *,
          gpointer) {
        return GST_RTSP_FILTER_REMOVE;
      }
      , NULL);
  GstRTSPSessionPool *pool = gst_rtsp_server_get_session_pool (server);
  gst_rtsp_session_pool_cleanup (pool);
  g_object_unref (pool);
  nvrtspoutbin->server = NULL;
}

static GstPad *
gst_ds_nvrtspout_request_new_pad (GstElement * element,
    GstPadTemplate * templ, const gchar * name, const GstCaps * caps)
{

  GstDsNvRtspOutBin *nvrtspoutbin = GST_DS_NVRTSPOUT_BIN (element);
  GstElementClass *klass = GST_ELEMENT_GET_CLASS (element);

  GST_DEBUG_OBJECT (element, "Requesting new sink pad");

  if (templ == gst_element_class_get_pad_template (klass, "asink")) {

    /** Check whether audio pad already exists */
    if (nvrtspoutbin->bin_asink_pad) {

      GST_WARNING_OBJECT (nvrtspoutbin, "Already has an audio pad");
      nvrtspoutbin->has_audio = TRUE;
      return NULL;

    } else {                    /*otherwise create audio pad */


      nvrtspoutbin->bin_asink_pad =
          gst_ghost_pad_new_no_target_from_template ("asink", templ);
      gst_pad_set_active (nvrtspoutbin->bin_asink_pad, TRUE);
      gst_element_add_pad (GST_ELEMENT (nvrtspoutbin),
          nvrtspoutbin->bin_asink_pad);
      nvrtspoutbin->has_audio = TRUE;
      if (!create_audio_pipeline (nvrtspoutbin)) {
        g_printerr ("Unable to create audio pipeline");
        return NULL;
      }

      if (!gst_bin_sync_children_states (GST_BIN (nvrtspoutbin))) {
        GST_ELEMENT_ERROR (nvrtspoutbin, RESOURCE, FAILED,
            ("Failed to sync children failed."), (NULL));
        return NULL;
      }

      if (!start_rtsp_server (nvrtspoutbin)) {
        GST_ELEMENT_ERROR (nvrtspoutbin, RESOURCE, FAILED,
            ("Failed to start RTSP SERVER"), (NULL));
        return NULL;
      }
      return nvrtspoutbin->bin_asink_pad;
    }
  } else if (templ == gst_element_class_get_pad_template (klass, "vsink")) {

    /** Check whether video pad already exists */
    if (nvrtspoutbin->bin_vsink_pad) {

      GST_WARNING_OBJECT (nvrtspoutbin, "Already has a video pad");
      nvrtspoutbin->has_video = TRUE;
      return NULL;

    } else {                    /*otherwise create video pad */


      nvrtspoutbin->bin_vsink_pad =
          gst_ghost_pad_new_no_target_from_template ("vsink", templ);
      gst_pad_set_active (nvrtspoutbin->bin_vsink_pad, TRUE);
      gst_element_add_pad (GST_ELEMENT (nvrtspoutbin),
          nvrtspoutbin->bin_vsink_pad);
      nvrtspoutbin->has_video = TRUE;
      if (!create_video_pipeline (nvrtspoutbin)) {
        g_printerr ("Unable to create video pipeline");
        return NULL;
      }

      if (!gst_bin_sync_children_states (GST_BIN (nvrtspoutbin))) {
        GST_ELEMENT_ERROR (nvrtspoutbin, RESOURCE, FAILED,
            ("Failed to sync children failed."), (NULL));
        return NULL;
      }

      if (!start_rtsp_server (nvrtspoutbin)) {
        GST_ELEMENT_ERROR (nvrtspoutbin, RESOURCE, FAILED,
            ("Failed to start RTSP SERVER"), (NULL));
        return NULL;
      }
      return nvrtspoutbin->bin_vsink_pad;
    }

  } else {
    GST_WARNING_OBJECT (nvrtspoutbin, "Invalid template");
    return NULL;
  }

  return NULL;
}

static void
gst_ds_nvrtspout_release_pad (GstElement * element, GstPad * pad)
{
  GstDsNvRtspOutBin *nvrtspoutbin = GST_DS_NVRTSPOUT_BIN (element);
  gst_pad_set_active (pad, FALSE);

  if (pad == nvrtspoutbin->bin_vsink_pad) {

    if (GST_BIN_NUMCHILDREN (GST_BIN (nvrtspoutbin)) != 0) {
      gst_bin_remove_many (GST_BIN (nvrtspoutbin),
          nvrtspoutbin->video_bin.video_queue,
          nvrtspoutbin->video_bin.video_transform,
          nvrtspoutbin->video_bin.video_cap_filter,
          nvrtspoutbin->video_bin.video_encoder,
          nvrtspoutbin->video_bin.video_codecparse,
          nvrtspoutbin->video_bin.video_rtppay,
          nvrtspoutbin->video_bin.video_sink, NULL);
    }

    nvrtspoutbin->bin_vsink_pad = NULL;

  } else if (pad == nvrtspoutbin->bin_asink_pad) {

    if (GST_BIN_NUMCHILDREN (GST_BIN (nvrtspoutbin)) != 0) {
      gst_bin_remove_many (GST_BIN (nvrtspoutbin),
          nvrtspoutbin->audio_bin.audio_queue,
          nvrtspoutbin->audio_bin.audio_transform,
          nvrtspoutbin->audio_bin.audio_encoder,
          nvrtspoutbin->audio_bin.audio_parser,
          nvrtspoutbin->audio_bin.audio_rtppay,
          nvrtspoutbin->audio_bin.audio_sink, NULL);
    }
    nvrtspoutbin->bin_asink_pad = NULL;

  } else {
    GST_WARNING_OBJECT (pad, "Pad is not known audio or video pad");
  }

  gst_element_remove_pad (element, pad);
}

static GstStateChangeReturn
gst_ds_nvrtspout_change_state (GstElement * element, GstStateChange transition)
{
  GstDsNvRtspOutBin *nvrtspoutbin = GST_DS_NVRTSPOUT_BIN (element);
  GstStateChangeReturn ret;

  if (transition == GST_STATE_CHANGE_NULL_TO_READY) {
    if (nvrtspoutbin->has_audio || nvrtspoutbin->has_video)
      if (!start_rtsp_server (nvrtspoutbin)) {
        GST_ELEMENT_ERROR (nvrtspoutbin, RESOURCE, FAILED,
            ("Failed to start RTSP SERVER"), (NULL));
        return GST_STATE_CHANGE_FAILURE;
      }
  }

  ret = GST_ELEMENT_CLASS (parent_class)->change_state (element, transition);

  if (transition == GST_STATE_CHANGE_READY_TO_NULL) {
    stop_rtsp_server (nvrtspoutbin);
  }

  return ret;
}

static void
gst_ds_nvrtspout_bin_init (GstDsNvRtspOutBin * nvrtspoutbin)
{
  nvrtspoutbin->codec = DEFAULT_CODEC;
  nvrtspoutbin->audio_codec = DEFAULT_AUDIO_CODEC;
  nvrtspoutbin->enc_type = DEFAULT_ENC_TYPE;
  nvrtspoutbin->profile = DEFAULT_PROFILE;
  nvrtspoutbin->iframeinterval = DEFAULT_IFRAMEINTERVAL;
  nvrtspoutbin->idrframeinterval = DEFAULT_IDRFRAMEINTERVAL;
  nvrtspoutbin->rtsp_port = DEFAULT_RTSP_PORT;
  nvrtspoutbin->bitrate = DEFAULT_BITRATE;
  nvrtspoutbin->sync = DEFAULT_SYNC;
  nvrtspoutbin->qos = DEFAULT_QOS;
  nvrtspoutbin->gpu_id = DEFAULT_GPU_ID;
  nvrtspoutbin->mem_type = DEFAULT_NVBUF_MEMORY_TYPE;
  nvrtspoutbin->create_from_config = g_strdup (DEFAULT_CREATE_FROM_CONFIG);
  nvrtspoutbin->file_location = g_strdup (DEFAULT_OUTPUT_FILE);
  nvrtspoutbin->rtsp_mount_point = g_strdup (DEFAULT_RTSP_MOUNT_POINT);
  nvrtspoutbin->audio_bin.rate = DEFAULT_AUDIO_CLOCK_RATE;
  nvrtspoutbin->compute_hw = DEFAULT_COMPUTE_HW;

  GST_OBJECT_FLAG_SET (nvrtspoutbin, GST_ELEMENT_FLAG_SINK);
}
