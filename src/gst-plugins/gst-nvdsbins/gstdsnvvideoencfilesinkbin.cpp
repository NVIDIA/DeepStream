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
#include "gstdsnvvideoencfilesinkbin.h"
#include "gst-nvcommon.h"
#include "nvbufsurface.h"
#include "gstnvdsbinutils.h"

extern "C" GType gst_nvvideoconvert_get_type ();

GST_DEBUG_CATEGORY (gst_ds_nvvideoencfilesink_bin_debug);
#define GST_CAT_DEFAULT gst_ds_nvvideoencfilesink_bin_debug

/* Define our element type. Standard GObject/GStreamer boilerplate stuff */
#define gst_ds_nvvideoencfilesink_bin_parent_class parent_class
#define _do_init \
    GST_DEBUG_CATEGORY_INIT (gst_ds_nvvideoencfilesink_bin_debug, "nvvideoencfilesinkbin", 0, "nvvideoencfilesink element");
G_DEFINE_TYPE_WITH_CODE (GstDsNvVideoEncFilesinkBin,
    gst_ds_nvvideoencfilesink_bin, GST_TYPE_BIN, _do_init);

static void gst_ds_nvvideoencfilesink_bin_set_property (GObject * object,
    guint prop_id, const GValue * value, GParamSpec * spec);
static void gst_ds_nvvideoencfilesink_bin_get_property (GObject * object,
    guint prop_id, GValue * value, GParamSpec * spec);

static GstStateChangeReturn
gst_ds_nvvideoencfilesink_change_state (GstElement * element,
    GstStateChange transition);

#define GST_TYPE_NVDS_VIDEO_FILE_CONTAINER_TYPE (gst_nvds_video_enc_container_get_type())
static GType
gst_nvds_video_enc_container_get_type (void)
{
  static GType video_enc_container_type = 0;
  static const GEnumValue video_enc_containers[] = {
    {FILE_CONTAINER_MP4, "MP4 container", "mp4"},
    {FILE_CONTAINER_MKV, "Matroska container", "mkv"},
    {0, NULL, NULL},
  };

  if (!video_enc_container_type) {
    video_enc_container_type =
        g_enum_register_static ("GstNvDsVideoEncContainerType",
        video_enc_containers);
  }
  return video_enc_container_type;
}

#define GST_TYPE_NVDS_VIDEO_FILE_CODEC_TYPE (gst_nvds_video_enc_codec_get_type())
static GType
gst_nvds_video_enc_codec_get_type (void)
{
  static GType video_enc_codec_type = 0;
  static const GEnumValue video_enc_codecs[] = {
    {FILE_CODEC_H264, "H.264 encoder", "h264"},
    {FILE_CODEC_H265, "H.265 encoder", "h265"},
    {FILE_CODEC_MPEG4, "MPEG4 encoder", "mpeg4"},
    {0, NULL, NULL},
  };

  if (!video_enc_codec_type) {
    video_enc_codec_type = g_enum_register_static ("GstNvDsVideoEncCodecType",
        video_enc_codecs);
  }
  return video_enc_codec_type;
}

#define GST_TYPE_NVDS_VIDEO_FILE_ENC_TYPE (gst_nvds_video_file_enc_type_get_type())
static GType
gst_nvds_video_file_enc_type_get_type (void)
{
  static GType video_enc_type_type = 0;
  static const GEnumValue video_enc_types[] = {
    {FILE_ENCODER_HW, "Hardware accelerated encoder", "hw"},
    {FILE_ENCODER_SW, "Software encoder", "sw"},
    {0, NULL, NULL},
  };

  if (!video_enc_type_type) {
    video_enc_type_type = g_enum_register_static ("GstNvDsVideoEncTypeType",
        video_enc_types);
  }
  return video_enc_type_type;
}

#define GST_TYPE_NVDS_VIDEO_FILE_ENC_PROFILE (gst_nvds_video_file_enc_profile_get_type())
static GType
gst_nvds_video_file_enc_profile_get_type (void)
{
  static GType video_enc_profile_type = 0;
  static const GEnumValue video_enc_profiles[] = {
    {FILE_PROFILE_BASELINE, "Baseline Profile (H.264 only)", "baseline"},
    {FILE_PROFILE_MAIN, "Main Profile (H.264 & H.265)", "main"},
    {FILE_PROFILE_HIGH, "High Profile (H.264 only)", "high"},
    {FILE_PROFILE_MAIN10, "Main10 Profile (H.265 only))",
        "main10"},
    {0, NULL, NULL},
  };

  if (!video_enc_profile_type) {
    video_enc_profile_type =
        g_enum_register_static ("GstNvDsVideoEncProfileType",
        video_enc_profiles);
  }
  return video_enc_profile_type;
}

enum
{
  PROP_FIRST,
  PROP_CONTAINER,
  PROP_CODEC,
  PROP_ENC_TYPE,
  PROP_BITRATE,
  PROP_PROFILE,
  PROP_IFRAMEINTERVAL,
  PROP_OUTPUT_FILE,
  PROP_SYNC,
  PROP_QOS,
  PROP_GPU_DEVICE_ID,
  PROP_NVBUF_MEMORY_TYPE,
  PROP_CREATE_FROM_CONFIG,
  PROP_LAST
};

#define DEFAULT_CONTAINER FILE_CONTAINER_MP4
#define DEFAULT_CODEC FILE_CODEC_H264
#define DEFAULT_ENC_TYPE FILE_ENCODER_HW
#define DEFAULT_PROFILE FILE_PROFILE_BASELINE
#define DEFAULT_BITRATE 0
#define DEFAULT_IFRAMEINTERVAL 0
#define DEFAULT_SYNC FALSE
#define DEFAULT_QOS FALSE
#define DEFAULT_GPU_ID 0
#define DEFAULT_NVBUF_MEMORY_TYPE NVBUF_MEM_DEFAULT
#define DEFAULT_OUTPUT_FILE ""
#define DEFAULT_CREATE_FROM_CONFIG ""

static void
gst_ds_nvvideoencfilesink_bin_class_init (GstDsNvVideoEncFilesinkBinClass *
    klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;

  gobject_class = G_OBJECT_CLASS (klass);
  gstelement_class = GST_ELEMENT_CLASS (klass);

  gobject_class->set_property =
      GST_DEBUG_FUNCPTR (gst_ds_nvvideoencfilesink_bin_set_property);
  gobject_class->get_property =
      GST_DEBUG_FUNCPTR (gst_ds_nvvideoencfilesink_bin_get_property);

  g_object_class_install_property (gobject_class, PROP_CONTAINER,
      g_param_spec_enum ("container", "Container",
          "Type of container to use",
          GST_TYPE_NVDS_VIDEO_FILE_CONTAINER_TYPE,
          DEFAULT_CONTAINER,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_CODEC,
      g_param_spec_enum ("codec", "Codec",
          "Type of codec to use",
          GST_TYPE_NVDS_VIDEO_FILE_CODEC_TYPE,
          DEFAULT_CODEC,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_ENC_TYPE,
      g_param_spec_enum ("enc-type", "Encoder Type",
          "Type of encoder to use",
          GST_TYPE_NVDS_VIDEO_FILE_ENC_TYPE,
          DEFAULT_ENC_TYPE,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_PROFILE,
      g_param_spec_enum ("profile", "Profile",
          "Encoder profile to use",
          GST_TYPE_NVDS_VIDEO_FILE_ENC_PROFILE,
          DEFAULT_PROFILE,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_BITRATE,
      g_param_spec_uint ("bitrate", "Bitrate",
          "Encoding bitrate in bits/sec", 0, G_MAXUINT, DEFAULT_BITRATE,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_IFRAMEINTERVAL,
      g_param_spec_uint ("iframeinterval", "I-Frame Interval",
          "Encoding Intra Frame occurance frequency (H/W encoder only)",
          0, G_MAXUINT, DEFAULT_IFRAMEINTERVAL,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));


  g_object_class_install_property (gobject_class, PROP_SYNC,
      g_param_spec_boolean ("sync", "Sync",
          "Sync on the clock", DEFAULT_SYNC,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_QOS,
      g_param_spec_boolean ("qos", "QoS",
          "Generate Quality - of - Service events upstream", DEFAULT_QOS,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  PROP_NVDS_GPU_ID_INSTALL (gobject_class);

  PROP_NVBUF_MEMORY_TYPE_INSTALL (gobject_class);

  g_object_class_install_property (gobject_class, PROP_OUTPUT_FILE,
      g_param_spec_string ("output-file", "Output File",
          "Type of Video sink to use",
          DEFAULT_OUTPUT_FILE,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  forward_pad_template (gstelement_class, gst_nvvideoconvert_get_type (),
      "sink", "sink");

  gstelement_class->change_state =
      GST_DEBUG_FUNCPTR (gst_ds_nvvideoencfilesink_change_state);

  /* Set metadata describing the element */
  gst_element_class_set_details_simple (gstelement_class,
      "NvVideoEncFilesink Bin", "NvVideoEncFilesink Bin",
      "Nvidia DeepStreamSDK Video Sinks Bin. Internal Pipeline: queue->nvvideoconvert->encoder->codecparse->mux->filesink",
      "NVIDIA Corporation. Deepstream for Tesla forum: "
      "https://devtalk.nvidia.com/default/board/209");
}

static void
gst_ds_nvvideoencfilesink_bin_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstDsNvVideoEncFilesinkBin *nvvideoencfilesinkbin =
      GST_DS_NVVIDEOENCFILESINK_BIN (object);

  switch (prop_id) {
    case PROP_CONTAINER:
      nvvideoencfilesinkbin->container =
          (FileContainerType) g_value_get_enum (value);
      break;
    case PROP_CODEC:
      nvvideoencfilesinkbin->codec = (FileCodecType) g_value_get_enum (value);
      break;
    case PROP_PROFILE:
      nvvideoencfilesinkbin->profile =
          (FileProfileType) g_value_get_enum (value);
      break;
    case PROP_BITRATE:
      nvvideoencfilesinkbin->bitrate = g_value_get_uint (value);
      break;
    case PROP_IFRAMEINTERVAL:
      nvvideoencfilesinkbin->iframeinterval = g_value_get_uint (value);
      break;
    case PROP_ENC_TYPE:
      nvvideoencfilesinkbin->enc_type =
          (FileEncodeType) g_value_get_enum (value);
      break;
    case PROP_SYNC:
      nvvideoencfilesinkbin->sync = g_value_get_boolean (value);
      break;
    case PROP_QOS:
      nvvideoencfilesinkbin->qos = g_value_get_boolean (value);
      break;
    case PROP_GPU_DEVICE_ID:
      nvvideoencfilesinkbin->gpu_id = g_value_get_uint (value);
      break;
    case PROP_NVBUF_MEMORY_TYPE:
      nvvideoencfilesinkbin->mem_type =
          (NvBufSurfaceMemType) g_value_get_enum (value);
      break;
    case PROP_CREATE_FROM_CONFIG:
      g_free (nvvideoencfilesinkbin->create_from_config);
      nvvideoencfilesinkbin->create_from_config = g_value_dup_string (value);
      break;
    case PROP_OUTPUT_FILE:
      g_free (nvvideoencfilesinkbin->file_location);
      nvvideoencfilesinkbin->file_location = g_value_dup_string (value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}


static void
gst_ds_nvvideoencfilesink_bin_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstDsNvVideoEncFilesinkBin *nvvideoencfilesinkbin =
      GST_DS_NVVIDEOENCFILESINK_BIN (object);

  switch (prop_id) {
    case PROP_CONTAINER:
      g_value_set_enum (value, nvvideoencfilesinkbin->container);
      break;
    case PROP_CODEC:
      g_value_set_enum (value, nvvideoencfilesinkbin->codec);
      break;
    case PROP_ENC_TYPE:
      g_value_set_enum (value, nvvideoencfilesinkbin->enc_type);
      break;
    case PROP_PROFILE:
      g_value_set_enum (value, nvvideoencfilesinkbin->profile);
      break;
    case PROP_BITRATE:
      g_value_set_uint (value, nvvideoencfilesinkbin->bitrate);
      break;
    case PROP_IFRAMEINTERVAL:
      g_value_set_uint (value, nvvideoencfilesinkbin->iframeinterval);
      break;
    case PROP_SYNC:
      g_value_set_boolean (value, nvvideoencfilesinkbin->sync);
      break;
    case PROP_QOS:
      g_value_set_boolean (value, nvvideoencfilesinkbin->qos);
      break;
    case PROP_GPU_DEVICE_ID:
      g_value_set_uint (value, nvvideoencfilesinkbin->gpu_id);
      break;
    case PROP_NVBUF_MEMORY_TYPE:
      g_value_set_enum (value, nvvideoencfilesinkbin->mem_type);
      break;
    case PROP_OUTPUT_FILE:
      g_value_set_string (value, nvvideoencfilesinkbin->file_location);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
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

static gboolean
populate_bin (GstDsNvVideoEncFilesinkBin * fbin)
{
  gchar elem_name[128];

  g_snprintf (elem_name, sizeof (elem_name), "%s-queue",
      GST_ELEMENT_NAME (fbin));
  fbin->queue = gst_element_factory_make ("queue", elem_name);
  if (!fbin->queue) {
    GST_ELEMENT_ERROR (fbin, RESOURCE, NOT_FOUND,
        ("Failed to create 'queue' element"), (NULL));
    return FALSE;
  }
  gst_bin_add (GST_BIN (fbin), fbin->queue);

  g_snprintf (elem_name, sizeof (elem_name), "%s-transform",
      GST_ELEMENT_NAME (fbin));
  fbin->transform = gst_element_factory_make ("nvvideoconvert", elem_name);
  if (!fbin->transform) {
    GST_ELEMENT_ERROR (fbin, RESOURCE, NOT_FOUND,
        ("Failed to create 'nvvideoconvert' element"), (NULL));
    return FALSE;
  }
  gst_bin_add (GST_BIN (fbin), fbin->transform);
  if (g_object_class_find_property (G_OBJECT_GET_CLASS (fbin->transform),
          "gpu-id")) {
    g_object_set (G_OBJECT (fbin->transform), "gpu-id", fbin->gpu_id, NULL);
  }
  if (g_object_class_find_property (G_OBJECT_GET_CLASS (fbin->transform),
          "nvbuf-memory-type")) {
    g_object_set (G_OBJECT (fbin->transform), "nvbuf-memory-type",
        fbin->mem_type, NULL);
  }

  g_snprintf (elem_name, sizeof (elem_name), "%s-capsfilter",
      GST_ELEMENT_NAME (fbin));
  fbin->cap_filter = gst_element_factory_make ("capsfilter", elem_name);
  if (!fbin->cap_filter) {
    GST_ELEMENT_ERROR (fbin, RESOURCE, NOT_FOUND,
        ("Failed to create 'capsfilter' element"), (NULL));
    return FALSE;
  }
  gst_bin_add (GST_BIN (fbin), fbin->cap_filter);

  if (fbin->codec == FILE_CODEC_MPEG4 && fbin->enc_type == FILE_ENCODER_HW) {
    GST_ELEMENT_WARNING (fbin, LIBRARY, SETTINGS,
        ("Overriding to software encoder."
            " MPEG4 is supported by software encoder only"), (NULL));
    fbin->enc_type = FILE_ENCODER_SW;
  }

  g_snprintf (elem_name, sizeof (elem_name), "%s-encoder",
      GST_ELEMENT_NAME (fbin));
  FileEncodeType enc_type = fbin->enc_type;
  switch (fbin->codec) {
    case FILE_CODEC_H264:
      if (enc_type == FILE_ENCODER_SW) {
        fbin->encoder =
            gst_element_factory_make ("x264enc", elem_name);
      } else {
        fbin->encoder =
            gst_element_factory_make ("nvv4l2h264enc", elem_name);
        if (!fbin->encoder) {
          GST_ELEMENT_WARNING (fbin, LIBRARY, SETTINGS,
              ("Could not create HW encoder. Falling back to SW encoder"), (NULL));
          fbin->encoder =
            gst_element_factory_make ("x264enc", elem_name);
          fbin->enc_type = FILE_ENCODER_SW;
        }
      }
      break;
    case FILE_CODEC_H265:
      if (enc_type == FILE_ENCODER_SW) {
        fbin->encoder =
            gst_element_factory_make ("x265enc", elem_name);
      } else {
        fbin->encoder =
            gst_element_factory_make ("nvv4l2h265enc", elem_name);
        if (!fbin->encoder) {
          GST_ELEMENT_WARNING (fbin, LIBRARY, SETTINGS,
              ("Could not create HW encoder. Falling back to SW encoder"), (NULL));
          fbin->encoder =
            gst_element_factory_make ("x265enc", elem_name);
          fbin->enc_type = FILE_ENCODER_SW;
        }
      }
      break;
    case FILE_CODEC_MPEG4:
      fbin->encoder = gst_element_factory_make ("avenc_mpeg4", elem_name);
      break;
    default:
      break;
  }

  if (!fbin->encoder) {
    GST_ELEMENT_ERROR (fbin, RESOURCE, NOT_FOUND,
        ("Failed to create encoder element"), (NULL));
    return FALSE;
  }

  GstCaps *caps;
  if (fbin->codec == FILE_CODEC_MPEG4 || fbin->enc_type == FILE_ENCODER_SW)
    caps = gst_caps_from_string ("video/x-raw, format=I420");
  else
    caps = gst_caps_from_string ("video/x-raw(memory:NVMM), format=I420");
  g_object_set (G_OBJECT (fbin->cap_filter), "caps", caps, NULL);
  gst_caps_unref (caps);

  gst_bin_add (GST_BIN (fbin), fbin->encoder);
  if (g_object_class_find_property (G_OBJECT_GET_CLASS (fbin->encoder),
          "gpu-id")) {
    g_object_set (G_OBJECT (fbin->encoder), "gpu-id", fbin->gpu_id, NULL);
  }

  NVGSTDS_ELEM_ADD_PROBE (fbin, fbin->encoder, "sink", seek_query_drop_prob,
      GST_PAD_PROBE_TYPE_QUERY_UPSTREAM, fbin);

  if (fbin->enc_type == FILE_ENCODER_HW) {
    switch (fbin->profile) {

      case FILE_PROFILE_BASELINE:
        if (fbin->codec == FILE_CODEC_H265) {
          GST_ELEMENT_WARNING (fbin, LIBRARY, SETTINGS,
              ("H.265 does not support 'Baseline' profile. Overriding to 'Main' profile."),
              (NULL));
          g_object_set (G_OBJECT (fbin->encoder), "profile", 0, NULL);
        } else {
          g_object_set (G_OBJECT (fbin->encoder), "profile", 0, NULL);
        }
        break;
      case FILE_PROFILE_MAIN:
        g_object_set (G_OBJECT (fbin->encoder), "profile",
            fbin->codec == FILE_CODEC_H265 ? 0 : 2, NULL);
        break;
      case FILE_PROFILE_HIGH:
        if (fbin->codec == FILE_CODEC_H265) {
          GST_ELEMENT_WARNING (fbin, LIBRARY, SETTINGS,
              ("H.265 does not support 'Baseline' profile. Overriding to 'Main' profile."),
              (NULL));
          g_object_set (G_OBJECT (fbin->encoder), "profile", 0, NULL);
        } else {
          g_object_set (G_OBJECT (fbin->encoder), "profile", 4, NULL);
        }
        break;
      case FILE_PROFILE_MAIN10:
        if (fbin->codec == FILE_CODEC_H265) {
          g_object_set (G_OBJECT (fbin->encoder), "profile", 1, NULL);
        } else {
          GST_ELEMENT_WARNING (fbin, LIBRARY, SETTINGS,
              ("H.264 does not support 'Main10' profile. Overriding to 'Baseline' profile."),
              (NULL));
          g_object_set (G_OBJECT (fbin->encoder), "profile", 0, NULL);
        }
        break;
      default:
        break;
    }

    g_object_set (G_OBJECT (fbin->encoder), "iframeinterval",
        fbin->iframeinterval, NULL);
    g_object_set (G_OBJECT (fbin->encoder), "bitrate", fbin->bitrate, NULL);
  } else {
    if (fbin->codec == FILE_CODEC_MPEG4)
      g_object_set (G_OBJECT (fbin->encoder), "bitrate", fbin->bitrate, NULL);
    else {
      //bitrate is in kbits/sec for software encoder x264enc and x265enc
      g_object_set (G_OBJECT (fbin->encoder), "bitrate",
          fbin->bitrate / 1000, "key-int-max", fbin->iframeinterval, NULL);
    }
  }

  g_snprintf (elem_name, sizeof (elem_name), "%s-codecparser",
      GST_ELEMENT_NAME (fbin));
  switch (fbin->codec) {
    case FILE_CODEC_H264:
      fbin->codecparse = gst_element_factory_make ("h264parse", elem_name);
      break;
    case FILE_CODEC_H265:
      fbin->codecparse = gst_element_factory_make ("h265parse", elem_name);
      break;
    case FILE_CODEC_MPEG4:
      fbin->codecparse =
          gst_element_factory_make ("mpeg4videoparse", elem_name);
      break;
    default:
      break;
  }
  if (!fbin->codecparse) {
    GST_ELEMENT_ERROR (fbin, RESOURCE, NOT_FOUND,
        ("Failed to create codecparser element"), (NULL));
    return FALSE;
  }
  gst_bin_add (GST_BIN (fbin), fbin->codecparse);

  g_snprintf (elem_name, sizeof (elem_name), "%s-muxer",
      GST_ELEMENT_NAME (fbin));
  switch (fbin->container) {
    case FILE_CONTAINER_MP4:
      fbin->mux = gst_element_factory_make ("qtmux", elem_name);
      break;
    case FILE_CONTAINER_MKV:
      fbin->mux = gst_element_factory_make ("matroskamux", elem_name);
      break;
    default:
      break;
  }
  if (!fbin->mux) {
    GST_ELEMENT_ERROR (fbin, RESOURCE, NOT_FOUND,
        ("Failed to create container element"), (NULL));
    return FALSE;
  }
  gst_bin_add (GST_BIN (fbin), fbin->mux);


  g_snprintf (elem_name, sizeof (elem_name), "%s-sink",
      GST_ELEMENT_NAME (fbin));
  fbin->sink = gst_element_factory_make ("filesink", elem_name);
  if (!fbin->sink) {
    GST_ELEMENT_ERROR (fbin, RESOURCE, NOT_FOUND,
        ("Failed to create 'filesink' element"), (NULL));
    return FALSE;
  }
  gst_bin_add (GST_BIN (fbin), fbin->sink);
  g_object_set (G_OBJECT (fbin->sink),
      "location", fbin->file_location,
      "sync", fbin->sync, "async", FALSE, NULL);

  NVGSTDS_LINK_ELEMENT (fbin->queue, fbin->transform, FALSE);
  NVGSTDS_LINK_ELEMENT (fbin->transform, fbin->cap_filter, FALSE);
  NVGSTDS_LINK_ELEMENT (fbin->cap_filter, fbin->encoder, FALSE);
  NVGSTDS_LINK_ELEMENT (fbin->encoder, fbin->codecparse, FALSE);
  NVGSTDS_LINK_ELEMENT (fbin->codecparse, fbin->mux, FALSE);
  NVGSTDS_LINK_ELEMENT (fbin->mux, fbin->sink, FALSE);

  NVGSTDS_BIN_SET_GHOST_PAD_TARGET (fbin, fbin->bin_sink_pad, fbin->queue,
      "sink", FALSE);

  return TRUE;
}

static GstStateChangeReturn
gst_ds_nvvideoencfilesink_change_state (GstElement * element,
    GstStateChange transition)
{
  GstDsNvVideoEncFilesinkBin *nvvideoencfilesinkbin =
      GST_DS_NVVIDEOENCFILESINK_BIN (element);
  GstStateChangeReturn ret;

  if (transition == GST_STATE_CHANGE_NULL_TO_READY) {
    if (!populate_bin (nvvideoencfilesinkbin)) {
      return GST_STATE_CHANGE_FAILURE;
    }
  }

  if (transition == GST_STATE_CHANGE_PAUSED_TO_READY) {
    gst_element_set_state (nvvideoencfilesinkbin->sink, GST_STATE_PLAYING);
    gst_element_send_event (nvvideoencfilesinkbin->codecparse,
        gst_event_new_eos ());
  }

  ret = GST_ELEMENT_CLASS (parent_class)->change_state (element, transition);

  if (transition == GST_STATE_CHANGE_READY_TO_NULL) {
    remove_all_children (GST_BIN (nvvideoencfilesinkbin));
  }

  return ret;
}

static void
gst_ds_nvvideoencfilesink_bin_init (GstDsNvVideoEncFilesinkBin *
    nvvideoencfilesinkbin)
{
  nvvideoencfilesinkbin->container = DEFAULT_CONTAINER;
  nvvideoencfilesinkbin->codec = DEFAULT_CODEC;
  nvvideoencfilesinkbin->enc_type = DEFAULT_ENC_TYPE;
  nvvideoencfilesinkbin->profile = DEFAULT_PROFILE;
  nvvideoencfilesinkbin->iframeinterval = DEFAULT_IFRAMEINTERVAL;
  nvvideoencfilesinkbin->bitrate = DEFAULT_BITRATE;
  nvvideoencfilesinkbin->sync = DEFAULT_SYNC;
  nvvideoencfilesinkbin->qos = DEFAULT_QOS;
  nvvideoencfilesinkbin->gpu_id = DEFAULT_GPU_ID;
  nvvideoencfilesinkbin->mem_type = DEFAULT_NVBUF_MEMORY_TYPE;
  nvvideoencfilesinkbin->create_from_config =
      g_strdup (DEFAULT_CREATE_FROM_CONFIG);
  nvvideoencfilesinkbin->file_location = g_strdup (DEFAULT_OUTPUT_FILE);

  nvvideoencfilesinkbin->bin_sink_pad =
      gst_ghost_pad_new_no_target_from_template ("sink",
      gst_element_class_get_pad_template (GST_ELEMENT_GET_CLASS
          (nvvideoencfilesinkbin), "sink"));
  gst_element_add_pad (GST_ELEMENT (nvvideoencfilesinkbin),
      nvvideoencfilesinkbin->bin_sink_pad);

  GST_OBJECT_FLAG_SET (nvvideoencfilesinkbin, GST_ELEMENT_FLAG_SINK);
}
