/*
 * Copyright (C) 2014-2023 Collabora Ltd.
 *     Author: Nicolas Dufresne <nicolas.dufresne@collabora.com>
 *
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
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>
#include <string.h>
#include "gstv4l2h26xparser.h"
#include "gstv4l2object.h"
#include "gstv4l2videodec.h"
#include "gstnvdsseimeta.h"
#include "gst-nvcustomevent.h"
#include "nvhelper_dyn.h"

#include "stdlib.h"

#include <string.h>
#include <stdint.h>
#include <gst/gst-i18n-plugin.h>

GST_DEBUG_CATEGORY_STATIC (gst_v4l2_video_dec_debug);
#define GST_CAT_DEFAULT gst_v4l2_video_dec_debug
#define ENABLE_DRAIN 1
#ifdef USE_V4L2_TARGET_NV


static gboolean
gst_v4l2_video_remove_padding (GstCapsFeatures * features,
    GstStructure * structure, gpointer user_data);

typedef enum {
  CAP_BUF_DYNAMIC_ALLOC_DISABLED,
  CAP_BUF_DYNAMIC_ALLOC_ENABLED_FOR_FW_PLAYBACK,
  CAP_BUF_DYNAMIC_ALLOC_ENABLED_FOR_RW_PLAYBACK,
  CAP_BUF_DYNAMIC_ALLOC_ENABLED_FOR_FW_RW_PLAYBACK
} CaptureBufferDynamicAllocationModes;

#define DEFAULT_SKIP_FRAME_TYPE V4L2_SKIP_FRAMES_TYPE_NONE
#define DEFAULT_DISABLE_DPB FALSE
#define DEFAULT_FULL_FRAME FALSE
#define DEFAULT_FRAME_TYPR_REPORTING FALSE
#define DEFAULT_ERROR_CHECK FALSE
#define DEFAULT_MAX_PERFORMANCE FALSE
#define DEFAULT_CAP_BUF_DYNAMIC_ALLOCATION CAP_BUF_DYNAMIC_ALLOC_DISABLED
#define GST_TYPE_V4L2_VID_DEC_SKIP_FRAMES (gst_video_dec_skip_frames ())
#define GST_TYPE_V4L2_DEC_CAP_BUF_DYNAMIC_ALLOC (gst_video_dec_capture_buffer_dynamic_allocation ())

gboolean default_cudadec_low_latency;
gboolean default_sei_extract_data;
gint default_num_extra_surfaces;

static gboolean enable_latency_measurement = FALSE;
static gboolean enable_latency_print = FALSE;
extern uint8_t *parse_sei_data (uint8_t *bs, guint size, uint32_t *payload_size,
        char *sei_uuid_string, guint32 pixelformat);
guint64 extract_sei_json_int (const char *input, const char *field_name);
guint64 convert_string_to_ts (GstV4l2VideoDec * self, const char *input);

#ifdef USE_V4L2_TARGET_NV
GstVideoCodecFrame *
gst_v4l2_video_dec_find_nearest_frame (GstBuffer * buf, GList * frames);

static void
gst_v4l2_video_dec_clean_older_frames (GstV4l2VideoDec * self,
    GstBuffer * buf, GList * frames);

#endif

static gdouble get_current_system_timestamp(void)
{
  struct timeval t1;
  double elapsedTime = 0;
  gettimeofday(&t1, NULL);
  elapsedTime = (t1.tv_sec) * 1000.0;
  elapsedTime += (t1.tv_usec) / 1000.0;
  return elapsedTime;
}

static GType
gst_video_dec_skip_frames (void)
{
  static GType qtype = 0;

  if (qtype == 0) {
    static const GEnumValue values[] = {
      {V4L2_SKIP_FRAMES_TYPE_NONE, "Decode all frames", "decode_all"},
      {V4L2_SKIP_FRAMES_TYPE_NONREF, "Decode non-ref frames",
          "decode_non_ref"},
      {V4L2_SKIP_FRAMES_TYPE_DECODE_IDR_ONLY, "decode key frames",
          "decode_key"},
      {0, NULL, NULL}
    };

    qtype = g_enum_register_static ("SkipFrame", values);
  }
  return qtype;
}

static GType
gst_video_dec_capture_buffer_dynamic_allocation (void)
{
  static GType qtype = 0;

  if (qtype == 0) {
    static const GEnumValue values[] = {
      {CAP_BUF_DYNAMIC_ALLOC_DISABLED,
       "Capture buffer dynamic allocation disabled", "cap_buf_dyn_alloc_disabled"},
      {CAP_BUF_DYNAMIC_ALLOC_ENABLED_FOR_FW_PLAYBACK,
       "Capture buffer dynamic allocation enabled for forward playback", "fw_cap_buf_dyn_alloc_enabled"},
      {CAP_BUF_DYNAMIC_ALLOC_ENABLED_FOR_RW_PLAYBACK,
       "Capture buffer dynamic allocation enabled for reverse playback", "rw_cap_buf_dyn_alloc_enabled"},
      {CAP_BUF_DYNAMIC_ALLOC_ENABLED_FOR_FW_RW_PLAYBACK,
       "Capture buffer dynamic allocation enabled for forward and reverse playback", "fw_rw_cap_buf_dyn_alloc_enabled"},
      {0, NULL, NULL}
    };

    qtype = g_enum_register_static ("CaptureBufferDynamicAllocationModes", values);
  }
  return qtype;
}

/* Properties specifically applicable on GPU*/
#define GST_TYPE_V4L2_VID_CUDADEC_MEM_TYPE (gst_video_cudadec_mem_type ())

#define DEFAULT_CUDADEC_MEM_TYPE V4L2_CUDA_MEM_TYPE_DEVICE

#define DEFAULT_CUDADEC_GPU_ID   0
#define MAX_CUDADEC_NUM_SURFACES 20
#define DEFAULT_CUDADEC_NUM_SURFACES 20

static GType
gst_video_cudadec_mem_type (void)
{
  static GType qtype = 0;

  if (qtype == 0) {
    NvBufSurfaceDeviceInfo deviceInfo;
    memset(&deviceInfo, 0, sizeof(deviceInfo));
    if (NvBufSurfaceGetDeviceInfo(&deviceInfo) != 0) {
      GST_WARNING("NvBufSurfaceGetDeviceInfo failed, using default mem types");
    }

    if (deviceInfo.platformType >= NVBUF_PLATFORM_TYPE_T264) {
    static const GEnumValue values[] = {
      {V4L2_CUDA_MEM_TYPE_DEVICE, "Memory type Device", "memtype_device"},
      {V4L2_CUDA_MEM_TYPE_PINNED, "Memory type Host Pinned",
          "memtype_pinned"},
      {V4L2_CUDA_MEM_TYPE_UNIFIED, "Memory type Unified",
          "memtype_unified"},
      {V4L2_CUDA_MEM_TYPE_SURFACE_ARRAY, "Memory type Surface Array",
          "memtype_surface_array"},
      {V4L2_CUDA_MEM_TYPE_ARRAY, "Memory type Array", "memtype_array"},
      {0, NULL, NULL}
    };

    qtype = g_enum_register_static ("CudaDecMemType", values);
  }
  else {
      static const GEnumValue values[] = {
        {V4L2_CUDA_MEM_TYPE_DEVICE, "Memory type Device", "memtype_device"},
        {V4L2_CUDA_MEM_TYPE_PINNED, "Memory type Host Pinned",
            "memtype_pinned"},
        {V4L2_CUDA_MEM_TYPE_UNIFIED, "Memory type Unified",
            "memtype_unified"},
        {0, NULL, NULL}
      };

      qtype = g_enum_register_static ("CudaDecMemType", values);
    }
  }

  return qtype;
}

#define CAPTURE_CAPS \
    "video/x-raw(memory:NVMM), " \
    "width = (gint) [ 1, MAX ], " \
    "height = (gint) [ 1, MAX ], " \
    "framerate = (fraction) [ 0, MAX ];"

//Caps on Tegra/dGPU
static GstStaticPadTemplate gst_v4l2dec_sink_template =
    GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS ("image/jpeg"
        ";"
        "video/x-h264,"
        "stream-format = (string) { byte-stream },"
        "alignment = (string) { au }"
        ";"
        "video/x-h265,"
        "stream-format = (string) { byte-stream },"
        "alignment = (string) { au }"
        ";"
        "video/mpeg, "
        "mpegversion= (int) 4, "
        "systemstream=(boolean) false, "
        "parsed=(boolean) true, "
        "width=(gint) [1,MAX],"
        "height=(gint) [1,MAX]"
        ";"
        "video/mpeg, "
        "mpegversion= (int) [1, 2], "
        "systemstream=(boolean) false, "
        "parsed=(boolean) true, "
        "width=(gint) [1,MAX],"
        "height=(gint) [1,MAX]"
        ";"
        "video/x-divx, "
        "divxversion=(int) [4, 5], "
        "width=(int) [1,MAX], " "height=(int) [1,MAX]"
        ";"
        "video/x-av1,"
        "stream-format = (string) { obu-stream },"
        "alignment = (string) { frame }"
        ";"
        "video/x-vp8"
        ";" "video/x-vp9," "width=(gint) [1,MAX]," "height=(gint) [1,MAX]" ";")
    );

static GstStaticPadTemplate gst_v4l2dec_src_template =
GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (CAPTURE_CAPS));
#endif

typedef struct
{
  gchar *device;
  GstCaps *sink_caps;
  GstCaps *src_caps;
  const gchar *longname;
  const gchar *description;
} GstV4l2VideoDecCData;

enum
{
  PROP_0,
  V4L2_STD_OBJECT_PROPS,
#ifdef USE_V4L2_TARGET_NV
/*Common properties*/
  PROP_SKIP_FRAME,
  PROP_DROP_FRAME_INTERVAL,
  PROP_NUM_EXTRA_SURFACES,
/*Properties exposed on Tegra only */
  PROP_DISABLE_DPB,
  PROP_USE_FULL_FRAME,
  PROP_ENABLE_FRAME_TYPE_REPORTING,
  PROP_ENABLE_ERROR_CHECK,
  PROP_ENABLE_MAX_PERFORMANCE,
  PROP_OPEN_MJPEG_BLOCK,
  PROP_IS_GDR_STREAM,
/*Properties exposed on dGPU only*/
  PROP_CUDADEC_MEM_TYPE,
  PROP_CUDADEC_GPU_ID,
  PROP_CUDADEC_LOW_LATENCY,
  PROP_EXTRACT_SEI_TYPE5_DATA,
  PROP_ATTACH_SEI_TIMESTAMPS,
  PROP_SEI_UUID_STRING,
  PROP_CAP_BUF_DYNAMIC_ALLOCATION,
#endif
};

#define gst_v4l2_video_dec_parent_class parent_class
G_DEFINE_ABSTRACT_TYPE (GstV4l2VideoDec, gst_v4l2_video_dec,
    GST_TYPE_VIDEO_DECODER);

#ifdef USE_V4L2_TARGET_NV
GType
gst_v4l2_dec_output_io_mode_get_type (void)
{
  static GType v4l2_dec_output_io_mode = 0;

  if (!v4l2_dec_output_io_mode) {
    static const GEnumValue dec_output_io_modes[] = {
      {GST_V4L2_IO_AUTO, "GST_V4L2_IO_AUTO", "auto"},
      {GST_V4L2_IO_MMAP, "GST_V4L2_IO_MMAP", "mmap"},
      {GST_V4L2_IO_USERPTR, "GST_V4L2_IO_USERPTR", "userptr"},
      {0, NULL, NULL}
    };

    v4l2_dec_output_io_mode = g_enum_register_static ("GstNvV4l2DecOutputIOMode",
        dec_output_io_modes);
  }
  return v4l2_dec_output_io_mode;
}

GType
gst_v4l2_dec_capture_io_mode_get_type (void)
{
  static GType v4l2_dec_capture_io_mode = 0;

  if (!v4l2_dec_capture_io_mode) {
    static const GEnumValue dec_capture_io_modes[] = {
      {GST_V4L2_IO_AUTO, "GST_V4L2_IO_AUTO", "auto"},
      {GST_V4L2_IO_MMAP, "GST_V4L2_IO_MMAP", "mmap"},
      {0, NULL, NULL}
    };

    v4l2_dec_capture_io_mode = g_enum_register_static ("GstNvV4l2DecCaptureIOMode",
        dec_capture_io_modes);
  }
  return v4l2_dec_capture_io_mode;
}
#endif

/* prototypes */
static GstFlowReturn gst_v4l2_video_dec_finish (GstVideoDecoder * decoder);

#ifdef USE_V4L2_GST_HEADER_VER_1_8
/**
 * TODO: This function gst_pad_get_task_state is introduced in newer gstreamer
 * version which we use on Tegra but on Desktop we are still at 1.8.3 (16.04)
 * At the moment I am copying the code here but eventually when we move to 1.14,
 * this shouldn't be needed.
 * gst_pad_get_task_state:
 * @pad: the #GstPad to get task state from
 *
 * Get @pad task state. If no task is currently
 * set, GST_TASK_STOPPED is returned.
 *
 * Returns: The current state of @pad's task.
 */
GstTaskState
gst_pad_get_task_state (GstPad * pad)
{
  GstTask *task;
  GstTaskState res;

  g_return_val_if_fail (GST_IS_PAD (pad), GST_TASK_STOPPED);

  GST_OBJECT_LOCK (pad);
  task = GST_PAD_TASK (pad);
  if (task == NULL)
    goto no_task;
  res = gst_task_get_state (task);
  GST_OBJECT_UNLOCK (pad);

  return res;

no_task:
  {
    GST_DEBUG_OBJECT (pad, "pad has no task");
    GST_OBJECT_UNLOCK (pad);
    return GST_TASK_STOPPED;
  }
}
#endif

static void
gst_v4l2_video_dec_set_property_tegra (GObject * object,
    guint prop_id, const GValue * value, GParamSpec * pspec)
{
  GstV4l2VideoDec *self = GST_V4L2_VIDEO_DEC (object);

  switch (prop_id) {
    case PROP_CAPTURE_IO_MODE:
      if (!gst_v4l2_object_set_property_helper (self->v4l2capture,
              prop_id, value, pspec)) {
        G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      }
      break;

#ifdef USE_V4L2_TARGET_NV
    case PROP_SKIP_FRAME:
      self->skip_frames = g_value_get_enum (value);
      break;

    case PROP_DROP_FRAME_INTERVAL:
      self->drop_frame_interval = g_value_get_uint (value);
      break;

    case PROP_NUM_EXTRA_SURFACES:
      self->num_extra_surfaces = g_value_get_uint (value);
      break;

    case PROP_DISABLE_DPB:
      self->disable_dpb = g_value_get_boolean (value);
      break;

    case PROP_USE_FULL_FRAME:
      self->enable_full_frame = g_value_get_boolean (value);
      break;

    case PROP_ENABLE_FRAME_TYPE_REPORTING:
      self->enable_frame_type_reporting = g_value_get_boolean (value);
      self->v4l2capture->Enable_frame_type_reporting =
          g_value_get_boolean (value);
      break;

    case PROP_ENABLE_ERROR_CHECK:
      self->enable_error_check = g_value_get_boolean (value);
      self->v4l2capture->Enable_error_check = g_value_get_boolean (value);
      break;

    case PROP_ENABLE_MAX_PERFORMANCE:
      self->enable_max_performance = g_value_get_boolean (value);
      break;

    case PROP_OPEN_MJPEG_BLOCK:
      self->v4l2output->open_mjpeg_block = g_value_get_boolean (value);
      break;

    case PROP_IS_GDR_STREAM:
      self->is_gdr_stream = g_value_get_boolean (value);
      self->enable_frame_type_reporting = g_value_get_boolean (value);
      break;

    case PROP_CAP_BUF_DYNAMIC_ALLOCATION:
      self->cap_buf_dynamic_allocation = g_value_get_enum (value);
      break;

    case PROP_EXTRACT_SEI_TYPE5_DATA:
      self->extract_sei_type5_data = g_value_get_boolean (value);
      break;

    case PROP_ATTACH_SEI_TIMESTAMPS:
      self->attach_sei_timestamps = g_value_get_boolean (value);
      break;

    case PROP_SEI_UUID_STRING:
      self->sei_uuid_string = (gchar *)g_value_dup_string (value);
      break;

#endif
      /* By default, only set on output */
    default:
      if (!gst_v4l2_object_set_property_helper (self->v4l2output,
              prop_id, value, pspec)) {
        G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      }
      break;
  }
}

static void
gst_v4l2_video_dec_set_property_cuvid (GObject * object,
    guint prop_id, const GValue * value, GParamSpec * pspec)
{
  GstV4l2VideoDec *self = GST_V4L2_VIDEO_DEC (object);

  switch (prop_id) {
    case PROP_CAPTURE_IO_MODE:
      if (!gst_v4l2_object_set_property_helper (self->v4l2capture,
              prop_id, value, pspec)) {
        G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      }
      break;

#ifdef USE_V4L2_TARGET_NV
    case PROP_SKIP_FRAME:
      self->skip_frames = g_value_get_enum (value);
      break;

    case PROP_DROP_FRAME_INTERVAL:
      self->drop_frame_interval = g_value_get_uint (value);
      break;

    case PROP_NUM_EXTRA_SURFACES:
      self->num_extra_surfaces = g_value_get_uint (value);
      break;

    case PROP_CUDADEC_MEM_TYPE:
      self->cudadec_mem_type = g_value_get_enum (value);
      break;

    case PROP_CUDADEC_GPU_ID:
      self->cudadec_gpu_id = g_value_get_uint (value);
      break;

    case PROP_CUDADEC_LOW_LATENCY:
    case PROP_DISABLE_DPB:
      self->cudadec_low_latency = g_value_get_boolean (value);
      break;

    case PROP_EXTRACT_SEI_TYPE5_DATA:
      self->extract_sei_type5_data = g_value_get_boolean (value);
      break;

    case PROP_ATTACH_SEI_TIMESTAMPS:
      self->attach_sei_timestamps = g_value_get_boolean (value);
      break;

    case PROP_SEI_UUID_STRING:
      self->sei_uuid_string = (gchar *)g_value_dup_string (value);
      break;

    case PROP_CAP_BUF_DYNAMIC_ALLOCATION:
      self->cap_buf_dynamic_allocation = g_value_get_enum (value);
      break;
#endif
      /* By default, only set on output */
    default:
      if (!gst_v4l2_object_set_property_helper (self->v4l2output,
              prop_id, value, pspec)) {
        G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      }
      break;
  }
}

static void
gst_v4l2_video_dec_get_property_tegra (GObject * object,
    guint prop_id, GValue * value, GParamSpec * pspec)
{
  GstV4l2VideoDec *self = GST_V4L2_VIDEO_DEC (object);

  switch (prop_id) {
    case PROP_CAPTURE_IO_MODE:
      if (!gst_v4l2_object_get_property_helper (self->v4l2capture,
              prop_id, value, pspec)) {
        G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      }
      break;

#ifdef USE_V4L2_TARGET_NV
    case PROP_SKIP_FRAME:
      g_value_set_enum (value, self->skip_frames);
      break;

    case PROP_DROP_FRAME_INTERVAL:
      g_value_set_uint (value, self->drop_frame_interval);
      break;

    case PROP_NUM_EXTRA_SURFACES:
      g_value_set_uint (value, self->num_extra_surfaces);
      break;

    case PROP_DISABLE_DPB:
      g_value_set_boolean (value, self->disable_dpb);
      break;

    case PROP_USE_FULL_FRAME:
      g_value_set_boolean (value, self->enable_full_frame);
      break;

    case PROP_ENABLE_FRAME_TYPE_REPORTING:
      g_value_set_boolean (value, self->enable_frame_type_reporting);
      break;

    case PROP_ENABLE_ERROR_CHECK:
      g_value_set_boolean (value, self->enable_error_check);
      break;

    case PROP_ENABLE_MAX_PERFORMANCE:
      g_value_set_boolean (value, self->enable_max_performance);
      break;

    case PROP_OPEN_MJPEG_BLOCK:
      g_value_set_boolean (value, self->v4l2output->open_mjpeg_block);
      break;

    case PROP_IS_GDR_STREAM:
      g_value_set_boolean (value, self->is_gdr_stream);
      break;

    case PROP_CAP_BUF_DYNAMIC_ALLOCATION:
      g_value_set_enum (value, self->cap_buf_dynamic_allocation);
      break;

    case PROP_EXTRACT_SEI_TYPE5_DATA:
      g_value_set_boolean (value, self->extract_sei_type5_data);
      break;

    case PROP_ATTACH_SEI_TIMESTAMPS:
      g_value_set_boolean (value, self->attach_sei_timestamps);
      break;

    case PROP_SEI_UUID_STRING:
      g_value_set_string (value, self->sei_uuid_string);
      break;
#endif
      /* By default read from output */
    default:
      if (!gst_v4l2_object_get_property_helper (self->v4l2output,
              prop_id, value, pspec)) {
        G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      }
      break;
  }
}

static void
gst_v4l2_video_dec_get_property_cuvid (GObject * object,
    guint prop_id, GValue * value, GParamSpec * pspec)
{
  GstV4l2VideoDec *self = GST_V4L2_VIDEO_DEC (object);

  switch (prop_id) {
    case PROP_CAPTURE_IO_MODE:
      if (!gst_v4l2_object_get_property_helper (self->v4l2capture,
              prop_id, value, pspec)) {
        G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      }
      break;

#ifdef USE_V4L2_TARGET_NV
    case PROP_SKIP_FRAME:
      g_value_set_enum (value, self->skip_frames);
      break;

    case PROP_DROP_FRAME_INTERVAL:
      g_value_set_uint (value, self->drop_frame_interval);
      break;

    case PROP_NUM_EXTRA_SURFACES:
      g_value_set_uint (value, self->num_extra_surfaces);
      break;

    case PROP_CUDADEC_MEM_TYPE:
      g_value_set_enum(value, self->cudadec_mem_type);
      break;

    case PROP_CUDADEC_GPU_ID:
      g_value_set_uint(value, self->cudadec_gpu_id);
      break;

    case PROP_CUDADEC_LOW_LATENCY:
    case PROP_DISABLE_DPB:
      g_value_set_boolean (value, self->cudadec_low_latency);
      break;

    case PROP_EXTRACT_SEI_TYPE5_DATA:
      g_value_set_boolean (value, self->extract_sei_type5_data);
      break;

    case PROP_ATTACH_SEI_TIMESTAMPS:
      g_value_set_boolean (value, self->attach_sei_timestamps);
      break;

    case PROP_SEI_UUID_STRING:
      g_value_set_string (value, self->sei_uuid_string);
      break;

    case PROP_CAP_BUF_DYNAMIC_ALLOCATION:
      g_value_set_enum (value, self->cap_buf_dynamic_allocation);
      break;
#endif
      /* By default read from output */
    default:
      if (!gst_v4l2_object_get_property_helper (self->v4l2output,
              prop_id, value, pspec)) {
        G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      }
      break;
  }
}

static gboolean
gst_v4l2_video_dec_open (GstVideoDecoder * decoder)
{
  GstV4l2VideoDec *self = GST_V4L2_VIDEO_DEC (decoder);
  GstCaps *codec_caps;

  GST_DEBUG_OBJECT (self, "Opening");

  if (!gst_v4l2_object_open (self->v4l2output))
    goto failure;

  if (!gst_v4l2_object_open_shared (self->v4l2capture, self->v4l2output))
    goto failure;

#ifdef USE_V4L2_TARGET_NV
  // Downstream plugins may export this environment variable
  if (is_cuvid == TRUE) {
    g_setenv ("DS_NEW_BUFAPI", "1", TRUE);
  }
#endif

  codec_caps = gst_pad_get_pad_template_caps (decoder->sinkpad);
  self->probed_sinkcaps = gst_v4l2_object_probe_caps (self->v4l2output,
      codec_caps);
  gst_caps_unref (codec_caps);

  if (gst_caps_is_empty (self->probed_sinkcaps))
    goto no_encoded_format;

  self->probed_srccaps = gst_v4l2_object_probe_caps (self->v4l2capture,
      gst_v4l2_object_get_raw_caps ());

  if (gst_caps_is_empty (self->probed_srccaps))
    goto no_raw_format;

  return TRUE;

no_encoded_format:
  GST_ELEMENT_ERROR (self, RESOURCE, SETTINGS,
      (_("Encoder on device %s has no supported input format"),
          self->v4l2output->videodev), (NULL));
  goto failure;


no_raw_format:
  GST_ELEMENT_ERROR (self, RESOURCE, SETTINGS,
      (_("Encoder on device %s has no supported output format"),
          self->v4l2output->videodev), (NULL));
  goto failure;

failure:
  if (GST_V4L2_IS_OPEN (self->v4l2output))
    gst_v4l2_object_close (self->v4l2output);

  if (GST_V4L2_IS_OPEN (self->v4l2capture))
    gst_v4l2_object_close (self->v4l2capture);

  gst_caps_replace (&self->probed_srccaps, NULL);
  gst_caps_replace (&self->probed_sinkcaps, NULL);

  return FALSE;
}

static gboolean
gst_v4l2_video_dec_close (GstVideoDecoder * decoder)
{
  GstV4l2VideoDec *self = GST_V4L2_VIDEO_DEC (decoder);

  GST_DEBUG_OBJECT (self, "Closing");

  gst_v4l2_object_close (self->v4l2output);
  gst_v4l2_object_close (self->v4l2capture);
  gst_caps_replace (&self->probed_srccaps, NULL);
  gst_caps_replace (&self->probed_sinkcaps, NULL);

  return TRUE;
}

static gboolean
gst_v4l2_video_dec_start (GstVideoDecoder * decoder)
{
  GstV4l2VideoDec *self = GST_V4L2_VIDEO_DEC (decoder);

  GST_DEBUG_OBJECT (self, "Starting");

  gst_v4l2_object_unlock (self->v4l2output);
  g_atomic_int_set (&self->active, TRUE);
  self->output_flow = GST_FLOW_OK;
#if USE_V4L2_TARGET_NV
  self->decoded_picture_cnt = 0;
#endif

  self->hash_pts_systemtime = g_hash_table_new_full (NULL, NULL, NULL, NULL);
  return TRUE;
}

static gboolean
gst_v4l2_video_dec_stop (GstVideoDecoder * decoder)
{
  GstV4l2VideoDec *self = GST_V4L2_VIDEO_DEC (decoder);

  GST_DEBUG_OBJECT (self, "Stopping");

  gst_v4l2_object_unlock (self->v4l2output);
  gst_v4l2_object_unlock (self->v4l2capture);

  /* Wait for capture thread to stop */
#ifdef USE_V4L2_TARGET_NV
  set_v4l2_video_mpeg_class (self->v4l2capture,
              V4L2_CID_MPEG_SET_POLL_INTERRUPT, 0);
#endif
  gst_pad_stop_task (decoder->srcpad);

  GST_VIDEO_DECODER_STREAM_LOCK (decoder);
  self->output_flow = GST_FLOW_OK;
  GST_VIDEO_DECODER_STREAM_UNLOCK (decoder);

  /* Should have been flushed already */
  g_assert (g_atomic_int_get (&self->active) == FALSE);

  gst_v4l2_object_stop (self->v4l2output);
  gst_v4l2_object_stop (self->v4l2capture);

  g_hash_table_remove_all (self->hash_pts_systemtime);
  g_hash_table_destroy (self->hash_pts_systemtime);

  if (self->input_state) {
    gst_video_codec_state_unref (self->input_state);
    self->input_state = NULL;
  }

  GST_DEBUG_OBJECT (self, "Stopped");

  return TRUE;
}

#ifdef USE_V4L2_TARGET_NV
gboolean set_v4l2_controls (GstV4l2VideoDec *self)
{
  GST_DEBUG_OBJECT(self, "set_v4l2_controls");
  {
    if (!set_v4l2_video_mpeg_class (self->v4l2output,
        V4L2_CID_MPEG_VIDEO_DISABLE_COMPLETE_FRAME_INPUT, 0)) {
      g_print ("S_EXT_CTRLS for DISABLE_COMPLETE_FRAME_INPUT failed\n");
      return FALSE;
    }
  }

  if (self->skip_frames != V4L2_SKIP_FRAMES_TYPE_NONE) {
    if (!set_v4l2_video_mpeg_class (self->v4l2output,
        V4L2_CID_MPEG_VIDEO_SKIP_FRAMES,
        self->skip_frames)) {
      g_print ("S_EXT_CTRLS for SKIP_FRAMES failed\n");
      return FALSE;
    }
  }

#if 0
  /* *
   * TODO: From low level library remove support of drop frame interval after
   * analyzing high CPU utilization in initial implementation.
   * */
  if (self->drop_frame_interval != 0) {
    if (!set_v4l2_video_mpeg_class (self->v4l2output,
        V4L2_CID_MPEG_VIDEODEC_DROP_FRAME_INTERVAL,
        self->drop_frame_interval)) {
      g_print ("S_EXT_CTRLS for DROP_FRAME_INTERVAL failed\n");
      return FALSE;
    }
  }
#endif
  if (self->disable_dpb != DEFAULT_DISABLE_DPB) {
    if (!set_v4l2_video_mpeg_class (self->v4l2output,
        V4L2_CID_MPEG_VIDEO_DISABLE_DPB,
        self->disable_dpb)) {
      g_print ("S_EXT_CTRLS for DISABLE_DPB failed\n");
      return FALSE;
    }
  }

  if (self->enable_full_frame != DEFAULT_FULL_FRAME) {
    if (!set_v4l2_video_mpeg_class (self->v4l2output,
        V4L2_CID_MPEG_VIDEO_DISABLE_COMPLETE_FRAME_INPUT,
        self->enable_full_frame)) {
      g_print ("S_EXT_CTRLS for DISABLE_COMPLETE_FRAME_INPUT failed\n");
      return FALSE;
    }
  }

  if (self->enable_frame_type_reporting) {
    if (!set_v4l2_video_mpeg_class (self->v4l2output,
        V4L2_CID_MPEG_VIDEO_ERROR_REPORTING,
        self->enable_frame_type_reporting)) {
      g_print ("S_EXT_CTRLS for ERROR_REPORTING failed\n");
      return FALSE;
    }
  }

  if (self->enable_error_check) {
    if (!set_v4l2_video_mpeg_class (self->v4l2output,
        V4L2_CID_MPEG_VIDEO_ERROR_REPORTING,
        self->enable_error_check)) {
      g_print ("S_EXT_CTRLS for ERROR_REPORTING failed\n");
      return FALSE;
    }
  }

  if (self->enable_max_performance != DEFAULT_MAX_PERFORMANCE) {
    if (!set_v4l2_video_mpeg_class (self->v4l2output,
        V4L2_CID_MPEG_VIDEO_MAX_PERFORMANCE,
        self->enable_max_performance)) {
      g_print ("S_EXT_CTRLS for MAX_PERFORMANCE failed\n");
      return FALSE;
    }
  }

  if (self->is_gdr_stream) {
    if (!set_v4l2_video_mpeg_class (self->v4l2output,
        V4L2_CID_MPEG_VIDEODEC_GDR_STREAM,
        self->is_gdr_stream)) {
      g_print ("S_EXT_CTRLS for GDR_STREAM failed\n");
      return FALSE;
    }
  }

  if (is_cuvid == TRUE) {
    if (!set_v4l2_video_mpeg_class (self->v4l2output,
          V4L2_CID_MPEG_VIDEO_CUDA_MEM_TYPE,
          self->cudadec_mem_type)) {
        g_print ("S_EXT_CTRLS for CUDA_MEM_TYPE failed\n");
        return FALSE;
    }

    if (!set_v4l2_video_mpeg_class (self->v4l2output,
          V4L2_CID_MPEG_VIDEO_CUDA_GPU_ID,
          self->cudadec_gpu_id)) {
        g_print ("S_EXT_CTRLS for CUDA_GPU_ID failed\n");
        return FALSE;
    }
  }

  return TRUE;
}
#endif

static gboolean
gst_v4l2_video_dec_set_format (GstVideoDecoder * decoder,
    GstVideoCodecState * state)
{
  GstV4l2Error error = GST_V4L2_ERROR_INIT;
  gboolean ret = TRUE;
  GstV4l2VideoDec *self = GST_V4L2_VIDEO_DEC (decoder);

  GST_DEBUG_OBJECT (self, "Setting format: %" GST_PTR_FORMAT, state->caps);

  if (self->input_state) {
#ifndef USE_V4L2_TARGET_NV
    if (gst_v4l2_object_caps_equal (self->v4l2output, state->caps)) {
      GST_DEBUG_OBJECT (self, "Compatible caps");
      goto done;
    }
#else
    GstV4l2BufferPool *v4l2pool = GST_V4L2_BUFFER_POOL(self->v4l2output->pool);
    GstV4l2Object *obj = v4l2pool->obj;
    if ((GST_V4L2_PIXELFORMAT(obj) == V4L2_PIX_FMT_H264) ||
        (GST_V4L2_PIXELFORMAT(obj) == V4L2_PIX_FMT_H265))
    {
      GST_INFO_OBJECT(self, "Pix format is h264 or h265. Bypassing set_format."
                            "Reset done from handle_frame");
      self->output_flow = GST_FLOW_OK;
      self->set_format = true;
      return TRUE;
    }

#endif
    gst_video_codec_state_unref (self->input_state);
    self->input_state = NULL;

    gst_v4l2_video_dec_finish (decoder);
    gst_v4l2_object_stop (self->v4l2output);

    /* The renegotiation flow don't blend with the base class flow. To
     * properly stop the capture pool we need to reclaim our buffers, which
     * will happend through the allocation query. The allocation query is
     * triggered by gst_video_decoder_negotiate() which requires the output
     * caps to be set, but we can't know this information as we rely on the
     * decoder, which requires the capture queue to be stopped.
     *
     * To workaround this issue, we simply run an allocation query with the
     * old negotiated caps in order to drain/reclaim our buffers. That breaks
     * the complexity and should not have much impact in performance since the
     * following allocation query will happen on a drained pipeline and won't
     * block. */
    {
      GstCaps *caps = gst_pad_get_current_caps (decoder->srcpad);
      if (caps) {
        GstQuery *query = gst_query_new_allocation (caps, FALSE);
        gst_pad_peer_query (decoder->srcpad, query);
        gst_query_unref (query);
        gst_caps_unref (caps);
      }
    }

    gst_v4l2_object_stop (self->v4l2capture);
    self->output_flow = GST_FLOW_OK;
  }

  ret = gst_v4l2_object_set_format (self->v4l2output, state->caps, &error);

  if (ret)
    self->input_state = gst_video_codec_state_ref (state);
  else
    gst_v4l2_error (self, &error);

#ifdef USE_V4L2_TARGET_NV
  ret = set_v4l2_controls(self);
#endif

#ifndef USE_V4L2_TARGET_NV
done:
#endif
  return ret;
}

static gboolean
gst_v4l2_video_dec_flush (GstVideoDecoder * decoder)
{
  GstV4l2VideoDec *self = GST_V4L2_VIDEO_DEC (decoder);

  GST_DEBUG_OBJECT (self, "Flushed");

  /* Ensure the processing thread has stopped for the reverse playback
   * discount case */
  if (gst_pad_get_task_state (decoder->srcpad) == GST_TASK_STARTED) {
    GST_VIDEO_DECODER_STREAM_UNLOCK (decoder);

    gst_v4l2_object_unlock (self->v4l2output);
    gst_v4l2_object_unlock (self->v4l2capture);
#ifdef USE_V4L2_TARGET_NV
    set_v4l2_video_mpeg_class (self->v4l2capture,
              V4L2_CID_MPEG_SET_POLL_INTERRUPT, 0);
#endif
    gst_pad_stop_task (decoder->srcpad);
    GST_VIDEO_DECODER_STREAM_LOCK (decoder);
  }

  self->output_flow = GST_FLOW_OK;

  gst_v4l2_object_unlock_stop (self->v4l2output);
  gst_v4l2_object_unlock_stop (self->v4l2capture);

#ifdef USE_V4L2_TARGET_NV
  if (is_cuvid == TRUE) {
    /* Flush out low-level cuvid decoder buffers */
    set_v4l2_video_mpeg_class (self->v4l2output, V4L2_CID_MPEG_VIDEODEC_FLUSH_BUFFERS, 1);
  }
#endif

  if (self->v4l2output->pool)
    gst_v4l2_buffer_pool_flush (self->v4l2output->pool);

  if (self->v4l2capture->pool)
    gst_v4l2_buffer_pool_flush (self->v4l2capture->pool);

  return TRUE;
}

static gboolean
gst_v4l2_video_dec_negotiate (GstVideoDecoder * decoder)
{
  GstV4l2VideoDec *self = GST_V4L2_VIDEO_DEC (decoder);

  /* We don't allow renegotiation without carefull disabling the pool */
  if (self->v4l2capture->pool &&
      gst_buffer_pool_is_active (GST_BUFFER_POOL (self->v4l2capture->pool)))
    return TRUE;

  return GST_VIDEO_DECODER_CLASS (parent_class)->negotiate (decoder);
}

static gboolean
gst_v4l2_decoder_cmd (GstV4l2Object * v4l2object, guint cmd, guint flags)
{
  struct v4l2_decoder_cmd dcmd = { 0, };

  GST_DEBUG_OBJECT (v4l2object->element,
      "sending v4l2 decoder command %u with flags %u", cmd, flags);

  if (!GST_V4L2_IS_OPEN (v4l2object))
    return FALSE;

  dcmd.cmd = cmd;
  dcmd.flags = flags;
  if (v4l2object->ioctl (v4l2object->video_fd, VIDIOC_DECODER_CMD, &dcmd) < 0)
    goto dcmd_failed;

  return TRUE;

dcmd_failed:
  if (errno == ENOTTY) {
    GST_INFO_OBJECT (v4l2object->element,
        "Failed to send decoder command %u with flags %u for '%s'. (%s)",
        cmd, flags, v4l2object->videodev, g_strerror (errno));
  } else {
    GST_ERROR_OBJECT (v4l2object->element,
        "Failed to send decoder command %u with flags %u for '%s'. (%s)",
        cmd, flags, v4l2object->videodev, g_strerror (errno));
  }
  return FALSE;
}

static GstFlowReturn
gst_v4l2_video_dec_finish (GstVideoDecoder * decoder)
{
  GstV4l2VideoDec *self = GST_V4L2_VIDEO_DEC (decoder);
  GstFlowReturn ret = GST_FLOW_OK;
  GstBuffer *buffer;

  if (gst_pad_get_task_state (decoder->srcpad) != GST_TASK_STARTED)
    goto done;

  GST_DEBUG_OBJECT (self, "Finishing decoding");

  GST_VIDEO_DECODER_STREAM_UNLOCK (decoder);

#ifndef USE_V4L2_TARGET_NV
  if (gst_v4l2_decoder_cmd (self->v4l2output, V4L2_DEC_CMD_STOP, 0)) {
#else
  if (gst_v4l2_decoder_cmd (self->v4l2output, V4L2_DEC_CMD_STOP,
          V4L2_DEC_CMD_STOP_TO_BLACK)) {
#endif
    GstTask *task = decoder->srcpad->task;
    if (!task) {
        goto stop_task;
    }

    /* If the decoder stop command succeeded, just wait until processing is
     * finished */
    GST_OBJECT_LOCK (task);
    while (GST_TASK_STATE (task) == GST_TASK_STARTED)
      GST_TASK_WAIT (task);
    GST_OBJECT_UNLOCK (task);
    ret = GST_FLOW_FLUSHING;
  } else {
    /* otherwise keep queuing empty buffers until the processing thread has
     * stopped, _pool_process() will return FLUSHING when that happened */
    while (ret == GST_FLOW_OK) {
      buffer = gst_buffer_new ();
      ret =
          gst_v4l2_buffer_pool_process (GST_V4L2_BUFFER_POOL (self->
              v4l2output->pool), &buffer);
      gst_buffer_unref (buffer);
    }
  }

  /* and ensure the processing thread has stopped in case another error
   * occured. */
stop_task:
  gst_v4l2_object_unlock (self->v4l2capture);
#ifdef USE_V4L2_TARGET_NV
  set_v4l2_video_mpeg_class (self->v4l2capture,
              V4L2_CID_MPEG_SET_POLL_INTERRUPT, 0);
#endif
  gst_pad_stop_task (decoder->srcpad);
  GST_VIDEO_DECODER_STREAM_LOCK (decoder);

  if (ret == GST_FLOW_FLUSHING)
    ret = self->output_flow;

  GST_DEBUG_OBJECT (decoder, "Done draining buffers");

  /* TODO Shall we cleanup any reffed frame to workaround broken decoders ? */

done:
  return ret;
}

static GstFlowReturn
gst_v4l2_video_dec_drain (GstVideoDecoder * decoder)
{
#if ENABLE_DRAIN
  GstV4l2VideoDec *self = GST_V4L2_VIDEO_DEC (decoder);

  GST_DEBUG_OBJECT (self, "Draining...");
  gst_v4l2_video_dec_finish (decoder);
  gst_v4l2_video_dec_flush (decoder);

  return GST_FLOW_OK;
#else
  return GST_FLOW_OK;
#endif
}

#ifdef USE_V4L2_TARGET_NV
GstVideoCodecFrame *
gst_v4l2_video_dec_find_nearest_frame (GstBuffer * buf, GList * frames)
{
  GstVideoCodecFrame *best = NULL;
  GstClockTimeDiff best_diff = G_MAXINT64;
  GstClockTime timestamp;
  GList *l;

  timestamp = buf->pts;

  for (l = frames; l; l = l->next) {
    GstVideoCodecFrame *tmp = l->data;
    GstClockTimeDiff diff = ABS (GST_CLOCK_DIFF (timestamp, tmp->pts));

    if (diff < best_diff) {
      best = tmp;
      best_diff = diff;

      if (diff == 0)
        break;
    }
  }

  if (best)
    gst_video_codec_frame_ref (best);

  g_list_foreach (frames, (GFunc) gst_video_codec_frame_unref, NULL);
  g_list_free (frames);

  return best;
}

static void
gst_v4l2_video_dec_clean_older_frames (GstV4l2VideoDec * self,
    GstBuffer * buf, GList * frames)
{
  GList *l;
  GstClockTime timestamp;

  timestamp = buf->pts;

  if (GST_CLOCK_TIME_IS_VALID (timestamp)) {
    /* We could release all frames stored with pts < timestamp since the
     * decoder will likely output frames in display order */
    for (l = frames; l; l = l->next) {
      GstVideoCodecFrame *tmp = l->data;

      if (tmp->pts < timestamp) {
        GST_LOG_OBJECT (self,
            "discarding ghost frame %p (#%d) PTS:%" GST_TIME_FORMAT " DTS:%"
            GST_TIME_FORMAT, tmp, tmp->system_frame_number,
            GST_TIME_ARGS (tmp->pts), GST_TIME_ARGS (tmp->dts));
        gst_video_decoder_release_frame (GST_VIDEO_DECODER (self), tmp);
      } else {
        gst_video_codec_frame_unref (tmp);
      }
    }
  } else {
    /* We will release all frames with invalid timestamp because we don't even
     * know if they will be output some day. */
    for (l = frames; l; l = l->next) {
      GstVideoCodecFrame *tmp = l->data;

      if (!GST_CLOCK_TIME_IS_VALID (tmp->pts)) {
        GST_LOG_OBJECT (self,
            "discarding frame %p (#%d) with invalid PTS:%" GST_TIME_FORMAT
            " DTS:%" GST_TIME_FORMAT, tmp, tmp->system_frame_number,
            GST_TIME_ARGS (tmp->pts), GST_TIME_ARGS (tmp->dts));
        gst_video_decoder_release_frame (GST_VIDEO_DECODER (self), tmp);
      } else {
        gst_video_codec_frame_unref (tmp);
      }
    }
  }

  g_list_free (frames);
}

#endif

#ifndef USE_V4L2_TARGET_NV
static GstVideoCodecFrame *
gst_v4l2_video_dec_get_oldest_frame (GstVideoDecoder * decoder)
{
  GstVideoCodecFrame *frame = NULL;
  GList *frames, *l;
  gint count = 0;

  frames = gst_video_decoder_get_frames (decoder);

  for (l = frames; l != NULL; l = l->next) {
    GstVideoCodecFrame *f = l->data;

    if (!frame || frame->pts > f->pts)
      frame = f;

    count++;
  }

  if (frame) {
    GST_LOG_OBJECT (decoder,
        "Oldest frame is %d %" GST_TIME_FORMAT " and %d frames left",
        frame->system_frame_number, GST_TIME_ARGS (frame->pts), count - 1);
    gst_video_codec_frame_ref (frame);
  }

  g_list_free_full (frames, (GDestroyNotify) gst_video_codec_frame_unref);

  return frame;
}
#endif

static void
gst_v4l2_video_dec_loop (GstVideoDecoder * decoder)
{
  GstV4l2VideoDec *self = GST_V4L2_VIDEO_DEC (decoder);
  GstV4l2BufferPool *v4l2_pool = GST_V4L2_BUFFER_POOL (self->v4l2capture->pool);
  GstBufferPool *pool;
  GstVideoCodecFrame *frame;
  GstBuffer *buffer = NULL;
  GstFlowReturn ret;

  GST_LOG_OBJECT (decoder, "Allocate output buffer");

  self->output_flow = GST_FLOW_OK;
  do {
    /* We cannot use the base class allotate helper since it taking the internal
     * stream lock. we know that the acquire may need to poll until more frames
     * comes in and holding this lock would prevent that.
     */
    pool = gst_video_decoder_get_buffer_pool (decoder);

    /* Pool may be NULL if we started going to READY state */
    if (pool == NULL) {
      ret = GST_FLOW_FLUSHING;
      goto beach;
    }

    ret = gst_buffer_pool_acquire_buffer (pool, &buffer, NULL);
    g_object_unref (pool);

    if (ret != GST_FLOW_OK)
      goto beach;

    GST_LOG_OBJECT (decoder, "Process output buffer");
    ret = gst_v4l2_buffer_pool_process (v4l2_pool, &buffer);

  } while (ret == GST_V4L2_FLOW_CORRUPTED_BUFFER);

  if (ret != GST_FLOW_OK)
    goto beach;

#ifdef USE_V4L2_TARGET_NV
  frame = gst_v4l2_video_dec_find_nearest_frame (buffer,
          gst_video_decoder_get_frames (GST_VIDEO_DECODER (self)));
  /* So we have a timestamped  buffer and get, or not, corresponding frame.
   * Assuming decoder output frames in display order, frames preceding this
   * frame could be discarded as they seems useless due to e.g interlaced
   * stream, corrupted input data...
   * In any cases, not likely to be seen again. so drop it before they pile up
   * and use all the memory. */
  gst_v4l2_video_dec_clean_older_frames (self, buffer,
          gst_video_decoder_get_frames (GST_VIDEO_DECODER (self)));
#else
  frame = gst_v4l2_video_dec_get_oldest_frame (decoder);
#endif

#ifdef USE_V4L2_TARGET_NV
  if (is_cuvid == FALSE) {
    if (frame && self->enable_frame_type_reporting) {
      g_print ("Frame %d\n", frame->system_frame_number);
    }
  }
#endif
  if (frame) {
    frame->output_buffer = buffer;
    buffer = NULL;

    if(enable_latency_measurement) /* TODO with better option */
    {
      g_mutex_lock(&self->pts_hashtable_lock);
      gsize frame_pts = frame->pts;
      gpointer in_time = g_hash_table_lookup (self->hash_pts_systemtime, GSIZE_TO_POINTER(frame_pts));
      gdouble input_time = (gdouble) GPOINTER_TO_SIZE(in_time);
      gdouble output_time = get_current_system_timestamp ();

      if (output_time < input_time)
      {
          gdouble time = G_MAXDOUBLE - input_time;
          if(enable_latency_print == TRUE)
          {
            g_print ("[%s] True Decode Latency = %f \n",
                    GST_OBJECT_NAME (self), output_time + time);
          } else {
            GST_DEBUG_OBJECT (self, "True Decode Latency = %f ms\n",
                    output_time + time);
          }
      }
      else
      {
          if(enable_latency_print == TRUE)
          {
            g_print ("[%s] True Decode Latency = %f \n",
                    GST_OBJECT_NAME (self), (output_time - input_time));
          } else {
            GST_DEBUG_OBJECT (self, "True Decode Latency = %f ms\n",
                    (output_time - input_time));
          }
      }

      //Remove the hash table entry for the frame pts
      g_hash_table_remove (self->hash_pts_systemtime, GSIZE_TO_POINTER(frame_pts));
      g_mutex_unlock(&self->pts_hashtable_lock);

      GstCaps *reference = gst_caps_new_simple ("video/x-raw",
          "component_name", G_TYPE_STRING, GST_ELEMENT_NAME(self),
          "frame_num", G_TYPE_INT, self->frame_num++,
          "in_timestamp", G_TYPE_DOUBLE, input_time,
          "out_timestamp", G_TYPE_DOUBLE, output_time,
          NULL);
      GstReferenceTimestampMeta * dec_meta =
        gst_buffer_add_reference_timestamp_meta (frame->output_buffer, reference,
            0, 0);
      if(dec_meta == NULL)
      {
        GST_DEBUG_OBJECT (decoder, "dec_meta: %p", dec_meta);
      }
      gst_caps_unref(reference);
    }

#if USE_V4L2_TARGET_NV

    GST_VIDEO_DECODER_STREAM_LOCK (decoder);
    if (!gst_buffer_copy_into (frame->output_buffer, frame->input_buffer,
          (GstBufferCopyFlags)GST_BUFFER_COPY_METADATA, 0, -1)) {
      GST_DEBUG_OBJECT (decoder, "Buffer metadata copy failed \n");
    }
    GST_VIDEO_DECODER_STREAM_UNLOCK (decoder);

    if ((self->drop_frame_interval == 0) ||
        (self->decoded_picture_cnt % self->drop_frame_interval == 0))
        ret = gst_video_decoder_finish_frame (decoder, frame);
    else
        ret = gst_video_decoder_drop_frame (GST_VIDEO_DECODER (self), frame);

    if (ret != GST_FLOW_OK)
      goto beach;

    self->decoded_picture_cnt += 1;
#else
    ret = gst_video_decoder_finish_frame (decoder, frame);
#endif

  } else {
    GST_WARNING_OBJECT (decoder, "Decoder is producing too many buffers");
    gst_buffer_unref (buffer);
  }

  return;

beach:
  GST_DEBUG_OBJECT (decoder, "Leaving output thread: %s",
      gst_flow_get_name (ret));

  gst_buffer_replace (&buffer, NULL);
  self->output_flow = ret;
  gst_v4l2_object_unlock (self->v4l2output);
  gst_pad_pause_task (decoder->srcpad);
}

static gboolean
gst_v4l2_video_remove_padding (GstCapsFeatures * features,
    GstStructure * structure, gpointer user_data)
{
  GstV4l2VideoDec *self = GST_V4L2_VIDEO_DEC (user_data);
  GstVideoAlignment *align = &self->v4l2capture->align;
  GstVideoInfo *info = &self->v4l2capture->info;
#ifndef USE_V4L2_TARGET_NV
  int width, height;

  if (!gst_structure_get_int (structure, "width", &width))
    return TRUE;

  if (!gst_structure_get_int (structure, "height", &height))
    return TRUE;
#else
  guint width = 0, height = 0;

  if (!gst_structure_get_int (structure, "width", (gint *) & width))
    return TRUE;

  if (!gst_structure_get_int (structure, "height", (gint *) & height))
    return TRUE;
#endif
  if (align->padding_left != 0 || align->padding_top != 0 ||
      height != info->height + align->padding_bottom)
    return TRUE;

  if (height == info->height + align->padding_bottom) {
    /* Some drivers may round up width to the padded with */
    if (width == info->width + align->padding_right)
      gst_structure_set (structure,
          "width", G_TYPE_INT, width - align->padding_right,
          "height", G_TYPE_INT, height - align->padding_bottom, NULL);
    /* Some drivers may keep visible width and only round up bytesperline */
#ifndef USE_V4L2_TARGET_NV
    else if (width == info->width)
#else
    else if (width == (guint) info->width)
#endif
      gst_structure_set (structure,
          "height", G_TYPE_INT, height - align->padding_bottom, NULL);
  }

  return TRUE;
}


static const char * const NvBufSurfaceMemType_names[] =
{
  [NVBUF_MEM_DEFAULT] = "nvbuf-mem-default",
  [NVBUF_MEM_CUDA_PINNED] = "nvbuf-mem-cuda-pinned",
  [NVBUF_MEM_CUDA_DEVICE] = "nvbuf-mem-cuda-device",
  [NVBUF_MEM_CUDA_UNIFIED] = "nvbuf-mem-cuda-unified",
  [NVBUF_MEM_SURFACE_ARRAY] = "nvbuf-mem-surface-array",
  [NVBUF_MEM_HANDLE] = "nvbuf-mem-handle",
  [NVBUF_MEM_SYSTEM] = "nvbuf-mem-system",
  [NVBUF_MEM_CUDA_ARRAY] = "nvbuf-mem-cuda-array",
};

static gboolean
gst_h265_parse_process_nal (GstV4l2VideoDec *self, H265NalUnit * nalu)
{
  H265SPS sps = { 0, };
  guint nal_type;
  H265Parser nalparser;
  H265ParserResult pres = H265_PARSER_ERROR;

  /* nothing to do for broken input */
  if (G_UNLIKELY (nalu->size < 2)) {
    return TRUE;
  }

  /* we have a peek as well */
  nal_type = nalu->type;

  switch (nal_type) {
    case H265_NAL_SPS:
      /* reset state, everything else is obsolete */

      pres = h265_parser_parse_sps (&nalparser, nalu, &sps, TRUE);

      /* arranged for a fallback sps.id, so use that one and only warn */
      if (pres != H265_PARSER_OK) {
        /* try to not parse VUI */
        pres = h265_parser_parse_sps (&nalparser, nalu, &sps, FALSE);
        if (pres != H265_PARSER_OK) {
          return FALSE;
        }
      }
      self->current_width = sps.width;
      self->current_height = sps.height;
      self->current_dpb_size =
          sps.max_dec_pic_buffering_minus1[sps.max_sub_layers_minus1] + 1;
      break;
  }

  return TRUE;
}

static gboolean
gst_h264_parse_process_nal (GstV4l2VideoDec *self, H264NalUnit * nalu)
{
  guint nal_type;
  H264SPS sps = { 0, };
  H264ParserResult pres;

  /* nothing to do for broken input */
  if (G_UNLIKELY (nalu->size < 2)) {
    GST_DEBUG_OBJECT (self, "not processing nal size %u", nalu->size);
    return TRUE;
  }

  /* we have a peek as well */
  nal_type = nalu->type;

  GST_DEBUG_OBJECT (self, "processing nal of type %u , size %u",
      nal_type, nalu->size);

  if (nal_type == H264_NAL_SPS)
  {
      GST_DEBUG_OBJECT (self, "GOT SPS frame\n");
  }

  switch (nal_type) {
    case H264_NAL_SPS:
      /* reset state, everything else is obsolete */
      pres = h264_parser_parse_sps (nalu, &sps, TRUE);

      /* arranged for a fallback sps.id, so use that one and only warn */
      if (pres != H264_PARSER_OK) {
        GST_WARNING_OBJECT (self, "failed to parse SPS:");
        return FALSE;
      }

      self->current_width = (sps.pic_width_in_mbs_minus1 + 1) << 4;
      self->current_height = (sps.pic_height_in_map_units_minus1 + 1) << 4;
      self->current_dpb_size = sps.num_ref_frames;

      break;
  }

  return TRUE;
}

static GstFlowReturn
gst_v4l2_h265_stream_parser (GstV4l2VideoDec *self, gpointer data, guint32 size)
{
    H265NalUnit nalu = {0};
    H265ParserResult pres = H265_PARSER_ERROR;
    gint current_off = 0;
    guint nalu_size = 0;
    GstFlowReturn ret = GST_FLOW_OK;

    while (TRUE)
    {
        pres = h265_parser_identify_nalu (NULL, data, current_off, size, &nalu);

        nalu_size += nalu.size + 4;
        if (nalu_size >= size)
            break;

        switch (pres) {
            case H265_PARSER_OK:
                GST_DEBUG_OBJECT (self, "complete nal (offset, size): (%u, %u) ",
                        nalu.offset, nalu.size);
                break;
            case H265_PARSER_NO_NAL_END:
                GST_WARNING_OBJECT (self, "Start of the nal found, but not the end");
                break;
            case H265_PARSER_ERROR:
                /* should not really occur either */
                GST_WARNING_OBJECT (self, "Error parsing H.265 stream. Invalid H.265 stream");
                break;
            case H265_PARSER_NO_NAL:
                GST_WARNING_OBJECT (self, "No H.265 NAL unit found");
                break;
            default:
                break;
        }

        GST_DEBUG_OBJECT (self, "%p complete nal found. Off: %u, Size: %u",
                data, nalu.offset, nalu.size);

        if (!gst_h265_parse_process_nal (self, &nalu)) {
            GST_WARNING_OBJECT (self,
                    "broken/invalid nal Type: %d, Size: %u will be dropped",
                    nalu.type, nalu.size);
        }
        current_off += nalu.size;
    }
    GST_DEBUG_OBJECT (self,
            "sps width = %d height = %d \n", self->current_width, self->current_height);

    return ret;
}

static GstFlowReturn
gst_v4l2_h264_stream_parser (GstV4l2VideoDec *self, gpointer data, guint32 size)
{
    H264NalUnit nalu;
    H264ParserResult pres;
    gint current_off = 0;
    guint nalu_size = 0;
    GstFlowReturn ret = GST_FLOW_OK;

    while (TRUE)
    {
        pres = h264_parser_identify_nalu (NULL, data, current_off, size, &nalu);

        nalu_size += nalu.size + 4;
        if (nalu_size >= size)
            break;

        switch (pres)
        {
          case H264_PARSER_OK:
            GST_DEBUG_OBJECT(self, "complete nal (offset, size): (%u, %u) ",
                             nalu.offset, nalu.size);
            break;
          case H264_PARSER_NO_NAL_END:
            GST_WARNING_OBJECT(self, "parser will assume that the end of the data is the end of the NAL unit");
            break;
          case H264_PARSER_ERROR:
            /* should not really occur either */
            GST_WARNING_OBJECT(self, "Error parsing H.264 stream. Invalid H.264 stream");
            break;
          case H264_PARSER_NO_NAL:
            GST_WARNING_OBJECT(self, "Error parsing H.264 stream. No H.264 NAL unit found");
            break;
        default:
            break;
        }

        GST_DEBUG_OBJECT (self, "%p complete nal found. Off: %u, Size: %u",
                data, nalu.offset, nalu.size);

        if (!gst_h264_parse_process_nal (self, &nalu)) {
            GST_WARNING_OBJECT (self,
                    "broken/invalid nal Type: %d, Size: %u will be dropped",
                    nalu.type, nalu.size);
        }
        current_off += nalu.size;
    }
    GST_DEBUG_OBJECT (self,
            "sps width = %d height = %d", self->current_width, self->current_height);

    return ret;
}

static gboolean
findvpxStartCode(GstV4l2Object *obj, const uint8_t* data, int size)
{
    for (int i = 0; i < size - 2; i++)
    {
        if (GST_V4L2_PIXELFORMAT(obj) == V4L2_PIX_FMT_VP8)
        {
            if (data[i] == VP8_START_BYTE_0 &&
                data[i + 1] == VP8_START_BYTE_1)
            {
                return TRUE;
            }
        }
        else if (GST_V4L2_PIXELFORMAT(obj) == V4L2_PIX_FMT_VP9)
        {
            if (data[i] == VP9_START_BYTE_0 &&
                data[i + 1] == VP9_START_BYTE_1 &&
                data[i + 2] == VP9_START_BYTE_2)
            {
                return TRUE;
            }
        }
    }
    return FALSE;  // Start code not found
}

guint64 extract_sei_json_int (const char *input, const char *field_name)
{
    char key[128];
    snprintf(key, sizeof(key), "\"%s\"", field_name);
    char *start = strstr(input, key);
    if (start) {
        start += strlen(key);
        while (*start == ' ' || *start == '\t')
            start++;
        if (*start == ':')
            start++;
        while (*start == ' ' || *start == '\t')
            start++;
        return (guint64) strtoll(start, NULL, 10);
    }
    return 0;
}

guint64 convert_string_to_ts (GstV4l2VideoDec * self, const char *input)
{
    guint64 timestamp = extract_sei_json_int (input, "timestamp");
    if (timestamp == 0) {
        GST_DEBUG_OBJECT (self, "timestamp key not found.");
    }

    return timestamp;
}

static GstFlowReturn
gst_v4l2_video_dec_handle_frame (GstVideoDecoder * decoder,
        GstVideoCodecFrame * frame)
{
    GstV4l2Error error = GST_V4L2_ERROR_INIT;
    GstV4l2VideoDec *self = GST_V4L2_VIDEO_DEC (decoder);
    GstFlowReturn ret = GST_FLOW_OK;
    gboolean vpx_ret = FALSE;
    gboolean processed = FALSE;
    GstBuffer *tmp;
    GstTaskState task_state;
#ifdef USE_V4L2_TARGET_NV
    gboolean trigger_drc = false;
    GstV4l2BufferPool *v4l2pool = GST_V4L2_BUFFER_POOL (self->v4l2output->pool);
    GstV4l2Object *obj = v4l2pool->obj;
#endif

    GST_DEBUG_OBJECT (self, "Handling frame %d", frame->system_frame_number);
#ifdef USE_V4L2_TARGET_NV
    if (is_cuvid == TRUE) {
        if (((GST_V4L2_PIXELFORMAT(obj) == V4L2_PIX_FMT_VP8) ||
            (GST_V4L2_PIXELFORMAT(obj) == V4L2_PIX_FMT_VP9)) && self->valid_vpx == FALSE)
        {
            GstMapInfo map = {0};
            if (!gst_buffer_map(frame->input_buffer, &map, GST_MAP_READ))
            {
              GST_ERROR_OBJECT(self, "couldnt map frame input_buffer\n");
            }

            vpx_ret = findvpxStartCode(obj, map.data, map.size);
            if (vpx_ret == FALSE)
            {
                gst_buffer_unmap (frame->input_buffer, &map);
                goto drop;
            }
            else if (vpx_ret == TRUE)
            {
                self->valid_vpx = TRUE;
            }

            gst_buffer_unmap (frame->input_buffer, &map);
        }
    }
    if (((GST_V4L2_PIXELFORMAT(obj) == V4L2_PIX_FMT_H264) ||
         (GST_V4L2_PIXELFORMAT(obj) == V4L2_PIX_FMT_H265)) &&
        !(GST_BUFFER_FLAG_IS_SET(GST_BUFFER_CAST(frame->input_buffer),
                                 GST_BUFFER_FLAG_DELTA_UNIT)))
    {
        GstMapInfo map = {0};
        if (!gst_buffer_map(frame->input_buffer, &map, GST_MAP_READ))
        {
            GST_ERROR_OBJECT(self, "couldnt map frame input_buffer\n");
        }

        if (GST_V4L2_PIXELFORMAT(obj) == V4L2_PIX_FMT_H264)
        {
            if (gst_v4l2_h264_stream_parser(self, map.data, map.size) != GST_FLOW_OK)
                GST_ERROR_OBJECT(self, "h264 stream parsing failed");
        }
        else if ((GST_V4L2_PIXELFORMAT(obj) == V4L2_PIX_FMT_H265))
        {
            if (gst_v4l2_h265_stream_parser(self, map.data, map.size) != GST_FLOW_OK)
                GST_ERROR_OBJECT(self, "h265 stream parsing failed");
        }
        gst_buffer_unmap(frame->input_buffer, &map);

        if (self->old_width || self->old_height)
        {
            if ((self->old_width != self->current_width) ||
                (self->old_height != self->current_height))
            {
                GST_INFO_OBJECT(self, "Decoder found new resolution. triggering DRC seq.");
                GST_INFO_OBJECT(self, "Old Resolution: [%d x %d], New Resolution: [%d x %d]",
                                self->old_width, self->old_height, self->current_width, self->current_height);
                trigger_drc = true;
            }
            else if (self->current_dpb_size > self->old_dpb_size)
            {
                /* Same size, but the new sequence needs a larger DPB than the
                 * previous one: restart the decoder like a resolution change. */
                GST_INFO_OBJECT(self, "Decoder found larger DPB. triggering DRC seq.");
                GST_INFO_OBJECT(self, "Old DPB: %u, New DPB: %u",
                                self->old_dpb_size, self->current_dpb_size);
                trigger_drc = true;
            }
        }

        self->old_width = self->current_width;
        self->old_height = self->current_height;
        self->old_dpb_size = self->current_dpb_size;

        if (trigger_drc == true)
        {
            GstStructure *structure = NULL;
            GstCaps *dec_sink_caps = gst_pad_get_current_caps(decoder->sinkpad);
            if (dec_sink_caps != NULL)
            {
                dec_sink_caps = gst_caps_make_writable(dec_sink_caps);
                structure = gst_caps_get_structure(dec_sink_caps, 0);
                gst_structure_set(structure, "width", G_TYPE_INT, self->current_width,
                                  "height", G_TYPE_INT, self->current_height,
                                  NULL);
                /* Replace coded size with visible size, we want to negotiate visible size
                 * with downstream, not coded size. */
                gst_caps_map_in_place(dec_sink_caps, gst_v4l2_video_remove_padding, self);
                GST_DEBUG_OBJECT(self, "dec_sink_caps: %s", gst_caps_to_string(dec_sink_caps));
            }
            self->idr_received = FALSE;

            gst_v4l2_video_dec_finish(decoder);
            gst_v4l2_object_stop(self->v4l2output);

            {
              const GValue *framerate = NULL;
              GstStructure *src_pad_st = NULL, *sink_pad_st = NULL;
              GstCaps *dec_src_caps = gst_pad_get_current_caps(decoder->srcpad);
              GstCaps *dec_sink_caps = gst_pad_get_current_caps(decoder->sinkpad);
              if (dec_src_caps != NULL)
              {
                gint curr_width = self->current_width;
                gint curr_height = self->current_height;
                dec_src_caps = gst_caps_make_writable(dec_src_caps);
                sink_pad_st = gst_caps_get_structure(dec_sink_caps, 0);
                src_pad_st = gst_caps_get_structure(dec_src_caps, 0);
                gst_structure_set(src_pad_st, "width", G_TYPE_INT, curr_width,
                                  "height", G_TYPE_INT, curr_height,
                                  NULL);
                framerate = gst_structure_get_value(sink_pad_st, "framerate");
                if (framerate)
                  gst_structure_set_value(src_pad_st, "framerate", framerate);
                /* Replace coded size with visible size, we want to negotiate visible size
                 * with downstream, not coded size. */
                gst_caps_map_in_place(dec_src_caps, gst_v4l2_video_remove_padding, self);
                gst_pad_set_caps(decoder->srcpad, dec_src_caps);

                GST_DEBUG_OBJECT(self, "dec_src_caps: %s", gst_caps_to_string(dec_src_caps));
                gst_caps_unref(dec_src_caps);
                gst_caps_unref(dec_sink_caps);
              }
            }

            {
                GstCaps *caps = gst_pad_get_current_caps(decoder->srcpad);
                if (caps)
                {
                  GstQuery *query = gst_query_new_allocation(caps, FALSE);
                  gst_pad_peer_query(decoder->srcpad, query);
                  gst_query_unref(query);
                  gst_caps_unref(caps);
                }
            }
            gst_v4l2_object_stop(self->v4l2capture);
            self->output_flow = GST_FLOW_OK;

            g_mutex_lock(&self->v4l2capture->cplane_stopped_lock);
            while (self->v4l2capture->capture_plane_stopped != TRUE)
            {
                g_cond_wait(&self->v4l2capture->cplane_stopped_cond,
                            &self->v4l2capture->cplane_stopped_lock);
            }
            self->v4l2capture->capture_plane_stopped = FALSE;
            g_mutex_unlock(&self->v4l2capture->cplane_stopped_lock);

            gst_v4l2_object_close(self->v4l2output);
            gst_v4l2_object_close(self->v4l2capture);
            if (!gst_v4l2_object_open(self->v4l2output))
                GST_ERROR_OBJECT(self, "gst_v4l2_object_open (self->v4l2output) failed\n");
            if (!gst_v4l2_object_open_shared(self->v4l2capture, self->v4l2output))
            {
                GST_ERROR_OBJECT(self, "gstv4l2object open shared failed\n");
                if (GST_V4L2_IS_OPEN(self->v4l2output))
                  gst_v4l2_object_close(self->v4l2output);
                return GST_FLOW_ERROR;
            }
            if (dec_sink_caps != NULL)
            {
                ret = gst_v4l2_object_set_format(self->v4l2output, dec_sink_caps, &error);
                gst_caps_unref(dec_sink_caps);
            }
            else
            {
                GST_ERROR_OBJECT(self, "Decoder sink caps == NULL");
            }

            if (ret)
                self->input_state = gst_video_codec_state_ref(self->input_state);
            else
                gst_v4l2_error(self, &error);

            set_v4l2_controls(self);
            GST_INFO_OBJECT(self, "Reset Done");

            trigger_drc = false;
            self->set_format = false;
        }
        else if (self->set_format == true)
        {
          GstStructure *sink_pad_st = NULL, *src_pad_st = NULL;
          const GValue *framerate = NULL;
          GstCaps *dec_sink_caps = gst_pad_get_current_caps(decoder->sinkpad);
          GstCaps *dec_src_caps = gst_pad_get_current_caps(decoder->srcpad);
          if (G_UNLIKELY(dec_sink_caps != NULL) && G_UNLIKELY(dec_src_caps != NULL))
          {
            GST_DEBUG_OBJECT(self, "dec_sink_caps: %s", gst_caps_to_string(dec_sink_caps));
            dec_src_caps = gst_caps_make_writable(dec_src_caps);
            sink_pad_st = gst_caps_get_structure(dec_sink_caps, 0);
            src_pad_st = gst_caps_get_structure(dec_src_caps, 0);
            framerate = gst_structure_get_value(sink_pad_st, "framerate");
            if (framerate)
              gst_structure_set_value(src_pad_st, "framerate", framerate);

            GST_DEBUG_OBJECT(self, "dec_src_caps: %s", gst_caps_to_string(dec_src_caps));

            gst_pad_set_caps(decoder->srcpad, dec_src_caps);

            gst_caps_unref(dec_sink_caps);
            gst_caps_unref(dec_src_caps);
          }
          set_v4l2_controls(self);
          self->set_format = false;
        }
    }

    /* CUVID and TEGRA decoders return format when SPS/PPS is received along with
     * a frame. In case of RTSP inputs we drop the DELTA units which are not
     * decodable independently until we receive I / IDR frame.
     */
    if (((GST_V4L2_PIXELFORMAT(obj) == V4L2_PIX_FMT_H264) ||
            (GST_V4L2_PIXELFORMAT(obj) == V4L2_PIX_FMT_H265) ||
            (GST_V4L2_PIXELFORMAT(obj) == V4L2_PIX_FMT_MPEG4)) &&
            self->is_gdr_stream == FALSE)
    {
        if ((GST_BUFFER_FLAG_IS_SET (GST_BUFFER_CAST(frame->input_buffer),
                        GST_BUFFER_FLAG_DELTA_UNIT)) &&
                (self->idr_received == FALSE))
        {
            GST_DEBUG_OBJECT (decoder, "Delta Unit Received, Dropping...");
            gst_video_decoder_drop_frame (decoder, frame);
            return GST_FLOW_OK;
        }
        self->idr_received = TRUE;
    }
#endif

  if (is_cuvid == TRUE) {
    if (self->skip_frames == V4L2_SKIP_FRAMES_TYPE_DECODE_IDR_ONLY) {
      // Decode only I Frames and drop others.
      if (GST_BUFFER_FLAG_IS_SET (GST_BUFFER_CAST(frame->input_buffer),
                                  GST_BUFFER_FLAG_DELTA_UNIT)) {
        gst_video_decoder_release_frame (GST_VIDEO_DECODER(self), frame);
        return GST_FLOW_OK;
      }
    }
  }

  if (enable_latency_measurement)
  {
      g_mutex_lock(&self->pts_hashtable_lock);
      gsize frame_pts = frame->pts;
      gsize current_time = get_current_system_timestamp ();
      g_hash_table_insert (self->hash_pts_systemtime, GSIZE_TO_POINTER(frame_pts),
                           GSIZE_TO_POINTER(current_time));
      g_mutex_unlock(&self->pts_hashtable_lock);
  }

  if (G_UNLIKELY (!g_atomic_int_get (&self->active)))
    goto flushing;

  if (G_UNLIKELY (!GST_V4L2_IS_ACTIVE (self->v4l2output))) {
    if (!self->input_state)
      goto not_negotiated;
    if (!gst_v4l2_object_set_format (self->v4l2output, self->input_state->caps,
            &error))
      goto not_negotiated;
  }

  if (G_UNLIKELY (!GST_V4L2_IS_ACTIVE (self->v4l2capture))) {
    GstBufferPool *pool = GST_BUFFER_POOL (self->v4l2output->pool);
    GstVideoInfo info = {0};
    GstVideoCodecState *output_state;
    GstBuffer *codec_data = NULL;
    GstCaps *acquired_caps, *available_caps, *caps, *filter;
    GstStructure *st;

    GST_DEBUG_OBJECT (self, "Sending header");

    codec_data = self->input_state->codec_data;

#ifdef USE_V4L2_TARGET_NV
    /* For av1 stream with webm container, we have observed that codec_data field is received
       along with the caps and is a part of input_state, thus sending this to cuvidparser as
       first frame. cuvidparser does not require it. We should send the input frame directly */
    if ((GST_V4L2_PIXELFORMAT(obj) == V4L2_PIX_FMT_AV1)) {
      codec_data = NULL;
    }
#endif

    /* We are running in byte-stream mode, so we don't know the headers, but
     * we need to send something, otherwise the decoder will refuse to
     * intialize.
     */
    if (codec_data) {
      GST_DEBUG_OBJECT (self, "codec_data field is valid. Using that as input for cuvidparser");
      gst_buffer_ref (codec_data);
    } else {
      codec_data = gst_buffer_ref (frame->input_buffer);
      processed = TRUE;
    }

    /* Ensure input internal pool is active */
    if (!gst_buffer_pool_is_active (pool)) {
      GstStructure *config = gst_buffer_pool_get_config (pool);
#ifndef USE_V4L2_TARGET_NV
      gst_buffer_pool_config_set_params (config, self->input_state->caps,
          self->v4l2output->info.size, 2, 2);
#else
      if (obj->mode != GST_V4L2_IO_USERPTR)
        obj->min_buffers = 2;

      if (V4L2_TYPE_IS_OUTPUT (obj->type)) {
        gst_buffer_pool_config_set_params (config, self->input_state->caps,
            self->v4l2output->info.size, obj->min_buffers, obj->min_buffers);
      }
#endif

      /* There is no reason to refuse this config */
      if (!gst_buffer_pool_set_config (pool, config))
        goto activate_failed;

      if (!gst_buffer_pool_set_active (pool, TRUE))
        goto activate_failed;
    }

    GST_VIDEO_DECODER_STREAM_UNLOCK (decoder);
    ret =
        gst_v4l2_buffer_pool_process (GST_V4L2_BUFFER_POOL (self->
            v4l2output->pool), &codec_data);
    GST_VIDEO_DECODER_STREAM_LOCK (decoder);

    gst_buffer_unref (codec_data);

    /* TODO: nvparser should return proper format of a stream with first
     * few bytes of stream header*/
#ifdef USE_V4L2_TARGET_NV
    if (!processed) {
      if (is_cuvid == TRUE)
        processed = TRUE;
      GST_VIDEO_DECODER_STREAM_UNLOCK (decoder);
      ret =
          gst_v4l2_buffer_pool_process (GST_V4L2_BUFFER_POOL (self->v4l2output->
              pool), &frame->input_buffer);
      GST_VIDEO_DECODER_STREAM_LOCK (decoder);

      if (ret == GST_FLOW_FLUSHING) {
        if (gst_pad_get_task_state (GST_VIDEO_DECODER_SRC_PAD (self)) !=
            GST_TASK_STARTED)
          ret = self->output_flow;
        goto drop;
      } else if (ret != GST_FLOW_OK) {
        GST_DEBUG_OBJECT (self, "gst_v4l2_buffer_pool_process failed\n");
        goto process_failed;
      }
    }

    if (V4L2_TYPE_IS_OUTPUT (obj->type)) {
      gint retval;
      gint dqevent_loop_count = 0;
      struct v4l2_event ev;
#ifndef USE_V4L2_TARGET_NV_X86
      /* This is WAR for Bug 3544450 */
      if(!processed)
      {
        processed = TRUE;
        /* Wait for DQEVENT for 1 sec */
        while (dqevent_loop_count < LOOP_COUNT_TO_WAIT_FOR_DQEVENT) {
          memset (&ev, 0, sizeof (ev));
          retval = obj->ioctl (obj->video_fd, VIDIOC_DQEVENT, &ev);
          dqevent_loop_count ++;
          if (retval != 0)
          {
            if (errno == EINVAL) {
                GST_DEBUG_OBJECT (self, "VIDIOC_DQEVENT failed. Event Type: %d\n", ev.type);
                goto process_failed;
	    }
            if (dqevent_loop_count == LOOP_COUNT_TO_WAIT_FOR_DQEVENT) {
                GST_DEBUG_OBJECT (self, "Stream format not found, dropping the frame\n");
                goto drop;
            }
            usleep (WAIT_TIME_PER_LOOP_FOR_DQEVENT); //TODO is this needed ?
            continue;
          }
          else
            break;
        }
      }
      else
#endif
      {
        while (1) {
          memset (&ev, 0, sizeof (ev));
          retval = obj->ioctl (obj->video_fd, VIDIOC_DQEVENT, &ev);
          if (retval != 0)
          {
            if (errno == EINVAL) {
                GST_DEBUG_OBJECT (self, "VIDIOC_DQEVENT failed. Event Type: %d\n", ev.type);
                goto process_failed;
            }
            if (is_cuvid != TRUE) {
              if (ev.sequence == 0) {
                GST_DEBUG_OBJECT (self, "Stream format not found, dropping the frame\n");
                goto drop;
              }
            }
            usleep(10*1000); //TODO is this needed ?
            continue;
          }
          else
            break;
        }
      }
    }
#endif

    /* For decoders G_FMT returns coded size, G_SELECTION returns visible size
     * in the compose rectangle. gst_v4l2_object_acquire_format() checks both
     * and returns the visible size as with/height and the coded size as
     * padding. */
    if (!gst_v4l2_object_acquire_format (self->v4l2capture, &info))
      goto not_negotiated;

    /* Create caps from the acquired format, remove the format field */
    acquired_caps = gst_video_info_to_caps (&info);
    GST_DEBUG_OBJECT (self, "Acquired caps: %" GST_PTR_FORMAT, acquired_caps);
    st = gst_caps_get_structure (acquired_caps, 0);
    gst_structure_remove_field (st, "format");

    /* Probe currently available pixel formats */
    available_caps = gst_v4l2_object_probe_caps (self->v4l2capture, NULL);
    available_caps = gst_caps_make_writable (available_caps);
    GST_DEBUG_OBJECT (self, "Available caps: %" GST_PTR_FORMAT, available_caps);

    /* Replace coded size with visible size, we want to negotiate visible size
     * with downstream, not coded size. */
    gst_caps_map_in_place (available_caps, gst_v4l2_video_remove_padding, self);

    filter = gst_caps_intersect_full (available_caps, acquired_caps,
        GST_CAPS_INTERSECT_FIRST);
    GST_DEBUG_OBJECT (self, "Filtered caps: %" GST_PTR_FORMAT, filter);
    gst_caps_unref (acquired_caps);
    gst_caps_unref (available_caps);
#ifndef USE_V4L2_TARGET_NV
    caps = gst_pad_peer_query_caps (decoder->srcpad, filter);
    gst_caps_unref (filter);
#else
    caps = gst_pad_peer_query_caps (decoder->srcpad,
        gst_pad_get_pad_template_caps (GST_VIDEO_DECODER_SRC_PAD (decoder)));
    gst_caps_unref (filter);

    if (gst_caps_is_empty (caps)) {
      GstPad *pad = GST_VIDEO_DECODER_SRC_PAD (decoder);
      caps = gst_pad_get_pad_template_caps (pad);
    }
#endif

    GST_DEBUG_OBJECT (self, "Possible decoded caps: %" GST_PTR_FORMAT, caps);
    if (gst_caps_is_empty (caps)) {
      gst_caps_unref (caps);
      goto not_negotiated;
    }

    /* Fixate pixel format */
    caps = gst_caps_fixate (caps);

    GST_DEBUG_OBJECT (self, "Chosen decoded caps: %" GST_PTR_FORMAT, caps);

    /* Try to set negotiated format, on success replace acquired format */
#ifndef USE_V4L2_TARGET_NV
    if (gst_v4l2_object_set_format (self->v4l2capture, caps, &error))
      gst_video_info_from_caps (&info, caps);
    else
      gst_v4l2_clear_error (&error);
#endif
    gst_caps_unref (caps);

    output_state = gst_video_decoder_set_output_state (decoder,
        info.finfo->format, info.width, info.height, self->input_state);

#ifdef USE_V4L2_TARGET_NV
    if (output_state->caps)
        gst_caps_unref (output_state->caps);
    /* set colorimetry information returned from low level lib
     * tegrav4l2. For now it is guarded using cuvid variable to
     * avoid any regression on x86.
     */
    if (!is_cuvid)
      output_state->info.colorimetry = info.colorimetry;
    output_state->caps = gst_video_info_to_caps (&output_state->info);
    GstCapsFeatures *features = gst_caps_features_new ("memory:NVMM", NULL);
    gst_caps_set_features (output_state->caps, 0, features);
    {
        // convert "cudadec_mem_type" to NvBufSurfaceMemType
        int buf_surface_mem_type = 0;
        switch(self->cudadec_mem_type)
        {
        case 0:
        {
            if(is_cuvid == TRUE)
                buf_surface_mem_type = NVBUF_MEM_CUDA_DEVICE;
            else
                buf_surface_mem_type = NVBUF_MEM_SURFACE_ARRAY;
            break;
        }
        case 1:
            buf_surface_mem_type = NVBUF_MEM_CUDA_PINNED;
            break;
        case 2:
            buf_surface_mem_type = NVBUF_MEM_CUDA_UNIFIED;
            break;
        case 3:
            buf_surface_mem_type = NVBUF_MEM_SURFACE_ARRAY;
            break;
        case 4:
            buf_surface_mem_type = NVBUF_MEM_CUDA_ARRAY;
            break;
        }
        gst_caps_set_simple(output_state->caps, "nvbuf-memory-type", G_TYPE_STRING , NvBufSurfaceMemType_names[buf_surface_mem_type], NULL);
        gst_caps_set_simple(output_state->caps, "gpu-id", G_TYPE_INT , self->cudadec_gpu_id, NULL);
    }
#endif
    /* Copy the rest of the information, there might be more in the future */
    output_state->info.interlace_mode = info.interlace_mode;
    gst_video_codec_state_unref (output_state);

    if (!gst_video_decoder_negotiate (decoder)) {
      if (GST_PAD_IS_FLUSHING (decoder->srcpad))
        goto flushing;
      else
        goto not_negotiated;
    }

    /* Ensure our internal pool is activated */
    if (!gst_buffer_pool_set_active (GST_BUFFER_POOL (self->v4l2capture->pool),
            TRUE))
      goto activate_failed;

#ifdef USE_V4L2_TARGET_NV
    if (self->v4l2capture->pool) {
      if (self->cap_buf_dynamic_allocation == CAP_BUF_DYNAMIC_ALLOC_ENABLED_FOR_FW_RW_PLAYBACK) {
        gst_v4l2_buffer_pool_enable_dynamic_allocation (GST_V4L2_BUFFER_POOL (self->v4l2capture->pool),
            TRUE);
      } else if (self->cap_buf_dynamic_allocation == CAP_BUF_DYNAMIC_ALLOC_ENABLED_FOR_RW_PLAYBACK && self->rate < 0) {
        gst_v4l2_buffer_pool_enable_dynamic_allocation (GST_V4L2_BUFFER_POOL (self->v4l2capture->pool),
            TRUE);
      } else if (self->cap_buf_dynamic_allocation == CAP_BUF_DYNAMIC_ALLOC_ENABLED_FOR_FW_PLAYBACK && self->rate > 0) {
        gst_v4l2_buffer_pool_enable_dynamic_allocation (GST_V4L2_BUFFER_POOL (self->v4l2capture->pool),
            TRUE);
      } else {
        gst_v4l2_buffer_pool_enable_dynamic_allocation (GST_V4L2_BUFFER_POOL (self->v4l2capture->pool),
            FALSE);
      }
    }
#endif
  }

  task_state = gst_pad_get_task_state (GST_VIDEO_DECODER_SRC_PAD (self));
  if (task_state == GST_TASK_STOPPED || task_state == GST_TASK_PAUSED) {
    /* It's possible that the processing thread stopped due to an error */
    if (self->output_flow != GST_FLOW_OK &&
        self->output_flow != GST_FLOW_FLUSHING) {
      GST_DEBUG_OBJECT (self, "Processing loop stopped with error, leaving");
      ret = self->output_flow;
      goto drop;
    }

#ifdef USE_V4L2_TARGET_NV
    set_v4l2_video_mpeg_class (self->v4l2capture,
              V4L2_CID_MPEG_SET_POLL_INTERRUPT, 1);
#endif

    GST_DEBUG_OBJECT (self, "Starting decoding thread");

    /* Start the processing task, when it quits, the task will disable input
     * processing to unlock input if draining, or prevent potential block */
    self->output_flow = GST_FLOW_FLUSHING;
    if (!gst_pad_start_task (decoder->srcpad,
            (GstTaskFunction) gst_v4l2_video_dec_loop, self, NULL))
      goto start_task_failed;
  }


  if ((self->extract_sei_type5_data == TRUE) && self->input_state && self->input_state->caps &&
      (gst_structure_has_name (gst_caps_get_structure (self->input_state->caps, 0), "video/x-h264") ||
       gst_structure_has_name (gst_caps_get_structure (self->input_state->caps, 0), "video/x-h265") ))
    {
      uint8_t *sei_type5_payload = NULL;
      GstMapInfo map = {0};
      frame->input_buffer = gst_buffer_make_writable (frame->input_buffer);
      if (!gst_buffer_map (frame->input_buffer, &map, GST_MAP_READ))
      {
          GST_DEBUG_OBJECT (self, "couldnt map\n");
          goto process_failed;
      }
      else
      {
          uint32_t payload_size = 0;
          uint8_t *stream_data = (uint8_t *)map.data;
          sei_type5_payload = parse_sei_data (stream_data, map.size,
                  &payload_size, self->sei_uuid_string, obj->fmtdesc->pixelformat);
          if (sei_type5_payload != NULL)
          {
              GST_DEBUG_OBJECT (self, "sei_type5_payload found\n");
              GstVideoSEIMeta *video_sei_meta =
                  (GstVideoSEIMeta *)gst_buffer_add_meta(
                  frame->input_buffer, GST_VIDEO_SEI_META_INFO, NULL);
              video_sei_meta->sei_metadata_type =
                  GST_USER_SEI_META;
              video_sei_meta->sei_metadata_size = payload_size;
              video_sei_meta->sei_metadata_ptr = sei_type5_payload;

              if (self->attach_sei_timestamps || getenv("UPDATE_SEI_TIMESTAMPS")) {
                self->ts = convert_string_to_ts (self, (const char *)sei_type5_payload);
                GST_BUFFER_PTS (frame->input_buffer) = self->ts;
                frame->pts = self->ts;
              }
          }
          gst_buffer_unmap (frame->input_buffer, &map);
      }
  }

  if (!processed) {
    GST_VIDEO_DECODER_STREAM_UNLOCK (decoder);
    ret =
        gst_v4l2_buffer_pool_process (GST_V4L2_BUFFER_POOL (self->v4l2output->
            pool), &frame->input_buffer);
    GST_VIDEO_DECODER_STREAM_LOCK (decoder);

    if (ret == GST_FLOW_FLUSHING) {
      if (gst_pad_get_task_state (GST_VIDEO_DECODER_SRC_PAD (self)) !=
          GST_TASK_STARTED)
        ret = self->output_flow;
      goto drop;
    } else if (ret != GST_FLOW_OK) {
        if (gst_pad_get_task_state (GST_VIDEO_DECODER_SINK_PAD(self)) ==
                GST_TASK_STOPPED)
        {
            ret = GST_FLOW_OK;
            goto drop;
        }
      GST_DEBUG_OBJECT (self, "gst_v4l2_buffer_pool_process failed\n");
      goto process_failed;
    }
  }


  /* No need to keep input around */
  tmp = frame->input_buffer;
  frame->input_buffer = gst_buffer_new ();
  gst_buffer_copy_into (frame->input_buffer, tmp,
      GST_BUFFER_COPY_FLAGS | GST_BUFFER_COPY_TIMESTAMPS |
      GST_BUFFER_COPY_META, 0, 0);
  gst_buffer_unref (tmp);

  gst_video_codec_frame_unref (frame);
  return ret;

  /* ERRORS */
not_negotiated:
  {
    GST_ERROR_OBJECT (self, "not negotiated");
    ret = GST_FLOW_NOT_NEGOTIATED;
    gst_v4l2_error (self, &error);
    goto drop;
  }
activate_failed:
  {
    GST_ELEMENT_ERROR (self, RESOURCE, SETTINGS,
        (_("Failed to allocate required memory.")),
        ("Buffer pool activation failed"));
    ret = GST_FLOW_ERROR;
    goto drop;
  }
flushing:
  {
    ret = GST_FLOW_FLUSHING;
    goto drop;
  }

start_task_failed:
  {
    GST_ELEMENT_ERROR (self, RESOURCE, FAILED,
        (_("Failed to start decoding thread.")), (NULL));
    ret = GST_FLOW_ERROR;
    goto drop;
  }
process_failed:
  {
    GST_ELEMENT_ERROR (self, RESOURCE, FAILED,
        (_("Failed to process frame.")), (NULL));
    ret = GST_FLOW_ERROR;
    goto drop;
  }
drop:
  {
    gst_video_decoder_drop_frame (decoder, frame);
    return ret;
  }
}

static gboolean
gst_v4l2_video_dec_decide_allocation (GstVideoDecoder * decoder,
    GstQuery * query)
{
  GstV4l2VideoDec *self = GST_V4L2_VIDEO_DEC (decoder);
  GstClockTime latency;
  gboolean ret = FALSE;

  if (gst_v4l2_object_decide_allocation (self->v4l2capture, query))
    ret = GST_VIDEO_DECODER_CLASS (parent_class)->decide_allocation (decoder,
        query);

  if (GST_CLOCK_TIME_IS_VALID (self->v4l2capture->duration)) {
    latency = self->v4l2capture->min_buffers * self->v4l2capture->duration;
    GST_DEBUG_OBJECT (self, "Setting latency: %" GST_TIME_FORMAT " (%"
        G_GUINT32_FORMAT " * %" G_GUINT64_FORMAT, GST_TIME_ARGS (latency),
        self->v4l2capture->min_buffers, self->v4l2capture->duration);
    gst_video_decoder_set_latency (decoder, latency, latency);
  } else {
    GST_WARNING_OBJECT (self, "Duration invalid, not setting latency");
  }

  return ret;
}

static gboolean
gst_v4l2_video_dec_src_query (GstVideoDecoder * decoder, GstQuery * query)
{
  gboolean ret = TRUE;
  GstV4l2VideoDec *self = GST_V4L2_VIDEO_DEC (decoder);

  switch (GST_QUERY_TYPE (query)) {
    case GST_QUERY_CAPS:{
      GstCaps *filter = NULL, *result = NULL;
      GstPad *pad = GST_VIDEO_DECODER_SRC_PAD (decoder);

      gst_query_parse_caps (query, &filter);

      if (self->probed_srccaps)
        result = gst_caps_ref (self->probed_srccaps);
      else
        result = gst_pad_get_pad_template_caps (pad);

      if (filter) {
        GstCaps *tmp = result;
        result =
            gst_caps_intersect_full (filter, tmp, GST_CAPS_INTERSECT_FIRST);
        gst_caps_unref (tmp);
      }

      GST_DEBUG_OBJECT (self, "Returning src caps %" GST_PTR_FORMAT, result);

      gst_query_set_caps_result (query, result);
      gst_caps_unref (result);
      break;
    }

    default:
      ret = GST_VIDEO_DECODER_CLASS (parent_class)->src_query (decoder, query);
      break;
  }

  return ret;
}

static GstCaps *
gst_v4l2_video_dec_sink_getcaps (GstVideoDecoder * decoder, GstCaps * filter)
{
  GstV4l2VideoDec *self = GST_V4L2_VIDEO_DEC (decoder);
  GstCaps *result;

  result = gst_video_decoder_proxy_getcaps (decoder, self->probed_sinkcaps,
      filter);

  GST_DEBUG_OBJECT (self, "Returning sink caps %" GST_PTR_FORMAT, result);

  return result;
}

static gboolean
gst_v4l2_video_dec_sink_event (GstVideoDecoder * decoder, GstEvent * event)
{
  GstV4l2VideoDec *self = GST_V4L2_VIDEO_DEC (decoder);
  gboolean ret;

  switch (GST_EVENT_TYPE (event)) {
    case GST_EVENT_FLUSH_START:
      break;
#ifdef USE_V4L2_TARGET_NV
    case GST_EVENT_GAP:
      GST_DEBUG_OBJECT (self, "GAP event received");
      if (GST_V4L2_IS_ACTIVE (self->v4l2capture))
      {
          GST_DEBUG_OBJECT (self, "CAPTURE PLANE buffer pool is active"
                 " so passing on the event to downstream");
          break;
      }
      else
      {
          GST_DEBUG_OBJECT (self, "CAPTURE PLANE buffer pool is inactive"
                  " so ignore the gap event");
          gst_event_unref(event);
          return TRUE;
      }
#endif
    default:
      break;
  }

  ret = GST_VIDEO_DECODER_CLASS (parent_class)->sink_event (decoder, event);

  switch (GST_EVENT_TYPE (event)) {
    case GST_EVENT_FLUSH_START:
      break;
    default:
      break;
  }

#ifdef USE_V4L2_TARGET_NV
  if ((GstNvCustomEventType)GST_EVENT_TYPE (event) == GST_NVEVENT_DEC_DROP_FRAME_INTERVAL_UPDATE) {
    gchar* stream_id = NULL;
    gst_nvevent_parse_dec_drop_frame_interval_update (event, &stream_id, &self->drop_frame_interval);
  }

  if ((GstNvCustomEventType)GST_EVENT_TYPE (event) == GST_NVEVENT_DEC_SKIP_FRAME_UPDATE) {
    gchar* stream_id = NULL;
    gst_nvevent_parse_dec_skip_frame_update (event, &stream_id, &self->skip_frames);
    /* Handle skip-frame event */
    if (self->skip_frames != V4L2_SKIP_FRAMES_TYPE_NONE) {
      if (!set_v4l2_video_mpeg_class (self->v4l2output,
          V4L2_CID_MPEG_VIDEO_SKIP_FRAMES,
          self->skip_frames)) {
        g_print ("S_EXT_CTRLS for SKIP_FRAMES failed\n");
        return FALSE;
      }
    }
  }

  if ((GstNvCustomEventType)GST_EVENT_TYPE (event) == GST_NVEVENT_DEC_ENABLE_LOW_LATENCY_MODE) {
    gchar* stream_id = NULL;
    gst_nvevent_parse_dec_enable_low_latency_mode (event, &stream_id, &self->cudadec_low_latency);
    /* TODO: Handle enable of low_latency mode */
  }
#endif

  return ret;
}

#ifdef USE_V4L2_TARGET_NV
static gboolean
gst_v4l2_video_dec_src_event (GstVideoDecoder * decoder, GstEvent * event)
{
  GstV4l2VideoDec *self = GST_V4L2_VIDEO_DEC (decoder);
  gboolean ret;

  switch (GST_EVENT_TYPE (event)) {
    case GST_EVENT_SEEK:
    {
      GstFormat format;
      gdouble rate = 0.0;
      GstSeekFlags flags;
      GstSeekType start_type, stop_type;
      gint64 start, stop;

      gst_event_parse_seek (event, &rate, &format, &flags, &start_type, &start,
          &stop_type, &stop);

      self->rate = rate;
      GST_DEBUG_OBJECT (self, "Seek event received with rate %f", rate);

      if (self->v4l2capture->pool) {
        if (self->cap_buf_dynamic_allocation == CAP_BUF_DYNAMIC_ALLOC_ENABLED_FOR_FW_RW_PLAYBACK) {
          gst_v4l2_buffer_pool_enable_dynamic_allocation (GST_V4L2_BUFFER_POOL (self->v4l2capture->pool),
              TRUE);
        } else if (self->cap_buf_dynamic_allocation == CAP_BUF_DYNAMIC_ALLOC_ENABLED_FOR_RW_PLAYBACK && self->rate < 0) {
          gst_v4l2_buffer_pool_enable_dynamic_allocation (GST_V4L2_BUFFER_POOL (self->v4l2capture->pool),
              TRUE);
        } else if (self->cap_buf_dynamic_allocation == CAP_BUF_DYNAMIC_ALLOC_ENABLED_FOR_FW_PLAYBACK && self->rate > 0) {
          gst_v4l2_buffer_pool_enable_dynamic_allocation (GST_V4L2_BUFFER_POOL (self->v4l2capture->pool),
              TRUE);
        } else {
          gst_v4l2_buffer_pool_enable_dynamic_allocation (GST_V4L2_BUFFER_POOL (self->v4l2capture->pool),
              FALSE);
        }
      }
      break;
    }
    default:
      break;
  }

  ret = GST_VIDEO_DECODER_CLASS (parent_class)->src_event (decoder, event);

  return ret;
}
#endif

static GstStateChangeReturn
gst_v4l2_video_dec_change_state (GstElement * element,
    GstStateChange transition)
{
  GstV4l2VideoDec *self = GST_V4L2_VIDEO_DEC (element);
  GstVideoDecoder *decoder = GST_VIDEO_DECODER (element);

  if (transition == GST_STATE_CHANGE_PAUSED_TO_READY) {
    g_atomic_int_set (&self->active, FALSE);
    gst_v4l2_object_unlock (self->v4l2output);
    gst_v4l2_object_unlock (self->v4l2capture);
#ifdef USE_V4L2_TARGET_NV
    set_v4l2_video_mpeg_class (self->v4l2capture,
              V4L2_CID_MPEG_SET_POLL_INTERRUPT, 0);
#endif
    gst_pad_stop_task (decoder->srcpad);
  }

  return GST_ELEMENT_CLASS (parent_class)->change_state (element, transition);
}

static void
gst_v4l2_video_dec_dispose (GObject * object)
{
  GstV4l2VideoDec *self = GST_V4L2_VIDEO_DEC (object);

  gst_caps_replace (&self->probed_sinkcaps, NULL);
  gst_caps_replace (&self->probed_srccaps, NULL);

  G_OBJECT_CLASS (parent_class)->dispose (object);
}

static void
gst_v4l2_video_dec_finalize (GObject * object)
{
  GstV4l2VideoDec *self = GST_V4L2_VIDEO_DEC (object);

#ifdef USE_V4L2_TARGET_NV
  g_cond_clear (&self->v4l2capture->cplane_stopped_cond);
  g_mutex_clear (&self->v4l2capture->cplane_stopped_lock);
  g_mutex_clear (&self->pts_hashtable_lock);
#endif

  gst_v4l2_object_destroy (self->v4l2capture);
  gst_v4l2_object_destroy (self->v4l2output);

  G_OBJECT_CLASS (parent_class)->finalize (object);
}

static void
gst_v4l2_video_dec_init (GstV4l2VideoDec * self)
{
  /* V4L2 object are created in subinstance_init */
#ifdef USE_V4L2_TARGET_NV
  if (is_cuvid == TRUE) {
    default_num_extra_surfaces = 0; //default for dGPU
    default_cudadec_low_latency = FALSE;
    default_sei_extract_data = FALSE;
  } else if (is_cuvid == FALSE) {
    default_num_extra_surfaces = 1; //default for Tegra
  }
  self->skip_frames = DEFAULT_SKIP_FRAME_TYPE;
  self->drop_frame_interval = 0;
  self->decoded_picture_cnt = 0;
  self->num_extra_surfaces = default_num_extra_surfaces;
  self->valid_vpx = FALSE;

  self->disable_dpb = DEFAULT_DISABLE_DPB;
  self->enable_full_frame = DEFAULT_FULL_FRAME;
  self->enable_frame_type_reporting = DEFAULT_FRAME_TYPR_REPORTING;
  self->enable_error_check = DEFAULT_ERROR_CHECK;
  self->enable_max_performance = DEFAULT_MAX_PERFORMANCE;
  self->cudadec_mem_type = DEFAULT_CUDADEC_MEM_TYPE;
  self->cudadec_gpu_id = DEFAULT_CUDADEC_GPU_ID;
  self->cudadec_low_latency = default_cudadec_low_latency;
  self->idr_received = FALSE;
  self->rate = 1;
  self->old_width = 0;
  self->old_height = 0;
  self->current_dpb_size = 0;
  self->old_dpb_size = 0;
  self->cap_buf_dynamic_allocation = DEFAULT_CAP_BUF_DYNAMIC_ALLOCATION;
#endif

  const gchar * latency = g_getenv("NVDS_ENABLE_LATENCY_MEASUREMENT");
  const gchar * silent_mode = g_getenv("NVDS_LATENCY_MEASUREMENT_SILENT");
  const gchar * latency_print = g_getenv("NVDS_ENABLE_LATENCY_PRINT");
  if(latency || silent_mode || latency_print)
  {
    enable_latency_measurement = TRUE;
  }
  if(latency_print)
  {
    enable_latency_print = TRUE;
  }
}

static void
gst_v4l2_video_dec_subinstance_init (GTypeInstance * instance, gpointer g_class)
{
  GstV4l2VideoDecClass *klass = GST_V4L2_VIDEO_DEC_CLASS (g_class);
  GstV4l2VideoDec *self = GST_V4L2_VIDEO_DEC (instance);
  GstVideoDecoder *decoder = GST_VIDEO_DECODER (instance);

  gst_video_decoder_set_packetized (decoder, TRUE);

  self->v4l2output = gst_v4l2_object_new (GST_ELEMENT (self),
      GST_OBJECT (GST_VIDEO_DECODER_SINK_PAD (self)),
      V4L2_BUF_TYPE_VIDEO_OUTPUT, klass->default_device,
      gst_v4l2_get_output, gst_v4l2_set_output, NULL);
  self->v4l2output->no_initial_format = TRUE;
  self->v4l2output->keep_aspect = FALSE;
#ifdef USE_V4L2_TARGET_NV
  self->v4l2output->open_mjpeg_block = TRUE;
#endif

  self->v4l2capture = gst_v4l2_object_new (GST_ELEMENT (self),
      GST_OBJECT (GST_VIDEO_DECODER_SRC_PAD (self)),
      V4L2_BUF_TYPE_VIDEO_CAPTURE, klass->default_device,
      gst_v4l2_get_input, gst_v4l2_set_input, NULL);
  self->v4l2capture->no_initial_format = TRUE;
  self->v4l2output->keep_aspect = FALSE;
#ifdef USE_V4L2_TARGET_NV
  g_mutex_init (&self->v4l2capture->cplane_stopped_lock);
  g_cond_init (&self->v4l2capture->cplane_stopped_cond);
  g_mutex_init (&self->pts_hashtable_lock);
#endif
}

static void
gst_v4l2_video_dec_class_init (GstV4l2VideoDecClass * klass)
{
  GstElementClass *element_class;
  GObjectClass *gobject_class;
  GstVideoDecoderClass *video_decoder_class;

  parent_class = g_type_class_peek_parent (klass);

  element_class = (GstElementClass *) klass;
  gobject_class = (GObjectClass *) klass;
  video_decoder_class = (GstVideoDecoderClass *) klass;

#ifndef USE_V4L2_TARGET_NV
  GST_DEBUG_CATEGORY_INIT (gst_v4l2_video_dec_debug, "v4l2videodec", 0,
      "V4L2 Video Decoder");
#else
  GST_DEBUG_CATEGORY_INIT (gst_v4l2_video_dec_debug, "v4l2videodec", 0,
      "NVIDIA V4L2 Video Decoder");
#endif

  gobject_class->dispose = GST_DEBUG_FUNCPTR (gst_v4l2_video_dec_dispose);
  gobject_class->finalize = GST_DEBUG_FUNCPTR (gst_v4l2_video_dec_finalize);

#ifdef USE_V4L2_TARGET_NV
  if (is_cuvid == FALSE) {
    gobject_class->set_property =
        GST_DEBUG_FUNCPTR (gst_v4l2_video_dec_set_property_tegra);
    gobject_class->get_property =
        GST_DEBUG_FUNCPTR (gst_v4l2_video_dec_get_property_tegra);
  } else if (is_cuvid == TRUE) {
    gobject_class->set_property =
        GST_DEBUG_FUNCPTR (gst_v4l2_video_dec_set_property_cuvid);
    gobject_class->get_property =
        GST_DEBUG_FUNCPTR (gst_v4l2_video_dec_get_property_cuvid);
  }

  g_object_class_install_property (gobject_class, PROP_SKIP_FRAME,
      g_param_spec_enum ("skip-frames",
          "Skip frames",
          "Type of frames to skip during decoding",
          GST_TYPE_V4L2_VID_DEC_SKIP_FRAMES,
          DEFAULT_SKIP_FRAME_TYPE,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
          GST_PARAM_MUTABLE_PLAYING));

  g_object_class_install_property (gobject_class, PROP_DROP_FRAME_INTERVAL,
      g_param_spec_uint ("drop-frame-interval",
          "Drop frames interval",
          "Interval to drop the frames,ex: value of 5 means every 5th frame will be given by decoder, rest all dropped",
          0,
          30, 30,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY));

  g_object_class_install_property (gobject_class, PROP_NUM_EXTRA_SURFACES,
      g_param_spec_uint ("num-extra-surfaces",
          "Number of extra surfaces",
          "Additional number of surfaces in addition to min decode surfaces given by the v4l2 driver",
          0,
          55, 55,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY));

  if (is_cuvid == FALSE) {
    g_object_class_install_property (gobject_class, PROP_DISABLE_DPB,
        g_param_spec_boolean ("disable-dpb",
            "Disable DPB buffer",
            "Set to disable DPB buffer for low latency",
            DEFAULT_DISABLE_DPB, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

    g_object_class_install_property (gobject_class, PROP_USE_FULL_FRAME,
        g_param_spec_boolean ("enable-full-frame",
            "Full Frame",
            "Whether or not the data is full framed",
            DEFAULT_FULL_FRAME, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

    g_object_class_install_property (gobject_class,
        PROP_ENABLE_FRAME_TYPE_REPORTING,
        g_param_spec_boolean ("enable-frame-type-reporting",
            "enable-frame-type-reporting", "Set to enable frame type reporting",
            DEFAULT_FRAME_TYPR_REPORTING,
            G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

    g_object_class_install_property (gobject_class, PROP_OPEN_MJPEG_BLOCK,
        g_param_spec_boolean ("mjpeg",
            "Open MJPEG Block",
            "Set to open MJPEG block",
            FALSE, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

    g_object_class_install_property (gobject_class, PROP_ENABLE_ERROR_CHECK,
        g_param_spec_boolean ("enable-error-check",
            "enable-error-check",
            "Set to enable error check",
            DEFAULT_ERROR_CHECK, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

    g_object_class_install_property (gobject_class, PROP_ENABLE_MAX_PERFORMANCE,
        g_param_spec_boolean ("enable-max-performance",
            "Enable max performance", "Set to enable max performance",
            DEFAULT_MAX_PERFORMANCE, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

    g_object_class_install_property (gobject_class, PROP_IS_GDR_STREAM,
        g_param_spec_boolean ("is-gdr-stream",
            "is-gdr-stream",
            "Set the flag to allow GDR decode for H264/HEVC",
            FALSE, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

    g_object_class_install_property (gobject_class, PROP_CAP_BUF_DYNAMIC_ALLOCATION,
      g_param_spec_enum ("capture-buffer-dynamic-allocation",
          "Enable capture buffer dynamic allocation",
          "Set to enable capture buffer dynamic allocation",
          GST_TYPE_V4L2_DEC_CAP_BUF_DYNAMIC_ALLOC,
          DEFAULT_CAP_BUF_DYNAMIC_ALLOCATION,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

    g_object_class_install_property (gobject_class, PROP_EXTRACT_SEI_TYPE5_DATA,
        g_param_spec_boolean ("extract-sei-type5-data",
            "extract-sei-type5-data",
            "Set to extract and attach SEI type5 unregistered data on output buffer",
            default_sei_extract_data,
            G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

    g_object_class_install_property (gobject_class, PROP_ATTACH_SEI_TIMESTAMPS,
        g_param_spec_boolean ("attach-sei-timestamps",
            "attach-sei-timestamps",
            "Set to overwrite buffer PTS with SEI extracted timestamp",
            FALSE,
            G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

    g_object_class_install_property (gobject_class, PROP_SEI_UUID_STRING,
        g_param_spec_string ("sei-uuid", "SEI UUID String",
            "Set 16 bytes UUID string for SEI Parsing, extract-sei-type5-data should be TRUE",
            NULL,
            (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  } else if (is_cuvid == TRUE) {
    g_object_class_install_property (gobject_class, PROP_CUDADEC_MEM_TYPE,
        g_param_spec_enum ("cudadec-memtype",
            "Memory type for cuda decoder buffers",
            "Set to specify memory type for cuda decoder buffers",
            GST_TYPE_V4L2_VID_CUDADEC_MEM_TYPE, DEFAULT_CUDADEC_MEM_TYPE,
            G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY));

    g_object_class_install_property (gobject_class, PROP_CUDADEC_GPU_ID,
        g_param_spec_uint ("gpu-id",
            "GPU Device ID",
            "Set to GPU Device ID for decoder ",
            0,
            G_MAXUINT, DEFAULT_CUDADEC_GPU_ID,
            G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY));

    g_object_class_install_property (gobject_class, PROP_CUDADEC_LOW_LATENCY,
        g_param_spec_boolean ("low-latency-mode",
            "CUVID Decode Low Latency Mode",
            "Set low latency mode for bitstreams having I and IPPP frames",
            default_cudadec_low_latency,
            G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

    g_object_class_install_property (gobject_class, PROP_DISABLE_DPB,
        g_param_spec_boolean ("disable-dpb",
            "Disable DPB buffer",
            "Set to disable DPB buffer for low latency",
            DEFAULT_DISABLE_DPB, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

    g_object_class_install_property (gobject_class, PROP_EXTRACT_SEI_TYPE5_DATA,
        g_param_spec_boolean ("extract-sei-type5-data",
            "extract-sei-type5-data",
            "Set to extract and attach SEI type5 unregistered data on output buffer",
            default_sei_extract_data,
            G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

    g_object_class_install_property (gobject_class, PROP_ATTACH_SEI_TIMESTAMPS,
        g_param_spec_boolean ("attach-sei-timestamps",
            "attach-sei-timestamps",
            "Set to overwrite buffer PTS with SEI extracted timestamp",
            FALSE,
            G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

    g_object_class_install_property (gobject_class, PROP_SEI_UUID_STRING,
        g_param_spec_string ("sei-uuid", "SEI UUID String",
            "Set 16 bytes UUID string for SEI Parsing, extract-sei-type5-data should be TRUE",
            NULL,
            (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  }
#endif

  video_decoder_class->open = GST_DEBUG_FUNCPTR (gst_v4l2_video_dec_open);
  video_decoder_class->close = GST_DEBUG_FUNCPTR (gst_v4l2_video_dec_close);
  video_decoder_class->start = GST_DEBUG_FUNCPTR (gst_v4l2_video_dec_start);
  video_decoder_class->stop = GST_DEBUG_FUNCPTR (gst_v4l2_video_dec_stop);
  video_decoder_class->finish = GST_DEBUG_FUNCPTR (gst_v4l2_video_dec_finish);
  video_decoder_class->flush = GST_DEBUG_FUNCPTR (gst_v4l2_video_dec_flush);
  video_decoder_class->drain = GST_DEBUG_FUNCPTR (gst_v4l2_video_dec_drain);
  video_decoder_class->set_format =
      GST_DEBUG_FUNCPTR (gst_v4l2_video_dec_set_format);
  video_decoder_class->negotiate =
      GST_DEBUG_FUNCPTR (gst_v4l2_video_dec_negotiate);
  video_decoder_class->decide_allocation =
      GST_DEBUG_FUNCPTR (gst_v4l2_video_dec_decide_allocation);
  /* FIXME propose_allocation or not ? */
  video_decoder_class->handle_frame =
      GST_DEBUG_FUNCPTR (gst_v4l2_video_dec_handle_frame);
  video_decoder_class->getcaps =
      GST_DEBUG_FUNCPTR (gst_v4l2_video_dec_sink_getcaps);
  video_decoder_class->src_query =
      GST_DEBUG_FUNCPTR (gst_v4l2_video_dec_src_query);
  video_decoder_class->sink_event =
      GST_DEBUG_FUNCPTR (gst_v4l2_video_dec_sink_event);
#ifdef USE_V4L2_TARGET_NV
  video_decoder_class->src_event =
      GST_DEBUG_FUNCPTR (gst_v4l2_video_dec_src_event);
#endif

  element_class->change_state =
      GST_DEBUG_FUNCPTR (gst_v4l2_video_dec_change_state);

  gst_v4l2_object_install_m2m_properties_helper (gobject_class);
#ifdef USE_V4L2_TARGET_NV
  if (is_cuvid == FALSE)
    gst_v4l2_object_install_m2m_dec_iomode_properties_helper (gobject_class);
#endif
}

static void
gst_v4l2_video_dec_subclass_init (gpointer g_class, gpointer data)
{
  GstV4l2VideoDecClass *klass = GST_V4L2_VIDEO_DEC_CLASS (g_class);
  GstElementClass *element_class = GST_ELEMENT_CLASS (g_class);
  GstV4l2VideoDecCData *cdata = data;

  klass->default_device = cdata->device;

#ifndef USE_V4L2_TARGET_NV
  /* Note: gst_pad_template_new() take the floating ref from the caps */
  gst_element_class_add_pad_template (element_class,
      gst_pad_template_new ("sink", GST_PAD_SINK, GST_PAD_ALWAYS,
          cdata->sink_caps));
  gst_element_class_add_pad_template (element_class,
      gst_pad_template_new ("src", GST_PAD_SRC, GST_PAD_ALWAYS,
          cdata->src_caps));

  gst_element_class_set_static_metadata (element_class, cdata->longname,
      "Codec/Decoder/Video", cdata->description,
      "Nicolas Dufresne <nicolas.dufresne@collabora.com>");

  gst_caps_unref (cdata->sink_caps);
  gst_caps_unref (cdata->src_caps);
#else
  gst_element_class_add_pad_template (element_class,
      gst_static_pad_template_get (&gst_v4l2dec_sink_template));
  gst_element_class_add_pad_template (element_class,
      gst_static_pad_template_get (&gst_v4l2dec_src_template));

  gst_element_class_set_static_metadata (element_class,
      "NVIDIA v4l2 video decoder" /*cdata->longname */ ,
      "Codec/Decoder/Video",
      "Decode video streams via V4L2 API" /*cdata->description */ ,
      "Nicolas Dufresne <nicolas.dufresne@collabora.com>, Viranjan Pagar <vpagar@nvidia.com>");
#endif

  g_free (cdata);
}

/* Probing functions */
gboolean
gst_v4l2_is_video_dec (GstCaps * sink_caps, GstCaps * src_caps)
{
  gboolean ret = FALSE;

  if (gst_caps_is_subset (sink_caps, gst_v4l2_object_get_codec_caps ())
      && gst_caps_is_subset (src_caps, gst_v4l2_object_get_raw_caps ()))
    ret = TRUE;

  return ret;
}

#ifndef USE_V4L2_TARGET_NV
static gchar *
gst_v4l2_video_dec_set_metadata (GstStructure * s, GstV4l2VideoDecCData * cdata,
    const gchar * basename)
{
  gchar *codec_name = NULL;
  gchar *type_name = NULL;

#define SET_META(codec) \
G_STMT_START { \
  cdata->longname = "V4L2 " codec " Decoder"; \
  cdata->description = "Decodes " codec " streams via V4L2 API"; \
  codec_name = g_ascii_strdown (codec, -1); \
} G_STMT_END

  if (gst_structure_has_name (s, "image/jpeg")) {
    SET_META ("JPEG");
  } else if (gst_structure_has_name (s, "video/mpeg")) {
    gint mpegversion = 0;
    gst_structure_get_int (s, "mpegversion", &mpegversion);

    if (mpegversion == 2) {
      SET_META ("MPEG2");
    } else {
      SET_META ("MPEG4");
    }
  } else if (gst_structure_has_name (s, "video/x-h263")) {
    SET_META ("H263");
  } else if (gst_structure_has_name (s, "video/x-h264")) {
    SET_META ("H264");
  } else if (gst_structure_has_name (s, "video/x-wmv")) {
    SET_META ("VC1");
  } else if (gst_structure_has_name (s, "video/x-vp8")) {
    SET_META ("VP8");
  } else if (gst_structure_has_name (s, "video/x-vp9")) {
    SET_META ("VP9");
  } else if (gst_structure_has_name (s, "video/x-bayer")) {
    SET_META ("BAYER");
  } else if (gst_structure_has_name (s, "video/x-sonix")) {
    SET_META ("SONIX");
  } else if (gst_structure_has_name (s, "video/x-pwc1")) {
    SET_META ("PWC1");
  } else if (gst_structure_has_name (s, "video/x-pwc2")) {
    SET_META ("PWC2");
  } else {
    /* This code should be kept on sync with the exposed CODEC type of format
     * from gstv4l2object.c. This warning will only occure in case we forget
     * to also add a format here. */
    gchar *s_str = gst_structure_to_string (s);
    g_warning ("Missing fixed name mapping for caps '%s', this is a GStreamer "
        "bug, please report at https://bugs.gnome.org", s_str);
    g_free (s_str);
  }

  if (codec_name) {
    type_name = g_strdup_printf ("v4l2%sdec", codec_name);
    if (g_type_from_name (type_name) != 0) {
      g_free (type_name);
      type_name = g_strdup_printf ("v4l2%s%sdec", basename, codec_name);
    }

    g_free (codec_name);
  }

  return type_name;
#undef SET_META
}

void
gst_v4l2_video_dec_register (GstPlugin * plugin, const gchar * basename,
    const gchar * device_path, GstCaps * sink_caps, GstCaps * src_caps)
{
  gint i;

  for (i = 0; i < gst_caps_get_size (sink_caps); i++) {
    GstV4l2VideoDecCData *cdata;
    GstStructure *s;
    GTypeQuery type_query;
    GTypeInfo type_info = { 0, };
    GType type, subtype;
    gchar *type_name;

    s = gst_caps_get_structure (sink_caps, i);

    cdata = g_new0 (GstV4l2VideoDecCData, 1);
    cdata->device = g_strdup (device_path);
    cdata->sink_caps = gst_caps_new_empty ();
    gst_caps_append_structure (cdata->sink_caps, gst_structure_copy (s));
    cdata->src_caps = gst_caps_ref (src_caps);
    type_name = gst_v4l2_video_dec_set_metadata (s, cdata, basename);

    /* Skip over if we hit an unmapped type */
    if (!type_name) {
      g_free (cdata);
      continue;
    }

    type = gst_v4l2_video_dec_get_type ();
    g_type_query (type, &type_query);
    memset (&type_info, 0, sizeof (type_info));
    type_info.class_size = type_query.class_size;
    type_info.instance_size = type_query.instance_size;
    type_info.class_init = gst_v4l2_video_dec_subclass_init;
    type_info.class_data = cdata;
    type_info.instance_init = gst_v4l2_video_dec_subinstance_init;

    subtype = g_type_register_static (type, type_name, &type_info, 0);
    if (!gst_element_register (plugin, type_name, GST_RANK_PRIMARY + 1,
            subtype))
      GST_WARNING ("Failed to register plugin '%s'", type_name);

    g_free (type_name);
  }
}

#else

void
gst_v4l2_video_dec_register (GstPlugin * plugin, const gchar * basename,
    const gchar * device_path, GstCaps * sink_caps, GstCaps * src_caps)
{
  GTypeQuery type_query = { 0, };
  GTypeInfo type_info = { 0, };
  GType type, subtype;
  GstV4l2VideoDecCData *cdata;

  cdata = g_new0 (GstV4l2VideoDecCData, 1);
  cdata->device = g_strdup (device_path);

  type = gst_v4l2_video_dec_get_type ();
  g_type_query (type, &type_query);
  memset (&type_info, 0, sizeof (type_info));
  type_info.class_size = type_query.class_size;
  type_info.instance_size = type_query.instance_size;
  type_info.class_init = gst_v4l2_video_dec_subclass_init;
  type_info.class_data = cdata;
  type_info.instance_init = gst_v4l2_video_dec_subinstance_init;

  subtype = g_type_register_static (type, "nvv4l2decoder", &type_info, 0);
  gst_element_register (plugin, "nvv4l2decoder", GST_RANK_PRIMARY + 11, subtype);
}
#endif
