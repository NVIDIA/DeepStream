/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <gst/rtp/gstrtcpbuffer.h>
#include <gst/rtsp/gstrtsptransport.h>

#include <gst/audio/audio.h>

#include "gstdsnvurisrcbin.h"
#include "nvdsgstutils.h"
#include "gst-nvcommon.h"
#include "gst-nvdscustommessage.h"
#include "gst-nvquery.h"
#include "gst-nvquery-internal.h"
#include "gst-nvdssr.h"
#include "gst-nvevent.h"

#include <memory>

#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>

//Default prop values
#define DEFAULT_NUM_EXTRA_SURFACES 1
#define DEFAULT_GPU_DEVICE_ID 0
#define DEFAULT_NVBUF_MEM_TYPE 0
#define DEFAULT_DROP_FRAME_INTERVAL 0
#define DEFAULT_SOURCE_TYPE 0
#define DEFAULT_DEC_SKIP_FRAME_TYPE 0
#define DEFAULT_BUFFER_MODE 3
#define DEFAULT_RTP_PROTOCOL 7
#define DEFAULT_LEAKY 0
#define DEFAULT_MAX_SIZE_BUFFERS 200
#define DEFAULT_LATENCY 100
#define DEFAULT_FILE_LOOP FALSE
#define DEFAULT_DISABLE_PASSTHROUGH FALSE
#define DEFAULT_LOW_LATENCY_MODE FALSE
#define DEFAULT_SMART_RECORD_MODE 0
#define DEFAULT_SMART_RECORD_PREFIX "Smart_Record"
#define DEFAULT_SMART_RECORD_CACHE 20
#define DEFAULT_SMART_RECORD_CONTAINER 0
#define DEFAULT_SMART_RECORD_DEFAULT_DURATION 20
#define DEFAULT_RTSP_RECONNECT_INTERVAL 0
#define DEFAULT_RTSP_INIT_RECONNECT_INTERVAL 0
#define DEFAULT_RTSP_RECONNECT_ATTEMPTS -1
#define DEFAULT_SOURCE_ID -1
#define DEFAULT_UDP_BUFFER_SIZE 524288
#define DEFAULT_DISABLE_AUDIO TRUE
#define DEFAULT_SEI_EXTRACT_DATA FALSE
#define DEFAULT_SEI_UUID NULL
#define DEFAULT_DROP_ON_LATENCY TRUE
#define DEFAULT_IPC_BUFFER_TIMESTAMP_COPY FALSE
#define DEFAULT_IPC_CONNECTION_ATTEMPTS -1
#define DEFAULT_IPC_CONNECTION_INTERVAL 1000000
#define DEFAULT_SIMULATE_FPS_INTERVAL_MS 0

#define GST_TYPE_NVDSURI_SOURCE_TYPE (gst_nvdsurisrc_get_type ())
#define GST_TYPE_NVDSURI_SKIP_FRAMES (gst_nvdsurisrc_dec_skip_frames ())
#define GST_TYPE_NVDSURI_RTP_PROTOCOL (gst_nvdsurisrc_rtp_protocol ())
#define GST_TYPE_NVDSURI_BUFFER_MODE (gst_nvdsurisrc_buffer_mode ())
#define GST_TYPE_NVDSURI_SMART_RECORD_TYPE (gst_nvdsurisrc_smart_record_type ())
#define GST_TYPE_NVDSURI_SMART_RECORD_MODE (gst_nvdsurisrc_smart_record_mode ())
#define GST_TYPE_NVDSURI_SMART_RECORD_CONTAINER (gst_nvdsurisrc_smart_record_container ())
#define GST_TYPE_NVDSURI_LEAKY (gst_nvdsurisrc_leaky ())

#define GST_TYPE_V4L2_VID_CUDADEC_MEM_TYPE (gst_video_cudadec_mem_type ())
#define DEFAULT_CUDADEC_MEM_TYPE 0

template <class T>
class GstObjectUPtr:public std::unique_ptr<T, void (*)(T *)>
{
public:
  GstObjectUPtr (T * t = nullptr): std::unique_ptr<T, void (*)(T *)>(t, (void (*)(T *)) gst_object_unref) {}
  operator T* () const { return this->get (); }
};

class GstCapsUPtr:public std::unique_ptr <GstCaps, void (*)(GstCaps *)>
{
public:
  GstCapsUPtr (GstCaps * t = nullptr): std::unique_ptr < GstCaps, void (*)(GstCaps *) > (t, gst_caps_unref) {}
  operator GstCaps* () const { return this->get (); }
};

using GstPadUPtr = GstObjectUPtr <GstPad>;

#define NVGSTDS_ELEM_ADD_PROBE(parent_elem, elem, pad, probe_func, probe_type, probe_data) \
    ({ \
      gulong probe_id = 0; \
      GstPad *gstpad = gst_element_get_static_pad (elem, pad); \
      if (!gstpad) { \
        GST_ELEMENT_ERROR(parent_elem, RESOURCE, FAILED, \
            ("Could not find '%s' in '%s'", pad, \
            GST_ELEMENT_NAME(elem)), (NULL)); \
      } else { \
        probe_id = gst_pad_add_probe(gstpad, (GstPadProbeType) (probe_type), probe_func, probe_data, NULL); \
        gst_object_unref (gstpad); \
      } \
      probe_id; \
    })

#define NVGSTDS_LINK_ELEMENT(elem1, elem2, ...) \
    ({ \
      if (!gst_element_link (elem1,elem2)) { \
        GstCaps * src_caps, *sink_caps; \
        src_caps = gst_pad_query_caps ((GstPad *) (elem1)->srcpads->data, NULL); \
        sink_caps = gst_pad_query_caps ((GstPad *) (elem2)->sinkpads->data, NULL); \
        GST_ELEMENT_ERROR (GST_ELEMENT_PARENT(elem1), STREAM, FAILED, \
            ("Failed to link '%s' (%s) and '%s' (%s)", \
                GST_ELEMENT_NAME (elem1), \
                gst_caps_to_string (src_caps), \
                GST_ELEMENT_NAME (elem2), \
                gst_caps_to_string (sink_caps)), (NULL)); \
        return __VA_ARGS__; \
      } \
    })

static GType
gst_video_cudadec_mem_type (void)
{
  static GType qtype = 0;

  if (qtype == 0) {
    static const GEnumValue values[] = {
      {0, "Memory type Device", "memtype_device"},
      {1, "Memory type Host Pinned", "memtype_pinned"},
      {2, "Memory type Unified", "memtype_unified"},
      {0, NULL, NULL}
    };

    qtype = g_enum_register_static ("GstNvUriSrcBinCudaDecMemType", values);
  }
  return qtype;
}

static GType
gst_nvdsurisrc_get_type (void)
{
  static gsize initialization_value = 0;
  static const GEnumValue source_type[] = {
    {SOURCE_TYPE_AUTO, "Select source type based on URI scheme", "auto"},
    {SOURCE_TYPE_URI, "Supports any URI supported by GStreamer", "uri"},
    {SOURCE_TYPE_RTSP, "Customize for RTSP, supports smart recording", "rtsp"},
    {0, NULL, NULL}
  };

  if (g_once_init_enter (&initialization_value)) {
    GType tmp = g_enum_register_static ("GstNvSourceType",
        source_type);
    g_once_init_leave (&initialization_value, tmp);
  }

  return (GType) initialization_value;
}

static GType
gst_nvdsurisrc_dec_skip_frames (void)
{
  static gsize initialization_value = 0;
  static const GEnumValue skip_type[] = {
    {DEC_SKIP_FRAMES_TYPE_NONE, "Decode all frames", "decode_all"},
    {DEC_SKIP_FRAMES_TYPE_NONREF, "Decode non-ref frames",
        "decode_non_ref"},
    {DEC_SKIP_FRAMES_TYPE_KEY_FRAME_ONLY, "decode key frames",
        "decode_key"},
    {0, NULL, NULL}
  };

  if (g_once_init_enter (&initialization_value)) {
    GType tmp = g_enum_register_static ("SkipFrames", skip_type);
    g_once_init_leave (&initialization_value, tmp);
  }
  return (GType) initialization_value;
}

static GType
gst_nvdsurisrc_rtp_protocol (void)
{
  static gsize initialization_value = 0;
  static const GEnumValue rtp_protocol[] = {
    {RTP_PROTOCOL_UNKNOWN, "unknown", "rtp-unknown"},
    {RTP_PROTOCOL_UDP, "udp", "rtp-udp"},
    {RTP_PROTOCOL_UDP_MCAST, "udp-mcast", "rtp-udp-mcast"},
    {RTP_PROTOCOL_TCP, "tcp", "rtp-tcp"},
    {RTP_PROTOCOL_UDP_UDPMCAST_TCP, "udp-udpmcast-tcp", "rtp-udp-udpmcast-tcp"},
    {RTP_PROTOCOL_HTTP, "http", "rtp-http"},
    {RTP_PROTOCOL_TLS, "tls", "rtp-tls"},
    {0, NULL, NULL}
  };

  if (g_once_init_enter (&initialization_value)) {
    GType tmp = g_enum_register_static ("RtpProtocol", rtp_protocol);
    g_once_init_leave (&initialization_value, tmp);
  }
  return (GType) initialization_value;
}

static GType
gst_nvdsurisrc_buffer_mode (void)
{
  static gsize initialization_value = 0;
  static const GEnumValue buffer_mode[] = {
    {BUFFER_MODE_UNKNOWN, "Only use RTP timestamps", "unknown"},
    {BUFFER_MODE_SLAVE, "Slave receiver to sender clock", "slave"},
    {BUFFER_MODE_BUFFER, "Do low/high watermark buffering", "buffer"},
    {BUFFER_MODE_AUTO, "Choose mode depending on stream live", "auto"},
    {BUFFER_MODE_SYNCED, "Synchronized sender and receiver clocks", "synced"},
    {0, NULL, NULL}
  };

  if (g_once_init_enter (&initialization_value)) {
    GType tmp = g_enum_register_static ("BufferMode", buffer_mode);
    g_once_init_leave (&initialization_value, tmp);
  }
  return (GType) initialization_value;
}

static GType
gst_nvdsurisrc_smart_record_type (void)
{
  static gsize initialization_value = 0;
  static const GEnumValue smart_rec_type[] = {
    {SMART_REC_DISABLE, "Disable Smart Record", "smart-rec-disable"},
    {SMART_REC_CLOUD, "Trigger Smart Record through cloud messages only",
        "smart-rec-cloud"},
    {SMART_REC_MULTI, "Trigger Smart Record through cloud and local events",
        "smart-rec-multi"},
    {0, NULL, NULL}
  };

  if (g_once_init_enter (&initialization_value)) {
    GType tmp = g_enum_register_static ("SmartRecordType", smart_rec_type);
    g_once_init_leave (&initialization_value, tmp);
  }
  return (GType) initialization_value;
}

static GType
gst_nvdsurisrc_smart_record_mode (void)
{
  static gsize initialization_value = 0;
  static const GEnumValue smart_rec_mode[] = {
    {SMART_REC_AUDIO_VIDEO, "Record audio and video if available",
        "smart-rec-mode-av"},
    {SMART_REC_VIDEO_ONLY, "Record video only if available",
        "smart-rec-mode-video"},
    {SMART_REC_AUDIO_ONLY, "Record audio only if available",
        "smart-rec-mode-audio"},
    {0, NULL, NULL}
  };

  if (g_once_init_enter (&initialization_value)) {
    GType tmp = g_enum_register_static ("SmartRecordMode", smart_rec_mode);
    g_once_init_leave (&initialization_value, tmp);
  }
  return (GType) initialization_value;
}

static GType
gst_nvdsurisrc_smart_record_container (void)
{
  static gsize initialization_value = 0;
  static const GEnumValue smart_rec_container[] = {
    {SMART_REC_MP4, "MP4 container", "smart-rec-mp4"},
    {SMART_REC_MKV, "MKV container", "smart-rec-mkv"},
    {0, NULL, NULL}
  };

  if (g_once_init_enter (&initialization_value)) {
    GType tmp = g_enum_register_static ("SmartRecordContainerType",
        smart_rec_container);
    g_once_init_leave (&initialization_value, tmp);
  }
  return (GType) initialization_value;
  }

static GType
gst_nvdsurisrc_leaky (void)
{
  static gsize initialization_value = 0;
  static const GEnumValue leaky[] = {
    {0, "No Leak", "no-leak"},
    {1, "Upstream Leak", "upstream-leak"},
    {2, "Downstream Leak", "downstream-leak"},
    {0, NULL, NULL}
  };

  if (g_once_init_enter (&initialization_value)) {
    GType tmp = g_enum_register_static ("Leaky", leaky);
    g_once_init_leave (&initialization_value, tmp);
  }
  return (GType) initialization_value;
}

static GstStaticPadTemplate gst_nvurisrc_bin_vsrc_template =
    GST_STATIC_PAD_TEMPLATE ("vsrc_%u",
    GST_PAD_SRC,
    GST_PAD_SOMETIMES,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES ("memory:NVMM",
            "{ " "I420,  NV12, P010_10LE, BGRx, RGBA, GRAY8 }") ";"
        GST_VIDEO_CAPS_MAKE ("{ "
            "I420, P010_10LE, NV12, BGRx, RGBA, GRAY8 }")));

static GstStaticPadTemplate gst_nvurisrc_bin_asrc_template =
GST_STATIC_PAD_TEMPLATE ("asrc_%u",
    GST_PAD_SRC,
    GST_PAD_SOMETIMES,
    GST_STATIC_CAPS (GST_AUDIO_CAPS_MAKE (GST_AUDIO_FORMATS_ALL)));

GST_DEBUG_CATEGORY (gst_ds_nvurisrc_bin_debug);
#define GST_CAT_DEFAULT gst_ds_nvurisrc_bin_debug

/* Define our element type. Standard GObject/GStreamer boilerplate stuff */
#define gst_ds_nvurisrc_bin_parent_class parent_class
#define _do_init \
    GST_DEBUG_CATEGORY_INIT (gst_ds_nvurisrc_bin_debug, "nvurisrcbin", 0, "nvurisrcbin element");
G_DEFINE_TYPE_WITH_CODE (GstDsNvUriSrcBin, gst_ds_nvurisrc_bin, GST_TYPE_BIN,
    _do_init);

static void gst_ds_nvurisrc_bin_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * spec);
static void gst_ds_nvurisrc_bin_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * spec);
static void gst_ds_nvurisrc_bin_finalize (GObject * object);
static GstStateChangeReturn
gst_ds_nvurisrc_bin_change_state (GstElement * element,
    GstStateChange transition);

static gboolean reset_source_pipeline (gpointer data);
static gboolean reset_ipc_source_pipeline (gpointer data);
static GstPadProbeReturn
rtspsrc_monitor_probe_func (GstPad * pad, GstPadProbeInfo * info,
    gpointer u_data);
static void cb_newpad2 (GstElement * decodebin, GstPad * pad, gpointer data);

static gboolean
populate_uri_bin_video (GstDsNvUriSrcBin * nvurisrcbin, GstPad * pad);
static gboolean
populate_uri_bin_audio (GstDsNvUriSrcBin * nvurisrcbin, GstPad * pad);

static GstPadProbeReturn src_pad_query_probe (GstPad *pad, GstPadProbeInfo *info,
    gpointer data);

enum
{
  /* actions */
  SIGNAL_START_SR,
  SIGNAL_STOP_SR,
  SIGNAL_SR_DONE,
  LAST_SIGNAL
};

guint gst_ds_nvurisrc_bin_signals[LAST_SIGNAL] = { 0 };

static NvDsSRStatus
gst_ds_nvurisrc_start_sr (GstDsNvUriSrcBin * ubin, NvDsSRSessionId * sessionId,
    guint startTime, guint duration, gpointer userData)
{
  if (!ubin->recordCtx) return NVDSSR_STATUS_INVALID_VAL;

  gpointer *wrapped_userdata = g_new (gpointer, 2);
  wrapped_userdata[0] = ubin;
  wrapped_userdata[1] = userData;

  NvDsSRStatus status = NvDsSRStart (ubin->recordCtx, sessionId, startTime, duration,
      wrapped_userdata);

  if (status ==  NVDSSR_STATUS_ERROR)
    g_free (wrapped_userdata);

  return status;
}

static NvDsSRStatus
gst_ds_nvurisrc_stop_sr (GstDsNvUriSrcBin * ubin, NvDsSRSessionId sessionId)
{
  if (!ubin->recordCtx) return NVDSSR_STATUS_INVALID_VAL;

  return NvDsSRStop (ubin->recordCtx, sessionId);
}

static void
gst_ds_nvurisrc_bin_handle_message (GstBin * bin, GstMessage * message)
{
  GstDsNvUriSrcBin *ubin = (GstDsNvUriSrcBin *) bin;

  /* If source watch is installed, convert error to warning since we
   * don't want app to stop the pipeline. */
  /* Note: GST_IS_OBJECT() check is essential as the ubin->elems might be
   * removed (floating refs) by the time we receive this callback
   */
    gboolean is_ipc = ubin->config->uri
      && g_str_has_prefix (ubin->config->uri, "ipc://");
  if (GST_IS_OBJECT (ubin->src_elem)
      && (GST_MESSAGE_SRC (message) == GST_OBJECT (ubin->src_elem))
      && (is_ipc || ubin->source_watch_id)
      && GST_MESSAGE_TYPE (message) == GST_MESSAGE_ERROR) {
    if (is_ipc)
    {
      gboolean ret = reset_ipc_source_pipeline (ubin);
      if (ret) {
        g_print ("EOS Event sent successfully to remove the source %s\n", ubin->config->uri);
      } else {
        g_print ("EOS Event send failed to remove the source %s\n", ubin->config->uri);        
      }
    }
    GST_MESSAGE_TYPE (message) = GST_MESSAGE_WARNING;
  }

  /* Allow decoded pads to be unlinked. */
  if ((GST_IS_OBJECT (ubin->cap_filter) && GST_IS_OBJECT (ubin->aqueue))
      && (GST_MESSAGE_SRC (message) == GST_OBJECT (ubin->cap_filter)
          || GST_MESSAGE_SRC (message) == GST_OBJECT (ubin->aqueue))
      && GST_MESSAGE_TYPE (message) == GST_MESSAGE_ERROR) {
    gchar *debug = NULL;
    GError *error = NULL;
    gst_message_parse_error (message, &error, &debug);
    if (g_strrstr (debug, "reason not-linked")) {
      gst_message_unref (message);
      return;
    }
  }

  GST_BIN_CLASS (parent_class)->handle_message (bin, message);
}

static void
gst_ds_nvurisrc_bin_class_init (GstDsNvUriSrcBinClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;
  GstBinClass *gstbin_class;

  gobject_class = G_OBJECT_CLASS (klass);
  gstelement_class = GST_ELEMENT_CLASS (klass);
  gstbin_class = GST_BIN_CLASS (klass);

  gobject_class->set_property =
      GST_DEBUG_FUNCPTR (gst_ds_nvurisrc_bin_set_property);
  gobject_class->get_property =
      GST_DEBUG_FUNCPTR (gst_ds_nvurisrc_bin_get_property);
  gobject_class->finalize = GST_DEBUG_FUNCPTR (gst_ds_nvurisrc_bin_finalize);
  gstelement_class->change_state =
      GST_DEBUG_FUNCPTR (gst_ds_nvurisrc_bin_change_state);
  gstbin_class->handle_message =
      GST_DEBUG_FUNCPTR (gst_ds_nvurisrc_bin_handle_message);

  gst_element_class_add_static_pad_template (gstelement_class,
      &gst_nvurisrc_bin_vsrc_template);
  gst_element_class_add_static_pad_template (gstelement_class,
      &gst_nvurisrc_bin_asrc_template);

  g_object_class_install_property (gobject_class, PROP_URI,
      g_param_spec_string ("uri", "URI of source",
          "URI of the file or rtsp source", NULL,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_NUM_EXTRA_SURF,
      g_param_spec_uint ("num-extra-surfaces",
          "Set extra decoder surfaces",
          "Number of surfaces in addition to minimum decode surfaces given by the decoder",
          0, G_MAXUINT, DEFAULT_NUM_EXTRA_SURFACES,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_GPU_DEVICE_ID,
      g_param_spec_uint ("gpu-id", "Set GPU Device ID",
          "Set GPU Device ID",
          0, G_MAXUINT, DEFAULT_GPU_DEVICE_ID,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_CUDADEC_MEM_TYPE,
      g_param_spec_enum ("cudadec-memtype",
          "Memory type for cuda decoder buffers",
          "Set to specify memory type for cuda decoder buffers",
          GST_TYPE_V4L2_VID_CUDADEC_MEM_TYPE, DEFAULT_CUDADEC_MEM_TYPE,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_DROP_FRAME_INTERVAL,
      g_param_spec_uint ("drop-frame-interval",
          "Set decoder drop frame interval",
          "Interval to drop the frames,ex: value of 5 means every 5th frame will be given by decoder, rest all dropped",
          0, 30, DEFAULT_DROP_FRAME_INTERVAL,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_SOURCE_TYPE,
      g_param_spec_enum ("type", "Set source type",
          "Set the type of source. Use source-type-rtsp to use smart record features",
          GST_TYPE_NVDSURI_SOURCE_TYPE, DEFAULT_SOURCE_TYPE,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_DEC_SKIP_FRAMES,
      g_param_spec_enum ("dec-skip-frames",
          "Type of frames to skip during decoding",
          "Type of frames to skip during decoding",
          GST_TYPE_NVDSURI_SKIP_FRAMES, DEFAULT_DEC_SKIP_FRAME_TYPE,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_BUFFER_MODE,
      g_param_spec_enum ("buffer-mode", "Control the buffering algorithm in use",
          "Control the buffering algorithm in use",
          GST_TYPE_NVDSURI_BUFFER_MODE, DEFAULT_BUFFER_MODE,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_RTSP_RECONNECT_ATTEMPTS,
      g_param_spec_int ("rtsp-reconnect-attempts", "Set rtsp reconnect attempts value",
          "Set rtsp reconnect attempt value",
          G_MININT, G_MAXINT, DEFAULT_RTSP_RECONNECT_ATTEMPTS,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_RTP_PROTOCOL,
      g_param_spec_enum ("select-rtp-protocol",
          "Transport Protocol to use for RTP",
          "Transport Protocol to use for RTP", GST_TYPE_NVDSURI_RTP_PROTOCOL,
          DEFAULT_RTP_PROTOCOL,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_LEAKY,
      g_param_spec_enum ("leaky",
          "Where the queue leaks, if at all",
          "Where the queue leaks, if at all", GST_TYPE_NVDSURI_LEAKY, DEFAULT_LEAKY,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_MAX_SIZE_BUFFERS,
      g_param_spec_uint ("max-size-buffers",
          "Maximum number of buffers in the queue",
          "Maximum number of buffers in the queue",
          0, G_MAXUINT, DEFAULT_MAX_SIZE_BUFFERS,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_FILE_LOOP,
      g_param_spec_boolean ("file-loop",
          "Loop file sources after EOS",
          "Loop file sources after EOS. Src type must be source-type-uri and uri starting with 'file:/'",
          DEFAULT_FILE_LOOP,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_DISABLE_PASSTHROUGH,
      g_param_spec_boolean ("disable-passthrough",
          "disable-passthrough",
          "Disable passthrough mode at init time, applicable for nvvideoconvert only.",
          DEFAULT_DISABLE_PASSTHROUGH,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property (gobject_class, PROP_DISABLE_AUDIO,
      g_param_spec_boolean ("disable-audio",
          "disable-audio",
          "Disable audio path mode at init time",
          DEFAULT_DISABLE_AUDIO,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property (gobject_class, PROP_SMART_RECORD,
      g_param_spec_enum ("smart-record",
          "Enable Smart Record",
          "Enable Smart Record and choose the type of events to respond to. Sources must be of type source-type-rtsp",
          GST_TYPE_NVDSURI_SMART_RECORD_TYPE, DEFAULT_SMART_RECORD_MODE,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_SMART_RECORD_DIR_PATH,
      g_param_spec_string ("smart-rec-dir-path",
          "Path of directory to save the recorded file",
          "Path of directory to save the recorded file.", NULL,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_SMART_RECORD_FILE_PREFIX,
      g_param_spec_string ("smart-rec-file-prefix",
          "Prefix of file name for recorded video",
          "By default, Smart_Record is the prefix. For unique file names every source must be provided with a unique prefix",
          DEFAULT_SMART_RECORD_PREFIX,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_SMART_RECORD_VIDEO_CACHE,
      g_param_spec_uint ("smart-rec-video-cache",
          "Size of video cache in seconds.",
          "Size of video cache in seconds. DEPRECATED: Use 'smart-rec-cache' instead",
          0, G_MAXUINT, DEFAULT_SMART_RECORD_CACHE,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_SMART_RECORD_CACHE,
      g_param_spec_uint ("smart-rec-cache",
          "Size of cache in seconds, applies to both audio and video cache",
          "Size of cache in seconds, applies to both audio and video cache",
          0, G_MAXUINT, DEFAULT_SMART_RECORD_CACHE,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_SMART_RECORD_CONTAINER,
      g_param_spec_enum ("smart-rec-container",
          "Container format of recorded video",
          "Container format of recorded video. MP4 and MKV containers are supported. Sources must be of type source-type-rtsp",
          GST_TYPE_NVDSURI_SMART_RECORD_CONTAINER,
          DEFAULT_SMART_RECORD_CONTAINER,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_SMART_RECORD_MODE,
      g_param_spec_enum ("smart-rec-mode",
          "Smart record mode",
          "Smart record mode",
          GST_TYPE_NVDSURI_SMART_RECORD_MODE, DEFAULT_SMART_RECORD_MODE,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class,
      PROP_SMART_RECORD_DEFAULT_DURATION,
      g_param_spec_uint ("smart-rec-default-duration",
          "In case a Stop event is not generated. This parameter will ensure the recording is stopped after a predefined default duration.",
          "In case a Stop event is not generated. This parameter will ensure the recording is stopped after a predefined default duration.",
          0, G_MAXUINT, DEFAULT_SMART_RECORD_DEFAULT_DURATION,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class,
      PROP_SMART_RECORD_STATUS,
      g_param_spec_boolean ("smart-rec-status",
          "Smart Record Status",
          "Boolean indicating if SR is currently",
          FALSE, (GParamFlags) (G_PARAM_READABLE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class,
      PROP_RTSP_RECONNECT_INTERVAL,
      g_param_spec_uint ("rtsp-reconnect-interval", "RTSP Reconnect Interval",
          "Timeout in seconds to wait since last data was received from an RTSP source before forcing a reconnection. 0=disable timeout",
          0, G_MAXUINT, DEFAULT_RTSP_RECONNECT_INTERVAL,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class,
      PROP_RTSP_INIT_RECONNECT_INTERVAL,
      g_param_spec_uint ("init-rtsp-reconnect-interval", "RTSP Reconnect Interval incase error received from rtspsrc",
          "Timeout in seconds to wait before forcing a reconnection when error received from rtsp source. 0=disable timeout",
          0, G_MAXUINT, DEFAULT_RTSP_INIT_RECONNECT_INTERVAL,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class,
      PROP_LATENCY,
      g_param_spec_uint ("latency", "Latency",
          "Jitterbuffer size in milliseconds; applicable only for RTSP streams.",
          0, G_MAXUINT, DEFAULT_LATENCY,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class,
      PROP_SOURCE_ID,
      g_param_spec_int ("source-id", "Source ID",
          "Unique ID for the input source",
          -1, G_MAXINT, DEFAULT_SOURCE_ID,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class,
      PROP_UDP_BUFFER_SIZE,
      g_param_spec_uint ("udp-buffer-size", "UDP Buffer Size",
          "UDP Buffer Size in bytes; applicable only for RTSP streams.",
          0, G_MAXUINT, DEFAULT_UDP_BUFFER_SIZE,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class,
      PROP_EXTRACT_SEI_TYPE5_DATA,
      g_param_spec_boolean ("extract-sei-type5-data",
          "extract-sei-type5-data",
          "Set to extract and attach SEI type5 unregistered data on output buffer",
          DEFAULT_SEI_EXTRACT_DATA,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class,
      PROP_LOW_LATENCY_MODE,
      g_param_spec_boolean ("low-latency-mode",
          "low-latency-mode",
          "Set low latency mode for bitstreams having I and IPPP frames on decoder",
          DEFAULT_LOW_LATENCY_MODE,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_SEI_UUID,
      g_param_spec_string ("sei-uuid",
          "Set sei uuid on decoder",
          "Set sei uuid on decoder", NULL,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));
  g_object_class_install_property (gobject_class, PROP_DROP_ON_LATENCY,
      g_param_spec_boolean ("drop-on-latency",
          "Drop buffers when maximum latency is reached",
          "Tells the jitterbuffer to never exceed the given latency in size",
          DEFAULT_DROP_ON_LATENCY,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  gst_ds_nvurisrc_bin_signals[SIGNAL_START_SR] =
      g_signal_new ("start-sr",
      G_TYPE_FROM_CLASS (klass),
      (GSignalFlags) (G_SIGNAL_RUN_LAST | G_SIGNAL_ACTION),
      G_STRUCT_OFFSET (GstDsNvUriSrcBinClass, start_sr),
      NULL, NULL, NULL, G_TYPE_NONE, 4, G_TYPE_POINTER, G_TYPE_UINT,
      G_TYPE_UINT, G_TYPE_POINTER);

  gst_ds_nvurisrc_bin_signals[SIGNAL_STOP_SR] =
      g_signal_new ("stop-sr",
      G_TYPE_FROM_CLASS (klass),
      (GSignalFlags) (G_SIGNAL_RUN_LAST | G_SIGNAL_ACTION),
      G_STRUCT_OFFSET (GstDsNvUriSrcBinClass, stop_sr),
      NULL, NULL, NULL, G_TYPE_NONE, 1, G_TYPE_UINT);

  gst_ds_nvurisrc_bin_signals[SIGNAL_SR_DONE] =
      g_signal_new ("sr-done",
      G_TYPE_FROM_CLASS (klass),
      (GSignalFlags) (G_SIGNAL_RUN_LAST),
      G_STRUCT_OFFSET (GstDsNvUriSrcBinClass, sr_done),
      NULL, NULL, NULL, G_TYPE_NONE, 2, G_TYPE_POINTER, G_TYPE_POINTER);

  g_object_class_install_property (gobject_class,
      PROP_IPC_BUFFER_TIMESTAMP_COPY,
      g_param_spec_boolean ("ipc-buffer-timestamp-copy", "Copy buffer timestamp for nvunixfdsrc plugin",
          "Copy buffer timestamp for nvunixfdsrc plugin",
          DEFAULT_IPC_BUFFER_TIMESTAMP_COPY, (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_IPC_SOCKET_PATH,
      g_param_spec_string ("ipc-socket-path",
          "Path to the control socket",
          "The path to the control socket used to control the shared memory "
          "transport. This may be modified during the NULL->READY transition",
          NULL,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_IPC_CONNECTION_ATTEMPTS,
    g_param_spec_int ("ipc-connection-attempts", "ipc-connection-attempts",
        "Max number of attempts for connection (-1 = unlimited)",
        -1, G_MAXINT, DEFAULT_IPC_CONNECTION_ATTEMPTS,
        (GParamFlags) (G_PARAM_READWRITE |
        G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_IPC_CONNECTION_INTERVAL,
    g_param_spec_uint64 ("ipc-connection-interval", "ipc-connection-interval",
        "connection interval between connection attempts in micro seconds",
        0, G_MAXUINT64, DEFAULT_IPC_CONNECTION_INTERVAL,
        (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_SIMULATE_FPS_INTERVAL_MS,
      g_param_spec_uint ("simulate-fps-interval-ms", "Simulate FPS Interval",
          "Frame interval in milliseconds for FPS simulation (e.g., 33 for 30 FPS, 0 to disable)",
          0, 1000, DEFAULT_SIMULATE_FPS_INTERVAL_MS,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  klass->start_sr = gst_ds_nvurisrc_start_sr;
  klass->stop_sr = gst_ds_nvurisrc_stop_sr;

  gst_element_class_set_details_simple (gstelement_class,
      "NvUriSrc Bin", "NvUriSrc Bin",
      "Nvidia DeepStreamSDK NvUriSrc Bin",
      "NVIDIA Corporation. Post on Deepstream for Tesla forum for any queries "
      "@ https://devtalk.nvidia.com/default/board/209/");

}

static void
gst_ds_nvurisrc_bin_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{

  GstDsNvUriSrcBin *nvurisrcbin = GST_DS_NVURISRC_BIN (object);
  GstDsNvUriSrcConfig *config = nvurisrcbin->config;
  switch (prop_id) {
    case PROP_URI:
      config->uri = g_value_dup_string (value);
      break;
    case PROP_NUM_EXTRA_SURF:
      config->num_extra_surfaces = g_value_get_uint (value);
      break;
    case PROP_DEC_SKIP_FRAMES:
      config->skip_frames_type =
          (NvDsUriSrcBinDecSkipFrame) g_value_get_enum (value);
      break;
    case PROP_GPU_DEVICE_ID:
      config->gpu_id = g_value_get_uint (value);
      break;
    case PROP_CUDADEC_MEM_TYPE:
      config->cuda_memory_type = g_value_get_enum (value);
      break;
    case PROP_DROP_FRAME_INTERVAL:
      config->drop_frame_interval = g_value_get_uint (value);
      break;
    case PROP_SOURCE_TYPE:
      config->src_type = (NvDsUriSrcBinType) g_value_get_enum (value);
      break;
    case PROP_RTP_PROTOCOL:
      config->rtp_protocol =
          (NvDsUriSrcBinRtpProtocol) g_value_get_enum (value);
      break;
    case PROP_LEAKY:
      config->leaky = (NvDsUriSrcBinLeaky) g_value_get_enum (value);
      break;
    case PROP_MAX_SIZE_BUFFERS:
      config->max_size_buffers = g_value_get_uint (value);
      break;
    case PROP_FILE_LOOP:
      config->loop = g_value_get_boolean (value);
      break;
    case PROP_SMART_RECORD:
      config->smart_record = (NvDsUriSrcBinSRType) g_value_get_enum (value);
      break;
    case PROP_SMART_RECORD_DIR_PATH:
      if (config->smart_rec_dir_path != NULL) {
        g_free (config->smart_rec_dir_path);
        config->smart_rec_dir_path = NULL;
      }
      config->smart_rec_dir_path = g_value_dup_string (value);
      break;
    case PROP_SMART_RECORD_FILE_PREFIX:
      if (config->smart_rec_file_prefix != NULL) {
        g_free (config->smart_rec_file_prefix);
        config->smart_rec_file_prefix = NULL;
      }
      config->smart_rec_file_prefix = g_value_dup_string (value);
      break;
    case PROP_SMART_RECORD_VIDEO_CACHE:
      g_warning
          ("%s: Deprecated property 'smart-rec-video-cache' set. Set property 'smart-rec-cache' instead.",
          GST_ELEMENT_NAME (nvurisrcbin));
    case PROP_SMART_RECORD_CACHE:
      config->smart_rec_cache_size = g_value_get_uint (value);
      break;
    case PROP_SMART_RECORD_CONTAINER:
      config->smart_rec_container =
          (NvDsUriSrcBinSRCont) g_value_get_enum (value);
      break;
    case PROP_SMART_RECORD_MODE:
      config->smart_rec_mode = (NvDsUriSrcBinSRMode) g_value_get_enum (value);
      break;
    case PROP_SMART_RECORD_DEFAULT_DURATION:
      config->smart_rec_def_duration = g_value_get_uint (value);
      break;
    case PROP_BUFFER_MODE:
      config->buffer_mode = (NvDsUriSrcBinBufferMode) g_value_get_enum (value);
      break;
    case PROP_RTSP_RECONNECT_INTERVAL:
      config->rtsp_reconnect_interval_sec = g_value_get_uint (value);
      config->rtsp_reconnect_interval_sec_org = config->rtsp_reconnect_interval_sec;
      break;
    case PROP_RTSP_INIT_RECONNECT_INTERVAL:
      config->init_rtsp_reconnect_interval_sec = g_value_get_uint (value);
      break;
    case PROP_RTSP_RECONNECT_ATTEMPTS:
      config->rtsp_reconnect_attempts = g_value_get_int (value);
      if (config->rtsp_reconnect_attempts == 0) {
        config->rtsp_reconnect_attempts = DEFAULT_RTSP_RECONNECT_ATTEMPTS;
      }
      break;
    case PROP_LATENCY:
      config->latency = g_value_get_uint (value);
      break;
    case PROP_UDP_BUFFER_SIZE:
      config->udp_buffer_size = g_value_get_uint (value);
      break;
    case PROP_SOURCE_ID:
      config->source_id = g_value_get_int (value);
      break;
    case PROP_DISABLE_PASSTHROUGH:
      config->disable_passthrough = g_value_get_boolean (value);
      break;
    case PROP_LOW_LATENCY_MODE:
      config->low_latency_mode = g_value_get_boolean (value);
      break;
    case PROP_DISABLE_AUDIO:
      config->disable_audio = g_value_get_boolean (value);
      break;
    case PROP_EXTRACT_SEI_TYPE5_DATA:
      config->extract_sei_type5_data = g_value_get_boolean (value);
      break;
    case PROP_SEI_UUID:
      config->sei_uuid = g_value_dup_string (value);
      break;
    case PROP_DROP_ON_LATENCY:
      config->drop_on_latency = g_value_get_boolean (value);
      break;
    case PROP_IPC_BUFFER_TIMESTAMP_COPY:
      config->ipc_buffer_timestamp_copy = g_value_get_boolean (value);
      break;
    case PROP_IPC_SOCKET_PATH:
      if (config->ipc_socket_path != NULL) {
        g_free(config->ipc_socket_path);
        config->ipc_socket_path = NULL;
      }
      config->ipc_socket_path = g_value_dup_string (value);
      break;
    case PROP_IPC_CONNECTION_ATTEMPTS:
      config->ipc_connection_attempts = g_value_get_int (value);
      break;
    case PROP_IPC_CONNECTION_INTERVAL:
      config->ipc_connection_interval = g_value_get_uint64 (value);
      break;
    case PROP_SIMULATE_FPS_INTERVAL_MS:
      config->simulate_fps_interval_ms = g_value_get_uint (value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }

}

static void
gst_ds_nvurisrc_bin_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{

  GstDsNvUriSrcBin *nvurisrcbin = GST_DS_NVURISRC_BIN (object);
  GstDsNvUriSrcConfig *config = nvurisrcbin->config;

  switch (prop_id) {
    case PROP_URI:
      g_value_set_string (value, config->uri);
      break;
    case PROP_NUM_EXTRA_SURF:
      g_value_set_uint (value, config->num_extra_surfaces);
      break;
    case PROP_DEC_SKIP_FRAMES:
      g_value_set_enum (value, config->skip_frames_type);
      break;
    case PROP_GPU_DEVICE_ID:
      g_value_set_uint (value, config->gpu_id);
      break;
    case PROP_CUDADEC_MEM_TYPE:
      g_value_set_enum (value, config->cuda_memory_type);
      break;
    case PROP_DROP_FRAME_INTERVAL:
      g_value_set_uint (value, config->drop_frame_interval);
      break;
    case PROP_SOURCE_TYPE:
      g_value_set_enum (value, config->src_type);
      break;
    case PROP_RTP_PROTOCOL:
      g_value_set_enum (value, config->rtp_protocol);
      break;
    case PROP_LEAKY:
      g_value_set_enum (value, config->leaky);
      break;
    case PROP_MAX_SIZE_BUFFERS:
      g_value_set_uint (value, config->max_size_buffers);
      break;
    case PROP_FILE_LOOP:
      g_value_set_boolean (value, config->loop);
      break;
    case PROP_SMART_RECORD:
      g_value_set_enum (value, config->smart_record);
      break;
    case PROP_SMART_RECORD_DIR_PATH:
      g_value_set_string (value, config->smart_rec_dir_path);
      break;
    case PROP_SMART_RECORD_FILE_PREFIX:
      g_value_set_string (value, config->smart_rec_file_prefix);
      break;
    case PROP_SMART_RECORD_VIDEO_CACHE:
    case PROP_SMART_RECORD_CACHE:
      g_value_set_uint (value, config->smart_rec_cache_size);
      break;
    case PROP_SMART_RECORD_CONTAINER:
      g_value_set_enum (value, config->smart_rec_container);
      break;
    case PROP_SMART_RECORD_MODE:
      g_value_set_enum (value, config->smart_rec_mode);
      break;
    case PROP_SMART_RECORD_DEFAULT_DURATION:
      g_value_set_uint (value, config->smart_rec_def_duration);
      break;
    case PROP_SMART_RECORD_STATUS:
      g_value_set_boolean (value, nvurisrcbin->recordCtx
          && nvurisrcbin->recordCtx->recordOn);
      break;
    case PROP_BUFFER_MODE:
      g_value_set_enum (value, config->buffer_mode);
      break;
    case PROP_RTSP_RECONNECT_INTERVAL:
      g_value_set_uint (value, config->rtsp_reconnect_interval_sec);
      break;
    case PROP_RTSP_INIT_RECONNECT_INTERVAL:
      g_value_set_uint (value, config->init_rtsp_reconnect_interval_sec);
      break;
    case PROP_RTSP_RECONNECT_ATTEMPTS:
      if (config->rtsp_reconnect_attempts == 0){
        g_value_set_int (value, DEFAULT_RTSP_RECONNECT_ATTEMPTS);
      } else {
        g_value_set_int (value, config->rtsp_reconnect_attempts);
      }
      break;
    case PROP_LATENCY:
      g_value_set_uint (value, config->latency);
      break;
    case PROP_UDP_BUFFER_SIZE:
      g_value_set_uint (value, config->udp_buffer_size);
      break;
    case PROP_SOURCE_ID:
      g_value_set_int (value, config->source_id);
      break;
    case PROP_DISABLE_PASSTHROUGH:
      g_value_set_boolean (value, config->source_id);
      break;
    case PROP_LOW_LATENCY_MODE:
      g_value_set_boolean (value, config->low_latency_mode);
      break;
    case PROP_DISABLE_AUDIO:
      g_value_set_boolean (value, config->disable_audio);
      break;
    case PROP_EXTRACT_SEI_TYPE5_DATA:
      g_value_set_boolean (value, config->extract_sei_type5_data);
      break;
    case PROP_SEI_UUID:
      g_value_set_string (value, config->sei_uuid);
      break;
    case PROP_DROP_ON_LATENCY:
      g_value_set_boolean (value, config->drop_on_latency);
      break;
    case PROP_IPC_BUFFER_TIMESTAMP_COPY:
      g_value_set_boolean (value, config->ipc_buffer_timestamp_copy);
      break;
    case PROP_IPC_SOCKET_PATH:
      g_value_set_string (value, config->ipc_socket_path);
      break;
    case PROP_IPC_CONNECTION_ATTEMPTS:
      g_value_set_int (value, config->ipc_connection_attempts);
      break;
    case PROP_IPC_CONNECTION_INTERVAL:
      g_value_set_uint64 (value, config->ipc_connection_interval);
      break;
    case PROP_SIMULATE_FPS_INTERVAL_MS:
      g_value_set_uint (value, config->simulate_fps_interval_ms);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }

}

static void
gst_ds_nvurisrc_bin_init (GstDsNvUriSrcBin * nvurisrcbin)
{
  g_object_set (G_OBJECT (nvurisrcbin), "async-handling", TRUE, nullptr);
  nvurisrcbin->config =
      (GstDsNvUriSrcConfig *) g_malloc0 (sizeof (GstDsNvUriSrcConfig));

  nvurisrcbin->config->uri = NULL;
  nvurisrcbin->config->num_extra_surfaces = DEFAULT_NUM_EXTRA_SURFACES;
  nvurisrcbin->config->gpu_id = DEFAULT_GPU_DEVICE_ID;
  nvurisrcbin->config->cuda_memory_type = DEFAULT_CUDADEC_MEM_TYPE;
  nvurisrcbin->config->drop_frame_interval = DEFAULT_DROP_FRAME_INTERVAL;
  nvurisrcbin->config->buffer_mode = (NvDsUriSrcBinBufferMode) DEFAULT_BUFFER_MODE;
  nvurisrcbin->config->leaky = (NvDsUriSrcBinLeaky) DEFAULT_LEAKY;
  nvurisrcbin->config->max_size_buffers = DEFAULT_MAX_SIZE_BUFFERS;
  nvurisrcbin->config->src_type = (NvDsUriSrcBinType) DEFAULT_SOURCE_TYPE;
  nvurisrcbin->config->skip_frames_type =
      (NvDsUriSrcBinDecSkipFrame) DEFAULT_DEC_SKIP_FRAME_TYPE;
  nvurisrcbin->config->rtp_protocol =
      (NvDsUriSrcBinRtpProtocol) DEFAULT_RTP_PROTOCOL;
  nvurisrcbin->config->smart_record =
      (NvDsUriSrcBinSRType) DEFAULT_SMART_RECORD_MODE;
  nvurisrcbin->config->smart_rec_dir_path = NULL;
  nvurisrcbin->config->smart_rec_file_prefix =
      g_strdup (DEFAULT_SMART_RECORD_PREFIX);
  nvurisrcbin->config->smart_rec_container =
      (NvDsUriSrcBinSRCont) DEFAULT_SMART_RECORD_CONTAINER;
  nvurisrcbin->config->smart_rec_def_duration =
      DEFAULT_SMART_RECORD_DEFAULT_DURATION;
  nvurisrcbin->config->rtsp_reconnect_interval_sec =
      DEFAULT_RTSP_RECONNECT_INTERVAL;
  nvurisrcbin->config->init_rtsp_reconnect_interval_sec =
      DEFAULT_RTSP_INIT_RECONNECT_INTERVAL;
  nvurisrcbin->config->rtsp_reconnect_attempts =
      DEFAULT_RTSP_RECONNECT_ATTEMPTS;
  nvurisrcbin->config->num_rtsp_reconnects = 0;
  nvurisrcbin->config->latency = DEFAULT_LATENCY;
  nvurisrcbin->config->disable_passthrough = DEFAULT_DISABLE_PASSTHROUGH;
  nvurisrcbin->config->low_latency_mode = DEFAULT_LOW_LATENCY_MODE;
  nvurisrcbin->config->disable_audio = DEFAULT_DISABLE_AUDIO;
  nvurisrcbin->config->source_id = DEFAULT_SOURCE_ID;
  nvurisrcbin->config->udp_buffer_size = DEFAULT_UDP_BUFFER_SIZE;
  nvurisrcbin->config->extract_sei_type5_data = DEFAULT_SEI_EXTRACT_DATA;
  nvurisrcbin->config->sei_uuid = DEFAULT_SEI_UUID;
  nvurisrcbin->config->drop_on_latency = DEFAULT_DROP_ON_LATENCY;
  nvurisrcbin->config->ipc_buffer_timestamp_copy = DEFAULT_IPC_BUFFER_TIMESTAMP_COPY;
  nvurisrcbin->config->ipc_socket_path = NULL;
  nvurisrcbin->config->ipc_connection_attempts = DEFAULT_IPC_CONNECTION_ATTEMPTS;
  nvurisrcbin->config->ipc_connection_interval = DEFAULT_IPC_CONNECTION_INTERVAL;
  nvurisrcbin->config->simulate_fps_interval_ms = DEFAULT_SIMULATE_FPS_INTERVAL_MS;
  g_mutex_init (&nvurisrcbin->bin_lock);

  GST_OBJECT_FLAG_SET (nvurisrcbin, GST_ELEMENT_FLAG_SOURCE);
}

static void
destroy_smart_record_bin (gpointer data)
{
  GstDsNvUriSrcBin *src_bin = (GstDsNvUriSrcBin *) data;
  if (src_bin->recordCtx)
    NvDsSRDestroy ((NvDsSRContext *) src_bin->recordCtx);
}

/* Free resources allocated during init. */
static void
gst_ds_nvurisrc_bin_finalize (GObject * object)
{
  GstDsNvUriSrcBin *nvurisrcbin = GST_DS_NVURISRC_BIN (object);

  if (nvurisrcbin->config) {
    if (nvurisrcbin->config->ipc_socket_path) {
      g_free(nvurisrcbin->config->ipc_socket_path);
      nvurisrcbin->config->ipc_socket_path = NULL;
    }
    if (nvurisrcbin->config->smart_rec_dir_path) {
      g_free(nvurisrcbin->config->smart_rec_dir_path);
      nvurisrcbin->config->smart_rec_dir_path = NULL;
    }
    if (nvurisrcbin->config->smart_rec_file_prefix) {
      g_free(nvurisrcbin->config->smart_rec_file_prefix);
      nvurisrcbin->config->smart_rec_file_prefix = NULL;
    }
    g_free (nvurisrcbin->config);
    nvurisrcbin->config = NULL;
  }

  destroy_smart_record_bin (nvurisrcbin);

  G_OBJECT_CLASS (parent_class)->finalize (object);
}


/*
 * Function to seek the source stream to start.
 * It is required to play the stream in loop.
 */
static gboolean
seek_decode (gpointer data)
{
  GstDsNvUriSrcBin *bin = (GstDsNvUriSrcBin *) data;
  gboolean ret = TRUE;

  gst_element_set_state (GST_ELEMENT (bin), GST_STATE_PAUSED);

  ret = gst_element_seek (GST_ELEMENT (bin), 1.0, GST_FORMAT_TIME,
      (GstSeekFlags) (GST_SEEK_FLAG_KEY_UNIT | GST_SEEK_FLAG_FLUSH),
      GST_SEEK_TYPE_SET, 0, GST_SEEK_TYPE_NONE, GST_CLOCK_TIME_NONE);

  if (!ret)
    GST_WARNING ("Error in seeking pipeline");

  gst_element_set_state (GST_ELEMENT (bin), GST_STATE_PLAYING);

  return FALSE;
}

/**
 * Probe function to drop certain events to support custom
 * logic of looping of each source stream.
 */
static GstPadProbeReturn
restart_stream_buf_prob (GstPad * pad, GstPadProbeInfo * info, gpointer u_data)
{
  GstEvent *event = GST_EVENT (info->data);
  GstDsNvUriSrcBin *bin = (GstDsNvUriSrcBin *) u_data;

  if ((info->type & GST_PAD_PROBE_TYPE_BUFFER)) {
    GST_BUFFER_PTS (GST_BUFFER (info->data)) += bin->prev_accumulated_base;
  }
  if ((info->type & GST_PAD_PROBE_TYPE_EVENT_BOTH)) {
    if (GST_EVENT_TYPE (event) == GST_EVENT_EOS) {
      g_timeout_add (1, seek_decode, bin);
    }

    if (GST_EVENT_TYPE (event) == GST_EVENT_SEGMENT) {
      GstSegment *segment = NULL;

      gst_event_parse_segment (event, (const GstSegment **) &segment);
      segment->base = bin->accumulated_base;
      bin->prev_accumulated_base = bin->accumulated_base;
      bin->accumulated_base += segment->stop;
    }
    switch (GST_EVENT_TYPE (event)) {
      case GST_EVENT_EOS:
        /* QOS events from downstream sink elements cause decoder to drop
         * frames after looping the file since the timestamps reset to 0.
         * We should drop the QOS events since we have custom logic for
         * looping individual sources. */
      case GST_EVENT_QOS:
      case GST_EVENT_SEGMENT:
      case GST_EVENT_FLUSH_START:
      case GST_EVENT_FLUSH_STOP:
        return GST_PAD_PROBE_DROP;
      default:
        break;
    }
  }
  return GST_PAD_PROBE_OK;
}

static void
cb_newpad (GstElement * decodebin, GstPad * pad, gpointer data)
{
  GstCaps *caps = gst_pad_query_caps (pad, NULL);
  const GstStructure *str = gst_caps_get_structure (caps, 0);
  const gchar *name = gst_structure_get_name (str);
  GstDsNvUriSrcBin *bin = (GstDsNvUriSrcBin *) data;

  if (!strncmp (name, "video", 5)
      && !bin->nvvidconv /** make sure we did not already populate for video */
      ) {
    if (!populate_uri_bin_video (bin, pad)) {
      GST_ELEMENT_ERROR (GST_ELEMENT (bin), STREAM, FAILED,
          ("Failed to populate and link video elements"), (NULL));
    }
  } else if (!strncmp (name, "audio", 5)
    && !bin->audio_convert /** make sure we did not already populate for audio */
    && !bin->config->disable_audio /** enable audio by setting disable-audio property to FALSE */
    ) {
    if (!populate_uri_bin_audio (bin, pad)) {
      GST_ELEMENT_ERROR (GST_ELEMENT (bin), STREAM, FAILED,
          ("Failed to populate and link video elements"), (NULL));
    }
  }

  gst_caps_unref (caps);
}

static void
cb_sourcesetup (GstElement * object, GstElement * arg0, gpointer data)
{
  GstDsNvUriSrcBin *bin = (GstDsNvUriSrcBin *) data;
  if (g_object_class_find_property (G_OBJECT_GET_CLASS (arg0), "latency")) {
    g_object_set (G_OBJECT (arg0), "latency", bin->config->latency, NULL);
  }
  if (g_object_class_find_property (G_OBJECT_GET_CLASS (arg0), "udp-buffer-size")) {
    g_object_set (G_OBJECT (arg0), "udp-buffer-size", bin->config->udp_buffer_size, NULL);
  }
}

static void
aparsebin_child_added (GstChildProxy * child_proxy, GObject * object,
    gchar * name, gpointer user_data)
{
  GstElementFactory *factory = gst_element_get_factory (GST_ELEMENT (object));
  GstDsNvUriSrcBin *bin = (GstDsNvUriSrcBin *) user_data;
  auto fname = GST_OBJECT_NAME (factory);
  if (g_strstr_len (fname, -1, "rtp") == fname
      && g_strstr_len (fname, -1, "depay")) {
    bin->adepay = GST_ELEMENT (object);

    if (bin->config->rtsp_reconnect_interval_sec > 0) {
      NVGSTDS_ELEM_ADD_PROBE (bin,
          bin->adepay, "src", rtspsrc_monitor_probe_func,
          GST_PAD_PROBE_TYPE_BUFFER | GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM, bin);
    }
  }
}

/**
 * Probe at decoder INPUT to throttle frame rate and simulate camera/RTSP behavior.
 * This introduces a configurable delay before each frame enters the decoder,
 * making file sources behave like real-time RTSP streams (e.g., 33ms for 30fps).
 * Latency measurement is handled by the standard NVDS_ENABLE_LATENCY_MEASUREMENT
 * mechanism via nvstreammux.
 */
static GstPadProbeReturn
decoder_input_fps_throttle_probe (GstPad * pad, GstPadProbeInfo * info, gpointer u_data)
{
  GstBuffer *buf = (GstBuffer *) info->data;
  GstDsNvUriSrcBin *bin = (GstDsNvUriSrcBin *) u_data;

  if (buf && bin && bin->config->simulate_fps_interval_ms > 0) {
    /* Throttle frame rate to simulate camera delivery rate (e.g., 33ms for 30fps) */
    guint sleep_time_us = bin->config->simulate_fps_interval_ms * 1000;
    g_usleep(sleep_time_us);
  }

  return GST_PAD_PROBE_OK;
}

static void
decodebin_child_added (GstChildProxy * child_proxy, GObject * object,
    gchar * name, gpointer user_data)
{
  GstDsNvUriSrcBin *bin = (GstDsNvUriSrcBin *) user_data;
  GstDsNvUriSrcConfig *config = bin->config;
  if (g_strrstr (name, "decodebin") == name) {
    g_signal_connect (G_OBJECT (object), "child-added",
        G_CALLBACK (decodebin_child_added), user_data);
  }
  if ((g_strrstr (name, "h264parse") == name) ||
      (g_strrstr (name, "h265parse") == name)) {
    g_object_set (object, "config-interval", -1, NULL);
  }
  if (g_strrstr (name, "fakesink") == name) {
    g_object_set (object, "enable-last-sample", FALSE, NULL);
  }
  if (g_strstr_len (name, -1, "nvv4l2decoder") == name) {
    if (config->skip_frames_type)
      g_object_set (object, "skip-frames", config->skip_frames_type, NULL);
    if (g_object_class_find_property (G_OBJECT_GET_CLASS (object),
            "enable-max-performance")) {
      g_object_set (object, "enable-max-performance", TRUE, NULL);
    }
    if (g_object_class_find_property (G_OBJECT_GET_CLASS (object), "gpu-id")) {
      g_object_set (object, "gpu-id", config->gpu_id, NULL);
    }
    if (g_object_class_find_property (G_OBJECT_GET_CLASS (object),
            "cudadec-memtype")) {
      g_object_set (G_OBJECT (object), "cudadec-memtype",
          config->cuda_memory_type, NULL);
    }
    g_object_set (object, "drop-frame-interval", config->drop_frame_interval,
        NULL);
    if (g_object_class_find_property(G_OBJECT_GET_CLASS (object), "low-latency-mode")) {
      g_object_set (object, "low-latency-mode", config->low_latency_mode,
          NULL);
    }
    if (g_object_class_find_property(G_OBJECT_GET_CLASS (object), "extract-sei-type5-data")) {
      g_object_set (object, "extract-sei-type5-data", config->extract_sei_type5_data,
          NULL);
    }
    if (g_object_class_find_property(G_OBJECT_GET_CLASS (object), "sei-uuid")) {
      g_object_set (object, "sei-uuid", config->sei_uuid, NULL);
    }
    g_object_set (object, "num-extra-surfaces", config->num_extra_surfaces,
        NULL);
    /* Seek only if file is the source. */
    if (config->loop && g_strstr_len (config->uri, -1, "file:/") == config->uri) {
      NVGSTDS_ELEM_ADD_PROBE (bin, GST_ELEMENT (object),
          "sink", restart_stream_buf_prob,
          (GstPadProbeType) (GST_PAD_PROBE_TYPE_EVENT_BOTH |
              GST_PAD_PROBE_TYPE_EVENT_FLUSH | GST_PAD_PROBE_TYPE_BUFFER), bin);
    }

    /* Add FPS throttle probe at decoder INPUT (sink pad) to simulate RTSP behavior.
     * This makes file sources deliver frames at a fixed rate (e.g., 30fps).
     * Latency measurement uses standard NVDS_ENABLE_LATENCY_MEASUREMENT mechanism. */
    if (config->simulate_fps_interval_ms > 0) {
      GstPad *decoder_sink_pad = gst_element_get_static_pad (GST_ELEMENT (object), "sink");
      if (decoder_sink_pad) {
        gst_pad_add_probe (decoder_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
            decoder_input_fps_throttle_probe, bin, NULL);
        GST_DEBUG_OBJECT (bin, "FPS throttle probe added at decoder INPUT");
        gst_object_unref (decoder_sink_pad);
      }
    }
  }
}

static gboolean
link_element_to_tee_src_pad (GstElement * tee, GstElement * sinkelem,
    const gchar * sinkpadname = "sink")
{
  GstPadTemplate *padtemplate = NULL;

  padtemplate = (GstPadTemplate *)
      gst_element_class_get_pad_template (GST_ELEMENT_GET_CLASS (tee),
      "src_%u");
  GstPadUPtr tee_src_pad =
      gst_element_request_pad (tee, padtemplate, NULL, NULL);
  if (!tee_src_pad) {
    GST_ELEMENT_ERROR (GST_ELEMENT_PARENT (tee), STREAM, FAILED,
        ("Failed to get sink pad from '%s'", GST_ELEMENT_NAME (tee)), (NULL));
    return FALSE;
  }

  GstPadUPtr sinkpad = gst_element_get_static_pad (sinkelem, sinkpadname);
  if (!sinkpad) {
    GST_ELEMENT_ERROR (GST_ELEMENT_PARENT (tee), STREAM, FAILED,
        ("Failed to get sink pad from '%s'",
            GST_ELEMENT_NAME (sinkelem)), (NULL));
    return FALSE;
  }

  if (gst_pad_link (tee_src_pad, sinkpad) != GST_PAD_LINK_OK) {
    GST_ELEMENT_ERROR (GST_ELEMENT_PARENT (tee), STREAM, FAILED,
        ("Failed to link '%s' and '%s'", GST_ELEMENT_NAME (tee),
            GST_ELEMENT_NAME (sinkelem)), (NULL));
    return FALSE;
  }

  return TRUE;
}

static gboolean
populate_uri_bin_video (GstDsNvUriSrcBin * nvurisrcbin, GstPad * pad)
{
  GstDsNvUriSrcConfig *config = nvurisrcbin->config;
  GstCapsUPtr caps;
  GstCapsFeatures *feature = NULL;

  nvurisrcbin->cap_filter =
      gst_element_factory_make ("queue", "nvurisrc_bin_queue");
  if (!nvurisrcbin->cap_filter) {
    GST_ELEMENT_ERROR (nvurisrcbin, RESOURCE, FAILED,
        ("Could not create 'queue'"), (NULL));
    return FALSE;
  }

  nvurisrcbin->nvvidconv =
      gst_element_factory_make ("nvvideoconvert",
      "nvurisrc_bin_nvvidconv_elem");

  if (!nvurisrcbin->nvvidconv) {
    GST_ELEMENT_ERROR (nvurisrcbin, RESOURCE, FAILED,
        ("Could not create element 'nvvideoconvert'"), (NULL));
    return FALSE;
  }
  g_object_set (G_OBJECT (nvurisrcbin->nvvidconv), "disable-passthrough", config->disable_passthrough, NULL);
  g_object_set (G_OBJECT (nvurisrcbin->nvvidconv), "gpu-id", config->gpu_id, NULL);

  caps = gst_caps_new_empty_simple ("video/x-raw");
  feature = gst_caps_features_new ("memory:NVMM", NULL);
  gst_caps_set_features (caps, 0, feature);

  nvurisrcbin->cap_filter1 =
      gst_element_factory_make ("capsfilter",
      "nvurisrc_bin_src_cap_filter_nvvidconv");
  if (!nvurisrcbin->cap_filter1) {
    GST_ELEMENT_ERROR (nvurisrcbin, RESOURCE, FAILED,
        ("Could not create 'queue'"), (NULL));
    return FALSE;
  }

  g_object_set (G_OBJECT (nvurisrcbin->cap_filter1), "caps", caps.get (), NULL);

  gst_bin_add_many (GST_BIN (nvurisrcbin), nvurisrcbin->cap_filter,
      nvurisrcbin->nvvidconv, nvurisrcbin->cap_filter1, NULL);

  if (!gst_element_sync_state_with_parent (nvurisrcbin->cap_filter)
      || !gst_element_sync_state_with_parent (nvurisrcbin->nvvidconv)
      || !gst_element_sync_state_with_parent (nvurisrcbin->cap_filter1)) {
    GST_ELEMENT_ERROR (nvurisrcbin, RESOURCE, FAILED,
        ("Failed to sync child states"), (NULL));
    return FALSE;
  }

  GstPadUPtr target_pad =
      gst_element_get_static_pad (nvurisrcbin->cap_filter1, "src");
  GstPad *src_pad = gst_ghost_pad_new_from_template ("vsrc_0", target_pad,
      gst_static_pad_template_get (&gst_nvurisrc_bin_vsrc_template));

  gst_pad_add_probe (src_pad, GST_PAD_PROBE_TYPE_QUERY_BOTH,
      src_pad_query_probe, nvurisrcbin, NULL);
  gst_pad_set_active (src_pad, TRUE);
  gst_element_add_pad (GST_ELEMENT (nvurisrcbin), src_pad);

  if (config->loop && g_strstr_len (config->uri, -1, "file:/") == config->uri) {
    nvurisrcbin->fakesink =
        gst_element_factory_make ("fakesink", "nvurisrc_bin__fakesink");
    if (!nvurisrcbin->fakesink) {
      GST_ELEMENT_ERROR (nvurisrcbin, RESOURCE, FAILED,
          ("Could not create 'fakesink'"), (NULL));
      return FALSE;
    }

    nvurisrcbin->fakesink_queue =
        gst_element_factory_make ("queue", "nvurisrc_bin_fakequeue");
    if (!nvurisrcbin->fakesink_queue) {
      GST_ELEMENT_ERROR (nvurisrcbin, RESOURCE, FAILED,
          ("Could not create 'queue'"), (NULL));
      return FALSE;
    }

    nvurisrcbin->tee = gst_element_factory_make ("tee", NULL);
    if (!nvurisrcbin->tee) {
      GST_ELEMENT_ERROR (nvurisrcbin, RESOURCE, FAILED,
          ("Could not create 'tee'"), (NULL));
      return FALSE;
    }
    gst_bin_add_many (GST_BIN (nvurisrcbin), nvurisrcbin->fakesink,
        nvurisrcbin->tee, nvurisrcbin->fakesink_queue, NULL);

    link_element_to_tee_src_pad (nvurisrcbin->tee, nvurisrcbin->fakesink_queue);
    NVGSTDS_LINK_ELEMENT (nvurisrcbin->fakesink_queue, nvurisrcbin->fakesink,
        FALSE);
    link_element_to_tee_src_pad (nvurisrcbin->tee, nvurisrcbin->cap_filter);

    g_object_set (G_OBJECT (nvurisrcbin->fakesink), "sync", FALSE, "async",
        FALSE, "enable-last-sample", FALSE, NULL);

    if (!gst_element_sync_state_with_parent (nvurisrcbin->tee)
        || !gst_element_sync_state_with_parent (nvurisrcbin->fakesink_queue)
        || !gst_element_sync_state_with_parent (nvurisrcbin->fakesink)) {
      GST_ELEMENT_ERROR (nvurisrcbin, RESOURCE, FAILED,
          ("Failed to sync child states"), (NULL));
      return FALSE;
    }
  }

  NVGSTDS_LINK_ELEMENT (nvurisrcbin->cap_filter, nvurisrcbin->nvvidconv, FALSE);
  NVGSTDS_LINK_ELEMENT (nvurisrcbin->nvvidconv, nvurisrcbin->cap_filter1,
      FALSE);
  GstPadUPtr sinkpad =
      gst_element_get_static_pad (nvurisrcbin->tee ? nvurisrcbin->
      tee : nvurisrcbin->cap_filter,
      "sink");
  if (gst_pad_link (pad, sinkpad) != GST_PAD_LINK_OK) {
    GST_ELEMENT_ERROR (nvurisrcbin, STREAM, FAILED,
        ("Failed to link decodebin to pipeline"), (NULL));
    return FALSE;
  } else {
    GST_DEBUG_OBJECT (nvurisrcbin, "Decodebin linked to pipeline");
  }

  return TRUE;
}

static gboolean
populate_uri_bin_audio (GstDsNvUriSrcBin * nvurisrcbin, GstPad * pad)
{
  if (nvurisrcbin->audio_convert)
    return TRUE;

  nvurisrcbin->audio_convert =
      gst_element_factory_make ("audioconvert", "nvurisrc_bin_audioconvert");
  if (!nvurisrcbin->audio_convert) {
    GST_ELEMENT_ERROR (nvurisrcbin, RESOURCE, FAILED,
        ("Could not create 'audioconvert'"), (NULL));
    return FALSE;
  }

  nvurisrcbin->audio_resample =
      gst_element_factory_make ("audioresample", "nvurisrc_bin_audioresample");
  if (!nvurisrcbin->audio_resample) {
    GST_ELEMENT_ERROR (nvurisrcbin, RESOURCE, FAILED,
        ("Could not create 'audioresample'"), (NULL));
    return FALSE;
  }

  gst_bin_add_many (GST_BIN (nvurisrcbin), nvurisrcbin->audio_convert,
      nvurisrcbin->audio_resample, NULL);

  if (!gst_element_sync_state_with_parent (nvurisrcbin->audio_convert)
      || !gst_element_sync_state_with_parent (nvurisrcbin->audio_resample)) {
    GST_ELEMENT_ERROR (nvurisrcbin, RESOURCE, FAILED,
        ("Failed to sync child states"), (NULL));
    return FALSE;
  }

  NVGSTDS_LINK_ELEMENT (nvurisrcbin->audio_convert, nvurisrcbin->audio_resample,
      FALSE);

  GstPadUPtr target_pad =
      gst_element_get_static_pad (nvurisrcbin->audio_resample, "src");
  GstPad *src_pad = gst_ghost_pad_new_from_template ("asrc_0", target_pad,
      gst_static_pad_template_get (&gst_nvurisrc_bin_asrc_template));

  gst_pad_add_probe (src_pad, GST_PAD_PROBE_TYPE_QUERY_BOTH,
      src_pad_query_probe, nvurisrcbin, NULL);

  gst_pad_set_active (src_pad, TRUE);
  gst_element_add_pad (GST_ELEMENT (nvurisrcbin), src_pad);

  GstPadUPtr sinkpad = gst_element_get_static_pad (nvurisrcbin->audio_convert,
      "sink");
  if (gst_pad_link (pad, sinkpad) != GST_PAD_LINK_OK) {
    GST_ELEMENT_ERROR (nvurisrcbin, STREAM, FAILED,
        ("Failed to link decodebin to pipeline"), (NULL));
    return FALSE;
  } else {
    GST_DEBUG_OBJECT (nvurisrcbin, "Decodebin linked to pipeline");
  }

  return TRUE;
}

static gboolean
populate_uri_bin (GstDsNvUriSrcBin * nvurisrcbin)
{
  GstDsNvUriSrcConfig *config = nvurisrcbin->config;

  nvurisrcbin->src_elem =
      gst_element_factory_make ("uridecodebin", "nvurisrc_bin_src_elem");
  if (!nvurisrcbin->src_elem) {
    GST_ELEMENT_ERROR (nvurisrcbin, RESOURCE, FAILED,
        ("Could not create element 'uridecodebin'"), (NULL));
    return FALSE;
  }

  if (config->uri && g_str_has_prefix (config->uri, "rtsp://")) {
    configure_source_for_ntp_sync (nvurisrcbin->src_elem);
  }

  g_object_set (G_OBJECT (nvurisrcbin->src_elem), "uri", config->uri, NULL);
  g_signal_connect (G_OBJECT (nvurisrcbin->src_elem), "pad-added",
      G_CALLBACK (cb_newpad), nvurisrcbin);
  g_signal_connect_swapped (G_OBJECT (nvurisrcbin->src_elem), "no-more-pads",
      G_CALLBACK (gst_element_no_more_pads), nvurisrcbin);
  g_signal_connect (G_OBJECT (nvurisrcbin->src_elem), "child-added",
      G_CALLBACK (decodebin_child_added), nvurisrcbin);
  g_signal_connect (G_OBJECT (nvurisrcbin->src_elem), "source-setup",
      G_CALLBACK (cb_sourcesetup), nvurisrcbin);

  gst_bin_add (GST_BIN (nvurisrcbin), nvurisrcbin->src_elem);

  GST_DEBUG_OBJECT (nvurisrcbin,
      "Decode bin created. Waiting for a new pad from decodebin to link");

  return TRUE;
}

static gpointer
smart_record_callback (NvDsSRRecordingInfo * info, gpointer userData)
{
  gpointer *wrapped_user_data = (gpointer *) userData;
  GObject *ubin = (GObject *) wrapped_user_data[0];
  gpointer udata = wrapped_user_data[1];
  g_free (wrapped_user_data);
  g_signal_emit_by_name (ubin, "sr-done", info, udata);

  return NULL;
}

/**
 * Function called at regular interval to check if NV_DS_SOURCE_RTSP type
 * source in the pipeline is down / disconnected. This function try to
 * reconnect the source by resetting that source pipeline.
 */
static gboolean
watch_source_status (gpointer data)
{
  GstDsNvUriSrcBin *src_bin = (GstDsNvUriSrcBin *) data;
  struct timeval current_time;
  gettimeofday (&current_time, NULL);
  static struct timeval last_reset_time_global = { 0, 0 };

  gdouble time_diff_msec_since_last_reset =
      1000.0 * (current_time.tv_sec - last_reset_time_global.tv_sec) +
      (current_time.tv_usec - last_reset_time_global.tv_usec) / 1000.0;
  if (src_bin->config->rtsp_reconnect_attempts == -1) {
    src_bin->config->rtsp_reconnect_attempts = INT_MAX;
  }
  if (src_bin->reconfiguring) {
    guint time_since_last_reconnect_sec =
        current_time.tv_sec - src_bin->last_reconnect_time.tv_sec;
    if (time_since_last_reconnect_sec > (guint) src_bin->config->rtsp_reconnect_interval_sec) {
      if (time_diff_msec_since_last_reset > 3000) {
        if (src_bin->config->rtsp_reconnect_attempts == -1 ||
           ++src_bin->config->num_rtsp_reconnects <=
           src_bin->config->rtsp_reconnect_attempts){
          last_reset_time_global = current_time;
          // source is still not up, reconfigure it again.
          reset_source_pipeline (src_bin);
        } else {
          GstObject *parent = NULL;
          GstBus *bus = NULL;
          gboolean attempt_exceeded =
            src_bin->config->num_rtsp_reconnects >
            src_bin->config->rtsp_reconnect_attempts ? 1 : 0;
          if (!(parent = gst_object_get_parent (GST_OBJECT(src_bin)))) {
            g_print("Unabled to get the parent of nvurisrcbin\n");
          } else {
            bus = gst_element_get_bus ((GstElement *) parent);
            gst_object_unref (parent);
          }
          NvDsRtspAttemptsInfo rtsp_info = {};
          rtsp_info.source_id =  src_bin->config->source_id;
          rtsp_info.attempt_exceeded = attempt_exceeded;

          if (bus) {
            gst_bus_post (bus, gst_nvmessage_reconnect_attempt_exceeded
              (GST_OBJECT(src_bin),&rtsp_info));
            gst_object_unref (bus);
          }
        return FALSE;
        }
      }
    }
  } else {
    gint time_since_last_buf_sec = 0;
    g_mutex_lock (&src_bin->bin_lock);
    if (src_bin->last_buffer_time.tv_sec != 0) {
      time_since_last_buf_sec =
          current_time.tv_sec - src_bin->last_buffer_time.tv_sec;
    }
    g_mutex_unlock (&src_bin->bin_lock);

    // Reset source bin if no buffers are received in the last
    // `rtsp_reconnect_interval_sec` seconds.
    if (src_bin->config->rtsp_reconnect_interval_sec > 0 &&
        time_since_last_buf_sec >=
        src_bin->config->rtsp_reconnect_interval_sec) {
      if (time_diff_msec_since_last_reset > 3000) {
        if (src_bin->config->rtsp_reconnect_attempts == -1 ||
           ++src_bin->config->num_rtsp_reconnects <=
           src_bin->config->rtsp_reconnect_attempts){
          last_reset_time_global = current_time;

          GST_ELEMENT_WARNING (src_bin, STREAM, FAILED,
              ("No data from source since last %u sec. Trying reconnection",
                time_since_last_buf_sec), (NULL));
          reset_source_pipeline (src_bin);
        } else {
          GstObject *parent = NULL;
          GstBus *bus = NULL;
          gboolean attempt_exceeded =
            src_bin->config->num_rtsp_reconnects >
            src_bin->config->rtsp_reconnect_attempts ? 1 : 0;
          if (!(parent = gst_object_get_parent (GST_OBJECT(src_bin)))) {
            g_print("Unabled to get the parent of nvurisrcbin\n");
          } else {
            bus = gst_element_get_bus ((GstElement *) parent);
            gst_object_unref (parent);
          }

          NvDsRtspAttemptsInfo rtsp_info = {};
          rtsp_info.source_id =  src_bin->config->source_id;
          rtsp_info.attempt_exceeded = attempt_exceeded;

          if (bus) {
            gst_bus_post (bus, gst_nvmessage_reconnect_attempt_exceeded
              (GST_OBJECT(src_bin),&rtsp_info));
            gst_object_unref (bus);
          }

          return FALSE;
        }
      }
    }

  }
  return TRUE;
}

/**
 * Function called at regular interval when source bin is
 * changing state async. This function watches the state of
 * the source bin and sets it to PLAYING if the state of source
 * bin stops at PAUSED when changing state ASYNC.
 */
static gboolean
watch_source_async_state_change (gpointer data)
{
  GstDsNvUriSrcBin *src_bin = (GstDsNvUriSrcBin *) data;
  GstState state = GST_STATE_NULL;
  GstState pending = GST_STATE_VOID_PENDING;
  GstStateChangeReturn ret;

  ret = gst_element_get_state (GST_ELEMENT (src_bin), &state, &pending, 0);

  GST_DEBUG_OBJECT (src_bin,
      "Bin %s %p: state:%s pending:%s ret:%s", GST_ELEMENT_NAME (src_bin),
      src_bin, gst_element_state_get_name (state),
      gst_element_state_get_name (pending),
      gst_element_state_change_return_get_name (ret));

  if (state == GST_STATE_NULL){
      src_bin->reconfiguring = FALSE;
      if (src_bin->source_watch_id){
        g_source_remove (src_bin->source_watch_id);
      }
      src_bin->source_watch_id = 0;
      return FALSE;
  }

  // Bin is still changing state ASYNC. Wait for some more time.
  if (ret == GST_STATE_CHANGE_ASYNC)
    return TRUE;

  // Bin state change failed / failed to get state
  if (ret == GST_STATE_CHANGE_FAILURE) {
    src_bin->async_state_watch_running = FALSE;
    return FALSE;
  }
  // Bin successfully changed state to PLAYING. Stop watching state
  if (state == GST_STATE_PLAYING) {
    src_bin->reconfiguring = FALSE;
    src_bin->async_state_watch_running = FALSE;
    src_bin->config->num_rtsp_reconnects = 0;
    return FALSE;
  }
  // Bin has stopped ASYNC state change but has not gone into
  // PLAYING. Expliclity set state to PLAYING and keep watching
  // state
  gst_element_set_state (GST_ELEMENT (src_bin), GST_STATE_PLAYING);

  return TRUE;
}

static gboolean
reset_ipc_source_pipeline (gpointer data)
{
  GstDsNvUriSrcBin *src_bin = (GstDsNvUriSrcBin *) data;
  gboolean ret = gst_element_send_event (GST_ELEMENT (src_bin->src_elem), gst_event_new_eos ());
  return ret;
}

static gboolean
reset_source_pipeline (gpointer data)
{
  GstDsNvUriSrcBin *src_bin = (GstDsNvUriSrcBin *) data;
  GstState state = GST_STATE_NULL;
  GstState pending = GST_STATE_VOID_PENDING;
  GstStateChangeReturn ret ;

  if (gst_element_set_state (GST_ELEMENT (src_bin->src_elem),
          GST_STATE_NULL) == GST_STATE_CHANGE_FAILURE) {
    GST_ERROR_OBJECT (src_bin, "Can't set source bin to NULL");
    return FALSE;
  }

  if (src_bin->depay && gst_element_set_state (GST_ELEMENT (src_bin->depay),
          GST_STATE_NULL) == GST_STATE_CHANGE_FAILURE) {
    GST_ERROR_OBJECT (src_bin, "Can't set source bin to NULL");
    return FALSE;
  }

  if (src_bin->adepay && gst_element_set_state (GST_ELEMENT (src_bin->adepay),
          GST_STATE_NULL) == GST_STATE_CHANGE_FAILURE) {
    GST_ERROR_OBJECT (src_bin, "Can't set source bin to NULL");
    return FALSE;
  }

  if (src_bin->decodebin && gst_element_set_state (GST_ELEMENT (src_bin->decodebin),
          GST_STATE_NULL) == GST_STATE_CHANGE_FAILURE) {
    GST_ERROR_OBJECT (src_bin, "Can't set source bin to NULL");
    return FALSE;
  }
  if (src_bin->cap_filter1) {
    gst_element_send_event (GST_ELEMENT (src_bin->cap_filter1),
                          gst_event_new_flush_start ());
    gst_element_send_event (GST_ELEMENT (src_bin->cap_filter1),
                          gst_event_new_flush_stop (TRUE));
  }
  g_print ("Resetting source %d, attempts: %d\n",src_bin->config->source_id,src_bin->config->num_rtsp_reconnects);

  g_mutex_lock (&src_bin->bin_lock);
  gettimeofday (&src_bin->last_buffer_time, NULL);
  gettimeofday (&src_bin->last_reconnect_time, NULL);
  g_mutex_unlock (&src_bin->bin_lock);

  if (src_bin->decodebin && gst_element_set_state (GST_ELEMENT (src_bin->decodebin),
          GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
    GST_ERROR_OBJECT (src_bin, "Can't set source bin to PLAYING");
    return FALSE;
  }

  if (src_bin->adepay && gst_element_set_state (GST_ELEMENT (src_bin->adepay),
          GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
    GST_ERROR_OBJECT (src_bin, "Can't set source bin to PLAYING");
    return FALSE;
  }

  if (src_bin->depay && gst_element_set_state (GST_ELEMENT (src_bin->depay),
          GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
    GST_ERROR_OBJECT (src_bin, "Can't set source bin to PLAYING");
    return FALSE;
  }

  if (gst_element_set_state (GST_ELEMENT (src_bin->src_elem),
          GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
    GST_ERROR_OBJECT (src_bin, "Can't set source bin to PLAYING");
    return FALSE;
  }

  if (src_bin->parser && !gst_element_send_event (GST_ELEMENT (src_bin->parser),
          gst_nvevent_new_stream_reset (0)))
    GST_ERROR_OBJECT (src_bin->parser,
        "Interrupted, Reconnection event not sent\n");

  ret = gst_element_get_state (GST_ELEMENT (src_bin), &state, &pending, 0);

  GST_DEBUG_OBJECT (src_bin,
      "Bin %s %p: state:%s pending:%s ret:%s", GST_ELEMENT_NAME (src_bin),
      src_bin, gst_element_state_get_name (state),
      gst_element_state_get_name (pending),
      gst_element_state_change_return_get_name (ret));

  if (ret == GST_STATE_CHANGE_ASYNC || ret == GST_STATE_CHANGE_NO_PREROLL) {
    if (!src_bin->async_state_watch_running)
      g_timeout_add (20, watch_source_async_state_change, src_bin);
    src_bin->async_state_watch_running = TRUE;
    src_bin->reconfiguring = TRUE;
  } else if (ret == GST_STATE_CHANGE_SUCCESS && state == GST_STATE_PLAYING) {
    src_bin->reconfiguring = FALSE;
  }
  return FALSE;
}

/* Returning FALSE from this callback will make rtspsrc ignore the stream.
 * Ignore audio and add the proper depay element based on codec. */
static gboolean
cb_rtspsrc_select_stream (GstElement * rtspsrc, guint num, GstCaps * caps,
    gpointer user_data)
{
  GstStructure *str = gst_caps_get_structure (caps, 0);
  GstDsNvUriSrcBin *src_bin = (GstDsNvUriSrcBin *) user_data;
  const gchar *media = gst_structure_get_string (str, "media");

  gboolean is_video = (!g_strcmp0 (media, "video"));
  gboolean is_audio = (!g_strcmp0 (media, "audio"));

  if (is_video || is_audio){
      src_bin->config->rtsp_reconnect_interval_sec = src_bin->config->rtsp_reconnect_interval_sec_org;
  }
  return (is_video || is_audio);
}

static void
cb_newpad_aparsebin (GstElement * decodebin, GstPad * pad, gpointer data)
{
  GstDsNvUriSrcBin *bin = (GstDsNvUriSrcBin *) data;

  if (!bin->atee) {
    bin->atee = gst_element_factory_make ("tee", "atee");
    if (!bin->atee) {
      GST_ELEMENT_ERROR (bin, STREAM, FAILED,
          ("Failed to create 'decodebin'"), (NULL));
      return;
    }
    gst_bin_add (GST_BIN (bin), bin->atee);

    if (!gst_element_sync_state_with_parent (bin->atee)) {
      GST_ELEMENT_ERROR (bin, RESOURCE, FAILED,
          ("Failed to sync child states"), (NULL));
      return;
    }
  }

  if (!bin->adecodebin) {
    bin->aqueue = gst_element_factory_make ("queue", "aqueue");
    if (!bin->aqueue) {
      GST_ELEMENT_ERROR (bin, STREAM, FAILED,
          ("Failed to create 'queue'"), (NULL));
      return;
    }

    bin->adecodebin = gst_element_factory_make ("decodebin", "adecodebin");
    if (!bin->adecodebin) {
      GST_ELEMENT_ERROR (bin, STREAM, FAILED,
          ("Failed to create 'decodebin'"), (NULL));
      return;
    }
    gst_bin_add_many (GST_BIN (bin), bin->aqueue, bin->adecodebin, NULL);

    if (!gst_element_sync_state_with_parent (bin->adecodebin) ||
        !gst_element_sync_state_with_parent (bin->aqueue)) {
      GST_ELEMENT_ERROR (bin, RESOURCE, FAILED,
          ("Failed to sync child states"), (NULL));
      return;
    }

    g_signal_connect (G_OBJECT (bin->adecodebin), "pad-added",
        G_CALLBACK (cb_newpad), bin);

    if (!gst_element_link_many (bin->atee, bin->aqueue, bin->adecodebin, NULL)) {
      GST_ELEMENT_ERROR (bin, RESOURCE, FAILED,
          ("Failed to link audio bin elements"), (NULL));
      return;
    }
  }

  if (bin->recordCtx && (bin->config->smart_rec_mode == 0 || bin->config->smart_rec_mode == 2)) {
    link_element_to_tee_src_pad (bin->atee, bin->recordCtx->recordbin, "asink");
  }

  GstPadUPtr sinkpad = gst_element_get_static_pad (bin->atee, "sink");
  if (gst_pad_link (pad, sinkpad) != GST_PAD_LINK_OK) {
    GST_ELEMENT_ERROR (bin, RESOURCE, FAILED,
        ("Failed to link audio bin elements"), (NULL));
  }
}

static gboolean
populate_rtspsrc_bin_audio (GstDsNvUriSrcBin * bin, GstPad * pad,
    const char *encoding_name)
{
  if (!bin->aparsebin) {
    bin->aparsebin = gst_element_factory_make ("parsebin", "aparsebin");
    if (!bin->aparsebin) {
      GST_ELEMENT_ERROR (bin, STREAM, FAILED,
          ("Failed to create 'decodebin'"), (NULL));
      return FALSE;
    }
    gst_bin_add (GST_BIN (bin), bin->aparsebin);
    if (!gst_element_sync_state_with_parent (bin->aparsebin)) {
      GST_ELEMENT_ERROR (bin, RESOURCE, FAILED,
          ("Failed to sync child states"), (NULL));
      return FALSE;
    }

    g_signal_connect (G_OBJECT (bin->aparsebin), "pad-added",
        G_CALLBACK (cb_newpad_aparsebin), bin);

    g_signal_connect (G_OBJECT (bin->aparsebin), "child-added",
        G_CALLBACK (aparsebin_child_added), bin);
  }

  GstPadUPtr sinkpad = gst_element_get_static_pad (bin->aparsebin, "sink");
  if (gst_pad_link (pad, sinkpad) != GST_PAD_LINK_OK) {
    return FALSE;
  }
  bin->audio_elem_populated = TRUE;
  return TRUE;
}

static gboolean
populate_rtspsrc_bin_video (GstDsNvUriSrcBin * bin, GstPad * pad,
    const char *encoding_name)
{
  GstCapsUPtr caps = NULL;
  GstCapsFeatures *feature = NULL;

  if (bin->video_elem_populated) {
    GstPadUPtr sinkpad = gst_element_get_static_pad (bin->depay, "sink");
    return gst_pad_link (pad, sinkpad) == GST_PAD_LINK_OK;
  }

  /* Create and add depay element only if it is not created yet. */
  if (!bin->depay) {
    /* Add the proper depay element based on codec. */
    if (!g_strcmp0 (encoding_name, "H264")) {
      bin->depay = gst_element_factory_make ("rtph264depay", "depay");
      bin->parser = gst_element_factory_make ("h264parse", "parser");
    } else if (!g_strcmp0 (encoding_name, "H265")) {
      bin->depay = gst_element_factory_make ("rtph265depay", "depay");
      bin->parser = gst_element_factory_make ("h265parse", "parser");
    } else if (!g_strcmp0 (encoding_name, "MP4V-ES")) {
      bin->depay = gst_element_factory_make ("rtpmp4vdepay", "depay");
      bin->parser = gst_element_factory_make ("mpeg4videoparse", "parser");
    } else {
      GST_ELEMENT_ERROR (bin, STREAM, FAILED, ("%s not supported",
              encoding_name), (NULL));
      return FALSE;
    }

    if (!bin->depay) {
      GST_ELEMENT_ERROR (bin, STREAM, FAILED, ("Failed to create 'depay'"),
          (NULL));
      return FALSE;
    }

    gst_bin_add_many (GST_BIN (bin), bin->depay, bin->parser, NULL);

    NVGSTDS_LINK_ELEMENT (bin->depay, bin->parser, FALSE);

    if (!gst_element_sync_state_with_parent (bin->depay)) {
      GST_ELEMENT_ERROR (bin, STREAM, FAILED,
          ("'depay' failed to sync state with parent"), (NULL));
      return FALSE;
    }
    if (!gst_element_sync_state_with_parent (bin->parser)) {
      GST_ELEMENT_ERROR (bin, STREAM, FAILED,
          ("'parser' failed to sync state with parent"), (NULL));
      return FALSE;
    }
  }

  GstPadUPtr sinkpad = gst_element_get_static_pad (bin->depay, "sink");
  if (gst_pad_link (pad, sinkpad) != GST_PAD_LINK_OK) {
    return FALSE;
  }

  bin->tee_rtsp_pre_decode =
      gst_element_factory_make ("tee", "tee_rtsp_pre_decode");
  if (!bin->tee_rtsp_pre_decode) {
    GST_ELEMENT_ERROR (bin, RESOURCE, FAILED,
        ("Failed to create 'tee_rtsp'"), (NULL));
    return FALSE;
  }

  bin->tee_rtsp_post_decode =
      gst_element_factory_make ("tee", "tee_rtsp_post_decode");
  if (!bin->tee_rtsp_post_decode) {
    GST_ELEMENT_ERROR (bin, RESOURCE, FAILED,
        ("Failed to create 'tee_rtsp'"), (NULL));
    return FALSE;
  }

  bin->dec_que = gst_element_factory_make ("queue", "dec_que");

  g_object_set (G_OBJECT (bin->dec_que), "leaky", bin->config->leaky, NULL);
  g_object_set (G_OBJECT (bin->dec_que), "max-size-buffers", bin->config->max_size_buffers, NULL);

  if (!bin->dec_que) {
    GST_ELEMENT_ERROR (bin, RESOURCE, FAILED,
        ("Failed to create 'queue'"), (NULL));
    return FALSE;
  }

  if (bin->config->rtsp_reconnect_interval_sec > 0) {
    NVGSTDS_ELEM_ADD_PROBE (bin,
        bin->depay, "src", rtspsrc_monitor_probe_func,
        GST_PAD_PROBE_TYPE_BUFFER | GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM, bin);
  }

  bin->decodebin = gst_element_factory_make ("decodebin", "decodebin");
  if (!bin->decodebin) {
    GST_ELEMENT_ERROR (bin, STREAM, FAILED,
        ("Failed to create 'decodebin'"), (NULL));
    return FALSE;
  }

  g_signal_connect (G_OBJECT (bin->decodebin), "child-added",
      G_CALLBACK (decodebin_child_added), bin);


  bin->cap_filter = gst_element_factory_make ("queue", "queue");
  if (!bin->cap_filter) {
    GST_ELEMENT_ERROR (bin, STREAM, FAILED,
        ("Failed to create 'queue'"), (NULL));
    return FALSE;
  }

  g_signal_connect (G_OBJECT (bin->decodebin), "pad-added",
      G_CALLBACK (cb_newpad2), bin);

  bin->nvvidconv = gst_element_factory_make ("nvvideoconvert", "nvvidconv");
  g_object_set (G_OBJECT (bin->nvvidconv), "disable-passthrough", bin->config->disable_passthrough, NULL);
  g_object_set (G_OBJECT (bin->nvvidconv), "gpu-id", bin->config->gpu_id, NULL);

  if (!bin->nvvidconv) {
    GST_ELEMENT_ERROR (bin, STREAM, FAILED,
        ("Could not create element 'nvvidconv_elem'"), (NULL));
    return FALSE;
  }
  caps = gst_caps_new_empty_simple ("video/x-raw");
  feature = gst_caps_features_new ("memory:NVMM", NULL);
  gst_caps_set_features (caps, 0, feature);

  bin->cap_filter1 =
      gst_element_factory_make ("capsfilter", "src_cap_filter_nvvidconv");
  if (!bin->cap_filter1) {
    GST_ELEMENT_ERROR (bin, STREAM, FAILED,
        ("Could not create 'queue'"), (NULL));
    return FALSE;
  }

  g_object_set (G_OBJECT (bin->cap_filter1), "caps", caps.get (), NULL);

  gst_bin_add_many (GST_BIN (bin),
      bin->tee_rtsp_pre_decode, bin->dec_que,
      bin->tee_rtsp_post_decode, bin->decodebin,
      bin->cap_filter, bin->nvvidconv, bin->cap_filter1, NULL);

  if (!gst_element_sync_state_with_parent (bin->tee_rtsp_pre_decode) ||
      !gst_element_sync_state_with_parent (bin->dec_que) ||
      !gst_element_sync_state_with_parent (bin->tee_rtsp_post_decode) ||
      !gst_element_sync_state_with_parent (bin->decodebin) ||
      !gst_element_sync_state_with_parent (bin->cap_filter) ||
      !gst_element_sync_state_with_parent (bin->nvvidconv) ||
      !gst_element_sync_state_with_parent (bin->cap_filter1)) {
    GST_ELEMENT_ERROR (bin, STREAM, FAILED,
        ("Failed to sync child states"), (NULL));
    return FALSE;
  }

  NVGSTDS_LINK_ELEMENT (bin->parser, bin->tee_rtsp_pre_decode, FALSE);

  link_element_to_tee_src_pad (bin->tee_rtsp_pre_decode, bin->dec_que);
  NVGSTDS_LINK_ELEMENT (bin->dec_que, bin->decodebin, FALSE);

  if (bin->recordCtx && (bin->config->smart_rec_mode == 0 || bin->config->smart_rec_mode == 1))
    link_element_to_tee_src_pad (bin->tee_rtsp_pre_decode,
        bin->recordCtx->recordbin);


  link_element_to_tee_src_pad (bin->tee_rtsp_post_decode, bin->cap_filter);
  NVGSTDS_LINK_ELEMENT (bin->cap_filter, bin->nvvidconv, FALSE);
  NVGSTDS_LINK_ELEMENT (bin->nvvidconv, bin->cap_filter1, FALSE);

  GstPadUPtr target_pad = gst_element_get_static_pad (bin->cap_filter1, "src");
  GstPad *src_pad = gst_ghost_pad_new_from_template ("vsrc_0", target_pad,
      gst_static_pad_template_get (&gst_nvurisrc_bin_vsrc_template));

  gst_pad_add_probe (src_pad, GST_PAD_PROBE_TYPE_QUERY_BOTH,
      src_pad_query_probe, bin, NULL);

  gst_pad_set_active (src_pad, TRUE);
  gst_element_add_pad (GST_ELEMENT (bin), src_pad);

  bin->video_elem_populated = TRUE;
  return TRUE;
}

static void
cb_newpad_rtspsrc (GstElement * decodebin, GstPad * pad, gpointer data)
{
  GstCapsUPtr caps = gst_pad_query_caps (pad, NULL);
  const GstStructure *str = gst_caps_get_structure (caps, 0);
  const gchar *name = gst_structure_get_name (str);
  const gchar *media = gst_structure_get_string (str, "media");
  const gchar *encoding_name = gst_structure_get_string (str, "encoding-name");
  GstDsNvUriSrcBin *bin = (GstDsNvUriSrcBin *) data;

  if (g_strrstr (name, "x-rtp") && !g_strcmp0 (media, "video")) {
    if (!populate_rtspsrc_bin_video (bin, pad, encoding_name)) {
      GST_ELEMENT_ERROR (bin, STREAM, FAILED,
          ("Failed to populate and link video elements"), (NULL));
    }
  }
  if (g_strrstr (name, "x-rtp") && !g_strcmp0 (media, "audio")
      && !bin->config->disable_audio) {
    if (!populate_rtspsrc_bin_audio (bin, pad, encoding_name)) {
      GST_ELEMENT_ERROR (bin, STREAM, FAILED,
          ("Failed to populate and link audio elements"), (NULL));
    }
  }
}

/**
 * Probe function to monitor data output from rtspsrc.
 */
static GstPadProbeReturn
rtspsrc_monitor_probe_func (GstPad * pad, GstPadProbeInfo * info,
    gpointer u_data)
{
  GstDsNvUriSrcBin *bin = (GstDsNvUriSrcBin *) u_data;
  if (info->type & GST_PAD_PROBE_TYPE_BUFFER) {
    g_mutex_lock (&bin->bin_lock);
    gettimeofday (&bin->last_buffer_time, NULL);
    bin->config->num_rtsp_reconnects = 0;
    g_mutex_unlock (&bin->bin_lock);
  }
  if (info->type & GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM) {
    if (GST_EVENT_TYPE (info->data) == GST_EVENT_EOS) {
      return GST_PAD_PROBE_DROP;
    }
  }
  return GST_PAD_PROBE_OK;
}

static void
cb_newpad2 (GstElement * decodebin, GstPad * pad, gpointer data)
{
  GstCapsUPtr caps = gst_pad_query_caps (pad, NULL);
  const GstStructure *str = gst_caps_get_structure (caps, 0);
  const gchar *name = gst_structure_get_name (str);

  if (!strncmp (name, "video", 5)) {
    GstDsNvUriSrcBin *bin = (GstDsNvUriSrcBin *) data;
    GstPadUPtr sinkpad =
        gst_element_get_static_pad (bin->tee_rtsp_post_decode, "sink");
    if (gst_pad_link (pad, sinkpad) != GST_PAD_LINK_OK) {
      GST_ELEMENT_ERROR (bin, STREAM, FAILED,
          ("Failed to link decodebin to pipeline"), (NULL));
    } else {
      GST_DEBUG_OBJECT (bin, "Decodebin linked to pipeline");
    }
  }
}

static GstPadProbeReturn
src_pad_query_probe (GstPad * pad, GstPadProbeInfo * info, gpointer data)
{
  GstDsNvUriSrcBin *bin = (GstDsNvUriSrcBin *) data;
  if (info->type & GST_PAD_PROBE_TYPE_QUERY_BOTH) {
    GstQuery *query = GST_QUERY (info->data);
    if (gst_nvquery_is_uri_from_streamid (query)) {
      gst_nvquery_uri_from_streamid_set (query, bin->config->uri);
      return GST_PAD_PROBE_HANDLED;
    }

    if (gst_nvquery_is_sourceid (query) && bin->config->source_id >= 0) {
      gst_nvquery_sourceid_set (query, bin->config->source_id);
      return GST_PAD_PROBE_HANDLED;
    }
  }
  return GST_PAD_PROBE_OK;
}

static gboolean
populate_ipc_bin (GstDsNvUriSrcBin * nvurisrcbin)
{
  const char *prefix = "ipc://";
  const char *ipc_socket_path = NULL;

  nvurisrcbin->src_elem =
      gst_element_factory_make ("nvunixfdsrc", NULL);
  if (!nvurisrcbin->src_elem) {
        GST_ELEMENT_ERROR (nvurisrcbin, RESOURCE, FAILED,
        ("Failed to create 'nvunixfdsrc'"), (NULL));
    return FALSE;
  }

  if (nvurisrcbin->config->ipc_socket_path) {
    ipc_socket_path = nvurisrcbin->config->ipc_socket_path;
    g_print ("Socket path from config, Setting Socket path %s\n",
              nvurisrcbin->config->ipc_socket_path);
  } else {
    ipc_socket_path = nvurisrcbin->config->uri + strlen(prefix);
    g_print ("Socket path from uri, Setting Socket path %s\n", ipc_socket_path);
  }

  if (nvurisrcbin->config->ipc_connection_attempts != DEFAULT_IPC_CONNECTION_ATTEMPTS) {
    g_object_set (G_OBJECT(nvurisrcbin->src_elem), "connection-attempts",
                  nvurisrcbin->config->ipc_connection_attempts,
                  NULL);
    g_print ("Setting Connection Attempts %d\n",
              nvurisrcbin->config->ipc_connection_attempts);
  }

  if (nvurisrcbin->config->ipc_connection_interval != DEFAULT_IPC_CONNECTION_INTERVAL) {
    g_object_set (G_OBJECT(nvurisrcbin->src_elem), "connection-interval",
                  nvurisrcbin->config->ipc_connection_interval,
                  NULL);
    g_print ("Setting Connection Interval %ld\n",
              nvurisrcbin->config->ipc_connection_interval);
  }
  g_object_set (G_OBJECT(nvurisrcbin->src_elem), "socket-path", ipc_socket_path,
                NULL);
  g_object_set (G_OBJECT(nvurisrcbin->src_elem), "buffer-timestamp-copy",
                nvurisrcbin->config->ipc_buffer_timestamp_copy,
                NULL);
  GST_DEBUG_OBJECT (nvurisrcbin, "IPC socket path = %s\n", ipc_socket_path);

  nvurisrcbin->cap_filter1 =
      gst_element_factory_make ("capsfilter", "src_caps_filter_nvunixfdsrc");
  if (!nvurisrcbin->cap_filter1) {
    GST_ELEMENT_ERROR (nvurisrcbin, STREAM, FAILED,
        ("Could not create 'src_caps_filter_nvunixfdsrc'"), (NULL));
    return FALSE;
  }

  GstCaps* caps = gst_caps_new_simple ("video/x-raw",
            "format", G_TYPE_STRING, "NV12", NULL);
  GstCapsFeatures *feature = NULL;
  feature = gst_caps_features_new ("memory:NVMM", NULL);
  gst_caps_set_features (caps, 0, feature);

  g_object_set (G_OBJECT (nvurisrcbin->cap_filter1), "caps", caps, NULL);
  gst_caps_unref (caps);
  gst_bin_add_many (GST_BIN (nvurisrcbin),
      nvurisrcbin->src_elem, nvurisrcbin->cap_filter1, NULL);


  NVGSTDS_LINK_ELEMENT (nvurisrcbin->src_elem, nvurisrcbin->cap_filter1, FALSE);

  GstPadUPtr target_pad = gst_element_get_static_pad (nvurisrcbin->cap_filter1, "src");
  GstPad *src_pad = gst_ghost_pad_new_from_template ("vsrc_0", target_pad,
      gst_static_pad_template_get (&gst_nvurisrc_bin_vsrc_template));

  gst_pad_add_probe (src_pad, GST_PAD_PROBE_TYPE_QUERY_BOTH,
      src_pad_query_probe, nvurisrcbin, NULL);

  gst_pad_set_active (src_pad, TRUE);
  gst_element_add_pad (GST_ELEMENT (nvurisrcbin), src_pad);

  nvurisrcbin->video_elem_populated = TRUE;
if (!gst_element_sync_state_with_parent (nvurisrcbin->src_elem)
      || !gst_element_sync_state_with_parent (nvurisrcbin->cap_filter1)) {
    GST_ELEMENT_ERROR (nvurisrcbin, RESOURCE, FAILED,
        ("Failed to sync child states"), (NULL));
    return FALSE;
  }

  return TRUE;
}

static gboolean
populate_rtsp_bin (GstDsNvUriSrcBin * nvurisrcbin)
{
  GstDsNvUriSrcConfig *config = nvurisrcbin->config;

  nvurisrcbin->src_elem = gst_element_factory_make ("rtspsrc", "src");
  if (!nvurisrcbin->src_elem) {
    GST_ELEMENT_ERROR (nvurisrcbin, RESOURCE, FAILED,
        ("Failed to create 'rtspsrc'"), (NULL));
    return FALSE;
  }

  g_signal_connect (G_OBJECT (nvurisrcbin->src_elem), "select-stream",
      G_CALLBACK (cb_rtspsrc_select_stream), nvurisrcbin);

  g_object_set (G_OBJECT (nvurisrcbin->src_elem), "location", config->uri,
      NULL);
  g_object_set (G_OBJECT (nvurisrcbin->src_elem), "latency", config->latency,
      NULL);
  g_object_set (G_OBJECT (nvurisrcbin->src_elem), "udp-buffer-size", config->udp_buffer_size,
      NULL);
  g_object_set (G_OBJECT (nvurisrcbin->src_elem), "drop-on-latency", config->drop_on_latency,
      NULL);
  g_object_set (G_OBJECT (nvurisrcbin->src_elem), "buffer-mode", config->buffer_mode,
      NULL);

  configure_source_for_ntp_sync (nvurisrcbin->src_elem);

  guint gst_protocol = 0;
  switch (config->rtp_protocol) {
    case RTP_PROTOCOL_UDP:
      gst_protocol = GST_RTSP_LOWER_TRANS_UDP;
      break;
    case RTP_PROTOCOL_UDP_MCAST:
      gst_protocol = GST_RTSP_LOWER_TRANS_UDP_MCAST;
      break;
    case RTP_PROTOCOL_TCP:
      gst_protocol = GST_RTSP_LOWER_TRANS_TCP;
      break;
    case RTP_PROTOCOL_UDP_UDPMCAST_TCP:
      gst_protocol = GST_RTSP_LOWER_TRANS_UDP | GST_RTSP_LOWER_TRANS_UDP_MCAST | GST_RTSP_LOWER_TRANS_TCP;
      break;
    case RTP_PROTOCOL_HTTP:
      gst_protocol = GST_RTSP_LOWER_TRANS_HTTP;
      break;
    case RTP_PROTOCOL_TLS:
      gst_protocol = GST_RTSP_LOWER_TRANS_TLS;
      break;
    default:
      GST_WARNING_OBJECT (nvurisrcbin->src_elem,
          "Unknown RTP protocol value: %d, using default", config->rtp_protocol);
      gst_protocol = GST_RTSP_LOWER_TRANS_UDP | GST_RTSP_LOWER_TRANS_UDP_MCAST | GST_RTSP_LOWER_TRANS_TCP;
      break;
  }

  g_object_set (G_OBJECT (nvurisrcbin->src_elem), "protocols", gst_protocol, NULL);
  GST_DEBUG_OBJECT (nvurisrcbin->src_elem,
      "RTP Protocol=0x%x (custom=%d, gst=0x%x)----\n",
      gst_protocol, config->rtp_protocol, gst_protocol);
  g_print("Protocol set: 0x%x\n", gst_protocol);

  g_signal_connect (G_OBJECT (nvurisrcbin->src_elem), "pad-added",
      G_CALLBACK (cb_newpad_rtspsrc), nvurisrcbin);

  gst_bin_add (GST_BIN (nvurisrcbin), nvurisrcbin->src_elem);


  if (config->smart_record) {
    NvDsSRInitParams params = { 0 };
    if (!config->smart_rec_dir_path) {
      GST_ELEMENT_ERROR (nvurisrcbin, LIBRARY, SETTINGS,
          ("Invalid path specified for %s ", "smart-rec-dir-path"), (NULL));
      return FALSE;
    }
    params.containerType = (NvDsSRContainerType) config->smart_rec_container;
    if (config->smart_rec_file_prefix)
      params.fileNamePrefix =
          g_strdup_printf ("%s_%d", config->smart_rec_file_prefix,
          config->source_id);
    params.dirpath = config->smart_rec_dir_path;
    params.cacheSize = config->smart_rec_cache_size;
    params.defaultDuration = config->smart_rec_def_duration;
    params.callback = smart_record_callback;
    if (NvDsSRCreate (&nvurisrcbin->recordCtx, &params) != NVDSSR_STATUS_OK) {
      GST_ELEMENT_ERROR (nvurisrcbin, RESOURCE, FAILED,
          ("Failed to create smart record bin"), (NULL));
      g_free (params.fileNamePrefix);
      return FALSE;
    }
    g_free (params.fileNamePrefix);
    gst_bin_add (GST_BIN (nvurisrcbin), nvurisrcbin->recordCtx->recordbin);
    if (!gst_element_sync_state_with_parent (nvurisrcbin->recordCtx->recordbin)) {
      GST_ELEMENT_ERROR (nvurisrcbin, STREAM, FAILED,
          ("Failed to sync child states"), (NULL));
      return FALSE;
    }
  }

  if (config->rtsp_reconnect_interval_sec > 0) {
    nvurisrcbin->source_watch_id =
        g_timeout_add (1000, watch_source_status, nvurisrcbin);
    g_mutex_lock (&nvurisrcbin->bin_lock);
    gettimeofday (&nvurisrcbin->last_buffer_time, NULL);
    gettimeofday (&nvurisrcbin->last_reconnect_time, NULL);
    g_mutex_unlock (&nvurisrcbin->bin_lock);
  }

  GST_DEBUG_OBJECT (nvurisrcbin,
      "Decode bin created. Waiting for a new pad from decodebin to link");

  return TRUE;
}

static inline void
s_remove_all_urisrcbin_children (GstBin * bin)
{
  GstIterator *it = gst_bin_iterate_elements (bin);
  GValue elem = G_VALUE_INIT;
  while (gst_iterator_next (it, &elem) == GST_ITERATOR_OK) {
    g_object_ref( GST_ELEMENT (g_value_get_object (&elem)));
    gst_bin_remove (bin, GST_ELEMENT (g_value_get_object (&elem)));

    GstStateChangeReturn state_return = GST_STATE_CHANGE_FAILURE;

    if ((state_return = gst_element_set_state ( GST_ELEMENT (g_value_get_object (&elem)),
              GST_STATE_NULL)) == GST_STATE_CHANGE_FAILURE) {
      g_print("Unable to set state to NULL, element : %s \n",GST_ELEMENT_NAME((g_value_get_object (&elem))));
    }
    g_object_unref( GST_ELEMENT (g_value_get_object (&elem)));
    gst_iterator_resync (it);
  }
  gst_iterator_free (it);
}

static GstStateChangeReturn
gst_ds_nvurisrc_bin_change_state (GstElement * element,
    GstStateChange transition)
{
  GstDsNvUriSrcBin *nvurisrcbin = GST_DS_NVURISRC_BIN (element);
  GstDsNvUriSrcConfig *config = nvurisrcbin->config;
  GstStateChangeReturn ret;

  if (transition == GST_STATE_CHANGE_NULL_TO_READY) {
    gboolean is_ipc = nvurisrcbin->config->uri
        && g_str_has_prefix (nvurisrcbin->config->uri, "ipc://");
    gboolean is_rtsp = nvurisrcbin->config->uri
        && g_str_has_prefix (nvurisrcbin->config->uri, "rtsp://");
    gboolean select_rtsp_mode = is_rtsp && (config->src_type == SOURCE_TYPE_RTSP
        || config->src_type == SOURCE_TYPE_AUTO);

    if (is_ipc)
    {
      if (!populate_ipc_bin (nvurisrcbin)) {
        return GST_STATE_CHANGE_FAILURE;
      }
    }
    else if (!select_rtsp_mode) {
      if (!populate_uri_bin (nvurisrcbin)) {
        return GST_STATE_CHANGE_FAILURE;
      }
    } else {
      if (!populate_rtsp_bin (nvurisrcbin)) {
        return GST_STATE_CHANGE_FAILURE;
      }
    }
  }


  ret = GST_ELEMENT_CLASS (parent_class)->change_state (element, transition);
  if (transition == GST_STATE_CHANGE_PAUSED_TO_READY) {
    nvurisrcbin->last_buffer_time = { 0, 0 };
  }

  if (transition == GST_STATE_CHANGE_READY_TO_NULL) {
    if (nvurisrcbin->source_watch_id) {
      if (nvurisrcbin->config->rtsp_reconnect_attempts == -1 ||
           nvurisrcbin->config->num_rtsp_reconnects <=
           nvurisrcbin->config->rtsp_reconnect_attempts){
        g_source_remove (nvurisrcbin->source_watch_id);
      }
    }
    nvurisrcbin->source_watch_id = 0;

    nvurisrcbin->src_elem=NULL;
    nvurisrcbin->cap_filter=NULL;
    nvurisrcbin->cap_filter1=NULL;
    nvurisrcbin->depay=NULL;
    nvurisrcbin->parser=NULL;
    nvurisrcbin->dec_que=NULL;
    nvurisrcbin->decodebin=NULL;
    nvurisrcbin->tee=NULL;
    nvurisrcbin->tee_rtsp_pre_decode=NULL;
    nvurisrcbin->tee_rtsp_post_decode=NULL;
    nvurisrcbin->fakesink_queue=NULL;
    nvurisrcbin->fakesink=NULL;
    nvurisrcbin->nvvidconv=NULL;
    nvurisrcbin->adepay=NULL;
    nvurisrcbin->aqueue=NULL;
    nvurisrcbin->aparsebin=NULL;
    nvurisrcbin->atee=NULL;
    nvurisrcbin->adecodebin=NULL;
    nvurisrcbin->audio_convert=NULL;
    nvurisrcbin->audio_resample=NULL;

    s_remove_all_urisrcbin_children (GST_BIN (nvurisrcbin));

    nvurisrcbin->depay = NULL;
    nvurisrcbin->tee = NULL;

    nvurisrcbin->video_elem_populated = FALSE;
    nvurisrcbin->audio_elem_populated = FALSE;

    if (nvurisrcbin->config->ipc_socket_path) {
      g_free(nvurisrcbin->config->ipc_socket_path);
      nvurisrcbin->config->ipc_socket_path = NULL;
    }

  }
  return ret;
}

/* Package and library details required for plugin_init */
#define PACKAGE "DeepStream SDK nvurisrcbin Bin"
#define LICENSE "Proprietary"
#define DESCRIPTION "Deepstream SDK nvurisrcbin Bin"
#define BINARY_PACKAGE "Deepstream SDK nvurisrcbin Bin"
#define URL "http://nvidia.com/"

static gboolean
nvurisrcbin_plugin_init (GstPlugin * plugin)
{
  if (!gst_element_register (plugin, "nvurisrcbin", GST_RANK_PRIMARY,
          GST_TYPE_DS_NVURISRC_BIN))
    return FALSE;
  return TRUE;
}
GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    nvdsgst_nvurisrcbin,
    DESCRIPTION, nvurisrcbin_plugin_init, "9.0", LICENSE,
    BINARY_PACKAGE, URL)

