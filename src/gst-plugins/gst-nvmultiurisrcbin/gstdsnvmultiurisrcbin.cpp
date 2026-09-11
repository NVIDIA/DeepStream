/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gst/audio/audio.h>
#include "gstdsnvmultiurisrcbin.h"
#include "nvdsgstutils.h"
#include "gst-nvcommon.h"
#include "gst-nvquery.h"
#include "gst-nvevent.h"
#include "gst-nvmessage.h"
#include "gst-nvdscustommessage.h"
#include "gst-nvdscustomevent.h"
#include "gst-nvcustomevent.h"
#include "nvds_rest_metrics.h"
#include "nvds_msgapi.h"
#include "dlfcn.h"

#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <ifaddrs.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include "nvds_rest_server.h"
#include "nvbufsurface.h"
#include "nvbufsurftransform.h"

#include "nvds_stats.h"
#include "nvds_utils.h"
#include <curl/curl.h>

#include <string>

#define HTTP_DOWNLOAD_DIR "/opt/nvidia/deepstream/deepstream/samples/streams/"

/* HTTP(S) file-download (curl) timeouts, in seconds. Exposed as the
 * "http-download-timeout" / "http-connect-timeout" properties; the defaults
 * preserve the previous hardcoded behavior. A value of 0 means "no limit". */
#define DEFAULT_HTTP_DOWNLOAD_TIMEOUT  300
#define DEFAULT_HTTP_CONNECT_TIMEOUT    30
/* Max concurrent HTTP(S) file downloads. 0 = use max-batch-size; any value is
 * clamped to max-batch-size (never fetch more files at once than the pipeline
 * can hold as active sources). */
#define DEFAULT_HTTP_MAX_CONCURRENT_DOWNLOADS 0

//Default prop values
#define DEFAULT_HTTP_IP "localhost"
#define DEFAULT_HTTP_PORT "9000"
#define DEFAULT_MAX_BATCH_SIZE (30)
#define DEFAULT_NUM_EXTRA_SURFACES 1
#define DEFAULT_GPU_DEVICE_ID 0
#define DEFAULT_DROP_FRAME_INTERVAL 0
#define DEFAULT_DROP_ON_LATENCY TRUE
#define DEFAULT_SOURCE_TYPE 0
#define DEFAULT_DEC_SKIP_FRAME_TYPE 0
#define DEFAULT_BUFFER_MODE 3
#define DEFAULT_LOW_LATENCY_MODE FALSE
#define DEFAULT_RTP_PROTOCOL 7
#define DEFAULT_LEAKY 0
#define DEFAULT_MAX_SIZE_BUFFERS 200
#define DEFAULT_LATENCY 100
#define DEFAULT_FILE_LOOP FALSE
#define DEFAULT_DISABLE_PASSTHROUGH FALSE
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
#define SOURCE_RESET_INTERVAL_SEC 60
#define DEFAULT_SEI_EXTRACT_DATA FALSE
#define DEFAULT_SEI_UUID NULL
#define DEFAULT_IPC_BUFFER_TIMESTAMP_COPY FALSE
#define DEFAULT_IPC_CONNECTION_ATTEMPTS -1
#define DEFAULT_IPC_CONNECTION_INTERVAL 1000000
#define DEFAULT_SENSORID_PADID_MAPPING FALSE
#define DEFAULT_SORT_BATCH_BUFFERS FALSE
#define DEFAULT_CACHE_BATCH_BUFFERS FALSE
#define DEFAULT_CACHED_BUFFER_TIMEOUT -1
#define DEFAULT_SEI_EXTRACT_SIM_TIME FALSE
#define DEFAULT_ALIGN_FIRST_BATCH FALSE
#define DEFAULT_SYNC_INPUTS_NTP 0 
#define DEFAULT_DROP_BACKWARD_SEI FALSE

/** nvstreammux props: */
#define DEFAULT_BATCH_METHOD 1
#define DEFAULT_BATCH_SIZE 0
#define DEFAULT_BATCHED_PUSH_TIMEOUT -1
#define DEFAULT_WIDTH 0
#define DEFAULT_HEIGHT 0
#define DEFAULT_QUERY_RESOLUTION FALSE
#define DEFAULT_GPU_DEVICE_ID 0
#define DEFAULT_LIVE_SOURCE FALSE
#define DEFAULT_ATTACH_SYS_TIME_STAMP TRUE
#define DEFAULT_ADAPTIVE_BATCH_SIZE FALSE
#define DEFAULT_ASYNC_PROCESS TRUE
#define DEFAULT_NO_PIPELINE_EOS FALSE
#define DEFAULT_FRAME_DURATION (0)
#define DEFAULT_BUFFER_POOL_SIZE (4)
#define MAX_NVBUFFERS 1024
#define MAX_POOL_BUFFERS (MAX_NVBUFFERS)
#define DEFAULT_CONFIG_FILE_PATH NULL

#define GST_TYPE_NVDSURI_SKIP_FRAMES (gst_nvdsurisrc_dec_skip_frames ())
#define GST_TYPE_NVDSURI_RTP_PROTOCOL (gst_nvdsurisrc_rtp_protocol ())
#define GST_TYPE_NVDSURI_LEAKY (gst_nvdsurisrc_leaky ())
#define GST_TYPE_NVDSURI_BUFFER_MODE (gst_nvdsurisrc_buffer_mode ())
#define GST_TYPE_NVDSURI_SMART_RECORD_TYPE (gst_nvdsurisrc_smart_record_type ())
#define GST_TYPE_NVDSURI_SMART_RECORD_MODE (gst_nvdsurisrc_smart_record_mode ())
#define GST_TYPE_NVDSURI_SMART_RECORD_CONTAINER (gst_nvdsurisrc_smart_record_container ())

#define GST_TYPE_V4L2_VID_CUDADEC_MEM_TYPE (gst_video_cudadec_mem_type ())
#define GST_TYPE_NVMULTIURISRCBIN_MODE (gst_nvmultiurisrcbin_mode ())
#define DEFAULT_CUDADEC_MEM_TYPE (0)
#define DEFAULT_NVBUF_MEM_TYPE (0)

#define DEFAULT_ENABLE_ERROR_PROPAGATION FALSE
typedef struct
{
  GstDsNvMultiUriBin *ubin;
  guint sourceId;
  gboolean didSourceElemError;
} DsNvMultiUriBinSourceInfo;
static gpointer GThreadFuncRemoveSource (gpointer data);
void free_sensor_info(gpointer data);

static void msgapi_cleanup();
static NvDsMsgApiHandle (*msgapi_connect_fn)(char*, nvds_msgapi_connect_cb_t, char*) = NULL;
static NvDsMsgApiErrorType (*msgapi_send_fn)(NvDsMsgApiHandle, char*, const uint8_t*, size_t) = NULL;
static NvDsMsgApiErrorType (*msgapi_disconnect_fn)(NvDsMsgApiHandle) = NULL;

static NvDsMsgApiHandle msgapi_conn_handle = NULL;
static void *msgapi_lib_handle = NULL;

static gboolean msgapi_init(GstDsNvMultiUriBin *nvmultiurisrcbin) {
  // Check if required properties are set
  if (!nvmultiurisrcbin->config->proto_lib || !nvmultiurisrcbin->config->conn_str) {
    GST_ERROR_OBJECT(nvmultiurisrcbin, "Initialization failed: proto-lib or conn-str not set\n");
    return FALSE;
  }

  // Load the MSGAPI library
  msgapi_lib_handle = dlopen(nvmultiurisrcbin->config->proto_lib, RTLD_LAZY);
  if (!msgapi_lib_handle) {
    GST_ERROR_OBJECT(nvmultiurisrcbin, "Failed to load library: %s\n", dlerror());
    GST_ERROR_OBJECT(nvmultiurisrcbin, "Make sure library exists at: %s\n", nvmultiurisrcbin->config->proto_lib);
    return FALSE;
  }
  // Resolve function symbols
  *(void**)(&msgapi_connect_fn) = dlsym(msgapi_lib_handle, "nvds_msgapi_connect");
  *(void**)(&msgapi_send_fn) = dlsym(msgapi_lib_handle, "nvds_msgapi_send");
  *(void**)(&msgapi_disconnect_fn) = dlsym(msgapi_lib_handle, "nvds_msgapi_disconnect");

  char *error = dlerror();
  if (error != NULL) {
    GST_ERROR_OBJECT(nvmultiurisrcbin, "Failed to resolve symbols: %s\n", error);
    dlclose(msgapi_lib_handle);
    msgapi_lib_handle = NULL;
    return FALSE;
  }
  msgapi_conn_handle = msgapi_connect_fn((char*)nvmultiurisrcbin->config->conn_str, NULL, NULL);
  if (!msgapi_conn_handle) {
    GST_ERROR_OBJECT(nvmultiurisrcbin, "Failed to connect to MSGAPI\n");
    return FALSE;
  }
  g_print("[MSGAPI] Connected to MSGAPI at %s...\n", nvmultiurisrcbin->config->conn_str);
  return TRUE;
}

/**
 * msgapi_send_message:
 * @sensor_id: The stream/sensor identifier
 * @source: Name of the component/pod reporting the error
 * @event: Detailed description of the error event
 * @is_default_topic: Not used (kept for backward compatibility)
 *
 * Sends an error event message to the MSGAPI broker with the following JSON schema:
 *
 * Error Event Message Schema:
 * {
 *   stream_id: "b0f05afe-cb0c-43fb-a0f3-e94bd437ab2e"  <string>,
 *   timestamp: "2023-04-13 07:32:51"                   <string>,
 *   type: "CRITICAL"                                   <enum("CRITICAL", "FUNCTIONAL")>,
 *   source: "<name of the pod reporting the error>"   <string>,
 *   event: "detailed description"                      <string>
 * }
 *
 * All error messages (stream-related and GStreamer-based) are sent to the
 * user-defined topic configured in the 'topic' property without any prefix.
 *
 * Returns: TRUE if message was sent successfully, FALSE otherwise
 */
static gboolean msgapi_send_message(GstDsNvMultiUriBin *nvmultiurisrcbin, const char *sensor_id, const char *source, const char *event, gboolean is_default_topic ) {
  if (!msgapi_send_fn) {
    GST_ERROR_OBJECT(nvmultiurisrcbin, "MSGAPI not initialized\n");
    g_print("[MSGAPI_ERROR] MSGAPI not initialized\n");
    return FALSE;
  }
  GDateTime *now = g_date_time_new_now_local();
  gchar *timestamp = g_date_time_format(now, "%Y-%m-%d %H:%M:%S");
  g_date_time_unref(now);
  gchar *stream_id = g_strdup(sensor_id);

  // Get hostname with IP if source is "N/A" or NULL
  gchar *source_str = NULL;
  gboolean free_source = FALSE;
  if (!source || g_strcmp0(source, "N/A") == 0) {
    source_str = nvds_get_hostname_with_ip();
    free_source = TRUE;
  } else {
    source_str = (gchar*)source;
  }

  gchar *error_message = g_strdup_printf(
    "{\n"
    "  \"stream_id\" : \"%s\",\n"
    "  \"timestamp\" : \"%s\",\n"
    "  \"type\" : \"CRITICAL\",\n"
    "  \"source\" : \"%s\",\n"
    "  \"event\" : \"%s\"\n"
    "}\n",
    stream_id,
    timestamp,
    source_str,
    event
  );
  // Use user-defined topic directly for all error messages (no prefix)
  gchar *topic_cstr = NULL;
  if (nvmultiurisrcbin->config->topic) {
    topic_cstr = g_strdup(nvmultiurisrcbin->config->topic);
  } else {
    topic_cstr = g_strdup("deepstream");
  }
  NvDsMsgApiErrorType result = msgapi_send_fn(msgapi_conn_handle, (char*)topic_cstr, (const uint8_t*)error_message, strlen(error_message));
  if (result != NVDS_MSGAPI_OK) {
    GST_ERROR_OBJECT(nvmultiurisrcbin, "Failed to send message\n");
    g_print("[MSGAPI_ERROR] Failed to send message\n");
    g_free(stream_id);
    g_free(timestamp);
    g_free(error_message);
    g_free(topic_cstr);
    if (free_source) {
      g_free(source_str);
    }
    return FALSE;
  }
  g_free(stream_id);
  g_free(timestamp);
  g_free(error_message);
  g_free(topic_cstr);
  if (free_source) {
    g_free(source_str);
  }
  return TRUE;
}

static void msgapi_cleanup() {
  if (msgapi_conn_handle && msgapi_disconnect_fn) {
    g_print("[MSGAPI] Disconnecting from MSGAPI broker...\n");
    msgapi_disconnect_fn(msgapi_conn_handle);
    msgapi_conn_handle = NULL;
  }
  if (msgapi_lib_handle) {
    g_print("[MSGAPI] Unloading MSGAPI library...\n");
    dlclose(msgapi_lib_handle);
    msgapi_lib_handle = NULL;
  }
}

static GType
gst_nvmultiurisrcbin_mode (void)
{
  static GType qtype = 0;

  if (qtype == 0) {
    static const GEnumValue values[] = {
      {0, "Mode Video-Only",
            "Video streams are muxed together; audio streams are ignored; Default"},
      {1, "Mode Audio-Only",
            "Audio streams are muxed together; video streams are ignored"},
      {0, NULL, NULL}
    };

    qtype = g_enum_register_static ("GstNvMultiUriSrcBinModeType2", values);
  }
  return qtype;
}

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

    qtype = g_enum_register_static ("GstNvUriSrcBinCudaDecMemType2", values);
  }
  return qtype;
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
    GType tmp = g_enum_register_static ("SkipFrames2", skip_type);
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
    GType tmp = g_enum_register_static ("RtpProtocol2", rtp_protocol);
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
    GType tmp = g_enum_register_static ("Leaky2", leaky);
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
    GType tmp = g_enum_register_static ("BufferMode2", buffer_mode);
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
    GType tmp = g_enum_register_static ("SmartRecordType2", smart_rec_type);
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
    GType tmp = g_enum_register_static ("SmartRecordMode2", smart_rec_mode);
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
    GType tmp = g_enum_register_static ("SmartRecordContainerType2",
        smart_rec_container);
    g_once_init_leave (&initialization_value, tmp);
  }
  return (GType) initialization_value;
}

#define COMMON_AUDIO_CAPS \
  "channels = " GST_AUDIO_CHANNELS_RANGE ", " \
  "rate = (int) [ 1, MAX ]"

static GstStaticPadTemplate gst_nvmultiurisrc_bin_src_template =
    GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES ("memory:NVMM",
            "{ " "NV12, RGBA, I420 }") "; "
        "audio/x-raw(memory:NVMM), "
        "format = { " "S16LE, F32LE }, "
        "layout = (string) interleaved, " COMMON_AUDIO_CAPS));

GST_DEBUG_CATEGORY (gst_ds_nvmultiurisrc_bin_debug);
#define GST_CAT_DEFAULT gst_ds_nvmultiurisrc_bin_debug

/* Define our element type. Standard GObject/GStreamer boilerplate stuff */
#define gst_ds_nvmultiurisrc_bin_parent_class parent_class
#define _do_init \
    GST_DEBUG_CATEGORY_INIT (gst_ds_nvmultiurisrc_bin_debug, "nvmultiurisrcbin", 0, "nvmultiurisrcbin element");
G_DEFINE_TYPE_WITH_CODE (GstDsNvMultiUriBin, gst_ds_nvmultiurisrc_bin,
    GST_TYPE_BIN, _do_init);

static void gst_ds_nvmultiurisrc_bin_set_property (GObject * object,
    guint prop_id, const GValue * value, GParamSpec * spec);
static void gst_ds_nvmultiurisrc_bin_get_property (GObject * object,
    guint prop_id, GValue * value, GParamSpec * spec);
static void gst_ds_nvmultiurisrc_bin_finalize (GObject * object);
static GstStateChangeReturn
gst_ds_nvmultiurisrc_bin_change_state (GstElement * element,
    GstStateChange transition);
static void
gst_ds_nvmultiurisrc_bin_handle_message (GstBin * bin, GstMessage * message);
static void rest_api_server_start (GstDsNvMultiUriBin * nvmultiurisrcbin);

static void
gst_ds_nvmultiurisrc_bin_class_init (GstDsNvMultiUriBinClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;
  GstBinClass *gstbin_class;

  gobject_class = G_OBJECT_CLASS (klass);
  gstelement_class = GST_ELEMENT_CLASS (klass);
  gstbin_class = GST_BIN_CLASS (klass);

  gobject_class->set_property =
      GST_DEBUG_FUNCPTR (gst_ds_nvmultiurisrc_bin_set_property);
  gobject_class->get_property =
      GST_DEBUG_FUNCPTR (gst_ds_nvmultiurisrc_bin_get_property);
  gobject_class->finalize =
      GST_DEBUG_FUNCPTR (gst_ds_nvmultiurisrc_bin_finalize);
  gstelement_class->change_state =
      GST_DEBUG_FUNCPTR (gst_ds_nvmultiurisrc_bin_change_state);
  gstbin_class->handle_message =
      GST_DEBUG_FUNCPTR (gst_ds_nvmultiurisrc_bin_handle_message);

  gst_element_class_add_static_pad_template (gstelement_class,
      &gst_nvmultiurisrc_bin_src_template);
  //gst_element_class_add_static_pad_template (gstelement_class,
  //  &gst_nvmultiurisrc_bin_asrc_template);

  g_object_class_install_property (gobject_class, MULTIURIBIN_PROP_URI_LIST,
      g_param_spec_string ("uri-list", "comma separated URI list of sources",
          "URI of the file or rtsp source", NULL,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class,
      MULTIURIBIN_PROP_SENSOR_ID_LIST, g_param_spec_string ("sensor-id-list",
          "comma separated list of source sensor IDs",
          "this vector is one to one mapped with the uri-list", NULL,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class,
      MULTIURIBIN_PROP_SENSOR_NAME_LIST, g_param_spec_string ("sensor-name-list",
          "optional comma separated list of source sensor names",
          "this vector is one to one mapped with the uri-list", NULL,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, MULTIURIBIN_PROP_MODE,
      g_param_spec_enum ("mode", "Video-only or Audio-only modes available",
          "Set Video-only or Audio-only",
          GST_TYPE_NVMULTIURISRCBIN_MODE, NVDS_MULTIURISRCBIN_MODE_VIDEO,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, MULTIURIBIN_PROP_DISABLE_AUDIO,
      g_param_spec_boolean ("disable-audio",
          "disable-audio",
          "Disable audio path at init time",
          DEFAULT_DISABLE_AUDIO,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, MULTIURIBIN_PROP_HTTP_IP,
      g_param_spec_string ("ip-address", "Set REST API HTTP IP Address",
          "REST API HTTP Endpoint", DEFAULT_HTTP_IP,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class,
      MULTIURIBIN_PROP_DISABLE_PASSTHROUGH,
      g_param_spec_boolean ("disable-passthrough", "disable-passthrough",
          "Disable passthrough mode at init time, applicable for nvvideoconvert only.",
          DEFAULT_DISABLE_PASSTHROUGH,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, MULTIURIBIN_PROP_HTTP_PORT,
      g_param_spec_string ("port",
          "Set REST API HTTP Port number",
          "REST API HTTP Endpoint; Note: User may pass \"0\" to disable REST API Server",
          DEFAULT_HTTP_PORT,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class,
      MULTIURIBIN_PROP_HTTP_DOWNLOAD_TIMEOUT,
      g_param_spec_uint ("http-download-timeout", "HTTP Download Timeout",
          "Maximum time in seconds for the whole HTTP(S) file download "
          "(curl CURLOPT_TIMEOUT). 0 means no limit.",
          0, G_MAXUINT, DEFAULT_HTTP_DOWNLOAD_TIMEOUT,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class,
      MULTIURIBIN_PROP_HTTP_CONNECT_TIMEOUT,
      g_param_spec_uint ("http-connect-timeout", "HTTP Connect Timeout",
          "Maximum time in seconds for the HTTP(S) connect phase "
          "(curl CURLOPT_CONNECTTIMEOUT). 0 means no limit.",
          0, G_MAXUINT, DEFAULT_HTTP_CONNECT_TIMEOUT,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class,
      MULTIURIBIN_PROP_HTTP_MAX_CONCURRENT_DOWNLOADS,
      g_param_spec_uint ("http-max-concurrent-downloads",
          "HTTP Max Concurrent Downloads",
          "Maximum number of HTTP(S) source-file downloads that may run at "
          "once across concurrent stream/add requests. 0 = use max-batch-size; "
          "any value is clamped to max-batch-size.",
          0, G_MAXUINT, DEFAULT_HTTP_MAX_CONCURRENT_DOWNLOADS,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, MULTIURIBIN_PROP_SEI_UUID,
      g_param_spec_string ("sei-uuid",
          "Set sei uuid",
          "Set sei uuid on decoder.",
          DEFAULT_SEI_UUID,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class,
      MULTIURIBIN_PROP_MAX_BATCH_SIZE, g_param_spec_uint ("max-batch-size",
          "Set the maximum batch size to be used for nvstreammux",
          "Maximum number of sources to be supported with this instance of nvmultiurisrcbin",
          0, G_MAXUINT, DEFAULT_MAX_BATCH_SIZE,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));
  /** @{ For nvurisrcbin  */
  g_object_class_install_property (gobject_class,
      MULTIURIBIN_PROP_NUM_EXTRA_SURF, g_param_spec_uint ("num-extra-surfaces",
          "Set extra decoder surfaces",
          "Number of surfaces in addition to minimum decode surfaces given by the decoder",
          0, G_MAXUINT, DEFAULT_NUM_EXTRA_SURFACES,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class,
      MULTIURIBIN_PROP_GPU_DEVICE_ID, g_param_spec_uint ("gpu-id",
          "Set GPU Device ID", "Set GPU Device ID", 0, G_MAXUINT,
          DEFAULT_GPU_DEVICE_ID,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class,
      MULTIURIBIN_PROP_CUDADEC_MEM_TYPE, g_param_spec_enum ("cudadec-memtype",
          "Memory type for cuda decoder buffers",
          "Set to specify memory type for cuda decoder buffers",
          GST_TYPE_V4L2_VID_CUDADEC_MEM_TYPE, DEFAULT_CUDADEC_MEM_TYPE,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class,
      MULTIURIBIN_PROP_DROP_FRAME_INTERVAL,
      g_param_spec_uint ("drop-frame-interval",
          "Set decoder drop frame interval",
          "Interval to drop the frames,ex: value of 5 means every 5th frame will be given by decoder, rest all dropped",
          0, 30, DEFAULT_DROP_FRAME_INTERVAL,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, MULTIURIBIN_PROP_DROP_ON_LATENCY,
      g_param_spec_boolean ("drop-on-latency",
          "Drop buffers when maximum latency is reached",
          "Tells the jitterbuffer to never exceed the given latency in size",
          DEFAULT_DROP_ON_LATENCY,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class,
      MULTIURIBIN_PROP_EXTRACT_SEI_TYPE5_DATA_DEC,
      g_param_spec_boolean ("extract-sei-type5-data-dec",
          "Extract SEI data for decoder",
          "Set to extract and attach SEI type5 unregistered data on output buffer for decoder",
          DEFAULT_SEI_EXTRACT_DATA,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class,
      MULTIURIBIN_PROP_LOW_LATENCY_MODE,
      g_param_spec_boolean ("low-latency-mode", "low-latency-mode",
          "Set low latency mode for bitstreams having I and IPPP frames on decoder",
          DEFAULT_LOW_LATENCY_MODE,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class,
      MULTIURIBIN_PROP_DEC_SKIP_FRAMES, g_param_spec_enum ("dec-skip-frames",
          "Type of frames to skip during decoding",
          "Type of frames to skip during decoding",
          GST_TYPE_NVDSURI_SKIP_FRAMES, DEFAULT_DEC_SKIP_FRAME_TYPE,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class,
      MULTIURIBIN_PROP_BUFFER_MODE, g_param_spec_enum ("buffer-mode",
          "Control the buffering algorithm in use",
          "Control the buffering algorithm in use",
          GST_TYPE_NVDSURI_BUFFER_MODE , DEFAULT_BUFFER_MODE,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, MULTIURIBIN_PROP_RTP_PROTOCOL,
      g_param_spec_enum ("select-rtp-protocol",
          "Transport Protocol to use for RTP",
          "Transport Protocol to use for RTP", GST_TYPE_NVDSURI_RTP_PROTOCOL,
          DEFAULT_RTP_PROTOCOL,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, MULTIURIBIN_PROP_LEAKY,
      g_param_spec_enum ("leaky",
          "Where the queue leaks, if at all",
          "Where the queue leaks, if at all", GST_TYPE_NVDSURI_LEAKY, DEFAULT_LEAKY,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, MULTIURIBIN_PROP_MAX_SIZE_BUFFERS,
      g_param_spec_uint ("max-size-buffers",
          "Maximum number of buffers in the queue",
          "Maximum number of buffers in the queue",
          0, G_MAXUINT, DEFAULT_MAX_SIZE_BUFFERS,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));
  g_object_class_install_property (gobject_class, MULTIURIBIN_PROP_FILE_LOOP,
      g_param_spec_boolean ("file-loop",
          "Loop file sources after EOS",
          "Loop file sources after EOS. Src type must be source-type-uri and uri starting with 'file:/'",
          DEFAULT_FILE_LOOP,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class,
      MULTIURIBIN_PROP_RTSP_RECONNECT_INTERVAL,
      g_param_spec_uint ("rtsp-reconnect-interval", "RTSP Reconnect Interval",
          "Timeout in seconds to wait since last data was received from an RTSP source before forcing a reconnection. 0=disable timeout",
          0, G_MAXUINT, DEFAULT_RTSP_RECONNECT_INTERVAL,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class,
      MULTIURIBIN_PROP_RTSP_INIT_RECONNECT_INTERVAL,
      g_param_spec_uint ("init-rtsp-reconnect-interval", "RTSP Reconnect Interval incase error received from rtspsrc",
          "Timeout in seconds to wait before forcing a reconnection when error received from rtsp source. 0=disable timeout",
          0, G_MAXUINT, DEFAULT_RTSP_INIT_RECONNECT_INTERVAL,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class,
    MULTIURIBIN_PROP_RTSP_RECONNECT_ATTEMPTS,
    g_param_spec_int ("rtsp-reconnect-attempts", "Set RTSP Reconnect attempts",
        "Set RTSP reconnect attempts ",
        G_MININT, G_MAXINT, DEFAULT_RTSP_RECONNECT_ATTEMPTS,
        (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
            GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class,
      MULTIURIBIN_PROP_LATENCY,
      g_param_spec_uint ("latency", "Latency",
          "Jitterbuffer size in milliseconds; applicable only for RTSP streams.",
          0, G_MAXUINT, DEFAULT_LATENCY,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class,
      MULTIURIBIN_PROP_UDP_BUFFER_SIZE,
      g_param_spec_uint ("udp-buffer-size", "UDP Buffer Size",
          "UDP Buffer Size in bytes; applicable only for RTSP streams.",
          0, G_MAXUINT, DEFAULT_UDP_BUFFER_SIZE,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, MULTIURIBIN_PROP_SMART_RECORD,
      g_param_spec_enum ("smart-record",
          "Enable Smart Record",
          "Enable Smart Record and choose the type of events to respond to. Sources must be of type source-type-rtsp",
          GST_TYPE_NVDSURI_SMART_RECORD_TYPE, DEFAULT_SMART_RECORD_MODE,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class,
      MULTIURIBIN_PROP_SMART_RECORD_DIR_PATH,
      g_param_spec_string ("smart-rec-dir-path",
          "Path of directory to save the recorded file",
          "Path of directory to save the recorded file.", NULL,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class,
      MULTIURIBIN_PROP_SMART_RECORD_FILE_PREFIX,
      g_param_spec_string ("smart-rec-file-prefix",
          "Prefix of file name for recorded video",
          "By default, Smart_Record is the prefix. For unique file names every source must be provided with a unique prefix",
          DEFAULT_SMART_RECORD_PREFIX,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class,
      MULTIURIBIN_PROP_SMART_RECORD_VIDEO_CACHE,
      g_param_spec_uint ("smart-rec-video-cache",
          "Size of video cache in seconds.",
          "Size of video cache in seconds. DEPRECATED: Use 'smart-rec-cache' instead",
          0, G_MAXUINT, DEFAULT_SMART_RECORD_CACHE,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class,
      MULTIURIBIN_PROP_SMART_RECORD_CACHE, g_param_spec_uint ("smart-rec-cache",
          "Size of cache in seconds, applies to both audio and video cache",
          "Size of cache in seconds, applies to both audio and video cache", 0,
          G_MAXUINT, DEFAULT_SMART_RECORD_CACHE,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class,
      MULTIURIBIN_PROP_SMART_RECORD_CONTAINER,
      g_param_spec_enum ("smart-rec-container",
          "Container format of recorded video",
          "Container format of recorded video. MP4 and MKV containers are supported. Sources must be of type source-type-rtsp",
          GST_TYPE_NVDSURI_SMART_RECORD_CONTAINER,
          DEFAULT_SMART_RECORD_CONTAINER,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class,
      MULTIURIBIN_PROP_SMART_RECORD_MODE, g_param_spec_enum ("smart-rec-mode",
          "Smart record mode", "Smart record mode",
          GST_TYPE_NVDSURI_SMART_RECORD_MODE, DEFAULT_SMART_RECORD_MODE,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class,
      MULTIURIBIN_PROP_SMART_RECORD_DEFAULT_DURATION,
      g_param_spec_uint ("smart-rec-default-duration",
          "In case a Stop event is not generated. This parameter will ensure the recording is stopped after a predefined default duration.",
          "In case a Stop event is not generated. This parameter will ensure the recording is stopped after a predefined default duration.",
          0, G_MAXUINT, DEFAULT_SMART_RECORD_DEFAULT_DURATION,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class,
      MULTIURIBIN_PROP_IPC_BUFFER_TIMESTAMP_COPY,
      g_param_spec_boolean ("ipc-buffer-timestamp-copy", "Copy buffer timestamp for nvunixfdsrc plugin",
          "Copy buffer timestamp for nvunixfdsrc plugin",
          DEFAULT_IPC_BUFFER_TIMESTAMP_COPY, (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, MULTIURIBIN_PROP_IPC_SOCKET_PATH,
      g_param_spec_string ("ipc-socket-path",
          "Path to the control socket",
          "The path to the control socket used to control the shared memory "
          "transport. This may be modified during the NULL->READY transition",
          NULL,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, MULTIURIBIN_PROP_IPC_CONNECTION_ATTEMPTS,
    g_param_spec_int ("ipc-connection-attempts", "ipc-connection-attempts",
        "Max number of attempts for connection (-1 = unlimited)",
        -1, G_MAXINT, DEFAULT_IPC_CONNECTION_ATTEMPTS,
        (GParamFlags) (G_PARAM_READWRITE |
        G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, MULTIURIBIN_PROP_IPC_CONNECTION_INTERVAL,
    g_param_spec_uint64 ("ipc-connection-interval", "ipc-connection-interval",
        "connection interval between connection attempts in micro seconds",
        0, G_MAXUINT64, DEFAULT_IPC_CONNECTION_INTERVAL,
        (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class,
      MULTIURIBIN_PROP_SENSORID_PADID_MAPPING,
      g_param_spec_boolean ("sensorID-padID-mapping", 
          "SensorID PadID Mapping",
          "One to One Mapping between Sensor ID and Mux Pad ID, must start from 0 Eg: Camera0, Camera1, Camera2",
          DEFAULT_SENSORID_PADID_MAPPING, (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, MULTIURIBIN_PROP_ENABLE_ERROR_PROPAGATION,
      g_param_spec_boolean ("enable-error-propagation",
          "Enable Error Propagation",
          "Enable automatic error message propagation to message broker (Kafka/Redis) when stream errors occur. "
          "Requires proto-lib, conn-str, and topic properties to be configured.",
          DEFAULT_ENABLE_ERROR_PROPAGATION, (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, MULTIURIBIN_PROP_PROTO_LIB,
      g_param_spec_string ("proto-lib",
          "Protocol Adapter Library Path",
          "Path to the message protocol adapter library for connecting to message broker. "
          "Example: /opt/nvidia/deepstream/deepstream/lib/libnvds_kafka_proto.so",
          NULL, (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, MULTIURIBIN_PROP_CONN_STR,
      g_param_spec_string ("conn-str",
          "Message Broker Connection String",
          "Connection string for the message broker in the format: <hostname>;<port>. "
          "Example: localhost;9092",
          NULL, (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, MULTIURIBIN_PROP_TOPIC,
      g_param_spec_string ("topic",
          "Message Broker Topic",
          "Topic name for publishing error messages to the message broker. ",
          NULL, (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, MULTIURIBIN_PROP_SIMULATE_FPS_INTERVAL_MS,
      g_param_spec_uint ("simulate-fps-interval-ms", "Simulate FPS Interval",
          "Frame interval in milliseconds for FPS simulation (e.g., 33 for 30 FPS, 0 to disable)",
          0, 1000, 0,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, MULTIURIBIN_PROP_IPC_FRAME_COPY,
      g_param_spec_boolean ("ipc-frame-copy",
          "IPC decoder frame copy",
          "Publish each source's DECODED frames over IPC (per-source nvunixfdsink, "
          "tapped right after the decoder). Socket: /tmp/nvds_ipc_<sensorId|source-id>.sock",
          FALSE,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  /** @} For nvurisrcbin */

  /** @{ For nvstreammux */
  g_object_class_install_property (gobject_class, PROP_BATCHED_PUSH_TIMEOUT,
      g_param_spec_int ("batched-push-timeout",
          "(nvstreammux) Batched Push Timeout",
          "Timeout in microseconds to wait after the first buffer is available\n"
          "\t\t\tto push the batch even if the complete batch is not formed.\n"
          "\t\t\tSet to -1 to wait infinitely", -1, G_MAXINT,
          DEFAULT_BATCHED_PUSH_TIMEOUT,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_WIDTH,
      g_param_spec_uint ("width", "(nvstreammux) Width",
          "Width of each frame in output batched buffer. This property MUST be set.",
          0, G_MAXUINT, DEFAULT_WIDTH,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_HEIGHT,
      g_param_spec_uint ("height", "(nvstreammux) Height",
          "Height of each frame in output batched buffer. This property MUST be set.",
          0, G_MAXUINT, DEFAULT_HEIGHT,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_ENABLE_PADDING,
      g_param_spec_boolean ("enable-padding", "(nvstreammux) Enable Padding",
          "Maintain input aspect ratio when scaling by padding with black bands.",
          FALSE, (GParamFlags) (G_PARAM_READWRITE)));

  g_object_class_install_property (gobject_class, PROP_NUM_SURFACES_PER_FRAME,
      g_param_spec_uint ("num-surfaces-per-frame",
          "(nvstreammux) Max number of surfaces per frame",
          "Max number of surfaces per frame", 1, 4, 1,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_LIVE_SOURCE,
      g_param_spec_boolean ("live-source", "(nvstreammux) live source",
          "Boolean property to inform muxer that sources are live.",
          DEFAULT_LIVE_SOURCE,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_SYNC_INPUTS,
      g_param_spec_boolean ("sync-inputs", "(nvstreammux) Synchronize Inputs",
          "Boolean property to force sychronization of input frames.",
          0, (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_BUFFER_POOL_SIZE,
      g_param_spec_uint ("buffer-pool-size", "Buffer Pool Size",
          "Maximum number of buffers from muxer's output pool",
          DEFAULT_BUFFER_POOL_SIZE, MAX_POOL_BUFFERS, DEFAULT_BUFFER_POOL_SIZE,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_MAX_LATNECY,
      g_param_spec_uint ("max-latency", "maximum lantency",
          "Additional latency in live mode to allow upstream to take longer to produce buffers for the current position (in nanoseconds)",
          0, G_MAXUINT, 0 /*200000000 */ ,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_ATTACH_SYS_TIME_STAMP,
      g_param_spec_boolean ("attach-sys-ts",
          "Set system timestamp as ntp timestamp",
          "If set to TRUE, system timestamp will be attached as ntp timestamp.\n"
          "\t\t\tIf set to FALSE, ntp timestamp from rtspsrc, if available, will be attached.",
          DEFAULT_ATTACH_SYS_TIME_STAMP,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_CONFIG_FILE_PATH,
      g_param_spec_string ("config-file-path", "Set config file path",
          "Configuation file path (applicable for new nvstreammux)", DEFAULT_CONFIG_FILE_PATH,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  PROP_NVBUF_MEMORY_TYPE_INSTALL (gobject_class);
  PROP_COMPUTE_HW_INSTALL (gobject_class);

  g_object_class_install_property (gobject_class, PROP_INTERPOLATION_METHOD,
      g_param_spec_enum ("interpolation-method", "Interpolation-method",
          "Set interpolation methods",
          GST_TYPE_INTERPOLATION_METHOD, NvBufSurfTransformInter_Bilinear,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_CONTROLLABLE | G_PARAM_CONSTRUCT)));

  g_object_class_install_property (gobject_class, PROP_FRAME_NUM_RESET_ON_EOS,
      g_param_spec_boolean ("frame-num-reset-on-eos",
          "Frame Number Reset on EOS",
          "Reset frame numbers to 0 for a source from which EOS is received (For debugging purpose only)",
          FALSE, (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class,
      PROP_FRAME_NUM_RESET_ON_STREAM_RESET,
      g_param_spec_boolean ("frame-num-reset-on-stream-reset",
          "Frame Number Reset on stream reset",
          "Reset frame numbers to 0 for a source which needs to be reset. (For debugging purpose only)\n"
          "Needs to be paired with tracking-id-reset-mode=1 in the tracker config.",
          FALSE, (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_FRAME_DURATION,
      g_param_spec_int ("frame-duration", "Frame duration",
          "Duration of input frames in milliseconds for use in NTP timestamp correction based on frame rate.\n"
          "\t\t\tIf set to 0, frame duration is inferred automatically from PTS values (default).\n"
          "\t\t\tIf set to -1, disables frame rate based NTP timestamp correction.",
          -1, G_MAXINT, DEFAULT_FRAME_DURATION,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_ASYNC_PROCESS,
      g_param_spec_boolean ("async-process",
          "(nvstreammux) Asynchronous Process",
          "Boolean property to enable/disable asynchronous processing of input frames for performance.",
          DEFAULT_ASYNC_PROCESS,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_NO_PIPELINE_EOS,
      g_param_spec_boolean ("drop-pipeline-eos",
          "(nvstreammux) No Pipeline EOS",
          "Boolean property so that EOS is not propagated downstream when all source pads are at EOS.",
          DEFAULT_NO_PIPELINE_EOS,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class,
      PROP_EXTRACT_SEI_TYPE5_DATA_MUX,
      g_param_spec_boolean ("extract-sei-type5-data",
          "Extract SEI data for Muxer",
          "Set to extract and attach SEI type5 unregistered data on output buffer for Mux",
          DEFAULT_SEI_EXTRACT_DATA,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_SORT_BATCH_BUFFERS,
      g_param_spec_boolean ("sort-batch", "Sort Batch Buffer in ascending order of pad-ids",
          "Sort Batch Buffer in ascending order",
          DEFAULT_SORT_BATCH_BUFFERS, (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_CACHE_BUFFERS,
      g_param_spec_boolean ("cache-buffer", "Render previous frame in case new frame is delayed",
          "Render previous frame in case new frame is delayed",
          DEFAULT_CACHE_BATCH_BUFFERS, (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_CACHE_BUFFERS_TIMEOUT,
      g_param_spec_int ("cache-buffer-timeout", "Cache Batch Buffer Timeout",
          "Timeout in microseconds to reuse previous rendered buffer\n"
          "\t\t\tPost timeout and when no initial buffer available , grey image to be rendered\n"
          "\t\t\tSet to -1 to wait infinitely",
          -1, G_MAXINT, DEFAULT_CACHED_BUFFER_TIMEOUT,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_EXTRACT_SIM_TIME,
        g_param_spec_boolean ("extract-sei-sim-time",
            "extract-sei-sim-time",
            "Set to extract and attach simulation time on output buffer",
            DEFAULT_SEI_EXTRACT_SIM_TIME,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_ALIGN_FIRST_BUFFER,
      g_param_spec_boolean ("align-first-buffer", "Align First Buffer",
          "Align only the first buffer of all sources within frame rate range",
          DEFAULT_ALIGN_FIRST_BATCH, (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_SYNC_INPUTS_NTP,
      g_param_spec_int ("sync-inputs-ntp", "PTS difference between incoming NTP frames",
          "Time in microseconds to synchronize incoming inputs based on ntp",
          0, G_MAXINT, DEFAULT_SYNC_INPUTS_NTP,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_DROP_BACKWARD_SEI,
      g_param_spec_boolean ("drop-backward-sei", "Drop Backward SEI",
          "Drop SEI messages with timestamps older than current frame",
          DEFAULT_DROP_BACKWARD_SEI,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  /** @} */

  gst_element_class_set_details_simple (gstelement_class,
      "NvMultiUri Bin", "NvMultiUri Bin",
      "Nvidia DeepStreamSDK NvMultiUri Bin",
      "NVIDIA Corporation. Post on Deepstream for Tesla forum for any queries "
      "@ https://devtalk.nvidia.com/default/board/209/");
}

static void
gst_ds_nvmultiurisrc_bin_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{

  GstDsNvMultiUriBin *nvmultiurisrcbin = GST_DS_NVMULTIURISRC_BIN (object);
  GstDsNvUriSrcConfig *config = nvmultiurisrcbin->config;
  GstDsNvStreammuxConfig *muxConfig = nvmultiurisrcbin->muxConfig;

  switch (prop_id) {
    case MULTIURIBIN_PROP_URI_LIST:
      if (nvmultiurisrcbin->uriList) {
        g_free (nvmultiurisrcbin->uriList);
        nvmultiurisrcbin->uriList = NULL;
      }
      if (nvmultiurisrcbin->uriListV) {
        g_strfreev (nvmultiurisrcbin->uriListV);
        nvmultiurisrcbin->uriListV = NULL;
      }
      nvmultiurisrcbin->uriList = g_value_dup_string (value);
      nvmultiurisrcbin->uriListV =
          g_strsplit (nvmultiurisrcbin->uriList, ",", -1);
      break;
    case MULTIURIBIN_PROP_SENSOR_ID_LIST:
      if (nvmultiurisrcbin->sensorIdList) {
        g_free (nvmultiurisrcbin->sensorIdList);
        nvmultiurisrcbin->sensorIdList = NULL;
      }
      if (nvmultiurisrcbin->sensorIdListV) {
        g_strfreev (nvmultiurisrcbin->sensorIdListV);
        nvmultiurisrcbin->sensorIdListV = NULL;
      }
      nvmultiurisrcbin->sensorIdList = g_value_dup_string (value);
      nvmultiurisrcbin->sensorIdListV =
          g_strsplit (nvmultiurisrcbin->sensorIdList, ",", -1);
      break;
    case MULTIURIBIN_PROP_SENSOR_NAME_LIST:
      if (nvmultiurisrcbin->sensorNameList) {
        g_free (nvmultiurisrcbin->sensorNameList);
        nvmultiurisrcbin->sensorNameList = NULL;
      }
      if (nvmultiurisrcbin->sensorNameListV) {
        g_strfreev (nvmultiurisrcbin->sensorNameListV);
        nvmultiurisrcbin->sensorNameListV = NULL;
      }
      nvmultiurisrcbin->sensorNameList = g_value_dup_string (value);
      nvmultiurisrcbin->sensorNameListV =
          g_strsplit (nvmultiurisrcbin->sensorNameList, ",", -1);
      break;
    case MULTIURIBIN_PROP_MODE:
      nvmultiurisrcbin->mode = (NvDsMultiUriMode) g_value_get_enum (value);
      break;
    case MULTIURIBIN_PROP_HTTP_IP:
      if (nvmultiurisrcbin->httpIp) {
        g_free (nvmultiurisrcbin->httpIp);
      }
      nvmultiurisrcbin->httpIp = g_value_dup_string (value);
      break;
    case MULTIURIBIN_PROP_HTTP_PORT:
      if (nvmultiurisrcbin->httpPort) {
        g_free (nvmultiurisrcbin->httpPort);
      }
      nvmultiurisrcbin->httpPort = g_value_dup_string (value);
      break;
    case MULTIURIBIN_PROP_HTTP_DOWNLOAD_TIMEOUT:
      nvmultiurisrcbin->httpDownloadTimeout = g_value_get_uint (value);
      break;
    case MULTIURIBIN_PROP_HTTP_CONNECT_TIMEOUT:
      nvmultiurisrcbin->httpConnectTimeout = g_value_get_uint (value);
      break;
    case MULTIURIBIN_PROP_HTTP_MAX_CONCURRENT_DOWNLOADS:
      nvmultiurisrcbin->httpMaxConcurrentDownloads = g_value_get_uint (value);
      break;
    case MULTIURIBIN_PROP_MAX_BATCH_SIZE:
      muxConfig->maxBatchSize = g_value_get_uint (value);
      break;
    case MULTIURIBIN_PROP_BUFFER_MODE:
      config->buffer_mode = (NvDsUriSrcBinBufferMode) g_value_get_enum (value);
      break;
    case MULTIURIBIN_PROP_RTSP_RECONNECT_INTERVAL:
      config->rtsp_reconnect_interval_sec = g_value_get_uint (value);
      break;
    case MULTIURIBIN_PROP_RTSP_INIT_RECONNECT_INTERVAL:
      config->init_rtsp_reconnect_interval_sec = g_value_get_uint (value);
      break;
    case MULTIURIBIN_PROP_RTSP_RECONNECT_ATTEMPTS:
      config->rtsp_reconnect_attempts = g_value_get_int (value);
      break;
    case MULTIURIBIN_PROP_NUM_EXTRA_SURF:
      config->num_extra_surfaces = g_value_get_uint (value);
      break;
    case MULTIURIBIN_PROP_DEC_SKIP_FRAMES:
      config->skip_frames_type =
          (NvDsUriSrcBinDecSkipFrame) g_value_get_enum (value);
      break;
    case MULTIURIBIN_PROP_GPU_DEVICE_ID:
      config->gpu_id = g_value_get_uint (value);
      muxConfig->gpu_id = g_value_get_uint (value);
      break;
    case MULTIURIBIN_PROP_CUDADEC_MEM_TYPE:
      config->cuda_memory_type = g_value_get_enum (value);
      break;
    case MULTIURIBIN_PROP_DROP_FRAME_INTERVAL:
      config->drop_frame_interval = g_value_get_uint (value);
      break;
    case MULTIURIBIN_PROP_DROP_ON_LATENCY:
      config->drop_on_latency = g_value_get_boolean (value);
      break;
    case MULTIURIBIN_PROP_EXTRACT_SEI_TYPE5_DATA_DEC:
      config->extract_sei_type5_data = g_value_get_boolean (value);
      break;
    case MULTIURIBIN_PROP_RTP_PROTOCOL:
      config->rtp_protocol =
          (NvDsUriSrcBinRtpProtocol) g_value_get_enum (value);
      break;
    case MULTIURIBIN_PROP_LEAKY:
      config->leaky = (NvDsUriSrcBinLeaky) g_value_get_enum (value);
      break;
    case MULTIURIBIN_PROP_MAX_SIZE_BUFFERS:
      config->max_size_buffers = g_value_get_uint (value);
      break;
    case MULTIURIBIN_PROP_FILE_LOOP:
      config->loop = g_value_get_boolean (value);
      break;
    case MULTIURIBIN_PROP_LATENCY:
      config->latency = g_value_get_uint (value);
      break;
    case MULTIURIBIN_PROP_UDP_BUFFER_SIZE:
      config->udp_buffer_size = g_value_get_uint (value);
      break;
    case MULTIURIBIN_PROP_SMART_RECORD:
      config->smart_record = (NvDsUriSrcBinSRType) g_value_get_enum (value);
      break;
    case MULTIURIBIN_PROP_SMART_RECORD_DIR_PATH:
      config->smart_rec_dir_path = g_value_dup_string (value);
      break;
    case MULTIURIBIN_PROP_SMART_RECORD_FILE_PREFIX:
      config->smart_rec_file_prefix = g_value_dup_string (value);
      break;
    case MULTIURIBIN_PROP_SMART_RECORD_VIDEO_CACHE:
      g_warning
          ("%s: Deprecated property 'smart-rec-video-cache' set. Set property 'smart-rec-cache' instead.",
          GST_ELEMENT_NAME (nvmultiurisrcbin));
      break;
    case MULTIURIBIN_PROP_SMART_RECORD_CACHE:
      config->smart_rec_cache_size = g_value_get_uint (value);
      break;
    case MULTIURIBIN_PROP_SMART_RECORD_CONTAINER:
      config->smart_rec_container =
          (NvDsUriSrcBinSRCont) g_value_get_enum (value);
      break;
    case MULTIURIBIN_PROP_SMART_RECORD_MODE:
      config->smart_rec_mode = (NvDsUriSrcBinSRMode) g_value_get_enum (value);
      break;
    case MULTIURIBIN_PROP_SMART_RECORD_DEFAULT_DURATION:
      config->smart_rec_def_duration = g_value_get_uint (value);
      break;
    case MULTIURIBIN_PROP_DISABLE_PASSTHROUGH:
      config->disable_passthrough = g_value_get_boolean (value);
      break;
    case MULTIURIBIN_PROP_LOW_LATENCY_MODE:
      config->low_latency_mode = g_value_get_boolean (value);
      break;
    case MULTIURIBIN_PROP_SEI_UUID:
      config->sei_uuid = g_value_dup_string (value);
      break;
    case MULTIURIBIN_PROP_DISABLE_AUDIO:
      config->disable_audio = g_value_get_boolean (value);
      break;
    case MULTIURIBIN_PROP_IPC_BUFFER_TIMESTAMP_COPY:
      config->ipc_buffer_timestamp_copy = g_value_get_boolean (value);
      break;
    case MULTIURIBIN_PROP_IPC_SOCKET_PATH:
      config->ipc_socket_path = g_value_dup_string (value);
      break;
    case MULTIURIBIN_PROP_IPC_CONNECTION_ATTEMPTS:
      config->ipc_connection_attempts = g_value_get_int (value);
      break;
    case MULTIURIBIN_PROP_IPC_CONNECTION_INTERVAL:
      config->ipc_connection_interval = g_value_get_uint64 (value);
      break;
    case MULTIURIBIN_PROP_SENSORID_PADID_MAPPING:
      config->sensorIdToPadIdMapping = g_value_get_boolean (value);
      break;
    case MULTIURIBIN_PROP_ENABLE_ERROR_PROPAGATION:
      config->enable_error_propagation = g_value_get_boolean (value);
      break;
    case MULTIURIBIN_PROP_PROTO_LIB:
      config->proto_lib = g_value_dup_string (value);
      break;
    case MULTIURIBIN_PROP_CONN_STR:
      config->conn_str = g_value_dup_string (value);
      break;
    case MULTIURIBIN_PROP_TOPIC:
      config->topic = g_value_dup_string (value);
      break;
    case MULTIURIBIN_PROP_SIMULATE_FPS_INTERVAL_MS:
      config->simulate_fps_interval_ms = g_value_get_uint (value);
      break;
    case MULTIURIBIN_PROP_IPC_FRAME_COPY:
      muxConfig->ipc_frame_copy = g_value_get_boolean (value);
      break;
    case PROP_BATCH_SIZE:
      muxConfig->batch_size = g_value_get_uint (value);
      break;
    case PROP_BATCHED_PUSH_TIMEOUT:
      muxConfig->batched_push_timeout = g_value_get_int (value);
      break;
    case PROP_WIDTH:
      muxConfig->pipeline_width = g_value_get_uint (value);
      break;
    case PROP_HEIGHT:
      muxConfig->pipeline_height = g_value_get_uint (value);
      break;
    case PROP_NUM_SURFACES_PER_FRAME:
      muxConfig->num_surfaces_per_frame = g_value_get_uint (value);
      break;
    case PROP_LIVE_SOURCE:
      muxConfig->live_source = g_value_get_boolean (value);
      break;
    case PROP_SYNC_INPUTS:
      muxConfig->sync_inputs = g_value_get_boolean (value);
      break;
    case PROP_ATTACH_SYS_TIME_STAMP:
      muxConfig->attach_sys_ts_as_ntp = g_value_get_boolean (value);
      break;
    case PROP_CONFIG_FILE_PATH:
      muxConfig->config_file_path = g_value_dup_string (value);
      break;
    case PROP_ENABLE_PADDING:
      muxConfig->enable_padding = g_value_get_boolean (value);
      break;
    case PROP_COMPUTE_HW:
      muxConfig->compute_hw = g_value_get_enum (value);
      break;
    case PROP_NVBUF_MEMORY_TYPE:
      muxConfig->nvbuf_memory_type = g_value_get_enum (value);
      break;
    case PROP_INTERPOLATION_METHOD:
      muxConfig->interpolation_method = g_value_get_enum (value);
      break;
    case PROP_BUFFER_POOL_SIZE:
      muxConfig->buffer_pool_size = g_value_get_uint (value);
      break;
    case PROP_MAX_LATNECY:
      muxConfig->max_latency = g_value_get_uint (value);
      break;
    case PROP_FRAME_NUM_RESET_ON_EOS:
      muxConfig->frame_num_reset_on_eos = g_value_get_boolean (value);
      break;
    case PROP_FRAME_NUM_RESET_ON_STREAM_RESET:
      muxConfig->frame_num_reset_on_stream_reset = g_value_get_boolean (value);
      break;
    case PROP_ASYNC_PROCESS:
      muxConfig->async_process = g_value_get_boolean (value);
      break;
    case PROP_NO_PIPELINE_EOS:
      muxConfig->no_pipeline_eos = g_value_get_boolean (value);
      break;
    case PROP_FRAME_DURATION:
    {
      gint ms_value = g_value_get_int (value);
      if (ms_value < 0) {
        muxConfig->frame_duration = GST_CLOCK_TIME_NONE;
      } else {
        muxConfig->frame_duration = (GstClockTime) ms_value *GST_MSECOND;
      }
      break;
    }
    case PROP_EXTRACT_SEI_TYPE5_DATA_MUX:
      muxConfig->extract_sei_type5_data = g_value_get_boolean (value);
      break;
      break;
    case PROP_SORT_BATCH_BUFFERS:
      muxConfig->sort_batch = g_value_get_boolean(value);
      break;
    case PROP_CACHE_BUFFERS:
      muxConfig->buffer_cache = g_value_get_boolean(value);
      break;
    case PROP_CACHE_BUFFERS_TIMEOUT:
      muxConfig->buffer_cache_timeout = g_value_get_int (value);
      break;
    case PROP_EXTRACT_SIM_TIME:
      muxConfig->extract_sei_sim_time = g_value_get_boolean (value);
      break;
    case PROP_ALIGN_FIRST_BUFFER:
      muxConfig->align_first_buffer = g_value_get_boolean(value);
      break;
    case PROP_SYNC_INPUTS_NTP:
      muxConfig->sync_inputs_ntp = g_value_get_int (value);
      break;
    case PROP_DROP_BACKWARD_SEI:
      muxConfig->drop_backward_sei = g_value_get_boolean (value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }

}

static void
gst_ds_nvmultiurisrc_bin_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{

  GstDsNvMultiUriBin *nvmultiurisrcbin = GST_DS_NVMULTIURISRC_BIN (object);
  GstDsNvUriSrcConfig *config = nvmultiurisrcbin->config;
  GstDsNvStreammuxConfig *muxConfig = nvmultiurisrcbin->muxConfig;

  switch (prop_id) {
    case MULTIURIBIN_PROP_URI_LIST:
      g_value_set_string (value, nvmultiurisrcbin->uriList);
      break;
    case MULTIURIBIN_PROP_SENSOR_ID_LIST:
      g_value_set_string (value, nvmultiurisrcbin->sensorIdList);
      break;
    case MULTIURIBIN_PROP_SENSOR_NAME_LIST:
      g_value_set_string (value, nvmultiurisrcbin->sensorNameList);
      break;
    case MULTIURIBIN_PROP_MODE:
      g_value_set_enum (value, (guint) nvmultiurisrcbin->mode);
      break;
    case MULTIURIBIN_PROP_HTTP_IP:
      g_value_set_string (value, nvmultiurisrcbin->httpIp);
      break;
    case MULTIURIBIN_PROP_HTTP_PORT:
      g_value_set_string (value, nvmultiurisrcbin->httpPort);
      break;
    case MULTIURIBIN_PROP_HTTP_DOWNLOAD_TIMEOUT:
      g_value_set_uint (value, nvmultiurisrcbin->httpDownloadTimeout);
      break;
    case MULTIURIBIN_PROP_HTTP_CONNECT_TIMEOUT:
      g_value_set_uint (value, nvmultiurisrcbin->httpConnectTimeout);
      break;
    case MULTIURIBIN_PROP_HTTP_MAX_CONCURRENT_DOWNLOADS:
      g_value_set_uint (value, nvmultiurisrcbin->httpMaxConcurrentDownloads);
      break;
    case MULTIURIBIN_PROP_MAX_BATCH_SIZE:
      g_value_set_uint (value, muxConfig->maxBatchSize);
      break;
    case MULTIURIBIN_PROP_NUM_EXTRA_SURF:
      g_value_set_uint (value, config->num_extra_surfaces);
      break;
    case MULTIURIBIN_PROP_DEC_SKIP_FRAMES:
      g_value_set_enum (value, config->skip_frames_type);
      break;
    case MULTIURIBIN_PROP_GPU_DEVICE_ID:
      //Note: config->gpu_id will be same as muxConfig->gpu_id
      g_value_set_uint (value, config->gpu_id);
      break;
    case MULTIURIBIN_PROP_CUDADEC_MEM_TYPE:
      g_value_set_enum (value, config->cuda_memory_type);
      break;
    case MULTIURIBIN_PROP_SEI_UUID:
      g_value_set_string (value, config->sei_uuid);
      break;
    case MULTIURIBIN_PROP_DROP_FRAME_INTERVAL:
      g_value_set_uint (value, config->drop_frame_interval);
      break;
    case MULTIURIBIN_PROP_DROP_ON_LATENCY:
      g_value_set_boolean (value, config->drop_on_latency);
      break;
    case MULTIURIBIN_PROP_EXTRACT_SEI_TYPE5_DATA_DEC:
      g_value_set_boolean (value, config->extract_sei_type5_data);
      break;
    case MULTIURIBIN_PROP_RTP_PROTOCOL:
      g_value_set_enum (value, config->rtp_protocol);
      break;
    case MULTIURIBIN_PROP_LEAKY:
      g_value_set_enum (value, config->leaky);
      break;
    case MULTIURIBIN_PROP_MAX_SIZE_BUFFERS:
      g_value_set_uint (value, config->max_size_buffers);
      break;
    case MULTIURIBIN_PROP_FILE_LOOP:
      g_value_set_boolean (value, config->loop);
      break;
    case MULTIURIBIN_PROP_BUFFER_MODE:
      g_value_set_enum (value, config->buffer_mode);
      break;
    case MULTIURIBIN_PROP_RTSP_RECONNECT_INTERVAL:
      g_value_set_uint (value, config->rtsp_reconnect_interval_sec);
      break;
    case MULTIURIBIN_PROP_RTSP_INIT_RECONNECT_INTERVAL:
      g_value_set_uint (value, config->init_rtsp_reconnect_interval_sec);
      break;
    case MULTIURIBIN_PROP_RTSP_RECONNECT_ATTEMPTS:
      g_value_set_int (value, config->rtsp_reconnect_attempts);
      break;
    case MULTIURIBIN_PROP_LATENCY:
      g_value_set_uint (value, config->latency);
      break;
    case MULTIURIBIN_PROP_UDP_BUFFER_SIZE:
      g_value_set_uint (value, config->udp_buffer_size);
      break;
    case MULTIURIBIN_PROP_SMART_RECORD:
      g_value_set_enum (value, config->smart_record);
      break;
    case MULTIURIBIN_PROP_SMART_RECORD_DIR_PATH:
      g_value_set_string (value, config->smart_rec_dir_path);
      break;
    case MULTIURIBIN_PROP_SMART_RECORD_FILE_PREFIX:
      g_value_set_string (value, config->smart_rec_file_prefix);
      break;
    case MULTIURIBIN_PROP_SMART_RECORD_VIDEO_CACHE:
    case MULTIURIBIN_PROP_SMART_RECORD_CACHE:
      g_value_set_uint (value, config->smart_rec_cache_size);
      break;
    case MULTIURIBIN_PROP_SMART_RECORD_CONTAINER:
      g_value_set_enum (value, config->smart_rec_container);
      break;
    case MULTIURIBIN_PROP_SMART_RECORD_MODE:
      g_value_set_enum (value, config->smart_rec_mode);
      break;
    case MULTIURIBIN_PROP_SMART_RECORD_DEFAULT_DURATION:
      g_value_set_uint (value, config->smart_rec_def_duration);
      break;
    case MULTIURIBIN_PROP_DISABLE_PASSTHROUGH:
      g_value_set_boolean (value, config->disable_passthrough);
      break;
    case MULTIURIBIN_PROP_LOW_LATENCY_MODE:
      g_value_set_boolean (value, config->low_latency_mode);
      break;
    case MULTIURIBIN_PROP_DISABLE_AUDIO:
      g_value_set_boolean (value, config->disable_audio);
      break;
    case MULTIURIBIN_PROP_IPC_BUFFER_TIMESTAMP_COPY:
      g_value_set_boolean (value, config->ipc_buffer_timestamp_copy);
      break;
    case MULTIURIBIN_PROP_IPC_SOCKET_PATH:
      g_value_set_string (value, config->ipc_socket_path);
      break;
    case MULTIURIBIN_PROP_IPC_CONNECTION_ATTEMPTS:
      g_value_set_int (value, config->ipc_connection_attempts);
      break;
    case MULTIURIBIN_PROP_IPC_CONNECTION_INTERVAL:
      g_value_set_uint64 (value, config->ipc_connection_interval);
      break;
    case MULTIURIBIN_PROP_SENSORID_PADID_MAPPING:
      g_value_set_boolean (value, config->sensorIdToPadIdMapping);
      break;
    case MULTIURIBIN_PROP_ENABLE_ERROR_PROPAGATION:
      g_value_set_boolean (value, config->enable_error_propagation);
      break;
    case MULTIURIBIN_PROP_PROTO_LIB:
      g_value_set_string (value, config->proto_lib);
      break;
    case MULTIURIBIN_PROP_CONN_STR:
      g_value_set_string (value, config->conn_str);
      break;
    case MULTIURIBIN_PROP_TOPIC:
      g_value_set_string (value, config->topic);
      break;
    case MULTIURIBIN_PROP_SIMULATE_FPS_INTERVAL_MS:
      g_value_set_uint (value, config->simulate_fps_interval_ms);
      break;
    case MULTIURIBIN_PROP_IPC_FRAME_COPY:
      g_value_set_boolean (value, muxConfig->ipc_frame_copy);
      break;
    case PROP_BATCH_SIZE:
      g_value_set_uint (value, muxConfig->batch_size);
      break;
    case PROP_BATCHED_PUSH_TIMEOUT:
      g_value_set_int (value, muxConfig->batched_push_timeout);
      break;
    case PROP_WIDTH:
      g_value_set_uint (value, muxConfig->pipeline_width);
      break;
    case PROP_HEIGHT:
      g_value_set_uint (value, muxConfig->pipeline_height);
      break;
    case PROP_NUM_SURFACES_PER_FRAME:
      g_value_set_uint (value, muxConfig->num_surfaces_per_frame);
      break;
    case PROP_ENABLE_PADDING:
      g_value_set_boolean (value, muxConfig->enable_padding);
      break;
    case PROP_LIVE_SOURCE:
      g_value_set_boolean (value, muxConfig->live_source);
      break;
    case PROP_SYNC_INPUTS:
      g_value_set_boolean (value, muxConfig->sync_inputs);
      break;
    case PROP_ATTACH_SYS_TIME_STAMP:
      g_value_set_boolean (value, muxConfig->attach_sys_ts_as_ntp);
      break;
    case PROP_CONFIG_FILE_PATH:
      g_value_set_string (value, muxConfig->config_file_path);
      break;
    case PROP_COMPUTE_HW:
      g_value_set_enum (value, muxConfig->compute_hw);
      break;
    case PROP_NVBUF_MEMORY_TYPE:
      g_value_set_enum (value, muxConfig->nvbuf_memory_type);
      break;
    case PROP_INTERPOLATION_METHOD:
      g_value_set_enum (value, muxConfig->interpolation_method);
      break;
    case PROP_BUFFER_POOL_SIZE:
      g_value_set_uint (value, muxConfig->buffer_pool_size);
      break;
    case PROP_MAX_LATNECY:
      g_value_set_uint (value, muxConfig->max_latency);
      break;
    case PROP_FRAME_NUM_RESET_ON_EOS:
      g_value_set_boolean (value, muxConfig->frame_num_reset_on_eos);
      break;
    case PROP_FRAME_NUM_RESET_ON_STREAM_RESET:
      g_value_set_boolean (value, muxConfig->frame_num_reset_on_stream_reset);
      break;
    case PROP_FRAME_DURATION:
    {
      gint ms_value = -1;
      ms_value = muxConfig->frame_duration / GST_MSECOND;
      g_value_set_int (value, ms_value);
      break;
    }
    case PROP_ASYNC_PROCESS:
      g_value_set_boolean (value, muxConfig->async_process);
      break;
    case PROP_NO_PIPELINE_EOS:
      g_value_set_boolean (value, muxConfig->no_pipeline_eos);
      break;
    case PROP_EXTRACT_SEI_TYPE5_DATA_MUX:
      g_value_set_boolean (value, muxConfig->extract_sei_type5_data);
      break;
    case PROP_SORT_BATCH_BUFFERS:
      g_value_set_boolean (value, muxConfig->sort_batch);
      break;
   case PROP_CACHE_BUFFERS:
      g_value_set_boolean (value, muxConfig->buffer_cache);
      break;
    case PROP_CACHE_BUFFERS_TIMEOUT:
      g_value_set_int (value, muxConfig->buffer_cache_timeout);
      break;
    case PROP_EXTRACT_SIM_TIME:
      g_value_set_boolean (value, muxConfig->extract_sei_sim_time);
      break;
    case PROP_ALIGN_FIRST_BUFFER:
      g_value_set_boolean (value, muxConfig->align_first_buffer);
      break;
    case PROP_SYNC_INPUTS_NTP:
      g_value_set_int (value, muxConfig->sync_inputs_ntp);
      break;
    case PROP_DROP_BACKWARD_SEI:
      g_value_set_boolean (value, muxConfig->drop_backward_sei);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }

}


static void
gst_ds_nvmultiurisrc_bin_init (GstDsNvMultiUriBin * nvmultiurisrcbin)
{
  g_object_set (G_OBJECT (nvmultiurisrcbin), "async-handling", TRUE, nullptr);
  nvmultiurisrcbin->config =
      (GstDsNvUriSrcConfig *) g_malloc0 (sizeof (GstDsNvUriSrcConfig));
  nvmultiurisrcbin->muxConfig =
      (GstDsNvStreammuxConfig *) g_malloc0 (sizeof (GstDsNvStreammuxConfig));

  nvmultiurisrcbin->muxConfig->nvbuf_memory_type = DEFAULT_NVBUF_MEM_TYPE;
  /** Note: Using max-batch-size config to workaround
   * the pipeline slow-down caused by add/remove without this
   * Issue tracked for next release
   */
  nvmultiurisrcbin->muxConfig->maxBatchSize = DEFAULT_MAX_BATCH_SIZE;
  nvmultiurisrcbin->muxConfig->async_process = DEFAULT_ASYNC_PROCESS;
  nvmultiurisrcbin->muxConfig->no_pipeline_eos = DEFAULT_NO_PIPELINE_EOS;
  nvmultiurisrcbin->muxConfig->attach_sys_ts_as_ntp =
      DEFAULT_ATTACH_SYS_TIME_STAMP;
  nvmultiurisrcbin->muxConfig->config_file_path = g_strdup (DEFAULT_CONFIG_FILE_PATH);

  nvmultiurisrcbin->config->uri = NULL;
  nvmultiurisrcbin->config->sei_uuid = DEFAULT_SEI_UUID;
  nvmultiurisrcbin->config->num_extra_surfaces = DEFAULT_NUM_EXTRA_SURFACES;
  nvmultiurisrcbin->config->gpu_id = DEFAULT_GPU_DEVICE_ID;
  nvmultiurisrcbin->config->cuda_memory_type = DEFAULT_CUDADEC_MEM_TYPE;
  nvmultiurisrcbin->config->drop_frame_interval = DEFAULT_DROP_FRAME_INTERVAL;
  nvmultiurisrcbin->config->drop_on_latency = DEFAULT_DROP_ON_LATENCY;
  nvmultiurisrcbin->config->buffer_mode = (NvDsUriSrcBinBufferMode) DEFAULT_BUFFER_MODE;
  nvmultiurisrcbin->config->leaky = (NvDsUriSrcBinLeaky) DEFAULT_LEAKY;
  nvmultiurisrcbin->config->max_size_buffers = DEFAULT_MAX_SIZE_BUFFERS;
  nvmultiurisrcbin->config->extract_sei_type5_data = DEFAULT_SEI_EXTRACT_DATA;
  nvmultiurisrcbin->config->low_latency_mode = DEFAULT_LOW_LATENCY_MODE;
  nvmultiurisrcbin->httpDownloadTimeout = DEFAULT_HTTP_DOWNLOAD_TIMEOUT;
  nvmultiurisrcbin->httpConnectTimeout = DEFAULT_HTTP_CONNECT_TIMEOUT;
  nvmultiurisrcbin->httpMaxConcurrentDownloads =
      DEFAULT_HTTP_MAX_CONCURRENT_DOWNLOADS;
  nvmultiurisrcbin->httpActiveDownloads = 0;
  nvmultiurisrcbin->muxConfig->extract_sei_type5_data = DEFAULT_NO_PIPELINE_EOS;
  nvmultiurisrcbin->muxConfig->extract_sei_sim_time = DEFAULT_SEI_EXTRACT_SIM_TIME;
  nvmultiurisrcbin->muxConfig->align_first_buffer = DEFAULT_ALIGN_FIRST_BATCH;
  nvmultiurisrcbin->muxConfig->sort_batch = DEFAULT_SORT_BATCH_BUFFERS;
  nvmultiurisrcbin->muxConfig->buffer_cache = DEFAULT_CACHE_BATCH_BUFFERS;
  nvmultiurisrcbin->muxConfig->buffer_cache_timeout = DEFAULT_CACHED_BUFFER_TIMEOUT;
  nvmultiurisrcbin->muxConfig->sync_inputs_ntp = DEFAULT_SYNC_INPUTS_NTP;
  nvmultiurisrcbin->muxConfig->drop_backward_sei = DEFAULT_DROP_BACKWARD_SEI;
  nvmultiurisrcbin->config->skip_frames_type =
      (NvDsUriSrcBinDecSkipFrame) DEFAULT_DEC_SKIP_FRAME_TYPE;
  nvmultiurisrcbin->config->rtp_protocol =
      (NvDsUriSrcBinRtpProtocol) DEFAULT_RTP_PROTOCOL;
  nvmultiurisrcbin->config->rtsp_reconnect_interval_sec =
      DEFAULT_RTSP_RECONNECT_INTERVAL;
  nvmultiurisrcbin->config->init_rtsp_reconnect_interval_sec =
      DEFAULT_RTSP_INIT_RECONNECT_INTERVAL;
  nvmultiurisrcbin->config->rtsp_reconnect_attempts =
      DEFAULT_RTSP_RECONNECT_ATTEMPTS;
  nvmultiurisrcbin->config->latency = DEFAULT_LATENCY;
  nvmultiurisrcbin->config->disable_passthrough = DEFAULT_DISABLE_PASSTHROUGH;
  nvmultiurisrcbin->config->disable_audio = DEFAULT_DISABLE_AUDIO;
  nvmultiurisrcbin->config->udp_buffer_size = DEFAULT_UDP_BUFFER_SIZE;
  nvmultiurisrcbin->config->ipc_buffer_timestamp_copy = DEFAULT_IPC_BUFFER_TIMESTAMP_COPY;
  nvmultiurisrcbin->config->ipc_socket_path = NULL;
  nvmultiurisrcbin->config->ipc_connection_attempts = DEFAULT_IPC_CONNECTION_ATTEMPTS;
  nvmultiurisrcbin->config->ipc_connection_interval = DEFAULT_IPC_CONNECTION_INTERVAL;
  nvmultiurisrcbin->config->sensorIdToPadIdMapping = DEFAULT_SENSORID_PADID_MAPPING;
  nvmultiurisrcbin->config->enable_error_propagation = DEFAULT_ENABLE_ERROR_PROPAGATION;
  nvmultiurisrcbin->config->proto_lib = NULL;
  nvmultiurisrcbin->config->conn_str = NULL;
  nvmultiurisrcbin->config->topic = NULL;
  nvmultiurisrcbin->mode = NVDS_MULTIURISRCBIN_MODE_VIDEO;
  nvmultiurisrcbin->uriList = NULL;
  nvmultiurisrcbin->uriListV = NULL;
  nvmultiurisrcbin->sensorIdList = NULL;
  nvmultiurisrcbin->sensorIdListV = NULL;
  nvmultiurisrcbin->sensorNameList = NULL;
  nvmultiurisrcbin->sensorNameListV = NULL;
  nvmultiurisrcbin->sourceIdCounter = 0;
  nvmultiurisrcbin->bin_src_pad =
      gst_ghost_pad_new_no_target_from_template ("src",
      gst_static_pad_template_get (&gst_nvmultiurisrc_bin_src_template));
  gst_element_add_pad (GST_ELEMENT (nvmultiurisrcbin),
      nvmultiurisrcbin->bin_src_pad);
  nvmultiurisrcbin->restServer = NULL;
  nvmultiurisrcbin->staticBindsEmitted = FALSE;
  nvmultiurisrcbin->httpIp = g_strdup (DEFAULT_HTTP_IP);
  nvmultiurisrcbin->httpPort = g_strdup (DEFAULT_HTTP_PORT);
  g_mutex_init (&nvmultiurisrcbin->bin_lock);
  g_mutex_init (&nvmultiurisrcbin->httpDlLock);
  g_cond_init (&nvmultiurisrcbin->httpDlCond);
  g_mutex_init(&g_shared_comp_latency_data.mutex);

  GST_OBJECT_FLAG_SET (nvmultiurisrcbin, GST_ELEMENT_FLAG_SOURCE);
}

/* Free resources allocated during init. */
static void
gst_ds_nvmultiurisrc_bin_finalize (GObject * object)
{
  GstDsNvMultiUriBin *nvmultiurisrcbin = GST_DS_NVMULTIURISRC_BIN (object);
  if (nvmultiurisrcbin->config->ipc_socket_path) {
    g_free(nvmultiurisrcbin->config->ipc_socket_path);
    nvmultiurisrcbin->config->ipc_socket_path = NULL;
  }
  g_free (nvmultiurisrcbin->config);
  g_mutex_clear (&nvmultiurisrcbin->httpDlLock);
  g_cond_clear (&nvmultiurisrcbin->httpDlCond);

  if (nvmultiurisrcbin->uriList) {
    g_free (nvmultiurisrcbin->uriList);
    nvmultiurisrcbin->uriList = NULL;
  }
  if (nvmultiurisrcbin->uriListV) {
    g_strfreev (nvmultiurisrcbin->uriListV);
    nvmultiurisrcbin->uriListV = NULL;
  }

  if (nvmultiurisrcbin->sensorIdList) {
    g_free (nvmultiurisrcbin->sensorIdList);
    nvmultiurisrcbin->sensorIdList = NULL;
  }
  if (nvmultiurisrcbin->sensorIdListV) {
    g_strfreev (nvmultiurisrcbin->sensorIdListV);
    nvmultiurisrcbin->sensorIdListV = NULL;
  }

  if (nvmultiurisrcbin->sensorNameList) {
    g_free (nvmultiurisrcbin->sensorNameList);
    nvmultiurisrcbin->sensorNameList = NULL;
  }
  if (nvmultiurisrcbin->sensorNameListV) {
    g_strfreev (nvmultiurisrcbin->sensorNameListV);
    nvmultiurisrcbin->sensorNameListV = NULL;
  }

  if (nvmultiurisrcbin->httpIp) {
    g_free (nvmultiurisrcbin->httpIp);
  }

  if (nvmultiurisrcbin->httpPort) {
    g_free (nvmultiurisrcbin->httpPort);
  }

  if (nvmultiurisrcbin->muxConfig->config_file_path) {
    g_free (nvmultiurisrcbin->muxConfig->config_file_path);
  }

  G_OBJECT_CLASS (parent_class)->finalize (object);
}

/* For sources added STATICALLY via uri-list/sensor-id-list, forward each
 * source's per-stream model binding (camera_id + camera_name) downstream as a
 * serialized nv-stream-model-bind event -- exactly what the REST stream/add
 * path does (see s_stream_api_impl). Without this a downstream model element
 * (e.g. nvmodelmux) only ever sees GST_NVEVENT_PAD_ADDED, which carries the
 * numeric source_id but NOT the camera_id, so it cannot apply per-sensor
 * [stream-model-<id>] selection and every static source falls back to the
 * default model.
 *
 * Emitted ONCE, after the pipeline reaches PLAYING: by then the bin src pad is
 * linked + flowing, so the event actually propagates (pushing it at
 * NULL_TO_READY, before the downstream link exists, would just be dropped). The
 * downstream element treats a bind that arrives after PAD_ADDED as a reconcile
 * and reroutes that one stream zero-drop, so ordering vs PAD_ADDED is safe. */
static void
gst_ds_nvmultiurisrc_bin_forward_static_binds (GstDsNvMultiUriBin *
    nvmultiurisrcbin)
{
  guint i, sensorIdListVLen, sensorNameListVLen;

  if (!nvmultiurisrcbin->sensorIdListV
      || !nvmultiurisrcbin->nvmultiurisrcbinCreator
      || !nvmultiurisrcbin->bin_src_pad)
    return;                       /* nothing bound statically -> no-op */

  sensorIdListVLen = g_strv_length (nvmultiurisrcbin->sensorIdListV);
  sensorNameListVLen = nvmultiurisrcbin->sensorNameListV
      ? g_strv_length (nvmultiurisrcbin->sensorNameListV) : 0;

  for (i = 0; i < sensorIdListVLen; i++) {
    const gchar *sid = nvmultiurisrcbin->sensorIdListV[i];
    const gchar *snm = (i < sensorNameListVLen)
        ? nvmultiurisrcbin->sensorNameListV[i] : NULL;
    GstDsNvUriSrcConfig *sc;
    GstEvent *bev;

    if (!sid || !*sid)            /* skip empty entries (e.g. a trailing ';') */
      continue;

    /* resolve the ACTUAL assigned source_id for this camera (same lookup the
     * REST path uses); if the sensor was not actually added, skip it. */
    sc = gst_nvmultiurisrcbincreator_get_source_config_by_sensorid
        (nvmultiurisrcbin->nvmultiurisrcbinCreator, sid);
    if (!sc)
      continue;

    bev = gst_nvevent_new_stream_model_bind (sc->source_id, sid,
        (snm && *snm) ? snm : sid, NULL);
    if (bev) {
      GST_DEBUG_OBJECT (nvmultiurisrcbin,
          "forwarding static stream-model-bind: source_id=%u camera_id=%s",
          sc->source_id, sid);
      gst_pad_push_event ((GstPad *) (nvmultiurisrcbin->bin_src_pad), bev);
    }
    gst_nvmultiurisrcbincreator_src_config_free (sc);
  }
}

static GstStateChangeReturn
gst_ds_nvmultiurisrc_bin_change_state (GstElement * element,
    GstStateChange transition)
{
  GstDsNvMultiUriBin *nvmultiurisrcbin = GST_DS_NVMULTIURISRC_BIN (element);
  GstStateChangeReturn ret;

  if (transition == GST_STATE_CHANGE_NULL_TO_READY) {
    GST_DEBUG_OBJECT (nvmultiurisrcbin,
        "GST_STATE_CHANGE_NULL_TO_READY %s %d\n", __func__, __LINE__);

    // Initialize msgapi if configured
    if (nvmultiurisrcbin->config->enable_error_propagation) {
      if(!msgapi_init(nvmultiurisrcbin)) {
        g_print("[MSGAPI_ERROR] Failed to initialize msgapi\n");
      }
    }

    gboolean is_rtsp = nvmultiurisrcbin->config->uri
        ? g_str_has_prefix (nvmultiurisrcbin->config->uri, "rtsp://") : FALSE;

    (void) is_rtsp;
    guint binNameForCreatorLen =
        strlen (GST_ELEMENT_NAME ((GstElement *) nvmultiurisrcbin));
    gchar *binNameForCreator =
        g_strndup (GST_ELEMENT_NAME ((GstElement *) nvmultiurisrcbin),
        binNameForCreatorLen + 10);
    if ( g_strlcat (binNameForCreator, (const gchar *) "_creator",
        binNameForCreatorLen + 10) < binNameForCreatorLen + 10 ) {
      GST_DEBUG_OBJECT (nvmultiurisrcbin,
        "Unable to  append string \"_creator\" to the bin name %s %d\n", __func__, __LINE__);
    }
    /** Initialize the Bin Manipulation API for stream add/remove */
    nvmultiurisrcbin->nvmultiurisrcbinCreator =
        gst_nvmultiurisrcbincreator_init (binNameForCreator,
        nvmultiurisrcbin->mode, nvmultiurisrcbin->muxConfig);

    g_free (binNameForCreator);
    if (!nvmultiurisrcbin->nvmultiurisrcbinCreator) {
      GST_ERROR_OBJECT (nvmultiurisrcbin,
          "Failed to create nvmultiurisrcbin handler\n");
      return GST_STATE_CHANGE_FAILURE;
    }

    gst_bin_add (GST_BIN (nvmultiurisrcbin),
        gst_nvmultiurisrcbincreator_get_bin (nvmultiurisrcbin->
            nvmultiurisrcbinCreator));

    /** Add the initial list of sources from uri-list */
    if (nvmultiurisrcbin->uriListV) {
      guint i = 0;
      guint sensorIdListVLen =
          nvmultiurisrcbin->sensorIdListV ? g_strv_length (nvmultiurisrcbin->
          sensorIdListV) : 0;
      guint sensorNameListVLen =
          nvmultiurisrcbin->sensorNameListV ? g_strv_length (nvmultiurisrcbin->
          sensorNameListV) : 0;
        /** Add the initial sources from uri-list */
      for (i = 0; nvmultiurisrcbin->uriListV[i] != NULL; i++) {
          /** Add the source */
        nvmultiurisrcbin->config->uri = nvmultiurisrcbin->uriListV[i];
        if (i < sensorIdListVLen && nvmultiurisrcbin->sensorIdListV[i]) {
          nvmultiurisrcbin->config->sensorId =
              nvmultiurisrcbin->sensorIdListV[i];
        }
        if (i < sensorNameListVLen && nvmultiurisrcbin->sensorNameListV[i]) {
          nvmultiurisrcbin->config->sensorName =
              nvmultiurisrcbin->sensorNameListV[i];
        }
        nvmultiurisrcbin->config->source_id = 0;
        g_mutex_lock (&nvmultiurisrcbin->bin_lock);
        gboolean ret =
            gst_nvmultiurisrcbincreator_add_source (nvmultiurisrcbin->
            nvmultiurisrcbinCreator, nvmultiurisrcbin->config);
        g_mutex_unlock (&nvmultiurisrcbin->bin_lock);
        if (ret == FALSE) {
          GST_WARNING_OBJECT (nvmultiurisrcbin, "Failed to add uri [%s]\n",
              nvmultiurisrcbin->uriListV[i]);
        } else {
          GST_DEBUG_OBJECT (nvmultiurisrcbin, "Successfully added uri [%s]\n",
              nvmultiurisrcbin->uriListV[i]);
        }
          /** clean the config place-holders that we change for each source */
        nvmultiurisrcbin->config->uri = NULL;
        nvmultiurisrcbin->config->sensorId = NULL;
        nvmultiurisrcbin->config->sensorName = NULL;
      }
      gst_nvmultiurisrcbincreator_sync_children_states (nvmultiurisrcbin->
          nvmultiurisrcbinCreator);
      gst_element_call_async (GST_ELEMENT (nvmultiurisrcbin),
          (GstElementCallAsyncFunc) gst_bin_sync_children_states, NULL, NULL);
    }

    /** Link the internal bin's proxy src pad with the outer bin's src pad
     * Note: This is done even if uri-list was NULL
     * to allow streams to be added later runtime
     */
    {
      gboolean ret =
          gst_ghost_pad_set_target (GST_GHOST_PAD (nvmultiurisrcbin->
              bin_src_pad),
          gst_nvmultiurisrcbincreator_get_source_pad (nvmultiurisrcbin->
              nvmultiurisrcbinCreator));
      if (ret == FALSE) {
        GST_WARNING_OBJECT (nvmultiurisrcbin,
            "Failed to set nvstreammux src pad as bin src pad\n");
      } else {
        GST_DEBUG_OBJECT (nvmultiurisrcbin,
            "Successfully set nvstreammux src pad as bin src pad\n");
      }
    }

    /** Initialize the HTTP server for REST API */
    if (g_strcmp0 (nvmultiurisrcbin->httpPort, "0") != 0) {
      //Valid HTTP port passed; start REST API server
      rest_api_server_start (nvmultiurisrcbin);
      if (!nvmultiurisrcbin->restServer) {
        g_print ("REST Server start failed, "
                 "Users may still have all sources mentioned with uri-list in the pipeline\n");
      }
    } else {
      GST_WARNING_OBJECT (nvmultiurisrcbin,
          "Not starting REST Server as port was passed \"0\"; "
          "Users may still have all sources mentioned with uri-list in the pipeline\n");
    }
  }


  if (transition == GST_STATE_CHANGE_PLAYING_TO_PAUSED) {
    GST_DEBUG_OBJECT (nvmultiurisrcbin,
        "GST_STATE_CHANGE_PLAYING_TO_PAUSED %s %d\n", __func__, __LINE__);
  }
  if (transition == GST_STATE_CHANGE_READY_TO_PAUSED) {
    GST_DEBUG_OBJECT (nvmultiurisrcbin,
        "GST_STATE_CHANGE_READY_TO_PAUSED %s %d\n", __func__, __LINE__);
  }
  if (transition == GST_STATE_CHANGE_PAUSED_TO_PLAYING) {
    GST_DEBUG_OBJECT (nvmultiurisrcbin,
        "GST_STATE_CHANGE_PAUSED_TO_PLAYING %s %d\n", __func__, __LINE__);
  }
  if (transition == GST_STATE_CHANGE_PAUSED_TO_READY) {
    GST_DEBUG_OBJECT (nvmultiurisrcbin,
        "GST_STATE_CHANGE_PAUSED_TO_READY %s %d\n", __func__, __LINE__);
  }
  if (transition == GST_STATE_CHANGE_READY_TO_NULL) {
    GST_DEBUG_OBJECT (nvmultiurisrcbin,
        "GST_STATE_CHANGE_READY_TO_NULL %s %d\n", __func__, __LINE__);
    if (nvmultiurisrcbin->nvmultiurisrcbinCreator) {
      gst_nvmultiurisrcbincreator_deinit (nvmultiurisrcbin->
          nvmultiurisrcbinCreator);
      nvmultiurisrcbin->nvmultiurisrcbinCreator = NULL;
    }
    if (nvmultiurisrcbin->restServer) {
      nvds_rest_server_stop ((NvDsRestServer *) nvmultiurisrcbin->restServer);
      nvmultiurisrcbin->restServer = NULL;
    }
    // Cleanup msgapi if it was initialized
    if (nvmultiurisrcbin->config->enable_error_propagation) {
      msgapi_cleanup();
    }
    /* allow the static binds to be re-forwarded on a subsequent restart */
    nvmultiurisrcbin->staticBindsEmitted = FALSE;
  }
  ret = GST_ELEMENT_CLASS (parent_class)->change_state (element, transition);

  /* Once the pipeline is PLAYING (bin src pad linked + flowing), forward the
   * per-sensor model bindings for statically added (uri-list) sources -- once.
   * Done AFTER the parent transition so the downstream link is fully active. */
  if (transition == GST_STATE_CHANGE_PAUSED_TO_PLAYING
      && ret != GST_STATE_CHANGE_FAILURE
      && !nvmultiurisrcbin->staticBindsEmitted) {
    gst_ds_nvmultiurisrc_bin_forward_static_binds (nvmultiurisrcbin);
    nvmultiurisrcbin->staticBindsEmitted = TRUE;
  }
  return ret;
}

gpointer
GThreadFuncRemoveSource (gpointer data)
{
  DsNvMultiUriBinSourceInfo *sourceInfo = (DsNvMultiUriBinSourceInfo *) data;
  GstDsNvMultiUriBin *ubin = sourceInfo->ubin;
  gboolean ret;

  /**
   * revoking the use of gst_nvmultiurisrcbincreator_remove_source_without_forced_state_change()
   * All test cases in gsttest PASS with this change
   * Note: Multiple calls to gst_nvmultiurisrcbincreator_remove_source() is handled now in the API.
   */
  ret = gst_nvmultiurisrcbincreator_remove_source
      (ubin->nvmultiurisrcbinCreator, sourceInfo->sourceId);
  if (ret == FALSE) {
    GST_WARNING_OBJECT (ubin,
        "Failed to remove sensor; the sensor might have gotten removed already\n");
  } else {
    GST_DEBUG_OBJECT (ubin, "Successfully removed sensor\n");
  }
  g_free (sourceInfo);
  return NULL;
}

static void
gst_ds_nvmultiurisrc_bin_set_init_reconnect_interval (GstDsNvUriSrcBin *
    nvurisrcbin)
{
  if (nvurisrcbin && nvurisrcbin->config &&
      nvurisrcbin->config->init_rtsp_reconnect_interval_sec > 0) {
    // Use the faster initial reconnect timeout only when explicitly configured.
    nvurisrcbin->config->rtsp_reconnect_interval_sec =
        nvurisrcbin->config->init_rtsp_reconnect_interval_sec;
  }
}

static void
gst_ds_nvmultiurisrc_bin_handle_message (GstBin * bin, GstMessage * message)
{
  GstDsNvMultiUriBin *ubin = (GstDsNvMultiUriBin *) bin;
  (void) ubin;

  if (GST_MESSAGE_TYPE (message) == GST_MESSAGE_WARNING) {
    GstDsNvUriSrcBin *nvurisrcbin = NULL;
    // "nvurisrc_bin_src_elem" is the uridecodebin of the nvurisrcbin. "src" is the rtspsrc of the nvurisrcbin. In either case, the nvurisrcbin will be the src->parent
    if (g_strcmp0 (GST_OBJECT_NAME (message->src),
          "nvurisrc_bin_src_elem") == 0
          || g_strcmp0 (GST_OBJECT_NAME (message->src), "src") == 0) {
      nvurisrcbin = (GstDsNvUriSrcBin *) message->src->parent;
    }
    gst_ds_nvmultiurisrc_bin_set_init_reconnect_interval (nvurisrcbin);
  }
  if (GST_MESSAGE_TYPE (message) == GST_MESSAGE_ERROR) {
    gchar *debug = NULL;
    GError *error = NULL;

    gst_message_parse_error (message, &error, &debug);
    g_printerr ("nvmultiurisrcbin ERROR from element %s: %s\n",
        GST_OBJECT_NAME (message->src), error->message);

    gboolean didSourceElemError = FALSE;
    if ((g_strcmp0 (GST_OBJECT_NAME (message->src), "source") == 0
        || g_strcmp0 (GST_OBJECT_NAME (message->src), "src") == 0)
	&& g_strrstr (error->message, "GStreamer error:")) {
      didSourceElemError = TRUE;
    } else if (g_strrstr (error->message, "Invalid URI")) {
      didSourceElemError = TRUE;
    } else if (g_strrstr (error->message, "No URI handler implemented for")) {
      didSourceElemError = TRUE;
    } else if (g_strrstr (error->message, "Resource not found")) {
      didSourceElemError = TRUE;
    } else if (g_strrstr (error->message, "Could not open resource for")) {
      didSourceElemError = TRUE;
    } else if (g_strrstr (error->message, "No supported authentication protocol was found")) {
      didSourceElemError = TRUE;
    } else if (g_strrstr (error->message, "Could not get/set settings from/on resource")) {
      didSourceElemError = TRUE;
    }

    if (didSourceElemError) {
      GST_DEBUG_OBJECT (ubin,
          "Converting ERROR to WARNING; We need to be able to continue adding valid streams\n");
      GST_MESSAGE_TYPE (message) = GST_MESSAGE_WARNING; // Change error to warning, so app can continue running

      // Retrieve the nvurisrcbin corresponding to the source so that we may get the source id and remove the broken stream
      GstDsNvUriSrcBin *nvurisrcbin = NULL;
      // If the source of the message is named "source", this is the filesrc of the uridecodebin of the nvurisrcbin. So, the nvurisrcbin will be the src->parent->parent
      if (g_strcmp0 (GST_OBJECT_NAME (message->src), "source") == 0) {
        nvurisrcbin = (GstDsNvUriSrcBin *) message->src->parent->parent;
      }
      // "nvurisrc_bin_src_elem" is the uridecodebin of the nvurisrcbin. "src" is the rtspsrc of the nvurisrcbin. In either case, the nvurisrcbin will be the src->parent
      else if (g_strcmp0 (GST_OBJECT_NAME (message->src),
              "nvurisrc_bin_src_elem") == 0
          || g_strcmp0 (GST_OBJECT_NAME (message->src), "src") == 0) {
        nvurisrcbin = (GstDsNvUriSrcBin *) message->src->parent;
      }

      gst_ds_nvmultiurisrc_bin_set_init_reconnect_interval (nvurisrcbin);
      // Remove the stream
      if (nvurisrcbin != NULL && nvurisrcbin->config) {
        g_mutex_lock (&ubin->bin_lock);
        GST_DEBUG_OBJECT (ubin, "removing source %d\n",
            nvurisrcbin->config->source_id);
        {
          DsNvMultiUriBinSourceInfo *sourceInfo =
              (DsNvMultiUriBinSourceInfo *)
              g_malloc0 (sizeof (DsNvMultiUriBinSourceInfo));
          sourceInfo->ubin = ubin;
          sourceInfo->sourceId = nvurisrcbin->config->source_id;
          sourceInfo->didSourceElemError = didSourceElemError;
          /** Remove the stream from a separate short lived thread */
          g_thread_new (NULL, GThreadFuncRemoveSource, sourceInfo);
        }
        g_mutex_unlock (&ubin->bin_lock);
      }
    }
    g_error_free (error);
    g_free (debug);
  }
  if (GST_MESSAGE_TYPE (message) == GST_MESSAGE_ELEMENT) {
    if (gst_nvmessage_is_reconnect_attempt_exceeded (message)) {
      NvDsRtspAttemptsInfo rtsp_info = {0};
      gboolean ret = FALSE;
      if (gst_nvmessage_parse_reconnect_attempt_exceeded (message, &rtsp_info)) {
        if (rtsp_info.attempt_exceeded){
          g_print("Reconnection attempt exceeded, removing source %d\n",
            rtsp_info.source_id);
            DsNvMultiUriBinSourceInfo *sourceInfo =
              (DsNvMultiUriBinSourceInfo *)
              g_malloc0 (sizeof (DsNvMultiUriBinSourceInfo));
            sourceInfo->ubin = ubin;
            sourceInfo->sourceId = rtsp_info.source_id;
            ret = gst_nvmultiurisrcbincreator_remove_source
              (ubin->nvmultiurisrcbinCreator, sourceInfo->sourceId);
            if (ret == FALSE) {
              GST_WARNING_OBJECT (ubin,
              "Failed to remove sensor; the sensor might have gotten removed already\n");
              } else {
                GST_DEBUG_OBJECT (ubin, "Successfully removed sensor\n");
            }
            g_free(sourceInfo);
        }
      }
    }
  }

  GST_BIN_CLASS (parent_class)->handle_message (bin, message);
}

static size_t
http_download_write_callback (void *ptr, size_t size, size_t nmemb, FILE *stream)
{
  return fwrite (ptr, size, nmemb, stream);
}

/**
 * Parse ISO 8601 timestamp string to NTP time in nanoseconds.
 * Supports format: "YYYY-MM-DDTHH:MM:SS.sssZ" (e.g., "2024-06-09T18:32:11.123Z")
 *
 * NTP epoch starts at January 1, 1900, while Unix epoch starts at January 1, 1970.
 * The difference is 2,208,988,800 seconds (70 years).
 *
 * @param iso8601_str The ISO 8601 timestamp string
 * @return NTP time in nanoseconds, or 0 if parsing fails
 */
static guint64
parse_iso8601_to_ntp_ns (const gchar *iso8601_str)
{
  if (!iso8601_str || strlen (iso8601_str) == 0) {
    return 0;
  }

  GDateTime *dt = g_date_time_new_from_iso8601 (iso8601_str, NULL);
  if (!dt) {
    g_print ("[WARN] Failed to parse ISO 8601 timestamp: %s\n", iso8601_str);
    return 0;
  }

  /* Get seconds since Unix epoch (1970) */
  gint64 unix_epoch_secs = g_date_time_to_unix (dt);

  /* Get microseconds part */
  gint microseconds = g_date_time_get_microsecond (dt);

  g_date_time_unref (dt);

  /* Convert to NTP epoch by adding 70 years offset */
  guint64 ntp_secs = (guint64) unix_epoch_secs;

  /* Convert to nanoseconds */
  guint64 ntp_ns = ntp_secs * 1000000000ULL + (guint64) microseconds * 1000ULL;

  return ntp_ns;
}

/**
 * Check if an HTTP/HTTPS URL points to a downloadable media file
 * (as opposed to a live/streaming source like HLS, DASH, etc.).
 * Only downloadable file URLs should be fetched via curl; everything
 * else is passed directly to GStreamer which handles it natively.
 */
static gboolean
is_downloadable_http_url (const gchar *url)
{
  static const gchar *downloadable_extensions[] = {
    ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv",
    ".webm", ".mpg", ".mpeg", ".3gp", ".ts", ".mts",
    ".m4v", ".vob", ".ogv", NULL
  };

  if (!url)
    return FALSE;

  /* Extract path portion (before '?' query params) */
  const gchar *query = strchr (url, '?');
  gsize path_len = query ? (gsize) (query - url) : strlen (url);

  /* Check if the URL path ends with a known media file extension */
  for (gint i = 0; downloadable_extensions[i] != NULL; i++) {
    gsize ext_len = strlen (downloadable_extensions[i]);
    if (path_len > ext_len) {
      const gchar *path_end = url + path_len - ext_len;
      if (g_ascii_strncasecmp (path_end, downloadable_extensions[i], ext_len) == 0)
        return TRUE;
    }
  }

  return FALSE;
}

static gboolean
download_http_file (GstDsNvMultiUriBin *nvmultiurisrcbin, const gchar *url,
    const gchar *camera_id, gchar **local_file_uri)
{
  CURL *curl;
  FILE *fp;
  CURLcode res;
  gchar *filename = NULL;
  gchar *local_path = NULL;
  const gchar *url_filename = NULL;

  *local_file_uri = NULL;

  url_filename = g_strrstr (url, "/");
  if (url_filename && strlen (url_filename) > 1) {
    url_filename++;
    /* Strip query params (?...) and fragment (#...) from filename */
    const gchar *query = strchr (url_filename, '?');
    const gchar *fragment = strchr (url_filename, '#');
    gsize name_len = strlen (url_filename);
    if (query)
      name_len = MIN (name_len, (gsize) (query - url_filename));
    if (fragment)
      name_len = MIN (name_len, (gsize) (fragment - url_filename));

    if (name_len > 0)
      filename = g_strndup (url_filename, name_len);
  }

  /* Fallback: generate a unique name using camera_id + timestamp */
  if (!filename || strlen (filename) == 0) {
    g_free (filename);
    GDateTime *now = g_date_time_new_now_local ();
    gchar *timestamp = g_date_time_format (now, "%Y%m%d_%H%M%S");
    g_date_time_unref (now);
    filename = g_strdup_printf ("stream_%s_%s.mp4",
        (camera_id && strlen (camera_id) > 0) ? camera_id : "unknown",
        timestamp);
    g_free (timestamp);
  }

  /* Truncate filename if it exceeds NAME_MAX (255) */
  if (strlen (filename) > 255) {
    const gchar *ext = g_strrstr (filename, ".");
    if (ext && (gsize) (ext - filename) < 255 && strlen (ext) < 20) {
      gsize stem_max = 255 - strlen (ext);
      gchar *truncated_stem = g_strndup (filename, stem_max);
      gchar *new_filename = g_strdup_printf ("%s%s", truncated_stem, ext);
      g_free (truncated_stem);
      g_free (filename);
      filename = new_filename;
    } else {
      gchar *truncated = g_strndup (filename, 255);
      g_free (filename);
      filename = truncated;
    }
    GST_WARNING_OBJECT (nvmultiurisrcbin,
        "Filename from URL was too long, truncated to: %s\n", filename);
  }

  local_path = g_strdup_printf ("%s%s", HTTP_DOWNLOAD_DIR, filename);
  g_free (filename);

  GST_DEBUG_OBJECT (nvmultiurisrcbin, "Downloading %s to %s\n", url, local_path);
  g_print ("[HTTP_DOWNLOAD] Downloading %s to %s\n", url, local_path);

  curl = curl_easy_init ();
  if (!curl) {
    GST_ERROR_OBJECT (nvmultiurisrcbin, "Failed to initialize curl\n");
    g_free (local_path);
    return FALSE;
  }

  fp = fopen (local_path, "wb");
  if (!fp) {
    GST_ERROR_OBJECT (nvmultiurisrcbin, "Failed to open file %s for writing\n", local_path);
    curl_easy_cleanup (curl);
    g_free (local_path);
    return FALSE;
  }

  curl_easy_setopt (curl, CURLOPT_URL, url);
  curl_easy_setopt (curl, CURLOPT_WRITEFUNCTION, http_download_write_callback);
  curl_easy_setopt (curl, CURLOPT_WRITEDATA, fp);
  curl_easy_setopt (curl, CURLOPT_FOLLOWLOCATION, 1L);
  curl_easy_setopt (curl, CURLOPT_SSL_VERIFYPEER, 0L);
  curl_easy_setopt (curl, CURLOPT_SSL_VERIFYHOST, 0L);
  /* Timeouts (seconds) are configurable via the "http-download-timeout" and
   * "http-connect-timeout" properties; a value of 0 means "no limit". */
  long http_timeout = (long) nvmultiurisrcbin->httpDownloadTimeout;
  long connect_timeout = (long) nvmultiurisrcbin->httpConnectTimeout;
  curl_easy_setopt (curl, CURLOPT_TIMEOUT, http_timeout);
  curl_easy_setopt (curl, CURLOPT_CONNECTTIMEOUT, connect_timeout);
  GST_DEBUG_OBJECT (nvmultiurisrcbin,
      "HTTP download timeouts: total=%lds connect=%lds",
      http_timeout, connect_timeout);

  res = curl_easy_perform (curl);

  fclose (fp);

  if (res != CURLE_OK) {
    GST_ERROR_OBJECT (nvmultiurisrcbin, "curl download failed: %s\n", curl_easy_strerror (res));
    g_print ("[HTTP_DOWNLOAD_ERROR] curl download failed: %s\n", curl_easy_strerror (res));
    curl_easy_cleanup (curl);
    remove (local_path);
    g_free (local_path);
    return FALSE;
  }

  /* Check HTTP response code */
  long http_code = 0;
  curl_easy_getinfo (curl, CURLINFO_RESPONSE_CODE, &http_code);
  curl_easy_cleanup (curl);

  if (http_code != 200) {
    GST_ERROR_OBJECT (nvmultiurisrcbin, "HTTP download failed with status code: %ld\n", http_code);
    g_print ("[HTTP_DOWNLOAD_ERROR] HTTP download failed with status code: %ld\n", http_code);
    remove (local_path);
    g_free (local_path);
    return FALSE;
  }

  *local_file_uri = g_strdup_printf ("file://%s", local_path);
  g_print ("[HTTP_DOWNLOAD] Successfully downloaded to %s\n", local_path);
  g_free (local_path);

  return TRUE;
}

/* Effective concurrent-download limit. Defaults to (0) and is always clamped to
 * max-batch-size: there is no point fetching more files at once than the
 * pipeline can ever hold as active sources. Returns 0 only if max-batch-size is
 * itself 0 (treated as unlimited). */
static guint
nv_http_download_limit (GstDsNvMultiUriBin * self)
{
  guint batch = self->muxConfig->maxBatchSize;
  guint lim = self->httpMaxConcurrentDownloads;
  if (lim == 0 || (batch > 0 && lim > batch))
    lim = batch;
  return lim;
}

/* Counting semaphore bounding concurrent HTTP(S) downloads to
 * nv_http_download_limit(). acquire() blocks the calling REST worker thread
 * while the limit is reached, WITHOUT holding bin_lock, so other adds proceed
 * and downloads run in parallel up to the cap. */
static void
nv_http_download_acquire_slot (GstDsNvMultiUriBin * self)
{
  guint lim = nv_http_download_limit (self);
  if (lim == 0)
    return;
  g_mutex_lock (&self->httpDlLock);
  while (self->httpActiveDownloads >= lim)
    g_cond_wait (&self->httpDlCond, &self->httpDlLock);
  self->httpActiveDownloads++;
  g_mutex_unlock (&self->httpDlLock);
}

static void
nv_http_download_release_slot (GstDsNvMultiUriBin * self)
{
  if (nv_http_download_limit (self) == 0)
    return;
  g_mutex_lock (&self->httpDlLock);
  if (self->httpActiveDownloads > 0)
    self->httpActiveDownloads--;
  g_cond_signal (&self->httpDlCond);
  g_mutex_unlock (&self->httpDlLock);
}

/* Native REST model-plane handler (/api/v1/model/load|unload|update): forward
 * the request's "value" object VERBATIM (JSON) in-band as a serialized
 * downstream event on the bin src pad -- the same data-plane path the
 * enc/dec/roi REST APIs use, so it reaches the downstream model-managing
 * element in ANY host app, independent of who owns the pipeline bus.
 * nvmultiurisrcbin owns no model bins and adds nothing here: model-plane
 * requests carry no stream scope by design (routing is s_route_api_impl). */
/* Map a nvmodelmux control-query verdict onto the REST response. Shared by every
 * model-plane op (load / unload / update): the element answers {http, err_code,
 * reason, hint} -- 202 => accepted (may still warm asynchronously), anything else
 * => rejected, and we surface the element's OWN reason to the caller verbatim.
 *   @answered  did a downstream element actually handle the query?
 *   @ok/@fail  the op's success/failure status enums (for the async ledger). */
/* Write the REST response from a nvmodelmux control-query verdict. Shared by every
 * model-plane op (load / unload / update): the element answers {http_code,
 * err_code, reason, hint} -- 202 => accepted (may still warm asynchronously),
 * anything else => rejected, and we surface the element's OWN reason to the caller
 * verbatim.
 *   @element_answered  did a downstream element actually handle the query?
 *   @success_status / @failure_status  the op's status enums (async ledger). */
static void
s_set_response_from_verdict (NvDsServerModelInfo * model_info, GstQuery * query,
    bool element_answered, NvDsServerModelStatus success_status,
    NvDsServerModelStatus failure_status, const char *op_name)
{
  gint http_code = 0;
  const gchar *err_code = NULL, *reason = NULL, *hint = NULL;

  if (element_answered
      && gst_nvquery_control_parse_response (query, &http_code, &err_code, &reason,
          &hint)) {
    if (http_code == 202) {                   /* admitted */
      model_info->status = success_status;
      model_info->err_info.code = StatusAccepted;
      model_info->model_log = (reason && *reason) ? reason : "request accepted";
    } else {                                  /* rejected -- element's own reason */
      model_info->status = failure_status;
      model_info->err_info.code = StatusBadRequest;
      model_info->err_info.err_code =
          (err_code && *err_code) ? err_code : "REQUEST_REJECTED";
      model_info->model_log = (reason && *reason) ? reason : "request rejected";
      if (hint && *hint)
        model_info->err_info.hint = hint;
    }
  } else {                                    /* nothing downstream answered */
    model_info->status = failure_status;
    model_info->err_info.code = StatusInternalServerError;
    model_info->err_info.err_code = "REQUEST_FAILED";
    model_info->model_log = std::string (op_name) + ": no model element answered "
        "(is nvmodelmux present and the pipeline PLAYING?)";
    model_info->err_info.hint =
        "ensure nvmodelmux is downstream and the pipeline is PLAYING";
  }
}

/* Native REST /api/v1/model/{load,unload,update}: forward the request to the
 * model element as a SYNCHRONOUS control query and return the element's real
 * verdict (accept + context, or reject + a specific reason) -- not a blind 202.
 * The element decides admission on the spot; any slow work (engine warm, TRT
 * teardown, OTA swap) still runs asynchronously inside the element. */
static void
s_model_api_impl (NvDsServerModelInfo * model_info, void *ctx)
{
  GstDsNvMultiUriBin *nvmultiurisrcbin = (GstDsNvMultiUriBin *) ctx;
  GstPad *src_pad = (GstPad *) (nvmultiurisrcbin->bin_src_pad);
  bool is_load = model_info->uri.find ("/model/load") != std::string::npos;
  bool is_update = model_info->uri.find ("/model/update") != std::string::npos;

  const char *op_name = is_load ? "model/load"
      : is_update ? "model/update" : "model/unload";
  NvDsServerModelStatus success_status = is_load ? MODEL_LOAD_SUCCESS
      : is_update ? MODEL_UPDATE_SUCCESS : MODEL_UNLOAD_SUCCESS;
  NvDsServerModelStatus failure_status = is_load ? MODEL_LOAD_FAIL
      : is_update ? MODEL_UPDATE_FAIL : MODEL_UNLOAD_FAIL;

  GstQuery *query = gst_nvquery_control_new (op_name, model_info->value_json.c_str ());
  bool element_answered = src_pad && gst_pad_peer_query (src_pad, query);
  s_set_response_from_verdict (model_info, query, element_answered,
      success_status, failure_status, op_name);
  gst_query_unref (query);
}

/* Native REST /api/v1/stream/route handler (the declarative stream-routing
 * request). nvmultiurisrcbin owns the camera_id -> source_id map, so ATOMIC
 * ADMISSION for stream existence happens here: every camera_id in every route
 * must resolve to a live source or the WHOLE request is rejected (an invalid
 * request must never partially apply). Each route entry is AUGMENTED with the
 * resolved "source_ids" array, then the request is forwarded in-band. Routes
 * with "streams": "all" are forwarded unresolved (the consuming element owns
 * the set of ATTACHED streams, which is the correct meaning of "all"). */
static void
s_route_api_impl (NvDsServerRouteInfo * route_info, void *ctx)
{
  GstDsNvMultiUriBin *nvmultiurisrcbin = (GstDsNvMultiUriBin *) ctx;
  Json::Value value;
  Json::CharReaderBuilder rbuilder;
  std::string errs;
  std::unique_ptr < Json::CharReader > reader (rbuilder.newCharReader ());

  if (!reader->parse (route_info->value_json.c_str (),
          route_info->value_json.c_str () + route_info->value_json.size (),
          &value, &errs)) {
    route_info->status = STREAM_ROUTE_FAIL;
    route_info->route_log = "STREAM_ROUTE_FAIL, internal payload re-parse error";
    route_info->err_info.code = StatusInternalServerError;
    route_info->err_info.err_code = "REQUEST_FAILED";
    return;
  }

  /* resolve camera_ids -> source_ids per route, under the bin lock */
  g_mutex_lock (&nvmultiurisrcbin->bin_lock);
  for (size_t r = 0; r < route_info->routes.size (); ++r) {
    const NvDsServerRouteEntry & entry = route_info->routes[r];
    if (entry.all_streams)
      continue;
    Json::Value resolved (Json::arrayValue);
    for (size_t i = 0; i < entry.camera_ids.size (); ++i) {
      GstDsNvUriSrcConfig *sc =
          gst_nvmultiurisrcbincreator_get_source_config_by_sensorid
          (nvmultiurisrcbin->nvmultiurisrcbinCreator,
          entry.camera_ids[i].c_str ());
      if (!sc) {
        /* ATOMIC ADMISSION: one unknown camera_id rejects the whole request */
        g_mutex_unlock (&nvmultiurisrcbin->bin_lock);
        route_info->status = STREAM_ROUTE_FAIL;
        route_info->route_log = "STREAM_ROUTE_FAIL, unknown stream '" +
            entry.camera_ids[i] + "' (no live source with that camera_id)";
        route_info->err_info.code = StatusBadRequest;
        route_info->err_info.err_code = "STREAM_UNKNOWN";
        /* point at stream/route's GET, not model/status: model/status reports the
         * element's DISPLAY name under its camera_id key, which this API will not
         * route by. stream/route's read returns the routable camera_ids. */
        route_info->err_info.hint = "add the stream first (stream/add), or "
            "check GET /api/v1/stream/route for the routable camera ids "
            "(model/status reports the display name, which is not routable)";
        return;
      }
      resolved.append (sc->source_id);
      gst_nvmultiurisrcbincreator_src_config_free (sc);
    }
    value["routes"][(Json::ArrayIndex) r]["source_ids"] = resolved;
  }
  g_mutex_unlock (&nvmultiurisrcbin->bin_lock);

  Json::StreamWriterBuilder wbuilder;
  wbuilder["indentation"] = "";
  std::string augmented = Json::writeString (wbuilder, value);

  /* SYNCHRONOUS admission: ask the model element to validate the route NOW and
   * return its real verdict (unknown model ref? if_revision conflict? bad
   * scope?) instead of a blind 202. The actual reroute still settles async. The
   * stream-existence check (STREAM_UNKNOWN) above stays here -- only nvmultiuri
   * owns the camera_id -> source_id map. */
  GstQuery *query = gst_nvquery_control_new ("stream/route", augmented.c_str ());
  GstPad *src_pad = (GstPad *) (nvmultiurisrcbin->bin_src_pad);
  gint http_code = 0;
  const gchar *err_code = NULL, *reason = NULL, *hint = NULL;
  bool answered = src_pad && gst_pad_peer_query (src_pad, query)
      && gst_nvquery_control_parse_response (query, &http_code, &err_code, &reason,
          &hint);

  if (answered && http_code == 202) {           /* admitted */
    route_info->status = STREAM_ROUTE_SUCCESS;
    route_info->err_info.code = StatusAccepted;
    route_info->route_log = (reason && *reason) ? reason : "stream/route accepted";
  } else if (answered) {                        /* rejected -- element's reason */
    route_info->status = STREAM_ROUTE_FAIL;
    route_info->err_info.code = StatusBadRequest;
    route_info->err_info.err_code =
        (err_code && *err_code) ? err_code : "STREAM_ROUTE_REJECTED";
    route_info->route_log = (reason && *reason) ? reason : "stream/route rejected";
    if (hint && *hint)
      route_info->err_info.hint = hint;
  } else {                                      /* nothing downstream answered */
    route_info->status = STREAM_ROUTE_FAIL;
    route_info->err_info.code = StatusInternalServerError;
    route_info->err_info.err_code = "REQUEST_FAILED";
    route_info->route_log = "stream/route: no model element answered "
        "(is nvmodelmux present and the pipeline PLAYING?)";
    route_info->err_info.hint =
        "ensure nvmodelmux is downstream and the pipeline is PLAYING";
  }
  gst_query_unref (query);
}

static void
s_stream_api_impl (NvDsServerStreamInfo * stream_info, void *ctx)
{

  GstDsNvMultiUriBin *nvmultiurisrcbin = (GstDsNvMultiUriBin *) ctx;
  (void) nvmultiurisrcbin;

  g_mutex_lock (&nvmultiurisrcbin->bin_lock);

  if (stream_info->uri.find ("/api/v1/") != std::string::npos) {
    /** check stream_info->value_change to identify stream add/remove */
    if (g_strrstr (stream_info->value_change.c_str (), "add")
        || g_strrstr (stream_info->value_change.c_str (), "streaming")) {

      GstDsNvUriSrcConfig **sourceConfigs = NULL;
      guint numSourceConfigs = 0;
      /** Check if we can accomodate more sources */
      if (gst_nvmultiurisrcbincreator_get_active_sources_list (nvmultiurisrcbin->
              nvmultiurisrcbinCreator, &numSourceConfigs, &sourceConfigs)) {
        gst_nvmultiurisrcbincreator_src_config_list_free (nvmultiurisrcbin->
            nvmultiurisrcbinCreator, numSourceConfigs, sourceConfigs);
        if (numSourceConfigs >= nvmultiurisrcbin->muxConfig->maxBatchSize) {
          /* VIA-S-5: batch full. RTSP is a live source that never self-EOS's, so
           * keep the fail-fast reject for it. File/HTTP overflow is instead
           * queued by the creator's add_source and admitted in waves as running
           * sources finish (EOS) and free slots -- so let it fall through to the
           * add path below. */
          gboolean isRtspOverflow =
              g_str_has_prefix (stream_info->value_camera_url.c_str (), "rtsp://")
              || g_str_has_prefix (stream_info->value_camera_url.c_str (), "rtsps://");
          if (isRtspOverflow) {
            GST_WARNING_OBJECT (nvmultiurisrcbin, "Failed to add sensor id=[%s]; "
                "We have [%d] active sources and max-batch-size is configured to [%d]\n",
                stream_info->value_camera_id.c_str (),
                numSourceConfigs, nvmultiurisrcbin->muxConfig->maxBatchSize);
            stream_info->status = STREAM_ADD_FAIL;
            stream_info->stream_log = "STREAM_ADD_FAIL, Active sources exceded max-batch-size of nvstreammux";
            stream_info->err_info.code = StatusInternalServerError;
            if (nvmultiurisrcbin->config->enable_error_propagation) {
              msgapi_send_message(nvmultiurisrcbin, stream_info->value_camera_id.c_str(), "N/A", stream_info->stream_log.c_str(),FALSE);
            }
            g_mutex_unlock (&nvmultiurisrcbin->bin_lock);
            return;
          }
        }
      }

      /** Check if sensor id already exist */
      GstDsNvUriSrcConfig *sourceConfig = NULL;
      if ((sourceConfig =
              gst_nvmultiurisrcbincreator_get_source_config_by_sensorid
              (nvmultiurisrcbin->nvmultiurisrcbinCreator,
                  stream_info->value_camera_id.c_str ()))) {
        GST_WARNING_OBJECT (nvmultiurisrcbin,
            "Failed to add sensor id=[%s]; Already added\n",
            stream_info->value_camera_id.c_str ());
        stream_info->status = STREAM_ADD_FAIL;
        stream_info->stream_log = "STREAM_ADD_FAIL, Duplicate Camera id, unable to add stream";
        stream_info->err_info.code = StatusInternalServerError;
        if (nvmultiurisrcbin->config->enable_error_propagation) {
          msgapi_send_message(nvmultiurisrcbin, stream_info->value_camera_id.c_str(), "N/A", stream_info->stream_log.c_str(),FALSE);
        }
        g_mutex_unlock (&nvmultiurisrcbin->bin_lock);
        gst_nvmultiurisrcbincreator_src_config_free (sourceConfig);
        return;
      }

      /** Add the source */
      gchar *actual_uri = NULL;
      gboolean is_http = g_str_has_prefix (stream_info->value_camera_url.c_str (), "http://");
      gboolean is_https = g_str_has_prefix (stream_info->value_camera_url.c_str (), "https://");
      gboolean is_http_url = is_http || is_https;
      gboolean is_downloadable = is_downloadable_http_url (stream_info->value_camera_url.c_str ());
      if (is_http_url && is_downloadable) {
        gchar *local_file_uri = NULL;
        /* Download WITHOUT holding bin_lock so concurrent stream/add requests
         * fetch their files in parallel (bounded by http-max-concurrent-
         * downloads); re-acquire the lock for the actual pipeline add. */
        g_mutex_unlock (&nvmultiurisrcbin->bin_lock);
        nv_http_download_acquire_slot (nvmultiurisrcbin);
        gboolean dl_ok = download_http_file (nvmultiurisrcbin,
            stream_info->value_camera_url.c_str (),
            stream_info->value_camera_id.c_str (), &local_file_uri);
        nv_http_download_release_slot (nvmultiurisrcbin);
        g_mutex_lock (&nvmultiurisrcbin->bin_lock);
        if (dl_ok) {
          actual_uri = local_file_uri;
          GST_DEBUG_OBJECT (nvmultiurisrcbin,
              "HTTP/HTTPS file downloaded, using local file: %s\n", actual_uri);
        } else {
          GST_WARNING_OBJECT (nvmultiurisrcbin,
              "Failed to download HTTP/HTTPS URL: %s\n", stream_info->value_camera_url.c_str ());
          stream_info->status = STREAM_ADD_FAIL;
          stream_info->stream_log = "STREAM_ADD_FAIL, Failed to download HTTP/HTTPS file";
          stream_info->err_info.code = StatusInternalServerError;
          if (nvmultiurisrcbin->config->enable_error_propagation) {
            msgapi_send_message (nvmultiurisrcbin, stream_info->value_camera_id.c_str (), "N/A",
                stream_info->stream_log.c_str (), FALSE);
          }
          g_mutex_unlock (&nvmultiurisrcbin->bin_lock);
          return;
        }
      } else {
        actual_uri = g_strdup (stream_info->value_camera_url.c_str ());
        if (is_http_url) {
          GST_DEBUG_OBJECT (nvmultiurisrcbin,
              "Non-file HTTP/HTTPS URL, passing directly to GStreamer: %s\n",
              actual_uri);
        }
      }

      nvmultiurisrcbin->config->uri = actual_uri;
      nvmultiurisrcbin->config->sensorId =
          (gchar *) stream_info->value_camera_id.c_str ();
      nvmultiurisrcbin->config->sensorName =
          (gchar *) stream_info->value_camera_name.c_str ();
      /* Forward the full per-stream metadata (raw JSON) so it reaches the app via
       * the stream-add bus message (NvDsSensorInfo.sensor_metadata) and downstream.
       * Borrowed pointer here; the creator deep-copies it into the source config. */
      nvmultiurisrcbin->config->sensorMetadata =
          stream_info->metadata_json.empty ()? NULL :
          (gchar *) stream_info->metadata_json.c_str ();
      nvmultiurisrcbin->config->source_id = 0;

      /* Parse and set creation_time if provided */
      if (!stream_info->value_creation_time.empty ()) {
        nvmultiurisrcbin->config->creation_time_ns =
            parse_iso8601_to_ntp_ns (stream_info->value_creation_time.c_str ());
        GST_DEBUG_OBJECT (nvmultiurisrcbin,
            "Parsed creation_time [%s] to %" G_GUINT64_FORMAT " ns\n",
            stream_info->value_creation_time.c_str (),
            nvmultiurisrcbin->config->creation_time_ns);
      } else {
        nvmultiurisrcbin->config->creation_time_ns = 0;
      }

      gboolean ret =
          gst_nvmultiurisrcbincreator_add_source (nvmultiurisrcbin->
          nvmultiurisrcbinCreator, nvmultiurisrcbin->config);
      if (ret == FALSE) {
        GST_WARNING_OBJECT (nvmultiurisrcbin,
            "Failed to add sensor id=[%s] uri=[%s]\n",
            nvmultiurisrcbin->config->sensorId, nvmultiurisrcbin->config->uri);
        stream_info->status = STREAM_ADD_FAIL;
        stream_info->stream_log = "STREAM_ADD_FAIL, Failed to add source stream";
        stream_info->err_info.code = StatusInternalServerError;
        if (nvmultiurisrcbin->config->enable_error_propagation) {
          msgapi_send_message(nvmultiurisrcbin, stream_info->value_camera_id.c_str(), "N/A", stream_info->stream_log.c_str(),FALSE);
        }
      } else {
        GST_DEBUG_OBJECT (nvmultiurisrcbin,
            "Successfully added sensor id=[%s] uri=[%s]\n",
            nvmultiurisrcbin->config->sensorId, nvmultiurisrcbin->config->uri);
        stream_info->status = STREAM_ADD_SUCCESS;
        stream_info->stream_log = "STREAM_ADD_SUCCESS";
        stream_info->err_info.code = StatusOk;

        /* Forward the per-stream model binding IN-BAND as a serialized
         * downstream event (same data-plane path as the model load/unload/
         * update events), so a model element can apply per-stream selection even
         * when the host owns the bus (the stream-add bus message never reaches
         * it then; PAD_ADDED carries only source_id). Keyed by the assigned
         * source_id so the element correlates it with PAD_ADDED. */
        GstDsNvUriSrcConfig *sc =
            gst_nvmultiurisrcbincreator_get_source_config_by_sensorid
            (nvmultiurisrcbin->nvmultiurisrcbinCreator,
            stream_info->value_camera_id.c_str ());
        if (sc) {
          GstEvent *bev = gst_nvevent_new_stream_model_bind (sc->source_id,
              stream_info->value_camera_id.c_str (),
              stream_info->value_camera_name.c_str (),
              stream_info->metadata_json.empty ()? NULL :
              stream_info->metadata_json.c_str ());
          if (bev)
            gst_pad_push_event ((GstPad *) (nvmultiurisrcbin->bin_src_pad), bev);
          gst_nvmultiurisrcbincreator_src_config_free (sc);
        }
      }
      gst_nvmultiurisrcbincreator_sync_children_states (nvmultiurisrcbin->
          nvmultiurisrcbinCreator);
      gst_element_call_async (GST_ELEMENT (nvmultiurisrcbin),
          (GstElementCallAsyncFunc) gst_bin_sync_children_states, NULL, NULL);
      /** clean the config place-holders that we change for each source */
      g_free (actual_uri);
      nvmultiurisrcbin->config->uri = NULL;
      nvmultiurisrcbin->config->sensorId = NULL;
    } else if (g_strrstr (stream_info->value_change.c_str (), "remove")
        || stream_info->uri.find ("/stream/remove") != std::string::npos) {
      /** Remove lookup with guarded fallback:
       * 1) Prefer legacy URL+camera_id when camera_url is present (normal
       *    RTSP/file flows where add/remove URLs match).
       * 2) Fall back to camera_id-only when URL is empty (VST camera_remove)
       *    or when HTTP/HTTPS download rewrote the stored URI to file:// so
       *    the original request URL no longer matches. */
      GstDsNvUriSrcConfig const *sourceConfig = NULL;
      if (!stream_info->value_camera_url.empty ()) {
        sourceConfig =
            gst_nvmultiurisrcbincreator_get_source_config (nvmultiurisrcbin->
            nvmultiurisrcbinCreator,
            stream_info->value_camera_url.c_str (),
            stream_info->value_camera_id.c_str ());
      }
      if (!sourceConfig) {
        sourceConfig =
            gst_nvmultiurisrcbincreator_get_source_config_by_sensorid
            (nvmultiurisrcbin->nvmultiurisrcbinCreator,
            stream_info->value_camera_id.c_str ());
      }
      if (sourceConfig) {
        gboolean ret =
            gst_nvmultiurisrcbincreator_remove_source (nvmultiurisrcbin->
            nvmultiurisrcbinCreator,
            sourceConfig->source_id);
        //Note: after call to gst_nvmultiurisrcbincreator_remove_source, sourceConfig will be invalid
        gst_nvmultiurisrcbincreator_src_config_free((GstDsNvUriSrcConfig*)sourceConfig);
        if (ret == FALSE) {
          GST_WARNING_OBJECT (nvmultiurisrcbin, "Failed to remove sensor\n");
          stream_info->status = STREAM_REMOVE_FAIL;
          stream_info->stream_log = "STREAM_REMOVE_FAIL,Failed to remove source stream";
          stream_info->err_info.code = StatusInternalServerError;
          if (nvmultiurisrcbin->config->enable_error_propagation) {
            msgapi_send_message(nvmultiurisrcbin, stream_info->value_camera_id.c_str(), "N/A", stream_info->stream_log.c_str(),FALSE);
          }
        } else {
          GST_DEBUG_OBJECT (nvmultiurisrcbin, "Successfully removed sensor\n");
          stream_info->status = STREAM_REMOVE_SUCCESS;
          stream_info->stream_log = "STREAM_REMOVE_SUCCESS";
          stream_info->err_info.code = StatusOk;
          gst_nvmultiurisrcbincreator_sync_children_states (nvmultiurisrcbin->
              nvmultiurisrcbinCreator);
          gst_element_call_async (GST_ELEMENT (nvmultiurisrcbin),
              (GstElementCallAsyncFunc) gst_bin_sync_children_states, NULL, NULL);
        }
      } else {
        GST_WARNING_OBJECT (nvmultiurisrcbin,
            "No record found; Failed to remove sensor id=[%s]\n",
            stream_info->value_camera_id.c_str ());
        stream_info->status = STREAM_REMOVE_FAIL;
        stream_info->stream_log = "STREAM_REMOVE_FAIL, No record found. Failed to remove source stream";
        stream_info->err_info.code = StatusInternalServerError;
        if (nvmultiurisrcbin->config->enable_error_propagation) {
          msgapi_send_message(nvmultiurisrcbin, stream_info->value_camera_id.c_str(), "N/A", stream_info->stream_log.c_str(),FALSE);
        }
      }
    } else {
      GST_WARNING_OBJECT (nvmultiurisrcbin,
          "Sensor API change string not supported\n");
      stream_info->stream_log = "Sensor API change string not supported";
      stream_info->err_info.code = StatusBadRequest;
      if (nvmultiurisrcbin->config->enable_error_propagation) {
        msgapi_send_message(nvmultiurisrcbin, stream_info->value_camera_id.c_str(), "N/A", stream_info->stream_log.c_str(),FALSE);
      }
    }
  } else {
    g_print ("Unsupported REST API version \n");
    stream_info->err_info.code = StatusBadRequest;
  }
  g_mutex_unlock (&nvmultiurisrcbin->bin_lock);
  return;
}

static void
s_roi_api_impl (NvDsServerRoiInfo * roi_info, void *ctx)
{
  GstDsNvMultiUriBin *nvmultiurisrcbin = (GstDsNvMultiUriBin *) ctx;
  (void) nvmultiurisrcbin;

  guint sourceId = std::stoi (roi_info->stream_id);

  if (roi_info->uri.find ("/api/v1/") != std::string::npos) {
    if (!find_source (nvmultiurisrcbin->nvmultiurisrcbinCreator, sourceId)) {
      roi_info->status = ROI_UPDATE_FAIL;
      roi_info->err_info.code = StatusInternalServerError;
      roi_info->roi_log = "ROI_UPDATE_FAIL, Unable to find stream id for ROI updation";
      if (nvmultiurisrcbin->config->enable_error_propagation) {
        msgapi_send_message(nvmultiurisrcbin, "N/A", "N/A", roi_info->roi_log.c_str(),TRUE);
      }
    } else {
      gchar* sensorId = gst_nvmultiurisrcbincreator_get_sensor_id_from_source_id(nvmultiurisrcbin->nvmultiurisrcbinCreator,sourceId);
      RoiDimension roi_dim[roi_info->roi_count];

      for (int i = 0; i < (int) roi_info->roi_count; i++) {
        if ( g_strlcpy (roi_dim[i].roi_id, roi_info->vect[i].roi_id,
            sizeof (roi_dim[i].roi_id)) < sizeof (roi_dim[i].roi_id) ){
          GST_DEBUG_OBJECT (nvmultiurisrcbin, "Failed to copy roi id(s) \n");
        }
        roi_dim[i].left = roi_info->vect[i].left;
        roi_dim[i].top = roi_info->vect[i].top;
        roi_dim[i].width = roi_info->vect[i].width;
        roi_dim[i].height = roi_info->vect[i].height;
      }

      GstEvent *nvevent =
          gst_nvevent_new_roi_update ((char *) roi_info->stream_id.c_str (),
          roi_info->roi_count, roi_dim);

      if (!nvevent) {
        roi_info->roi_log = "ROI_UPDATE_FAIL, nv-roi-update event creation failed";
        roi_info->status = ROI_UPDATE_FAIL;
        roi_info->err_info.code = StatusInternalServerError;
        if (nvmultiurisrcbin->config->enable_error_propagation) {
          msgapi_send_message(nvmultiurisrcbin, sensorId, "N/A", roi_info->roi_log.c_str(),FALSE);
        }
      }

      if (!gst_pad_push_event ((GstPad *) (nvmultiurisrcbin->bin_src_pad),
              nvevent)) {
        switch (roi_info->roi_flag) {
          case ROI_UPDATE:
            g_print ("[WARN] nv-roi-update event not pushed downstream.. !! \n");
            roi_info->roi_log = "ROI_UPDATE_FAIL, nv-roi-update event not pushed";
            roi_info->status = ROI_UPDATE_FAIL;
            roi_info->err_info.code = StatusInternalServerError;
            if (nvmultiurisrcbin->config->enable_error_propagation) {
              msgapi_send_message(nvmultiurisrcbin, sensorId, "N/A", roi_info->roi_log.c_str(),FALSE);
            }
            break;
          default:
            break;
        }
      } else {
        switch (roi_info->roi_flag) {
          case ROI_UPDATE:
            roi_info->status = ROI_UPDATE_SUCCESS;
            roi_info->roi_log = "ROI_UPDATE_SUCCESS";
            roi_info->err_info.code = StatusOk;
            break;
          default:
            break;
        }
      }
    }

    gst_nvmultiurisrcbincreator_sync_children_states (nvmultiurisrcbin->
        nvmultiurisrcbinCreator);
    gst_element_call_async (GST_ELEMENT (nvmultiurisrcbin),
        (GstElementCallAsyncFunc) gst_bin_sync_children_states, NULL, NULL);
  } else {
    g_print ("Unsupported REST API version \n");
    roi_info->err_info.code = StatusBadRequest;
  }

}

static void
s_dec_api_impl (NvDsServerDecInfo * dec_info, void *ctx)
{

  GstDsNvMultiUriBin *nvmultiurisrcbin = (GstDsNvMultiUriBin *) ctx;
  (void) nvmultiurisrcbin;
  guint sourceId = std::stoi (dec_info->stream_id);
  gchar* sensorId = gst_nvmultiurisrcbincreator_get_sensor_id_from_source_id(nvmultiurisrcbin->nvmultiurisrcbinCreator,sourceId);
  /* Default to 500 until a branch sets the real status; prevents a silent
   * 200 OK on an unhandled/unknown decoder flag. */
  dec_info->err_info.code = StatusInternalServerError;
  if (dec_info->uri.find ("/api/v1/") != std::string::npos) {
    if (!set_nvuribin_dec_prop (nvmultiurisrcbin->nvmultiurisrcbinCreator,
            sourceId, dec_info)) {
      switch (dec_info->dec_flag) {
        case DROP_FRAME_INTERVAL:
          g_print ("[WARN] drop-frame-interval not set on decoder .. !! \n");
          dec_info->status = DROP_FRAME_INTERVAL_UPDATE_FAIL;
          dec_info->dec_log = "DROP_FRAME_INTERVAL_UPDATE_FAIL, drop-frame-interval not set on decoder";
          dec_info->err_info.code = StatusInternalServerError;
          if (nvmultiurisrcbin->config->enable_error_propagation) {
            msgapi_send_message(nvmultiurisrcbin, sensorId, "N/A", dec_info->dec_log.c_str(),FALSE);
          }
          break;
        case SKIP_FRAMES:
          g_print ("[WARN] skip-frame not set on decoder .. !! \n");
          dec_info->status = SKIP_FRAMES_UPDATE_FAIL;
          dec_info->dec_log = "SKIP_FRAMES_UPDATE_FAIL, skip-frame not set on decoder";
          dec_info->err_info.code = StatusInternalServerError;
          if (nvmultiurisrcbin->config->enable_error_propagation) {
            msgapi_send_message(nvmultiurisrcbin, sensorId, "N/A", dec_info->dec_log.c_str(),FALSE);
          }
          break;
        case LOW_LATENCY_MODE:
          g_print ("[WARN] low-latency-mode not set on decoder .. !! \n");
          dec_info->status = LOW_LATENCY_MODE_UPDATE_FAIL;
          dec_info->dec_log = "LOW_LATENCY_MODE_UPDATE_FAIL, low-latency-mode not set on decoder";
          dec_info->err_info.code = StatusInternalServerError;
          if (nvmultiurisrcbin->config->enable_error_propagation) {
            msgapi_send_message(nvmultiurisrcbin, sensorId, "N/A", dec_info->dec_log.c_str(),FALSE);
          }
          break;
        default:
          break;
      }
    } else {
      switch (dec_info->dec_flag) {
        case DROP_FRAME_INTERVAL:
          dec_info->status =
              dec_info->status !=
              DROP_FRAME_INTERVAL_UPDATE_FAIL ? DROP_FRAME_INTERVAL_UPDATE_SUCCESS
              : DROP_FRAME_INTERVAL_UPDATE_FAIL;
          if ( dec_info->status == DROP_FRAME_INTERVAL_UPDATE_SUCCESS ){
            dec_info->err_info.code = StatusOk;
            dec_info->dec_log = "DROP_FRAME_INTERVAL_UPDATE_SUCCESS";
          } else{
            dec_info->err_info.code = StatusInternalServerError;
            dec_info->dec_log = "DROP_FRAME_INTERVAL_UPDATE_FAIL, Error while setting drop-frame-interval property";
            if (nvmultiurisrcbin->config->enable_error_propagation) {
              msgapi_send_message(nvmultiurisrcbin, sensorId, "N/A", dec_info->dec_log.c_str(),FALSE);
            }
          }
          break;
        case SKIP_FRAMES:
          dec_info->status =
              dec_info->status !=
              SKIP_FRAMES_UPDATE_FAIL ? SKIP_FRAMES_UPDATE_SUCCESS :
              SKIP_FRAMES_UPDATE_FAIL;
          if ( dec_info->status == SKIP_FRAMES_UPDATE_SUCCESS ){
            dec_info->err_info.code = StatusOk;
            dec_info->dec_log = "SKIP_FRAMES_UPDATE_SUCCESS";
          } else{
            dec_info->err_info.code = StatusInternalServerError;
            dec_info->dec_log = "SKIP_FRAMES_UPDATE_FAIL, Error while setting skip-frame property";
            if (nvmultiurisrcbin->config->enable_error_propagation) {
              msgapi_send_message(nvmultiurisrcbin, sensorId, "N/A", dec_info->dec_log.c_str(),FALSE);
            }
          }
          break;
        case LOW_LATENCY_MODE:
          dec_info->status =
              dec_info->status !=
              LOW_LATENCY_MODE_UPDATE_FAIL ? LOW_LATENCY_MODE_UPDATE_SUCCESS :
              LOW_LATENCY_MODE_UPDATE_FAIL;
          if ( dec_info->status == LOW_LATENCY_MODE_UPDATE_SUCCESS ){
            dec_info->err_info.code = StatusOk;
            dec_info->dec_log = "LOW_LATENCY_MODE_UPDATE_SUCCESS";
          } else{
            dec_info->err_info.code = StatusInternalServerError;
            dec_info->dec_log = "LOW_LATENCY_MODE_UPDATE_FAIL, Error while setting skip-frame property";
            if (nvmultiurisrcbin->config->enable_error_propagation) {
              msgapi_send_message(nvmultiurisrcbin, sensorId, "N/A", dec_info->dec_log.c_str(),FALSE);
            }
          }
          break;
        default:
          break;
      }
    }

    gst_nvmultiurisrcbincreator_sync_children_states (nvmultiurisrcbin->
        nvmultiurisrcbinCreator);
    gst_element_call_async (GST_ELEMENT (nvmultiurisrcbin),
        (GstElementCallAsyncFunc) gst_bin_sync_children_states, NULL, NULL);
  } else {
    g_print ("Unsupported REST API version\n");
    dec_info->err_info.code = StatusBadRequest;
  }

}

static void
s_infer_api_impl (NvDsServerInferInfo * infer_info, void *ctx)
{

  GstDsNvMultiUriBin *nvmultiurisrcbin = (GstDsNvMultiUriBin *) ctx;
  (void) nvmultiurisrcbin;
  guint sourceId = std::stoi (infer_info->stream_id);

  if (infer_info->uri.find ("/api/v1/") != std::string::npos) {
    if (!find_source (nvmultiurisrcbin->nvmultiurisrcbinCreator, sourceId)) {
      infer_info->status = INFER_INTERVAL_UPDATE_FAIL;
      infer_info->err_info.code = StatusInternalServerError;
      infer_info->infer_log = "INFER_INTERVAL_UPDATE_FAIL, Unable to find stream id for infer property updation";
      if (nvmultiurisrcbin->config->enable_error_propagation) {
        msgapi_send_message(nvmultiurisrcbin, "N/A", "N/A", infer_info->infer_log.c_str(),TRUE);
      }
    } else {
      gchar* sensorId = gst_nvmultiurisrcbincreator_get_sensor_id_from_source_id(nvmultiurisrcbin->nvmultiurisrcbinCreator,sourceId);
      GstEvent *nvevent =
          gst_nvevent_infer_interval_update ((char *) infer_info->stream_id.
          c_str (), infer_info->interval);

      if (!nvevent) {
        infer_info->status = INFER_INTERVAL_UPDATE_FAIL;
        infer_info->infer_log = "INFER_INTERVAL_UPDATE_FAIL, nv-infer-interval-update event creation failed";
        infer_info->err_info.code = StatusInternalServerError;
        if (nvmultiurisrcbin->config->enable_error_propagation) {
          msgapi_send_message(nvmultiurisrcbin, sensorId, "N/A", infer_info->infer_log.c_str(),FALSE);
        }
      }

      if (!gst_pad_push_event ((GstPad *) (nvmultiurisrcbin->bin_src_pad),
              nvevent)) {
        g_print
            ("[WARN] nv-infer-interval-update event not pushed downstream.. !! \n");
        infer_info->status = INFER_INTERVAL_UPDATE_FAIL;
        infer_info->infer_log = "INFER_INTERVAL_UPDATE_FAIL, nv-infer-interval-update event not pushed";
        infer_info->err_info.code = StatusInternalServerError;
        if (nvmultiurisrcbin->config->enable_error_propagation) {
          msgapi_send_message(nvmultiurisrcbin, sensorId, "N/A", infer_info->infer_log.c_str(),FALSE);
        }
      } else {
        infer_info->status = INFER_INTERVAL_UPDATE_SUCCESS;
        infer_info->infer_log = "INFER_INTERVAL_UPDATE_SUCCESS";
        infer_info->err_info.code = StatusOk;
      }
    }
    gst_nvmultiurisrcbincreator_sync_children_states (nvmultiurisrcbin->
      nvmultiurisrcbinCreator);
    gst_element_call_async (GST_ELEMENT (nvmultiurisrcbin),
      (GstElementCallAsyncFunc) gst_bin_sync_children_states, NULL, NULL);
  } else {
    g_print ("Unsupported REST API version\n");
    infer_info->err_info.code = StatusBadRequest;
  }
}

static void
s_inferserver_api_impl (NvDsServerInferServerInfo * inferserver_info, void *ctx)
{

  GstDsNvMultiUriBin *nvmultiurisrcbin = (GstDsNvMultiUriBin *) ctx;
  (void) nvmultiurisrcbin;
  guint sourceId = std::stoi (inferserver_info->stream_id);
  /* Default to 500 until a branch sets the real status; prevents a silent
   * 200 OK on an unhandled/unknown inferserver flag. */
  inferserver_info->err_info.code = StatusInternalServerError;

  if (inferserver_info->uri.find ("/api/v1/") != std::string::npos) {
    if (!find_source (nvmultiurisrcbin->nvmultiurisrcbinCreator, sourceId)) {
      if (inferserver_info->inferserver_flag == INFERSERVER_INTERVAL) {
        inferserver_info->status = INFERSERVER_INTERVAL_UPDATE_FAIL;
        inferserver_info->err_info.code = StatusInternalServerError;
        inferserver_info->inferserver_log = "INFERSERVER_INTERVAL_UPDATE_FAIL, Unable to find stream id for infer (inferserver) property updation";
        if (nvmultiurisrcbin->config->enable_error_propagation) {
          msgapi_send_message(nvmultiurisrcbin, "N/A", "N/A", inferserver_info->inferserver_log.c_str(),TRUE);
        }
      }
    } else {
      gchar* sensorId = gst_nvmultiurisrcbincreator_get_sensor_id_from_source_id(nvmultiurisrcbin->nvmultiurisrcbinCreator,sourceId);
      GstEvent *nvevent =
          gst_nvevent_infer_interval_update ((char *) inferserver_info->stream_id.
          c_str (), inferserver_info->interval);
      if (!nvevent) {
        inferserver_info->status = INFERSERVER_INTERVAL_UPDATE_FAIL;
        inferserver_info->inferserver_log =
            "INFERSERVER_INTERVAL_UPDATE_FAIL, nv-infer-interval-update event (inferserver) creation failed";
        inferserver_info->err_info.code = StatusInternalServerError;
        if (nvmultiurisrcbin->config->enable_error_propagation) {
          msgapi_send_message(nvmultiurisrcbin, sensorId, "N/A", inferserver_info->inferserver_log.c_str(),FALSE);
        }
      }

      if (!gst_pad_push_event ((GstPad *) (nvmultiurisrcbin->bin_src_pad),
              nvevent)) {
        g_print
            ("[WARN] nv-infer-interval-update (inferserver) event not pushed downstream.. !! \n");
        inferserver_info->status = INFERSERVER_INTERVAL_UPDATE_FAIL;
        inferserver_info->inferserver_log =
            "INFERSERVER_INTERVAL_UPDATE_FAIL, nv-infer-interval-update event not pushed";
        inferserver_info->err_info.code = StatusInternalServerError;
        if (nvmultiurisrcbin->config->enable_error_propagation) {
          msgapi_send_message(nvmultiurisrcbin, sensorId, "N/A", inferserver_info->inferserver_log.c_str(),FALSE);
        }
      } else {
        inferserver_info->status = INFERSERVER_INTERVAL_UPDATE_SUCCESS;
        inferserver_info->inferserver_log = "INFERSERVER_INTERVAL_UPDATE_SUCCESS";
        inferserver_info->err_info.code = StatusOk;
      }
    }
    gst_nvmultiurisrcbincreator_sync_children_states (nvmultiurisrcbin->
      nvmultiurisrcbinCreator);
    gst_element_call_async (GST_ELEMENT (nvmultiurisrcbin),
      (GstElementCallAsyncFunc) gst_bin_sync_children_states, NULL, NULL);
  } else {
    g_print("Unsupported REST API version\n");
    inferserver_info->err_info.code = StatusBadRequest;
  }
}

static void
s_nvtracker_api_impl( NvDsServerNvTrackerInfo* nvtracker_info,  void *ctx)
{
  GstDsNvMultiUriBin *nvmultiurisrcbin = (GstDsNvMultiUriBin *) ctx;
  (void) nvmultiurisrcbin;
  guint sourceId = std::stoi (nvtracker_info->stream_id);
  /* Default to 500 until a branch sets the real status; prevents a silent
   * 200 OK on an unhandled/unknown nvtracker flag. */
  nvtracker_info->err_info.code = StatusInternalServerError;

  if (nvtracker_info->uri.find ("/api/v1/") != std::string::npos)
  {
    if (!find_source (nvmultiurisrcbin->nvmultiurisrcbinCreator, sourceId))
    {
      if (nvtracker_info->nvTracker_flag == NVTRACKER_CONFIG)
      {
        nvtracker_info->status = NVTRACKER_CONFIG_UPDATE_FAIL;
        nvtracker_info->err_info.code = StatusInternalServerError;
        nvtracker_info->nvTracker_log =
          "NVTRACKER_CONFIG_UPDATE_FAIL, Unable to find stream id for nvTracker property updation";
        if (nvmultiurisrcbin->config->enable_error_propagation) {
          msgapi_send_message(nvmultiurisrcbin, "N/A", "N/A", nvtracker_info->nvTracker_log.c_str(),TRUE);
        }
      }
    }
    else
    {
      gchar* sensorId = gst_nvmultiurisrcbincreator_get_sensor_id_from_source_id(nvmultiurisrcbin->nvmultiurisrcbinCreator,sourceId);
      GstEvent *nvevent = gst_nvevent_nvtracker_config_update
                                                        (
                                                        (char *) nvtracker_info->stream_id.c_str(),
                                                        (char *) nvtracker_info->config_path.c_str()
                                                        );

      if (!nvevent)
      {
        nvtracker_info->status = NVTRACKER_CONFIG_UPDATE_FAIL;
        nvtracker_info->nvTracker_log =
            "NVTRACKER_CONFIG_UPDATE_FAIL, nv-tracker-config-update event creation failed";
        nvtracker_info->err_info.code = StatusInternalServerError;
        if (nvmultiurisrcbin->config->enable_error_propagation) {
          msgapi_send_message(nvmultiurisrcbin, sensorId, "N/A", nvtracker_info->nvTracker_log.c_str(),FALSE);
        }
      }

      /* send nv-tracker-update event */
      if (!gst_pad_push_event ((GstPad *) (nvmultiurisrcbin->bin_src_pad), nvevent))
      {
        g_print("[WARN] nv-tracker-config-update event not pushed downstream.. !! \n");
        nvtracker_info->status = NVTRACKER_CONFIG_UPDATE_FAIL;
        nvtracker_info->nvTracker_log ="NVTRACKER_CONFIG_UPDATE_FAIL, nv-tracker-config-update event not pushed";
        nvtracker_info->err_info.code = StatusInternalServerError;
        if (nvmultiurisrcbin->config->enable_error_propagation) {
          msgapi_send_message(nvmultiurisrcbin, sensorId, "N/A", nvtracker_info->nvTracker_log.c_str(),FALSE);
        }
      }
      else
      {
        nvtracker_info->status = NVTRACKER_CONFIG_UPDATE_SUCCESS;
        nvtracker_info->err_info.code = StatusOk;
        nvtracker_info->nvTracker_log = "NVTRACKER_CONFIG_UPDATE_SUCCESS";
      }
      gst_nvmultiurisrcbincreator_sync_children_states (nvmultiurisrcbin->nvmultiurisrcbinCreator);
      gst_element_call_async (GST_ELEMENT (nvmultiurisrcbin),
        (GstElementCallAsyncFunc) gst_bin_sync_children_states, NULL, NULL);
    }
  }
  else
  {
    g_print("Unsupported REST API version\n");
    nvtracker_info->err_info.code = StatusBadRequest;
  }
}

static void
s_conv_api_impl (NvDsServerConvInfo * conv_info, void *ctx)
{

  GstDsNvMultiUriBin *nvmultiurisrcbin = (GstDsNvMultiUriBin *) ctx;
  (void) nvmultiurisrcbin;
  guint sourceId = std::stoi (conv_info->stream_id);
  gchar* sensorId = gst_nvmultiurisrcbincreator_get_sensor_id_from_source_id(nvmultiurisrcbin->nvmultiurisrcbinCreator,sourceId);
  /* Default to 500 until a branch sets the real status; prevents a silent
   * 200 OK on an unhandled/unknown converter flag. */
  conv_info->err_info.code = StatusInternalServerError;

  if (conv_info->uri.find ("/api/v1/") != std::string::npos) {
    if (!set_nvuribin_conv_prop (nvmultiurisrcbin->nvmultiurisrcbinCreator,
            sourceId, conv_info)) {
      switch (conv_info->conv_flag) {
        case SRC_CROP:
          g_print ("[WARN] source-crop update failed .. !! \n");
          conv_info->conv_log = "SRC_CROP_UPDATE_FAIL, source-crop update failed";
          conv_info->status = SRC_CROP_UPDATE_FAIL;
          conv_info->err_info.code = StatusInternalServerError;
          if (nvmultiurisrcbin->config->enable_error_propagation) {
            msgapi_send_message(nvmultiurisrcbin, sensorId, "N/A", conv_info->conv_log.c_str(),FALSE);
          }
          break;
        case DEST_CROP:
          g_print ("[WARN] source-crop update failed .. !! \n");
          conv_info->conv_log = "DEST_CROP_UPDATE_FAIL, dest-crop update failed";
          conv_info->status = DEST_CROP_UPDATE_FAIL;
          conv_info->err_info.code = StatusInternalServerError;
          if (nvmultiurisrcbin->config->enable_error_propagation) {
            msgapi_send_message(nvmultiurisrcbin, sensorId, "N/A", conv_info->conv_log.c_str(),FALSE);
          }
          break;
        case FLIP_METHOD:
          g_print ("[WARN] flip-method update failed .. !! \n");
          conv_info->conv_log = "FLIP_METHOD_UPDATE_FAIL, flip-method update failed";
          conv_info->status = FLIP_METHOD_UPDATE_FAIL;
          conv_info->err_info.code = StatusInternalServerError;
          if (nvmultiurisrcbin->config->enable_error_propagation) {
            msgapi_send_message(nvmultiurisrcbin, sensorId, "N/A", conv_info->conv_log.c_str(),FALSE);
          }
          break;
        case INTERPOLATION_METHOD:
          g_print ("[WARN] interpolation-method update failed .. !! \n");
          conv_info->conv_log = "INTERPOLATION_METHOD_UPDATE_FAIL, interpolation-method update failed";
          conv_info->status = INTERPOLATION_METHOD_UPDATE_FAIL;
          conv_info->err_info.code = StatusInternalServerError;
          if (nvmultiurisrcbin->config->enable_error_propagation) {
            msgapi_send_message(nvmultiurisrcbin, sensorId, "N/A", conv_info->conv_log.c_str(),FALSE);
          }
          break;
        default:
          break;
      }
    } else {
      switch (conv_info->conv_flag) {
        case SRC_CROP:
          conv_info->status =
              conv_info->status !=
              SRC_CROP_UPDATE_FAIL ? SRC_CROP_UPDATE_SUCCESS :
              SRC_CROP_UPDATE_FAIL;
              if ( conv_info->status == SRC_CROP_UPDATE_SUCCESS ){
                conv_info->err_info.code = StatusOk;
                conv_info->conv_log = "SRC_CROP_UPDATE_SUCCESS";
              } else{
                conv_info->err_info.code = StatusInternalServerError;
                conv_info->conv_log = "SRC_CROP_UPDATE_FAIL, Error while setting src-crop property";
                if (nvmultiurisrcbin->config->enable_error_propagation) {
                  msgapi_send_message(nvmultiurisrcbin, sensorId, "N/A", conv_info->conv_log.c_str(),FALSE);
                }
              }
          break;
        case DEST_CROP:
          conv_info->status =
              conv_info->status !=
              DEST_CROP_UPDATE_FAIL ? DEST_CROP_UPDATE_SUCCESS :
              DEST_CROP_UPDATE_FAIL;
          if ( conv_info->status == DEST_CROP_UPDATE_SUCCESS ){
            conv_info->conv_log = "DEST_CROP_UPDATE_SUCCESS";
            conv_info->err_info.code = StatusOk;
          } else{
            conv_info->err_info.code = StatusInternalServerError;
            conv_info->conv_log = "DEST_CROP_UPDATE_FAIL, Error while setting dest-crop property";
            if (nvmultiurisrcbin->config->enable_error_propagation) {
              msgapi_send_message(nvmultiurisrcbin, sensorId, "N/A", conv_info->conv_log.c_str(),FALSE);
            }
          }
          break;
        case FLIP_METHOD:
          conv_info->status =
              conv_info->status !=
              FLIP_METHOD_UPDATE_FAIL ? FLIP_METHOD_UPDATE_SUCCESS :
              FLIP_METHOD_UPDATE_FAIL;
          if ( conv_info->status == FLIP_METHOD_UPDATE_SUCCESS ){
            conv_info->err_info.code = StatusOk;
            conv_info->conv_log = "FLIP_METHOD_UPDATE_SUCCESS";
          } else{
            conv_info->err_info.code = StatusInternalServerError;
            conv_info->conv_log = "FLIP_METHOD_UPDATE_FAIL, Error while setting flip-method property";
            if (nvmultiurisrcbin->config->enable_error_propagation) {
              msgapi_send_message(nvmultiurisrcbin, sensorId, "N/A", conv_info->conv_log.c_str(),FALSE);
            }
          }
          break;
        case INTERPOLATION_METHOD:
          conv_info->status =
              conv_info->status !=
              INTERPOLATION_METHOD_UPDATE_FAIL ?
              INTERPOLATION_METHOD_UPDATE_SUCCESS :
              INTERPOLATION_METHOD_UPDATE_FAIL;
          if ( conv_info->status == INTERPOLATION_METHOD_UPDATE_SUCCESS ){
            conv_info->err_info.code = StatusOk;
            conv_info->conv_log = "INTERPOLATION_METHOD_UPDATE_SUCCESS";
          } else{
            conv_info->err_info.code = StatusInternalServerError;
            conv_info->conv_log = "INTERPOLATION_METHOD_UPDATE_FAIL, Error while setting interpolation-method property";
            if (nvmultiurisrcbin->config->enable_error_propagation) {
              msgapi_send_message(nvmultiurisrcbin, sensorId, "N/A", conv_info->conv_log.c_str(),FALSE);
            }
          }
          break;
        default:
          break;
      }
    }
    gst_nvmultiurisrcbincreator_sync_children_states (nvmultiurisrcbin->
      nvmultiurisrcbinCreator);
    gst_element_call_async (GST_ELEMENT (nvmultiurisrcbin),
      (GstElementCallAsyncFunc) gst_bin_sync_children_states, NULL, NULL);
  } else {
    g_print("Unsupported REST API version\n");
    conv_info->err_info.code = StatusBadRequest;
  }

}

static void
s_mux_api_impl (NvDsServerMuxInfo * mux_info, void *ctx)
{

  GstDsNvMultiUriBin *nvmultiurisrcbin = (GstDsNvMultiUriBin *) ctx;
  (void) nvmultiurisrcbin;
  /* Default to 500 until a branch sets the real status; prevents a silent
   * 200 OK on an unhandled/unknown mux flag. */
  mux_info->err_info.code = StatusInternalServerError;

  if (mux_info->uri.find ("/api/v1/") != std::string::npos) {
    if (!set_nvuribin_mux_prop (nvmultiurisrcbin->nvmultiurisrcbinCreator,
        mux_info)) {
      switch (mux_info->mux_flag) {
        case BATCHED_PUSH_TIMEOUT:
          g_print ("[WARN] batched-push-timeout update failed .. !! \n");
          mux_info->mux_log = "BATCHED_PUSH_TIMEOUT_UPDATE_FAIL, batched-push-timeout value not updated";
          mux_info->status = BATCHED_PUSH_TIMEOUT_UPDATE_FAIL;
          mux_info->err_info.code = StatusInternalServerError;
          if (nvmultiurisrcbin->config->enable_error_propagation) {
            msgapi_send_message(nvmultiurisrcbin, "N/A", "N/A", mux_info->mux_log.c_str(),TRUE);
          }
          break;
        case MAX_LATENCY:
          g_print ("[WARN] max-latency update failed .. !! \n");
          mux_info->mux_log = "MAX_LATENCY_UPDATE_FAIL, MAX_LATENCY_UPDATE_FAIL, max-latency value not updated";
          mux_info->status = MAX_LATENCY_UPDATE_FAIL;
          mux_info->err_info.code = StatusInternalServerError;
          if (nvmultiurisrcbin->config->enable_error_propagation) {
            msgapi_send_message(nvmultiurisrcbin, "N/A", "N/A", mux_info->mux_log.c_str(),TRUE);
          }
          break;
        default:
          break;
      }
    } else {
      switch (mux_info->mux_flag) {
        case BATCHED_PUSH_TIMEOUT:
          mux_info->status =
              mux_info->status !=
              BATCHED_PUSH_TIMEOUT_UPDATE_FAIL ?
              BATCHED_PUSH_TIMEOUT_UPDATE_SUCCESS :
              BATCHED_PUSH_TIMEOUT_UPDATE_FAIL;
          if ( mux_info->status == BATCHED_PUSH_TIMEOUT_UPDATE_SUCCESS ){
            mux_info->err_info.code = StatusOk;
            mux_info->mux_log = "BATCHED_PUSH_TIMEOUT_UPDATE_SUCCESS";
          } else{
            mux_info->err_info.code = StatusInternalServerError;
            mux_info->mux_log = "BATCHED_PUSH_TIMEOUT_UPDATE_SUCCESS, Error while setting batched-push-timeout property";
            if (nvmultiurisrcbin->config->enable_error_propagation) {
              msgapi_send_message(nvmultiurisrcbin, "N/A", "N/A", mux_info->mux_log.c_str(),TRUE);
            }
          }
          break;
        case MAX_LATENCY:
          mux_info->status =
              mux_info->status !=
              MAX_LATENCY_UPDATE_FAIL ? MAX_LATENCY_UPDATE_SUCCESS :
              MAX_LATENCY_UPDATE_FAIL;
          if ( mux_info->status == MAX_LATENCY_UPDATE_SUCCESS ){
            mux_info->err_info.code = StatusOk;
            mux_info->mux_log = "MAX_LATENCY_UPDATE_SUCCESS";
          } else{
            mux_info->err_info.code = StatusInternalServerError;
            mux_info->mux_log = "MAX_LATENCY_UPDATE_FAIL, Error while setting max-latency property";
            if (nvmultiurisrcbin->config->enable_error_propagation) {
              msgapi_send_message(nvmultiurisrcbin, "N/A", "N/A", mux_info->mux_log.c_str(),TRUE);
            }
          }
          break;
        default:
          break;
      }
    }
    gst_nvmultiurisrcbincreator_sync_children_states (nvmultiurisrcbin->
      nvmultiurisrcbinCreator);
    gst_element_call_async (GST_ELEMENT (nvmultiurisrcbin),
      (GstElementCallAsyncFunc) gst_bin_sync_children_states, NULL, NULL);
  } else {
    g_print ("Unsupported REST API version\n");
    mux_info->err_info.code = StatusBadRequest;
  }
}

static void
s_enc_api_impl (NvDsServerEncInfo * enc_info, void *ctx)
{

  GstDsNvMultiUriBin *nvmultiurisrcbin = (GstDsNvMultiUriBin *) ctx;
  (void) nvmultiurisrcbin;
  guint sourceId = std::stoi (enc_info->stream_id);
  GstEvent *nvevent = NULL;
  /* Default to 500 until a branch sets the real status; prevents a silent
   * 200 OK on an unhandled/unknown encoder flag. */
  enc_info->err_info.code = StatusInternalServerError;

  if (enc_info->uri.find ("/api/v1/") != std::string::npos) {
    if (!find_source (nvmultiurisrcbin->nvmultiurisrcbinCreator, sourceId)) {
      switch (enc_info->enc_flag) {
        case BITRATE:
          enc_info->enc_log = "BITRATE_UPDATE_FAIL, Not able to find source id";
          enc_info->status = BITRATE_UPDATE_FAIL;
          enc_info->err_info.code = StatusInternalServerError;
          if (nvmultiurisrcbin->config->enable_error_propagation) {
            msgapi_send_message(nvmultiurisrcbin, "N/A", "N/A", enc_info->enc_log.c_str(),TRUE);
          }
          break;
        case FORCE_IDR:
          enc_info->enc_log = "FORCE_IDR_UPDATE_FAIL, Not able to find source id";
          enc_info->status = FORCE_IDR_UPDATE_FAIL;
          enc_info->err_info.code = StatusInternalServerError;
          if (nvmultiurisrcbin->config->enable_error_propagation) {
            msgapi_send_message(nvmultiurisrcbin, "N/A", "N/A", enc_info->enc_log.c_str(),TRUE);
          }
          break;
        case FORCE_INTRA:
          enc_info->enc_log = "FORCE_INTRA_UPDATE_FAIL, Not able to find source id";
          enc_info->status = FORCE_INTRA_UPDATE_FAIL;
          enc_info->err_info.code = StatusInternalServerError;
          if (nvmultiurisrcbin->config->enable_error_propagation) {
            msgapi_send_message(nvmultiurisrcbin, "N/A", "N/A", enc_info->enc_log.c_str(),TRUE);
          }
          break;
        case IFRAME_INTERVAL:
          enc_info->enc_log = "IFRAME_INTERVAL_UPDATE_FAIL, Not able to find source id";
          enc_info->status = IFRAME_INTERVAL_UPDATE_FAIL;
          enc_info->err_info.code = StatusInternalServerError;
          if (nvmultiurisrcbin->config->enable_error_propagation) {
            msgapi_send_message(nvmultiurisrcbin, "N/A", "N/A", enc_info->enc_log.c_str(),TRUE);
          }
          break;
        default:
          break;
      }
    } else {
      gchar* sensorId = gst_nvmultiurisrcbincreator_get_sensor_id_from_source_id(nvmultiurisrcbin->nvmultiurisrcbinCreator,sourceId);
      switch (enc_info->enc_flag) {
        case BITRATE:
          nvevent =
              gst_nvevent_enc_bitrate_update ((char *) enc_info->stream_id.
              c_str (), enc_info->bitrate);
          if (!gst_pad_push_event ((GstPad *) (nvmultiurisrcbin->bin_src_pad),
                  nvevent)) {
            g_print
                ("[WARN] nv-enc-bitrate-update event not pushed downstream.bitrate update failed on encoder.. !! \n");
            enc_info->enc_log =
                !nvevent ? "BITRATE_UPDATE_FAIL, nv-enc-bitrate-update event creation failed" :
                "BITRATE_UPDATE_FAIL, nv-enc-bitrate-update event not pushed";
            enc_info->status = BITRATE_UPDATE_FAIL;
            enc_info->err_info.code = StatusInternalServerError;
            if (nvmultiurisrcbin->config->enable_error_propagation) {
              msgapi_send_message(nvmultiurisrcbin, sensorId, "N/A", enc_info->enc_log.c_str(),FALSE);
            }
          } else {
            enc_info->status = BITRATE_UPDATE_SUCCESS;
            enc_info->err_info.code = StatusOk;
            enc_info->enc_log = "BITRATE_UPDATE_SUCCESS";
          }
          break;
        case FORCE_IDR:
          nvevent =
              gst_nvevent_enc_force_idr ((char *) enc_info->stream_id.c_str (),
              enc_info->force_idr);
          if (!gst_pad_push_event ((GstPad *) (nvmultiurisrcbin->bin_src_pad),
                  nvevent)) {
            g_print
                ("[WARN] nv-enc-force-idr event not pushed downstream.force IDR frame failed on encoder .. !! \n");
            enc_info->enc_log =
                !nvevent ? "FORCE_IDR_UPDATE_FAIL, nv-enc-force-idr event creation failed" :
                "FORCE_IDR_UPDATE_FAIL, nv-enc-force-idr event not pushed";
            enc_info->status = FORCE_IDR_UPDATE_FAIL;
            enc_info->err_info.code = StatusInternalServerError;
            if (nvmultiurisrcbin->config->enable_error_propagation) {
              msgapi_send_message(nvmultiurisrcbin, sensorId, "N/A", enc_info->enc_log.c_str(),FALSE);
            }
          } else {
            enc_info->status = FORCE_IDR_UPDATE_SUCCESS;
            enc_info->err_info.code = StatusOk;
            enc_info->enc_log = "FORCE_IDR_UPDATE_SUCCESS";
          }
          break;
        case FORCE_INTRA:
          nvevent =
              gst_nvevent_enc_force_intra ((char *) enc_info->stream_id.c_str (),
              enc_info->force_intra);
          if (!gst_pad_push_event ((GstPad *) (nvmultiurisrcbin->bin_src_pad),
                  nvevent)) {
            g_print
                ("[WARN] nv-enc-force-intra event not pushed downstream.force intra frame failed on encoder .. !! \n");
            enc_info->enc_log =
                !nvevent ? "FORCE_INTRA_UPDATE_FAIL, nv-enc-force-intra event creation failed" :
                "FORCE_INTRA_UPDATE_FAIL, nv-enc-force-intra event not pushed";
            enc_info->status = FORCE_INTRA_UPDATE_FAIL;
            enc_info->err_info.code = StatusInternalServerError;
            if (nvmultiurisrcbin->config->enable_error_propagation) {
              msgapi_send_message(nvmultiurisrcbin, sensorId, "N/A", enc_info->enc_log.c_str(),FALSE);
            }
          } else {
            enc_info->status = FORCE_INTRA_UPDATE_SUCCESS;
            enc_info->err_info.code = StatusOk;
            enc_info->enc_log = "FORCE_INTRA_UPDATE_SUCCESS";
          }
          break;
        case IFRAME_INTERVAL:
          nvevent =
              gst_nvevent_enc_iframeinterval_update ((char *) enc_info->stream_id.
              c_str (), enc_info->iframeinterval);
          if (!gst_pad_push_event ((GstPad *) (nvmultiurisrcbin->bin_src_pad),
                  nvevent)) {
            g_print
                ("[WARN] nv-enc-iframeinterval-update event not pushed downstream.iframe interval update failed on encoder .. !! \n");
            enc_info->enc_log =
                !nvevent ? "IFRAME_INTERVAL_UPDATE_FAIL, nv-enc-iframeinterval-update event creation failed" :
                "IFRAME_INTERVAL_UPDATE_FAIL, nv-enc-iframeinterval-update event not pushed";
            enc_info->status = IFRAME_INTERVAL_UPDATE_FAIL;
            enc_info->err_info.code = StatusInternalServerError;
            if (nvmultiurisrcbin->config->enable_error_propagation) {
              msgapi_send_message(nvmultiurisrcbin, sensorId, "N/A", enc_info->enc_log.c_str(),FALSE);
            }
          } else {
            enc_info->status = IFRAME_INTERVAL_UPDATE_SUCCESS;
            enc_info->err_info.code = StatusOk;
            enc_info->enc_log = "IFRAME_INTERVAL_UPDATE_SUCCESS";
          }
          break;
        default:
          break;
      }
    }
    gst_nvmultiurisrcbincreator_sync_children_states (nvmultiurisrcbin->
      nvmultiurisrcbinCreator);
    gst_element_call_async (GST_ELEMENT (nvmultiurisrcbin),
      (GstElementCallAsyncFunc) gst_bin_sync_children_states, NULL, NULL);
  } else {
    g_print("Unsupported REST API version\n");
    enc_info->err_info.code = StatusBadRequest;
  }
}

static void
s_osd_api_impl (NvDsServerOsdInfo * osd_info, void *ctx)
{

  GstDsNvMultiUriBin *nvmultiurisrcbin = (GstDsNvMultiUriBin *) ctx;
  (void) nvmultiurisrcbin;

  guint sourceId = std::stoi (osd_info->stream_id);
  /* Default to 500 until a branch sets the real status; prevents a silent
   * 200 OK on an unhandled/unknown osd flag. */
  osd_info->err_info.code = StatusInternalServerError;

  if (osd_info->uri.find ("/api/v1/") != std::string::npos) {
    if (!find_source (nvmultiurisrcbin->nvmultiurisrcbinCreator, sourceId)) {
      if (osd_info->osd_flag == PROCESS_MODE) {
        osd_info->status = PROCESS_MODE_UPDATE_FAIL;
        osd_info->err_info.code = StatusInternalServerError;
        osd_info->osd_log = "PROCESS_MODE_UPDATE_FAIL, Unable to find stream id for osd property updation";
        if (nvmultiurisrcbin->config->enable_error_propagation) {
          msgapi_send_message(nvmultiurisrcbin, "N/A", "N/A", osd_info->osd_log.c_str(),TRUE);
        }
      }
    } else {
      gchar* sensorId = gst_nvmultiurisrcbincreator_get_sensor_id_from_source_id(nvmultiurisrcbin->nvmultiurisrcbinCreator,sourceId);
      GstEvent *nvevent =
          gst_nvevent_osd_process_mode_update ((char *) osd_info->stream_id.
          c_str (), osd_info->process_mode);
      if (!nvevent) {
        osd_info->status = PROCESS_MODE_UPDATE_FAIL;
        osd_info->osd_log = "PROCESS_MODE_UPDATE_FAIL, nv-osd-process-mode-update event creation failed";
        osd_info->err_info.code = StatusInternalServerError;
        if (nvmultiurisrcbin->config->enable_error_propagation) {
          msgapi_send_message(nvmultiurisrcbin, sensorId, "N/A", osd_info->osd_log.c_str(),FALSE);
        }
      }

      if (!gst_pad_push_event ((GstPad *) (nvmultiurisrcbin->bin_src_pad),
              nvevent)) {
        g_print
            ("[WARN] nv-osd-process-mode-update event not pushed downstream.. !! \n");
        osd_info->status = PROCESS_MODE_UPDATE_FAIL;
        osd_info->osd_log = "PROCESS_MODE_UPDATE_FAIL, nv-osd-process-mode-update event not pushed";
        osd_info->err_info.code = StatusInternalServerError;
        if (nvmultiurisrcbin->config->enable_error_propagation) {
          msgapi_send_message(nvmultiurisrcbin, sensorId, "N/A", osd_info->osd_log.c_str(),FALSE);
        }
      } else {
        osd_info->status = PROCESS_MODE_UPDATE_SUCCESS;
        osd_info->err_info.code = StatusOk;
        osd_info->osd_log = "PROCESS_MODE_UPDATE_SUCCESS";
      }

    }
  } else {
    g_print ("Unsupported REST API version\n");
    osd_info->err_info.code = StatusBadRequest;
  }
}

static void
s_analytics_api_impl(NvDsServerAnalyticsInfo* analytics_info, void* ctx) {
  GstDsNvMultiUriBin* nvmultiurisrcbin = (GstDsNvMultiUriBin*)ctx;

  if (analytics_info->uri.find("/api/v1/") != std::string::npos) {
    GstEvent* nvevent = gst_nvevent_analytics_reload_config_update((char*)analytics_info->config_file_path.c_str());

    if (!nvevent) {
      analytics_info->status = RELOAD_CONFIG_UPDATE_FAIL;
      analytics_info->analytics_log = "RELOAD_CONFIG_UPDATE_FAIL, analytics-config-update event creation failed";
      analytics_info->err_info.code = StatusInternalServerError;
      if (nvmultiurisrcbin->config->enable_error_propagation) {
        msgapi_send_message(nvmultiurisrcbin, "N/A", "N/A", analytics_info->analytics_log.c_str(),TRUE);
      }
      return;
    }

    if (!gst_pad_push_event((GstPad*)(nvmultiurisrcbin->bin_src_pad), nvevent)) {
      analytics_info->status = RELOAD_CONFIG_UPDATE_FAIL;
      analytics_info->analytics_log = "RELOAD_CONFIG_UPDATE_FAIL, analytics-config-update event not pushed";
      analytics_info->err_info.code = StatusInternalServerError;
      if (nvmultiurisrcbin->config->enable_error_propagation) {
        msgapi_send_message(nvmultiurisrcbin, "N/A", "N/A", analytics_info->analytics_log.c_str(),TRUE);
      }
    } else {
      analytics_info->status = RELOAD_CONFIG_UPDATE_SUCCESS;
      analytics_info->analytics_log = "RELOAD_CONFIG_UPDATE_SUCCESS";
      analytics_info->err_info.code = StatusOk;
    }

    gst_nvmultiurisrcbincreator_sync_children_states(nvmultiurisrcbin->nvmultiurisrcbinCreator);
    gst_element_call_async(GST_ELEMENT(nvmultiurisrcbin),
        (GstElementCallAsyncFunc)gst_bin_sync_children_states, NULL, NULL);
  } else {
    g_print("Unsupported REST API version\n");
    analytics_info->err_info.code = StatusBadRequest;
  }
}

static void
s_text_embedding_api_impl(NvDsServerTextEmbeddingInfo* text_embedding_info, void* ctx) {
  GstDsNvMultiUriBin* nvmultiurisrcbin = (GstDsNvMultiUriBin*)ctx;
  if (text_embedding_info->uri.find("/api/v1/") != std::string::npos) {

    // Create the text embedding query with request parameters
    GstQuery *query = gst_nvquery_text_embedding_new (text_embedding_info->text_input.c_str(),
                                                       text_embedding_info->model.c_str());


    // Send query downstream to nvinfer plugin via bin_src_pad
    gboolean query_ret = gst_pad_peer_query ((GstPad *)(nvmultiurisrcbin->bin_src_pad), query);

    if (query_ret) {
      // Parse the response from nvinfer plugin
      const gchar *id = NULL;
      const gchar *response_model = NULL;
      guint64 created = 0;
      const GValue *data = NULL;

      if (gst_nvquery_text_embedding_parse (query, &id, &created, &response_model, &data)) {
        // Successfully parsed response from downstream plugin
        const gchar *embeddings_json = NULL;
        text_embedding_info->id = id ? id : "";
        text_embedding_info->created = created;
        text_embedding_info->status = TEXT_EMBEDDING_GENERATE_SUCCESS;
        text_embedding_info->text_embedding_log = "Text embedding generated successfully";
        text_embedding_info->err_info.code = StatusOk;

        // Convert GValue data array to Json::Value if data is available
        if (data != NULL) {
          if (G_VALUE_HOLDS_STRING(data)) {
            embeddings_json = g_value_get_string(data);
            // Parse JSON string to Json::Value
            Json::Reader reader;
            if (!reader.parse(embeddings_json, text_embedding_info->data)) {
              g_print("Failed to parse embeddings JSON: %s\n", reader.getFormattedErrorMessages().c_str());
              text_embedding_info->data = Json::Value(Json::arrayValue);
            }
          } else {
            text_embedding_info->data = Json::Value(Json::arrayValue);
          }
        } else {
          text_embedding_info->data = Json::Value(Json::arrayValue);
        }
      } else {
        g_print("[WARN] Failed to parse text embedding query response\n");
        text_embedding_info->status = TEXT_EMBEDDING_GENERATE_FAIL;
        text_embedding_info->text_embedding_log = "Failed to parse response from nvinfer plugin";
        text_embedding_info->err_info.code = StatusInternalServerError;
        text_embedding_info->data = Json::Value(Json::arrayValue);
      }
    } else {
      g_print("[WARN] Text embedding query not handled by downstream elements\n");
      text_embedding_info->status = TEXT_EMBEDDING_GENERATE_FAIL;
      text_embedding_info->text_embedding_log = "Text embedding query not handled by downstream elements";
      text_embedding_info->err_info.code = StatusInternalServerError;
      text_embedding_info->data = Json::Value(Json::arrayValue);
    }

    // Clean up the query
    gst_query_unref (query);

  } else {
    g_print("Unsupported REST API version\n");
    text_embedding_info->status = TEXT_EMBEDDING_GENERATE_FAIL;
    text_embedding_info->text_embedding_log = "Unsupported REST API version";
    text_embedding_info->err_info.code = StatusBadRequest;
  }
}

static void
s_image_embedding_api_impl(NvDsServerImageEmbeddingInfo* image_embedding_info, void* ctx) {
  GstDsNvMultiUriBin* nvmultiurisrcbin = (GstDsNvMultiUriBin*)ctx;
  if (image_embedding_info->uri.find("/api/v1/") != std::string::npos) {

    GstQuery *query = gst_nvquery_image_embedding_new (
        image_embedding_info->image_path.c_str(),
        image_embedding_info->model.c_str(),
        image_embedding_info->has_bbox,
        image_embedding_info->bbox_left,
        image_embedding_info->bbox_top,
        image_embedding_info->bbox_width,
        image_embedding_info->bbox_height);

    gboolean query_ret = gst_pad_peer_query ((GstPad *)(nvmultiurisrcbin->bin_src_pad), query);

    if (query_ret) {
      const gchar *id = NULL;
      const gchar *response_model = NULL;
      guint64 created = 0;
      const GValue *data = NULL;

      if (gst_nvquery_image_embedding_parse (query, &id, &created, &response_model, &data)) {
        const gchar *embeddings_json = NULL;
        image_embedding_info->id = id ? id : "";
        image_embedding_info->created = created;
        image_embedding_info->status = IMAGE_EMBEDDING_GENERATE_SUCCESS;
        image_embedding_info->image_embedding_log = "Image embedding generated successfully";
        image_embedding_info->err_info.code = StatusOk;

        if (data != NULL && G_VALUE_HOLDS_STRING(data)) {
          embeddings_json = g_value_get_string(data);
          Json::Reader reader;
          if (!reader.parse(embeddings_json, image_embedding_info->data)) {
            g_print("Failed to parse image embeddings JSON: %s\n", reader.getFormattedErrorMessages().c_str());
            image_embedding_info->data = Json::Value(Json::arrayValue);
          }
        } else {
          image_embedding_info->data = Json::Value(Json::arrayValue);
        }
      } else {
        g_print("[WARN] Failed to parse image embedding query response\n");
        image_embedding_info->status = IMAGE_EMBEDDING_GENERATE_FAIL;
        image_embedding_info->image_embedding_log = "Failed to parse response from vision encoder plugin";
        image_embedding_info->err_info.code = StatusInternalServerError;
        image_embedding_info->data = Json::Value(Json::arrayValue);
      }
    } else {
      g_print("[WARN] Image embedding query not handled by downstream elements\n");
      image_embedding_info->status = IMAGE_EMBEDDING_GENERATE_FAIL;
      image_embedding_info->image_embedding_log = "Image embedding query not handled by downstream elements";
      image_embedding_info->err_info.code = StatusInternalServerError;
      image_embedding_info->data = Json::Value(Json::arrayValue);
    }

    gst_query_unref (query);

  } else {
    g_print("Unsupported REST API version\n");
    image_embedding_info->status = IMAGE_EMBEDDING_GENERATE_FAIL;
    image_embedding_info->image_embedding_log = "Unsupported REST API version";
    image_embedding_info->err_info.code = StatusBadRequest;
  }
}

static void
s_appinstance_api_impl (NvDsServerAppInstanceInfo * appinstance_info, void *ctx)
{

  GstDsNvMultiUriBin *nvmultiurisrcbin = (GstDsNvMultiUriBin *) ctx;
  (void) nvmultiurisrcbin;
  /* Default to 500 until a branch sets the real status; prevents a silent
   * 200 OK on an unhandled/unknown appinstance flag. */
  appinstance_info->err_info.code = StatusInternalServerError;

  if (appinstance_info->uri.find ("/api/v1/") != std::string::npos) {
    if (!s_force_eos_handle (nvmultiurisrcbin->nvmultiurisrcbinCreator,
            appinstance_info)) {
      if (appinstance_info->appinstance_flag == QUIT_APP) {
      appinstance_info->app_log =
          "QUIT_FAIL, Unable to handle force-pipeline-eos nvmultiurisrcbin";
      appinstance_info->status = QUIT_FAIL;
      appinstance_info->err_info.code = StatusInternalServerError;
      if(nvmultiurisrcbin->config->enable_error_propagation) {
        msgapi_send_message(nvmultiurisrcbin, "N/A", "N/A", appinstance_info->app_log.c_str(),TRUE);
      }
      }
    }
    else{
      if (appinstance_info->appinstance_flag == QUIT_APP) {
      appinstance_info->status = QUIT_SUCCESS;
      appinstance_info->err_info.code = StatusOk;
      appinstance_info->app_log = "QUIT_SUCCESS";
      }
    }
  } else {
    g_print ("Unsupported REST API version\n");
    appinstance_info->err_info.code = StatusBadRequest;
  }
}

void free_sensor_info(gpointer data) {
    NvDsSensorInfo* sensorInfo = (NvDsSensorInfo*)data;
    if (sensorInfo) {
        g_free((char*)sensorInfo->sensor_name);
        g_free((char*)sensorInfo->uri);
        g_free((char*)sensorInfo->sensor_metadata);
        g_free(sensorInfo);
    }
}

/* Read one key from a URI query-string ("...?a=1&b=2"); "" if absent. Values are taken
 * verbatim (our status filter keys -- model/version/stream/source_id -- carry no
 * percent-encoded characters). */
static std::string
s_uri_query_get (const std::string & uri, const char *key)
{
  size_t qpos = uri.find ('?');
  if (qpos == std::string::npos)
    return "";
  std::string qs = uri.substr (qpos + 1);
  std::string k = std::string (key) + "=";
  size_t p = 0;
  while (p <= qs.size ()) {
    size_t amp = qs.find ('&', p);
    size_t len = (amp == std::string::npos) ? std::string::npos : (amp - p);
    std::string tok = qs.substr (p, len);
    if (tok.compare (0, k.size (), k) == 0)
      return tok.substr (k.size ());
    if (amp == std::string::npos)
      break;
    p = amp + 1;
  }
  return "";
}

/* TRUE when the query string contains the key at all, regardless of its value.
 *
 * s_uri_query_get returns "" for BOTH "key absent" and "key present but empty",
 * so it cannot distinguish `?` from `?stream=`. Treating the latter as "no
 * filter" makes an empty filter FAIL OPEN and return every route. This lets the
 * caller tell the two apart and fail closed. */
static gboolean
s_uri_query_has (const std::string & uri, const char *key)
{
  size_t qpos = uri.find ('?');
  if (qpos == std::string::npos)
    return FALSE;
  std::string qs = uri.substr (qpos + 1);
  std::string k = std::string (key) + "=";
  size_t p = 0;
  while (p <= qs.size ()) {
    size_t amp = qs.find ('&', p);
    size_t len = (amp == std::string::npos) ? std::string::npos : (amp - p);
    /* "stream=..." OR a bare "stream" with no '=' at all -- both are the key
     * being PRESENT. Matching only the "key=" form would let `?stream` fall
     * through as "no filter" and return every route. */
    if (qs.compare (p, k.size (), k) == 0 || qs.substr (p, len) == key)
      return TRUE;
    if (amp == std::string::npos)
      break;
    p = amp + 1;
  }
  return FALSE;
}

/* Percent-decode one query-string value (also '+' -> space).
 *
 * s_uri_query_get takes values verbatim -- fine for model/status, whose filter
 * keys (model / version / source_id) never carry reserved characters. But
 * stream/add accepts ANY non-empty camera_id, so a route lookup for an id
 * containing a space, '&', '=' or '%' could not round-trip without decoding.
 * Applied ONLY to stream/route's ?stream=; model/status's parsing is untouched. */
static std::string
s_uri_unescape (const std::string & in)
{
  std::string out;
  out.reserve (in.size ());
  for (size_t i = 0; i < in.size (); i++) {
    if (in[i] == '+') {
      out += ' ';
    } else if (in[i] == '%' && i + 2 < in.size ()
        && g_ascii_isxdigit (in[i + 1]) && g_ascii_isxdigit (in[i + 2])) {
      out += (char) ((g_ascii_xdigit_value (in[i + 1]) << 4)
          | g_ascii_xdigit_value (in[i + 2]));
      i += 2;
    } else {
      out += in[i];
    }
  }
  return out;
}

/* Ask the downstream app for a model-status snapshot. Used by the model/status
 * GET only -- stream/route is a SEPARATE protocol: it builds its own
 * stream-route query and the element answers that plane whole, so there is no
 * shared projection step to factor out here. Returns FALSE if nothing answered
 * or the answer was not parseable. */
static gboolean
s_query_model_status (GstDsNvMultiUriBin * nvmultiurisrcbin,
    const gchar * f_model, const gchar * f_ver, const gchar * f_stream,
    gint sid, Json::Value * out, gboolean * parse_failed)
{
  GstQuery *query = gst_nvquery_model_status_new_filtered (f_model, f_ver, f_stream, sid);
  const gchar *json = NULL;
  GstPad *srcpad = (GstPad *) nvmultiurisrcbin->bin_src_pad;
  gboolean ok = FALSE;

  *parse_failed = FALSE;
  if (srcpad && gst_pad_peer_query (srcpad, query)
      && gst_nvquery_model_status_parse_response (query, &json) && json && *json) {
    Json::Reader reader;
    if (reader.parse (std::string (json), *out))
      ok = TRUE;
    else
      *parse_failed = TRUE;
  }
  gst_query_unref (query);
  return ok;
}


static void
s_get_request_api_impl (NvDsServerGetRequestInfo * get_request_info, void *ctx)
{
  GstDsNvMultiUriBin *nvmultiurisrcbin = (GstDsNvMultiUriBin *) ctx;
  (void) nvmultiurisrcbin;

  /* stream/route (GET): the ROUTING plane. Answered whole by the model-managing
   * element downstream -- it owns both the routing state and, since it records
   * the routable camera_id per stream, the identifiers the response is keyed by.
   * Nothing is projected or joined here; this forwards the body, exactly as the
   * model/status branch below does. Fails safe (FAIL) if no handler answers. */
  if (get_request_info->get_request_flag == GET_STREAM_ROUTE_INFO) {
    /* ?stream=<camera_id> -- percent-decoded, and PRESENCE-checked so that a
     * present-but-empty `?stream=` is a filter matching nothing rather than an
     * absent filter matching everything. An embedded NUL is rejected for the
     * same reason: c_str() would hand the element "" and it would answer with
     * every route. Both are passed down as a value no stream can hold. */
    const std::string raw = s_uri_query_get (get_request_info->uri, "stream");
    std::string f_stream = s_uri_unescape (raw);
    gboolean has_stream_filter = s_uri_query_has (get_request_info->uri, "stream");
    GstQuery *query;
    const gchar *json = NULL;
    GstPad *srcpad = (GstPad *) nvmultiurisrcbin->bin_src_pad;
    gboolean ok = FALSE, parse_failed = FALSE;
    Json::Value root;

    /* An unusable value stays a PRESENT filter carrying "", which the element
     * treats as matching nothing. Truncating at an embedded NUL is what must be
     * avoided: c_str() would hand it a shorter string that might match a real
     * camera. */
    if (has_stream_filter && f_stream.find ('\0') != std::string::npos)
      f_stream.clear ();

    query = gst_nvquery_stream_route_new_filtered (has_stream_filter ? f_stream.c_str () : NULL);
    if (srcpad && gst_pad_peer_query (srcpad, query)
        && gst_nvquery_stream_route_parse_response (query, &json) && json && *json) {
      Json::Reader reader;
      /* Parseable is not the same as VALID. A downstream element that answered
       * with {}, null, "ok", or a version-skewed body is a failure, not an empty
       * routing plane -- forwarding it as 200 would tell a reconciler the fleet
       * has no routes and invite it to erase desired state. Require the shape
       * this endpoint promises: an object carrying a routes[] array. */
      if (reader.parse (std::string (json), root)) {
        if (root.isObject () && root.isMember ("routes")
            && root["routes"].isArray ())
          ok = TRUE;
        else
          parse_failed = TRUE;
      } else {
        parse_failed = TRUE;
      }
    }
    gst_query_unref (query);

    if (ok) {
      get_request_info->stream_info = root;
      get_request_info->status = GET_STREAM_ROUTE_INFO_SUCCESS;
      get_request_info->err_info.code = StatusOk;
      get_request_info->get_request_log = "GET_STREAM_ROUTE_INFO_SUCCESS";
    } else {
      /* Machine-readable identity alongside the prose: the POST on this uri
       * already answers failures as {code,message,hint}, so the GET carries the
       * same vocabulary and one client-side parser covers both verbs. The two
       * failures are distinguishable -- a malformed body is the element's bug,
       * an unanswered query means nothing downstream handles the plane. */
      get_request_info->status = GET_STREAM_ROUTE_INFO_FAIL;
      get_request_info->err_info.code = StatusInternalServerError;
      get_request_info->err_info.err_code = parse_failed
          ? "ROUTE_SNAPSHOT_MALFORMED" : "ROUTE_SNAPSHOT_UNAVAILABLE";
      get_request_info->err_info.hint = parse_failed
          ? "the model-managing element returned a body that is not valid JSON"
          : "no downstream element answered the stream-route query -- is nvmodelmux in the pipeline?";
      get_request_info->get_request_log = parse_failed
          ? "GET_STREAM_ROUTE_INFO_FAIL, malformed route json"
          : "GET_STREAM_ROUTE_INFO_FAIL, application did not answer (no downstream handler?)";
    }
    return;
  }

  /* model/status: the model pool lives in the downstream app, not here. Ask it
   * synchronously via a custom query sent downstream from our src pad; the app
   * answers (JSON) at its bin boundary. Fails safe (FAIL) if no handler answers.
   * Optional GET filters (v1 vocabulary -- flat projections of the {name,
   * version} ref, since query strings cannot nest):
   *   ?model=X[&version=N] -> that model (join semantics downstream);
   *   ?stream=Y | ?source_id=N -> that stream. */
  if (get_request_info->get_request_flag == GET_MODEL_STATUS_INFO) {
    std::string f_model  = s_uri_query_get (get_request_info->uri, "model");
    std::string f_ver    = s_uri_query_get (get_request_info->uri, "version");
    std::string f_stream = s_uri_query_get (get_request_info->uri, "stream");
    std::string f_sid    = s_uri_query_get (get_request_info->uri, "source_id");
    gint sid = f_sid.empty () ? -1 : atoi (f_sid.c_str ());
    Json::Value root;
    gboolean parse_failed = FALSE;

    /* version is a refinement of model (separate params, never combined).
     * The query/peer_query/parse sequence is shared with stream/route's GET
     * (s_query_model_status) -- same construction, same lifetime, same
     * success/failure classification as before, just factored out so the two
     * readers cannot drift apart. */
    if (s_query_model_status (nvmultiurisrcbin,
            f_model.empty ()  ? NULL : f_model.c_str (),
            f_ver.empty ()    ? NULL : f_ver.c_str (),
            f_stream.empty () ? NULL : f_stream.c_str (), sid, &root,
            &parse_failed)) {
      get_request_info->stream_info = root;
      get_request_info->status = GET_MODEL_STATUS_INFO_SUCCESS;
      get_request_info->err_info.code = StatusOk;
      get_request_info->get_request_log = "GET_MODEL_STATUS_INFO_SUCCESS";
    } else {
      get_request_info->status = GET_MODEL_STATUS_INFO_FAIL;
      get_request_info->err_info.code = StatusInternalServerError;
      get_request_info->get_request_log = parse_failed
          ? "GET_MODEL_STATUS_INFO_FAIL, malformed status json"
          : "GET_MODEL_STATUS_INFO_FAIL, application did not answer (no downstream handler?)";
    }
    return;
  }

  if (get_request_info->uri.find ("/api/v1/") != std::string::npos) {
    GList *stream_info_list = NULL;

    if (gst_nvmultiurisrcbincreator_get_source_info_list (
            nvmultiurisrcbin->nvmultiurisrcbinCreator, &stream_info_list)) {
      // Create a JSON object to store the stream information
      Json::Value root;
      root["stream-count"] = g_list_length(stream_info_list);
      // Create a JSON array to store stream info
      Json::Value streamInfo(Json::arrayValue);

      if (get_request_info->get_request_flag == GET_LIVE_STREAM_INFO) {
        get_request_info->status = GET_LIVE_STREAM_INFO_SUCCESS;
        get_request_info->err_info.code = StatusOk;
        get_request_info->get_request_log = "GET_LIVE_STREAM_INFO_SUCCESS";

        // Log information from the retrieved list
        for (GList *temp = stream_info_list; temp; temp = g_list_next (temp)) {
          NvDsSensorInfo *sensor_info = (NvDsSensorInfo *) temp->data;
          //Create Json stream info objects for each active stream
          Json::Value stream;
          if (sensor_info->source_id <=UINT_MAX)
            stream["source_id"] = sensor_info->source_id;
          else
            stream["source_id"] = -1;
          if (sensor_info->sensor_id)
            stream["camera_id"] = std::string(sensor_info->sensor_id);
          else
            stream["camera_id"] = std::string("");

          if (sensor_info->sensor_name)
            stream["camera_name"] = std::string(sensor_info->sensor_name);
          else
            stream["camera_name"] = std::string("");

          //Add the stream info objects into stream info JSON array
          streamInfo.append(stream);

        }
        // Add the stream info JSON array to the root object
        root["stream-info"] = streamInfo;
        get_request_info->stream_info = root;

      } else if (get_request_info->get_request_flag == GET_READY_INFO) {

        GstState curr_state = GST_STATE_NULL;
        GstState pending_state = GST_STATE_NULL;
        gst_element_get_state(GST_ELEMENT(nvmultiurisrcbin), &curr_state, &pending_state, GST_CLOCK_TIME_NONE);

        const gchar * state_name =gst_element_state_get_name(curr_state);

        Json::Value root;
        if ((!g_strcmp0 (state_name, "PLAYING")) || (!g_strcmp0 (state_name, "PAUSED"))){
          root["ds-ready"]="YES";
        } else {
          root["ds-ready"]="NO";
        }
        get_request_info->stream_info = root;

        get_request_info->status = GET_READY_INFO_SUCCESS;
        get_request_info->err_info.code = StatusOk;
        get_request_info->get_request_log = "GET_READY_INFO_SUCCESS";

      } else if (get_request_info->get_request_flag == GET_METADATA_INFO) {
        Json::Value root;
        root["version"] = "1.0.0";
        root["sub-version"] = "";

        Json::Value licenseInfo;
        licenseInfo["name"] = "NVIDIA-Proprietary";
        licenseInfo["path"] = "/opt/mm/LICENSE";
        licenseInfo["url"] = "file:///opt/mm/LICENSE";
        root["licenseInfo"] = licenseInfo;

        get_request_info->stream_info = root;
        get_request_info->status = GET_METADATA_INFO_SUCCESS;
        get_request_info->err_info.code = StatusOk;
        get_request_info->get_request_log = "GET_METADATA_INFO_SUCCESS";

      } else if (get_request_info->get_request_flag == GET_METRICS_INFO) {
        get_request_info->status = GET_METRICS_INFO_SUCCESS;
        get_request_info->err_info.code = StatusOk;
        get_request_info->get_request_log = "GET_METRICS_INFO_SUCCESS";

        // Get latency and FPS data using local counts (do NOT pass global fields directly)
        guint latency_count = 0;
        guint fps_count = 0;
        NvDsMetricsFrameLatency *current_latency_data = nvds_get_shared_frame_latency_data(&latency_count);
        NvDsMetricsFpsData *current_fps_data = nvds_get_shared_fps_data(&fps_count);

        // Prepare stream-info list mapping (sensor names/ids)
        GList *stream_info_list = NULL;
        std::map<guint, NvDsSensorInfo*> source_id_to_sensor_info;
        if (gst_nvmultiurisrcbincreator_get_source_info_list(
                nvmultiurisrcbin->nvmultiurisrcbinCreator, &stream_info_list)) {
            for (GList *temp = stream_info_list; temp; temp = g_list_next(temp)) {
                NvDsSensorInfo *sensor_info = (NvDsSensorInfo *)temp->data;
                if (sensor_info)
                    source_id_to_sensor_info[sensor_info->source_id] = sensor_info;
            }
        }

        if (current_latency_data && latency_count > 0) {
            // Create the root JSON object
            Json::Value metricsInfo(Json::objectValue);
            Json::Value dsMetric(Json::arrayValue);

            // Helper lambda to find fps by source id
            auto find_fps = [&](guint sid) -> double {
                if (!current_fps_data || fps_count == 0) return 0.0;
                for (guint j = 0; j < fps_count; j++) {
                    if ((guint)current_fps_data[j].source_id == sid) {
                        return (double) current_fps_data[j].fps_val;
                    }
                }
                return 0.0;
            };

            // Build ds-metric array and filter out data for inactive sources
            guint active_stream_count = 0;
            for (guint i = 0; i < latency_count; i++) {
                guint source_id = current_latency_data[i].source_id;

                // Only include data for sources that are still in the active list
                if (source_id_to_sensor_info.find(source_id) == source_id_to_sensor_info.end()) {
                    continue;
                }

                Json::Value metric(Json::objectValue);
                // Add source_id
                metric["source_id"] = static_cast<Json::UInt>(source_id);

                // Add sensor_name and sensor_id (look up from stream info map)
                NvDsSensorInfo *sensor_info = source_id_to_sensor_info[source_id];
                metric["sensor_name"] = sensor_info && sensor_info->sensor_name ? std::string(sensor_info->sensor_name) : "";
                metric["sensor_id"] = sensor_info && sensor_info->sensor_id ? std::string(sensor_info->sensor_id) : "";

                // Add fps (lookup by source_id)
                metric["fps"] = static_cast<double>(find_fps(source_id));

                // Add latency_ms (assumed already in milliseconds)
                metric["latency_ms"] = static_cast<double>(current_latency_data[i].latency);

                // Add frame_number
                metric["frame_number"] = static_cast<Json::UInt>(current_latency_data[i].frame_num);

                // Append
                dsMetric.append(metric);
                active_stream_count++;
            }

            // Create system stats placeholder
            Json::Value systemStats(Json::objectValue);
            systemStats["gpu_util"] = get_gpu_utilization(nvmultiurisrcbin->muxConfig->gpu_id);  // GPU utilization percentage
            systemStats["GPU_gb"] = get_gpu_memory(nvmultiurisrcbin->muxConfig->gpu_id);
            systemStats["cpu_util"] = get_cpu_utilization();  // CPU utilization percentage
            systemStats["RAM_gb"] = get_ram_usage();  // RAM usage in GB

            // Build final JSON with actual active stream count
            metricsInfo["stream-count"] = static_cast<Json::UInt>(active_stream_count);
            metricsInfo["stream-stats"] = dsMetric;
            metricsInfo["system-stats"] = systemStats;

            // Update the response under mutex to avoid races when other threads read/write
            g_mutex_lock(&g_shared_frame_latency_data.frame_metrics_mutex);
            get_request_info->stream_info.clear();
            get_request_info->stream_info = metricsInfo;
            g_mutex_unlock(&g_shared_frame_latency_data.frame_metrics_mutex);

            get_request_info->status = GET_METRICS_INFO_SUCCESS;
            get_request_info->err_info.code = StatusOk;
            get_request_info->get_request_log = "GET_METRICS_INFO_SUCCESS";
        } else {
            // No data available -> return empty structured response
            Json::Value metricsInfo(Json::objectValue);
            Json::Value dsMetric(Json::arrayValue);
            Json::Value systemStats(Json::objectValue);
            systemStats["gpu_util"] = get_gpu_utilization(nvmultiurisrcbin->muxConfig->gpu_id);
            systemStats["GPU_gb"] = get_gpu_memory(nvmultiurisrcbin->muxConfig->gpu_id);
            systemStats["cpu_util"] = get_cpu_utilization();  // CPU utilization percentage
            systemStats["RAM_gb"] = get_ram_usage();  // RAM usage in GB

            metricsInfo["stream-count"] = static_cast<Json::UInt>(0);
            metricsInfo["stream-stats"] = dsMetric;
            metricsInfo["system-stats"] = systemStats;

            g_mutex_lock(&g_shared_frame_latency_data.frame_metrics_mutex);
            get_request_info->stream_info.clear();
            get_request_info->stream_info = metricsInfo;
            g_mutex_unlock(&g_shared_frame_latency_data.frame_metrics_mutex);

            get_request_info->status = GET_METRICS_INFO_SUCCESS;
            get_request_info->err_info.code = StatusOk;
            get_request_info->get_request_log = "GET_METRICS_INFO_SUCCESS - No data available";
        }

        if (stream_info_list) {
            g_list_free_full(stream_info_list, (GDestroyNotify)free_sensor_info);
            stream_info_list = NULL;
        }
      }
      else {
          // Handle other GET flags properly
          switch(get_request_info->get_request_flag) {
              case GET_LIVE_STREAM_INFO:
                  get_request_info->status = GET_LIVE_STREAM_INFO_FAIL;
                  break;
              case  GET_READY_INFO:
                  get_request_info->status = GET_READY_INFO_FAIL;
                  break;
              case GET_METRICS_INFO:
                  get_request_info->status = GET_METRICS_INFO_FAIL;
                  break;
              case GET_METADATA_INFO:
                  get_request_info->status = GET_METADATA_INFO_FAIL;
                  break;
              default:
                  break;
          }
          get_request_info->err_info.code = StatusInternalServerError;
          get_request_info->get_request_log = "Unable to fetch the data";
      }
      /* Release the list fetched at the top of this block: every branch above is
       * done with it, and GET_METRICS_INFO frees only its OWN inner list, which
       * shadows this one -- so without this it leaked on every GET. */
      if (stream_info_list) {
        g_list_free_full (stream_info_list, (GDestroyNotify) free_sensor_info);
        stream_info_list = NULL;
      }
    } else {
      /* Failed to fetch the source info list -> report 500 instead of
       * silently defaulting to 200 OK. */
      switch (get_request_info->get_request_flag) {
        case GET_LIVE_STREAM_INFO:
          get_request_info->status = GET_LIVE_STREAM_INFO_FAIL;
          break;
        case GET_READY_INFO:
          get_request_info->status = GET_READY_INFO_FAIL;
          break;
        case GET_METRICS_INFO:
          get_request_info->status = GET_METRICS_INFO_FAIL;
          break;
        case GET_METADATA_INFO:
          get_request_info->status = GET_METADATA_INFO_FAIL;
          break;
        default:
          break;
      }
      get_request_info->err_info.code = StatusInternalServerError;
      get_request_info->get_request_log = "Unable to fetch the source info list";
    }
  }
  if (get_request_info->uri.find ("/ready") != std::string::npos ||
      get_request_info->uri.find ("/live") != std::string::npos ||
       get_request_info->uri.find ("/startup") != std::string::npos) {
    GstState curr_state = GST_STATE_NULL;
    GstState pending_state = GST_STATE_NULL;
    gst_element_get_state(GST_ELEMENT(nvmultiurisrcbin), &curr_state, &pending_state, GST_CLOCK_TIME_NONE);

    const gchar * state_name =gst_element_state_get_name(curr_state);

    if(get_request_info->get_request_flag == GET_READY_INFO) {
      if ((!g_strcmp0 (state_name, "PLAYING")) || (!g_strcmp0 (state_name, "PAUSED"))){
        get_request_info->stream_info["ds-ready"]="YES";
        get_request_info->err_info.code = StatusOk;
      } else {
        get_request_info->stream_info["ds-ready"]="NO";
        /* Not ready (pipeline not PLAYING/PAUSED) -> 503 so a readiness httpGet
         * probe withholds traffic from the pod until it is ready. */
        get_request_info->err_info.code = StatusServiceUnavailable;
      }
      get_request_info->status = GET_READY_INFO_SUCCESS;
      get_request_info->get_request_log = "GET_READY_INFO_SUCCESS";

    } else if (get_request_info->get_request_flag == GET_LIVE_INFO) {
      if ( (!g_strcmp0 (state_name, "PLAYING")) ){
        get_request_info->stream_info["ds-liveness"]="YES";
        get_request_info->err_info.code = StatusOk;
      } else {
        get_request_info->stream_info["ds-liveness"]="NO";
        /* Pipeline not PLAYING -> report 503 so an httpGet probe detects "not
         * live" from the status code, not just the body. NOTE: pair /live with a
         * Kubernetes startupProbe (e.g. /startup) so this does not restart the
         * pod during the normal NULL->READY->PAUSED->PLAYING startup. */
        get_request_info->err_info.code = StatusServiceUnavailable;
      }
      get_request_info->status = GET_LIVE_INFO_SUCCESS;
      get_request_info->get_request_log = "GET_LIVE_INFO_SUCCESS";

    } else if (get_request_info->get_request_flag == GET_STARTUP_INFO) {
      if ((!g_strcmp0 (state_name, "PLAYING")) || (!g_strcmp0 (state_name, "PAUSED"))){
        get_request_info->stream_info["ds-startup"]="YES";
        get_request_info->err_info.code = StatusOk;
      } else {
        get_request_info->stream_info["ds-startup"]="NO";
        /* Not started yet -> 503 so a startupProbe keeps waiting (and gates
         * liveness/readiness) until the pipeline reaches PLAYING/PAUSED. */
        get_request_info->err_info.code = StatusServiceUnavailable;
      }
      get_request_info->status = GET_STARTUP_INFO_SUCCESS;
      get_request_info->get_request_log = "GET_STARTUP_INFO_SUCCESS";

    } else {
      switch(get_request_info->get_request_flag) {
        case GET_READY_INFO:
          get_request_info->status = GET_READY_INFO_FAIL;
          break;
        case GET_LIVE_INFO:
          get_request_info->status = GET_LIVE_INFO_FAIL;
          break;
        case GET_STARTUP_INFO:
          get_request_info->status = GET_STARTUP_INFO_FAIL;
          break;
        default:
          break;
        }
        get_request_info->err_info.code = StatusInternalServerError;
        get_request_info->get_request_log = "Unable to fetch the data";
    }
  }
  /* URI matched neither a /api/v1/ request nor a /ready|/live|/startup probe
   * -> unsupported endpoint; report 400 instead of defaulting to 200 OK. */
  if (get_request_info->uri.find ("/api/v1/") == std::string::npos &&
      get_request_info->uri.find ("/ready") == std::string::npos &&
      get_request_info->uri.find ("/live") == std::string::npos &&
      get_request_info->uri.find ("/startup") == std::string::npos) {
    g_print ("Unsupported REST API version \n");
    get_request_info->err_info.code = StatusBadRequest;
    get_request_info->get_request_log = "Unsupported REST API version";
  }
}

static void
rest_api_server_start (GstDsNvMultiUriBin * nvmultiurisrcbin)
{
  NvDsServerCallbacks server_cb = { };

  server_cb.stream_cb =
      [nvmultiurisrcbin] (NvDsServerStreamInfo * stream_info, void *ctx) {
    s_stream_api_impl (stream_info, (void *) nvmultiurisrcbin);
  };
  server_cb.model_cb =
      [nvmultiurisrcbin] (NvDsServerModelInfo * model_info, void *ctx) {
    s_model_api_impl (model_info, (void *) nvmultiurisrcbin);
  };
  server_cb.route_cb =
      [nvmultiurisrcbin] (NvDsServerRouteInfo * route_info, void *ctx) {
    s_route_api_impl (route_info, (void *) nvmultiurisrcbin);
  };
  server_cb.roi_cb =[nvmultiurisrcbin] (NvDsServerRoiInfo * roi_info, void *ctx) {
    s_roi_api_impl (roi_info, (void *) nvmultiurisrcbin);
  };
  server_cb.dec_cb =[nvmultiurisrcbin] (NvDsServerDecInfo * dec_info, void *ctx) {
    s_dec_api_impl (dec_info, (void *) nvmultiurisrcbin);
  };
  server_cb.infer_cb =[nvmultiurisrcbin] (NvDsServerInferInfo * infer_info, void *ctx) {
    s_infer_api_impl (infer_info, (void *) nvmultiurisrcbin);
  };
  server_cb.inferserver_cb =
      [nvmultiurisrcbin] (NvDsServerInferServerInfo * inferserver_info, void *ctx) {
    s_inferserver_api_impl (inferserver_info, (void *) nvmultiurisrcbin);
  };
  server_cb.nvTracker_cb = [nvmultiurisrcbin] (NvDsServerNvTrackerInfo * nvtracker_info, void *ctx){
    s_nvtracker_api_impl(nvtracker_info, (void *) nvmultiurisrcbin);
  };
  server_cb.conv_cb =[nvmultiurisrcbin] (NvDsServerConvInfo * conv_info, void *ctx) {
    s_conv_api_impl (conv_info, (void *) nvmultiurisrcbin);
  };
  server_cb.enc_cb =[nvmultiurisrcbin] (NvDsServerEncInfo * enc_info, void *ctx) {
    s_enc_api_impl (enc_info, (void *) nvmultiurisrcbin);
  };
  server_cb.mux_cb =[nvmultiurisrcbin] (NvDsServerMuxInfo * mux_info, void *ctx) {
    s_mux_api_impl (mux_info, (void *) nvmultiurisrcbin);
  };

  server_cb.osd_cb =[nvmultiurisrcbin] (NvDsServerOsdInfo * osd_info, void *ctx) {
    s_osd_api_impl (osd_info, (void *) nvmultiurisrcbin);
  };
  server_cb.analytics_cb = [nvmultiurisrcbin] (NvDsServerAnalyticsInfo * analytics_info, void *ctx) {
    s_analytics_api_impl (analytics_info, (void *) nvmultiurisrcbin);
  };
  server_cb.text_embedding_cb = [nvmultiurisrcbin] (NvDsServerTextEmbeddingInfo * text_embedding_info, void *ctx) {
    s_text_embedding_api_impl (text_embedding_info, (void *) nvmultiurisrcbin);
  };
  server_cb.image_embedding_cb = [nvmultiurisrcbin] (NvDsServerImageEmbeddingInfo * image_embedding_info, void *ctx) {
    s_image_embedding_api_impl (image_embedding_info, (void *) nvmultiurisrcbin);
  };

  server_cb.appinstance_cb =
      [nvmultiurisrcbin] (NvDsServerAppInstanceInfo * appinstance_info, void *ctx) {
    s_appinstance_api_impl (appinstance_info, (void *) nvmultiurisrcbin);
  };

  server_cb.get_request_cb =
      [nvmultiurisrcbin] (NvDsServerGetRequestInfo * get_request_info, void *ctx) {
        if(ctx){
          *(void **)ctx = (void *) nvmultiurisrcbin->nvmultiurisrcbinCreator;
        }
    s_get_request_api_impl (get_request_info, (void *) nvmultiurisrcbin);
  };

  NvDsServerConfig server_conf = { };
  server_conf.ip = std::string (nvmultiurisrcbin->httpIp);
  server_conf.port = std::string (nvmultiurisrcbin->httpPort);
  nvmultiurisrcbin->restServer =
      (void *) nvds_rest_server_start (&server_conf, &server_cb, nvmultiurisrcbin->nvmultiurisrcbinCreator);
}


#ifdef ENABLE_GST_NVMULTIURISRCBIN_UNIT_TESTS

#ifndef PACKAGE
#define PACKAGE "nvdsbins"
#endif

#define VERSION "1.0"
#define LICENSE "MIT"
#define DESCRIPTION "NVIDIA Multiurisrcbin"
#define BINARY_PACKAGE "NVIDIA DeepStream Bins"
#define URL "http://nvidia.com/"

extern "C" gboolean
plugin_init_2 (GstPlugin * plugin)
{
  if (!gst_element_register (plugin, "nvmultiurisrcbin", GST_RANK_PRIMARY,
          GST_TYPE_DS_NVMULTIURISRC_BIN))
    return FALSE;

  return TRUE;
}

extern "C" gboolean gGstNvMultiUriSrcBinStaticInit ();
gboolean
gGstNvMultiUriSrcBinStaticInit ()
{
  return gst_plugin_register_static (GST_VERSION_MAJOR, GST_VERSION_MINOR,
      "nvdsgst_deepstream_bins2",
      DESCRIPTION, plugin_init_2, "9.1", LICENSE, BINARY_PACKAGE, PACKAGE,
      URL);
}
#endif
