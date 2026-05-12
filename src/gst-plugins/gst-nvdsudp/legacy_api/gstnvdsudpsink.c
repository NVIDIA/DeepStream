/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#define _GNU_SOURCE
#include <gst/gst.h>
#include <gst/base/gstbasesink.h>
#include <gst/sdp/sdp.h>
#include <stdio.h>
#include "gstnvdsudpsink.h"

#include <string.h>
#include <arpa/inet.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include "nvbufsurface.h"

GST_DEBUG_CATEGORY_STATIC (gst_nvdsudpsink_debug_category);
#define GST_CAT_DEFAULT gst_nvdsudpsink_debug_category

#define DEFAULT_LOCAL_IFACE_IP     NULL
#define DEFAULT_AUTO_MULTICAST     FALSE
#define DEFAULT_SOCKET             NULL
#define DEFAULT_CLOSE_SOCKET       TRUE
#define DEFAULT_LOOP               TRUE
#define UDP_DEFAULT_HOST           "0.0.0.0"
#define UDP_DEFAULT_PORT           5004
#define DEFAULT_PAYLOAD_SIZE       1400
#define DEFAULT_PACKETS            10
#define DEFAULT_CHUNK_SIZE         100
#define DEFAULT_SDP_FILE           NULL
#define FHD_HEIGHT                 (1080)
#define FHD_WIDTH                  (1920)
#define SLEEP_THRESHOLD_MS         (5)
#define DEFAULT_FRAMES_PER_BLOCK   (10)
#define DEFAULT_PACKETS_PER_LINE   (4)
#define RTP_HEADER_SIZE            (12)
#define MAX_CPU_CORE               (1023)
#define RTP_2110_20_MIN_HEADER_SIZE (20)
#define DEFAULT_PAYLOAD_TYPE       (96)
#define GPU_ID_INVALID             (-1)

#define river_align_down_pow2(_n, _alignment) \
    ( (_n) & ~((_alignment) - 1) )

#define river_align_up_pow2(_n, _alignment) \
    river_align_down_pow2((_n) + (_alignment) - 1, _alignment)

#define RMAX_CPUELT(_cpu)  ((_cpu) / RMAX_NCPUBITS)
#define RMAX_CPUMASK(_cpu) ((rmax_cpu_mask_t) 1 << ((_cpu) % RMAX_NCPUBITS))
#define RMAX_CPU_SET(_cpu, _cpusetp) \
    do { \
        size_t _cpu2 = (_cpu); \
        if (_cpu2 < (8 * sizeof (struct rmax_cpu_set_t))) { \
            (((rmax_cpu_mask_t *)((_cpusetp)->rmax_bits))[RMAX_CPUELT(_cpu2)] |= \
                                      RMAX_CPUMASK(_cpu2)); \
        } \
    } while (0)

enum
{
  PROP_0,
  PROP_LOCAL_IFACE_IP,
  PROP_HOST,
  PROP_PORT,
  PROP_PAYLOAD_SIZE,
  PROP_CHUNK_SIZE,
  PROP_PACKET_PER_CHUNK,
  PROP_PACKET_PER_LINE,
  PROP_SDP_FILE,
  PROP_INTERNAL_THREAD_CORE,
  PROP_PTP_SOURCE,
  PROP_RENDER_THREAD_CORE,
  PROP_GPU_ID,
  /* These are dummy properties. These have been defined just to avoid warnings
   from rtspsrc */
  PROP_AUTO_MULTICAST,
  PROP_TTL,
  PROP_LOOP,
  PROP_SOCKET,
  PROP_CLOSE_SOCKET,
  PROP_PASS_RTP_TIMESTAMP,
  PROP_RTP_TIMESTAMP_OFFSET,
};

static void gst_nvdsudpsink_set_property (GObject * object,
    guint property_id, const GValue * value, GParamSpec * pspec);
static void gst_nvdsudpsink_get_property (GObject * object,
    guint property_id, GValue * value, GParamSpec * pspec);
static void gst_nvdsudpsink_finalize (GObject * object);
static gboolean gst_nvdsudpsink_start (GstBaseSink * bsink);
static gboolean gst_nvdsudpsink_stop (GstBaseSink * bsink);
static GstFlowReturn gst_nvdsudpsink_render (GstBaseSink * bsink,
    GstBuffer * buffer);
static GstFlowReturn gst_nvdsudpsink_render_list (GstBaseSink * bsink,
    GstBufferList * buffer_list);

static gboolean gst_nvdsudpsink_set_caps (GstBaseSink * sink, GstCaps * caps);
static GstFlowReturn
gst_nvdsudpsink_render_raw_frame (GstBaseSink *bsink, GstBuffer *buffer);

static gpointer render_thread (gpointer data);

static void
gst_nvdsudpsink_uri_handler_init (gpointer g_iface, gpointer iface_data);

static guint
parse_sdp_source_filter_ips(const char *sdp_content, char **source_ips, guint max_ips);

static GstStaticPadTemplate gst_nvdsudpsink_sink_template =
GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS ("ANY")
    );

G_DEFINE_TYPE_WITH_CODE (GstNvDsUdpSink, gst_nvdsudpsink, GST_TYPE_BASE_SINK,
  G_IMPLEMENT_INTERFACE (GST_TYPE_URI_HANDLER, gst_nvdsudpsink_uri_handler_init));


/*** GSTURIHANDLER INTERFACE *************************************************/

/**
 * @brief Sets the URI for the UDP sink
 * @param sink The GstNvDsUdpSink instance
 * @param uri The URI to set
 * @param error Error pointer to store any errors
 * @return TRUE if URI was set successfully, FALSE otherwise
 */
static gboolean
gst_nvdsudpsink_set_uri (GstNvDsUdpSink * sink, const gchar * uri, GError ** error)
{
  gchar *host;
  guint16 port;

  if (!sink->localIfaceIp) {
    if (error != NULL) {
      g_set_error (error, GST_URI_ERROR, GST_URI_ERROR_BAD_STATE,
            "local interface ip not set");
    }
    return FALSE;
  }

  if (!gst_udp_parse_uri (uri, &host, &port))
    goto wrong_uri;

  g_free (sink->host);
  sink->host = host;
  sink->port = port;

  g_free (sink->uri);
  sink->uri = g_strdup (uri);

  return TRUE;

wrong_uri:
  {
    GST_ELEMENT_ERROR (sink, RESOURCE, READ, (NULL),
        ("error parsing uri %s", uri));
    g_set_error_literal (error, GST_URI_ERROR, GST_URI_ERROR_BAD_URI,
        "Could not parse UDP URI");
    return FALSE;
  }

  return TRUE;
}

static GstURIType
gst_nvdsudpsink_uri_get_type (GType type)
{
  return GST_URI_SINK;
}

/**
 * @brief Gets the supported protocols for the sink
 * @param type The GType of the sink
 * @return Array of supported protocols (currently only "udp")
 */
static const gchar *const *
gst_nvdsudpsink_uri_get_protocols (GType type)
{
  static const gchar *protocols[] = { "udp", NULL };

  return protocols;
}

/**
 * @brief Gets the current URI of the sink
 * @param handler The URI handler instance
 * @return The current URI as a string
 */
static gchar *
gst_nvdsudpsink_uri_get_uri (GstURIHandler * handler)
{
  GstNvDsUdpSink *sink = GST_NVDSUDPSINK (handler);

  return g_strdup (sink->uri);
}

/**
 * @brief Sets the URI for the sink through the URI handler interface
 * @param handler The URI handler instance
 * @param uri The URI to set
 * @param error Error pointer to store any errors
 * @return TRUE if URI was set successfully, FALSE otherwise
 */
static gboolean
gst_nvdsudpsink_uri_set_uri (GstURIHandler * handler, const gchar * uri,
    GError ** error)
{
  return gst_nvdsudpsink_set_uri (GST_NVDSUDPSINK (handler), uri, error);
}

static void
gst_nvdsudpsink_uri_handler_init (gpointer g_iface, gpointer iface_data)
{
  GstURIHandlerInterface *iface = (GstURIHandlerInterface *) g_iface;

  iface->get_type = gst_nvdsudpsink_uri_get_type;
  iface->get_protocols = gst_nvdsudpsink_uri_get_protocols;
  iface->get_uri = gst_nvdsudpsink_uri_get_uri;
  iface->set_uri = gst_nvdsudpsink_uri_set_uri;
}

static void
gst_nvdsudpsink_class_init (GstNvDsUdpSinkClass * klass)
{
  GObjectClass *gobject_class = G_OBJECT_CLASS (klass);
  GstBaseSinkClass *base_sink_class = GST_BASE_SINK_CLASS (klass);

  gobject_class->set_property = gst_nvdsudpsink_set_property;
  gobject_class->get_property = gst_nvdsudpsink_get_property;
  gobject_class->finalize = gst_nvdsudpsink_finalize;

  g_object_class_install_property (gobject_class, PROP_AUTO_MULTICAST,
      g_param_spec_boolean ("auto-multicast",
          "Automatically join/leave multicast groups",
          "Automatically join/leave the multicast groups, FALSE means user"
          " has to do it himself", DEFAULT_AUTO_MULTICAST,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_SOCKET,
      g_param_spec_object ("socket", "Socket Handle",
          "Socket to use for UDP sending. (NULL == allocate)",
          G_TYPE_SOCKET, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_CLOSE_SOCKET,
      g_param_spec_boolean ("close-socket", "Close socket",
          "Close socket if passed as property on state change",
          DEFAULT_CLOSE_SOCKET, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property(gobject_class, PROP_LOCAL_IFACE_IP,
      g_param_spec_string("local-iface-ip", "Local interface IP address",
          "IP Address associated with network interface through which to"
          " receive the data.",
          DEFAULT_LOCAL_IFACE_IP, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (G_OBJECT_CLASS (klass), PROP_HOST,
      g_param_spec_string ("host", "host",
          "The host/IP/Multicast group to send the packets to",
          UDP_DEFAULT_HOST, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (G_OBJECT_CLASS (klass), PROP_PORT,
      g_param_spec_int ("port", "port", "The port to send the packets to",
          0, 65535, UDP_DEFAULT_PORT,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (G_OBJECT_CLASS (klass), PROP_PAYLOAD_SIZE,
      g_param_spec_uint ("payload-size", "Payload Size",
          "Size of payload in RTP / UDP packet", 0, G_MAXUINT16,
          DEFAULT_PAYLOAD_SIZE, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (G_OBJECT_CLASS (klass), PROP_PACKET_PER_CHUNK,
      g_param_spec_uint ("packets-per-chunk", "Packets per chunk",
          "Number of packets per memory chunk", 1, G_MAXUINT16,
          DEFAULT_PACKETS, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (G_OBJECT_CLASS (klass), PROP_PACKET_PER_LINE,
      g_param_spec_uint ("packets-per-line", "Packets per line",
          "Number of packets per line, required for Rivermax media APIs", 1, G_MAXUINT16,
          DEFAULT_PACKETS_PER_LINE, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (G_OBJECT_CLASS (klass), PROP_CHUNK_SIZE,
      g_param_spec_uint ("chunk-size", "Chunk Size",
          "Number of memory chunks to allocate", 1, G_MAXUINT16,
          DEFAULT_CHUNK_SIZE, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_LOOP,
      g_param_spec_boolean ("loop", "Multicast Loopback",
          "Used for setting the multicast loop parameter. TRUE = enable,"
          " FALSE = disable", DEFAULT_LOOP,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (G_OBJECT_CLASS (klass), PROP_SDP_FILE,
      g_param_spec_string ("sdp-file", "SDP File",
          "SDP file to parse the connection details. Set this property to use\n"
          "\t\t\tRivermax media APIs for transmission. By default Rivermax Generic\n"
          "\t\t\tAPIs are used.",
          DEFAULT_SDP_FILE, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (G_OBJECT_CLASS (klass), PROP_INTERNAL_THREAD_CORE,
      g_param_spec_int ("internal-thread-core", "Internal thread core",
          "CPU core to run Rivermax internal thread, (-1 = disabled)", -1,
          MAX_CPU_CORE, -1, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property(gobject_class, PROP_PTP_SOURCE,
      g_param_spec_string("ptp-src", "PTP source",
          "IP Address of PTP source.",
          DEFAULT_PTP_SRC, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property(gobject_class, PROP_RENDER_THREAD_CORE,
      g_param_spec_string("render-thread-core", "Render thread cores",
          "Comma seperated list of CPU cores for rendering thread.",
          NULL, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_PASS_RTP_TIMESTAMP,
      g_param_spec_boolean ("pass-rtp-timestamp",
          "Pass RTP Timestamp",
          "When enabled, use RTP timestamp from upstream metadata",
          FALSE,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_RTP_TIMESTAMP_OFFSET,
      g_param_spec_uint64 ("rtp-timestamp-offset",
          "RTP Timestamp Offset (ns)",
          "Offset in nanoseconds to add to RTP timestamp when pass-rtp-timestamp is enabled (default: 0ns)",
          0, G_MAXUINT64, 0,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property(gobject_class, PROP_GPU_ID,
      g_param_spec_int("gpu-id", "GPU ID",
          "GPU ID to use for GPUDirect (-1 = disabled)",
          -1, G_MAXINT, GPU_ID_INVALID,
		  G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  gst_element_class_add_static_pad_template (GST_ELEMENT_CLASS(klass),
      &gst_nvdsudpsink_sink_template);

  gst_element_class_set_static_metadata (GST_ELEMENT_CLASS(klass),
      "UDP packet sender", "Sink/Network",
      "Send data over the network via UDP using Mellanox Rivermax APIs",
      "NVIDIA Corporation. Post on Deepstream for Tesla forum for any queries "
      "@ https://devtalk.nvidia.com/default/board/209/");

  base_sink_class->start = GST_DEBUG_FUNCPTR (gst_nvdsudpsink_start);
  base_sink_class->stop = GST_DEBUG_FUNCPTR (gst_nvdsudpsink_stop);
  base_sink_class->render = GST_DEBUG_FUNCPTR (gst_nvdsudpsink_render);
  base_sink_class->render_list = GST_DEBUG_FUNCPTR (gst_nvdsudpsink_render_list);
  base_sink_class->set_caps = GST_DEBUG_FUNCPTR (gst_nvdsudpsink_set_caps);

  GST_DEBUG_CATEGORY_INIT (gst_nvdsudpsink_debug_category, "nvdsudpsink", 0,
      "debug category for nvdsudpsink element");
}

static void
gst_nvdsudpsink_init (GstNvDsUdpSink *sink)
{
  sink->localIfaceIp = g_strdup (g_getenv ("LOCAL_IFACE_IP"));
  sink->nChunks = DEFAULT_CHUNK_SIZE;
  sink->packetsPerChunk = DEFAULT_PACKETS;
  sink->payloadSize = DEFAULT_PAYLOAD_SIZE;
  sink->host = g_strdup (UDP_DEFAULT_HOST);
  sink->port = UDP_DEFAULT_PORT;
  sink->uri = g_strdup_printf ("udp://%s:%d", sink->host, sink->port);
  sink->socket = DEFAULT_SOCKET;
  sink->close_socket = DEFAULT_CLOSE_SOCKET;
  sink->auto_multicast = DEFAULT_AUTO_MULTICAST;
  sink->loop = DEFAULT_LOOP;
  sink->sdpFile = DEFAULT_SDP_FILE;
  sink->isGenericApi = TRUE;
  sink->internalThreadCore = -1;
  sink->nextChunk = 0;
  sink->adapter = NULL;
  sink->ptpSrc = NULL;
  sink->renderThreadCore = NULL;
  sink->lastError = 0;
  sink->rThread = NULL;
  sink->isRtpStream = TRUE;
  sink->packetsPerLine = DEFAULT_PACKETS_PER_LINE;
  sink->pass_rtp_timestamp = FALSE;
  sink->rtp_timestamp_offset = 0;
  sink->gpu_id = GPU_ID_INVALID;
  sink->is_nvmm = false;
  sink->memblock = NULL;
  sink->header_block = NULL;
  sink->cuda_stream = NULL;
  for (guint i = 0; i < MAX_ST2022_7_STREAMS; i++) {
    sink->source_ips[i] = NULL;
    sink->mKey[i] = RMX_MKEY_INVALID;
    sink->hKey[i] = RMX_MKEY_INVALID;
  }
  sink->streamId = INVALID_STREAM_ID;
  sink->num_streams = 1;
  sink->is_dup = FALSE;
}

void
gst_nvdsudpsink_set_property (GObject * object, guint property_id,
    const GValue * value, GParamSpec * pspec)
{
  GstNvDsUdpSink *sink = GST_NVDSUDPSINK (object);

  GST_DEBUG_OBJECT (sink, "set_property");

  switch (property_id) {
    case PROP_AUTO_MULTICAST:
      sink->auto_multicast = g_value_get_boolean (value);
      break;
    case PROP_CLOSE_SOCKET:
      sink->close_socket = g_value_get_boolean (value);
      break;
    case PROP_SOCKET:
      if (sink->socket != NULL && sink->close_socket) {
        GError *err = NULL;

        if (!g_socket_close (sink->socket, &err)) {
          GST_ERROR ("failed to close socket %p: %s", sink->socket,
              err->message);
          g_clear_error (&err);
        }
      }
      if (sink->socket)
        g_object_unref (sink->socket);
      sink->socket = g_value_dup_object (value);
      break;
    case PROP_LOCAL_IFACE_IP:
      g_free (sink->localIfaceIp);
      sink->localIfaceIp = g_value_dup_string (value);
      g_strstrip (sink->localIfaceIp);
      if (!g_strcmp0 (sink->localIfaceIp, "")) {
        g_free (sink->localIfaceIp);
        sink->localIfaceIp = NULL;
      }
      break;
    case PROP_HOST:
    {
      const gchar *host;
      host = g_value_get_string (value);
      g_free (sink->host);
      sink->host = g_strdup (host);
      g_free (sink->uri);
      sink->uri =
          g_strdup_printf ("udp://%s:%d", sink->host, sink->port);
      break;
    }
    case PROP_PORT:
      sink->port = g_value_get_int (value);
      g_free (sink->uri);
      sink->uri =
          g_strdup_printf ("udp://%s:%d", sink->host, sink->port);
      break;
    case PROP_PAYLOAD_SIZE:
      sink->payloadSize = g_value_get_uint (value);
      break;
    case PROP_PACKET_PER_CHUNK:
      sink->packetsPerChunk = g_value_get_uint (value);
      break;
    case PROP_PACKET_PER_LINE:
      sink->packetsPerLine = g_value_get_uint (value);
      break;
    case PROP_CHUNK_SIZE:
      sink->nChunks = g_value_get_uint (value);
      break;
    case PROP_LOOP:
      sink->loop = g_value_get_boolean (value);
      break;
    case PROP_SDP_FILE:
      g_free (sink->sdpFile);
      sink->sdpFile = g_value_dup_string (value);
      g_strstrip (sink->sdpFile);
      if (!g_strcmp0 (sink->sdpFile, "")) {
        g_free (sink->sdpFile);
        sink->sdpFile = NULL;
      }
      break;
    case PROP_INTERNAL_THREAD_CORE:
      sink->internalThreadCore = g_value_get_int (value);
      break;
    case PROP_PTP_SOURCE:
      g_free (sink->ptpSrc);
      sink->ptpSrc = g_value_dup_string (value);
      break;
    case PROP_RENDER_THREAD_CORE:
      g_free (sink->renderThreadCore);
      sink->renderThreadCore = g_value_dup_string (value);
      break;
    case PROP_PASS_RTP_TIMESTAMP:
      sink->pass_rtp_timestamp = g_value_get_boolean (value);
      break;
    case PROP_RTP_TIMESTAMP_OFFSET:
      sink->rtp_timestamp_offset = g_value_get_uint64 (value);
      break;
    case PROP_GPU_ID:
      sink->gpu_id = g_value_get_int (value);
      if (sink->gpu_id >= 0)
        sink->isGpuDirect = TRUE;
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, property_id, pspec);
      break;
  }
}

void
gst_nvdsudpsink_get_property (GObject * object, guint property_id,
    GValue * value, GParamSpec * pspec)
{
  GstNvDsUdpSink *sink = GST_NVDSUDPSINK (object);

  GST_DEBUG_OBJECT (sink, "get_property");

  switch (property_id) {
    case PROP_AUTO_MULTICAST:
      g_value_set_boolean (value, sink->auto_multicast);
      break;
    case PROP_CLOSE_SOCKET:
      g_value_set_boolean (value, sink->close_socket);
      break;
    case PROP_SOCKET:
      g_value_set_object (value, sink->socket);
      break;
    case PROP_LOCAL_IFACE_IP:
      g_value_set_string (value, sink->localIfaceIp);
      break;
    case PROP_HOST:
      g_value_set_string (value, sink->host);
      break;
    case PROP_PORT:
      g_value_set_int (value, sink->port);
      break;
    case PROP_PAYLOAD_SIZE:
      g_value_set_uint (value, sink->payloadSize);
      break;
    case PROP_PACKET_PER_CHUNK:
      g_value_set_uint (value, sink->packetsPerChunk);
      break;
    case PROP_PACKET_PER_LINE:
      g_value_set_uint (value, sink->packetsPerLine);
      break;
    case PROP_CHUNK_SIZE:
      g_value_set_uint (value, sink->nChunks);
      break;
    case PROP_LOOP:
      g_value_set_boolean (value, sink->loop);
      break;
    case PROP_SDP_FILE:
      g_value_set_string (value, sink->sdpFile);
      break;
    case PROP_INTERNAL_THREAD_CORE:
      g_value_set_int (value, sink->internalThreadCore);
      break;
    case PROP_PTP_SOURCE:
      g_value_set_string (value, sink->ptpSrc);
      break;
    case PROP_RENDER_THREAD_CORE:
      g_value_set_string (value, sink->renderThreadCore);
      break;
    case PROP_PASS_RTP_TIMESTAMP:
      g_value_set_boolean (value, sink->pass_rtp_timestamp);
      break;
    case PROP_RTP_TIMESTAMP_OFFSET:
      g_value_set_uint64 (value, sink->rtp_timestamp_offset);
      break;
    case PROP_GPU_ID:
      g_value_set_int (value, sink->gpu_id);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, property_id, pspec);
      break;
  }
}

void
gst_nvdsudpsink_finalize (GObject * object)
{
  GstNvDsUdpSink *sink = GST_NVDSUDPSINK (object);

  GST_DEBUG_OBJECT (sink, "finalize");

  g_free (sink->localIfaceIp);
  sink->localIfaceIp = NULL;

  if (sink->socket)
    g_object_unref (sink->socket);
  sink->socket = NULL;

  g_free (sink->uri);
  sink->uri = NULL;

  g_free (sink->host);
  sink->host = NULL;

  g_free (sink->ptpSrc);
  sink->ptpSrc = NULL;

  if (sink->adapter) {
    g_object_unref (sink->adapter);
    sink->adapter = NULL;
  }

  G_OBJECT_CLASS (gst_nvdsudpsink_parent_class)->finalize (object);
}

static gboolean gst_nvdsudpsink_set_caps (GstBaseSink * bsink, GstCaps * caps)
{
  GstNvDsUdpSink *sink = GST_NVDSUDPSINK (bsink);

  GstStructure* structure = gst_caps_get_structure (caps, 0);
  const gchar* mimeType = gst_structure_get_name (structure);

  if (!g_strcmp0 (mimeType, "video/x-raw") ||
      !g_strcmp0 (mimeType, "audio/x-raw")) {

    GstCapsFeatures *inFeature = gst_caps_features_new ("memory:NVMM", NULL);
    if (gst_caps_features_is_equal (gst_caps_get_features (caps, 0), inFeature)) {
      if (sink->gpu_id == GPU_ID_INVALID) {
        GST_ERROR_OBJECT (sink, "Received NVMM memory but gpu-id is not set");
        gst_caps_features_free (inFeature);
        return FALSE;
      }
      sink->is_nvmm = true;
    }
    gst_caps_features_free (inFeature);

    if (!g_strcmp0 (mimeType, "video/x-raw")) {
      sink->streamParams.streamType = VIDEO_2110_20_STREAM;
    } else if (!g_strcmp0 (mimeType, "audio/x-raw")) {
      sink->streamParams.streamType = AUDIO_2110_30_31_STREAM;
      if (sink->adapter) {
        gst_adapter_clear (sink->adapter);
      } else {
        sink->adapter = gst_adapter_new ();
      }
    }
    sink->isRtpStream = FALSE;
  }
  return TRUE;
}

static uint16_t get_cache_line_size (void)
{
  uint16_t size = (uint16_t) sysconf (_SC_LEVEL1_DCACHE_LINESIZE);

  return size;
}

/**
 * @brief Gets the current TAI time in nanoseconds
 *
 * If ptpSrc is set, get the time from the PTP source
 * otherwise, get the time from the system
 *
 * @param sink The sink instance
 * @return The current TAI time in nanoseconds
 */
static uint64_t get_tai_time_ns (GstNvDsUdpSink *sink)
{
  if (sink->ptpSrc) {
    uint64_t time = 0;
    if (RMAX_OK != rmax_get_time (RMAX_CLOCK_PTP, &time)) {
      GST_ERROR_OBJECT (sink, "Failed to retrieve Rivermax time");
    }
    return time;
  } else {
    struct timespec ts;
    clock_gettime (CLOCK_REALTIME, &ts);
    return (uint64_t) ((ts.tv_sec + LEAP_SECONDS) * GST_SECOND + ts.tv_nsec);
  }
}

/**
 * @brief Aligns a timestamp to Rivermax time
 *
 * Basically, it subtracts the leap seconds from the timestamp.
 * @param time The timestamp to align
 * @return The aligned timestamp
 */
static uint64_t align_to_rmax_time (uint64_t time)
{
    return time - LEAP_SECONDS * GST_SECOND;
}

/**
 * @brief Calculates the first packet time for the stream
 *
 * It calculates first packet timestamp based on expected next frame time as per reference clock.
 *
 * @param sink The sink instance
 */
static void calculate_first_packet_time (GstNvDsUdpSink *sink)
{
  StreamParams *sParams = &sink->streamParams;
  double time_ns = get_tai_time_ns (sink);
  GST_DEBUG_OBJECT(sink, "tai time in ns %" GST_TIME_FORMAT,
                      GST_TIME_ARGS((GstClockTime)(time_ns)));
  time_ns += GST_SECOND;
  double t_frame_ns = sParams->frameTimeInterval;
  if (sParams->videoType != PROGRESSIVE) {
    t_frame_ns *= 2;
  }

  uint64_t N = (uint64_t)(time_ns / t_frame_ns + 1);
  double first_packet_start_time_ns = N * t_frame_ns;
  uint32_t packets_in_frame = sParams->packetsPerFrame;
  if (sParams->videoType != PROGRESSIVE) {
    packets_in_frame *= 2;
  }

  if (VIDEO_2110_20_STREAM == sParams->streamType) {
    double r_active;
    double tro_default_multiplier;
    if (sParams->videoType == PROGRESSIVE) {
      r_active = (1080.0 / 1125.0);
      if (sParams->height >= FHD_HEIGHT) { // As defined by SMPTE 2110-21 6.3.2
          tro_default_multiplier = (43.0 / 1125.0);
      } else {
          tro_default_multiplier = (28.0 / 750.0);
      }
    } else {
      if (sParams->height >= FHD_HEIGHT) { // As defined by SMPTE 2110-21 6.3.3
          r_active = (1080.0 / 1125.0);
          tro_default_multiplier = (22.0 / 1125.0);
      } else if (sParams->height >= 576) {
          r_active = (576.0 / 625.0);
          tro_default_multiplier = (26.0 / 625.0);
      } else {
          r_active = (487.0 / 525.0);
          tro_default_multiplier = (20.0 / 525.0);
      }
    }
    uint16_t video_tro_default_modification = 4;
    double trs_ns = (t_frame_ns * r_active) / packets_in_frame;
    double tro = (tro_default_multiplier * t_frame_ns) - (video_tro_default_modification * trs_ns);
    first_packet_start_time_ns += tro;
  }
  sParams->firstPacketTime = first_packet_start_time_ns;
}

static gboolean
get_video_params_from_sdp_caps (GstCaps *srcCaps, GstNvDsUdpSink *sink)
{
  g_return_val_if_fail (srcCaps != NULL, FALSE);

  const gchar *str;
  StreamParams *sParams = &sink->streamParams;

  GstStructure *structure = gst_caps_get_structure (srcCaps, 0);

  if (!(str = gst_structure_get_string (structure, "width"))) {
    GST_ERROR ("No width in sdp message");
    return FALSE;
  }

  sParams->width = atoi (str);

  if (!(str = gst_structure_get_string(structure, "height"))) {
    GST_ERROR ("No height in sdp message");
    return FALSE;
  }
  sParams->height = atoi(str);

  if (!(str = gst_structure_get_string(structure, "depth"))) {
    GST_ERROR ("No depth in sdp message");
    return FALSE;
  }
  sParams->depth = atoi (str);

  if (!(str = gst_structure_get_string (structure, "sampling"))) {
    GST_ERROR ("No sampling in sdp message");
    return FALSE;
  }

  sParams->format = g_strdup (str);

  if (gst_structure_get_string (structure, "interlace")) {
    sParams->videoType = INTERLACE;
  }

  if (!(str = gst_structure_get_string (structure, "exactframerate"))) {
    GST_ERROR ("No exactframerate media attribute in sdp message");
    return FALSE;
  }

  if (g_strrstr (str, "/")) {
    gint num, den;
    if (sscanf (str, "%d/%d", &num, &den) >= 2) {
      if (den == 0)
        return FALSE;

      sParams->fps = (double) num / den;
    } else {
      GST_ERROR ("can't parse exactframerate media attribute");
      return FALSE;
    }
  } else {
    sParams->fps = g_ascii_strtod (str, NULL);
  }

  if (sParams->width == 0 || sParams->height == 0 || sParams->fps == 0) {
    GST_ERROR ("wrong value for width (%u), height (%u) or fps (%f)",
        sParams->width, sParams->height, sParams->fps);
    return FALSE;
  }

  return TRUE;
}

static gboolean
get_audio_params_from_sdp_caps (GstCaps *srcCaps, GstNvDsUdpSink *sink)
{
  g_return_val_if_fail (srcCaps != NULL, FALSE);

  gint channels, tmp;
  const gchar *str;
  StreamParams *sParams = &sink->streamParams;

  GstStructure *structure = gst_caps_get_structure (srcCaps, 0);

  if ((str = gst_structure_get_string (structure, "encoding-params"))) {
    channels = atoi (str);
  } else if (gst_structure_get_int (structure, "channels", &tmp)) {
    channels = tmp;
  } else {
    GST_ERROR ("No channels details in sdp message");
    return FALSE;
  }
  sParams->audioChannels = channels;

  if (!(str = gst_structure_get_string(structure, "encoding-name"))) {
    GST_ERROR ("No encoding-name in sdp message");
    return FALSE;
  }

  if (!g_strcmp0 (str, "AM824")) {
    sParams->depth = 32;
  } else {
    sParams->depth = atoi (str+1);
  }

  return TRUE;
}

/**
 * Extracts source-filter IP addresses from an SDP file
 *
 * @param sdp_content The SDP file content as a string
 * @param source_ips Array to store the extracted IP addresses (caller must free each string)
 * @param max_ips Maximum number of IPs to extract
 *
 * @return Number of IPs found and stored in the array
 */
static guint parse_sdp_source_filter_ips(const char *sdp_content, char **source_ips, guint max_ips) {
    const char *source_filter_tag = "a=source-filter:";
    const char *pos = sdp_content;
    guint count = 0;

    while (count < max_ips && (pos = strstr(pos, source_filter_tag)) != NULL) {
        // Move position to start of the source-filter line
        pos += strlen(source_filter_tag);

        // Find end of line
        const char *eol = strpbrk(pos, "\r\n");
        if (!eol) {
            eol = pos + strlen(pos); // End of string if no newline found
        }

        // Copy the line for tokenization
        int line_len = eol - pos;
        char *line = g_malloc(line_len + 1);
        memcpy(line, pos, line_len);
        line[line_len] = '\0';
        // Tokenize to extract IP address (last token)
        char *last_token = NULL;
        char *token = strtok(line, " ");
        while (token != NULL) {
            last_token = token;
            token = strtok(NULL, " ");
        }

        // Store the IP if found
        if (last_token) {
            source_ips[count] = g_strdup(last_token);
            count++;
        }

        g_free(line);
        pos = eol; // Move to end of current line for next iteration
    }

    return count;
}

static gboolean
parse_sdp_file (GstNvDsUdpSink *sink)
{
  gchar *sdpTxt = NULL;
  GstSDPResult result;
  GstSDPMessage *sdpMsg;
  gboolean ret = FALSE;
  StreamParams *sParams = &sink->streamParams;

  if (!g_file_get_contents (sink->sdpFile, &sdpTxt, NULL, NULL)) {
    GST_ERROR_OBJECT (sink, "Error in reading contents of sdp file - %s",
        sink->sdpFile);
    return FALSE;
  }

  // Check if SDP file contains the "DUP" marker to enable redundancy
  if (g_strstr_len(sdpTxt, -1, "DUP") != NULL) {
    sink->is_dup = TRUE;
    sink->num_streams = parse_sdp_source_filter_ips(sdpTxt, sink->source_ips, MAX_ST2022_7_STREAMS);
    if (sink->num_streams == 0) {
      GST_ERROR_OBJECT(sink, "No streams found in SDP file");
      g_free (sdpTxt);
      return FALSE;
    } else if ((sink->num_streams > 0) && (sink->num_streams < MAX_ST2022_7_STREAMS)) {
      GST_WARNING_OBJECT(sink, "Streams found in SDP file: [%u] not equal to max streams for ST2022-7: [%u]",
                                sink->num_streams, MAX_ST2022_7_STREAMS);
    }
    /* Safety check to ensure we don't exceed array bounds */
    if (sink->num_streams > MAX_ST2022_7_STREAMS) {
      GST_ERROR_OBJECT(sink, "Too many streams found: %u. Maximum allowed is %d",
                       sink->num_streams, MAX_ST2022_7_STREAMS);
      g_free (sdpTxt);
      return FALSE;
    }
  } else {
    sink->is_dup = FALSE;
    sink->source_ips[0] = sink->localIfaceIp;
  }

  result = gst_sdp_message_new_from_text (sdpTxt, &sdpMsg);
  if (result != GST_SDP_OK) {
    GST_ERROR ("Error (%d) in creating sdp message.", result);
    g_free (sdpTxt);
    return FALSE;
  }

  const GstSDPMedia *media = gst_sdp_message_get_media (sdpMsg, 0);
  if (!media) {
    GST_ERROR ("Error!! No media in sdp message");
    goto error;
  }

  gint pt = atoi (gst_sdp_media_get_format (media, 0));
  if (pt < 96 || pt > 127) {
    GST_ERROR ("Wrong value (%d) of payload type. It should be in range of 96-127", pt);
    goto error;
  }

  GstCaps *caps = gst_sdp_media_get_caps_from_media (media, pt);
  gint rate, tmp;
  const gchar *str;

  GstStructure *structure = gst_caps_get_structure (caps, 0);

  if ((str = gst_structure_get_string (structure, "clock-rate"))) {
    rate = atoi(str);
  } else if (gst_structure_get_int (structure, "clock-rate", &tmp)) {
    rate = tmp;
  } else {
    GST_ERROR ("No clock-rate in sdp message");
    goto error;
  }
  sParams->sampleRate = rate;
  sParams->payloadType = pt;

  if (!g_strcmp0 (media->media, "video")) {
    sParams->streamType = VIDEO_2110_20_STREAM;
    sParams->videoType = PROGRESSIVE;
    if (!get_video_params_from_sdp_caps (caps, sink))
      goto error;
  } else if (!g_strcmp0 (media->media, "audio")) {
    sParams->streamType = AUDIO_2110_30_31_STREAM;
    const gchar *ptime = gst_sdp_media_get_attribute_val (media, "ptime");
    if (!ptime) {
      GST_ERROR ("No attribute ptime in sdp message");
      goto error;
    }

    gdouble tmpPtime = atof (ptime);
    if (tmpPtime <= 0) {
      GST_ERROR ("wrong value of ptime - %s", ptime);
      goto error;
    }
    sParams->ptime = (guint64) (tmpPtime * 1000000 + 0.5); // convert from milli to nano seconds

    if (!get_audio_params_from_sdp_caps (caps, sink))
      goto error;
  } else {
    GST_ERROR ("media %s not supported", media->media);
    goto error;
  }

  ret = TRUE;

error:
  gst_sdp_message_free (sdpMsg);
  g_free (sdpTxt);
  return ret;
}

/**
 * @brief Initializes a Rivermax output stream
 *
 * This function initializes the Rivermax output stream by:
 * 1. Setting up stream parameters
 * 2. Parsing the SDP file to get stream configuration
 * 3. Configuring stream-specific parameters based on stream type (video/audio)
 * 4. Setting up frame timing and packet sizes
 *
 * @param sink The GstNvDsUdpSink instance
 * @return TRUE if initialization was successful, FALSE otherwise
 */
static gboolean
initialize_rivermax_out_stream (GstNvDsUdpSink *sink)
{
  rmax_status_t status;
  struct rmax_buffer_attr buffer_attr;
  struct rmax_mem_block *block = NULL;
  struct rmax_out_stream_params params;
  gboolean ret;
  StreamParams *sParams = &sink->streamParams;
  uint16_t *payload_sizes = NULL;
  uint16_t *header_sizes = NULL;
  gchar *sdpTxt = NULL;

  memset(&buffer_attr, 0, sizeof(buffer_attr));
  memset(&params, 0, sizeof(params));
  memset(sParams, 0, sizeof(StreamParams));

  sParams->videoType = PROGRESSIVE;
  sParams->payloadType = DEFAULT_PAYLOAD_TYPE;
  sParams->ssrc = g_random_int ();
  sParams->seq = g_random_int ();

  ret = parse_sdp_file (sink);
  if (!ret) {
    GST_ERROR ("Failed to parse sdp file - %s", sink->sdpFile);
    return ret;
  }

  switch (sParams->streamType) {
    case VIDEO_2110_20_STREAM: {
      int lines_in_chunk = 4;
      sParams->packetsPerFrame = sink->packetsPerLine * sParams->height;
      sParams->chunkSize = lines_in_chunk * sink->packetsPerLine;

      if ((sParams->packetsPerFrame % sParams->chunkSize)) {
        GST_ERROR ("packets per frame(%u) must be multiple of chunk size(%u)",
          sParams->packetsPerFrame, sParams->chunkSize);
        return FALSE;
      }
      sParams->frameTimeInterval = 1000000000.0 / sParams->fps;
      if (sParams->videoType != PROGRESSIVE) {
        sParams->packetsPerFrame /= 2;
        sParams->frameTimeInterval /= 2;
      }
    }
      break;
    case AUDIO_2110_30_31_STREAM:
      sParams->packetsPerFrame = 1000000000 / sParams->ptime;
      // Due to limitation of rtp payloader component which provides one packet at a time.
      sParams->chunkSize = 1;
      int samples_in_packet = sParams->sampleRate / sParams->packetsPerFrame;
      sParams->frameTimeInterval = sParams->ptime * sParams->packetsPerFrame;
      sink->payloadSize = ((sParams->depth * sParams->audioChannels *
                                               samples_in_packet) / 8) + RTP_HEADER_SIZE;
      break;
    default:
      GST_ERROR ("stream type not supported");
      return FALSE;
  }

  uint16_t data_stride_size = sink->isGpuDirect ? (sink->payloadSize - RTP_2110_20_MIN_HEADER_SIZE) : river_align_up_pow2 (sink->payloadSize, get_cache_line_size());
  sParams->framesPerMemblock = DEFAULT_FRAMES_PER_BLOCK;
  if (sParams->videoType != PROGRESSIVE) {
    sParams->framesPerMemblock *= 2;
  }
  sParams->headerStride = 0;
  sParams->chunksPerFrame = sParams->packetsPerFrame / sParams->chunkSize;
  sParams->chunksPerMemblock = sParams->framesPerMemblock * sParams->chunksPerFrame;
  sParams->payloadStride = data_stride_size;

  uint32_t packets_in_mem_block = sParams->packetsPerFrame * sParams->framesPerMemblock;
  block = g_new0 (struct rmax_mem_block, 1);

  if (sink->isGpuDirect) {
    cudaError_t ret = cudaSuccess;
    struct in_addr inAddr[MAX_ST2022_7_STREAMS];

    for (guint i = 0; i < MAX_ST2022_7_STREAMS; i++) {
      memset(&inAddr[i], 0, sizeof(inAddr[i]));
    }

    ret = cudaStreamCreate(&sink->cuda_stream);
    CHECK_CUDA (ret, "failed to create cuda stream");

    size_t memSize = packets_in_mem_block * data_stride_size;
    struct cudaDeviceProp props;
    ret = cudaGetDeviceProperties(&props, sink->gpu_id);
    CHECK_CUDA (ret, "failed to get device properties.");

    if (props.integrated) {
      sink->alignedMemSize = memSize;
    } else {
      sink->alignedMemSize = gpu_aligned_size(sink->gpu_id, memSize);
    }
    sink->memblock = gpu_allocate_memory(sink->gpu_id, sink->alignedMemSize, 0);
    if (!sink->memblock) {
      GST_ERROR_OBJECT(sink, "Data host memory allocation failed");
      goto error;
    }

    // Register data memory
    for (guint i = 0; i < sink->num_streams; i++) {
      inet_aton(sink->source_ips[i], &inAddr[i]);
      status = rmax_register_memory(sink->memblock, sink->alignedMemSize, inAddr[i], &(sink->mKey[i]));
      if (status != RMAX_OK) {
        GST_ERROR_OBJECT (sink, "Error in registering data memory for IP: %s, status = %d\n", sink->source_ips[i], status);
        goto error;
      }
    }
    buffer_attr.attr_flags = RMAX_OUT_BUFFER_ATTR_DATA_MKEY_IS_SET;

    sParams->headerStride = river_align_up_pow2(RTP_2110_20_MIN_HEADER_SIZE, get_cache_line_size());
    size_t headerMemSize = packets_in_mem_block * sParams->headerStride;
    sink->header_block = g_malloc0 (headerMemSize);
    if (!sink->header_block) {
      GST_ERROR_OBJECT(sink, "Header host memory allocation failed");
      goto error;
    }

    // Register header memory
    for (guint i = 0; i < sink->num_streams; i++) {
      status = rmax_register_memory(sink->header_block, headerMemSize, inAddr[i], &(sink->hKey[i]));
      if (status != RMAX_OK) {
        GST_ERROR_OBJECT (sink, "Error in registering header memory for IP: %s, status = %d\n", sink->source_ips[i], status);
        goto error;
      }
    }
    buffer_attr.attr_flags = (rmax_out_buffer_attr_flags)(buffer_attr.attr_flags | RMAX_OUT_BUFFER_ATTR_APP_HDR_MKEY_IS_SET);

    header_sizes = g_new0 (uint16_t, packets_in_mem_block);
    for (uint32_t i = 0; i < packets_in_mem_block; i++) {
      header_sizes[i] = RTP_2110_20_MIN_HEADER_SIZE;
    }

    block->app_hdr_ptr= (void*)sink->header_block;
    block->data_ptr = (void*)sink->memblock;

    for (guint i = 0; i < sink->num_streams; i++) {
      block->data_mkey[i] = sink->mKey[i];
      block->app_hdr_mkey[i] = sink->hKey[i];
    }
  }

  payload_sizes = g_new0 (uint16_t, packets_in_mem_block);
  for (uint32_t i = 0; i < packets_in_mem_block; i++) {
    payload_sizes[i] = sink->isGpuDirect ? (sink->payloadSize - RTP_2110_20_MIN_HEADER_SIZE) : sink->payloadSize;
  }
  block->chunks_num = sParams->chunksPerMemblock;
  block->app_hdr_size_arr = header_sizes;
  block->data_size_arr = payload_sizes;

  buffer_attr.chunk_size_in_strides = sParams->chunkSize;
  buffer_attr.mem_block_array = block;
  buffer_attr.mem_block_array_len = 1;
  buffer_attr.data_stride_size = sParams->payloadStride;
  buffer_attr.app_hdr_stride_size = sParams->headerStride;
  struct rmax_qos_attr q = { 0, 0 };

  if (!g_file_get_contents (sink->sdpFile, &sdpTxt, NULL, NULL)) {
    GST_ERROR_OBJECT(sink, "Error in reading contents of sdp file - %s",
                     sink->sdpFile);
    goto error;
  }


  params.sdp_chr = sdpTxt;
  params.buffer_attr = &buffer_attr;
  params.qos = &q;
  params.num_packets_per_frame = sParams->packetsPerFrame;
  params.media_block_index = 0;
  params.source_port_arr = g_new0(uint16_t, sink->num_streams);
  params.source_port_arr_sz = sink->num_streams;

  // Setup source ports for primary and redundant streams (if is_dup is true)
  if (sink->is_dup) {
    if (sink->port != UDP_DEFAULT_PORT) {
      for (guint i = 0; i < sink->num_streams; i++) {
        params.source_port_arr[i] = sink->port + i;
      }
    }
  }

  if (sParams->videoType != PROGRESSIVE)
    params.num_packets_per_frame *= 2;

  status = rmax_out_create_stream_ex (&params, &sink->streamId);
  if (status != RMAX_OK) {
    g_print("Failed in creating stream, status: %d\n", status);
    goto error;
  }

  // Validate the source and destination IP addresses registered with Rivermax
  if (sink->is_dup) {
    struct sockaddr_in source_address;
    struct sockaddr_in destination_address;
    memset(&source_address, 0, sizeof(source_address));
    memset(&destination_address, 0, sizeof(destination_address));

    for (guint j = 0; j < sink->num_streams; j++) {
      status = rmax_out_query_address(sink->streamId, j, &source_address, &destination_address);
      if (status != RMAX_OK) {
        GST_ERROR_OBJECT(sink, "Failed to query stream source port for stream %d", j);
        goto error;
      }

      char src_ip[INET_ADDRSTRLEN];
      char dst_ip[INET_ADDRSTRLEN];
      inet_ntop(AF_INET, &(source_address.sin_addr), src_ip, INET_ADDRSTRLEN);
      inet_ntop(AF_INET, &(destination_address.sin_addr), dst_ip, INET_ADDRSTRLEN);
      uint16_t src_port = ntohs(source_address.sin_port);
      uint16_t dst_port = ntohs(destination_address.sin_port);

      GST_INFO_OBJECT(sink, "Stream %d - Source: %s:%d, Destination: %s:%d",
                     j, src_ip, src_port, dst_ip, dst_port);
    }
  }

  g_free(payload_sizes);
  g_free(header_sizes);
  g_free(sdpTxt);
  g_free(block);

  return TRUE;

error:
  if (sink->streamId != INVALID_STREAM_ID) {
    rmax_out_destroy_stream(sink->streamId);
    sink->streamId = INVALID_STREAM_ID;
  }

  if (sink->memblock) {
    sink->isGpuDirect ? cudaFreeMmap ((uint64_t *)&sink->memblock, sink->alignedMemSize) : g_free(sink->memblock);
    sink->memblock = NULL;
  }

  if (sink->header_block) {
    g_free(sink->header_block);
    sink->header_block = NULL;
  }

  if (payload_sizes) {
    g_free(payload_sizes);
    payload_sizes = NULL;
  }

  if (header_sizes) {
    g_free(header_sizes);
    header_sizes = NULL;
  }

  if (sdpTxt) {
    g_free(sdpTxt);
    sdpTxt = NULL;
  }

  if (block) {
    g_free(block);
    block = NULL;
  }

  if (params.source_port_arr) {
    g_free(params.source_port_arr);
    params.source_port_arr = NULL;
  }

  return FALSE;
}

/**
 * @brief Initializes a Rivermax generic output stream
 *
 * This function sets up a Rivermax generic output stream using the Rivermax generic APIs.
 * It configures the local and remote addresses, creates the stream, allocates memory for packets,
 * and registers the memory with Rivermax.
 *
 * @param sink The GstNvDsUdpSink instance containing stream configuration
 * @return TRUE if stream initialization was successful, FALSE otherwise
 */
static gboolean
initialize_rivermax_generic_stream (GstNvDsUdpSink *sink)
{
  rmax_status_t status;
  struct rmax_out_gen_stream_params params;
  struct sockaddr_in remoteAddr;
  struct in_addr inAddr;
  rmax_mkey_id mKey = RMX_MKEY_INVALID;
  guint i, j;
  guint memSize;
  guint memOffset = 0;

  memset(&sink->localNicAddr, 0, sizeof(sink->localNicAddr));
  sink->localNicAddr.sin_family = AF_INET;
  sink->localNicAddr.sin_addr.s_addr = inet_addr(sink->localIfaceIp);

  memset (&remoteAddr, 0, sizeof(remoteAddr));
  remoteAddr.sin_family = AF_INET;
  remoteAddr.sin_port = htons ((uint16_t) sink->port);

  if (!g_strcmp0 (sink->host, "0.0.0.0"))
    remoteAddr.sin_addr.s_addr = inet_addr (sink->localIfaceIp);
  else {
    GInetAddress *addr = gst_udp_resolve_name (sink, sink->host);
    if (!addr) {
      GST_ERROR_OBJECT (sink, "Failed to resolve %s", sink->host);
      return FALSE;
    }
    remoteAddr.sin_addr.s_addr = *(in_addr_t *) g_inet_address_to_bytes (addr);
    g_object_unref (addr);
  }

  memset(&params, 0, sizeof(params));
  params.local_addr = (struct sockaddr *) &sink->localNicAddr;
  params.max_chunk_size = sink->packetsPerChunk;
  params.remote_addr = (struct sockaddr *) &remoteAddr;
  params.flags = RMAX_OUT_STREAM_REM_ADDR;
  params.size_in_chunks = sink->nChunks;

  status = rmax_out_create_gen_stream (&params, &sink->streamId);
  if (status != RMAX_OK) {
    GST_ERROR_OBJECT(sink, "Failed to create output stream - error %d", status);
    rmax_cleanup ();
    return FALSE;
  }

  // Allocate the memory for packets.
  memSize = sink->nChunks * sink->packetsPerChunk * sink->payloadSize;
  sink->memblock = g_malloc0 (memSize);

  inet_aton (sink->localIfaceIp, &inAddr);
  status = rmax_register_memory (sink->memblock, memSize, inAddr, &mKey);
  if (status != RMAX_OK) {
    GST_ERROR("error in registering memory, status = %d\n", status);
    g_free (sink->memblock);
    rmax_out_destroy_stream (sink->streamId);
    rmax_cleanup ();
    return FALSE;
  }

  sink->mKey[0] = mKey;
  sink->chunks = g_new0 (struct rmax_chunk, sink->nChunks);
  for (i = 0; i < sink->nChunks; i++) {
    struct rmax_chunk *chunk = &sink->chunks[i];
    chunk->size = sink->packetsPerChunk;
    chunk->chunk_ctx = NULL;
    chunk->packets = g_new0 (struct rmax_packet, sink->packetsPerChunk);

    for (j = 0; j < sink->packetsPerChunk; j++) {
      struct rmax_packet *packet = &chunk->packets[j];
      packet->iovec = g_new0 (struct rmax_iov, 1);;
      packet->count = 1;

      packet->iovec->addr = (uint64_t) sink->memblock + memOffset;
      packet->iovec->length = sink->payloadSize;
      packet->iovec->mid = mKey;

      memOffset += sink->payloadSize;
    }
  }
  return TRUE;
}

/**
 * @brief Sets thread affinity for a thread
 * @param coreList Comma-separated list of CPU cores
 * @return TRUE if affinity was set successfully, FALSE otherwise
 */
static gboolean
set_thread_affinity (gchar *coreList)
{
  g_return_val_if_fail (coreList != NULL, FALSE);

  int ret = 0;
  cpu_set_t *cpu_set;
  size_t cpu_alloc_size;
  cpu_set = CPU_ALLOC (RMAX_CPU_SETSIZE);
  if (!cpu_set) {
    g_print ("failed to allocate cpu_set\n");
    return FALSE;
  }
  cpu_alloc_size = CPU_ALLOC_SIZE (RMAX_CPU_SETSIZE);
  CPU_ZERO_S (cpu_alloc_size, cpu_set);

  gchar **tokens = g_strsplit (g_strstrip (coreList), ",", 0);
  gchar **tmp = tokens;
  gint cpuCore = 0;
  while (*tmp) {
    cpuCore = atoi (*tmp);
    if (cpuCore >= 0 && cpuCore < RMAX_CPU_SETSIZE) {
      CPU_SET_S (cpuCore, cpu_alloc_size, cpu_set);
    }
    tmp++;
  }
  g_strfreev (tokens);

  pthread_t thread_handle = pthread_self ();
  if (CPU_COUNT (cpu_set)) {
    ret = pthread_setaffinity_np (thread_handle, cpu_alloc_size, cpu_set);
    if (ret) {
      g_print ("failed to set thread affinity, errno: %d\n", ret);
    }
  }
  CPU_FREE (cpu_set);
  return ret ? FALSE : TRUE;
}

static gboolean
gst_nvdsudpsink_start (GstBaseSink * bsink)
{
  rmax_status_t status;
  gboolean ret;
  struct rmax_init_config initConfig;
  GstNvDsUdpSink *sink = GST_NVDSUDPSINK (bsink);

  GST_DEBUG_OBJECT (sink, "start");
  if (!sink->localIfaceIp) {
    GST_ERROR_OBJECT (sink, "NULL IP address of local interface.");
    return FALSE;
  }

  if (sink->sdpFile)
    sink->isGenericApi = FALSE;

  if (sink->isGpuDirect && (sink->streamParams.streamType != VIDEO_2110_20_STREAM)) {
    GST_ERROR_OBJECT (sink, "GPU-Direct is currently only supported for video stream. Falling back to non GPU direct path");
    sink->isGpuDirect = FALSE;
  }

  memset(&initConfig, 0, sizeof(initConfig));
  initConfig.flags |= RIVERMAX_HANDLE_SIGNAL;

  if (sink->internalThreadCore >= 0) {
    RMAX_CPU_SET (sink->internalThreadCore, &initConfig.cpu_mask);
    initConfig.flags |= RIVERMAX_CPU_MASK;
  }

  status = rmax_init (&initConfig);
  if (status != RMAX_OK) {
    GST_ERROR_OBJECT (sink, "Failed to initialize Rivermax - error %d", status);
    return FALSE;
  }

  if (!sink->isGenericApi && sink->ptpSrc) {
    int err;
    struct rmax_clock_t clock;
    memset(&clock, 0, sizeof(clock));
    clock.clock_type = RIVERMAX_PTP_CLOCK;

    err = inet_pton (AF_INET, sink->ptpSrc,
                &clock.clock_u.rmax_ptp_clock.device_ip_addr);
    if (!err) {
      GST_ERROR_OBJECT (sink, "Invalid PTP source address (%s)", sink->ptpSrc);
      rmax_cleanup ();
      return FALSE;
    }

    status = rmax_set_clock (&clock);
    GST_DEBUG_OBJECT (sink, "rmax_set_clock(RIVERMAX_PTP_CLOCK) status: %d", status);
    /* If multiple instances are running, the clock return busy while trying to set the clock second time.
      Ignore the busy status in this case. */
    if ((status != RMAX_OK) && (status != RMAX_ERR_BUSY)) {
      GST_WARNING_OBJECT (sink, "Failed to set PTP clock - status %d", status);
      GST_WARNING_OBJECT (sink, "PTP clock is not supported, using SYSTEM clock");

      memset (&clock, 0, sizeof(clock));
      clock.clock_type = RIVERMAX_SYSTEM_CLOCK;

      g_free (sink->ptpSrc);
      sink->ptpSrc = NULL;

      status = rmax_set_clock (&clock);
      GST_DEBUG_OBJECT (sink, "rmax_set_clock(RIVERMAX_SYSTEM_CLOCK) status: %d", status);
      /* If multiple instances are running, the clock return busy while trying to set the clock second time.
      Ignore the busy status in this case. */
      if ((status != RMAX_OK) && (status != RMAX_ERR_BUSY)) {
        GST_ERROR_OBJECT (sink, "Failed to set SYSTEM clock - status %d", status);
        rmax_cleanup ();
        return FALSE;
      }
    }
  }

  if (sink->isGenericApi) {
    ret = initialize_rivermax_generic_stream (sink);
  } else {
    ret = initialize_rivermax_out_stream (sink);
  }

  if (!ret) {
    GST_ERROR_OBJECT (sink, "Failed to initialize stream");
    rmax_cleanup ();
    return ret;
  }

  if (sink->renderThreadCore) {
    g_mutex_init (&sink->qLock);
    g_cond_init (&sink->qCond);
    sink->bufferQ = g_queue_new ();
    sink->isRunning = TRUE;
    sink->rThread = g_thread_new (NULL, render_thread, sink);
  }

  return TRUE;
}

static gboolean
gst_nvdsudpsink_stop (GstBaseSink * bsink)
{
  rmax_status_t status;
  struct in_addr inAddr;
  guint i, j;
  GstNvDsUdpSink *sink = GST_NVDSUDPSINK (bsink);

  GST_DEBUG_OBJECT (sink, "stop");

  if (sink->rThread) {
    sink->isRunning = FALSE;
    g_cond_signal (&sink->qCond);
    g_thread_join (sink->rThread);
    sink->rThread = NULL;

    g_mutex_lock (&sink->qLock);
    g_queue_free_full (sink->bufferQ, (GDestroyNotify) gst_buffer_unref);
    g_mutex_unlock (&sink->qLock);
    g_mutex_clear (&sink->qLock);
    g_cond_clear (&sink->qCond);
  }

  if (sink->streamId != INVALID_STREAM_ID) {
    do {
      status = rmax_out_destroy_stream(sink->streamId);

      if (status == RMAX_ERR_BUSY)
        sleep(1);
    } while (status == RMAX_ERR_BUSY);
  }

  for (guint i = 0; i < sink->num_streams; i++) {
    inet_aton (sink->source_ips[i], &inAddr);
    if (sink->mKey[i] != RMX_MKEY_INVALID) {
      status = rmax_deregister_memory(sink->mKey[i], inAddr);
      if (status != RMAX_OK) {
        GST_ERROR_OBJECT (sink, "Failed to deregister data memory, status = %d\n", status);
      }
      sink->mKey[i] = RMX_MKEY_INVALID;
    }
    if (sink->hKey[i] != RMX_MKEY_INVALID) {
      status = rmax_deregister_memory (sink->hKey[i], inAddr);
      if (status != RMAX_OK) {
        GST_ERROR_OBJECT (sink, "Failed to deregister header memory, status = %d\n", status);
      }
      sink->hKey[i] = RMX_MKEY_INVALID;
    }
  }

  if (sink->isGenericApi) {
    for (i = 0; i < sink->nChunks; i++) {
      struct rmax_chunk *chunk = &sink->chunks[i];
      for (j = 0; j < sink->packetsPerChunk; j++) {
        g_free(chunk->packets[j].iovec);
      }
      g_free(chunk->packets);
    }
    g_free (sink->chunks);
  }

  if (sink->memblock) {
    sink->isGpuDirect ? cudaFreeMmap ((uint64_t *)&sink->memblock, sink->alignedMemSize) : g_free(sink->memblock);
    sink->memblock = NULL;
  }

  if (sink->header_block) {
    g_free(sink->header_block);
    sink->header_block = NULL;
  }

  if (sink->socket != NULL && sink->close_socket) {
    GError *err = NULL;
    if (!g_socket_close (sink->socket, &err)) {
      GST_ERROR("failed to close socket %p: %s", sink->socket,
                err->message);
      g_clear_error(&err);
    }
  }

  if (sink->socket)
    g_object_unref (sink->socket);
  sink->socket = NULL;

  if (sink->cuda_stream) {
    cuStreamDestroy(sink->cuda_stream);
    sink->cuda_stream = NULL;
  }

  rmax_cleanup ();
  return TRUE;
}

/**
 * @brief Renders a buffer to the UDP sink
 *
 * This function handles the rendering of a buffer to the UDP sink. It supports both RTP and raw stream modes.
 * In RTP mode, it wraps the RTP packet in a buffer list and passes it to render_list.
 * In raw mode, it either queues the buffer for processing by the render
 * thread (if separate render thread is enabled) or directly renders the frame.
 *
 * @param bsink The base sink instance
 * @param buffer The buffer to render
 * @return GstFlowReturn indicating the result of the render operation
 */
static GstFlowReturn
gst_nvdsudpsink_render (GstBaseSink * bsink, GstBuffer * buffer)
{
  GstFlowReturn ret;
  GstNvDsUdpSink *sink = GST_NVDSUDPSINK (bsink);

  GST_DEBUG_OBJECT (sink, "render");

  if (sink->lastError) {
    return GST_FLOW_ERROR;
  }

  if (sink->isRtpStream) {
    GstBufferList *bList = gst_buffer_list_new ();
    gst_buffer_ref (buffer);
    gst_buffer_list_add (bList, buffer);
    ret = gst_nvdsudpsink_render_list (bsink, bList);
    gst_buffer_list_unref (bList);
  } else {
    if (sink->rThread) {
      gst_buffer_ref (buffer);
      g_mutex_lock (&sink->qLock);
      // Free flowing pipeline can cause buffer build up.
      // limit to max 5 frames.
      if (g_queue_get_length (sink->bufferQ) >= 5) {
        g_cond_wait (&sink->qCond, &sink->qLock);
      }
      g_queue_push_tail (sink->bufferQ, buffer);
      g_cond_signal(&sink->qCond);
      g_mutex_unlock (&sink->qLock);
      return GST_FLOW_OK;
    }
    ret = gst_nvdsudpsink_render_raw_frame (bsink, buffer);
  }

  return ret;
}

/**
 * @brief Converts a time value in nanoseconds to an RTP timestamp
 *
 * This function converts a time value in nanoseconds to an RTP timestamp value
 * based on the provided sample rate. The timestamp is wrapped to 32 bits and
 * decreased by 1 tick to prevent future timestamps due to calculation imprecision.
 *
 * @param time_ns Time value in nanoseconds
 * @param sample_rate Sample rate in Hz
 * @return RTP timestamp value
 */
static gdouble
time_to_rtp_timestamp (gdouble time_ns, guint sample_rate)
{
    gdouble time_sec = time_ns / (gdouble) GST_SECOND;
    gdouble timestamp = time_sec * (gdouble) sample_rate;
    gdouble mask = 0x100000000;
    // We decrease one tick from the timestamp to prevent cases where the timestamp
    // lands up in the future due to calculation imprecision
    timestamp = fmod(timestamp, mask) - 1;
    return timestamp;
}

/**
 * @brief Builds an RTP packet header
 *
 * This function constructs an RTP header according to RFC 3550 and SMPTE 2110-20/30/31 specifications.
 * For video streams, it also adds the SMPTE 2110-20 payload header with extended sequence number,
 * SRD length, row number, and offset information.
 *
 * @param buf Buffer to store the RTP header
 * @param line Line number in the video frame (for video streams)
 * @param offset Byte offset within the line (for video streams)
 * @param packetNum Current packet number within the frame
 * @param fieldIdx Field index for interlaced video (0 for first field, 1 for second field)
 * @param sink The GstNvDsUdpSink instance containing stream parameters
 */
static void
build_rtp_header (guint8 *buf, guint line, guint offset,
    guint packetNum, guint fieldIdx, GstNvDsUdpSink *sink)
{
  g_return_if_fail (buf != NULL);
  g_return_if_fail (sink != NULL);

  StreamParams *sParams = &sink->streamParams;

  // RTP header - 12 bytes
  /*
  0                   1                   2                   3
  0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
  | V |P|X|  CC   |M|     PT      |            SEQ                |
  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
  |                           timestamp                           |
  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
  |                           ssrc                                |
  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+*/
  memset(buf, 0, RTP_2110_20_MIN_HEADER_SIZE);

  buf[0] = 0x80; // 10000000 - version2, no padding, no extension
  buf[1] = sParams->payloadType;
  *(guint16 *)(buf + 2) = g_htons (sParams->seq);
  *(guint32 *)(buf + 4) = g_htonl ((guint32) sParams->timestampTick);
  *(guint32 *)(buf + 8) = g_htonl ((guint32) sParams->ssrc);

  // Payload Header - 8 bytes
  /*
   0                   1                   2                   3
   0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
   |    Extended Sequence Number   |           SRD Length          |
   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
   |F|     SRD Row Number          |C|         SRD Offset          |
   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+ */

  if (sParams->streamType == VIDEO_2110_20_STREAM) {
    *(guint16 *)(buf + 12) = g_htons (sParams->extSeqNumber);
    *(guint16 *)(buf + 14) = g_htons (sink->payloadSize - RTP_2110_20_MIN_HEADER_SIZE);
    *(guint16 *)(buf + 16) = g_htons (line);
    buf[16] |= ((fieldIdx << 7) & 0x80);
    // C = 0 Always because of no continuation.
    *(guint16 *)(buf + 18) = g_htons (offset);
  }

  if (++packetNum == sParams->packetsPerFrame) {
    // last packet in frame / field, set marker bit
    buf[1] |= 0x80;
  }
}

/**
 * @brief Renders a raw frame to the UDP sink
 *
 * This function handles the rendering of a raw frame to the UDP sink. It calculates
 * the appropriate timing for frame transmission, manages packetization of the frame
 * data, and handles both video and audio stream types. For video streams, it handles
 * progressive and interlaced formats.
 *
 * @param bsink The base sink instance
 * @param buffer The buffer containing the raw frame data to render
 * @return GstFlowReturn indicating the result of the render operation
 */
static GstFlowReturn
gst_nvdsudpsink_render_raw_frame (GstBaseSink *bsink, GstBuffer *buffer)
{
  rmax_status_t status;
  GstMapInfo info = GST_MAP_INFO_INIT;
  GstNvDsUdpSink *sink = GST_NVDSUDPSINK (bsink);
  StreamParams *sParams = &sink->streamParams;

  if (!sParams->firstPacketTime) {
    if (sink->pass_rtp_timestamp) {
      GstClockTime base_time = gst_element_get_base_time(GST_ELEMENT(sink));
      sParams->firstPacketTime = base_time + GST_BUFFER_PTS (buffer);
      GST_DEBUG_OBJECT(sink, "firstPacketTime in HH:MM:SS.NS: %" GST_TIME_FORMAT,
                      GST_TIME_ARGS((GstClockTime)(sParams->firstPacketTime)));

      GstRTPTimestampMeta *meta = gst_buffer_get_rtp_timestamp_meta(buffer);
      if (meta) {
          GST_DEBUG_OBJECT(sink, "Using RTP timestamp from metadata: %u", meta->rtp_timestamp);
          sParams->timestampTick = meta->rtp_timestamp;
          if (meta->leap_seconds_adjusted) {
            sParams->firstPacketTime += (LEAP_SECONDS*GST_SECOND);
            sParams->pass_rtp_ts_offset = 0;
            GST_DEBUG_OBJECT(sink, "Adjusted firstPacketTime: %" GST_TIME_FORMAT, GST_TIME_ARGS(sParams->firstPacketTime));
          } else {
            sParams->pass_rtp_ts_offset = (LEAP_SECONDS * GST_SECOND);
          }
      } else {
          GST_DEBUG_OBJECT(sink, "No RTP timestamp metadata found");
          //Update timestampTick using legacy method. This can be slightly different from the actual rtp_ticks.
          sParams->timestampTick = time_to_rtp_timestamp(sParams->firstPacketTime, sParams->sampleRate);
          sParams->pass_rtp_ts_offset = (LEAP_SECONDS * GST_SECOND);
      }
       if (sink->rtp_timestamp_offset > 0) {
         gdouble offset_ticks = (sink->rtp_timestamp_offset * sParams->sampleRate) / (gdouble)GST_SECOND;
         sParams->timestampTick += (guint32)offset_ticks;
         GST_DEBUG_OBJECT(sink, "Applied RTP timestamp offset: %" G_GUINT64_FORMAT " ns -> %.0f ticks, new timestampTick: %f", sink->rtp_timestamp_offset, offset_ticks, sParams->timestampTick);
       }
    } else {
      calculate_first_packet_time(sink);
      GST_DEBUG_OBJECT(sink, "firstPacketTime in HH:MM:SS.NS: %" GST_TIME_FORMAT,
                      GST_TIME_ARGS((GstClockTime)(sParams->firstPacketTime)));
      sParams->timestampTick = time_to_rtp_timestamp(sParams->firstPacketTime, sParams->sampleRate);
      GST_DEBUG_OBJECT(sink, "RTP TS start: sParams->timestampTick: %f", sParams->timestampTick);
      sParams->pass_rtp_ts_offset = 0;
    }
  }

  /* When pass-rtp-timestamp is enabled, firstPacketTime will be in UTC time, unless meta->leap_seconds_adjusted = true
    (which means upstream nvdsudpsrc received RTP timestamp in tai time and LEAP_SEC is adjusted for running time calculation).
     We have to calculate send_time_ns with an offset when pass-rtp-timestamp is enabled, otherwise it will always be less than time_now_ns
     which is in tai_time */
  gdouble send_time_ns = sParams->firstPacketTime +
                            (sParams->frameTimeInterval * sParams->frameCount) + sParams->pass_rtp_ts_offset;

  uint64_t time_now_ns = get_tai_time_ns (sink);
  GST_DEBUG_OBJECT(sink, "send_time_ns: %f, time_now_ns: %lu", send_time_ns, time_now_ns);
  if (send_time_ns > time_now_ns) {
    uint64_t sleep_time = send_time_ns - time_now_ns;
    // If less than 2ms to packetize and commit, log it for debbugging purpose.
    if (sleep_time < 2000000)
      GST_DEBUG_OBJECT (sink, "received the frame %lu with remaining time %lu ns to render",
          sParams->frameCount, sleep_time);
    if (sleep_time > (SLEEP_THRESHOLD_MS * 1000000)) {
      sleep_time -= SLEEP_THRESHOLD_MS * 1000000;
      g_usleep (sleep_time / 1000);
    }
  } else {
    if (!sink->pass_rtp_timestamp) {
      GST_WARNING_OBJECT (sink, "frame %lu late by %f ns", sParams->frameCount,
          send_time_ns - time_now_ns);
    }
  }

  guint line = 0;
  guint offset = 0;
  guint packetNum = 0;
  guint numFields = 1;
  guint fieldIdx = 0;
  guint rawPayloadSize, bytesPerLine = 0;
  gboolean isAdapterMemory = FALSE;

  gst_buffer_map (buffer, &info, GST_MAP_READ);
  guint8 *inDataPtr = NULL;
  guint bsize = 0;
  NvBufSurface *surf = NULL;

  inDataPtr = info.data;
  bsize = info.size;

  if (sink->is_nvmm) {
    surf = (NvBufSurface *) info.data;
    inDataPtr = surf->surfaceList[0].dataPtr;
    if (sParams->streamType == VIDEO_2110_20_STREAM)
      bsize = (sink->payloadSize - RTP_2110_20_MIN_HEADER_SIZE) * sink->packetsPerLine * surf->surfaceList[0].height;
  }

  gst_buffer_unmap (buffer, &info);

  if (sParams->streamType == AUDIO_2110_30_31_STREAM) {
    rawPayloadSize = sink->payloadSize - RTP_HEADER_SIZE;
    guint available = gst_adapter_available (sink->adapter);

    if (available == 0 && (bsize % rawPayloadSize) == 0) {
      // we don't have any previous cached data and current buffer size is
      // multiple of payload size, we can directly send the packets.
      isAdapterMemory = FALSE;
    } else {
      gst_buffer_ref (buffer);
      gst_adapter_push (sink->adapter, buffer);
      available += bsize;
      if (available < rawPayloadSize) {
        // Don't have sufficient data.
        return GST_FLOW_OK;
      }
      if ((available % rawPayloadSize) == 0) {
        inDataPtr = gst_adapter_take (sink->adapter, available);
        bsize = available;
      } else {
        guint nbytes = (available / rawPayloadSize) * rawPayloadSize;
        inDataPtr = gst_adapter_take (sink->adapter, nbytes);
        bsize = nbytes;
      }
      isAdapterMemory = TRUE;
    }
  } else {
    rawPayloadSize = sink->payloadSize - RTP_2110_20_MIN_HEADER_SIZE;
    bytesPerLine = sink->packetsPerLine * rawPayloadSize;

    if (sParams->videoType != PROGRESSIVE)
      numFields = 2;
  }

  const guint nextPixelOffset = sParams->width / sink->packetsPerLine;
  const gdouble bytesPerPixelReal = (gdouble) bytesPerLine / sParams->width;

  gboolean isNewBuffer = true;

  do {
    void *payload = NULL;
    void *appHeader = NULL;
    do {
      status = rmax_out_get_next_chunk (sink->streamId, &payload, &appHeader);
      if (status == RMAX_OK)
        break;

      if (status == RMAX_SIGNAL) {
        GST_DEBUG_OBJECT (sink, "Received CTRL-C");
        return GST_FLOW_EOS;
      }
    } while (status != RMAX_OK);

    if (sink->isGpuDirect) {
      cudaError_t ret = cudaSuccess;
      if (sParams->streamType == VIDEO_2110_20_STREAM) {
        if (isNewBuffer) {
          ret = cudaMemcpy2DAsync(
            (void*)payload,   // dst: Destination pointer (GPU memory allocated for network packets)
            bytesPerLine,     // dpitch: Destination pitch in bytes
            inDataPtr,        // src: Source pointer
            (sink->is_nvmm) ? surf->surfaceList[0].pitch : bytesPerLine, // spitch: Source pitch in bytes.
            bytesPerLine,     // width: Width of the 2D memory copy in bytes (actual data per line)
            sParams->height,  // height: Number of rows to copy
            cudaMemcpyDefault, // kind: Type of transfer (device to device, host to device etc)
            sink->cuda_stream // stream: CUDA stream for asynchronous operation
          );
          isNewBuffer = false;
          CHECK_CUDA(ret, "failed to copy chunk from device to device");
        }
      } else {
        GST_ERROR_OBJECT(sink, "Unsupported stream type");
        goto error;
      }
    }

    for (guint i = 0; i < sParams->chunkSize; i++) {
      guint8 *dataptr = NULL;
      guint8 *outPtr = (guint8 *) payload + (i * sParams->payloadStride);
      guint8 *outHdrPtr = appHeader ? (guint8 *) appHeader + (i * sParams->headerStride) : NULL;
      build_rtp_header (sink->isGpuDirect ? outHdrPtr : outPtr, line, offset, packetNum, fieldIdx, sink);

      if (sParams->streamType == AUDIO_2110_30_31_STREAM) {
        outPtr += RTP_HEADER_SIZE;
        dataptr = inDataPtr + packetNum * rawPayloadSize;
        sParams->timestampTick += ((sParams->sampleRate * sParams->ptime) / (gdouble) GST_SECOND);
      } else {
        guint lineOffset = line * numFields + fieldIdx;
        outPtr += RTP_2110_20_MIN_HEADER_SIZE;
        dataptr = inDataPtr + lineOffset * bytesPerLine + (guint)(offset * bytesPerPixelReal);

        // Next pixel offset for video
        offset += nextPixelOffset;
        if (sParams->seq == G_MAXUINT16) {
          sParams->extSeqNumber++;
        }
      }

      /* We don't copy anything in this loop if GpuDirect is enabled.
      It's just to build rtp header and increment all the required counters */
      if (!(sink->isGpuDirect)) {
        memcpy ((void *) outPtr, dataptr, rawPayloadSize);
      }
      bsize -= rawPayloadSize;
      sParams->seq++;
      packetNum++;
      if ((packetNum % sink->packetsPerLine) == 0) {
        line++;
        offset = 0;
      }
    }

    do {
      uint64_t timeout = 0;
      if (!(sParams->chunkNum % sParams->chunksPerFrame)) {
        timeout = (uint64_t) send_time_ns;
        // verify window is at least 600 nano away.
        if (timeout - 600 < get_tai_time_ns (sink)) {
          timeout = 0;
        } else {
          /*
           * When timer handler callback is not used we have a mismatch between
           * media_sender clock (TAI) and rivermax clock (UTC).
           * To fix this we are calling to align_to_rmax_time function to convert
           * @time from TAI to UTC
           */
          if (!sink->ptpSrc) {
            timeout = align_to_rmax_time (timeout);
            GST_DEBUG_OBJECT(sink, "align_to_rmax_time: timeout: %" GST_TIME_FORMAT, GST_TIME_ARGS((GstClockTime)timeout));
          }
        }
      }

      if (sink->isGpuDirect) {
        cudaError_t ret = cudaSuccess;
        ret = cuStreamSynchronize(sink->cuda_stream);
        CHECK_CUDA(ret, "failed to synchronize cuda stream");
      }

      status = rmax_out_commit (sink->streamId, timeout, 0);
      if (status == RMAX_OK) {
        break;
      }

      if (status == RMAX_SIGNAL) {
        GST_DEBUG_OBJECT (sink, "Received CTRL-C");
        rmax_out_cancel_unsent_chunks (sink->streamId);
        return GST_FLOW_EOS;
      } else if (status == RMAX_ERR_HW_COMPLETION_ISSUE) {
        GST_ERROR_OBJECT (sink, "error in commiting chunk, status = %d", status);
        rmax_out_cancel_unsent_chunks (sink->streamId);
        return GST_FLOW_ERROR;
      }

      if (status == RMAX_ERR_HW_SEND_QUEUE_FULL) {
        g_usleep (1);
      }
    } while (status != RMAX_OK);

    sParams->chunkNum++;
    if ((sParams->chunkNum % sParams->chunksPerFrame) == 0) {
      send_time_ns += sParams->frameTimeInterval;
      sParams->frameCount++;
      sParams->chunkNum = 0;
      if (sParams->streamType == VIDEO_2110_20_STREAM) {
        gdouble tick = sParams->sampleRate / sParams->fps;
        if (sParams->videoType != PROGRESSIVE)
          tick /= 2;
        sParams->timestampTick += tick;
        line = 0;
        packetNum = 0;
        fieldIdx ^= 1;
      }
    }
  } while (bsize >= rawPayloadSize);

  if (isAdapterMemory)
    g_free (inDataPtr);

  return GST_FLOW_OK;

error:
  return GST_FLOW_ERROR;
}

/**
 * @brief Renders a buffer list using Rivermax media API
 *
 * This function handles the rendering of a buffer list to the network using Rivermax media API.
 * It processes the buffers in chunks, aligns the timing with the stream's requirements,
 * and sends the data over the network with proper timing synchronization.
 *
 * @param bsink The base sink instance
 * @param bList The buffer list containing data to be sent
 * @return GstFlowReturn indicating the result of the render operation
 */
static GstFlowReturn
render_using_media_api (GstBaseSink * bsink, GstBufferList * bList)
{
  rmax_status_t status;
  GstBuffer *buf;
  GstMapInfo info = GST_MAP_INFO_INIT;
  GstNvDsUdpSink *sink = GST_NVDSUDPSINK (bsink);
  StreamParams *sParams = &sink->streamParams;

  gdouble send_time_ns;
  guint i, remainder;
  guint bufIdx = 0;
  void *payload;
  void *appHeader;

  remainder = gst_buffer_list_length (bList);
  if (remainder % sParams->chunkSize) {
    g_print ("packets in list should be multiple of chunk size: %d - len :%d \n",
     sParams->chunkSize, remainder);
    return GST_FLOW_ERROR;
  }

  if (!sParams->firstPacketTime) {
    calculate_first_packet_time (sink);
    sParams->timestampTick = time_to_rtp_timestamp (sParams->firstPacketTime, sParams->sampleRate);
  }

  send_time_ns = sParams->firstPacketTime + sParams->frameTimeInterval * sParams->frameCount;

  uint64_t time_now_ns = get_tai_time_ns (sink);
  if (send_time_ns > time_now_ns) {
    uint64_t sleep_time = send_time_ns - time_now_ns;
    if (sleep_time > (SLEEP_THRESHOLD_MS * 1000000)) {
      sleep_time -= SLEEP_THRESHOLD_MS * 1000000;
      g_usleep (sleep_time / 1000);
    }
  }

  do {
    do {
      status = rmax_out_get_next_chunk (sink->streamId, &payload, &appHeader);
      if (status == RMAX_OK)
        break;

      if (status == RMAX_SIGNAL) {
        GST_DEBUG_OBJECT (sink, "Received CTRL-C");
        return GST_FLOW_EOS;
      }
    } while (status != RMAX_OK);

    for (i = 0; i < sParams->chunkSize; i++) {
      buf = gst_buffer_list_get (bList, bufIdx++);
      gst_buffer_map (buf, &info, GST_MAP_READ);

      *(uint32_t *)(info.data + 4) = GUINT32_TO_BE ((uint32_t) sParams->timestampTick);

      if (sParams->streamType == VIDEO_2110_20_STREAM) {
        *(guint16 *)(info.data + 12) = g_htons(sParams->extSeqNumber);
        guint16 seqNum = g_ntohs (*(guint16 *)(info.data + 2));
        if (seqNum == G_MAXUINT16) {
          sParams->extSeqNumber++;
        }
      }

      if (sParams->streamType == AUDIO_2110_30_31_STREAM) {
        sParams->timestampTick += ((sParams->sampleRate * sParams->ptime) / (gdouble) GST_SECOND);
      }

      uint8_t *ptr = (uint8_t *) payload + (i * sParams->payloadStride);
      memcpy ((void *) ptr, info.data, info.size);
      gst_buffer_unmap (buf, &info);
    }

    do {
      uint64_t timeout = 0;
      if (!(sParams->chunkNum % sParams->chunksPerFrame)) {
        timeout = (uint64_t) send_time_ns;
        // verify window is at least 600 nano away.
        if (timeout - 600 < get_tai_time_ns (sink)) {
          timeout = 0;
        } else {
          /*
           * When timer handler callback is not used we have a mismatch between
           * media_sender clock (TAI) and rivermax clock (UTC).
           * To fix this we are calling to align_to_rmax_time function to convert
           * @time from TAI to UTC
           */
          timeout = align_to_rmax_time (timeout);
        }
      }

      status = rmax_out_commit (sink->streamId, timeout, 0);
      if (status == RMAX_OK) {
        break;
      }

      if (status == RMAX_SIGNAL) {
        GST_DEBUG_OBJECT (sink, "Received CTRL-C");
        rmax_out_cancel_unsent_chunks (sink->streamId);
        return GST_FLOW_EOS;
      } else if (status == RMAX_ERR_HW_COMPLETION_ISSUE) {
        GST_ERROR_OBJECT (sink, "error in commiting chunk, status = %d", status);
        rmax_out_cancel_unsent_chunks (sink->streamId);
        return GST_FLOW_ERROR;
      }

      if (status == RMAX_ERR_HW_SEND_QUEUE_FULL) {
        g_usleep (10);
        continue;
      }
    } while (status != RMAX_OK);

    sParams->chunkNum++;
    remainder -= sParams->chunkSize;
    if ((sParams->chunkNum % sParams->chunksPerFrame) == 0) {
      send_time_ns += sParams->frameTimeInterval;
      sParams->frameCount++;
      sParams->chunkNum = 0;
      if (sParams->streamType == VIDEO_2110_20_STREAM) {
        gdouble tick = sParams->sampleRate / sParams->fps;
        if (sParams->videoType != PROGRESSIVE)
          tick /= 2;
        sParams->timestampTick += tick;
      }
    }
  } while (remainder > 0);

  return GST_FLOW_OK;
}

/**
 * @brief Renders a list of buffers to the UDP sink
 *
 * @param bsink The base sink instance
 * @param bList The buffer list to render
 * @return GstFlowReturn indicating the result of the render operation
 */
static GstFlowReturn
gst_nvdsudpsink_render_list (GstBaseSink * bsink, GstBufferList * bList)
{
  rmax_status_t status;
  struct rmax_chunk *chunk;
  GstBuffer *buf;
  GstMapInfo info = GST_MAP_INFO_INIT;
  GstNvDsUdpSink *sink = GST_NVDSUDPSINK (bsink);

  guint i, remainder;
  guint count = sink->packetsPerChunk;
  guint bufIdx = 0;

  if (!sink->isGenericApi) {
    return render_using_media_api (bsink, bList);
  }

  remainder = gst_buffer_list_length (bList);

  do {
    if (remainder < sink->packetsPerChunk)
      count = remainder;

    chunk = &sink->chunks[sink->nextChunk];
    for (i = 0; i < count; i++) {
      buf = gst_buffer_list_get (bList, bufIdx++);
      gst_buffer_map (buf, &info, GST_MAP_READ);
      memcpy ((void *)chunk->packets[i].iovec->addr, info.data, info.size);
      chunk->packets[i].iovec->length = info.size;
      gst_buffer_unmap (buf, &info);
    }
    chunk->size = count;
    do {
      status = rmax_out_commit_chunk (sink->streamId, 0, chunk, 0);
      if (status == RMAX_ERR_HW_SEND_QUEUE_FULL) {
        g_usleep (10);
        continue;
      }

      if (status != RMAX_OK) {
        GST_ERROR("error in commiting chunk, status = %d \n", status);
        return GST_FLOW_ERROR;
      }
    } while (status != RMAX_OK);

    sink->nextChunk++;
    sink->nextChunk %= sink->nChunks;
    remainder -= count;
  } while (remainder > 0);

  return GST_FLOW_OK;
}

/**
 * @brief Thread function for handling rendering when separate rendering thread is required
 *
 * This function is created when a separate rendering thread is needed to handle the
 * processing and sending of buffers. It:
 * 1. Sets thread affinity for optimal performance
 * 2. Processes buffers from the queue in a loop
 * 3. Handles synchronization with the main thread
 *
 * @param bsink The base sink instance passed as thread data
 * @return NULL on completion or error
 */
static gpointer
render_thread (gpointer bsink)
{
  GstFlowReturn ret;
  GstBuffer *buffer;
  GstNvDsUdpSink *sink = GST_NVDSUDPSINK (bsink);

  if (!set_thread_affinity (sink->renderThreadCore)) {
    GST_ERROR_OBJECT (sink, "failed in setting thread affinity");
    sink->lastError = -1;
    return NULL;
  }

  while (sink->isRunning) {
    g_mutex_lock (&sink->qLock);
    while (g_queue_is_empty (sink->bufferQ)) {
      g_cond_wait (&sink->qCond, &sink->qLock);
      if (!sink->isRunning) {
        g_mutex_unlock (&sink->qLock);
        return NULL;
      }
    }

    buffer = (GstBuffer *) g_queue_pop_head (sink->bufferQ);
    g_cond_signal (&sink->qCond);
    g_mutex_unlock (&sink->qLock);
    ret = gst_nvdsudpsink_render_raw_frame (GST_BASE_SINK (sink), buffer);
    gst_buffer_unref (buffer);
    if (ret != GST_FLOW_OK) {
      sink->lastError = -1;
      return NULL;
    }
  }

  return NULL;
}
