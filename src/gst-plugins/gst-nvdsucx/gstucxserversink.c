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

/**
 * SECTION:element-gstucxserversink
 * @title: ucxserversink
 * @see_also: #basesink
 *
 * The ucxserversink element transmits information over UCX.
 *
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <gst/gst.h>
#include <gst/base/gstbasesink.h>
#include "gstucxserversink.h"
#include "gstucx.h"
#include <ucp/api/ucp.h>
#include "gstnvdsmeta.h"
#include "nvdsmeta.h"
#include <string.h>

GST_DEBUG_CATEGORY_STATIC (gst_ucx_server_sink_debug_category);
#define GST_CAT_DEFAULT gst_ucx_server_sink_debug_category

/* prototypes */
static void gst_ucx_server_sink_set_property (GObject * object,
    guint property_id, const GValue * value, GParamSpec * pspec);
static void gst_ucx_server_sink_get_property (GObject * object,
    guint property_id, GValue * value, GParamSpec * pspec);
static void gst_ucx_server_sink_finalize (GObject * object);
static gboolean gst_ucx_server_sink_start (GstBaseSink * sink);
static gboolean gst_ucx_server_sink_stop (GstBaseSink * sink);
static GstFlowReturn gst_ucx_server_sink_render (GstBaseSink * sink,
    GstBuffer * buffer);
static gboolean gst_ucx_server_sink_event (GstBaseSink * sink,
    GstEvent * event);
static void do_server_sink_work (gpointer user_data);

#define gst_ucx_server_sink_parent_class parent_class
G_DEFINE_TYPE (GstUcxServerSink, gst_ucx_server_sink, GST_TYPE_BASE_SINK);

/* pad templates */
static GstStaticPadTemplate gst_ucx_server_sink_sink_template =
    GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES
        (GST_CAPS_FEATURE_MEMORY_NVMM,
            VIDEO_FORMATS) ";" GST_AUDIO_CAPS_MAKE_WITH_FEATURES ("",
            AUDIO_FORMAT,
            AUDIO_CHANNELS) ";"
        GST_AUDIO_CAPS_MAKE_WITH_FEATURES (GST_CAPS_FEATURE_MEMORY_NVMM,
            AUDIO_FORMAT,
            AUDIO_CHANNELS) ";" GST_TEXT_CAPS_MAKE (" pango-markup, utf8")));

/* class initialization */
static void
gst_ucx_server_sink_class_init (GstUcxServerSinkClass * klass)
{
  GObjectClass *gobject_class = G_OBJECT_CLASS (klass);
  GstBaseSinkClass *base_sink_class = GST_BASE_SINK_CLASS (klass);

  gst_element_class_add_static_pad_template (GST_ELEMENT_CLASS (klass),
      &gst_ucx_server_sink_sink_template);

  gobject_class->set_property = gst_ucx_server_sink_set_property;
  gobject_class->get_property = gst_ucx_server_sink_get_property;

  g_object_class_install_property (gobject_class, PROP_ADDR,
      g_param_spec_string ("addr", "addr", "The IP address to listen on",
          UCX_DEFAULT_ADDR,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_HIGH_PERF,
    g_param_spec_boolean ("high-perf", "high-perf", "Enables (when true) or"
        "disables (when false) the RDMA transport for the plugin",
        UCX_DEFAULT_HIGH_PERF, (GParamFlags) (G_PARAM_READWRITE |
            G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_PORT,
      g_param_spec_int ("port", "port",
          "The port to listen to (default = 7174)",
          0, UCX_MAX_PORT, UCX_DEFAULT_PORT,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_BUF_TYPE,
      g_param_spec_enum ("buf-type",
          "Type of data to be handled by UCX - Video/Audio/Text",
          "Type of data to be handled by UCX - Video/Audio/Text",
          GST_TYPE_NVDSUCX_BUF_TYPE, NVDSUCX_BUF_TYPE_VIDEO,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_NUM_CONNS,
      g_param_spec_uint ("num-conns", "num-conns",
          "The number of connections to handle from clients",
          1, UCX_MAX_CONNS, UCX_DEFAULT_CONNS,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  PROP_NVDS_GPU_ID_INSTALL (gobject_class);

  g_object_class_install_property (gobject_class, PROP_MAX_EP_NUM,
    g_param_spec_uint ("max-ep-num", "max-ep-num",
          "Maximum number of UCX End Points. For optimal performance: "
          "max-ep-num should be greater than num-conn times nvbuf-batch-size",
          MIN_MAX_EP_NUM, MAX_MAX_EP_NUM, DEFAULT_MAX_EP_NUM,
          G_PARAM_READWRITE));

  g_object_class_install_property (gobject_class, PROP_NVBUF_BATCH_SIZE,
      g_param_spec_int ("nvbuf-batch-size", "nvbuf-batch-size",
          "The maximal batch size of a NV buffer", MIN_NVBUF_BATCH_SIZE,
          MAX_NVBUF_BATCH_SIZE, DEFAULT_NVBUF_BATCH_SIZE, G_PARAM_READWRITE));

  gst_element_class_set_static_metadata (GST_ELEMENT_CLASS (klass),
      "UCX server sink", "Sink/Network",
      "Send data as a server over the network via UCX",
      "NVIDIA Corporation. Post on Deepstream for Tesla forum for any queries "
      "@ https://devtalk.nvidia.com/default/board/209/");

  gobject_class->finalize = gst_ucx_server_sink_finalize;
  base_sink_class->start = GST_DEBUG_FUNCPTR (gst_ucx_server_sink_start);
  base_sink_class->stop = GST_DEBUG_FUNCPTR (gst_ucx_server_sink_stop);
  base_sink_class->render = GST_DEBUG_FUNCPTR (gst_ucx_server_sink_render);
  base_sink_class->event = GST_DEBUG_FUNCPTR (gst_ucx_server_sink_event);

  GST_DEBUG_CATEGORY_INIT (gst_ucx_server_sink_debug_category,
      "ucxserversink", 0, "UCX sink");
}

void
gst_ucx_server_sink_set_property (GObject * object, guint property_id,
    const GValue * value, GParamSpec * pspec)
{
  GstUcxServerSink *ucxserversink = GST_UCX_SERVER_SINK (object);

  GST_DEBUG_OBJECT (ucxserversink, "set_property");

  if (gst_ucx_server_set_property (&ucxserversink->server, property_id, value,
          pspec) ||
      gst_ucx_sink_set_property (&ucxserversink->sink, property_id, value,
          pspec))
    return;
  else
    G_OBJECT_WARN_INVALID_PROPERTY_ID (object, property_id, pspec);
}

void
gst_ucx_server_sink_get_property (GObject * object, guint property_id,
    GValue * value, GParamSpec * pspec)
{
  GstUcxServerSink *ucxserversink = GST_UCX_SERVER_SINK (object);

  GST_DEBUG_OBJECT (ucxserversink, "get_property");

  if (gst_ucx_server_get_property (&ucxserversink->server, property_id, value,
          pspec) ||
      gst_ucx_sink_get_property (&ucxserversink->sink, property_id, value,
          pspec))
    return;
  else
    G_OBJECT_WARN_INVALID_PROPERTY_ID (object, property_id, pspec);

}

void
gst_ucx_server_sink_init (GstUcxServerSink * ucxserversink)
{
  GST_DEBUG_OBJECT (ucxserversink, "init");

  gst_ucx_init ();
  gst_ucx_server_init (&ucxserversink->server);
  gst_ucx_sink_init (&ucxserversink->sink);

  return;
}

void
gst_ucx_server_sink_finalize (GObject * object)
{
  GstUcxServerSink *ucxserversink = GST_UCX_SERVER_SINK (object);

  GST_DEBUG_OBJECT (ucxserversink, "finalize");

  g_free (ucxserversink->server.addr);
  ucxserversink->server.addr = NULL;

  G_OBJECT_CLASS (gst_ucx_server_sink_parent_class)->finalize (object);
}

static gboolean
gst_ucx_server_sink_event (GstBaseSink * sink, GstEvent * event)
{
  GList *list_entry;
  GstUcxServerSink *ucxserversink = GST_UCX_SERVER_SINK (sink);
  NvDsUcxAmEventDesc *event_data = NULL;
  gboolean send_event = TRUE;
  gst_ucp_conn_ctx_t *conn = NULL;
  ucs_status_t status = UCS_OK;

  GST_DEBUG_OBJECT (ucxserversink, "sink event: %s",
      GST_EVENT_TYPE_NAME (event));

  event_data = g_new (NvDsUcxAmEventDesc, 1);
  if (!event_data) {
    GST_ERROR ("failed to allocate event data");
    return FALSE;
  }

  if (GST_EVENT_TYPE (event) == GST_EVENT_CAPS) {
    if (!gst_ucx_sink_cap_event_handling (&ucxserversink->sink, sink, event,
        event_data, &send_event)) {
      g_free(event_data);
      return FALSE;
    }
    if (send_event)
      ucxserversink->server.caps = event_data->data;
  } else {
    gst_ucx_set_nvevent_data (event, event_data, &send_event);
  }

  if ((send_event) && (event_data != NULL)) {
    for (list_entry = ucxserversink->server.priv->gst_ucx_server_conn_list;
        list_entry != NULL; list_entry = list_entry->next) {
      conn = (gst_ucp_conn_ctx_t *) list_entry->data;
      if (conn == NULL) {
        GST_ERROR_OBJECT (ucxserversink,
            "received caps event while there is no connection");
        break;
      }
      status = gst_ucp_send_event (conn->data_worker[0], conn->ep[0],
          event_data, sizeof (*event_data));
      if (status != UCS_OK) {
        GST_ERROR ("failed to send event to source, status: %s",
            ucs_status_string (status));
      }
    }

    /* Add events to list to send to new connections */
    g_mutex_lock (&ucxserversink->server.priv->event_lock);
    ucxserversink->server.priv->gst_ucx_nvevents_list =
        g_list_append (ucxserversink->server.priv->gst_ucx_nvevents_list,
        event_data);
    g_mutex_unlock (&ucxserversink->server.priv->event_lock);
  } else {
    g_free(event_data);
  }
  return GST_BASE_SINK_CLASS (parent_class)->event (sink, event);
}

/* Server start */

static gboolean
gst_ucx_server_sink_start (GstBaseSink * sink)
{
  int i;
  gboolean ret = TRUE;
  NVTX_WRAPPER_START;
  GstUcxServerSink *ucxserversink = GST_UCX_SERVER_SINK (sink);

  GST_DEBUG_OBJECT (ucxserversink, "start");

  if (start_server (ucxserversink, FALSE)) {
    ret = FALSE;
    goto exit;
  }

  if (start_sink(&ucxserversink->sink)) {
    stop_server (&ucxserversink->server);
    ret = FALSE;
    goto exit;
  }

  g_mutex_lock (&ucxserversink->server.priv->conn_lock);
  while (g_list_length (ucxserversink->server.priv->gst_ucx_server_conn_list) <
      1) {
    /* Waiting for listener thread to signal on new connection */
    /* XXX: If clients connect on-demand then no need to wait here. */
    g_cond_wait (&ucxserversink->server.priv->conn_cond,
        &ucxserversink->server.priv->conn_lock);
  }
  g_mutex_unlock (&ucxserversink->server.priv->conn_lock);

exit:
  NVTX_WRAPPER_END;
  return ret;
}

/* End of start */

/* Server render */
static void
do_server_sink_work (gpointer user_data)
{
  NvDsBatchMeta *batch_meta = NULL;
  void *rdma_buffer = NULL;
  size_t rdma_buffer_size = 0;
  void *rdma_meta_buffer = NULL;
  size_t rdma_meta_buffer_size = 0;

  GstUcxThreadDataSink *thread_data = (GstUcxThreadDataSink *) user_data;
  GstUcxServerSink *ucxserversink = (GstUcxServerSink *) thread_data->ucxparent;
  GstMapInfo *map_info = thread_data->mapdata;
  GstBuffer *buffer = (GstBuffer *) thread_data->buffer;
  int batch_id = thread_data->batch_id;
  int index;

  GList *list_entry;
  ucs_status_t ret;

  cudaError_t cudaReturn = cudaSetDevice (ucxserversink->sink.gpu_id);
  if (cudaReturn != cudaSuccess) {
    GST_ERROR ("Cuda set device failed for device %d",
        ucxserversink->sink.gpu_id);
    goto err;
  }

  if (gst_buffer_get_nvds_meta (buffer) != NULL) {
    batch_meta = gst_buffer_get_nvds_batch_meta (buffer);
  }

  if (batch_id == 0 && ucxserversink->sink.prep_am_meta != NULL
      && batch_meta != NULL) {
    if (ucxserversink->sink.prep_am_meta (&ucxserversink->
            sink.am_hdrs[batch_id].am_hdr_meta_buf, &rdma_meta_buffer,
            &rdma_meta_buffer_size, batch_meta, batch_id)) {
      GST_ERROR_OBJECT (ucxserversink, "prep_am_meta failed batch %d",
          batch_id);
      goto err;
    }
  }

  if (ucxserversink->sink.prep_am (buffer, map_info,
          &ucxserversink->sink.am_hdrs[batch_id].am_hdr_buf, &rdma_buffer,
          &rdma_buffer_size, batch_meta, batch_id,
          ucxserversink->sink.num_surfaces_per_frame, rdma_meta_buffer_size)) {
    GST_ERROR_OBJECT (ucxserversink, "prep_am failed batch %d", batch_id);
    goto err;
  }

  for (list_entry = ucxserversink->server.priv->gst_ucx_server_conn_list;
      list_entry != NULL; list_entry = list_entry->next) {
    gst_ucp_conn_ctx_t *conn = (gst_ucp_conn_ctx_t *) list_entry->data;
    index = batch_id % conn->ep_num;
    ret =
        gst_ucp_send (conn->data_worker[index], conn->ep[index], UCP_AM_ID,
        ucxserversink->sink.am_hdrs[batch_id].am_hdr_buf,
        ucxserversink->sink.am_hdrs[batch_id].am_hdr_buf_size, rdma_buffer,
        rdma_buffer_size);
    if (ret != UCS_OK) {
      GST_ERROR_OBJECT (ucxserversink,
          "failed to send with status %s batch_id %d", ucs_status_string (ret),
          batch_id);
      conn->err = ret;
      continue;
    }

    if (rdma_meta_buffer_size > 0) {
      ret =
          gst_ucp_send (conn->data_worker[index], conn->ep[index],
          UCP_AM_ID_META, ucxserversink->sink.am_hdrs[batch_id].am_hdr_meta_buf,
          ucxserversink->sink.am_hdrs[batch_id].am_hdr_meta_buf_size,
          rdma_meta_buffer, rdma_meta_buffer_size);
      if (ret != UCS_OK) {
        GST_ERROR_OBJECT (ucxserversink,
            "failed to send metadata status %s batch_id %d",
            ucs_status_string (ret), batch_id);
        conn->err = ret;
      }
    }
  }
err:
  return;
}

static GstFlowReturn
gst_ucx_server_sink_render (GstBaseSink * sink, GstBuffer * buffer)
{
  GstUcxServerSink *ucxserversink = GST_UCX_SERVER_SINK (sink);
  GstMapInfo map_info;
  GList *l;
  GList *next;
  int i;
  gst_ucp_conn_ctx_t *conn;
  guint timeout_tid;
  volatile int batch_end_timeout = 0;
  GstFlowReturn ret = GST_FLOW_OK;
  NVTX_WRAPPER_START;
  GST_DEBUG_OBJECT (ucxserversink, "render");

  g_mutex_lock (&ucxserversink->server.priv->conn_lock);
  while (g_list_length (ucxserversink->server.priv->gst_ucx_server_conn_list) <
      ucxserversink->server.num_conns) {
    /* Waiting for listener thread to signal on new connection */
    g_cond_wait (&ucxserversink->server.priv->conn_cond,
        &ucxserversink->server.priv->conn_lock);
  }

  g_mutex_unlock (&ucxserversink->server.priv->conn_lock);

  /*
   * Batching start -
   * a. Send batch start message to all conns
   *    or
   * a. Send first batch message to all conns
   * b. For each batch elem
   *   1. Prep the buffer with surface, metadata
   *   2. For every conn
   *     i. Send buffer
   *   3. increment batch sent
   * c. Wait for batch end message from conns
   */

  if (!gst_buffer_map (buffer, &map_info, GST_MAP_READ)) {
    GST_ERROR_OBJECT (ucxserversink, "Failed to map the gst buffer");
    ret = GST_FLOW_ERROR;
    goto exit;
  }

  for (i = 0; i < ucxserversink->sink.nvbuf_batch_size; i++) {
    ucxserversink->sink.threads_data[i].ucxparent = (void *) ucxserversink;
    ucxserversink->sink.threads_data[i].batch_id = i;
    ucxserversink->sink.threads_data[i].buffer = (void *) buffer;
    ucxserversink->sink.threads_data[i].mapdata = &map_info;
    ucxserversink->sink.threads_data[i].thread_id =
        gst_task_pool_push (ucxserversink->sink.thread_pool,
        (GstTaskPoolFunction) do_server_sink_work,
        &ucxserversink->sink.threads_data[i], NULL);
  }

  for (i = 0; i < ucxserversink->sink.nvbuf_batch_size; i++)
    gst_task_pool_join (ucxserversink->sink.thread_pool,
        ucxserversink->sink.threads_data[i].thread_id);

  // Batch end timeout per connection
  l = ucxserversink->server.priv->gst_ucx_server_conn_list;
  while (l != NULL) {
    next = l->next;
    conn = (gst_ucp_conn_ctx_t *) l->data;
    if (conn->err != UCS_OK) {
      free_gst_ucp_conn_ctx (l->data);
      ucxserversink->server.priv->gst_ucx_server_conn_list =
          g_list_delete_link (ucxserversink->server.
          priv->gst_ucx_server_conn_list, l);
      l = next;
      continue;
    }
    timeout_tid =
        g_timeout_add_seconds (BATCH_END_TIMEOUT_SEC, batch_end_check_func,
        (void *) &batch_end_timeout);

    while ((conn->complete == 0) && (!batch_end_timeout)) {
      //for(i = 0; i < conn->ep_num; i++)
      //ucp_worker_progress (conn->data_worker[i]);
      ucp_worker_progress (conn->data_worker[0]);
    }

    if (conn->complete) {
      conn->complete = 0;
      g_source_remove (timeout_tid);
      GST_DEBUG_OBJECT (ucxserversink, "Received batch end");
    } else {
      GST_ERROR_OBJECT (ucxserversink,
          "timeout - not received batch end event");
      ret = GST_FLOW_ERROR;
    }
    l = next;
  }

  memset (ucxserversink->sink.threads_data, 0,
      sizeof (GstUcxThreadDataSink) * ucxserversink->sink.nvbuf_batch_size);
  gst_buffer_unmap (buffer, &map_info);
exit:
  NVTX_WRAPPER_END;
  return ret;
}

/* End of render */

/* Sink stop */
static gboolean
gst_ucx_server_sink_stop (GstBaseSink * sink)
{
  GstUcxServerSink *ucxserversink = GST_UCX_SERVER_SINK (sink);
  NVTX_WRAPPER_START;
  GST_DEBUG_OBJECT (ucxserversink, "stop");
  stop_server (&ucxserversink->server);
  stop_sink(&ucxserversink->sink);
  NVTX_WRAPPER_END;
  return TRUE;
}

/* End of stop */
