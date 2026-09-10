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
 * SECTION:element-gstucxclientsink
 * @title: ucxclientsink
 * @see_also: #basesink
 *
 * The ucxclientsink element transmits information over RDMA.
 *
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <gst/gst.h>
#include <gst/base/gstbasesink.h>
#include "gstucxclientsink.h"
#include "gstucx.h"
#include <ucp/api/ucp.h>
#include "gstnvdsmeta.h"
#include "nvdsmeta.h"

GST_DEBUG_CATEGORY_STATIC (gst_ucx_client_sink_debug_category);
#define GST_CAT_DEFAULT gst_ucx_client_sink_debug_category

/* prototypes */
static void gst_ucx_client_sink_set_property (GObject * object,
    guint property_id, const GValue * value, GParamSpec * pspec);
static void gst_ucx_client_sink_get_property (GObject * object,
    guint property_id, GValue * value, GParamSpec * pspec);
static void gst_ucx_client_sink_finalize (GObject * object);
static gboolean gst_ucx_client_sink_start (GstBaseSink * sink);
static gboolean gst_ucx_client_sink_stop (GstBaseSink * sink);
static GstFlowReturn gst_ucx_client_sink_render (GstBaseSink * sink,
    GstBuffer * buffer);
static gboolean gst_ucx_client_sink_event (GstBaseSink * sink,
    GstEvent * event);
static void do_client_sink_work (gpointer data, gpointer user_data);

static GQuark dsmeta_quark = 0;
#define gst_ucx_client_sink_parent_class parent_class
G_DEFINE_TYPE (GstUcxClientSink, gst_ucx_client_sink, GST_TYPE_BASE_SINK);

/* pad templates */
static GstStaticPadTemplate gst_ucx_client_sink_sink_template =
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
gst_ucx_client_sink_class_init (GstUcxClientSinkClass * klass)
{
  GObjectClass *gobject_class = G_OBJECT_CLASS (klass);
  GstBaseSinkClass *base_sink_class = GST_BASE_SINK_CLASS (klass);

  gst_element_class_add_static_pad_template (GST_ELEMENT_CLASS (klass),
      &gst_ucx_client_sink_sink_template);

  gobject_class->set_property = gst_ucx_client_sink_set_property;
  gobject_class->get_property = gst_ucx_client_sink_get_property;

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

  PROP_NVDS_GPU_ID_INSTALL (gobject_class);

  g_object_class_install_property (gobject_class, PROP_NVBUF_BATCH_SIZE,
      g_param_spec_int ("nvbuf-batch-size", "nvbuf-batch-size",
          "The maximal batch size of a NV buffer", MIN_NVBUF_BATCH_SIZE,
          MAX_NVBUF_BATCH_SIZE, DEFAULT_NVBUF_BATCH_SIZE, G_PARAM_READWRITE));

  g_object_class_install_property (gobject_class, PROP_MAX_EP_NUM,
    g_param_spec_uint ("max-ep-num", "max-ep-num",
          "Maximum number of UCX End Points. For optimal performance: "
          "max-ep-num should be greater than num-conn times nvbuf-batch-size",
          MIN_MAX_EP_NUM, MAX_MAX_EP_NUM, DEFAULT_MAX_EP_NUM,
          G_PARAM_READWRITE));

  gst_element_class_set_static_metadata (GST_ELEMENT_CLASS (klass),
      "UCX client sink", "Sink/Network",
      "Send data as a client over the network via UCX",
      "<NVIDIA Corporation. Post on Deepstream for Tesla forum for any queries "
      "@ https://devtalk.nvidia.com/default/board/209/");

  gobject_class->finalize = gst_ucx_client_sink_finalize;
  base_sink_class->start = GST_DEBUG_FUNCPTR (gst_ucx_client_sink_start);
  base_sink_class->stop = GST_DEBUG_FUNCPTR (gst_ucx_client_sink_stop);
  base_sink_class->render = GST_DEBUG_FUNCPTR (gst_ucx_client_sink_render);
  base_sink_class->event = GST_DEBUG_FUNCPTR (gst_ucx_client_sink_event);

  GST_DEBUG_CATEGORY_INIT (gst_ucx_client_sink_debug_category,
      "ucxclientsink", 0, "UCX sink");
}

void
gst_ucx_client_sink_set_property (GObject * object, guint property_id,
    const GValue * value, GParamSpec * pspec)
{
  GstUcxClientSink *ucxclientsink = GST_UCX_CLIENT_SINK (object);

  GST_DEBUG_OBJECT (ucxclientsink, "set_property");

  if (gst_ucx_client_set_property (&ucxclientsink->client, property_id, value,
          pspec) ||
      gst_ucx_sink_set_property (&ucxclientsink->sink, property_id, value,
          pspec))
    return;
  else
    G_OBJECT_WARN_INVALID_PROPERTY_ID (object, property_id, pspec);
}

void
gst_ucx_client_sink_get_property (GObject * object, guint property_id,
    GValue * value, GParamSpec * pspec)
{
  GstUcxClientSink *ucxclientsink = GST_UCX_CLIENT_SINK (object);

  GST_DEBUG_OBJECT (ucxclientsink, "get_property");

  if (gst_ucx_client_get_property (&ucxclientsink->client, property_id, value,
          pspec)
      || gst_ucx_sink_get_property (&ucxclientsink->sink, property_id, value,
          pspec))
    return;
  else
    G_OBJECT_WARN_INVALID_PROPERTY_ID (object, property_id, pspec);

}

static void
gst_ucx_client_sink_init (GstUcxClientSink * ucxclientsink)
{
  GST_DEBUG_OBJECT (ucxclientsink, "init");

  gst_ucx_init ();
  gst_ucx_client_init (&ucxclientsink->client);
  gst_ucx_sink_init (&ucxclientsink->sink);

  return;
}

void
gst_ucx_client_sink_finalize (GObject * object)
{
  GstUcxClientSink *ucxclientsink = GST_UCX_CLIENT_SINK (object);

  GST_DEBUG_OBJECT (ucxclientsink, "finalize");

  g_free (ucxclientsink->client.addr);
  ucxclientsink->client.addr = NULL;

  G_OBJECT_CLASS (gst_ucx_client_sink_parent_class)->finalize (object);
}

/* Client start */

static gboolean
gst_ucx_client_sink_start (GstBaseSink * sink)
{
  GstUcxClientSink *ucxclientsink = GST_UCX_CLIENT_SINK (sink);
  gboolean ret = TRUE;
  NVTX_WRAPPER_START;
  GST_DEBUG_OBJECT (ucxclientsink, "start");

  dsmeta_quark = g_quark_from_static_string (NVDS_META_STRING);

  if (start_client (ucxclientsink, FALSE)) {
    ret = FALSE;
    goto exit;
  }

  if (start_sink (&ucxclientsink->sink)){
    ret = FALSE;
    goto exit;
  }

exit:
  NVTX_WRAPPER_END;
  return ret;
}


/* End of start */

static gboolean
gst_ucx_client_sink_event (GstBaseSink * sink, GstEvent * event)
{
  GList *list_entry;
  GstUcxClientSink *ucxclientsink = GST_UCX_CLIENT_SINK (sink);
  NvDsUcxAmEventDesc event_data = { 0 };
  gboolean send_event = TRUE;
  gst_ucp_conn_ctx_t *conn = NULL;
  ucs_status_t status = UCS_OK;

  GST_DEBUG_OBJECT (ucxclientsink, "sink event: %s",
      GST_EVENT_TYPE_NAME (event));

  if (GST_EVENT_TYPE (event) == GST_EVENT_CAPS) {
    if (!gst_ucx_sink_cap_event_handling (&ucxclientsink->sink, sink, event,
        &event_data, &send_event)) {
      return FALSE;
    }
  } else {
    gst_ucx_set_nvevent_data (event, &event_data, &send_event);
  }

  if (send_event) {
    gst_ucp_conn_ctx_t *conn = &ucxclientsink->client.priv->gst_ucp_conn_ctx;
    gst_ucp_send_event (conn->data_worker[0], conn->ep[0], &event_data,
        sizeof (event_data));
  }

  if (GST_BASE_SINK_CLASS (parent_class)->event)
    return GST_BASE_SINK_CLASS (parent_class)->event (sink, event);
  else
    return TRUE;
}

/* Client render */

static void
do_client_sink_work (gpointer user_data)
{
  NvDsBatchMeta *batch_meta = NULL;
  void *rdma_buffer = NULL;
  size_t rdma_buffer_size = 0;
  void *rdma_meta_buffer = NULL;
  size_t rdma_meta_buffer_size = 0;

  GstUcxThreadDataSink *thread_data = (GstUcxThreadDataSink *) user_data;
  GstUcxClientSink *ucxclientsink = (GstUcxClientSink *) thread_data->ucxparent;
  GstMapInfo *map_info = thread_data->mapdata;
  GstBuffer *buffer = (GstBuffer *) thread_data->buffer;
  int batch_id = thread_data->batch_id;
  int index;
  gst_ucp_conn_ctx_t *conn = &ucxclientsink->client.priv->gst_ucp_conn_ctx;

  ucs_status_t ret;
  cudaError_t cudaReturn = cudaSetDevice (ucxclientsink->sink.gpu_id);
  if (cudaReturn != cudaSuccess) {
    GST_ERROR ("Cuda set device failed for device %d",
        ucxclientsink->sink.gpu_id);
    conn->err = UCS_ERR_NO_MESSAGE;
    return;
  }

  if (gst_buffer_get_nvds_meta (buffer) != NULL) {
    batch_meta = gst_buffer_get_nvds_batch_meta (buffer);
  }

  if (batch_id == 0 && ucxclientsink->sink.prep_am_meta != NULL
      && batch_meta != NULL) {
    if (ucxclientsink->sink.prep_am_meta (&ucxclientsink->
            sink.am_hdrs[batch_id].am_hdr_meta_buf, &rdma_meta_buffer,
            &rdma_meta_buffer_size, batch_meta, batch_id)) {
      GST_ERROR_OBJECT (ucxclientsink, "prep_am_meta failed batch %d",
          batch_id);
      conn->err = UCS_ERR_NO_MESSAGE;
      return;
    }
  }

  if (ucxclientsink->sink.prep_am (buffer, map_info,
          &ucxclientsink->sink.am_hdrs[batch_id].am_hdr_buf, &rdma_buffer,
          &rdma_buffer_size, batch_meta, batch_id,
          ucxclientsink->sink.num_surfaces_per_frame, rdma_meta_buffer_size)) {
    GST_ERROR_OBJECT (ucxclientsink, "prep_am failed batch %d", batch_id);
    conn->err = UCS_ERR_NO_MESSAGE;
    return;
  }

  index = batch_id % conn->ep_num;
  ret=
      gst_ucp_send (conn->data_worker[index], conn->ep[index], UCP_AM_ID,
      ucxclientsink->sink.am_hdrs[batch_id].am_hdr_buf,
      ucxclientsink->sink.am_hdrs[batch_id].am_hdr_buf_size, rdma_buffer,
      rdma_buffer_size);
  if (ret != UCS_OK) {
    GST_ERROR_OBJECT (ucxclientsink,
        "failed to send with status %s batch_id %d", ucs_status_string (ret),
        batch_id);
    conn->err = ret;
    return;
  }

  if (rdma_meta_buffer_size > 0) {
    ret=
        gst_ucp_send (conn->data_worker[index], conn->ep[index],
        UCP_AM_ID_META, ucxclientsink->sink.am_hdrs[batch_id].am_hdr_meta_buf,
        ucxclientsink->sink.am_hdrs[batch_id].am_hdr_meta_buf_size,
        rdma_meta_buffer, rdma_meta_buffer_size);
    if (ret!= UCS_OK) {
      GST_ERROR_OBJECT (ucxclientsink,
          "failed to send metadata status %s batch_id %d",
          ucs_status_string (ret), batch_id);
      conn->err = ret;
    }
  }
}

static GstFlowReturn
gst_ucx_client_sink_render (GstBaseSink * sink, GstBuffer * buffer)
{
  GstUcxClientSink *ucxclientsink = GST_UCX_CLIENT_SINK (sink);
  GstMapInfo map_info;
  gst_ucp_conn_ctx_t *conn = &ucxclientsink->client.priv->gst_ucp_conn_ctx;
  guint timeout_tid;
  volatile int batch_end_timeout = 0;
  GstFlowReturn ret = GST_FLOW_OK;
  int i;
  NVTX_WRAPPER_START;

  GST_DEBUG_OBJECT (ucxclientsink, "render");

  /*
   * Batching start -
   * a. Send first batch message to all server
   * b. For each batch elem
   *   1. Prep the buffer with surface, metadata
   *   2. Send buffer
   * c. Wait for batch end message from server
   */

  if (!gst_buffer_map (buffer, &map_info, GST_MAP_READ)) {
    GST_DEBUG ("Failed to map the gst buffer");
    ret = GST_FLOW_ERROR;
    goto exit;
  }

  for (i = 0; i < ucxclientsink->sink.nvbuf_batch_size; i++) {
    ucxclientsink->sink.threads_data[i].ucxparent = (void *) ucxclientsink;
    ucxclientsink->sink.threads_data[i].batch_id = i;
    ucxclientsink->sink.threads_data[i].buffer = (void *) buffer;
    ucxclientsink->sink.threads_data[i].mapdata = &map_info;
    ucxclientsink->sink.threads_data[i].thread_id =
        gst_task_pool_push (ucxclientsink->sink.thread_pool,
        (GstTaskPoolFunction) do_client_sink_work,
        &ucxclientsink->sink.threads_data[i], NULL);
  }

  for (i = 0; i < ucxclientsink->sink.nvbuf_batch_size; i++)
    gst_task_pool_join (ucxclientsink->sink.thread_pool,
        ucxclientsink->sink.threads_data[i].thread_id);

  if (conn->err != UCS_OK) {
    ret = GST_FLOW_ERROR;
    goto unmap_exit;
  }
  timeout_tid =
      g_timeout_add_seconds (BATCH_END_TIMEOUT_SEC, batch_end_check_func,
          (void *) &batch_end_timeout);

  while ((conn->complete == 0) && (!batch_end_timeout)) {
    ucp_worker_progress (conn->data_worker[0]);
  }

  if (conn->complete) {
    conn->complete = 0;
    g_source_remove (timeout_tid);
    GST_DEBUG_OBJECT (ucxclientsink, "Received batch end");
  } else {
    GST_ERROR_OBJECT (ucxclientsink,
        "timeout - not received batch end event");
    ret = GST_FLOW_ERROR;
  }

unmap_exit:
  memset (ucxclientsink->sink.threads_data, 0,
      sizeof (GstUcxThreadDataSink) * ucxclientsink->sink.nvbuf_batch_size);
  gst_buffer_unmap (buffer, &map_info);
exit:
  NVTX_WRAPPER_END;
  return ret;

}

/* End of render */

/* Sink stop */

static gboolean
gst_ucx_client_sink_stop (GstBaseSink * sink)
{
  NVTX_WRAPPER_START;
  GstUcxClientSink *ucxclientsink = GST_UCX_CLIENT_SINK (sink);
  GST_DEBUG_OBJECT (ucxclientsink, "stop");
  stop_client (&ucxclientsink->client);
  stop_sink (&ucxclientsink->sink);
  NVTX_WRAPPER_END;
  return TRUE;
}
