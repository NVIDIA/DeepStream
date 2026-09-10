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
 * SECTION:element-gstucxserversrc
 * @title: ucxserversrc
 * @see_also: #basesrc
 *
 * The ucxserversrc element recive information via UCX.
 *
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <gst/gst.h>
#include "gstucxserversrc.h"
#include "gstucx.h"
#include <gst-nvquery.h>
#include <ucp/api/ucp.h>

GST_DEBUG_CATEGORY_STATIC (gst_ucx_server_src_debug_category);
#define GST_CAT_DEFAULT gst_ucx_server_src_debug_category

static void gst_ucx_server_src_finalize (GObject * object);
static gboolean gst_ucx_server_src_start (GstBaseSrc * src);
static gboolean gst_ucx_server_src_stop (GstBaseSrc * src);
gboolean gst_ucx_server_src_negotiate_caps (GstBaseSrc * src);
static GstFlowReturn gst_ucx_server_src_create (GstPushSrc * src,
    GstBuffer ** buf);
static void gst_ucx_server_src_set_property (GObject * object,
    guint property_id, const GValue * value, GParamSpec * pspec);
static void gst_ucx_server_src_get_property (GObject * object,
    guint property_id, GValue * value, GParamSpec * pspec);
static gboolean gst_ucx_server_src_query (GstBaseSrc * parent,
    GstQuery * query);

#define gst_ucx_server_src_parent_class parent_class

G_DEFINE_TYPE (GstUcxServerSrc, gst_ucx_server_src, GST_TYPE_PUSH_SRC);

/* pad templates */
static GstStaticPadTemplate gst_ucx_server_src_src_template =
    GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
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
gst_ucx_server_src_class_init (GstUcxServerSrcClass * klass)
{
  GObjectClass *gobject_class = G_OBJECT_CLASS (klass);
  GstBaseSrcClass *base_src_class = GST_BASE_SRC_CLASS (klass);
  GstPushSrcClass *push_src_class = GST_PUSH_SRC_CLASS (klass);

  gst_element_class_add_static_pad_template (GST_ELEMENT_CLASS (klass),
      &gst_ucx_server_src_src_template);

  gst_element_class_set_static_metadata (GST_ELEMENT_CLASS (klass),
      "UCX server source", "Source/Network",
      "Receive data as a server over the network via UCX",
      "<NVIDIA Corporation. Post on Deepstream for Tesla forum for any queries "
      "@ https://devtalk.nvidia.com/default/board/209/");

  gobject_class->set_property = gst_ucx_server_src_set_property;
  gobject_class->get_property = gst_ucx_server_src_get_property;

  g_object_class_install_property (gobject_class, PROP_ADDR,
      g_param_spec_string ("addr", "addr", "The IP address to connect to",
          UCX_DEFAULT_ADDR,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_HIGH_PERF,
    g_param_spec_boolean ("high-perf", "high-perf", "Enables (when true) or"
        "disables (when false) the RDMA transport for the plugin",
        UCX_DEFAULT_HIGH_PERF, (GParamFlags) (G_PARAM_READWRITE |
            G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_PORT,
      g_param_spec_int ("port", "port",
          "The port to connect to (default = 7174)",
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

  PROP_NVBUF_MEMORY_TYPE_INSTALL (gobject_class);

  g_object_class_install_property (gobject_class, PROP_NUM_NVBUF,
      g_param_spec_int ("num-nvbuf", "num-nvbuf",
          "The number of NV buffers to allocate. 0 is unlimited", MIN_NUM_NVBUF,
          MAX_NUM_NVBUF, DEFAULT_NUM_NVBUF, G_PARAM_READWRITE));

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

  g_object_class_install_property (gobject_class, PROP_RAW_BUF_SIZE,
      g_param_spec_uint ("raw-buf-size", "raw-buf-size", "raw buffer size",
          0, DEFAULT_RAW_BUFFER_SIZE, DEFAULT_RAW_BUFFER_SIZE,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  gobject_class->finalize = gst_ucx_server_src_finalize;
  base_src_class->start = GST_DEBUG_FUNCPTR (gst_ucx_server_src_start);
  base_src_class->stop = GST_DEBUG_FUNCPTR (gst_ucx_server_src_stop);
  base_src_class->negotiate =
      GST_DEBUG_FUNCPTR (gst_ucx_server_src_negotiate_caps);
  base_src_class->query = GST_DEBUG_FUNCPTR (gst_ucx_server_src_query);
  push_src_class->create = GST_DEBUG_FUNCPTR (gst_ucx_server_src_create);

  GST_DEBUG_CATEGORY_INIT (gst_ucx_server_src_debug_category,
      "ucxserversrc", 0, "NET source");
}

/* prototypes */

void
gst_ucx_server_src_set_property (GObject * object, guint property_id,
    const GValue * value, GParamSpec * pspec)
{
  GstUcxServerSrc *ucxserversrc = GST_UCX_SERVER_SRC (object);

  GST_DEBUG_OBJECT (ucxserversrc, "set_property");

  if (gst_ucx_server_set_property (&ucxserversrc->server, property_id, value,
          pspec) ||
      gst_ucx_src_set_property (&ucxserversrc->src, property_id, value, pspec))
    return;
  else {
    G_OBJECT_WARN_INVALID_PROPERTY_ID (object, property_id, pspec);
    GST_ERROR("G_OBJECT_WARN_INVALID_PROPERTY_ID");
  }
}

void
gst_ucx_server_src_get_property (GObject * object, guint property_id,
    GValue * value, GParamSpec * pspec)
{
  GstUcxServerSrc *ucxserversrc = GST_UCX_SERVER_SRC (object);

  GST_DEBUG_OBJECT (ucxserversrc, "get_property");

  if (gst_ucx_server_get_property (&ucxserversrc->server, property_id, value,
          pspec) ||
      gst_ucx_src_get_property (&ucxserversrc->src, property_id, value, pspec))
    return;
  else
    G_OBJECT_WARN_INVALID_PROPERTY_ID (object, property_id, pspec);
}

static void
gst_ucx_server_src_init (GstUcxServerSrc * ucxserversrc)
{
  GST_DEBUG_OBJECT (ucxserversrc, "init");

  gst_ucx_init ();
  gst_ucx_server_init (&ucxserversrc->server);
  gst_ucx_src_init (&ucxserversrc->src);

  return;
}

static gboolean
gst_ucx_server_src_query (GstBaseSrc * parent, GstQuery * query)
{
  GstUcxServerSrc *ucxserversrc = GST_UCX_SERVER_SRC (parent);

  GST_DEBUG_OBJECT (ucxserversrc, "src_query");

  if (gst_nvquery_is_batch_size (query)) {
    GST_DEBUG_OBJECT (ucxserversrc, "batch size query, batch size is %d",
        ucxserversrc->src.nvbuf_batch_size);
    gst_nvquery_batch_size_set (query, ucxserversrc->src.nvbuf_batch_size);
    return TRUE;
  }

  return GST_BASE_SRC_CLASS (parent_class)->query (parent, query);
}

void
gst_ucx_server_src_finalize (GObject * object)
{
  GstUcxServerSrc *ucxserversrc = GST_UCX_SERVER_SRC (object);

  GST_DEBUG_OBJECT (ucxserversrc, "finalize");

  g_free (ucxserversrc->server.addr);
  ucxserversrc->server.addr = NULL;

  G_OBJECT_CLASS (gst_ucx_server_src_parent_class)->finalize (object);
}

/* Server src start */
static gboolean
gst_ucx_server_src_start (GstBaseSrc * src)
{
  gboolean ret = TRUE;
  GstUcxServerSrc *ucxserversrc = GST_UCX_SERVER_SRC (src);
  NVTX_WRAPPER_START;
  GST_DEBUG_OBJECT (ucxserversrc, "start");

  if (start_server (ucxserversrc, TRUE)) {
    ret = FALSE;
    goto exit;
  }

  if (start_src (&ucxserversrc->src, src)) {
    stop_server (&ucxserversrc->server);
    ret = FALSE;
    goto exit;
  }

  g_mutex_lock (&ucxserversrc->server.priv->conn_lock);
  while (ucxserversrc->server.priv->gst_ucx_server_conn_list == NULL) {
    /* Waiting for listener thread to signal on new connection */
    g_cond_wait (&ucxserversrc->server.priv->conn_cond, &ucxserversrc->server.priv->conn_lock);
  }
  g_mutex_unlock (&ucxserversrc->server.priv->conn_lock);

exit:
  NVTX_WRAPPER_END;
  return ret;
}

/* End of server src start */

static gboolean
gst_ucx_server_src_stop (GstBaseSrc * src)
{
  NVTX_WRAPPER_START;
  GstUcxServerSrc *ucxserversrc = GST_UCX_SERVER_SRC (src);
  GST_DEBUG_OBJECT (ucxserversrc, "stop");
  stop_server (&ucxserversrc->server);
  stop_src (&ucxserversrc->src);
  NVTX_WRAPPER_END;
  return TRUE;
}

gboolean
gst_ucx_server_src_negotiate_caps (GstBaseSrc * src)
{
  GstUcxServerSrc *ucxserversrc = GST_UCX_SERVER_SRC (src);
  GST_DEBUG_OBJECT (ucxserversrc, "negotiate");
  return gst_ucx_src_negotiate_caps (src, &ucxserversrc->src);
}

/* Server src create */
static void
do_server_src_work (gpointer user_data) {
  GstUcxThreadDataSource *thread_data = (GstUcxThreadDataSource *) user_data;
  GstUcxServerSrc *ucxserversrc = (GstUcxServerSrc *) thread_data->ucxparent;
  GstBuffer **buffer = (GstBuffer **) thread_data->buffer;
  int batch_id = thread_data->batch_id;
  GList *list_entry;
  gst_ucp_conn_ctx *conn;
  list_entry = ucxserversrc->server.priv->gst_ucx_server_conn_list;
  conn = (gst_ucp_conn_ctx_t *) list_entry->data;
  int index = batch_id % conn->ep_num;

  cudaError_t cudaReturn = cudaSetDevice (ucxserversrc->src.gpu_id);
  if (cudaReturn != cudaSuccess) {
    GST_ERROR_OBJECT (ucxserversrc, "Cuda set device failed for device %d",
        ucxserversrc->src.gpu_id);
    gst_buffer_unref (*buffer);
    return;
  }

  thread_data->status =
      gst_ucp_recv (conn->data_worker[index], buffer, ucxserversrc->src.prep_am,
      &ucxserversrc->src.am_desc[index],
      &conn->conn_lock,  &conn->err);

  if (thread_data->status != UCS_OK) {
    GST_DEBUG_OBJECT (ucxserversrc, "batch_id %d return with status %s",
        batch_id, ucs_status_string (thread_data->status));
    return;
  }

  if (gst_ucx_is_recv_meta (&ucxserversrc->src, index)) {
    conn->main_worker = index;
    thread_data->status =
        gst_ucp_recv_meta (conn->data_worker[index], buffer,
        ucxserversrc->src.prep_am_meta,
        &ucxserversrc->src.am_desc_meta,  &conn->err);

    if (thread_data->status != UCS_OK) {
      GST_DEBUG_OBJECT (ucxserversrc,
          "Failed receiveing metdata with status %s batch_id %d",
          ucs_status_string (thread_data->status), batch_id);
      return;
    }
  }
}


static GstFlowReturn
gst_ucx_server_src_create (GstPushSrc * src, GstBuffer ** buf)
{
  GstUcxServerSrc *ucxserversrc = GST_UCX_SERVER_SRC (src);
  GstEvent *nvevent;
  int i;
  GstFlowReturn ret = GST_FLOW_OK;
  GList *list_entry;
  gst_ucp_conn_ctx *conn;
  NVTX_WRAPPER_START;
  GST_DEBUG_OBJECT (ucxserversrc, "create");

  ret = gst_buffer_pool_acquire_buffer (ucxserversrc->src.pool, buf, NULL);
  if (ret != GST_FLOW_OK) {
    GST_WARNING_OBJECT (ucxserversrc, "err acquiring from buffer pool");
    goto exit;
  }

  g_mutex_lock (&ucxserversrc->server.priv->conn_lock);
  while (ucxserversrc->server.priv->gst_ucx_server_conn_list == NULL) {
    /* Waiting for listener thread to signal on new connection */
    g_cond_wait (&ucxserversrc->server.priv->conn_cond, &ucxserversrc->server.priv->conn_lock);
  }
  g_mutex_unlock (&ucxserversrc->server.priv->conn_lock);

  /*
   * Batching support -
   * a. Loop for the batch size as provided as property
   * b. Add buffers within the current gst buffer
   * c. Send batch complete message to server
   */

  for (i = 0; i < ucxserversrc->src.nvbuf_batch_size; i++) {
    ucxserversrc->src.threads_data[i].ucxparent = (void *) ucxserversrc;
    ucxserversrc->src.threads_data[i].batch_id = i;
    ucxserversrc->src.threads_data[i].buffer = (void *) buf;
    ucxserversrc->src.threads_data[i].thread_id =
        gst_task_pool_push (ucxserversrc->src.thread_pool,
        (GstTaskPoolFunction) do_server_src_work,
        &ucxserversrc->src.threads_data[i], NULL);
  }

  for (i = 0; i < ucxserversrc->src.nvbuf_batch_size; i++) {
    gst_task_pool_join (ucxserversrc->src.thread_pool,
        ucxserversrc->src.threads_data[i].thread_id);
    if (ucxserversrc->src.threads_data[i].status != UCS_OK)
      ucxserversrc->src.is_eos = TRUE;
  }

  if (ucxserversrc->src.is_eos) {
    for (i = 0; i < ucxserversrc->src.nvbuf_batch_size; i++) {
      nvevent = gst_nvevent_new_stream_eos (i);
      gst_pad_push_event (GST_BASE_SRC_PAD (src), nvevent);
    }
    ret = GST_FLOW_EOS;
    goto exit;
  }

  list_entry = ucxserversrc->server.priv->gst_ucx_server_conn_list;
  if (!list_entry) {
    GST_ERROR_OBJECT (ucxserversrc, "All connections are disconected");
    return GST_FLOW_ERROR;
  }
  conn = (gst_ucp_conn_ctx_t *) list_entry->data;

  if (gst_ucp_send_only (conn->data_worker[conn->main_worker],
          conn->ep[conn->main_worker], EVENT_AM_BATCH_END) != UCS_OK) {
    GST_ERROR_OBJECT (ucxserversrc, "failed gst_ucp_send_end_batch");
    ret = GST_FLOW_ERROR;
    gst_buffer_unref (*buf);
  }

exit:
  NVTX_WRAPPER_END;
  return ret;
}

/* End of client create */