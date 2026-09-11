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
 * SECTION:element-gstucxlientsrc
 *
 * The ucxclientsrc element receives data over UCX.
 *
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <gst/gst.h>
#include <gst/base/gstbasesrc.h>
#include <gst/base/gstpushsrc.h>
#include <gst/video/video.h>
#include "gstucxclientsrc.h"
#include "gstnvdsbufferpool.h"
#include "gst-nvcommon.h"

GST_DEBUG_CATEGORY_STATIC (gst_ucx_client_src_debug_category);
#define GST_CAT_DEFAULT gst_ucx_client_src_debug_category

/* prototypes */


static void gst_ucx_client_src_finalize (GObject * object);
static gboolean gst_ucx_client_src_start (GstBaseSrc * src);
static gboolean gst_ucx_client_src_stop (GstBaseSrc * src);
static gboolean gst_ucx_negotiate_caps (GstBaseSrc * src);
static GstFlowReturn gst_ucx_client_src_create (GstPushSrc * src,
    GstBuffer ** buf);

static void do_client_src_work (gpointer user_data);

static void gst_ucx_client_src_set_property (GObject * object,
    guint property_id, const GValue * value, GParamSpec * pspec);
static void gst_ucx_client_src_get_property (GObject * object,
    guint property_id, GValue * value, GParamSpec * pspec);

#define gst_ucx_client_src_parent_class parent_class
G_DEFINE_TYPE (GstUcxClientSrc, gst_ucx_client_src, GST_TYPE_PUSH_SRC);


/* pad templates */
static GstStaticPadTemplate gst_ucx_client_src_src_template =
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
gst_ucx_client_src_class_init (GstUcxClientSrcClass * klass)
{
  GObjectClass *gobject_class = G_OBJECT_CLASS (klass);
  GstBaseSrcClass *base_src_class = GST_BASE_SRC_CLASS (klass);
  GstPushSrcClass *push_src_class = GST_PUSH_SRC_CLASS (klass);

  gst_element_class_add_static_pad_template (GST_ELEMENT_CLASS (klass),
      &gst_ucx_client_src_src_template);

  gst_element_class_set_static_metadata (GST_ELEMENT_CLASS (klass),
      "UCX client source", "Source/Network",
      "Receive data as a client over the network via UCX",
      "<NVIDIA Corporation. Post on Deepstream for Tesla forum for any queries "
      "@ https://devtalk.nvidia.com/default/board/209/");

  gobject_class->set_property = gst_ucx_client_src_set_property;
  gobject_class->get_property = gst_ucx_client_src_get_property;

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
          "max-ep-num should be greater than nvbuf-batch-size",
          MIN_MAX_EP_NUM, MAX_MAX_EP_NUM, DEFAULT_MAX_EP_NUM,
          G_PARAM_READWRITE));

  g_object_class_install_property (gobject_class, PROP_RAW_BUF_SIZE,
      g_param_spec_uint ("raw-buf-size", "raw-buf-size", "raw buffer size",
          0, DEFAULT_RAW_BUFFER_SIZE, DEFAULT_RAW_BUFFER_SIZE,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  gobject_class->finalize = gst_ucx_client_src_finalize;
  base_src_class->start = GST_DEBUG_FUNCPTR (gst_ucx_client_src_start);
  base_src_class->stop = GST_DEBUG_FUNCPTR (gst_ucx_client_src_stop);
  base_src_class->negotiate = GST_DEBUG_FUNCPTR (gst_ucx_negotiate_caps);
  push_src_class->create = GST_DEBUG_FUNCPTR (gst_ucx_client_src_create);

  GST_DEBUG_CATEGORY_INIT (gst_ucx_client_src_debug_category,
      "ucxclientsrc", 0, "UCX source");
}

void
gst_ucx_client_src_set_property (GObject * object, guint property_id,
    const GValue * value, GParamSpec * pspec)
{
  GstUcxClientSrc *ucxclientsrc = GST_UCX_CLIENT_SRC (object);

  GST_DEBUG_OBJECT (ucxclientsrc, "set_property, id: %d", property_id);

  if (gst_ucx_client_set_property (&ucxclientsrc->client, property_id, value,
          pspec) ||
      gst_ucx_src_set_property (&ucxclientsrc->src, property_id, value, pspec))
    return;

  else
    G_OBJECT_WARN_INVALID_PROPERTY_ID (object, property_id, pspec);
}

void
gst_ucx_client_src_get_property (GObject * object, guint property_id,
    GValue * value, GParamSpec * pspec)
{
  GstUcxClientSrc *ucxclientsrc = GST_UCX_CLIENT_SRC (object);

  GST_DEBUG_OBJECT (ucxclientsrc, "get_property");

  if (gst_ucx_client_get_property (&ucxclientsrc->client, property_id, value,
          pspec) ||
      gst_ucx_src_get_property (&ucxclientsrc->src, property_id, value, pspec))
    return;

  else
    G_OBJECT_WARN_INVALID_PROPERTY_ID (object, property_id, pspec);

}

static void
gst_ucx_client_src_init (GstUcxClientSrc * ucxclientsrc)
{
  GST_DEBUG_OBJECT (ucxclientsrc, "init");

  gst_ucx_init ();
  gst_ucx_client_init (&ucxclientsrc->client);
  gst_ucx_src_init (&ucxclientsrc->src);

  return;
}

void
gst_ucx_client_src_finalize (GObject * object)
{
  GstUcxClientSrc *ucxclientsrc = GST_UCX_CLIENT_SRC (object);

  GST_DEBUG_OBJECT (ucxclientsrc, "finalize");

  g_free (ucxclientsrc->client.addr);
  ucxclientsrc->client.addr = NULL;

  G_OBJECT_CLASS (gst_ucx_client_src_parent_class)->finalize (object);
}

/* Client start */

static gboolean
gst_ucx_client_src_start (GstBaseSrc * src)
{
  NVTX_WRAPPER_START;
  gboolean ret = TRUE;
  GstUcxClientSrc *ucxclientsrc = GST_UCX_CLIENT_SRC (src);

  GST_DEBUG_OBJECT (ucxclientsrc, "start");

  if (start_client (ucxclientsrc, TRUE)) {
    ret = FALSE;
    goto exit;
  }

  if (start_src (&ucxclientsrc->src, src)){
    ret = FALSE;
    goto exit;
  }

exit:
  NVTX_WRAPPER_END;
  return ret;
}

/* End of client start */

static gboolean
gst_ucx_negotiate_caps (GstBaseSrc * src)
{
  GstUcxClientSrc *ucxclientsrc = GST_UCX_CLIENT_SRC (src);
  GST_DEBUG_OBJECT (ucxclientsrc, "negotiate");

  return gst_ucx_src_negotiate_caps (src, &ucxclientsrc->src);
}

/* Client create */
static void
do_client_src_work (gpointer user_data)
{
  GstUcxThreadDataSource *thread_data = (GstUcxThreadDataSource *) user_data;
  GstUcxClientSrc *ucxclientsrc = (GstUcxClientSrc *) thread_data->ucxparent;
  GstBuffer **buffer = (GstBuffer **) thread_data->buffer;
  int batch_id = thread_data->batch_id;
  int index = batch_id % ucxclientsrc->client.priv->gst_ucp_conn_ctx.ep_num;
  ucs_status_t status;

  cudaError_t cudaReturn = cudaSetDevice (ucxclientsrc->src.gpu_id);
  if (cudaReturn != cudaSuccess) {
    GST_ERROR_OBJECT (ucxclientsrc, "Cuda set device failed for device %d",
        ucxclientsrc->src.gpu_id);
    gst_buffer_unref (*buffer);
    return;
  }
  status =
      gst_ucp_recv (ucxclientsrc->client.priv->
      gst_ucp_conn_ctx.data_worker[index], buffer, ucxclientsrc->src.prep_am,
      &ucxclientsrc->src.am_desc[index],
      &ucxclientsrc->client.priv->gst_ucp_conn_ctx.conn_lock,
      &ucxclientsrc->client.priv->gst_ucp_conn_ctx.err);

  if (status != UCS_OK) {
    GST_DEBUG_OBJECT (ucxclientsrc,
        "Failed receiving data with status %s batch_id %d",
        ucs_status_string (status), batch_id);
    ucxclientsrc->client.priv->gst_ucp_conn_ctx.err = status;
    return;
  }

  if (gst_ucx_is_recv_meta (&ucxclientsrc->src, index)) {
    ucxclientsrc->client.priv->gst_ucp_conn_ctx.main_worker = index;
    status =
        gst_ucp_recv_meta (ucxclientsrc->client.priv->
        gst_ucp_conn_ctx.data_worker[index], buffer,
        ucxclientsrc->src.prep_am_meta, &ucxclientsrc->src.am_desc_meta,
        &ucxclientsrc->client.priv->gst_ucp_conn_ctx.err);

    if (status != UCS_OK) {
      GST_DEBUG_OBJECT (ucxclientsrc,
          "Failed receiving metdata with status %s batch_id %d",
          ucs_status_string (status), batch_id);
      ucxclientsrc->client.priv->gst_ucp_conn_ctx.err = status;
      return;
    }
  }
}

static GstFlowReturn
gst_ucx_client_src_create (GstPushSrc * src, GstBuffer ** buf)
{
  GstUcxClientSrc *ucxclientsrc = GST_UCX_CLIENT_SRC (src);
  GstFlowReturn ret = GST_FLOW_OK;
  GstEvent *nvevent;
  int i;
  NVTX_WRAPPER_START;
  GST_DEBUG_OBJECT (ucxclientsrc, "create");

  ret = gst_buffer_pool_acquire_buffer (ucxclientsrc->src.pool, buf, NULL);
  if (ret != GST_FLOW_OK) {
    GST_WARNING_OBJECT (ucxclientsrc, "err acquiring from buffer pool");
    goto exit;
  }

  /*
   * Batching support -
   * a. Loop for the batch size as provided in the first message of the batch
   * b. Add buffers within the current gst buffer
   * c. Send batch complete message to server
   */

  for (i = 0; i < ucxclientsrc->src.nvbuf_batch_size; i++) {
    ucxclientsrc->src.threads_data[i].ucxparent = (void *) ucxclientsrc;
    ucxclientsrc->src.threads_data[i].batch_id = i;
    ucxclientsrc->src.threads_data[i].buffer = (void *) buf;
    ucxclientsrc->src.threads_data[i].thread_id =
        gst_task_pool_push (ucxclientsrc->src.thread_pool,
        (GstTaskPoolFunction) do_client_src_work,
        &ucxclientsrc->src.threads_data[i], NULL);
  }

  for (i = 0; i < ucxclientsrc->src.nvbuf_batch_size; i++)
    gst_task_pool_join (ucxclientsrc->src.thread_pool,
        ucxclientsrc->src.threads_data[i].thread_id);

  if (ucxclientsrc->client.priv->gst_ucp_conn_ctx.err == UCS_ERR_CANCELED ||
      ucxclientsrc->client.priv->gst_ucp_conn_ctx.err ==
      UCS_ERR_CONNECTION_RESET) {
    for (i = 0; i < ucxclientsrc->src.nvbuf_batch_size; i++) {
      nvevent = gst_nvevent_new_stream_eos (i);
      gst_pad_push_event (GST_BASE_SRC_PAD (src), nvevent);
    }
    ret = GST_FLOW_EOS;
    goto exit;
  }

  if (gst_ucp_send_only (ucxclientsrc->client.priv->
          gst_ucp_conn_ctx.data_worker[ucxclientsrc->client.priv->
              gst_ucp_conn_ctx.main_worker],
          ucxclientsrc->client.priv->gst_ucp_conn_ctx.ep[ucxclientsrc->
              client.priv->gst_ucp_conn_ctx.main_worker],
          EVENT_AM_BATCH_END) != UCS_OK) {

    GST_ERROR_OBJECT (ucxclientsrc, "failed gst_ucp_send_end_batch");
    ret = GST_FLOW_ERROR;
    gst_buffer_unref (*buf);
  }
exit:
  NVTX_WRAPPER_END;
  return ret;
}

/* End of client create */

static gboolean
gst_ucx_client_src_stop (GstBaseSrc * src)
{
  NVTX_WRAPPER_START;
  GstUcxClientSrc *ucxclientsrc = GST_UCX_CLIENT_SRC (src);
  GST_DEBUG_OBJECT (ucxclientsrc, "stop");
  stop_client (&ucxclientsrc->client);
  stop_src (&ucxclientsrc->src) ;
  NVTX_WRAPPER_END;
  return TRUE;
}