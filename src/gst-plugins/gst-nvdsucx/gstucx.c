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


#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "gstucx.h"
#include <stdlib.h>
#include <ucp/api/ucp.h>
#include "gstucxclientsrc.h"
#include "gstucxclientsink.h"
#include "gstucxserversrc.h"
#include "gstucxserversink.h"

#define MAX_OUT_CONNS 64

GST_DEBUG_CATEGORY_STATIC (gst_ucx_common_debug_category);
#define GST_CAT_DEFAULT gst_ucx_common_debug_category

static gboolean gst_ucx_src_send_event_caps (GstCaps * caps, GstBaseSrc * src);
static gboolean gst_ucx_src_send_nvevent (NvDsUcxAmEventDesc * event,
    GstBaseSrc * src);

static const GEnumValue nvdsucx_buf_type_vals[] = {
  {NVDSUCX_BUF_TYPE_VIDEO,
      "Video buffers - NvBufSurface", "nvdsucx-buf-video"},
  {NVDSUCX_BUF_TYPE_AUDIO_NV,
      "Audio buffers - NvBufAudio", "nvdsucx-buf-nv-audio"},
  {NVDSUCX_BUF_TYPE_AUDIO_RAW,
      "Audio buffers - raw memory", "nvdsucx-buf-raw-audio"},
  {NVDSUCX_BUF_TYPE_TEXT,
      "Text buffers - gchar", "nvdsucx-buf-text"},
  {0, NULL, NULL},
};

int
get_thread_num (gpointer ucxdata, bool is_src, bool is_server)
{
  int nvbuf_batch_size;
  int max_ep_num;
  if (!ucxdata) {
    GST_ERROR ("null data");
    return -1;
  }
  GstUcxServerSink *ucxserversink = (GstUcxServerSink *) ucxdata;
  GstUcxServerSrc *ucxserversrc = (GstUcxServerSrc *) ucxdata;
  GstUcxClientSink *ucxclientsink = (GstUcxClientSink *) ucxdata;
  GstUcxClientSrc *ucxclientsrc = (GstUcxClientSrc *) ucxdata;
  if (is_server) {
    nvbuf_batch_size =
        is_src ? ucxserversrc->src.nvbuf_batch_size : ucxserversink->
        sink.nvbuf_batch_size;
    max_ep_num =
        is_src ? ucxserversrc->server.max_ep_num : ucxserversink->
        server.max_ep_num;
  }
  else {
    nvbuf_batch_size =
        is_src ? ucxclientsrc->src.nvbuf_batch_size : ucxclientsink->
        sink.nvbuf_batch_size;
    max_ep_num =
        is_src ? ucxclientsrc->client.max_ep_num : ucxclientsink->
        client.max_ep_num;
  }
  return MIN (max_ep_num, nvbuf_batch_size);
}

static GstUcxSrc *
get_gstucxsrc (gpointer ucxdata, bool is_server)
{
  if (!ucxdata) {
    GST_ERROR ("null data");
    return NULL;
  }
  GstUcxServerSrc *ucxserversrc = (GstUcxServerSrc *) ucxdata;
  GstUcxClientSrc *ucxclientsrc = (GstUcxClientSrc *) ucxdata;

  return is_server ? &ucxserversrc->src : &ucxclientsrc->src;
}

static GstUcxServer *
get_gstucxserver (gpointer ucxdata, bool is_src)
{
  if (!ucxdata) {
    GST_ERROR ("null data");
    return NULL;
  }
  GstUcxServerSink *ucxserversink = (GstUcxServerSink *) ucxdata;
  GstUcxServerSrc *ucxserversrc = (GstUcxServerSrc *) ucxdata;
  return is_src ? &ucxserversrc->server : &ucxserversink->server;
}

static GstUcxClient *
get_gstucxclient (gpointer ucxdata, bool is_src)
{
  if (!ucxdata) {
    GST_ERROR ("null data");
    return NULL;
  }
  GstUcxClientSink *ucxclientsink = (GstUcxClientSink *) ucxdata;
  GstUcxClientSrc *ucxclientsrc = (GstUcxClientSrc *) ucxdata;
  return is_src ? &ucxclientsrc->client : &ucxclientsink->client;
}

GType
gst_nvdsucx_buf_type_get_type (void)
{
  static GType etype = 0;

  if (etype == 0)
    etype = g_enum_register_static ("GstNvDsUcxBufType", nvdsucx_buf_type_vals);

  return etype;
}

int
gst_ucx_create_thread_pool (GstTaskPool ** pool, gint nvbuf_batch_size)
{
  GError *error = NULL;
  *pool = gst_ucx_shared_task_pool_new ();
  gst_ucx_shared_task_pool_set_max_threads (GST_UCX_SHARED_TASK_POOL (*pool),
      nvbuf_batch_size);

  gst_task_pool_prepare (*pool, &error);
  if (error != NULL) {
    GST_ERROR ("Failed preparing UCX thread pool with %s", error->message);
    g_error_free (error);
    gst_task_pool_cleanup (*pool);
    gst_object_unref (*pool);
    return 1;
  }

  return 0;
}

gboolean
gst_ucx_is_recv_meta (GstUcxSrc * src, int batch_id)
{
  if (src->buf_type == NVDSUCX_BUF_TYPE_VIDEO
      && src->am_desc[batch_id].nv_dsucx_buf.nvbuf_surf.has_user_meta == 1)
    return TRUE;

  if (src->buf_type == NVDSUCX_BUF_TYPE_AUDIO_NV
      && src->am_desc[batch_id].nv_dsucx_buf.nvbuf_audio.has_user_meta == 1)
    return TRUE;

  return FALSE;
}

/* UCP stuff */

typedef struct test_req
{
  volatile int complete;
} ucp_test_req_t;

ucs_status_t
gst_ucp_init_context (ucp_context_h * ucp_context, gboolean high_perf)
{
  ucp_params_t ucp_params;
  ucs_status_t status;
  ucp_config_t *config = NULL;

  if (!high_perf) {
    status = ucp_config_read (NULL, NULL, &config);
    if (status != UCS_OK) {
      GST_ERROR ("Failed to set UCX transports");
      return status;
    }
    status = ucp_config_modify(config, "TLS", UCX_HIGH_PERF_DISABLE_TLS);
    if (status != UCS_OK) {
        GST_ERROR ("Failed to set UCX transports");
    }
  }

  memset (&ucp_params, 0, sizeof (ucp_params));
  ucp_params.field_mask = UCP_PARAM_FIELD_FEATURES;
  ucp_params.features = UCP_FEATURE_AM;
  ucp_params.field_mask |= UCP_PARAM_FIELD_MT_WORKERS_SHARED;
  ucp_params.mt_workers_shared = 1;

  status = ucp_init (&ucp_params, config, ucp_context);
  if (status != UCS_OK) {
    GST_ERROR ("failed to ucp_init (%s)", ucs_status_string (status));
    ucp_context = NULL;
  }
  return status;
}

static uint64_t
rand_client_num (void)
{
  uint64_t r30 = (uint64_t)RAND_MAX * rand () + rand ();
  uint64_t s30 = rand () & 0x3;
  return (r30 << 34) + (s30 << 32);
}

static ucs_status_t
gst_ucp_init_worker (ucp_context_h ucp_context, ucp_worker_h * ucp_worker,
    uint64_t client_id)
{
  ucp_worker_params_t worker_params;
  ucs_status_t status;

  memset (&worker_params, 0, sizeof (worker_params));
  worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
  worker_params.thread_mode = UCS_THREAD_MODE_SINGLE;
  if (client_id) {
    worker_params.field_mask |= UCP_WORKER_PARAM_FIELD_CLIENT_ID;
    worker_params.client_id = client_id;
  }

  status = ucp_worker_create (ucp_context, &worker_params, ucp_worker);
  if (status != UCS_OK)
    GST_ERROR ("failed to ucp_worker_create (%s)",
        ucs_status_string (status));
  return status;
}

static void
gst_ucp_ep_close (ucp_worker_h ucp_worker, ucp_ep_h ep)
{
  ucp_request_param_t param;
  ucs_status_t status;
  void *close_req;

  param.op_attr_mask = UCP_OP_ATTR_FIELD_FLAGS;
  param.flags = UCP_EP_CLOSE_FLAG_FORCE;
  close_req = ucp_ep_close_nbx (ep, &param);
  if (UCS_PTR_IS_PTR (close_req)) {
    do {
      ucp_worker_progress (ucp_worker);
      status = ucp_request_check_status (close_req);
    } while (status == UCS_INPROGRESS);
    ucp_request_free (close_req);
  } else if (UCS_PTR_STATUS (close_req) != UCS_OK) {
    GST_WARNING ("failed to close ep %p", (void *) ep);
  }

  return;
}

void
gst_ucp_reset_am_desc (GstUcxSrc *src)
{
  NvDsUcxAmDesc * am_desc = src->am_desc;
  if (!am_desc)
    return;

  for (int i = 0; i < src->nvbuf_batch_size; i++) {
    am_desc[i].desc = NULL;
    am_desc[i].length = 0;
    am_desc[i].complete = 1;
  }
  src->am_desc_meta.desc = NULL;
  src->am_desc_meta.length = 0;
  src->am_desc_meta.complete = 1;
  return;
}

static ucs_status_t
gst_ucp_am_src_cb_audio_nv (void *arg, const void *header,
    size_t header_length, void *data, size_t length,
    const ucp_am_recv_param_t * param)
{
  NvBufAudioDsUcx *nvbufaudiodsucx;
  NvDsUcxAmDesc *am_desc;
  ucs_status_t ret = UCS_INPROGRESS;
  NVTX_WRAPPER_START;

  if (header_length != sizeof (NvBufAudioDsUcx)) {
    GST_ERROR ("Truncated header received for audio, "
        "expected %ld received %ld", header_length, sizeof (NvBufAudioDsUcx));
    ret = UCS_ERR_MESSAGE_TRUNCATED;
    goto exit;
  }

  nvbufaudiodsucx = (NvBufAudioDsUcx *) header;
  am_desc = (NvDsUcxAmDesc *) arg;

  if (!am_desc) {
    GST_ERROR ("AM desc is missing for audio buffer");
    ret = UCS_ERR_NO_RESOURCE;
    goto exit;
  }

  am_desc->nv_dsucx_buf.nvbuf_audio.numFilled = nvbufaudiodsucx->numFilled;
  am_desc->nv_dsucx_buf.nvbuf_audio.batchSize = nvbufaudiodsucx->batchSize;
  am_desc->nv_dsucx_buf.nvbuf_audio.batchId = nvbufaudiodsucx->batchId;
  am_desc->nv_dsucx_buf.nvbuf_surf.has_user_meta = 0;

  if (nvbufaudiodsucx->batchId >= nvbufaudiodsucx->numFilled) {
    GST_DEBUG ("Received AM hdr [%d] | invalid batch_id : numFilled %d",
        nvbufaudiodsucx->batchId, nvbufaudiodsucx->numFilled);
    am_desc->length = 0;
    am_desc->complete = 2;
    ret = UCS_OK;
    goto exit;
  }

  copy_gstbuf_info (&nvbufaudiodsucx->nvgstbuffer_params, NULL,
      &am_desc->nv_dsucx_buf.nvbuf_audio.nvgstbuffer_params, NULL);


  am_desc->nv_dsucx_buf.nvbuf_audio.num_surfaces_per_frame =
      nvbufaudiodsucx->num_surfaces_per_frame;
  am_desc->nv_dsucx_buf.nvbuf_audio.has_user_meta =
      nvbufaudiodsucx->has_user_meta;

  copy_nvbufaudio_params (&nvbufaudiodsucx->nvbufaudio_params,
      &am_desc->nv_dsucx_buf.nvbuf_audio.nvbufaudio_params);

  GST_DEBUG ("Received AM hdr [%d] | numFilled %d length %ld",
      am_desc->nv_dsucx_buf.nvbuf_audio.batchId,
      am_desc->nv_dsucx_buf.nvbuf_audio.numFilled, length);

  am_desc->desc = data;
  am_desc->length = length;
  am_desc->complete = 1;

exit:
  NVTX_WRAPPER_END;
  return ret;
}

static ucs_status_t
gst_ucp_am_src_cb_video (void *arg, const void *header, size_t header_length,
    void *data, size_t length, const ucp_am_recv_param_t * param)
{
  NvBufSurfaceDsUcx *nvbufsurfacedsucx;
  NvDsUcxAmDesc *am_desc;
  ucs_status_t ret = UCS_INPROGRESS;
  NVTX_WRAPPER_START;

  if (header_length != sizeof (NvBufSurfaceDsUcx)) {
    GST_ERROR ("Truncated header received for video, "
        "expected %ld received %ld", header_length, sizeof (NvBufSurfaceDsUcx));
    ret = UCS_ERR_MESSAGE_TRUNCATED;
    goto exit;
  }

  nvbufsurfacedsucx = (NvBufSurfaceDsUcx *) header;
  am_desc = (NvDsUcxAmDesc *) arg;

  if (!am_desc) {
    GST_ERROR ("AM desc is missing for video");
    ret = UCS_ERR_NO_RESOURCE;
    goto exit;
  }

  am_desc->nv_dsucx_buf.nvbuf_surf.numFilled = nvbufsurfacedsucx->numFilled;
  am_desc->nv_dsucx_buf.nvbuf_surf.batchSize = nvbufsurfacedsucx->batchSize;
  am_desc->nv_dsucx_buf.nvbuf_surf.batchId = nvbufsurfacedsucx->batchId;
  am_desc->nv_dsucx_buf.nvbuf_surf.has_user_meta = 0;

  if (nvbufsurfacedsucx->batchId >= nvbufsurfacedsucx->numFilled) {
    GST_DEBUG ("Received AM hdr [%d] | invalid batch_id : numFilled %d",
        nvbufsurfacedsucx->batchId, nvbufsurfacedsucx->numFilled);
    am_desc->length = 0;
    am_desc->complete = 2;
    ret = UCS_OK;
    goto exit;
  }

  copy_gstbuf_info (&nvbufsurfacedsucx->nvgstbuffer_params, NULL,
      &am_desc->nv_dsucx_buf.nvbuf_surf.nvgstbuffer_params, NULL);

  am_desc->nv_dsucx_buf.nvbuf_surf.num_surfaces_per_frame =
      nvbufsurfacedsucx->num_surfaces_per_frame;
  am_desc->nv_dsucx_buf.nvbuf_surf.has_user_meta =
      nvbufsurfacedsucx->has_user_meta;

  copy_nvbufsurf_params (&nvbufsurfacedsucx->nvbufsurf_params,
      &am_desc->nv_dsucx_buf.nvbuf_surf.nvbufsurf_params);

  GST_DEBUG ("Received AM hdr [%d] | numFilled %d length %ld",
      am_desc->nv_dsucx_buf.nvbuf_surf.batchId,
      am_desc->nv_dsucx_buf.nvbuf_surf.numFilled, length);

  am_desc->desc = data;
  am_desc->length = length;
  am_desc->complete = 1;

exit:
  NVTX_WRAPPER_END;
  return ret;
}

static ucs_status_t
gst_ucp_am_src_cb_meta (void *arg, const void *header,
    size_t header_length, void *data, size_t length,
    const ucp_am_recv_param_t * param)
{
  NvDsCustomMetaUcx *nvdscustommeta;
  NvDsUcxAmDesc *am_desc;
  ucs_status_t ret = UCS_INPROGRESS;
  NVTX_WRAPPER_START;

  if (!(param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV)) {
    ret = UCS_ERR_UNSUPPORTED;
    goto exit;
  }

  if (header_length != sizeof (NvDsCustomMetaUcx)) {
    GST_ERROR ("Truncated header received for metadata, "
        "expected %ld received %ld", header_length, sizeof (NvDsCustomMetaUcx));
    ret = UCS_ERR_MESSAGE_TRUNCATED;
    goto exit;
  }

  nvdscustommeta = (NvDsCustomMetaUcx *) header;
  am_desc = (NvDsUcxAmDesc *) arg;
  if (!am_desc) {
    GST_ERROR ("AM desc is missing");
    ret = UCS_ERR_NO_RESOURCE;
    goto exit;
  }

  am_desc->nv_dscustom_buf.payloadType = nvdscustommeta->payloadType;
  am_desc->nv_dscustom_buf.payloadSize = nvdscustommeta->payloadSize;
  am_desc->nv_dscustom_buf.batchId = nvdscustommeta->batchId;

  GST_DEBUG
      ("Received AM meta hdr batchId %d payloadType %X payloadSize %d length %ld",
      am_desc->nv_dsucx_buf.nvbuf_surf.batchId, nvdscustommeta->payloadType,
      nvdscustommeta->payloadSize, length);

  am_desc->desc = data;
  am_desc->length = length;
  am_desc->complete = 1;

exit:
  NVTX_WRAPPER_END;
  return ret;
}

static ucs_status_t
gst_ucp_am_src_cb_raw (void *arg, const void *header,
    size_t header_length, void *data, size_t length,
    const ucp_am_recv_param_t * param)
{
  NvBufRawDsUcx *nvbufraw;
  NvDsUcxAmDesc *am_desc;
  ucs_status_t ret = UCS_INPROGRESS;
  NVTX_WRAPPER_START;

  if (!(param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV)) {
    ret = UCS_ERR_UNSUPPORTED;
    goto exit;
  }

  if (header_length != sizeof (NvBufRawDsUcx)) {
    GST_ERROR ("Truncated header received for raw data, "
        "expected %ld received %ld", header_length, sizeof (NvBufRawDsUcx));
    ret = UCS_ERR_MESSAGE_TRUNCATED;
    goto exit;
  }

  nvbufraw = (NvBufRawDsUcx *) header;
  am_desc = (NvDsUcxAmDesc *) arg;
  if (!am_desc) {
    GST_ERROR ("AM desc is missing for raw header");
    ret = UCS_ERR_NO_RESOURCE;
    goto exit;
  }

  copy_gstbuf_info (&nvbufraw->nvgstbuffer_params, NULL,
      &am_desc->nv_dsucx_buf.nvbuf_raw.nvgstbuffer_params, NULL);
  am_desc->nv_dsucx_buf.nvbuf_raw.size = nvbufraw->size;

  am_desc->desc = data;
  am_desc->length = length;
  am_desc->complete = 1;

exit:
  NVTX_WRAPPER_END;
  return ret;
}

static ucs_status_t
gst_ucp_am_cb_server_end_batch (void *arg, const void *header,
    size_t header_length, void *data, size_t length,
    const ucp_am_recv_param_t * param)
{
  gst_ucp_conn_ctx_t *conn_ctx;
  conn_ctx = (gst_ucp_conn_ctx_t *) arg;

  conn_ctx->complete = 1;

  return UCS_OK;
}

static ucs_status_t
gst_ucp_am_src_cb_handle_event (void *arg, const void *header,
    size_t header_length, void *data, size_t length,
    const ucp_am_recv_param_t * param)
{
  GstBaseSrc *src;
  int subset = 0;
  gboolean ret = TRUE;

  if ((param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV)) {
    GST_ERROR ("RNDV flags is on for caps event");
    return UCS_ERR_UNSUPPORTED;
  }

  src = GST_BASE_SRC_CAST (arg);
  NvDsUcxAmEventDesc *event_data = (NvDsUcxAmEventDesc *) data;

  if (event_data->type == GST_EVENT_CAPS) {
    GST_DEBUG ("Handle caps: [%s]", event_data->data);
    ret =
        gst_ucx_src_send_event_caps (gst_caps_from_string (event_data->data),
        src);
  } else {
    GST_DEBUG ("Handling NVEvent: %d", event_data->type);
    ret = gst_ucx_src_send_nvevent (event_data, src);
  }

  if (!ret) {
    GST_ERROR ("Failed to forward event to src pipeline");
  }

  return UCS_OK;
}

static ucs_status_t
gst_ucp_register_am_handler (ucp_worker_h data_worker, GstNvDsUcxAmId am_id,
    void *arg, unsigned flags, NvDsUcxAmHandlerFunc am_handler)
{
  ucp_am_handler_param_t param;
  uint64_t field_mask = UCP_AM_HANDLER_PARAM_FIELD_ID |
      UCP_AM_HANDLER_PARAM_FIELD_CB | UCP_AM_HANDLER_PARAM_FIELD_ARG;

  if (flags != 0) {
    field_mask |= UCP_AM_HANDLER_PARAM_FIELD_FLAGS;
    param.flags = flags;
  }
  param.field_mask = field_mask;
  param.id = am_id;
  param.arg = arg;
  param.cb = am_handler;

  return ucp_worker_set_am_recv_handler (data_worker, &param);
}

static ucs_status_t
gst_ucp_init_src_surface_am (ucp_worker_h data_worker, GstNvDsUcxAmId am_id,
    void *arg, GstNvDsUcxBufType buf_type)
{
  ucs_status_t status;

  switch (buf_type) {
    case NVDSUCX_BUF_TYPE_VIDEO:
      status =
          gst_ucp_register_am_handler (data_worker, am_id, arg, 0,
          gst_ucp_am_src_cb_video);
      break;
    case NVDSUCX_BUF_TYPE_AUDIO_NV:
      status =
          gst_ucp_register_am_handler (data_worker, am_id, arg, 0,
          gst_ucp_am_src_cb_audio_nv);
      break;
    case NVDSUCX_BUF_TYPE_AUDIO_RAW:
    case NVDSUCX_BUF_TYPE_TEXT:
      status =
          gst_ucp_register_am_handler (data_worker, am_id, arg, 0,
          gst_ucp_am_src_cb_raw);
      break;
    default:
      return UCS_ERR_INVALID_PARAM;
  }

  return status;
}

ucs_status_t
gst_ucp_init_src_am (gpointer ucxdata, gst_ucp_conn_ctx_t * conn,
    bool is_server)
{
  ucp_am_handler_param_t param;
  ucs_status_t status;
  GstUcxSrc *ucxsrc = get_gstucxsrc (ucxdata, is_server);
  if (!ucxsrc) {
    GST_ERROR ("Could not get ucxsrc object");
    return UCS_ERR_NO_MESSAGE;
  }

  if (!ucxsrc->am_desc) {
    int thread_num = get_thread_num (ucxdata, TRUE, is_server);
    if (thread_num == -1)
      return UCS_ERR_NO_MESSAGE;
    ucxsrc->am_desc = g_new0 (NvDsUcxAmDesc, thread_num);
    if (!ucxsrc->am_desc) {
      GST_ERROR ("failed to alloc am_desc");
      return UCS_ERR_NO_MEMORY;
    }
  }

  if ((status =
          gst_ucp_init_src_surface_am (conn->data_worker[conn->ep_num],
              UCP_AM_ID, &(ucxsrc->am_desc[conn->ep_num]),
              ucxsrc->buf_type)) != UCS_OK) {
    return status;
  }

  return gst_ucp_register_am_handler (conn->data_worker[conn->ep_num],
      UCP_AM_ID_META, &ucxsrc->am_desc_meta, 0, gst_ucp_am_src_cb_meta);
}

static gboolean
gst_ucx_src_send_nvevent (NvDsUcxAmEventDesc * nvevent, GstBaseSrc * src)
{
  GstUcxSrc *ucxsrc;
  GstEvent *event = NULL;
  guint source_id = nvevent->sourceId;

  ucxsrc = (GstUcxSrc *) src;

  switch (nvevent->type) {
    case GST_NVEVENT_PAD_ADDED:
      event = gst_nvevent_new_pad_added (source_id);
      GST_DEBUG ("Pad added event for source: %d", source_id);
      break;
    case GST_NVEVENT_PAD_DELETED:
      event = gst_nvevent_new_pad_deleted (source_id);
      GST_DEBUG ("Pad deleted for source: %d", source_id);
      break;
    case GST_NVEVENT_STREAM_EOS:
      event = gst_nvevent_new_stream_eos (source_id);
      GST_DEBUG ("Stream eos for source: %d", source_id);
      ucxsrc->is_eos = TRUE;
      gst_ucp_reset_am_desc (ucxsrc);
      break;
    case GST_NVEVENT_STREAM_RESET:
      event = gst_nvevent_new_stream_reset (source_id);
      GST_DEBUG ("Stream reset for source: %d", source_id);
      ucxsrc->is_eos = TRUE;
      gst_ucp_reset_am_desc (ucxsrc);
      break;
    case GST_NVEVENT_STREAM_SEGMENT:
      event = gst_nvevent_new_stream_segment (source_id, &nvevent->segment);
      GST_DEBUG ("Stream segment for source: %d", source_id);
      break;
    case GST_NVEVENT_STREAM_START:
      event = gst_nvevent_new_stream_start (source_id, nvevent->data);
      GST_DEBUG ("Stream start for source: %d stream: %s",
          source_id, nvevent->data);
      break;
  }

  if (event != NULL) {
    GST_DEBUG ("Pushing event to pipeline: %p type %d", event,
        GST_EVENT_TYPE (event));
    return gst_pad_push_event (GST_BASE_SRC_PAD (src), event);
  }

  return TRUE;
}

static gboolean
gst_ucx_src_send_event_caps (GstCaps * caps, GstBaseSrc * src)
{
  GstUcxSrc *ucxsrc;
  GstEvent *event;
  gboolean res;

  ucxsrc = (GstUcxSrc *) src;

  /* Deactivate already configured pool */
  gst_buffer_pool_set_active (ucxsrc->pool, FALSE);
  gst_object_unref (ucxsrc->pool);

  res = gst_ucx_src_buffer_pool_new (ucxsrc, caps);
  if (!res)
    return FALSE;

  event = gst_event_new_caps (caps);
  return gst_pad_push_event (GST_BASE_SRC_PAD (src), event);
}

static ucs_status_t
request_wait (ucp_worker_h ucp_worker, void *request, ucp_test_req_t * ctx)
{
  ucs_status_t status;

  /* if operation was completed immediately */
  if (request == NULL) {
    return UCS_OK;
  }

  if (UCS_PTR_IS_ERR (request)) {
    return UCS_PTR_STATUS (request);
  }

  while (ctx->complete == 0) {
    ucp_worker_progress (ucp_worker);
  }
  status = ucp_request_check_status (request);

  ucp_request_free (request);

  return status;
}

static ucs_status_t
request_finalize (ucp_worker_h ucp_worker, ucp_test_req_t * request,
    ucp_test_req_t * ctx)
{
  ucs_status_t status;

  status = request_wait (ucp_worker, request, ctx);
  if (status != UCS_OK) {
    GST_ERROR ("unable to process UCX message (%s)",
        ucs_status_string (status));
    return status;
  }

  return UCS_OK;
}

static void
am_recv_cb (void *request, ucs_status_t status, size_t length, void *user_data)
{
  ucp_test_req_t *ctx = (ucp_test_req_t *) user_data;

  ctx->complete = 1;
}

static void
am_send_cb (void *request, ucs_status_t status, void *user_data)
{
  ucp_test_req_t *ctx = (ucp_test_req_t *) user_data;

  ctx->complete = 1;
}

ucs_status_t
gst_ucp_send_event (ucp_worker_h ucp_worker, ucp_ep_h ep, void *msg,
    size_t msg_len)
{
  ucp_request_param_t param;
  ucp_test_req_t *request;
  ucp_test_req_t ctx;

  ctx.complete = 0;

  param.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA;
  param.user_data = &ctx;
  param.cb.send = (ucp_send_nbx_callback_t) am_send_cb;

  /* The server xmits RDMA SEND */
  request =
      (ucp_test_req_t *) ucp_am_send_nbx (ep, EVENT_AM_ID, NULL, 0, msg,
      msg_len, &param);
  return request_finalize (ucp_worker, request, &ctx);
}

ucs_status_t
gst_ucp_send_only (ucp_worker_h ucp_worker, ucp_ep_h ep, unsigned int am_tag)
{
  ucp_request_param_t param;
  ucp_test_req_t *request;
  ucp_test_req_t ctx;
  ctx.complete = 0;

  param.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA;

  param.user_data = &ctx;
  param.cb.send = (ucp_send_nbx_callback_t) am_send_cb;

  request =
      (ucp_test_req_t *) ucp_am_send_nbx (ep, am_tag, NULL, 0, NULL, 0, &param);

  return request_finalize (ucp_worker, request, &ctx);
}

ucs_status_t
gst_ucp_send (ucp_worker_h ucp_worker, ucp_ep_h ep, GstNvDsUcxAmId am_id,
    void *am_hdr_buf, size_t am_hdr_buf_size, void *rdma_buffer,
    size_t rdma_buffer_size)
{
  ucp_request_param_t param;
  ucp_test_req_t *request;
  ucp_test_req_t ctx;

  ctx.complete = 0;

  param.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA;

  if (rdma_buffer_size != 0) {
    param.op_attr_mask |= UCP_OP_ATTR_FIELD_FLAGS;
    param.flags = UCP_AM_SEND_FLAG_RNDV;
  }

  param.user_data = &ctx;
  param.cb.send = (ucp_send_nbx_callback_t) am_send_cb;

  /* The server xmits RDMA SEND with RNDV bit and a header message */
  request =
      (ucp_test_req_t *) ucp_am_send_nbx (ep, am_id, am_hdr_buf,
      am_hdr_buf_size, rdma_buffer, rdma_buffer_size, &param);

  return request_finalize (ucp_worker, request, &ctx);
}

ucs_status_t
gst_ucp_recv (ucp_worker_h ucp_worker,
    GstBuffer ** gst_buffer, NvDsUcxSrcPrepAmFunc prep_am,
    NvDsUcxAmDesc * am_desc, GMutex * meta_lock,
    ucs_status_t *conn_status)
{
  ucp_request_param_t param;
  ucp_test_req_t *request;
  ucp_test_req_t ctx;
  void *rdma_data_ptr;
  ucs_status_t ret;

  if (!am_desc) {
    GST_ERROR ("missing am desc batch_id");
    return UCS_ERR_NO_RESOURCE;
  }

  /*
   * When the server first SEND arrives, gst_ucp_am_src_cb
   * is triggered and sets NvDsUcxAmDesc.complete
   */
  while ((!am_desc->complete) && (*conn_status == UCS_OK)) {
    ucp_worker_progress (ucp_worker);
  }

  if (*conn_status != UCS_OK) {
    return *conn_status;
  }

  if (am_desc->complete == 2) {
    GST_DEBUG ("exit on batch_id > numFilled");
    am_desc->complete = 0;
    return UCS_OK;
  }

  if (am_desc->desc == NULL && am_desc->length == 0) {
    am_desc->complete = 0;
    return UCS_ERR_CONNECTION_RESET;
  }
  ret = prep_am (gst_buffer, &rdma_data_ptr, &am_desc->nv_dsucx_buf, meta_lock);
  if (ret != UCS_OK) {
    GST_DEBUG ("prep_am failed");
    am_desc->complete = 0;
    return ret;
  }

  am_desc->complete = 0;
  /*
   * The src executes RDMA READ
   * source = NvDsUcxAmDesc.desc (buffer descriptor at the server)
   * destination = buf (local buffer)
   * length = NvDsUcxAmDesc.length
   */
  ctx.complete = 0;
  param.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                       UCP_OP_ATTR_FIELD_USER_DATA |
                       UCP_OP_ATTR_FLAG_NO_IMM_CMPL;
  param.user_data = &ctx;
  param.cb.recv_am = am_recv_cb;

  request =
      (ucp_test_req_t *) ucp_am_recv_data_nbx (ucp_worker, am_desc->desc,
      rdma_data_ptr, am_desc->length, &param);
  ret = request_finalize (ucp_worker, request, &ctx);

  GST_DEBUG
      ("recv data on rdma_data_ptr %p length %ld with status %s",
      rdma_data_ptr, am_desc->length, ucs_status_string (ret));

  return ret;
}

ucs_status_t
gst_ucp_recv_meta (ucp_worker_h ucp_worker,
    GstBuffer ** gst_buffer, NvDsUcxSrcPrepAmMetaFunc prep_am_meta,
    NvDsUcxAmDesc * am_desc, ucs_status_t *conn_status)
{
  ucp_request_param_t param;
  ucp_test_req_t *request;
  ucp_test_req_t ctx;
  void *rdma_data_ptr;
  ucs_status_t ret;

  if (!am_desc) {
    GST_ERROR ("missing am desc in gst_ucp_recv_meta");
    return UCS_ERR_NO_RESOURCE;
  }

  /*
   * When the server first SEND arrives, gst_ucp_am_src_cb
   * is triggered and sets NvDsUcxAmDesc.complete
   */
  while ((!am_desc->complete) && (*conn_status == UCS_OK)) {
    ucp_worker_progress (ucp_worker);
  }

  if (*conn_status != UCS_OK) {
    return *conn_status;
  }

  if (am_desc->desc == NULL && am_desc->length == 0) {
    am_desc->complete = 0;
    return UCS_ERR_CONNECTION_RESET;
  }

  ret = prep_am_meta (gst_buffer, &rdma_data_ptr, am_desc->nv_dscustom_buf);
  if (ret != UCS_OK) {
    am_desc->complete = 0;
    return ret;
  }

  am_desc->complete = 0;

  /*
   * The src executes RDMA READ
   * source = NvDsUcxAmDesc.desc (buffer descriptor at the server)
   * destination = buf (local buffer)
   * length = NvDsUcxAmDesc.length
   */
  ctx.complete = 0;
  param.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                       UCP_OP_ATTR_FIELD_USER_DATA |
                       UCP_OP_ATTR_FLAG_NO_IMM_CMPL;
  param.user_data = &ctx;
  param.cb.recv_am = am_recv_cb;
  request =
      (ucp_test_req_t *) ucp_am_recv_data_nbx (ucp_worker, am_desc->desc,
      rdma_data_ptr, am_desc->length, &param);

  ret = request_finalize (ucp_worker, request, &ctx);

  GST_DEBUG
      ("recv metadata rdma_data_ptr %p length %ld with status %s",
      rdma_data_ptr, am_desc->length, ucs_status_string (ret));

  return ret;
}


/* GstBuffer copying */
static void
copy_gstbuf_info (NvGstBufferParams * inNvGst,
    GstBuffer * inGstBuf, NvGstBufferParams * outNvGst, GstBuffer * outGstBuf)
{
  if (inNvGst) {
    if (outNvGst) {
      memcpy (outNvGst, inNvGst, sizeof (*outNvGst));
    } else if (outGstBuf) {
      outGstBuf->pts = inNvGst->pts;
      outGstBuf->dts = inNvGst->dts;
      outGstBuf->duration = inNvGst->duration;
      outGstBuf->offset = inNvGst->offset;
      outGstBuf->offset_end = inNvGst->offset_end;
    }
  } else if (inGstBuf && outNvGst) {
    outNvGst->pts = inGstBuf->pts;
    outNvGst->dts = inGstBuf->dts;
    outNvGst->duration = inGstBuf->duration;
    outNvGst->offset = inGstBuf->offset;
    outNvGst->offset_end = inGstBuf->offset_end;
  }
}

/* NvBufSurface copying */
static void
copy_nvbufsurf_params (NvBufSurfaceParams * in, NvBufSurfaceParams * out)
{
  out->width = in->width;
  out->height = in->height;
  out->pitch = in->pitch;
  out->colorFormat = in->colorFormat;
  out->layout = in->layout;
  out->dataSize = in->dataSize;

  out->planeParams.num_planes = in->planeParams.num_planes;
  for (int j = 0; j < in->planeParams.num_planes; j++) {
    out->planeParams.width[j] = in->planeParams.width[j];
    out->planeParams.height[j] = in->planeParams.height[j];
    out->planeParams.pitch[j] = in->planeParams.pitch[j];
    out->planeParams.offset[j] = in->planeParams.offset[j];
    out->planeParams.psize[j] = in->planeParams.psize[j];
    out->planeParams.bytesPerPix[j] = in->planeParams.bytesPerPix[j];
  }

  return;
}

/* NvBufAudio copying */
static void
copy_nvbufaudio_params (NvBufAudioParams * in, NvBufAudioParams * out)
{
  out->layout = in->layout;
  out->format = in->format;
  out->bpf = in->bpf;
  out->channels = in->channels;
  out->rate = in->rate;
  out->dataSize = in->dataSize;
  out->padId = in->padId;
  out->sourceId = in->sourceId;
  out->ntpTimestamp = in->ntpTimestamp;
  out->bufPts = in->bufPts;
  out->duration = in->duration;
}

/* GstUcx plugin stuff */

bool
gst_ucx_server_set_property (GstUcxServer * ucxserver, guint property_id,
    const GValue * value, GParamSpec * pspec)
{
  bool ret = true;

  switch (property_id) {
    case PROP_ADDR:
      if (!g_value_get_string (value)) {
        g_warning ("host property cannot be NULL");
        break;
      }
      g_free (ucxserver->addr);
      ucxserver->addr = g_value_dup_string (value);
      break;
    case PROP_PORT:
      ucxserver->port = g_value_get_int (value);
      break;
    case PROP_NUM_CONNS:
      ucxserver->num_conns = g_value_get_uint (value);
      break;
    case PROP_MAX_EP_NUM:
      ucxserver->max_ep_num = g_value_get_uint (value);
      break;
    case PROP_HIGH_PERF:
      ucxserver->high_perf = g_value_get_boolean (value);
      break;
    default:
      ret = false;
      break;
  }

  return ret;
}

bool
gst_ucx_client_set_property (GstUcxClient * ucxclient, guint property_id,
    const GValue * value, GParamSpec * pspec)
{
  bool ret = true;

  switch (property_id) {
    case PROP_ADDR:
      if (!g_value_get_string (value)) {
        g_warning ("host property cannot be NULL");
        break;
      }
      g_free (ucxclient->addr);
      ucxclient->addr = g_value_dup_string (value);
      break;
    case PROP_PORT:
      ucxclient->port = g_value_get_int (value);
      break;
    case PROP_MAX_EP_NUM:
      ucxclient->max_ep_num = g_value_get_uint (value);
      break;
    case PROP_HIGH_PERF:
      ucxclient->high_perf = g_value_get_boolean (value);
      break;
    default:
      ret = false;
      break;
  }

  return ret;
}

bool
gst_ucx_src_set_property (GstUcxSrc * ucxsrc, guint property_id,
    const GValue * value, GParamSpec * pspec)
{
  bool ret = true;
  switch (property_id) {
    case PROP_BUF_TYPE:
      ucxsrc->buf_type = (GstNvDsUcxBufType) g_value_get_enum (value);
      break;
    case PROP_GPU_DEVICE_ID:
      ucxsrc->gpu_id = g_value_get_uint (value);
      break;
    case PROP_RAW_BUF_SIZE:
      ucxsrc->raw_buf_size = g_value_get_uint (value);
      break;
    case PROP_NVBUF_MEMORY_TYPE:
      ucxsrc->mem_type = (NvBufSurfaceMemType) (g_value_get_enum (value));
      break;
    case PROP_NUM_NVBUF:
      ucxsrc->num_nvbuf = g_value_get_int (value);
      break;
    case PROP_NVBUF_BATCH_SIZE:
      ucxsrc->nvbuf_batch_size = g_value_get_int (value);
      break;
    default:
      ret = false;
      break;
  }

  return ret;
}

bool
gst_ucx_sink_set_property (GstUcxSink * ucxsink, guint property_id,
    const GValue * value, GParamSpec * pspec)
{
  bool ret = true;

  switch (property_id) {
    case PROP_BUF_TYPE:
      ucxsink->buf_type = (GstNvDsUcxBufType) g_value_get_enum (value);
      break;
    case PROP_GPU_DEVICE_ID:
      ucxsink->gpu_id = g_value_get_uint (value);
      break;
    case PROP_NVBUF_BATCH_SIZE:
      ucxsink->nvbuf_batch_size = g_value_get_int (value);
      break;
    default:
      ret = false;
      break;
  }

  return ret;
}

bool
gst_ucx_server_get_property (GstUcxServer * ucxserver, guint property_id,
    GValue * value, GParamSpec * pspec)
{
  bool ret = true;

  switch (property_id) {
    case PROP_ADDR:
      g_value_set_string (value, ucxserver->addr);
      break;
    case PROP_PORT:
      g_value_set_int (value, ucxserver->port);
      break;
    case PROP_NUM_CONNS:
      g_value_set_uint (value, ucxserver->num_conns);
      break;
    case PROP_MAX_EP_NUM:
      g_value_set_uint (value, ucxserver->max_ep_num);
      break;
    case PROP_HIGH_PERF:
      g_value_set_boolean (value, ucxserver->high_perf);
      break;
    default:
      ret = false;
      break;
  }

  return ret;
}

bool
gst_ucx_sink_get_property (GstUcxSink * ucxsink, guint property_id,
    GValue * value, GParamSpec * pspec)
{
  bool ret = true;

  switch (property_id) {
    case PROP_BUF_TYPE:
      g_value_set_enum (value, ucxsink->buf_type);
      break;
    case PROP_GPU_DEVICE_ID:
      g_value_set_uint (value, ucxsink->gpu_id);
      break;
    case PROP_NVBUF_BATCH_SIZE:
      g_value_set_int (value, ucxsink->nvbuf_batch_size);
      break;
    default:
      ret = false;
      break;
  }

  return ret;
}

bool
gst_ucx_client_get_property (GstUcxClient * ucxclient, guint property_id,
    GValue * value, GParamSpec * pspec)
{
  bool ret = true;

  switch (property_id) {
    case PROP_ADDR:
      g_value_set_string (value, ucxclient->addr);
      break;
    case PROP_PORT:
      g_value_set_int (value, ucxclient->port);
      break;
    case PROP_MAX_EP_NUM:
      g_value_set_uint (value, ucxclient->max_ep_num);
      break;
    case PROP_HIGH_PERF:
      g_value_set_boolean (value, ucxclient->high_perf);
      break;
    default:
      ret = false;
      break;
  }

  return ret;
}

bool
gst_ucx_src_get_property (GstUcxSrc * ucxsrc, guint property_id,
    GValue * value, GParamSpec * pspec)
{
  bool ret = true;

  switch (property_id) {
    case PROP_BUF_TYPE:
      g_value_set_enum (value, ucxsrc->buf_type);
      break;
    case PROP_GPU_DEVICE_ID:
      g_value_set_uint (value, ucxsrc->gpu_id);
      break;
    case PROP_RAW_BUF_SIZE:
      g_value_set_uint (value, ucxsrc->raw_buf_size);
      break;
    case PROP_NVBUF_MEMORY_TYPE:
      g_value_set_enum (value, ucxsrc->mem_type);
      break;
    case PROP_NUM_NVBUF:
      g_value_set_int (value, ucxsrc->num_nvbuf);
      break;
    case PROP_NVBUF_BATCH_SIZE:
      g_value_set_int (value, ucxsrc->nvbuf_batch_size);
      break;
    default:
      ret = false;
      break;
  }

  return ret;
}

void
gst_ucx_init ()
{
  GST_DEBUG_CATEGORY_INIT (gst_ucx_common_debug_category,
      "ucxcommon", 0, "NET common");
}

void
gst_ucx_sink_init (GstUcxSink * ucxsink)
{
  ucxsink->nvbuf_batch_size = DEFAULT_NVBUF_BATCH_SIZE;
  ucxsink->gpu_id = DEFAULT_GPU_ID;
  return;
}

void
gst_ucx_src_init (GstUcxSrc * ucxsrc)
{
  ucxsrc->gpu_id = DEFAULT_GPU_ID;
  ucxsrc->mem_type = DEFAULT_NVBUF_MEMORY_TYPE;
  ucxsrc->num_nvbuf = DEFAULT_NUM_NVBUF;
  ucxsrc->nvbuf_batch_size = DEFAULT_NVBUF_BATCH_SIZE;
  ucxsrc->raw_buf_size = DEFAULT_RAW_BUFFER_SIZE;
}

void
gst_ucx_server_init (GstUcxServer * ucxserver)
{
  ucxserver->port = UCX_DEFAULT_PORT;
  ucxserver->addr = g_strdup (UCX_DEFAULT_ADDR);
  ucxserver->num_conns = UCX_DEFAULT_CONNS;
  ucxserver->max_ep_num = DEFAULT_MAX_EP_NUM;
  ucxserver->high_perf = UCX_DEFAULT_HIGH_PERF;
}

void
gst_ucx_client_init (GstUcxClient * ucxclient)
{
  ucxclient->addr = g_strdup (UCX_DEFAULT_ADDR);
  ucxclient->port = UCX_DEFAULT_PORT;
  ucxclient->max_ep_num = DEFAULT_MAX_EP_NUM;
  ucxclient->high_perf = UCX_DEFAULT_HIGH_PERF;
}

gboolean
gst_ucx_src_negotiate_caps (GstBaseSrc * src, GstUcxSrc * ucxsrc)
{
  GstCaps *thiscaps = NULL;
  GstCaps *peercaps = NULL;
  GstCaps *intercaps = NULL;
  gboolean res;

  thiscaps = gst_pad_query_caps (GST_BASE_SRC_PAD (src), NULL);
  if (thiscaps == NULL) {
    GST_WARNING ("thiscaps is NULL");
    return FALSE;
  }

  peercaps = gst_pad_peer_query_caps (GST_BASE_SRC_PAD (src), NULL);
  if (peercaps == NULL) {
    GST_WARNING ("peercaps is NULL");
    return FALSE;
  }

  intercaps = gst_caps_intersect (thiscaps, peercaps);
  gst_caps_unref (thiscaps);
  gst_caps_unref (peercaps);

  if (!intercaps) {
    GST_WARNING ("intercaps is NULL");
    return FALSE;
  }

  /* Set the framerate to defult value, this value will be ovveride by sender
   * caps in handle caps*/
  gst_caps_set_simple (intercaps, "framerate", GST_TYPE_FRACTION, 30, 1, NULL);
  GST_DEBUG ("intercaps %s", gst_caps_to_string (intercaps));

  res = gst_ucx_src_buffer_pool_new (ucxsrc, intercaps);

  gst_caps_unref (intercaps);

  return res;
}

gboolean
gst_ucx_sink_cap_event_handling (GstUcxSink * sink, GstBaseSink * basesink,
    GstEvent * event, NvDsUcxAmEventDesc * event_data, gboolean * send_event)
{
  GstCaps *caps = NULL;
  gboolean found_surf = FALSE;
  gst_event_parse_caps (event, &caps);
  if (caps) {
    gchar *caps_str = gst_caps_to_string (caps);
    gint msg_len = strlen (caps_str) + 1;
    if (msg_len > EVENT_DATA_MAX_LEN) {
      GST_ERROR ("Caps string length %d longer than supported length %d",
          msg_len, EVENT_DATA_MAX_LEN);
      return FALSE;
    }

    strcpy (event_data->data, caps_str);
    event_data->type = GST_EVENT_CAPS;
    GST_DEBUG ("incoming sink caps = [%s]\n len =%d", caps_str, msg_len);
    GstStructure *str = gst_caps_get_structure (caps, 0);
    gint num_surfaces_per_frame;
    if (gst_structure_get_int (str, "num-surfaces-per-frame",
        &num_surfaces_per_frame)) {
      sink->num_surfaces_per_frame = num_surfaces_per_frame;
      GST_DEBUG ("num surfaces per frame from caps: %d",
          sink->num_surfaces_per_frame);
      found_surf = TRUE;
    }
  } else {
    *send_event = FALSE;
  }

  if (!found_surf) {
    GstQuery *query = gst_nvquery_num_surfaces_per_buffer_new ();
    if (gst_pad_peer_query (GST_BASE_SINK_PAD (basesink), query)) {
      gst_nvquery_num_surfaces_per_buffer_parse (query,
          &sink->num_surfaces_per_frame);
      GST_DEBUG ("num surfaces per frame query: %d",
          sink->num_surfaces_per_frame);
    } else {
      sink->num_surfaces_per_frame = 1;
    }
    gst_query_unref (query);
  }
  return TRUE;
}

void
gst_ucx_set_nvevent_data (GstEvent * event, NvDsUcxAmEventDesc * event_data,
    gboolean * send_event)
{
  guint source_id;

  *send_event = TRUE;

  switch GST_EVENT_TYPE
    (event) {
    case GST_NVEVENT_PAD_ADDED:
      gst_nvevent_parse_pad_added (event, &source_id);
      GST_DEBUG ("Pad added for source: %d", source_id);
      break;
    case GST_NVEVENT_PAD_DELETED:
      gst_nvevent_parse_pad_deleted (event, &source_id);
      GST_DEBUG ("Pad deleted for source: %d", source_id);
      break;
    case GST_NVEVENT_STREAM_EOS:
      gst_nvevent_parse_stream_eos (event, &source_id);
      GST_DEBUG ("Stream eos for source: %d", source_id);
      break;
    case GST_NVEVENT_STREAM_RESET:
      gst_nvevent_parse_stream_reset (event, &source_id);
      GST_DEBUG ("Stream reset for source: %d", source_id);
      break;
    case GST_NVEVENT_STREAM_SEGMENT:
      GstSegment * segment;
      gst_nvevent_parse_stream_segment (event, &source_id, &segment);
      memcpy (&event_data->segment, segment, sizeof (*segment));
      GST_DEBUG ("Stream segment for source: %d", source_id);
      break;
    case GST_NVEVENT_STREAM_START:
      gchar * stream_id;
      gst_nvevent_parse_stream_start (event, &source_id, &stream_id);
      strncpy(event_data->data, stream_id, sizeof(event_data->data) - 1);
      GST_DEBUG ("Stream start for source: %d stream: %s",
          source_id, stream_id);
      break;
    default:
      GST_DEBUG ("Unsupported event received: %d", GST_EVENT_TYPE (event));
      *send_event = FALSE;
      event_data = { 0 };
    }

  if (*send_event) {
    event_data->type = GST_EVENT_TYPE (event);
    event_data->sourceId = source_id;
  }
}

static const char *
sockaddr_get_ip_str (const struct sockaddr_storage *sock_addr,
    char *ip_str, size_t max_size)
{
  struct sockaddr_in addr_in;
  struct sockaddr_in6 addr_in6;
  switch (sock_addr->ss_family) {
    case AF_INET:
      memcpy (&addr_in, sock_addr, sizeof (struct sockaddr_in));
      inet_ntop (AF_INET, &addr_in.sin_addr, ip_str, max_size);
      return ip_str;
    case AF_INET6:
      memcpy (&addr_in6, sock_addr, sizeof (struct sockaddr_in6));
      inet_ntop (AF_INET6, &addr_in6.sin6_addr, ip_str, max_size);
      return ip_str;
    default:
      return "Invalid address family";
  }
}

static const char *
sockaddr_get_port_str (const struct sockaddr_storage *sock_addr,
    char *port_str, size_t max_size)
{
  struct sockaddr_in addr_in;
  struct sockaddr_in6 addr_in6;
  switch (sock_addr->ss_family) {
    case AF_INET:
      memcpy (&addr_in, sock_addr, sizeof (struct sockaddr_in));
      snprintf (port_str, max_size, "%d", ntohs (addr_in.sin_port));
      return port_str;
    case AF_INET6:
      memcpy (&addr_in6, sock_addr, sizeof (struct sockaddr_in6));
      snprintf (port_str, max_size, "%d", ntohs (addr_in6.sin6_port));
      return port_str;
    default:
      return "Invalid address family";
  }
}

/* GstUcxSink */

int
gst_ucx_sink_prep_am_video (GstBuffer * gst_buffer,
    GstMapInfo * map_info, void **am_hdr_buf, void **rdma_buffer,
    size_t *rdma_buffer_size, NvDsBatchMeta * batch_meta, int batch_id,
    guint num_surfaces_per_frame, size_t rdma_meta_buffer_size)
{
  NvBufSurface *nvbufsurf;
  NvBufSurfaceDsUcx *nvbufsurfdsucx;

  nvbufsurf = (NvBufSurface *) map_info->data;

  nvbufsurfdsucx = (NvBufSurfaceDsUcx *) (*am_hdr_buf);

  copy_gstbuf_info (NULL, gst_buffer, &nvbufsurfdsucx->nvgstbuffer_params,
      NULL);

  nvbufsurfdsucx->numFilled = nvbufsurf->numFilled;
  nvbufsurfdsucx->batchSize = nvbufsurf->batchSize;
  nvbufsurfdsucx->batchId = batch_id;

  if (batch_id < nvbufsurf->numFilled) {
    copy_nvbufsurf_params (&nvbufsurf->surfaceList[batch_id],
        &nvbufsurfdsucx->nvbufsurf_params);
    *rdma_buffer = nvbufsurf->surfaceList[batch_id].dataPtr;
    *rdma_buffer_size = nvbufsurf->surfaceList[batch_id].dataSize;
  }
  if (batch_meta != NULL) {
    nvbufsurfdsucx->num_surfaces_per_frame = num_surfaces_per_frame;
  } else {
    /*
     * This = 0 indicates an absence of batch metadata to the client
     * not an absence of video frames
     */
    nvbufsurfdsucx->num_surfaces_per_frame = 0;
  }
  nvbufsurfdsucx->has_user_meta = (rdma_meta_buffer_size > 0 && batch_id == 0);

  GST_DEBUG
      ("batchId %d out of numFilled %d - rdma_buffer %p, "
      "rdma_buffer_size %ld has_user_meta %d",
      batch_id, nvbufsurf->numFilled, *rdma_buffer, *rdma_buffer_size,
      nvbufsurfdsucx->has_user_meta);
  return 0;
}

int
gst_ucx_sink_prep_am_meta (void **am_hdr_meta_buf, void **rdma_buffer,
    size_t *rdma_buffer_size, NvDsBatchMeta * batch_meta, int batch_id)
{
  NvDsUserMetaList *bMetaList;
  NvDsUserMeta *user_meta = NULL;
  NVDS_CUSTOM_PAYLOAD *metadata = NULL;
  NvDsCustomMetaUcx *metaucx;

  metaucx = (NvDsCustomMetaUcx *) (*am_hdr_meta_buf);

  for (bMetaList = (NvDsUserMetaList *) batch_meta->batch_user_meta_list;
      bMetaList != NULL;
      bMetaList = bMetaList->next) {
    if (!bMetaList->data) {
      GST_ERROR ("Metadata is not serialized correctly batch_id %d", batch_id);
      return 1;
    }
    user_meta = (NvDsUserMeta *)bMetaList->data;

    if (user_meta->base_meta.meta_type == NVDS_USER_CUSTOM_META) {
      metadata = (NVDS_CUSTOM_PAYLOAD *) user_meta->user_meta_data;
      metaucx->payloadType = metadata->payloadType;
      metaucx->payloadSize = metadata->payloadSize;
      metaucx->batchId = batch_id;
      *rdma_buffer = metadata->payload;
      *rdma_buffer_size = metadata->payloadSize;
      GST_DEBUG ("batchId %d payloadType %X payloadSize %d rdma_buffer %p rdma_buffer_size %ld",
      batch_id, metaucx->payloadType, metaucx->payloadSize, *rdma_buffer,
      *rdma_buffer_size);
      return 0;
    }
  }

  *rdma_buffer = NULL;
  *rdma_buffer_size = 0;
  GST_DEBUG ("Metadata not found");
  return 0;
}

static int
gst_ucx_sink_prep_am_raw (GstBuffer * gst_buffer, GstMapInfo * map_info,
    void **am_hdr_buf, void **rdma_buffer,
    size_t *rdma_buffer_size, NvDsBatchMeta * batch_meta, int batch_id,
    guint num_surfaces_per_frame, size_t rdma_meta_buffer_size)
{
  NvBufRawDsUcx *nvbufraw = (NvBufRawDsUcx *) (*am_hdr_buf);

  *rdma_buffer = (void *) map_info->data;
  *rdma_buffer_size = map_info->size;

  copy_gstbuf_info (NULL, gst_buffer, &nvbufraw->nvgstbuffer_params, NULL);
  nvbufraw->size = *rdma_buffer_size;

  return 0;
}

int
gst_ucx_sink_prep_am_audio_nv (GstBuffer * gst_buffer,
    GstMapInfo * map_info, void **am_hdr_buf,
    void **rdma_buffer, size_t *rdma_buffer_size,
    NvDsBatchMeta * batch_meta, int batch_id, guint num_surfaces_per_frame,
    size_t rdma_meta_buffer_size)
{
  NvBufAudio *nvbufaudio;
  NvBufAudioDsUcx *nvbufaudiodsucx;

  nvbufaudio = (NvBufAudio *) map_info->data;

  nvbufaudiodsucx = (NvBufAudioDsUcx *) (*am_hdr_buf);

  copy_gstbuf_info (NULL, gst_buffer, &nvbufaudiodsucx->nvgstbuffer_params,
      NULL);

  nvbufaudiodsucx->numFilled = nvbufaudio->numFilled;
  nvbufaudiodsucx->batchSize = nvbufaudio->batchSize;
  nvbufaudiodsucx->batchId = batch_id;

  if (batch_id < nvbufaudio->numFilled) {
    copy_nvbufaudio_params (&nvbufaudio->audioBuffers[batch_id],
        &nvbufaudiodsucx->nvbufaudio_params);
    *rdma_buffer = nvbufaudio->audioBuffers[batch_id].dataPtr;
    *rdma_buffer_size = nvbufaudio->audioBuffers[batch_id].dataSize;
  }
  if (batch_meta != NULL) {
    nvbufaudiodsucx->num_surfaces_per_frame = num_surfaces_per_frame;
  } else {
    /*
     * This = 0 indicates an absence of batch metadata to the client
     * not an absence of video frames
     */
    nvbufaudiodsucx->num_surfaces_per_frame = 0;
  }
  nvbufaudiodsucx->has_user_meta = (rdma_meta_buffer_size > 0 && batch_id == 0);

  GST_DEBUG
      ("batchId %d out of numFilled %d - rdma_buffer %p, rdma_buffer_size %ld has_user_meta %d",
      batch_id, nvbufaudio->numFilled, *rdma_buffer, *rdma_buffer_size,
      nvbufaudiodsucx->has_user_meta);
  return 0;
}


static uint64_t
get_client_id_from_conn_req (ucp_conn_request_h conn_request)
{
  ucp_conn_request_attr_t attr;
  attr.field_mask = UCP_CONN_REQUEST_ATTR_FIELD_CLIENT_ID;
  if (UCS_OK != ucp_conn_request_query (conn_request, &attr)) {
    GST_ERROR ("failed to query the connection request");
    return -1;
  }
  return attr.client_id;
}

/* GstUcxServer */
static gst_ucp_conn_ctx_t *
server_get_conn_from_list (GList * gst_ucx_server_conn_list, uint64_t client_id)
{
  gst_ucp_conn_ctx_t *conn;
  GList *l;
  if (!gst_ucx_server_conn_list)
    return NULL;

  for (l = gst_ucx_server_conn_list; l != NULL; l = l->next) {
    conn = (gst_ucp_conn_ctx_t *) l->data;
    if (conn->client_id == (client_id >> CLIENT_ID_SIZE)) {
      GST_DEBUG ("Found connection with the same client_id");
      return conn;
    }
  }
  return NULL;
}

static GList *
server_get_glist_conn_from_list (GList * gst_ucx_server_conn_list, uint64_t client_id)
{
  gst_ucp_conn_ctx_t *conn;
  GList *l;
  if (!gst_ucx_server_conn_list)
    return NULL;

  for (l = gst_ucx_server_conn_list; l != NULL; l = l->next) {
    conn = (gst_ucp_conn_ctx_t *) l->data;
    if (conn->client_id == (client_id >> CLIENT_ID_SIZE)) {
      GST_DEBUG ("Found connection with the same client_id");
      return l;
    }
  }
  return NULL;
}

static void
server_conn_handle_cb (ucp_conn_request_h conn_request, void *arg)
{
  gst_ucx_conn_req_ctx_t *context = (gst_ucx_conn_req_ctx_t *) arg;
  ucp_conn_request_attr_t attr;
  char ip_str[IP_STRING_LEN];
  char port_str[PORT_STRING_LEN];
  ucs_status_t status;

  attr.field_mask =
      UCP_CONN_REQUEST_ATTR_FIELD_CLIENT_ADDR |
      UCP_CONN_REQUEST_ATTR_FIELD_CLIENT_ID;
  status = ucp_conn_request_query (conn_request, &attr);
  if (status == UCS_OK) {
    GST_DEBUG
        ("Server received a connection request from client at address %s:%s client_id %ld",
        sockaddr_get_ip_str (&attr.client_address, ip_str, sizeof (ip_str)),
        sockaddr_get_port_str (&attr.client_address, port_str,
            sizeof (port_str)), attr.client_id);
  } else if (status != UCS_ERR_UNSUPPORTED) {
    GST_WARNING ("failed to query the connection request (%s)",
        ucs_status_string (status));
  }

  g_mutex_lock (&context->req_lock);
  if (g_list_length (context->gst_ucx_conn_req_list) < MAX_OUT_CONNS) {
    context->gst_ucx_conn_req_list =
        g_list_prepend (context->gst_ucx_conn_req_list, conn_request);
  } else {
    GST_DEBUG ("Rejecting a connection request. Too many pending requests");
    status = ucp_listener_reject (context->listener, conn_request);
    if (status != UCS_OK) {
      GST_ERROR ("server failed to reject a connection request: (%s)",
          ucs_status_string (status));
    }
  }
  g_mutex_unlock (&context->req_lock);
}

static void
set_listen_addr (const char *address_str, int port,
    struct sockaddr_in *listen_addr)
{
  /* The server will listen on INADDR_ANY */
  memset (listen_addr, 0, sizeof (struct sockaddr_in));
  listen_addr->sin_family = AF_INET;
  listen_addr->sin_addr.s_addr =
      (address_str) ? inet_addr (address_str) : INADDR_ANY;
  listen_addr->sin_port = htons (port);
}

static ucs_status_t
start_server_listener (ucp_worker_h ucp_worker,
    gst_ucx_conn_req_ctx_t * context, gchar * ip, int port)
{
  struct sockaddr_in listen_addr;
  ucp_listener_params_t params;
  ucp_listener_attr_t attr;
  ucs_status_t status;
  char ip_str[IP_STRING_LEN];
  char port_str[PORT_STRING_LEN];

  set_listen_addr ((const char *) ip, port, &listen_addr);

  params.field_mask = UCP_LISTENER_PARAM_FIELD_SOCK_ADDR |
      UCP_LISTENER_PARAM_FIELD_CONN_HANDLER;
  params.sockaddr.addr = (const struct sockaddr *) &listen_addr;
  params.sockaddr.addrlen = sizeof (listen_addr);
  params.conn_handler.cb = server_conn_handle_cb;
  params.conn_handler.arg = context;

  /* Create a listener on the server side to listen on the given address. */
  status = ucp_listener_create (ucp_worker, &params, &context->listener);
  if (status != UCS_OK) {
    GST_ERROR ("failed to listen (%s)", ucs_status_string (status));
    goto out;
  }

  /* Query the created listener to get the port it is listening on. */
  attr.field_mask = UCP_LISTENER_ATTR_FIELD_SOCKADDR;
  status = ucp_listener_query (context->listener, &attr);
  if (status != UCS_OK) {
    GST_ERROR ("failed to query the listener (%s)",
        ucs_status_string (status));
    ucp_listener_destroy (context->listener);
    goto out;
  }

  GST_DEBUG ("Server is listening on IP %s port %s",
      sockaddr_get_ip_str (&attr.sockaddr, ip_str, IP_STRING_LEN),
      sockaddr_get_port_str (&attr.sockaddr, port_str, PORT_STRING_LEN));

  GST_DEBUG ("Waiting for connection...");
out:
  return status;
}

static void
handle_peer_conn_reset (GstUcxServer * ucxserver, ucp_ep_h ep)
{
  GList *l;
  GList *next;
  gst_ucp_conn_ctx_t *conn = NULL;      // Make compilers happy
  int i;

  l = ucxserver->priv->gst_ucx_server_conn_list;
  while (l != NULL) {
    next = l->next;
    conn = (gst_ucp_conn_ctx_t *) l->data;
    for (i = 0; i < conn->ep_num; i++) {
      if (conn->ep[i] != ep)
        continue;
      clear_gst_ucp_conn_ctx(l);
      ucxserver->priv->gst_ucx_server_conn_list =
          g_list_delete_link (ucxserver->priv->gst_ucx_server_conn_list, l);
      free(l);
      break;
    }
    l = next;
  }
  return;
}

static void
server_err_cb (void *arg, ucp_ep_h ep, ucs_status_t status)
{
  gst_ucp_conn_ctx_t * conn;
  int i;

  if (!arg) {
    GST_ERROR ("arg is empty");
    return;
  }
  conn = (gst_ucp_conn_ctx_t *) arg;

  GST_DEBUG ("ep invoked with status %d (%s)", status,
      ucs_status_string (status));

  conn->err = status;
  return;
}


void query_ep(ucp_ep_h ep)
{

  ucp_ep_attr_t attr;
  char ip_str[IP_STRING_LEN];
  char port_str[PORT_STRING_LEN];
  char ip_str2[IP_STRING_LEN];
  char port_str2[PORT_STRING_LEN];
  ucs_status_t status;

  attr.field_mask = UCP_EP_ATTR_FIELD_LOCAL_SOCKADDR |
                       UCP_EP_ATTR_FIELD_REMOTE_SOCKADDR;
  status = ucp_ep_query(ep, &attr);
  if (status != UCS_OK) {
     GST_ERROR ("EP QUERY FAILED");
  } else {
    GST_DEBUG ("EP local %s:%s     remote %s:%s",
        sockaddr_get_ip_str (&attr.local_sockaddr, ip_str, sizeof (ip_str)),
        sockaddr_get_port_str (&attr.local_sockaddr, port_str, sizeof (port_str)),
        sockaddr_get_ip_str (&attr.remote_sockaddr, ip_str2, sizeof (ip_str2)),
        sockaddr_get_port_str (&attr.remote_sockaddr, port_str2, sizeof (port_str2)));
  }
}

ucs_status_t blocking_ep_flush (gst_ucp_conn_ctx_t * conn_ctx)
{
  ucp_request_param_t param;
  void *request;
  int i;
  if (!conn_ctx->ep[conn_ctx->ep_num]) {
    return UCS_ERR_NO_RESOURCE;
  }
  param.op_attr_mask = 0;
  request = ucp_ep_flush_nbx(conn_ctx->ep[conn_ctx->ep_num], &param);
  if (request == NULL) {
    return UCS_OK;
  } else if (UCS_PTR_IS_ERR(request)) {
    return UCS_PTR_STATUS(request);
  } else {
    ucs_status_t status = UCS_INPROGRESS;
    do {
      ucp_worker_progress(conn_ctx->data_worker[conn_ctx->ep_num]);
      for(i = 0; i <= conn_ctx->ep_num && status == UCS_INPROGRESS; i++) {
        ucp_worker_progress(conn_ctx->data_worker[i]);
        status = ucp_request_check_status(request);
      }
    } while (status == UCS_INPROGRESS);
    ucp_request_free(request);
    return status;
  }
  return UCS_OK;
}


static ucs_status_t
server_create_ep (gst_ucp_conn_ctx_t * conn_ctx, ucp_conn_request_h conn_request)
{
  ucp_ep_params_t ep_params;
  ucs_status_t status;

  /* Server creates an ep to the client on the data worker.
   * This is not the worker the listener was created on.
   * The client side should have initiated the connection, leading
   * to this ep's creation */
  ep_params.field_mask = UCP_EP_PARAM_FIELD_ERR_HANDLER |
                         UCP_EP_PARAM_FIELD_CONN_REQUEST;

  ep_params.conn_request = conn_request;
  ep_params.err_handler.cb = server_err_cb;
  ep_params.err_handler.arg = conn_ctx;

  status = ucp_ep_create (conn_ctx->data_worker[conn_ctx->ep_num], &ep_params,
      &conn_ctx->ep[conn_ctx->ep_num]);

  if (status != UCS_OK) {
    GST_ERROR ("failed to create an endpoint on the server: (%s)",
        ucs_status_string (status));
  }
  status = blocking_ep_flush (conn_ctx);
  if (status != UCS_OK) {
    GST_ERROR ("failed to create an endpoint on the server: (%s)",
        ucs_status_string (status));
  }

  return status;
}

static ucs_status_t
init_conn_ctx (gst_ucp_conn_ctx_t * conn_ctx, int size)
{
  GST_DEBUG ("Creating new connection");
  conn_ctx->data_worker = g_new0 (ucp_worker_h, size);
  if (!conn_ctx->data_worker) {
    GST_ERROR ("failed to alloc ucp_worker_h");
    goto out;
  }

  conn_ctx->ep = g_new0 (ucp_ep_h, size);
  if (!conn_ctx->ep) {
    GST_ERROR ("failed to alloc ucp_ep_h");
    goto ep_err;
  }

  conn_ctx->complete = 0;
  conn_ctx->ep_num = 0;
  conn_ctx->complete = 0;
  conn_ctx->err = UCS_OK;
  g_mutex_init (&conn_ctx->conn_lock);
  return UCS_OK;

ep_err:
  g_free (conn_ctx->data_worker);
out:
  return UCS_ERR_NO_RESOURCE;
}


ucs_status_t
create_worker_and_register_cbs (gpointer ucxdata, ucp_context_h ucp_context,
    gst_ucp_conn_ctx_t *conn, uint64_t client_id, bool is_src, bool is_server)
{
  if (gst_ucp_init_worker (ucp_context, &conn->data_worker[conn->ep_num],
      client_id) != UCS_OK)
    return UCS_ERR_NO_MESSAGE;
  if (is_src) {
    if (gst_ucp_init_src_am (ucxdata, conn, is_server) != UCS_OK)
      return UCS_ERR_NO_MESSAGE;
    if (gst_ucp_register_am_handler (conn->data_worker[conn->ep_num], EVENT_AM_ID,
        ucxdata, 0, gst_ucp_am_src_cb_handle_event))
        return UCS_ERR_NO_MESSAGE;
  } else {
    if(gst_ucp_register_am_handler (conn->data_worker[conn->ep_num],
        EVENT_AM_BATCH_END, conn, 0,
        gst_ucp_am_cb_server_end_batch))
      return UCS_ERR_NO_MESSAGE;
  }
   return UCS_OK;
}

static void
create_and_add_new_connection (gpointer ucxdata,
    ucp_conn_request_h conn_request, bool is_src)
{
  gst_ucp_conn_ctx_t *conn;
  int msg_len = 0;
  int thread_num;
  uint64_t client_id;
  GstUcxServer *ucxserver = get_gstucxserver (ucxdata, is_src);
  if (!ucxserver) {
    GST_ERROR ("could not get ucxserver object");
    return;
  }

  thread_num = get_thread_num (ucxdata, is_src, TRUE);
  if (thread_num == -1)
    return;

  client_id = get_client_id_from_conn_req (conn_request);
  if (client_id == -1)
    return;

  conn =
      server_get_conn_from_list (ucxserver->priv->gst_ucx_server_queued_conn_list,
      client_id);
  if (!conn) {
    conn = g_new (gst_ucp_conn_ctx_t, 1);
    if (!conn) {
      GST_ERROR ("failed to alloc gst_ucp_conn_ctx_t");
      return;
    }
    if (UCS_OK != init_conn_ctx (conn, thread_num)) {
      g_free (conn);
      return;
    }

    conn->client_id = (client_id >> CLIENT_ID_SIZE);
  }

  if (ucxserver->total_ep_num >= ucxserver->max_ep_num) {
    GST_WARNING ("exceeded number of QPs - server reject a connection request");
    if (UCS_OK !=
        ucp_listener_reject (ucxserver->priv->server_conn_ctx.listener,
            conn_request))
      GST_ERROR ("server failed to reject a connection request");
  }


  if (create_worker_and_register_cbs(ucxdata, ucxserver->priv->ucp_context, conn,
      conn->client_id, is_src, TRUE)  != UCS_OK)
      goto err_worker;

  if (server_create_ep (conn, conn_request) != UCS_OK)
    goto err_create_ep;
  ucxserver->total_ep_num++;

  GST_DEBUG ("Adding connection %p to list to index number %d", conn,
      conn->ep_num);
  conn->ep_num++;

  if (conn->ep_num == 1) {
      ucxserver->priv->gst_ucx_server_queued_conn_list =
          g_list_prepend (ucxserver->priv->gst_ucx_server_queued_conn_list, conn);
  }
  if (conn->ep_num == thread_num) {
    g_mutex_lock (&ucxserver->priv->conn_lock);
    ucxserver->priv->gst_ucx_server_conn_list =
        g_list_prepend (ucxserver->priv->gst_ucx_server_conn_list, conn);

    ucxserver->priv->gst_ucx_server_queued_conn_list =
        g_list_delete_link (ucxserver->priv->gst_ucx_server_queued_conn_list,
        server_get_glist_conn_from_list (ucxserver->priv->gst_ucx_server_queued_conn_list,
        client_id));
    g_cond_signal (&ucxserver->priv->conn_cond);
    g_mutex_unlock (&ucxserver->priv->conn_lock);
  }


  /* Stop listener thread once we reach number of required connections */
  if (!is_src && ucxserver->caps && conn->ep_num == thread_num) {
    NvDsUcxAmEventDesc event;
    GList *list_entry;

    strcpy (event.data, ucxserver->caps);
    event.type = GST_EVENT_CAPS;
    GST_DEBUG ("new connection, sending caps = [%s]", ucxserver->caps);
    gst_ucp_send_event (conn->data_worker[0], conn->ep[0], &event,
        sizeof (event));

    /* Send other previous events as well */
    for (list_entry = ucxserver->priv->gst_ucx_nvevents_list;
        list_entry != NULL; list_entry = list_entry->next) {
      gst_ucp_send_event (conn->data_worker[0], conn->ep[0],
          (NvDsUcxAmEventDesc *) list_entry->data, sizeof (NvDsUcxAmEventDesc));
    }
  }

  /* Server main thread waiting for this signal on his create function */
  return;

err_create_ep:
  ucp_worker_destroy (conn->data_worker[conn->ep_num]);
err_worker:
  if (conn->ep_num == 0)
    g_free (conn);
out:
  return;
}

static gpointer
do_listen (gpointer ucxdata, bool is_src)
{
  GList *l;
  GList *next;
  GstUcxServer *ucxserver = get_gstucxserver (ucxdata, is_src);
  if (!ucxserver) {
    GST_ERROR ("could not get ucxserver object");
    return NULL;
  }

  while (1) {
    while (g_list_length (ucxserver->priv->server_conn_ctx.
            gst_ucx_conn_req_list) == 0 && !ucxserver->stop_listener_thread) {
      ucp_worker_progress (ucxserver->priv->server_conn_ctx.listen_worker);
    }

    /* Thread was stopped */
    if (ucxserver->stop_listener_thread == 1) {
      GST_DEBUG ("Listener thread was stopped");
      goto stop_thread;
    }
    ucp_conn_request_h conn_request = NULL;

    g_mutex_lock (&ucxserver->priv->server_conn_ctx.req_lock);
    l = ucxserver->priv->server_conn_ctx.gst_ucx_conn_req_list;
    while (l != NULL) {
      conn_request = (ucp_conn_request_h) l->data;

      /* Connect to pending request */
      create_and_add_new_connection (ucxdata, conn_request, is_src);

      next = l->next;
      ucxserver->priv->server_conn_ctx.gst_ucx_conn_req_list =
          g_list_delete_link (ucxserver->priv->
          server_conn_ctx.gst_ucx_conn_req_list, l);
      l = next;
    }
    g_mutex_unlock (&ucxserver->priv->server_conn_ctx.req_lock);
  }

stop_thread:
  return NULL;
}

static gpointer
do_listen_src (gpointer data)
{
  return do_listen (data, TRUE);
}

static gpointer
do_listen_sink (gpointer data)
{
  return do_listen (data, FALSE);
}

int
start_server (gpointer data, bool is_src)
{
  GstUcxServerSink *ucxserversink = (GstUcxServerSink *) data;
  GstUcxServerSrc *ucxserversrc = (GstUcxServerSrc *) data;
  GstUcxServer *ucxserver;

  ucxserver = is_src ? &ucxserversrc->server : &ucxserversink->server;

  ucxserver->priv = g_new0 (GstUcxServerPriv, 1);
  if (!ucxserver->priv)
    return 1;

  g_mutex_init (&ucxserver->priv->conn_lock);
  g_cond_init (&ucxserver->priv->conn_cond);
  ucxserver->priv->server_conn_ctx.gst_ucx_conn_req_list = NULL;
  ucxserver->priv->gst_ucx_nvevents_list = NULL;
  g_mutex_init (&ucxserver->priv->server_conn_ctx.req_lock);
  g_mutex_init (&ucxserver->priv->event_lock);

  /* Initialize UCP application context */
  if (gst_ucp_init_context (&ucxserver->priv->ucp_context, ucxserver->high_perf) != UCS_OK)
    goto err_priv;

  if (gst_ucp_init_worker (ucxserver->priv->ucp_context,
          &ucxserver->priv->server_conn_ctx.listen_worker, 0) != UCS_OK) {
    goto ucp_cleanup;
  }

  /* Create the UCP listener on the ucp_listen_worker */
  if (start_server_listener (ucxserver->priv->server_conn_ctx.listen_worker,
          &ucxserver->priv->server_conn_ctx,
          ucxserver->addr, ucxserver->port) != UCS_OK) {
    goto err_listener;
  }

  ucxserver->stop_listener_thread = 0;
  if (is_src) {
    ucxserver->listener_thread =
        g_thread_new ("listener_thread", do_listen_src, (gpointer) data);
  } else {
    ucxserver->listener_thread =
        g_thread_new ("listener_thread", do_listen_sink, (gpointer) data);
  }
  if (!ucxserver->listener_thread)
    goto err_thread;

  return 0;

err_thread:
  ucp_listener_destroy (ucxserver->priv->server_conn_ctx.listener);
err_listener:
  ucp_worker_destroy (ucxserver->priv->server_conn_ctx.listen_worker);
ucp_cleanup:
  ucp_cleanup (ucxserver->priv->ucp_context);
err_priv:
  g_mutex_clear (&ucxserver->priv->conn_lock);
  g_cond_clear (&ucxserver->priv->conn_cond);
  g_mutex_clear (&ucxserver->priv->server_conn_ctx.req_lock);
  g_mutex_clear (&ucxserver->priv->event_lock);
  g_free (ucxserver->priv);
  return 1;
}

void
clear_gst_ucp_conn_ctx (gpointer data) {
  gst_ucp_conn_ctx_t *conn = (gst_ucp_conn_ctx_t *) data;
  uint64_t flags;
  ucs_status_t status;
  if (!conn)
    return;
  if (conn->ep) {
    for (int i = 0; i < conn->ep_num; i++) {
      if (conn->ep[i]) {
        gst_ucp_ep_close (conn->data_worker[i], conn->ep[i]);
      }
    }
    g_free (conn->ep);
  }
  if (conn->data_worker) {
    for (int i = 0; i < conn->ep_num; i++) {
      if (conn->data_worker[i]) {
        ucp_worker_destroy (conn->data_worker[i]);
      }
    }
    g_free (conn->data_worker);
  }
  g_mutex_clear (&conn->conn_lock);
}


void
free_gst_ucp_conn_ctx (gpointer data)
{
  clear_gst_ucp_conn_ctx(data);
  g_free (data);
  return;
}


/* GstUcxSrc */

ucs_status_t
gst_ucx_src_prep_am_audio_nv (GstBuffer ** gst_buffer,
    void **rdma_data_ptr, NvDsUcxBuf * nv_dsucx_buf, GMutex * meta_lock)
{
  GstMapInfo map_info;
  NvBufAudio *nvbufaudio;

  if (!gst_buffer_map (*gst_buffer, &map_info, GST_MAP_READ))
    return UCS_ERR_NO_MEMORY;

  nvbufaudio = (NvBufAudio *) map_info.data;
  gst_buffer_unmap (*gst_buffer, &map_info);

  /* Only copy the gstbuffer metadata for first element of batch. */
  if (nv_dsucx_buf->nvbuf_audio.batchId == 0) {
    copy_gstbuf_info (&nv_dsucx_buf->nvbuf_audio.nvgstbuffer_params, NULL, NULL,
        *gst_buffer);
    nvbufaudio->numFilled = nv_dsucx_buf->nvbuf_audio.numFilled;
    nvbufaudio->batchSize = nv_dsucx_buf->nvbuf_audio.batchSize;
  }
  copy_nvbufaudio_params (&nv_dsucx_buf->nvbuf_audio.nvbufaudio_params,
      &nvbufaudio->audioBuffers[nv_dsucx_buf->nvbuf_audio.batchId]);

  (*rdma_data_ptr) =
      (void *) nvbufaudio->audioBuffers[nv_dsucx_buf->nvbuf_audio.
      batchId].dataPtr;

  GST_DEBUG
    ("From nv_dsucx_buf: numFilled: %d, batchSize: %d, batchId: %d, has_user_meta: %d",
    nv_dsucx_buf->nvbuf_audio.numFilled, nv_dsucx_buf->nvbuf_audio.batchSize,
    nv_dsucx_buf->nvbuf_audio.batchId, nv_dsucx_buf->nvbuf_audio.has_user_meta);
  /*
   * Create the NvDsAudioFrameMeta here.
   *
   * A non-zero value for num_surfaces_per_frame means that the buffer received
   * contains FrameMeta as well as the number of surfaces per frame.
   * A zero value is interpreted as FrameMeta being absent so the default value
   * of 1 is used for num_surfaces_per_frame.
   */
  if (nv_dsucx_buf->nvbuf_audio.num_surfaces_per_frame > 0) {
    NvDsBatchMeta *batch_meta = NULL;
    g_mutex_lock (meta_lock);
    batch_meta = gst_buffer_get_nvds_batch_meta (*gst_buffer);
    if (batch_meta == NULL) {
      // Create batch_meta for gst_buffer
      batch_meta = nvds_create_audio_batch_meta (nvbufaudio->batchSize);
      if (batch_meta == NULL) {
        GST_ERROR ("Failed to create batch meta data in clientsrc");
        g_mutex_unlock (meta_lock);
        return UCS_ERR_NO_MESSAGE;
      }
      NvDsMeta *meta = gst_buffer_add_nvds_meta (*gst_buffer, batch_meta, NULL,
          nvds_audio_batch_meta_copy_func, nvds_audio_batch_meta_release_func);

      meta->meta_type = NVDS_BATCH_GST_META;
      batch_meta->base_meta.batch_meta = batch_meta;
      batch_meta->base_meta.meta_type = NVDS_AUDIO_BATCH_META;
      batch_meta->base_meta.copy_func = nvds_audio_batch_meta_copy_func;
      batch_meta->base_meta.release_func = nvds_audio_batch_meta_release_func;
      batch_meta->max_frames_in_batch = nvbufaudio->batchSize *
          nv_dsucx_buf->nvbuf_audio.num_surfaces_per_frame;
    }
    NvDsAudioFrameMeta *frame_meta = nvds_acquire_audio_frame_meta_from_pool (batch_meta);
    frame_meta->batch_id = nv_dsucx_buf->nvbuf_audio.batchId;
    frame_meta->classifier_meta_list = NULL;
    nvds_add_audio_frame_meta_to_audio_batch (batch_meta, frame_meta);
    g_mutex_unlock (meta_lock);
  }

  return UCS_OK;
}

static gpointer
copy_ucx_user_meta (gpointer data, gpointer user_data)
{
  NvDsUserMeta *user_meta = (NvDsUserMeta *) data;
  NVDS_CUSTOM_PAYLOAD *src_user_metadata =
      (NVDS_CUSTOM_PAYLOAD *) user_meta->user_meta_data;
  NVDS_CUSTOM_PAYLOAD *dst_user_metadata =
      (NVDS_CUSTOM_PAYLOAD *) g_malloc0 (sizeof (NVDS_CUSTOM_PAYLOAD));
  dst_user_metadata->payloadType = src_user_metadata->payloadType;
  dst_user_metadata->payloadSize = src_user_metadata->payloadSize;
  dst_user_metadata->payload =
      (uint8_t *) g_malloc0 (src_user_metadata->payloadSize);
  memcpy (dst_user_metadata->payload, src_user_metadata->payload,
      src_user_metadata->payloadSize * sizeof (uint8_t));
  return (gpointer) dst_user_metadata;
}

static void
release_ucx_user_meta (gpointer data, gpointer user_data)
{
  NvDsUserMeta *user_meta = (NvDsUserMeta *) data;
  NVDS_CUSTOM_PAYLOAD *user_metadata =
      (NVDS_CUSTOM_PAYLOAD *) user_meta->user_meta_data;
  g_free (user_metadata->payload);
  user_metadata->payload = NULL;
  g_free (user_meta->user_meta_data);
  user_meta->user_meta_data = NULL;
  return;
}

ucs_status_t
gst_ucx_src_prep_am_meta_audio (GstBuffer ** gst_buffer,
    void **rdma_data_ptr, NvDsCustomMetaUcx nv_dsucx_meta)
{
  NvDsUserMeta *user_meta = NULL;
  NVDS_CUSTOM_PAYLOAD *ucx_metadata = NULL;

  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (*gst_buffer);

  GST_DEBUG ("Prep am meta for meta size %d type %X", nv_dsucx_meta.payloadSize,
      nv_dsucx_meta.payloadType);

  if (batch_meta == NULL) {
    GST_ERROR ("Failed to create batch meta data in clientsrc");
    return UCS_ERR_NO_MESSAGE;
  }

  ucx_metadata =
      (NVDS_CUSTOM_PAYLOAD *) g_malloc0 (sizeof (NVDS_CUSTOM_PAYLOAD));

  ucx_metadata->payloadType = nv_dsucx_meta.payloadType;
  ucx_metadata->payloadSize = nv_dsucx_meta.payloadSize;
  ucx_metadata->payload = (uint8_t *) g_malloc0 (ucx_metadata->payloadSize);

  *rdma_data_ptr = (void *) ucx_metadata->payload;

  user_meta = nvds_acquire_user_meta_from_pool (batch_meta);
  user_meta->user_meta_data = (void *) ucx_metadata;
  user_meta->base_meta.meta_type = NVDS_USER_CUSTOM_META;
  user_meta->base_meta.copy_func = (NvDsMetaCopyFunc) copy_ucx_user_meta;
  user_meta->base_meta.release_func =
      (NvDsMetaReleaseFunc) release_ucx_user_meta;

  nvds_add_user_meta_to_batch (batch_meta, user_meta);

  GST_DEBUG
      ("AM | batchId: %d payloadType %X payloadSize %d rdma_metadata_buffer %p",
      nv_dsucx_meta.batchId, nv_dsucx_meta.payloadType,
      nv_dsucx_meta.payloadSize, *rdma_data_ptr);

  return UCS_OK;
}

ucs_status_t
gst_ucx_src_prep_am_meta_video (GstBuffer ** gst_buffer,
    void **rdma_data_ptr, NvDsCustomMetaUcx nv_dsucx_meta)
{
  NVDS_CUSTOM_PAYLOAD *metadata = NULL;
  NvDsUserMeta *user_meta = NULL;

  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (*gst_buffer);

  GST_DEBUG ("Prep am meta for meta size %d type %X",
      nv_dsucx_meta.payloadSize, nv_dsucx_meta.payloadType);

  if (batch_meta == NULL) {
    GST_ERROR ("Failed to create batch meta data in clientsrc");
    return UCS_ERR_NO_MESSAGE;
  }

  metadata = (NVDS_CUSTOM_PAYLOAD *) g_malloc0 (sizeof (NVDS_CUSTOM_PAYLOAD));

  metadata->payloadType = nv_dsucx_meta.payloadType;
  metadata->payloadSize = nv_dsucx_meta.payloadSize;
  metadata->payload = (uint8_t *) g_malloc0 (metadata->payloadSize);

  *rdma_data_ptr = (void *) metadata->payload;

  user_meta = nvds_acquire_user_meta_from_pool (batch_meta);
  user_meta->user_meta_data = (void *) metadata;
  user_meta->base_meta.meta_type = NVDS_USER_CUSTOM_META;
  user_meta->base_meta.copy_func = (NvDsMetaCopyFunc) copy_ucx_user_meta;
  user_meta->base_meta.release_func =
      (NvDsMetaReleaseFunc) release_ucx_user_meta;

  nvds_add_user_meta_to_batch (batch_meta, user_meta);

  GST_DEBUG
      ("AM | batchId: %d payloadType %X payloadSize %d rdma_metadata_buffer %p",
      nv_dsucx_meta.batchId, nv_dsucx_meta.payloadType,
      nv_dsucx_meta.payloadSize, *rdma_data_ptr);

  return UCS_OK;
}

ucs_status_t
gst_ucx_src_prep_am_video (GstBuffer ** gst_buffer,
    void **rdma_data_ptr, NvDsUcxBuf * nv_dsucx_buf, GMutex * meta_lock)
{
  GstMapInfo map_info;
  NvBufSurface *nvbufsurf;

  if (!gst_buffer_map (*gst_buffer, &map_info, GST_MAP_READ))
    return UCS_ERR_NO_MEMORY;

  nvbufsurf = (NvBufSurface *) map_info.data;
  gst_buffer_unmap (*gst_buffer, &map_info);


  /* Only copy the gstbuffer metadata for first element of batch. */
  if (nv_dsucx_buf->nvbuf_surf.batchId == 0) {
    copy_gstbuf_info (&nv_dsucx_buf->nvbuf_surf.nvgstbuffer_params, NULL,
        NULL, *gst_buffer);
    nvbufsurf->numFilled = nv_dsucx_buf->nvbuf_surf.numFilled;
    nvbufsurf->batchSize = nv_dsucx_buf->nvbuf_surf.batchSize;
  }
  copy_nvbufsurf_params (&nv_dsucx_buf->nvbuf_surf.nvbufsurf_params,
      &nvbufsurf->surfaceList[nv_dsucx_buf->nvbuf_surf.batchId]);

  (*rdma_data_ptr) =
      nvbufsurf->surfaceList[nv_dsucx_buf->nvbuf_surf.batchId].dataPtr;

  GST_DEBUG
      ("AM [%d] | numFilled: %d batchSize: %d has_user_meta: %d num_surfaces_per_frame %d rdma_data_buffer %p",
      nv_dsucx_buf->nvbuf_surf.batchId, nv_dsucx_buf->nvbuf_surf.numFilled,
      nv_dsucx_buf->nvbuf_surf.batchSize, nv_dsucx_buf->nvbuf_surf.has_user_meta,
      nv_dsucx_buf->nvbuf_surf.num_surfaces_per_frame, *rdma_data_ptr);

  /*
   * Create the NvDsFrameMeta here.
   *
   * A non-zero value for num_surfaces_per_frame means that the buffer received
   * contains FrameMeta as well as the number of surfaces per frame.
   * A zero value is interpreted as FrameMeta being absent so the default value
   * of 1 is used for num_surfaces_per_frame.
   */
  if (nv_dsucx_buf->nvbuf_surf.num_surfaces_per_frame > 0) {
    NvDsBatchMeta *batch_meta = NULL;
    g_mutex_lock (meta_lock);
    batch_meta = gst_buffer_get_nvds_batch_meta (*gst_buffer);
    if (batch_meta == NULL) {
      // Create batch_meta for gst_buffer
      batch_meta = nvds_create_batch_meta (nvbufsurf->batchSize);
      if (batch_meta == NULL) {
        GST_ERROR ("Failed to create batch meta data in clientsrc");
        g_mutex_unlock (meta_lock);
        return UCS_ERR_NO_MESSAGE;
      }
      NvDsMeta *meta = gst_buffer_add_nvds_meta (*gst_buffer, batch_meta, NULL,
          nvds_batch_meta_copy_func, nvds_batch_meta_release_func);
      meta->meta_type = NVDS_BATCH_GST_META;
      batch_meta->base_meta.batch_meta = batch_meta;
      batch_meta->base_meta.copy_func = nvds_batch_meta_copy_func;
      batch_meta->base_meta.release_func = nvds_batch_meta_release_func;
      batch_meta->max_frames_in_batch = nvbufsurf->batchSize *
          nv_dsucx_buf->nvbuf_surf.num_surfaces_per_frame;
    }
    NvDsFrameMeta *frame_meta = nvds_acquire_frame_meta_from_pool (batch_meta);
    frame_meta->batch_id = nv_dsucx_buf->nvbuf_surf.batchId;
    nvds_add_frame_meta_to_batch (batch_meta, frame_meta);
    g_mutex_unlock (meta_lock);
  }

  return UCS_OK;
}

ucs_status_t
gst_ucx_src_prep_am_raw (GstBuffer ** gst_buffer,
    void **rdma_data_ptr, NvDsUcxBuf * nv_dsucx_buf, GMutex * meta_lock)
{
  GstMapInfo map_info;
  gsize max_size, curr_size;

  curr_size = gst_buffer_get_sizes (*gst_buffer, NULL, &max_size);
  if (nv_dsucx_buf->nvbuf_raw.size > max_size)
    gst_buffer_resize (*gst_buffer, 0, nv_dsucx_buf->nvbuf_raw.size);

  if (curr_size != nv_dsucx_buf->nvbuf_raw.size)
    gst_buffer_set_size (*gst_buffer, nv_dsucx_buf->nvbuf_raw.size);

  if (!gst_buffer_map (*gst_buffer, &map_info, GST_MAP_READ))
    return UCS_ERR_NO_MEMORY;

  (*rdma_data_ptr) = (void *) map_info.data;
  gst_buffer_unmap (*gst_buffer, &map_info);

  copy_gstbuf_info (&nv_dsucx_buf->nvbuf_raw.nvgstbuffer_params, NULL,
      NULL, *gst_buffer);

  return UCS_OK;
}

int
ucx_src_init_buffer_type_attr (GstUcxSrc * ucxsrc)
{
  switch (ucxsrc->buf_type) {
    case NVDSUCX_BUF_TYPE_VIDEO:
      ucxsrc->prep_am = gst_ucx_src_prep_am_video;
      ucxsrc->prep_am_meta = gst_ucx_src_prep_am_meta_video;
      break;
    case NVDSUCX_BUF_TYPE_AUDIO_NV:
      ucxsrc->prep_am = gst_ucx_src_prep_am_audio_nv;
      ucxsrc->prep_am_meta = gst_ucx_src_prep_am_meta_audio;
      break;
    case NVDSUCX_BUF_TYPE_AUDIO_RAW:
    case NVDSUCX_BUF_TYPE_TEXT:
      ucxsrc->prep_am = gst_ucx_src_prep_am_raw;
      break;
    default:
      return 1;
  }

  return 0;
}

static int
ucx_src_audio_nv_buffer_pool_new (GstUcxSrc * ucxsrc, GstCaps * caps)
{
  GstNvDsAudioAllocatorParams allocator_params;
  GstAllocationParams allocation_params;
  GstStructure *structure;
  const gchar *format;

  structure = gst_caps_get_structure (caps, 0);
  if (!gst_structure_get_int (structure, "rate",
          (gint *) & allocator_params.rate))
    return false;

  if (!gst_structure_get_int (structure, "channels",
          (gint *) & allocator_params.channels))
    return false;

  format = gst_structure_get_string (structure, "format");
  if (!strcmp (format, "S16LE"))
    allocator_params.format = NVBUF_AUDIO_S16LE;
  else if (!strcmp (format, "F32LE"))
    allocator_params.format = NVBUF_AUDIO_F32LE;
  else
    return false;

  allocator_params.batchSize = ucxsrc->nvbuf_batch_size;
  allocator_params.isContiguous = true;
  allocator_params.gpuId = ucxsrc->gpu_id;
  /* nvdsucxdemux handle only mem_sys audiobuffers */
  allocator_params.memType = NVDS_MEM_SYSTEM;
  allocator_params.layout = NVBUF_AUDIO_INTERLEAVED;
  allocator_params.bpf = DEFAULT_ALLOCATOR_BPF;
  allocator_params.bufferLength = DEFAULT_ALLOCATOR_BUF_LEN;

  GstStructure *config = NULL;

  ucxsrc->pool = gst_buffer_pool_new ();
  if (!ucxsrc->pool) {
    GST_ERROR ("failed to create new pool");
    return 1;
  }

  config = gst_buffer_pool_get_config (ucxsrc->pool);
  gst_buffer_pool_config_set_params (config, caps, sizeof (GstNvDsAudioMemory),
      ucxsrc->num_nvbuf, ucxsrc->num_nvbuf);

  GstAllocator *allocator = gst_nvdsaudio_allocator_new (&allocator_params);

  memset (&allocation_params, 0, sizeof (allocation_params));
  gst_buffer_pool_config_set_allocator (config, allocator, &allocation_params);
  if (!gst_buffer_pool_set_config (ucxsrc->pool, config)) {
    GST_ERROR ("bufferpool configuration failed");
    goto err_set_conf;
  }

  return 0;

err_set_conf:
  gst_object_unref (ucxsrc->pool);
  return 1;
}

static int
ucx_src_video_buffer_pool_new (GstUcxSrc * ucxsrc, GstCaps * caps)
{
  GstStructure *config = NULL;

  ucxsrc->pool = gst_nvds_buffer_pool_new ();
  if (!ucxsrc->pool) {
    GST_ERROR ("failed to create new pool");
    return 1;
  }

  config = gst_buffer_pool_get_config (ucxsrc->pool);
  gst_buffer_pool_config_set_params (config, caps, sizeof (NvBufSurface),
      ucxsrc->num_nvbuf, ucxsrc->num_nvbuf);

  gst_structure_set (config,
      "gpu-id", G_TYPE_UINT, ucxsrc->gpu_id,
      "memtype", G_TYPE_UINT, ucxsrc->mem_type,
      "batch-size", G_TYPE_UINT, ucxsrc->nvbuf_batch_size, NULL);

  if (!gst_buffer_pool_set_config (ucxsrc->pool, config)) {
    GST_ERROR ("bufferpool configuration failed");
    goto err_set_conf;
  }

  return 0;

err_set_conf:
  gst_object_unref (ucxsrc->pool);
  return 1;
}

static int
ucx_src_raw_buffer_pool_new (GstUcxSrc * ucxsrc, GstCaps * caps)
{
  GstStructure *config = NULL;

  ucxsrc->pool = gst_buffer_pool_new ();
  if (!ucxsrc->pool) {
    GST_ERROR ("failed to create new pool");
    return 1;
  }

  config = gst_buffer_pool_get_config (ucxsrc->pool);
  gst_buffer_pool_config_set_params (config, caps, ucxsrc->raw_buf_size,
      ucxsrc->num_nvbuf, 0);

  if (!gst_buffer_pool_set_config (ucxsrc->pool, config)) {
    GST_ERROR ("bufferpool configuration failed");
    goto err_set_conf;
  }

  return 0;

err_set_conf:
  gst_object_unref (ucxsrc->pool);
  return 1;
}

gboolean
gst_ucx_src_buffer_pool_new (GstUcxSrc * ucxsrc, GstCaps * caps)
{
  int res;
  switch (ucxsrc->buf_type) {
    case NVDSUCX_BUF_TYPE_VIDEO:
      res = ucx_src_video_buffer_pool_new (ucxsrc, caps);
      break;
    case NVDSUCX_BUF_TYPE_AUDIO_NV:
      res = ucx_src_audio_nv_buffer_pool_new (ucxsrc, caps);
      break;
    case NVDSUCX_BUF_TYPE_AUDIO_RAW:
    case NVDSUCX_BUF_TYPE_TEXT:
      res = ucx_src_raw_buffer_pool_new (ucxsrc, caps);
      break;
    default:
      return UCS_ERR_INVALID_PARAM;
  }

  if (res)
    return FALSE;

  if (gst_buffer_pool_set_active (ucxsrc->pool, TRUE) == FALSE) {
    GST_ERROR ("bufferpool activation failed");
    goto err_set_active;
  }

  return TRUE;

err_set_active:
  gst_object_unref (ucxsrc->pool);
  return FALSE;
}

gboolean
batch_end_check_func (gpointer arg)
{
  volatile int *timeout = (int *) arg;
  *timeout = 1;
  return FALSE;
}

/* GstUcxClient */

static void
set_connect_addr (const char *address_str, int port,
    struct sockaddr_in *connect_addr)
{
  memset (connect_addr, 0, sizeof (struct sockaddr_in));
  connect_addr->sin_family = AF_INET;
  connect_addr->sin_addr.s_addr = inet_addr (address_str);
  connect_addr->sin_port = htons (port);

  return;
}

static void
client_sink_err_cb (void *arg, ucp_ep_h ep, ucs_status_t status)
{
  GstUcxClientSink *ucxclientsink;

  if (!arg) {
    GST_ERROR ("ucxclientsink is NULL");
    return;
  }
  ucxclientsink = (GstUcxClientSink *) arg;

  GST_DEBUG_OBJECT (ucxclientsink,
      "client sink was invoked with status %d (%s)", status,
      ucs_status_string (status));

  ucxclientsink->client.priv->gst_ucp_conn_ctx.err = status;
  return;
}

static void
client_src_err_cb (void *arg, ucp_ep_h ep, ucs_status_t status)
{
  GstUcxClientSrc *ucxclientsrc;

  if (!arg) {
    GST_ERROR ("ucxclientsrc is NULL");
    return;
  }
  ucxclientsrc = (GstUcxClientSrc *) arg;

  GST_DEBUG_OBJECT (ucxclientsrc,
      "client src was invoked with status %d (%s)", status,
      ucs_status_string (status));

  gst_ucp_reset_am_desc (&ucxclientsrc->src);

  return;
}

static int
ucx_sink_init_am (GstUcxSink * ucxsink)
{
  int i = 0;
  ucxsink->am_hdrs = g_new0 (GstUcxSinkAmHdrParams, ucxsink->nvbuf_batch_size);
  switch (ucxsink->buf_type) {
    case NVDSUCX_BUF_TYPE_VIDEO:
      ucxsink->prep_am = gst_ucx_sink_prep_am_video;
      ucxsink->prep_am_meta = gst_ucx_sink_prep_am_meta;
      for (i = 0; i < ucxsink->nvbuf_batch_size; i++) {
        ucxsink->am_hdrs[i].am_hdr_buf = (void *) g_new0 (NvBufSurfaceDsUcx, 1);
        ucxsink->am_hdrs[i].am_hdr_buf_size = sizeof (NvBufSurfaceDsUcx);
        ucxsink->am_hdrs[i].am_hdr_meta_buf =
            (void *) g_new0 (NvDsCustomMetaUcx, 1);
        ucxsink->am_hdrs[i].am_hdr_meta_buf_size = sizeof (NvDsCustomMetaUcx);
      }
      break;
    case NVDSUCX_BUF_TYPE_AUDIO_NV:
      ucxsink->prep_am = gst_ucx_sink_prep_am_audio_nv;
      ucxsink->prep_am_meta = gst_ucx_sink_prep_am_meta;
      for (i = 0; i < ucxsink->nvbuf_batch_size; i++) {
        ucxsink->am_hdrs[i].am_hdr_buf = (void *) g_new0 (NvBufAudioDsUcx, 1);
        ucxsink->am_hdrs[i].am_hdr_buf_size = sizeof (NvBufAudioDsUcx);
        ucxsink->am_hdrs[i].am_hdr_meta_buf =
            (void *) g_new0 (NvDsCustomMetaUcx, 1);
        ucxsink->am_hdrs[i].am_hdr_meta_buf_size = sizeof (NvDsCustomMetaUcx);
      }
      break;
    case NVDSUCX_BUF_TYPE_AUDIO_RAW:
    case NVDSUCX_BUF_TYPE_TEXT:
      ucxsink->prep_am = gst_ucx_sink_prep_am_raw;
      for (i = 0; i < ucxsink->nvbuf_batch_size; i++) {
        ucxsink->am_hdrs[i].am_hdr_buf = (void *) g_new0 (NvBufRawDsUcx, 1);
        ucxsink->am_hdrs[i].am_hdr_buf_size = sizeof (NvBufRawDsUcx);
      }
      break;
    default:
      return 1;
  }

  return 0;
}

static void
ucx_sink_free_am (GstUcxSink * ucxsink)
{
  int i;
  switch (ucxsink->buf_type) {
    case NVDSUCX_BUF_TYPE_VIDEO:
    case NVDSUCX_BUF_TYPE_AUDIO_NV:
      for (i = 0; i < ucxsink->nvbuf_batch_size; i++) {
        g_free (ucxsink->am_hdrs[i].am_hdr_buf);
        g_free (ucxsink->am_hdrs[i].am_hdr_meta_buf);
      }
      break;
    case NVDSUCX_BUF_TYPE_AUDIO_RAW:
    case NVDSUCX_BUF_TYPE_TEXT:
      for (i = 0; i < ucxsink->nvbuf_batch_size; i++)
        g_free (ucxsink->am_hdrs[i].am_hdr_buf);
      break;
    default:
      break;
    }
  g_free (ucxsink->am_hdrs);
  return;
}


int
start_src (GstUcxSrc *ucxsrc, GstBaseSrc * src)
{
  if (ucxsrc->buf_type == NVDSUCX_BUF_TYPE_AUDIO_RAW)
    gst_base_src_set_format (src, GST_FORMAT_TIME);

  if (ucx_src_init_buffer_type_attr (ucxsrc))
    return 1;

  if (gst_ucx_create_thread_pool (&ucxsrc->thread_pool,
          ucxsrc->nvbuf_batch_size))
    return 1;

  ucxsrc->threads_data =
      g_new0 (GstUcxThreadDataSource, ucxsrc->nvbuf_batch_size);
  if (!ucxsrc->threads_data)
    goto err_threads_data;

  ucxsrc->is_eos = FALSE;

  return 0;

err_threads_data:
  gst_task_pool_cleanup (ucxsrc->thread_pool);
  gst_object_unref (ucxsrc->thread_pool);
  return 1;
}

int
start_sink (GstUcxSink *ucxsink)
{
  if (ucx_sink_init_am (ucxsink))
    return 1;

  if (gst_ucx_create_thread_pool (&ucxsink->thread_pool,
          ucxsink->nvbuf_batch_size))
    goto err;

  ucxsink->threads_data =
      g_new0 (GstUcxThreadDataSink, ucxsink->nvbuf_batch_size);
  if (!ucxsink->threads_data)
    goto err_threads_data;

  return 0;

err_threads_data:
  gst_task_pool_cleanup (ucxsink->thread_pool);
  gst_object_unref (ucxsink->thread_pool);
err:
  ucx_sink_free_am (ucxsink);
  return 1;
}


static ucs_status_t
client_create_ep (gst_ucp_conn_ctx_t * conn, gpointer ucxdata, bool is_src)
{
  ucp_ep_params_t ep_params;
  ucs_status_t status;
  struct sockaddr_in connect_addr;

  GstUcxClient *ucxclient = get_gstucxclient (ucxdata, is_src);
  set_connect_addr ((const char *) ucxclient->addr, ucxclient->port,
      &connect_addr);
  ep_params.field_mask = UCP_EP_PARAM_FIELD_FLAGS |
                         UCP_EP_PARAM_FIELD_SOCK_ADDR |
                         UCP_EP_PARAM_FIELD_ERR_HANDLER |
                         UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;

  ep_params.err_mode = UCP_ERR_HANDLING_MODE_PEER;
  ep_params.err_handler.cb = is_src ? client_src_err_cb : client_sink_err_cb;
  ep_params.err_handler.arg = ucxdata;
  ep_params.flags = UCP_EP_PARAMS_FLAGS_CLIENT_SERVER |
                    UCP_EP_PARAMS_FLAGS_SEND_CLIENT_ID;
  ep_params.sockaddr.addr = (struct sockaddr *) &connect_addr;
  ep_params.sockaddr.addrlen = sizeof (connect_addr);
  status = ucp_ep_create (conn->data_worker[conn->ep_num], &ep_params,
      &conn->ep[conn->ep_num]);
  if (status != UCS_OK) {
    GST_ERROR ("failed to create an endpoint on the server: (%s)",
          ucs_status_string (status));
      return status;
  }
  status = blocking_ep_flush (conn);
  if (status != UCS_OK) {
    GST_ERROR ("failed to create an endpoint on the server: (%s)",
        ucs_status_string (status));
    return status;
  }
  if (conn->err != UCS_OK) {
    GST_ERROR ("failed to connect status ucs status %s",
        ucs_status_string (ucxclient->priv->gst_ucp_conn_ctx.err));
    return status;
  }
  return UCS_OK;
}

int
start_client (gpointer ucxdata, bool is_src)
{
  ucp_ep_params_t ep_params;
  ucs_status_t status = UCS_OK;
  int thread_num;
  uint64_t client_id;
  gst_ucp_conn_ctx_t *conn;
  GstUcxClient *ucxclient = get_gstucxclient (ucxdata, is_src);
  if (!ucxclient) {
    GST_ERROR ("null data");
    return 1;
  }

  ucxclient->priv = g_new0 (GstUcxClientPriv, 1);
  if (!ucxclient->priv) {
    GST_ERROR_OBJECT (ucxclient, "failed to allocate private data");
    return 1;
  }
  g_mutex_init (&ucxclient->priv->conn_lock);
  g_cond_init (&ucxclient->priv->conn_cond);
  if (gst_ucp_init_context (&ucxclient->priv->ucp_context, ucxclient->high_perf) != UCS_OK)
    goto err_ucp_init;


  thread_num = get_thread_num (ucxdata, is_src, FALSE);
  if (thread_num == -1)
    goto err_cleanup;
  conn = &ucxclient->priv->gst_ucp_conn_ctx;
  client_id = rand_client_num ();
  GST_DEBUG ("Client %ld is creating endpoint to remote: IP %s port %d",
      client_id, ucxclient->addr, ucxclient->port);

  if (UCS_OK != init_conn_ctx (conn, thread_num))
    goto err_cleanup;

  while (conn->ep_num < thread_num) {
    if (create_worker_and_register_cbs(ucxdata, ucxclient->priv->ucp_context,
        conn, client_id + conn->ep_num, is_src, FALSE) != UCS_OK)
      goto err_ep;

    GST_DEBUG ("Sending connect request id %d out of %d", conn->ep_num + 1, thread_num);
    status = client_create_ep (conn, ucxdata, is_src);
    if (status != UCS_OK) {
      goto err_ep_create;
    }
    conn->ep_num++;
  }
  return status;

err_ep_create:
err_init_am:
  gst_ucp_ep_close (conn->data_worker[conn->ep_num], conn->ep[conn->ep_num]);
err_ep:
  ucp_worker_destroy (conn->data_worker[conn->ep_num]);
  clear_gst_ucp_conn_ctx (conn);
err_cleanup:
  ucp_cleanup (ucxclient->priv->ucp_context);
err_ucp_init:
  g_mutex_clear (&ucxclient->priv->conn_lock);
  g_cond_clear (&ucxclient->priv->conn_cond);
  g_free (ucxclient->priv);
  return 1;
}


static void
clear_server_conn_ctx(gst_ucx_conn_req_ctx_t *server_conn_ctx)
{
  GList *list_entry;
  ucs_status_t status;
  ucp_listener_destroy (server_conn_ctx->listener);
  ucp_worker_destroy (server_conn_ctx->listen_worker);

  /* Reject any outstanding connections. */
  for (list_entry = server_conn_ctx->gst_ucx_conn_req_list;
      list_entry != NULL; list_entry = list_entry->next) {
    status = ucp_listener_reject (server_conn_ctx->listener,
        (ucp_conn_request_h) list_entry->data);
    if (status != UCS_OK) {
      GST_ERROR ("server failed to reject a connection request: (%s)",
          ucs_status_string (status));
    }
  }

  /* Clear req mutex */
  g_mutex_clear (&server_conn_ctx->req_lock);
}

void
stop_sink (GstUcxSink* sink)
{
  int i;

  /* Clean thread pool */
  gst_task_pool_cleanup (sink->thread_pool);
  gst_object_unref (sink->thread_pool);
  g_free (sink->threads_data);
  ucx_sink_free_am (sink);
  return;
}

void
stop_src (GstUcxSrc* src)
{
  g_free (src->am_desc);

  /* Clean buffer pool */
  gst_buffer_pool_set_active (src->pool, FALSE);
  gst_object_unref (src->pool);

  /* Clean thread pool */
  gst_task_pool_cleanup (src->thread_pool);
  gst_object_unref (src->thread_pool);
  g_free (src->threads_data);
}


void
stop_server (GstUcxServer * ucxserver)
{
  /* Destroy listener resources */
  ucxserver->stop_listener_thread = 1;
  g_thread_join (ucxserver->listener_thread);

  clear_server_conn_ctx(&ucxserver->priv->server_conn_ctx);

  /* Destroy all event data */
  for (GList *list_entry = ucxserver->priv->gst_ucx_nvevents_list;
      list_entry != NULL; list_entry = list_entry->next) {
    g_free (list_entry->data);
  }
  g_mutex_clear (&ucxserver->priv->event_lock);
  /* Destroy all connection contexts */
  g_list_free_full (ucxserver->priv->gst_ucx_server_conn_list,
      free_gst_ucp_conn_ctx);

  ucp_cleanup (ucxserver->priv->ucp_context);
  g_mutex_clear (&ucxserver->priv->conn_lock);
  g_cond_clear (&ucxserver->priv->conn_cond);

  g_free (ucxserver->priv);
}

void
stop_client (GstUcxClient * ucxclient)
{
  /* Destroy ep and worker */
  clear_gst_ucp_conn_ctx (&ucxclient->priv->gst_ucp_conn_ctx);
  /* Destory UCP context */
  ucp_cleanup (ucxclient->priv->ucp_context);
  g_mutex_clear (&ucxclient->priv->conn_lock);
  g_cond_clear (&ucxclient->priv->conn_cond);
  /* Clear GstUcxClientPriv */
  g_free (ucxclient->priv);
}
