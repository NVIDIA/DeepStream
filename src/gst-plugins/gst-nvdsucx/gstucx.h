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

#ifndef __GST_UCX_H__
#define __GST_UCX_H__

#include <ucp/api/ucp.h>
#include <gst/gst.h>
#include <string.h>
#include <arpa/inet.h>
#include <gmodule.h>
#include <gst/video/video.h>
#include <gst/base/gstpushsrc.h>
#include <gst/base/gstbasesink.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "nvbufsurface.h"
#include <nvbufaudio.h>
#include "gstnvdsmeta.h"
#include "nvdsmeta.h"
#include "gstnvdsbufferpool.h"
#include "gstucxsharedtaskpool.h"
#include "gst-nvcommon.h"
#include "gst_nvdsaudio.h"
#include "nvbufaudio.h"
#include "nvdscustomusermeta.h"
#include "gst-nvevent.h"
#include "nvtx_helper.h"
#include "gst-nvquery.h"
#include "gst-nvquery-internal.h"

#define UCX_DEFAULT_ADDR   "127.0.0.1"
#define UCX_DEFAULT_HIGH_PERF   TRUE
#define UCX_HIGH_PERF_DISABLE_TLS "tcp,cuda"
#define UCX_MAX_PORT       (65535)
#define UCX_DEFAULT_PORT   (7174)
#define UCX_MAX_CONNS      (4)
#define UCX_DEFAULT_CONNS  (1)
#define IP_STRING_LEN       (45)
#define PORT_STRING_LEN     (8)
#define DEFAULT_ALLOCATOR_BPF (4)
#define DEFAULT_ALLOCATOR_BUF_LEN (441000)
#define DEFAULT_GPU_ID (0)
#define DEFAULT_NVBUF_MEMORY_TYPE NVBUF_MEM_CUDA_UNIFIED
#define MIN_NUM_NVBUF (0)
#define MAX_NUM_NVBUF (20)
#define DEFAULT_NUM_NVBUF (4)
#define MIN_NVBUF_BATCH_SIZE (1)
#define MAX_NVBUF_BATCH_SIZE INT_MAX
#define MIN_MAX_EP_NUM (1)
#define MAX_MAX_EP_NUM INT_MAX
#define DEFAULT_NVBUF_BATCH_SIZE (1)
#define DEFAULT_RAW_BUFFER_SIZE (1 << 13)
#define BATCH_END_TIMEOUT_SEC (10)
#define DEFAULT_MAX_EP_NUM (128)
#define CLIENT_ID_SIZE (32)

/* Caps */
#define GST_CAPS_FEATURE_MEMORY_NVMM "memory:NVMM"

/* Video */
#define VIDEO_FORMATS "{ NV12, RGBA, I420 }"

/* Audio */
#define AUDIO_FORMAT                                                  \
    "S8, U8, S16LE, S16BE, U16LE, U16BE, S24_32LE, S24_32BE, "        \
    "U24_32LE, U24_32BE, S32LE, S32BE, U32LE, U32BE, S24LE, S24BE, "  \
    "U24LE, U24BE, S20LE, S20BE, U20LE, U20BE, S18LE, S18BE, U18LE, " \
    "U18BE, F32LE, F32BE, F64LE, F64BE"

#define AUDIO_CHANNELS "[1, 2147483647]"

#define GST_AUDIO_CAPS_MAKE_WITH_FEATURES(features, format, channels)         \
    "audio/x-raw(" features "), "                          \
    "format = (string) { " format                          \
    "}"                                                    \
    ", "                                                   \
    "rate = [ 1, 2147483647 ], "                           \
    "layout = (string) { interleaved, non-interleaved }, " \
    "channels = (int)" channels

/* Text */
#define GST_TEXT_CAPS_MAKE(format) "text/x-raw, format=(string){" format "}"

/* NVTX */
#define NVTX_WRAPPER_START \
    uint64_t nvtx_id; \
    char context_name[256]; \
    strcpy(context_name, __func__); \
    nvtx_helper_start_end(context_name, &nvtx_id)

#define NVTX_WRAPPER_END \
    nvtx_helper_start_end(NULL, &nvtx_id)
/* Properties */

enum
{
  PROP_0,
  PROP_ADDR,
  PROP_PORT,
  PROP_BUF_TYPE,
  PROP_GPU_DEVICE_ID,
  PROP_NVBUF_MEMORY_TYPE,
  PROP_NUM_NVBUF,
  PROP_NVBUF_BATCH_SIZE,
  PROP_RAW_BUF_SIZE,
  PROP_NUM_CONNS,
  PROP_MAX_EP_NUM,
  PROP_HIGH_PERF,
};

typedef enum
{
  NVDSUCX_BUF_TYPE_VIDEO = 0,
  NVDSUCX_BUF_TYPE_AUDIO_NV = 1 << 0,
  NVDSUCX_BUF_TYPE_AUDIO_RAW = 1 << 1,
  NVDSUCX_BUF_TYPE_TEXT = 1 << 2
} GstNvDsUcxBufType;

#define GST_TYPE_NVDSUCX_BUF_TYPE (gst_nvdsucx_buf_type_get_type ())
GType gst_nvdsucx_buf_type_get_type (void);

/* UCX */

#define EVENT_DATA_MAX_LEN 1024

typedef enum
{
  UCP_AM_ID = 1,
  EVENT_AM_ID = 4,
  UCP_AM_ID_META = 7,
  EVENT_AM_BATCH_END = 9
} GstNvDsUcxAmId;

typedef struct gst_ucx_conn_req_ctx
{
  GList *gst_ucx_conn_req_list; /* List of ucp_conn_request_h */
  GMutex req_lock;
  ucp_listener_h listener;
  ucp_worker_h listen_worker;
} gst_ucx_conn_req_ctx_t;

typedef struct gst_ucp_conn_ctx
{
  ucp_worker_h *data_worker;
  ucp_ep_h *ep;
  int ep_num;
  ucs_status_t err;

  volatile int complete;        //server
  uint32_t client_id;           //client
  GMutex conn_lock;             //src
  int main_worker;              //src
} gst_ucp_conn_ctx_t;

typedef struct _GstUcxServerPriv
{
  ucp_context_h ucp_context;
  gst_ucx_conn_req_ctx_t server_conn_ctx;
  GList *gst_ucx_server_conn_list;
  GList *gst_ucx_server_queued_conn_list;
  GMutex conn_lock;
  GCond conn_cond;
  GList *gst_ucx_nvevents_list;
  GMutex event_lock;
} GstUcxServerPriv;

typedef struct _GstUcxClientPriv
{
  ucp_context_h ucp_context;
  gst_ucp_conn_ctx_t gst_ucp_conn_ctx;
  GMutex conn_lock;
  GCond conn_cond;
} GstUcxClientPriv;

typedef struct _NvGstBufferParams
{
  GstClockTime pts;
  GstClockTime dts;
  GstClockTime duration;
  guint64 offset;
  guint64 offset_end;
} NvGstBufferParams;

typedef struct _NvBufSurfaceDsUcx
{
  NvGstBufferParams nvgstbuffer_params;
  NvBufSurfaceParams nvbufsurf_params;
  uint32_t numFilled;
  uint32_t batchSize;
  uint32_t batchId;
  uint32_t num_surfaces_per_frame;
  uint32_t has_user_meta;
} NvBufSurfaceDsUcx;

typedef struct _NvBufRawDsUcx
{
  NvGstBufferParams nvgstbuffer_params;
  gsize size;
} NvBufRawDsUcx;

typedef struct _NvBufAudioDsUcx
{
  NvGstBufferParams nvgstbuffer_params;
  NvBufAudioParams nvbufaudio_params;
  uint32_t numFilled;
  uint32_t batchSize;
  uint32_t batchId;
  uint32_t num_surfaces_per_frame;
  uint32_t has_user_meta;
  uint32_t reserved;
} NvBufAudioDsUcx;

typedef union _NvDsUcxBuf
{
  NvBufSurfaceDsUcx nvbuf_surf;
  NvBufRawDsUcx nvbuf_raw;
  NvBufAudioDsUcx nvbuf_audio;
} NvDsUcxBuf;

typedef struct _NvDsCustomMetaUcx
{
  uint32_t batchId;
  uint32_t payloadType;
  uint32_t payloadSize;
} NvDsCustomMetaUcx;


typedef struct _NvDsUcxAmDesc
{
  volatile int complete;
  void *desc;
  size_t length;
  union
  {
    NvDsUcxBuf nv_dsucx_buf;
    NvDsCustomMetaUcx nv_dscustom_buf;
  };
} NvDsUcxAmDesc;

typedef struct _NvDsUcxAmEventDesc
{
  guint type;
  guint sourceId;
  union
  {
    gchar data[EVENT_DATA_MAX_LEN];
    GstSegment segment;
  };
} NvDsUcxAmEventDesc;

typedef ucs_status_t (*NvDsUcxSrcPrepAmFunc) (GstBuffer ** gst_buffer,
    void **rdma_data_ptr, NvDsUcxBuf * nv_dsucx_buf, GMutex * meta_lock);
typedef ucs_status_t (*NvDsUcxSrcPrepAmMetaFunc) (GstBuffer ** gst_buffer,
    void **rdma_data_ptr, NvDsCustomMetaUcx nv_dsucx_meta);
typedef ucs_status_t (*NvDsUcxAmHandlerFunc) (void *arg,
    const void *header, size_t header_length, void *data,
    size_t length, const ucp_am_recv_param_t * param);

/* UCP stuff */

ucs_status_t gst_ucp_init_context (ucp_context_h * ucp_context,
    gboolean high_perf);
ucs_status_t gst_ucp_send (ucp_worker_h ucp_worker, ucp_ep_h ep,
    GstNvDsUcxAmId am_id, void *am_hdr_buf, size_t am_hdr_buf_size,
    void *rdma_buffer, size_t rdma_buffer_size);
ucs_status_t gst_ucp_send_event (ucp_worker_h ucp_worker, ucp_ep_h ep,
    void *msg, size_t msg_len);
ucs_status_t gst_ucp_send_only (ucp_worker_h ucp_worker, ucp_ep_h ep,
    unsigned int am_tag);
ucs_status_t gst_ucp_recv (ucp_worker_h ucp_worker,
    GstBuffer ** gst_buffer, NvDsUcxSrcPrepAmFunc prep_am,
    NvDsUcxAmDesc * am_desc, GMutex * meta_lock, ucs_status_t *conn_status);
ucs_status_t gst_ucp_recv_meta (ucp_worker_h ucp_worker,
    GstBuffer ** gst_buffer, NvDsUcxSrcPrepAmMetaFunc prep_am_meta,
    NvDsUcxAmDesc * am_desc, ucs_status_t *conn_status);

/* End of UCP stuff */

#define UCX_THREAD_DATA_COMMON_FIELDS gint batch_id; \
                                      void *buffer; \
                                      void *ucxparent; \
                                      gpointer thread_id

typedef struct __GstUcxThreadDataSource
{
  UCX_THREAD_DATA_COMMON_FIELDS;
  ucs_status_t status;
} GstUcxThreadDataSource;

typedef struct _GstUcxThreadDataSink
{
  UCX_THREAD_DATA_COMMON_FIELDS;
  GstMapInfo *mapdata;
} GstUcxThreadDataSink;

G_BEGIN_DECLS typedef struct _GstUcxClient
{
  gint port;                    /* port property */
  gchar *addr;                  /* address property */
  GstUcxClientPriv *priv;       /* private */
  gint max_ep_num;              /* max number of ep */
  gboolean high_perf;           /* RDMA activation property */
} GstUcxClient;

typedef struct _GstUcxServer
{
  gint port;                    /* port property */
  gchar *addr;                  /* address property */
  volatile int stop_listener_thread;
  GThread *listener_thread;
  GstUcxServerPriv *priv;       /* private */
  gchar *caps;                  /* caps for new connection */
  guint num_conns;              /* Number of connections */
  gint total_ep_num;            /* in order to ensure we dont cross MAX_QP */
  gint max_ep_num;              /* max number of ep */
  gboolean high_perf;           /* RDMA activation property */
} GstUcxServer;

typedef struct _GstUcxSrc
{
  GstPushSrc push_ucxsrc;
  GstNvDsUcxBufType buf_type;   /* Type of data: Video/Audio/Text */
  guint gpu_id;                 /* GPU device ID */
  NvBufSurfaceMemType mem_type; /* NvBufSurface memory type */
  gint num_nvbuf;               /* Number of NV buffers */
  gint nvbuf_batch_size;        /* NvBufSurface maximal batch size */
  guint raw_buf_size;           /* raw buffer size */
  NvDsUcxAmDesc *am_desc;       /* am client descs from frame/meta */
  NvDsUcxAmDesc am_desc_meta;        /* am client descs for meta */
  guint num_surfaces_per_frame; /* Number of surfaces in a frame */
  NvDsUcxSrcPrepAmFunc prep_am;
  NvDsUcxSrcPrepAmMetaFunc prep_am_meta;
  bool is_eos;

  GstBufferPool *pool;          /* buffer pool */
  GstTaskPool *thread_pool;
  GstUcxThreadDataSource *threads_data;
} GstUcxSrc;

typedef struct _GstUcxSinkAmHdrParams
{
  void *am_hdr_buf;
  size_t am_hdr_buf_size;
  void *am_hdr_meta_buf;
  size_t am_hdr_meta_buf_size;
} GstUcxSinkAmHdrParams;

typedef struct _GstUcxSinkRdmaParams
{
  void *rdma_frame_buffer;
  size_t rdma_frame_buffer_size;
  void *rdma_meta_buffer;
  size_t rdma_meta_buffer_size;
} GstUcxRdmaParams;

typedef struct _GstUcxSink
{
  GstBaseSink base_ucxsink;
  GstNvDsUcxBufType buf_type;   /* Type of data: Video/Audio/Text */
  guint num_surfaces_per_frame; /* Number of surfaces in a frame */
  GstUcxSinkAmHdrParams *am_hdrs;       /* Active Message header buffer */
  int (*prep_am) (GstBuffer * gst_buffer, GstMapInfo * map_info,
      void **am_hdr_buf, void **rdma_buffer,
      size_t *rdma_buffer_size, NvDsBatchMeta * batch_meta, int batch_id,
      guint num_surfaces_per_frame, size_t rdma_meta_buffer_size);
  int (*prep_am_meta) (void **am_hdr_meta_buf,
      void **rdma_buffer, size_t *rdma_buffer_size,
      NvDsBatchMeta * batch_meta, int batch_id);
  gchar *caps;                  /* caps for new connection */
  guint gpu_id;                 /* GPU device ID */
  gint nvbuf_batch_size;        /* NvBufSurface maximal batch size */
  GstTaskPool *thread_pool;
  GstUcxThreadDataSink *threads_data;
} GstUcxSink;

int get_thread_num (gpointer ucxdata, bool is_src, bool is_server);
gboolean gst_ucx_is_recv_meta (GstUcxSrc * src, int batch_id);
void free_gst_ucp_conn_ctx (gpointer data);
void clear_gst_ucp_conn_ctx (gpointer data);
bool gst_ucx_server_set_property (GstUcxServer * ucxserver, guint property_id,
    const GValue * value, GParamSpec * pspec);
bool gst_ucx_client_set_property (GstUcxClient * ucxclient, guint property_id,
    const GValue * value, GParamSpec * pspec);
bool gst_ucx_src_set_property (GstUcxSrc * ucxsrc, guint property_id,
    const GValue * value, GParamSpec * pspec);
bool gst_ucx_sink_set_property (GstUcxSink * ucxsink, guint property_id,
    const GValue * value, GParamSpec * pspec);
bool gst_ucx_server_get_property (GstUcxServer * ucxserver, guint property_id,
    GValue * value, GParamSpec * pspec);
bool gst_ucx_client_get_property (GstUcxClient * ucxclient, guint property_id,
    GValue * value, GParamSpec * pspec);
bool gst_ucx_src_get_property (GstUcxSrc * ucxsrc, guint property_id,
    GValue * value, GParamSpec * pspec);
bool gst_ucx_sink_get_property (GstUcxSink * ucxsink, guint property_id,
    GValue * value, GParamSpec * pspec);
void gst_ucx_server_init (GstUcxServer * ucxserver);
void gst_ucx_sink_init (GstUcxSink * ucxsink);
void gst_ucx_src_init (GstUcxSrc * ucxsrc);
void gst_ucx_client_init (GstUcxClient * ucxclient);
gboolean gst_ucx_src_negotiate_caps (GstBaseSrc * src, GstUcxSrc * ucxsrc);
gboolean gst_ucx_sink_cap_event_handling (GstUcxSink * ucxsink,
    GstBaseSink * basesink, GstEvent * event, NvDsUcxAmEventDesc * event_data,
    gboolean * send_event);
void gst_ucx_set_nvevent_data (GstEvent * event,
    NvDsUcxAmEventDesc * event_data, gboolean * send_event);

int start_server (gpointer data, bool is_src);
int start_client (gpointer data, bool is_src);
int start_sink (GstUcxSink * ucxsink);
int start_src (GstUcxSrc * ucxsrc, GstBaseSrc * src);

void stop_client (GstUcxClient * ucxclient);
void stop_server (GstUcxServer * ucxserver);
void stop_sink (GstUcxSink * ucxsink);
void stop_src (GstUcxSrc * ucxsrc);

void gst_ucx_init ();
gboolean gst_ucx_src_buffer_pool_new (GstUcxSrc * ucxsrc, GstCaps * caps);

int ucx_src_init_buffer_type_attr (GstUcxSrc * ucxsrc);

int gst_ucx_create_thread_pool (GstTaskPool ** pool, gint nvbuf_batch_size);
gboolean batch_end_check_func (gpointer arg);

G_END_DECLS
/* End of GstPlugin */
/* Buffer handling utils */
// GstBuffer
static void copy_gstbuf_info (NvGstBufferParams * inNvGst,
    GstBuffer * inGstBuf, NvGstBufferParams * outNvGst, GstBuffer * outGstBuf);

// NvBufSurface
static void copy_nvbufsurf_params (NvBufSurfaceParams * in,
    NvBufSurfaceParams * out);

// NvBufAudio
static void copy_nvbufaudio_params (NvBufAudioParams * in,
    NvBufAudioParams * out);
#endif
