/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef __GST_NVSTREAMMUX_H__
#define __GST_NVSTREAMMUX_H__

#include <gst/gst.h>
#include <gst/video/video.h>
#include <time.h>
#include "cuda_runtime_api.h"
#include "nvbufsurface.h"
#include "nvbufsurftransform.h"
#include "nvmultistream_common.h"
#include "../gst-nvmultistream2/gstnvstreammux_ntp.h"

G_BEGIN_DECLS
#define GST_TYPE_NVSTREAMMUX \
  (gst_nvstreammux_get_type ())
#define GST_NVSTREAMMUX(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_NVSTREAMMUX,GstNvStreamMux))
#define GST_NVSTREAMMUX_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_NVSTREAMMUX,GstNvStreamMuxClass))
#define GST_IS_NVSTREAMMUX(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_NVSTREAMMUX))
#define GST_IS_NVSTREAMMUX_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_NVSTREAMMUX))
typedef struct _GstNvStreamMux GstNvStreamMux;
typedef struct _GstNvStreamMuxClass GstNvStreamMuxClass;

typedef enum
{
  BATCH_METHOD_NONE,
  BATCH_METHOD_ROUND_ROBIN,
  BATCH_METHOD_ONE_PER_SOURCE
} GstNvStreamMuxBatchMethod;

typedef struct
{
  guint pad_id;
  gboolean buffer_available=FALSE;
  gboolean buffer_available_pad=FALSE;

  GstVideoInfo in_videoinfo;

  gulong num_bufs_in_current_batch;
  gulong total_bufs_in_current_batch;

  GQueue buf_queue;
  GQueue sync_queue;
  GMutex queue_lock;
  GCond queue_cond;

  gulong curr_frame_no;

  gboolean got_eos;
  gboolean queue_empty;
  gboolean new_pad_added;

  gboolean stopping;

  GstNvDsNtpCalculator *ntp_calc;

  GstBufferPool *int_buf_pool;
  GstSegment segment;

  guint source_id;
  guint sei_frame_id;
  guint last_sei_frame_id;
  guint64 timestamp;
  guint64 last_timestamp;
  GstBuffer *cached_src_buffers = NULL;
  struct timeval cached_time;
  float sim_time;
  guint64 latent_sim_time_s;
  char timestamp_iso8601[30];
  guint64 timestamp_mega;
  guint pad_holding_iterations;  // Per-source failsafe: count holding events for THIS pad
  GstClockTime previous_sei_timestamp;
} GstNvStreamMuxPadData;

typedef struct
{
  NvBufSurfTransformSyncObj_t sync_obj;
  GstBuffer *buf; // Copy of the buffer pointer to tie it up with cooresponding sync object
  GstBuffer *inp_buf; // input buffer. Should be unreffed after tranformation is complete
} GstNvStreamMuxSyncInfo;

struct _GstNvStreamMux
{
  GstElement element;

  GstPad *srcpad;


  gulong num_bufs_in_current_batch;

  GMutex ctx_lock;
  GCond ctx_cond;
  GstFlowReturn last_flow_ret;

  gboolean stop_task;
  gboolean eos_sent;
  gboolean all_pads_eos;
  guint num_pads_eos;
  guint num_queues_empty;
  gboolean flushing;
  gboolean enable_padding;
  gboolean live_source;
  gboolean sys_ts;
  gboolean pad_task_created;
  gboolean enable_adaptive_batch_size;

  GstNvStreamMuxBatchMethod batch_method;
  GstNvMultiStreamPadFrameRates* pad_framerates;
  gint timeout_usec;
  GstClockTime max_latency;
  GstClockTime peer_latency_min;
  GstClockTime peer_latency_max;
  gboolean peer_latency_live;
  gboolean has_peer_latency;
  GstClockID timeout_clk_id;
  gboolean sync_inputs;
  gboolean align_inputs;
  gboolean async_process;
  gboolean no_pipeline_eos;

  GHashTable *pad_indexes;
  GstPad **sink_pads;
  guint current_loc;
  guint batch_size;
  guint current_batch_size;
  guint num_surfaces_per_frame;
  gint cuda_mem_type;
  gint is_integrated;
  gint compute_hw;
  gint interpolation_method;

  GstBufferPool *output_buf_pool;
  cudaStream_t stream;
  GstVideoInfo out_videoinfo;

  cudaStream_t nppStream;

  gulong frame_duration_nsec;
  gulong cur_frame_pts;

  guint width;
  guint height;
  guint gpu_id;

  gboolean sei_based_frameId;

  gboolean query_resolution;

  guint num_extra_bufs;
  guint num_extra_bufs_in_batch;

  GList *event_list;
  GList *latency_metadata_list;
  guint frame_num;
  gboolean segment_sent;
  GstSegment segment;

  guint buffer_pool_size;
  gboolean prev_batch_meta;

  GstClockTime ts_latency_offset;

  gboolean frame_num_reset_on_eos;
  gboolean frame_num_reset_on_stream_reset;
  /** Application specified frame duration used for NTP timestamp calculaion */
  GstClockTime frame_duration;
  gboolean extract_sei_type5_data;
  gboolean extract_sei_sim_time;
  gboolean sort_batch;
  gboolean buffer_cache;
  gboolean is_current_buffer_cached;
  gint buffer_cache_timeout;
  GstBufferPool *black_out_buf_pool;
  GstBuffer *gray_buffer = NULL;
  gboolean align_first_buffer;
  GstClockTime highest_first_pts;
  gboolean all_first_buffers_received;
  gboolean first_batch_aligned;
  gboolean first_batch;
  guint sync_inputs_ntp;
  GstClockTime base_ntp;
  GstClockTime fps;
  gboolean one_time_in_batch=0;
  guint holding_counter;
  GHashTable *processed_pads;
  guint holding_iterations; // Failsafe: count iterations with held buffers
  guint max_holding_iterations; // Failsafe: max iterations before forcing batch push
  guint failsafe_flush_count; // Failsafe: max iterations before forcing batch push
  gboolean drop_backward_sei;
};

struct _GstNvStreamMuxClass
{
  GstElementClass parent_class;
};

G_GNUC_INTERNAL GType gst_nvstreammux_get_type (void);

typedef struct
{
  GstMemory mem;
  NvBufSurface surf;
  GstBuffer **orig_buffer_ptrs;
  guint batchsize;
} GstNvStreamMemory;
GstBufferPool *gst_nvstreammux_buffer_pool_new (guint batch_size, GstCaps *caps, guint pool_size);
GstNvStreamMemory *gst_buffer_get_nvstream_memory (GstBuffer *buf);

G_END_DECLS
#endif
