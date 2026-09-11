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

//#define NEW_METADATA 1

#include "gstnvstreammux.h"
#include "gstnvstreampad.h"

#ifdef NEW_METADATA
#include "gstnvdsmeta.h"
#else
#include "gstnvstreammeta.h"
#include "gstnvdsmeta_int.h"
#endif

#include "nvbufsurftransform.h"
//#include "gstnvbufferpool.h"
#include "gst-nvquery.h"
#include "gst-nvquery-internal.h"
#include "gst-nvmessage.h"
#include "gst-nvevent.h"
#include "gst-nvcommon.h"
#include <stdio.h>
#include <string.h>
#include <npp.h>
#include <sys/time.h>
#include <math.h>
#include "nvtx_helper.h"
#include "nvbufsurface.h"
#include "gstnvdsseimeta.h"
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <iostream>
#include <sstream>

#include "gstnvdsbufferpool.h"
#include "nvds_dewarper_meta.h"
#include "nvds_latency_meta_internal.h"

#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wpointer-arith"
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wuninitialized"

#define MAX_NVBUFFERS 1024

GST_DEBUG_CATEGORY (gst_nvstreammux_debug);
#define GST_CAT_DEFAULT gst_nvstreammux_debug

#define USE_CUDA_BATCH 1
#define PAD_DATA_KEY "pad-data"

#define MAX_BUFFERS_IN_QUEUE 10

#define MIN_POOL_BUFFERS 2
#define MAX_POOL_BUFFERS (MAX_NVBUFFERS)
#define DEFAULT_BUFFER_POOL_SIZE (4)
#define DEFAULT_FAILSAFE_FLUSH_COUNT (DEFAULT_BUFFER_POOL_SIZE)

#define USE_NPPSTREAM
//#define DEBUG_STREAM_MUX //debug//bhushan

#define CEIL(a,b) (((a) + (b) - 1) / (b))

#define COND_BROADCAST(mux) g_cond_broadcast (&mux->ctx_cond); \
  if (mux->timeout_clk_id) \
      gst_clock_id_unschedule (mux->timeout_clk_id);

#define BUF_PTS_TO_RUNNING_TIME(buf) \
    gst_segment_to_running_time(&pad_data->segment, GST_FORMAT_TIME, GST_BUFFER_PTS (buf))

#define _do_init \
    GST_DEBUG_CATEGORY_INIT (gst_nvstreammux_debug, "nvstreammux", 0, "nvstreammux element");
#define gst_nvstreammux_parent_class parent_class
G_DEFINE_TYPE_WITH_CODE (GstNvStreamMux, gst_nvstreammux,
    GST_TYPE_ELEMENT, _do_init);

static gboolean REPEAT_MODE = FALSE;
static gboolean VMS_MODE = FALSE;

static GstStaticPadTemplate nvstreammux_sinkpad_template =
    GST_STATIC_PAD_TEMPLATE ("sink_%u",
    GST_PAD_SINK,
    GST_PAD_REQUEST,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES ("memory:NVMM",
            "{ " "NV12, RGBA, I420 }")));

static GstStaticPadTemplate nvstreammux_srcpad_template =
GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES ("memory:NVMM",
            "{ " "NV12, RGBA, I420 }")));

typedef struct
{
    int64_t frameId;
    int64_t timestamp;
} FrameInfoSeiPayload;

enum
{
  PROP_0,
  PROP_PUSH_MODE_BATCHED,
  PROP_BATCH_SIZE,
  PROP_BATCHED_PUSH_TIMEOUT,
  PROP_WIDTH,
  PROP_HEIGHT,
  PROP_ENABLE_PADDING,
  PROP_QUERY_RESOLUTION,
  PROP_GPU_DEVICE_ID,
  PROP_LIVE_SOURCE,
  PROP_ADAPTIVE_BATCH_SIZE,
  PROP_NUM_SURFACES_PER_FRAME,
  PROP_NVBUF_MEMORY_TYPE,
  PROP_COMPUTE_HW,
  PROP_INTERPOLATION_METHOD,
  PROP_BUFFER_POOL_SIZE,
  PROP_ATTACH_SYS_TIME_STAMP,
  PROP_SYNC_INPUTS,
  PROP_ALIGN_INPUTS,
  PROP_MAX_LATNECY,
  PROP_FRAME_NUM_RESET_ON_EOS,
  PROP_FRAME_NUM_RESET_ON_STREAM_RESET,
  PROP_FRAME_DURATION,
  PROP_ASYNC_PROCESS,
  PROP_NO_PIPELINE_EOS,
  PROP_EXTRACT_SEI_TYPE5_DATA,
  PROP_EXTRACT_SIM_TIME,
  PROP_SORT_BATCH_BUFFERS,
  PROP_CACHE_BUFFERS,
  PROP_CACHE_BUFFERS_TIMEOUT,
  PROP_ALIGN_FIRST_BUFFER,
  PROP_SYNC_INPUTS_NTP,
  PROP_FAILSAFE_FLUSH_COUNT,
  PROP_DROP_BACKWARD_SEI
};

#define DEFAULT_BATCH_METHOD BATCH_METHOD_ROUND_ROBIN
#define DEFAULT_BATCH_SIZE 0
#define DEFAULT_BATCHED_PUSH_TIMEOUT -1
#define DEFAULT_WIDTH 0
#define DEFAULT_HEIGHT 0
#define DEFAULT_QUERY_RESOLUTION FALSE
#define DEFAULT_GPU_DEVICE_ID 0
#define DEFAULT_LIVE_SOURCE FALSE
#define DEFAULT_ATTACH_SYS_TIME_STAMP TRUE
#define DEFAULT_ADAPTIVE_BATCH_SIZE FALSE
#define DEFAULT_FRAME_DURATION GST_CLOCK_TIME_NONE
#define DEFAULT_ASYNC_PROCESS TRUE
#define DEFAULT_NO_PIPELINE_EOS FALSE
#define DEFAULT_SEI_EXTRACT_DATA FALSE
#define DEFAULT_SEI_EXTRACT_SIM_TIME FALSE
#define DEFAULT_SORT_BATCH_BUFFERS FALSE
#define DEFAULT_CACHE_BATCH_BUFFERS FALSE
#define DEFAULT_CACHED_BUFFER_TIMEOUT -1
#define DEFAULT_SYNC_INPUTS_NTP 0 
#define DEFAULT_ALIGN_FIRST_BATCH FALSE
#define DEFAULT_DROP_BACKWARD_SEI FALSE

#define META_BEFORE_NVSTREAMMUX 1
static GQuark _dsmeta_quark = 0;

typedef struct _Input_TS_Identification Input_TS_Identification;

struct _Input_TS_Identification
{
  gdouble in_system_timestamp;
};

static gboolean
gst_nvstreammux_alloc_output_buffers (GstNvStreamMux * mux, GstBufferPool **pool, GstNvStreamMuxPadData *pad_data);
void buf_in_system_timestamp_free(Input_TS_Identification
    * buf_in_system_timestamp_ptr);

// Forward declaration for NTP timestamp extraction (defined later in file)
static GstClockTime get_ntp_timestamp_from_sei_meta(GstBuffer *buffer);

void buf_in_system_timestamp_free(Input_TS_Identification
    *buf_in_system_timestamp_ptr)
{
  g_slice_free (Input_TS_Identification, buf_in_system_timestamp_ptr);
}

static void set_colorimetry_on_buf (GstVideoColorimetry *pad_colorimetry, NvBufSurfaceColorFormat colorFmt)
{
  //Only RGBA NV12 support in muxer, if we do add the support just in case
  switch (colorFmt)
  {
    case NVBUF_COLOR_FORMAT_YUV420:
    case NVBUF_COLOR_FORMAT_NV12:
    case NVBUF_COLOR_FORMAT_NV12_10LE:
      pad_colorimetry->matrix = GST_VIDEO_COLOR_MATRIX_BT601;
      pad_colorimetry->range = GST_VIDEO_COLOR_RANGE_16_235;
      break;
    case NVBUF_COLOR_FORMAT_YUV420_ER:
    case NVBUF_COLOR_FORMAT_NV12_ER:
    case NVBUF_COLOR_FORMAT_NV12_10LE_ER:
      pad_colorimetry->matrix = GST_VIDEO_COLOR_MATRIX_BT601;
      pad_colorimetry->range = GST_VIDEO_COLOR_RANGE_0_255;
      break;
    case NVBUF_COLOR_FORMAT_YUV420_709:
    case NVBUF_COLOR_FORMAT_NV12_709:
    case NVBUF_COLOR_FORMAT_NV12_10LE_709:
      pad_colorimetry->matrix = GST_VIDEO_COLOR_MATRIX_BT709;
      pad_colorimetry->range = GST_VIDEO_COLOR_RANGE_16_235;
      break;
    case NVBUF_COLOR_FORMAT_YUV420_709_ER:
    case NVBUF_COLOR_FORMAT_NV12_709_ER:
    case NVBUF_COLOR_FORMAT_NV12_10LE_709_ER:
      pad_colorimetry->matrix = GST_VIDEO_COLOR_MATRIX_BT709;
      pad_colorimetry->range = GST_VIDEO_COLOR_RANGE_0_255;
      break;
      //Non YUV dont require colorimetry and we dont support
    default:
      break;
  }
#ifdef __aarch64__
  //200629884 - if ER format is received, ER-ER scaling fails for VIC
  // Make it SR
  pad_colorimetry->range = GST_VIDEO_COLOR_RANGE_16_235;
#endif

}

#define GST_TYPE_NVSTREAMMUX_BATCH_METHOD (gst_nvstreammux_batch_method_get_type())

static GType
gst_nvstreammux_batch_method_get_type (void)
{
  static GType nvstreammux_batch_method_type = 0;
  static const GEnumValue nvstreammux_batch_method[] = {
    {BATCH_METHOD_NONE,
          "Buffers will be forwarded as they arrive without any ordering",
        "none"},
    {BATCH_METHOD_ROUND_ROBIN,
          "Round robin method. Loop among all sources to form a batch.\n"
          "\t\t\t\t\t\t   Skip a source if no buffer is received from it.",
        "round-robin"},
    {BATCH_METHOD_ONE_PER_SOURCE,
        "Do not block on client output. Attach"
          "metadata from the next buffer after output is recieved", "blocking"},
    {0, NULL, NULL},
  };

  if (!nvstreammux_batch_method_type) {
    nvstreammux_batch_method_type =
        g_enum_register_static ("GstNvStreamMuxBatchMethod",
        nvstreammux_batch_method);
  }
  return nvstreammux_batch_method_type;
}

static void gst_nvstreammux_src_push_loop (gpointer user_data);

static GstFlowReturn
gst_nvstreammux_push_buffers (GstNvStreamMux * mux, GstBufferList * buf_list)
{
  guint num_fake_buffers = 0;

  while (gst_buffer_list_length (buf_list) < mux->batch_size) {
    GstBuffer *buf = gst_buffer_new ();
    //NvDsMeta *meta = gst_buffer_add_nvds_meta (buf, NULL);
    //meta->meta_type = NV_FAKE_BUFFER;

    num_fake_buffers++;
    gst_buffer_list_add (buf_list, buf);
  }

  GST_DEBUG_OBJECT (mux, "Pushing buffer list %p, length=%u fake=%u", buf_list,
      mux->batch_size, num_fake_buffers);
  return gst_pad_push_list (mux->srcpad, buf_list);
}


static gboolean
gst_nvstreammux_query_latency_unlocked (GstNvStreamMux * self, GstQuery * query)
{
  gboolean query_ret, live;
  GstClockTime our_latency = GST_CLOCK_TIME_NONE, min = GST_CLOCK_TIME_NONE, max = GST_CLOCK_TIME_NONE;
  query_ret = gst_pad_query_default (self->srcpad, GST_OBJECT (self), query);
  if (!query_ret) {
    GST_WARNING_OBJECT (self, "Latency query failed");
    return FALSE;
  }
  gst_query_parse_latency (query, &live, &min, &max);
  if (G_UNLIKELY (!GST_CLOCK_TIME_IS_VALID (min))) {
    GST_ERROR_OBJECT (self, "Invalid minimum latency %" GST_TIME_FORMAT, GST_TIME_ARGS (min));
    return FALSE;
  }
#if 0 //comes from user
  if (self->priv->upstream_latency_min > min) {
    GstClockTimeDiff diff =
        GST_CLOCK_DIFF (min, self->priv->upstream_latency_min);
    min += diff;
    if (GST_CLOCK_TIME_IS_VALID (max)) {
      max += diff;
    }
  }
#endif
  if (min > max && GST_CLOCK_TIME_IS_VALID (max)) {
    GST_ELEMENT_WARNING (self, CORE, CLOCK, (NULL),
        ("Impossible to configure latency: max %" GST_TIME_FORMAT " < min %"
            GST_TIME_FORMAT ". Add queues or other buffering elements.",
            GST_TIME_ARGS (max), GST_TIME_ARGS (min)));
    return FALSE;
  }
  our_latency = self->timeout_usec * 1000;
  self->peer_latency_live = live;
  self->peer_latency_min = min;
  self->peer_latency_max = max;
  self->has_peer_latency = TRUE;
  /* add our own */
  min += our_latency;
#if 0
  min += self->priv->sub_latency_min;
  if (GST_CLOCK_TIME_IS_VALID (self->priv->sub_latency_max)
      && GST_CLOCK_TIME_IS_VALID (max))
    max += self->priv->sub_latency_max + our_latency;
  else
    max = GST_CLOCK_TIME_NONE;
#else
  if (GST_CLOCK_TIME_IS_VALID (max))
          max += our_latency;
#endif
  //SRC_BROADCAST (self);
  GST_DEBUG_OBJECT (self, "configured latency live:%s min:%" G_GINT64_FORMAT
      " max:%" G_GINT64_FORMAT, live ? "true" : "false", min, max);
  gst_query_set_latency (query, live, min, max);
  return query_ret;
}

static GstClockTime
gst_nvstreammux_get_latency_unlocked (GstNvStreamMux *self)
{
  GstClockTime latency = GST_CLOCK_TIME_NONE;
  //g_return_val_if_fail (GST_IS_AGGREGATOR (self), 0);
  if (!self->has_peer_latency)
  {
    GstQuery *query = gst_query_new_latency ();
    gboolean ret;
    ret = gst_nvstreammux_query_latency_unlocked (self, query);
    gst_query_unref (query);
    if (!ret)
      return GST_CLOCK_TIME_NONE;
  }
  if (!self->has_peer_latency || !self->peer_latency_live)
    return GST_CLOCK_TIME_NONE;
  /* latency_min is never GST_CLOCK_TIME_NONE by construction */
  latency = self->peer_latency_min;
  /* add our own */
  latency += self->timeout_usec * 1000;
  return latency;
}

static gboolean
blit_buffer (GstNvStreamMux * mux, GstBuffer * in_buf, GstVideoInfo * in_vinfo, GstBuffer * out_buf, NvBufSurfTransformSyncObj_t *p_sync_obj)
{
  NvBufSurfTransform_Error ret;
  NvBufSurfTransformConfigParams tparams = { static_cast<NvBufSurfTransform_Compute>(mux->compute_hw), static_cast<int32_t>(mux->gpu_id), mux->stream};

  GstMapInfo in_info = GST_MAP_INFO_INIT;
  GstMapInfo out_info = GST_MAP_INFO_INIT;
  NvBufSurface *out_surf, *in_surf;

  gst_buffer_map (in_buf, &in_info, GST_MAP_READ);
  gst_buffer_map (out_buf, &out_info, GST_MAP_READ);


  out_surf = (NvBufSurface *) out_info.data;
  in_surf = (NvBufSurface *) in_info.data;

  if (in_surf == NULL || out_surf == NULL || in_surf->surfaceList->dataPtr == NULL)  {
    gst_buffer_unmap (in_buf, &in_info);
    gst_buffer_unmap (out_buf, &out_info);
    return FALSE;
  }

  out_surf->numFilled = in_surf->numFilled;

  /* Evaluate both checks unconditionally to avoid skipping side effects
   * (GST_ELEMENT_ERROR) due to short-circuit evaluation of ||. */
  gboolean in_surf_check  = CHECK_NVDS_MEMORY_AND_GPUID(mux, in_surf);
  gboolean out_surf_check = CHECK_NVDS_MEMORY_AND_GPUID(mux, out_surf);
  if (in_surf_check || out_surf_check)  {
    gst_buffer_unmap (in_buf, &in_info);
    gst_buffer_unmap (out_buf, &out_info);
    return FALSE;
  }

  guint src_width = GST_ROUND_DOWN_2 (in_surf->surfaceList->width);
  guint src_height = GST_ROUND_DOWN_2 (in_surf->surfaceList->height);
  guint dest_width = out_surf->surfaceList->width, dest_height = out_surf->surfaceList->height;
  guint dest_x = 0, dest_y = 0;
  NvBufSurfTransform_Error err;

  if (mux->enable_padding) {
    /** Note: Input buffer could be of pixel-aspect-ratio that is not 1/1
     * Example: width=800, height=600 and pixel-aspect-ratio=16/9
     * In such cases, the (dest_x, dest_y) (dest_width, dest_height) calculation
     * shall account for the input PAR.
     * Note: NvStreammux output buffer PAR is always 1/1
     */
    /** Input Pixel Aspect Ratio N/D */
    gdouble src_par_nd = ((gdouble)GST_VIDEO_INFO_PAR_N (in_vinfo)) / GST_VIDEO_INFO_PAR_D (in_vinfo);
    /** Input Pixel Aspect Ratio D/N */
    gdouble src_par_dn = ((gdouble)GST_VIDEO_INFO_PAR_D (in_vinfo)) / GST_VIDEO_INFO_PAR_N (in_vinfo);
    gulong compounded_ratio_numerator = src_width  * GST_VIDEO_INFO_PAR_N (in_vinfo) * dest_height;
    gulong compounded_ratio_denominator = dest_width  * GST_VIDEO_INFO_PAR_D (in_vinfo) * src_height;
    gboolean is_proportion = compounded_ratio_numerator == compounded_ratio_denominator;

    /* Calculate the destination width and height required to maintain
     * the aspect ratio. */
    double hdest = out_surf->surfaceList->width * src_height * src_par_dn / (double) src_width;
    double wdest = out_surf->surfaceList->height * src_width * src_par_nd / (double) src_height;
    int pixel_size;
    cudaError_t cudaReturn;

    if (hdest <= out_surf->surfaceList->height) {
      dest_width = out_surf->surfaceList->width;
      dest_height = hdest;
    } else {
      dest_width = wdest;
      dest_height = out_surf->surfaceList->height;
    }
    dest_width = GST_ROUND_DOWN_2 (dest_width);
    dest_height = GST_ROUND_DOWN_2 (dest_height);
    dest_x = GST_ROUND_DOWN_2 ((out_surf->surfaceList->width - dest_width) / 2);
    dest_y = GST_ROUND_DOWN_2 ((out_surf->surfaceList->height - dest_height) / 2);

    /* Pad the scaled image with black color w.r.t color format */
    if (!is_proportion) {
      switch (out_surf->surfaceList->colorFormat) {
        case NVBUF_COLOR_FORMAT_RGBA:
          NvBufSurfaceMemSet (out_surf, 0, 0, 0 );
          break;
        case NVBUF_COLOR_FORMAT_YUV420:
        case NVBUF_COLOR_FORMAT_YUV420_709:
        case NVBUF_COLOR_FORMAT_YUV420_2020:
          NvBufSurfaceMemSet (out_surf, 0, 0, 16 );
          NvBufSurfaceMemSet (out_surf, 0, 1, 128 );
          NvBufSurfaceMemSet (out_surf, 0, 2, 128 );
          break;
        case NVBUF_COLOR_FORMAT_YUV420_ER:
        case NVBUF_COLOR_FORMAT_YUV420_709_ER:
          NvBufSurfaceMemSet (out_surf, 0, 0, 0 );
          NvBufSurfaceMemSet (out_surf, 0, 1, 128 );
          NvBufSurfaceMemSet (out_surf, 0, 2, 128 );
          break;
        case NVBUF_COLOR_FORMAT_NV12:
        case NVBUF_COLOR_FORMAT_NV12_709:
        case NVBUF_COLOR_FORMAT_NV12_2020:
          NvBufSurfaceMemSet (out_surf, 0, 0, 16 );
          NvBufSurfaceMemSet (out_surf, 0, 1, 128 );
          break;
        case NVBUF_COLOR_FORMAT_NV12_ER:
        case NVBUF_COLOR_FORMAT_NV12_709_ER:
          NvBufSurfaceMemSet (out_surf, 0, 0, 0 );
          NvBufSurfaceMemSet (out_surf, 0, 1, 128 );
          break;
        default:
          break;
      }
    }
  }

  if (NvBufSurfTransformSetSessionParams(&tparams) != NvBufSurfTransformError_Success)
    return FALSE;

  NvBufSurfTransformParams transform_params = { 0 };

  NvBufSurfTransformRect src_rect[4]=
  {
	{0,0, in_surf->surfaceList->width, in_surf->surfaceList->height},
	{0,0, in_surf->surfaceList->width, in_surf->surfaceList->height},
	{0,0, in_surf->surfaceList->width, in_surf->surfaceList->height},
	{0,0, in_surf->surfaceList->width, in_surf->surfaceList->height},
  };
  NvBufSurfTransformRect dst_rect[4]= {
    {dest_y, dest_x, dest_width, dest_height},
    {dest_y, dest_x, dest_width, dest_height},
    {dest_y, dest_x, dest_width, dest_height},
    {dest_y, dest_x, dest_width, dest_height},
  };
  transform_params.transform_flag = NVBUFSURF_TRANSFORM_FILTER | NVBUFSURF_TRANSFORM_CROP_DST;
  transform_params.transform_flip = NvBufSurfTransform_None;
  transform_params.transform_filter = static_cast<NvBufSurfTransform_Inter>(mux->interpolation_method);
  transform_params.src_rect = src_rect;
  transform_params.dst_rect = dst_rect;

  if(mux->async_process)
  {
    ret = NvBufSurfTransformAsync (in_surf, out_surf, &transform_params, p_sync_obj);
    GST_DEBUG("Called NvBufSurfTransformAsync");
  }
  else
    ret = NvBufSurfTransform (in_surf, out_surf, &transform_params);
  if (ret != NvBufSurfTransformError_Success)
  {
    gst_buffer_unmap (in_buf, &in_info);
    gst_buffer_unmap (out_buf, &out_info);
    return FALSE;
  }

  gst_buffer_unmap (in_buf, &in_info);
  gst_buffer_unmap (out_buf, &out_info);

  if (!gst_buffer_copy_into (out_buf, in_buf,
      (GstBufferCopyFlags)(GST_BUFFER_COPY_TIMESTAMPS | GST_BUFFER_COPY_META), 0, -1)) {
    GST_ELEMENT_ERROR (mux, STREAM, FAILED,
        ("Failed to copy GstMeta and timestamps when scaling in muxer"), (nullptr));
    return FALSE;
  }

  return TRUE;
}


static GstFlowReturn
gst_nvstreammux_chain (GstPad * pad, GstObject * parent, GstBuffer * buffer)
{
  GstMeta *gst_meta = NULL;
  NvDsMeta *meta = NULL;
  gpointer state = NULL;
#ifndef NEW_METADATA
  FrameInfoMeta_Params * frameinfo = NULL;
#endif
  gboolean needs_conversion = FALSE;
  GstMapInfo in_info = GST_MAP_INFO_INIT;
  NvBufSurface *in_surf;
  GstNvStreamMux *mux = GST_NVSTREAMMUX (parent);
  GstNvStreamMuxSyncInfo *sync_info = NULL;

  gst_buffer_map (buffer, &in_info, GST_MAP_READ);
  in_surf = (NvBufSurface *) in_info.data;
  gst_buffer_unmap (buffer, &in_info);
  if (in_surf->numFilled != mux->num_surfaces_per_frame)
  {
    GST_ELEMENT_ERROR (mux, STREAM, FAILED,
        ("Input buffer number of surfaces (%d) must be equal to mux->num_surfaces_per_frame (%d)\n"
         "\tSet nvstreammux property num-surfaces-per-frame appropriately\n",
        in_surf->numFilled, mux->num_surfaces_per_frame), (nullptr));
    return GST_FLOW_ERROR;
  }

  GstNvStreamMuxPadData *pad_data =
      (GstNvStreamMuxPadData *) g_object_get_data (G_OBJECT (pad),
      PAD_DATA_KEY);

  GST_DEBUG_OBJECT(mux, "Got buffer %p from source %d pts = %" GST_TIME_FORMAT, buffer, pad_data->pad_id, GST_TIME_ARGS(GST_BUFFER_PTS(buffer)));

  if (mux->drop_backward_sei || mux->extract_sei_sim_time) {
    GstClockTime current_sei_ts = get_ntp_timestamp_from_sei_meta (buffer);

    if (mux->extract_sei_sim_time && !GST_CLOCK_TIME_IS_VALID (current_sei_ts)) {
      GST_WARNING_OBJECT (mux,
          "Dropping buffer from pad %d: invalid/missing SEI timestamp",
          pad_data->pad_id);
      /*printf ("Dropping buffer from pad %d: invalid/missing SEI timestamp\n",
          pad_data->pad_id);*/
      gst_buffer_unref (buffer);
      return GST_FLOW_OK;
    }

    if (mux->drop_backward_sei &&
        GST_CLOCK_TIME_IS_VALID (current_sei_ts) &&
        GST_CLOCK_TIME_IS_VALID (pad_data->previous_sei_timestamp) &&
        (gint64) current_sei_ts < (gint64) pad_data->previous_sei_timestamp) {
      GST_WARNING_OBJECT (mux,
          "Dropping buffer from pad %d: SEI timestamp %" G_GUINT64_FORMAT
          " went backwards (previous %" G_GUINT64_FORMAT ")",
          pad_data->pad_id, current_sei_ts, pad_data->previous_sei_timestamp);
      /*printf ("Dropping buffer from pad %d: SEI timestamp %" G_GUINT64_FORMAT
          " went backwards (previous %" G_GUINT64_FORMAT ")\n",
          pad_data->pad_id, current_sei_ts, pad_data->previous_sei_timestamp);*/
      gst_buffer_unref (buffer);
      return GST_FLOW_OK;
    }

    if (mux->drop_backward_sei && GST_CLOCK_TIME_IS_VALID (current_sei_ts)) {
      pad_data->previous_sei_timestamp = current_sei_ts;
    }
  }

  // Extract NTP timestamp early to track from arrival
  GstClockTime arrival_ntp = GST_CLOCK_TIME_NONE;
  if (mux->sync_inputs_ntp > 0) {
    // TODO: Function scope issue - will extract NTP in collect_buffers instead
    // arrival_ntp = get_ntp_timestamp_from_sei_meta(buffer);
    arrival_ntp = GST_CLOCK_TIME_NONE;
  }

  // Track first buffer from each pad only when sync-inputs-ntp is enabled
  if (mux->sync_inputs_ntp > 0) {
  // Use dynamic allocation based on actual number of sink pads
  static gboolean *first_buffer_seen = NULL;
  static GstClockTime *first_ntp_per_pad = NULL;
  static guint first_buffers_count = 0;
  static guint allocated_pad_count = 0;

  GstElement *element = GST_ELEMENT(mux);
  guint current_pad_count = element->numsinkpads;

  // Allocate or reallocate arrays if needed
  if (allocated_pad_count < current_pad_count) {
    first_buffer_seen = (gboolean*)g_realloc(first_buffer_seen, current_pad_count * sizeof(gboolean));
    first_ntp_per_pad = (GstClockTime*)g_realloc(first_ntp_per_pad, current_pad_count * sizeof(GstClockTime));

    // Initialize new elements
    for (guint i = allocated_pad_count; i < current_pad_count; i++) {
      first_buffer_seen[i] = FALSE;
      first_ntp_per_pad[i] = GST_CLOCK_TIME_NONE;
    }
    allocated_pad_count = current_pad_count;
  }

  if (pad_data->pad_id < current_pad_count && !first_buffer_seen[pad_data->pad_id]) {
    first_buffer_seen[pad_data->pad_id] = TRUE;
    first_ntp_per_pad[pad_data->pad_id] = arrival_ntp;
    first_buffers_count++;

    // First buffer tracking for race condition fix

    // If we have first buffers from multiple pads, show the differences
    if (first_buffers_count > 1) {
      for (guint i = 0; i < current_pad_count; i++) {
        if (first_buffer_seen[i] && i != pad_data->pad_id) {
          gint64 ntp_diff = (gint64)(arrival_ntp - first_ntp_per_pad[i]);
          // NTP timestamp difference calculated for race condition analysis
        }
      }
    }
  }
  }

  // Race condition fix applied in NTP extraction function

  Input_TS_Identification *buf_in_TS_ptr = NULL;
  if(nvds_enable_latency_measurement || nvds_latency_measurement_silent)
  {
    buf_in_TS_ptr = g_slice_new0 (Input_TS_Identification);
    buf_in_TS_ptr->in_system_timestamp = nvds_get_current_system_timestamp();
  }

  //g_print("************LATENCY IN = %lf\n", nvds_get_current_system_timestamp());

  g_mutex_lock (&mux->ctx_lock);
  if (pad_data->got_eos) {
    mux->num_pads_eos--;
    if (mux->enable_adaptive_batch_size)
      mux->current_batch_size = MAX(MIN (mux->batch_size,
          GST_ELEMENT(mux)->numsinkpads - mux->num_pads_eos), 1);

    if (pad_data->queue_empty)
      mux->num_queues_empty--;
    pad_data->got_eos = FALSE;
    pad_data->queue_empty = FALSE;
  }
  mux->all_pads_eos = FALSE;
  g_mutex_unlock (&mux->ctx_lock);

#ifndef NEW_METADATA
  while ((gst_meta = gst_buffer_iterate_meta (buffer, &state)))
  {
    meta = (NvDsMeta *) gst_meta;
    if (meta->meta_type == NV_FRAME_INFO_META)
    {
      frameinfo = (FrameInfoMeta_Params *) (meta->meta_data);
      //g_print (" MUX cam-id %d pad-id %d\n", frameinfo->camera_id, pad_data->pad_id);
    }
  }
#endif

  if (!USE_CUDA_BATCH) {
#ifndef NEW_METADATA
    GstNvStreamMeta *meta;
    meta = gst_buffer_add_nvstream_meta (buffer, 1);
    if (meta) {
      meta->stream_id[0] = pad_data->pad_id;
      if (frameinfo)
      {
        meta->camera_id[0] = frameinfo->camera_id;
      }
      // TODO - Add surface_type related changes
      // TODO - Add num_surfaces_per_frame setting
      meta->buf_pts[0] = GST_BUFFER_PTS (buffer);
      meta->num_filled = 1;
      meta->is_valid[0] = FALSE;
    }
#endif
  }

  if (pad_data->in_videoinfo.width != (gint) mux->width)
    needs_conversion = TRUE;
  if (pad_data->in_videoinfo.height != (gint) mux->height)
    needs_conversion = TRUE;
  if (pad_data->in_videoinfo.finfo->format != mux->out_videoinfo.finfo->format)
    needs_conversion = TRUE;

  if(needs_conversion) {
      if(mux->width != GST_ROUND_UP_8(mux->width))
      {
        mux->width = GST_ROUND_UP_8(mux->width);
        GST_ELEMENT_WARNING (mux, LIBRARY, SETTINGS,
            ("Rounding muxer output width to the next multiple of 8: %d", mux->width),
            (NULL));
      }

      if(mux->height != GST_ROUND_UP_4(mux->height))
      {
        mux->height = GST_ROUND_UP_4(mux->height);
        GST_ELEMENT_WARNING (mux, LIBRARY, SETTINGS,
            ("Rounding muxer output height to the next multiple of 4: %d", mux->height),
            (NULL));
      }
  }

  if(!needs_conversion)
  {
    /* The memory type check should compare with different types according to
     * the platform. For Jetson (integrated-gpu) it should compare with
     * Surface_Array whereas for DGPU usecase it should compare with
     * CUDA_Device. As DEFAULT memtype is different for both.
    */
    NvBufSurfaceMemType default_memtype = (mux->is_integrated)?NVBUF_MEM_SURFACE_ARRAY:NVBUF_MEM_CUDA_DEVICE;

    gboolean check_is_default =  (((gint) in_surf->memType == (gint) NVBUF_MEM_DEFAULT
      && (gint) mux->cuda_mem_type == (gint) default_memtype) || ((gint) in_surf->memType ==
      (gint) default_memtype && (gint) mux->cuda_mem_type == (gint) NVBUF_MEM_DEFAULT));

    if ((gint)in_surf->memType != (gint)mux->cuda_mem_type && !check_is_default) {
      GST_DEBUG ("Mismatch between configured memory type and i/p buffer memory type. ip_surf : %s ; muxer : %s. Hence, memory copy will be performed.",
           gst_nvbuf_memory_get_name(in_surf->memType), gst_nvbuf_memory_get_name(mux->cuda_mem_type));
           needs_conversion = TRUE;
    }
  }

  if (needs_conversion) {

    GstBuffer *int_buf;
    GstFlowReturn ret;
    NvBufSurfTransformSyncObj_t sync_obj = NULL;

    if (!pad_data->int_buf_pool){

      set_colorimetry_on_buf(&pad_data->in_videoinfo.colorimetry,
                             in_surf->surfaceList[0].colorFormat);
      g_mutex_lock(&mux->ctx_lock);
      if (!gst_nvstreammux_alloc_output_buffers(mux, &pad_data->int_buf_pool, pad_data))
      {
        g_free(buf_in_TS_ptr);
        buf_in_TS_ptr = NULL;
        g_mutex_unlock(&mux->ctx_lock);
        return GST_FLOW_ERROR;
      }
      g_mutex_unlock(&mux->ctx_lock);
    }

    ret = gst_buffer_pool_acquire_buffer (pad_data->int_buf_pool, &int_buf, NULL);
    if (int_buf == NULL) {
      g_print ("===== error acquiring buffer %s===\n", gst_flow_get_name (ret));
      g_free(buf_in_TS_ptr);
      buf_in_TS_ptr = NULL;
      return GST_FLOW_ERROR;
    }
    if (!blit_buffer (mux, buffer, &pad_data->in_videoinfo, int_buf, &sync_obj)) {
      g_free(buf_in_TS_ptr);
      buf_in_TS_ptr = NULL;
      return GST_FLOW_ERROR;
    }
    if (sync_obj){
      sync_info = (GstNvStreamMuxSyncInfo *)g_malloc0(sizeof(GstNvStreamMuxSyncInfo));
      sync_info->inp_buf = buffer;
      sync_info->buf = int_buf;
      sync_info->sync_obj = sync_obj;
    }
    else
    {
      gst_buffer_unref (buffer);
      sync_info = NULL;
    }
    buffer = int_buf;
  }
  else {
    GstMapInfo in_info = GST_MAP_INFO_INIT;
    NvBufSurface *in_surf;
    gst_buffer_map (buffer, &in_info, GST_MAP_READ);
    gst_buffer_unmap (buffer, &in_info);
    in_surf = (NvBufSurface *) in_info.data;

    if (in_surf == NULL) {
      g_free(buf_in_TS_ptr);
      buf_in_TS_ptr = NULL;
      return GST_FLOW_ERROR;
    }
  }

  g_mutex_lock (&pad_data->queue_lock);
  guint queue_len_before = g_queue_get_length(&pad_data->buf_queue);
  gboolean is_empty_before = g_queue_is_empty(&pad_data->buf_queue);
  //g_print("[CHAIN-QUEUE] pad_id=%d: BEFORE queue - len=%u, is_empty=%d (MAX=%d)\n",
          //pad_data->pad_id, queue_len_before, is_empty_before, MAX_BUFFERS_IN_QUEUE);

  if (!pad_data->stopping
      && queue_len_before > MAX_BUFFERS_IN_QUEUE) {
    //g_print("[CHAIN-BLOCKED] pad_id=%d: Queue FULL (%u > %d), WAITING for space...\n",
            //pad_data->pad_id, queue_len_before, MAX_BUFFERS_IN_QUEUE);
    g_cond_wait (&pad_data->queue_cond, &pad_data->queue_lock);
    //g_print("[CHAIN-UNBLOCKED] pad_id=%d: Wait completed, queue_len now = %u\n",
            //pad_data->pad_id, g_queue_get_length(&pad_data->buf_queue));
  }


  if(nvds_enable_latency_measurement || nvds_latency_measurement_silent) {
    gst_mini_object_set_qdata((GstMiniObject *)buffer,
        g_quark_from_string(GST_ELEMENT_NAME(mux)), buf_in_TS_ptr,
        (GDestroyNotify)buf_in_system_timestamp_free);
  }
  if (pad_data->stopping)
  {
    //g_print("[CHAIN-DROP] pad_id=%d: Pad STOPPING, dropping buffer %p (UNREF #1)\n",
            //pad_data->pad_id, buffer);
    gst_buffer_unref (buffer);
    if(sync_info){
      if(sync_info->sync_obj){
        NvBufSurfTransformSyncObjWait (sync_info->sync_obj, -1);
        NvBufSurfTransformSyncObjDestroy (&(sync_info->sync_obj));
      }
      gst_buffer_unref(sync_info->inp_buf);
      g_free(sync_info);
    }
  }
  else
  {
    g_queue_push_tail (&pad_data->buf_queue, buffer);
    if(sync_info)
      g_queue_push_tail (&pad_data->sync_queue, sync_info);

    guint queue_len_after = g_queue_get_length(&pad_data->buf_queue);
    gboolean is_empty_after = g_queue_is_empty(&pad_data->buf_queue);
    gpointer head_after = g_queue_peek_head(&pad_data->buf_queue);
    gpointer tail_after = g_queue_peek_tail(&pad_data->buf_queue);

    //g_print("[CHAIN-QUEUED] pad_id=%d: Buffer %p QUEUED, queue_len=%u, is_empty=%d, head=%p, tail=%p\n",
            //pad_data->pad_id, buffer, queue_len_after, is_empty_after, head_after, tail_after);

    // VERIFICATION: After queueing, queue should NOT be empty and should contain our buffer
    if (is_empty_after) {
      //g_print("[CHAIN-QUEUED-ERROR] pad_id=%d: Queue is_empty=TRUE after queuing buffer %p!\n",
              //pad_data->pad_id, buffer);
    }
    if (queue_len_after == 0) {
      //g_print("[CHAIN-QUEUED-ERROR] pad_id=%d: queue_len=0 after queuing buffer %p!\n",
              //pad_data->pad_id, buffer);
    }
    if (head_after == NULL) {
      //g_print("[CHAIN-QUEUED-ERROR] pad_id=%d: head=NULL after queuing buffer %p!\n",
              //pad_data->pad_id, buffer);
    }
    if (tail_after != buffer) {
      //g_print("[CHAIN-QUEUED-WARNING] pad_id=%d: tail=%p does not match queued buffer %p\n",
              //pad_data->pad_id, tail_after, buffer);
    }
  }
  g_mutex_unlock (&pad_data->queue_lock);

  //g_print("[CHAIN-EXIT] pad_id=%d: Returning flow_ret=%d\n", pad_data->pad_id, mux->last_flow_ret);

  g_mutex_lock (&mux->ctx_lock);
  //g_print("[CHAIN-SIGNAL] pad_id=%d: Broadcasting to wake up collection loop\n", pad_data->pad_id);
  COND_BROADCAST (mux);
  g_mutex_unlock (&mux->ctx_lock);

  return mux->last_flow_ret;
}

static void
gst_nvstreammux_reset_pad_data (GstNvStreamMuxPadData * pad_data)
{
  pad_data->got_eos = FALSE;
  pad_data->queue_empty = FALSE;
  pad_data->num_bufs_in_current_batch = 0;
  pad_data->stopping = TRUE;
  if (pad_data->int_buf_pool)
    g_object_unref(pad_data->int_buf_pool);
  pad_data->int_buf_pool = NULL;
  pad_data->num_bufs_in_current_batch = 0;
  pad_data->total_bufs_in_current_batch = 0;
  pad_data->curr_frame_no = 0;
  if (pad_data->ntp_calc) {
    gst_nvds_ntp_calculator_free (pad_data->ntp_calc);
  }
  pad_data->ntp_calc = NULL;

  g_mutex_lock (&pad_data->queue_lock);
  g_cond_broadcast (&pad_data->queue_cond);
  g_mutex_unlock (&pad_data->queue_lock);

  g_usleep (100);

  g_mutex_lock (&pad_data->queue_lock);
  while (!g_queue_is_empty (&pad_data->buf_queue)) {
    gst_buffer_unref (GST_BUFFER (g_queue_pop_head (&pad_data->buf_queue)));
  }
  while (!g_queue_is_empty (&pad_data->sync_queue)) {
    GstNvStreamMuxSyncInfo *sync_info = (GstNvStreamMuxSyncInfo *)g_queue_pop_head (&pad_data->sync_queue);
    g_assert(sync_info != NULL);
    g_assert(sync_info->sync_obj != NULL);
    NvBufSurfTransformSyncObjWait (sync_info->sync_obj, -1); // Question: Is 0 the correct value here??
    NvBufSurfTransformSyncObjDestroy (&(sync_info->sync_obj));
    gst_buffer_unref(sync_info->inp_buf);
    g_free((void*)sync_info);
  }
  g_cond_broadcast (&pad_data->queue_cond);
  g_mutex_unlock (&pad_data->queue_lock);
}

static void
gst_nvstreammux_free_pad_data (GstNvStreamMuxPadData * data)
{
  gst_nvstreammux_reset_pad_data (data);
  if (data->ntp_calc) {
    gst_nvds_ntp_calculator_free (data->ntp_calc);
  }

  g_mutex_clear (&data->queue_lock);
  g_cond_clear (&data->queue_cond);
  if (data)
  {
    g_free (data);
    data = NULL;
  }
}

static gboolean
gst_nvstreammux_alloc_output_buffers (GstNvStreamMux * mux, GstBufferPool **pool, GstNvStreamMuxPadData *pad_data)
{
  GstStructure *config;
  GstVideoInfo info;
  GstAllocator *allocator;
  GstVideoColorimetry colorimetry = mux->out_videoinfo.colorimetry;
  GstCaps *caps;

  mux->out_videoinfo.colorimetry = pad_data->in_videoinfo.colorimetry;
  caps=gst_video_info_to_caps (&mux->out_videoinfo);
  mux->out_videoinfo.colorimetry = colorimetry;

#if 0
//    GST_ERROR_OBJECT (mux, "Could not parse video info from caps %p, %" GST_PTR_FORMAT, caps, caps);
  if (!gst_video_info_from_caps (&info, caps)) {
    //GST_ERROR_OBJECT (mux, "Could not parse video info from caps %p, %s", caps, gst_caps_to_string);
//    return FALSE;
  }
#endif

  if (*pool) {
    gst_buffer_pool_set_active (*pool, FALSE);
    gst_object_unref (*pool);
  }

#if 0
  *pool = gst_nvm_buffer_pool_new ();

  config = gst_buffer_pool_get_config (mux->output_buf_pool);
  gst_buffer_pool_config_set_params (config, caps, sizeof (NvBufSurface *),
      MIN_POOL_BUFFERS, DEFAULT_BUFFER_POOL_SIZE);
#endif

#if 0
  allocator =
      gst_nvstream_allocator_new (info.width *
      GST_VIDEO_FORMAT_INFO_PSTRIDE (info.finfo, 0), info.width, info.height,
      info.size, mux->batch_size, CUDA_DEVICE_MEMORY, FALSE, mux->gpu_id);
  gst_buffer_pool_config_set_allocator (config, allocator, &allocation_params);

  gst_buffer_pool_set_config (mux->output_buf_pool, config);
  gst_buffer_pool_set_active (mux->output_buf_pool, TRUE);

  gst_object_unref (allocator);

  /* create the buffer pool configuration structure */
  config = gst_structure_new ("buf-pool/config0",
      "width", G_TYPE_UINT,
      info.width,
      "height", G_TYPE_UINT,
      info.height,
      "max-buffers", G_TYPE_UINT, DEFAULT_BUFFER_POOL_SIZE,
      "size", G_TYPE_UINT, info.size,
      "memtype", G_TYPE_UINT, mux->cuda_mem_type,
      "num-batched-buffers", G_TYPE_UINT, mux->batch_size,
      "device", G_TYPE_POINTER, mux,
      "gpu_id", G_TYPE_UINT, mux->gpu_id,
      "free-func", G_TYPE_POINTER, NULL,
      "name", G_TYPE_POINTER, GST_ELEMENT_NAME(mux),
      NULL);

  if (!gst_buffer_pool_set_config (*pool, config)) {
    GST_WARNING ("Nvstreammux bufferpool configuration failed");
    return FALSE;
  }

  gboolean is_active = gst_buffer_pool_set_active (*pool, TRUE);
  if (!is_active) {
    GST_WARNING (" Failed to allocate the buffers inside the Nvstreammux output pool");
    return FALSE;
  } else {
    GST_DEBUG (" Output buffer pool (%p) successfully created with %d buffers",
        *pool, DEFAULT_BUFFER_POOL_SIZE);
  }

#endif
      *pool = gst_nvds_buffer_pool_new();

      config = gst_buffer_pool_get_config (*pool);
      gst_buffer_pool_config_set_params (config, caps, sizeof (NvBufSurface), mux->buffer_pool_size, mux->buffer_pool_size);

      gst_structure_set (config,
          "memtype", G_TYPE_UINT, mux->cuda_mem_type,
          "gpu-id", G_TYPE_UINT, mux->gpu_id,
          "batch-size", G_TYPE_UINT, mux->num_surfaces_per_frame,
          "clear-chroma", G_TYPE_BOOLEAN, TRUE, NULL);

      if (!gst_buffer_pool_set_config (*pool, config)) {
        gst_caps_unref (caps);
        return FALSE;
      }
      gst_caps_unref (caps);

      gboolean is_active = gst_buffer_pool_set_active (*pool, TRUE);
      if (!is_active) {
        //GST_WARNING (" Failed to allocate the buffers inside the Nvstreammux output pool");
         GST_ELEMENT_ERROR (mux, RESOURCE, FAILED,
             ("Failed to allocate the buffers inside the Nvstreammux output pool"),
             (nullptr));
        return FALSE;
      } else {
        GST_DEBUG (" Output buffer pool (%p) successfully created with %d buffers",
            *pool, mux->buffer_pool_size);
      }

  return TRUE;
}

static GstCaps *
remove_width_height (GstCaps * caps, gboolean has_nvmm)
{
  GstCaps *ret_caps = gst_caps_new_empty ();
  guint i, n;

  if (gst_caps_is_any (caps)) {
    gst_caps_unref (ret_caps);
    return gst_caps_copy (caps);
  }

  n = gst_caps_get_size (caps);
  for (i = 0; i < n; i++) {
    GstStructure *str = gst_caps_get_structure (caps, i);

    if (i > 0 &&
        gst_caps_is_subset_structure_full (
          ret_caps, gst_caps_get_structure (caps, i),
          gst_caps_get_features (caps, i)))
      continue;

    str = gst_structure_copy (str);
    gst_structure_remove_fields (str, "width", "height", "pixel-aspect-ratio",
        "chroma-site", "interlace-mode", "colorimetry", NULL);
    gst_structure_set (str, "width", GST_TYPE_INT_RANGE, 0, G_MAXINT, NULL);
    gst_structure_set (str, "height", GST_TYPE_INT_RANGE, 0, G_MAXINT, NULL);
    if (!has_nvmm) {
      gst_caps_append_structure_full (ret_caps, str, NULL);
      str = gst_structure_copy (str);
    }
    gst_caps_append_structure_full (ret_caps, str,
        gst_caps_features_from_string ("memory:NVMM"));
  }

  return ret_caps;
}

static gboolean
caps_has_nvmm (GstCaps *caps){
  gint i,  n = gst_caps_get_size (caps);
  GstCapsFeatures *tft;
  for (i = 0; i < n; i++) {
    tft = gst_caps_get_features (caps, i);
      if (gst_caps_features_get_size (tft)){
        return TRUE;
      }
  }

  return FALSE;
}

static GstCaps *
add_nvmm (GstCaps * caps, gboolean has_nvmm)
{
  GstCaps *ret_caps = gst_caps_new_empty ();
  guint i = 0, n = 0;

  if (gst_caps_is_any (caps)) {
    gst_caps_unref (ret_caps);
    return caps;
  }

  n = gst_caps_get_size (caps);
  for (i = 0; i < n; i++) {
    GstCapsFeatures *features = gst_caps_get_features(caps, i);
    if (gst_caps_features_contains(features, "memory:NVMM")) {
      gst_caps_unref (ret_caps);
      return caps;
    }
    GstStructure *str = gst_caps_get_structure (caps, i);

    if (i > 0 &&
        gst_caps_is_subset_structure_full (
          ret_caps, gst_caps_get_structure (caps, i),
          gst_caps_get_features (caps, i)))
      continue;

    str = gst_structure_copy(str);

    gst_caps_append_structure_full(ret_caps, str,
        gst_caps_features_from_string("memory:NVMM"));
  }
  gst_caps_unref (caps);

  return ret_caps;
}

static GstCaps *
remove_framerate_memory_feautures (GstCaps * caps, gboolean has_nvmm)
{
  GstCaps *ret_caps = gst_caps_new_empty ();
  guint i, n;

  if (gst_caps_is_any (caps)) {
    gst_caps_unref (ret_caps);
    return gst_caps_copy (caps);
  }

  n = gst_caps_get_size (caps);
  for (i = 0; i < n; i++) {
    GstStructure *str = gst_caps_get_structure (caps, i);

    if (i > 0 &&
        gst_caps_is_subset_structure_full (
          ret_caps, gst_caps_get_structure (caps, i),
          gst_caps_get_features (caps, i)))
      continue;

    str = gst_structure_copy (str);
    gst_structure_remove_fields (str, "framerate", "colorimetry", "chroma-site",
        "batch-size", "nvbuf-memory-type", "gpu-id", NULL);
    gst_structure_set (str, "framerate", GST_TYPE_FRACTION_RANGE, 0, G_MAXINT,
        G_MAXINT, 1, NULL);

    if (!has_nvmm) {
       gst_caps_append_structure_full (ret_caps, str, NULL);
       str = gst_structure_copy (str);
    }
    gst_caps_append_structure_full (ret_caps, str,
        gst_caps_features_from_string ("memory:NVMM"));
  }

  return ret_caps;
}

static gboolean
gst_nvstreammux_src_query (GstPad * pad, GstObject * parent, GstQuery * query)
{
  GstNvStreamMux *mux = GST_NVSTREAMMUX (parent);

#ifdef USE_NPPSTREAM
  if (gst_nvquery_is_nppstream (query)) {
    if (mux->nppStream == NULL) {
      cudaStreamCreate (&mux->nppStream);
      //nppSetStream (mux->nppStream);
    }
#ifdef USE_COMMON_NPPSTREAM
    gst_nvquery_nppstream_set (query, mux->nppStream);
    return TRUE;
#else
    return FALSE;
#endif
  }
#endif

  if (gst_nvquery_is_batch_size (query)) {
    gst_nvquery_batch_size_set (query, mux->batch_size);
    return TRUE;
  }

  if (gst_nvquery_is_numStreams_size (query)) {

    gst_nvquery_numStreams_size_set (query, GST_ELEMENT (mux)->numsinkpads);
    return TRUE;
  }

  if (gst_nvquery_is_uri_from_streamid(query)) {
    guint streamid;
    gchar pad_name[32];
    if(!gst_nvquery_uri_from_streamid_parse_streamid(query, &streamid)) {
      return FALSE;
    }
    g_snprintf(pad_name, sizeof(pad_name), "sink_%u", streamid);

    for (GList *iter = GST_ELEMENT(mux)->sinkpads; iter; iter = iter->next) {
      if (!g_strcmp0(pad_name, GST_PAD_NAME(iter->data))) {
        return gst_pad_peer_query(GST_PAD(iter->data), query);
      }
    }
    return FALSE;
  }

  return gst_pad_query_default (pad, parent, query);
}

static gboolean
gst_nvstreammux_sink_query (GstPad * pad, GstObject * parent, GstQuery * query)
{
  GstNvStreamMux *mux = GST_NVSTREAMMUX (parent);
  GstNvStreamMuxPadData *pad_data =
      (GstNvStreamMuxPadData *) g_object_get_data (G_OBJECT (pad),
      PAD_DATA_KEY);

  if (gst_nvquery_is_batch_size (query)) {
    gst_nvquery_batch_size_set (query, 1);
    return TRUE;
  }
  if (gst_nvquery_is_ntp_sync (query)) {
    _NtpData ntpdata;

    g_mutex_lock (&mux->ctx_lock);
    gst_nvquery_ntp_sync_parse (query, &ntpdata);
    if (!pad_data->ntp_calc)
            pad_data->ntp_calc =
                gst_nvds_ntp_calculator_new (mux->sys_ts ? GST_NVDS_NTP_CALC_MODE_SYSTEM_TIME :
                        GST_NVDS_NTP_CALC_MODE_RTCP, mux->frame_duration,
                        GST_ELEMENT (mux), pad_data->pad_id);
    if (pad_data->ntp_calc)
      gst_nvds_ntp_calculator_add_ntp_sync_values (pad_data->ntp_calc,
              ntpdata.ntp_time_epoch_ns, ntpdata.frame_timestamp,
              ntpdata.avg_frame_time);

    g_mutex_unlock (&mux->ctx_lock);
    return TRUE;
  }

  g_mutex_lock (&mux->ctx_lock);
  if (GST_QUERY_TYPE (query) == GST_QUERY_CAPS) {
    GstCaps *ret_caps = NULL;
    GstCaps *caps = NULL;
    GstCaps *tmp_caps = NULL;
    GstCaps *filter = NULL;
    gboolean has_nvmm = FALSE;
    gst_query_parse_caps (query, &filter);

    if (filter) {
      has_nvmm = caps_has_nvmm (filter);
    }
    if (gst_pad_has_current_caps (mux->srcpad)) {
      ret_caps = gst_pad_get_current_caps (mux->srcpad);
    } else {
      GstQuery *peer_query = gst_query_new_caps (filter);
      if (gst_pad_peer_query (mux->srcpad, peer_query)) {
        gst_query_parse_caps_result (peer_query, &ret_caps);

        if (gst_caps_is_any (ret_caps) || gst_caps_is_empty (ret_caps)) {
          ret_caps = gst_pad_get_pad_template_caps (pad);
        }
      } else {
        ret_caps = gst_pad_get_pad_template_caps (pad);
      }
      // Take a ref as unreffing the query will delete caps.
      gst_caps_ref (ret_caps);
      gst_query_unref (peer_query);
    }

    tmp_caps = gst_pad_get_pad_template_caps (pad);
    ret_caps = add_nvmm(ret_caps, has_nvmm);
    caps = gst_caps_intersect_full (ret_caps, tmp_caps, GST_CAPS_INTERSECT_FIRST);
    gst_caps_unref (ret_caps);
    gst_caps_unref (tmp_caps);
    ret_caps = caps;

    caps = remove_framerate_memory_feautures (ret_caps, has_nvmm);
    gst_caps_unref (ret_caps);
    ret_caps = caps;

    if (mux->width && mux->height) {
      caps = remove_width_height (ret_caps, has_nvmm);
      gst_caps_unref (ret_caps);
      ret_caps = caps;
    }

    if (mux->query_resolution) {
      GstQuery *query = gst_nvquery_resolution_new ();
      if (gst_pad_peer_query (mux->srcpad, query)) {
        caps = remove_width_height (ret_caps, has_nvmm);
        gst_caps_unref (ret_caps);
        ret_caps = caps;
      }
      gst_query_unref (query);
    }

    if (filter) {
      caps = gst_caps_intersect (ret_caps, filter);
      gst_caps_unref (ret_caps);
      ret_caps = caps;
    }

    gst_query_set_caps_result (query, ret_caps);
    gst_caps_unref (ret_caps);
    g_mutex_unlock (&mux->ctx_lock);
    return TRUE;
  }

  if (GST_QUERY_TYPE (query) == GST_QUERY_ACCEPT_CAPS) {
    GstCaps *acaps;
    GstQuery *peer_query;
    gboolean result = FALSE;

    gst_query_parse_accept_caps (query, &acaps);


    acaps = gst_caps_copy (acaps);
    gst_caps_set_features (acaps, 0,
        gst_caps_features_from_string ("memory:NVMM"));

    if (mux->width && mux->height) {
      GstStructure *str = gst_caps_get_structure (acaps, 0);
      gst_structure_set (str, "width", G_TYPE_INT, mux->width,
          "height", G_TYPE_INT, mux->height, NULL);
    } else if (mux->query_resolution) {
      GstQuery *query = gst_nvquery_resolution_new ();
      if (gst_pad_peer_query (mux->srcpad, query)) {
        GstStructure *str = gst_caps_get_structure (acaps, 0);

        gst_structure_set (str, "width", G_TYPE_INT, 1,
            "height", G_TYPE_INT, 1, NULL);
      }
    }
    {
        GstStructure *str = gst_caps_get_structure (acaps, 0);
        gst_structure_remove_fields (str, "nvbuf-memory-type", "gpu-id",NULL);
    }

    peer_query = gst_query_new_accept_caps (acaps);
    if (gst_pad_peer_query (mux->srcpad, peer_query)) {
      gst_query_parse_accept_caps_result (peer_query, &result);
    }
    gst_query_unref (peer_query);
    gst_caps_unref (acaps);

    gst_query_set_accept_caps_result (query, result);
    g_mutex_unlock (&mux->ctx_lock);
    return TRUE;

  }
  g_mutex_unlock (&mux->ctx_lock);
  return gst_pad_query_default (pad, parent, query);
}

static void
handle_eos_event(GstNvStreamMux *mux, GstNvStreamMuxPadData *pad_data)
{
  GstElement *element = GST_ELEMENT (mux);
  g_mutex_lock (&pad_data->queue_lock);
  g_queue_push_tail (&pad_data->buf_queue, gst_event_new_eos());
  g_mutex_unlock (&pad_data->queue_lock);
  g_mutex_lock (&mux->ctx_lock);
  if (pad_data->got_eos) {
    COND_BROADCAST (mux);
    g_mutex_unlock (&mux->ctx_lock);
    return ;
  }
/*Commented this code, to be considered later if any case where stream specific EOS is not sent*/
#if 0
  if(pad_data->buffer_available==FALSE){
     GstEvent *event_eos = gst_nvevent_new_stream_eos (pad_data->pad_id);
     mux->event_list = g_list_prepend (mux->event_list, event_eos);
     pad_data->buffer_available=TRUE;
  }
#endif
  pad_data->got_eos = TRUE;
  mux->num_pads_eos++;
  mux->all_pads_eos = (mux->num_pads_eos == GST_ELEMENT(mux)->numsinkpads);
  if (mux->all_pads_eos && !mux->pad_task_created) {
    if (mux->no_pipeline_eos == FALSE){
      gst_pad_push_event (mux->srcpad, gst_event_new_eos ());
      mux->eos_sent = TRUE;
    }
  }
  if (mux->enable_adaptive_batch_size)
    mux->current_batch_size = MAX(MIN (mux->batch_size, element->numsinkpads - mux->num_pads_eos), 1);
  g_print("nvstreammux: Successfully handled EOS for source_id=%d\n", pad_data->source_id);
  COND_BROADCAST (mux);
  g_mutex_unlock (&mux->ctx_lock);
}

static gboolean
gst_nvstreammux_src_event (GstPad * pad, GstObject * parent, GstEvent * event)
{
  GstNvStreamMux *mux = GST_NVSTREAMMUX (parent);
  GstElement *element = GST_ELEMENT (parent);
  gboolean ret = TRUE;
  GstClockTime latency = GST_CLOCK_TIME_NONE;

  if (GST_EVENT_TYPE (event) == GST_EVENT_LATENCY)
  {
      gst_event_parse_latency (event, &latency);

      GST_DEBUG_OBJECT (mux, "latency %" GST_TIME_FORMAT, GST_TIME_ARGS (latency));

      mux->ts_latency_offset = latency;
  }
  gst_event_unref (event);
  return ret;
}

static GstFlowReturn create_black_frame(GstNvStreamMux *mux, GstBuffer **gray_buffer) {
    GstFlowReturn ret;
    GstBuffer *cached_out_buf = NULL;
    NvBufSurface *surf = NULL;
    GstMapInfo cached_out_buf_map = GST_MAP_INFO_INIT;

    ret = gst_buffer_pool_acquire_buffer(mux->black_out_buf_pool, &cached_out_buf, NULL);
    if (ret != GST_FLOW_OK) {
        GST_ERROR_OBJECT(mux, "Failed to acquire buffer from pool");
        return ret;
    }

    gst_buffer_map (cached_out_buf, &cached_out_buf_map, GST_MAP_READ);
    surf = (NvBufSurface *) cached_out_buf_map.data;
    surf->numFilled=1;
    surf->batchSize=1;

    switch (surf->surfaceList->colorFormat) {
      case NVBUF_COLOR_FORMAT_RGBA:
        NvBufSurfaceMemSet (surf, 0, 0, 200 );
        break;
      case NVBUF_COLOR_FORMAT_YUV420:
      case NVBUF_COLOR_FORMAT_YUV420_709:
      case NVBUF_COLOR_FORMAT_YUV420_2020:
        NvBufSurfaceMemSet (surf, 0, 0, 160 );
        NvBufSurfaceMemSet (surf, 0, 1, 200 );
        NvBufSurfaceMemSet (surf, 0, 2, 180 );
        break;
      case NVBUF_COLOR_FORMAT_YUV420_ER:
      case NVBUF_COLOR_FORMAT_YUV420_709_ER:
        NvBufSurfaceMemSet (surf, 0, 0, 160 );
        NvBufSurfaceMemSet (surf, 0, 1, 200 );
        NvBufSurfaceMemSet (surf, 0, 2, 200 );
        break;
      case NVBUF_COLOR_FORMAT_NV12:
      case NVBUF_COLOR_FORMAT_NV12_709:
      case NVBUF_COLOR_FORMAT_NV12_2020:
        NvBufSurfaceMemSet (surf, 0, 0, 127 );
        NvBufSurfaceMemSet (surf, 0, 1, 160 );
        break;
      case NVBUF_COLOR_FORMAT_NV12_ER:
      case NVBUF_COLOR_FORMAT_NV12_709_ER:
        NvBufSurfaceMemSet (surf, 0, 0, 127 );
        NvBufSurfaceMemSet (surf, 0, 1, 160 );
        break;
      default:
        break;
    }
    *gray_buffer = cached_out_buf;

    gst_buffer_unmap (cached_out_buf, &cached_out_buf_map);

    return GST_FLOW_OK;
}

static gboolean
gst_nvstreammux_sink_event (GstPad * pad, GstObject * parent, GstEvent * event)
{
  GstNvStreamMux *mux = GST_NVSTREAMMUX (parent);
  GstElement *element = GST_ELEMENT (parent);
  GstNvStreamMuxPadData *pad_data =
      (GstNvStreamMuxPadData *) g_object_get_data (G_OBJECT (pad),
      PAD_DATA_KEY);
  gboolean ret = TRUE;

  switch (GST_EVENT_TYPE (event)) {
    case GST_EVENT_CAPS:
    {
      GstCaps *caps = NULL;
      GstQuery *caps_query;
      GstStructure *caps_str;
      gboolean needs_conversion = FALSE;
      GstCapsFeatures *nvfeatures = gst_caps_features_from_string ("memory:NVMM");

      {
        GstQuery *query = gst_nvquery_sourceid_new();
        guint sourceid;
        if (gst_pad_peer_query(pad, query) &&
            gst_nvquery_sourceid_parse(query, &sourceid)) {
          pad_data->source_id = sourceid;
        }
        gst_query_unref (query);
      }

      gst_event_parse_caps (event, &caps);

      if (!gst_caps_features_is_equal (gst_caps_get_features (caps, 0),
            nvfeatures)) {
        gst_caps_features_free (nvfeatures);
        GST_ELEMENT_ERROR (mux, STREAM, WRONG_TYPE,
            ("NvStreamMux does not suppport raw buffers. Use nvvideoconvert "
             "before NvStreamMux to convert to NVMM buffers"), (nullptr));
        return FALSE;
      }
      gst_caps_features_free (nvfeatures);

      caps_str = gst_caps_get_structure (caps, 0);

      g_mutex_lock (&mux->ctx_lock);
      if (!gst_pad_has_current_caps (mux->srcpad)) {
        if (caps) {
          GstEvent *event;
          GstStructure *str;
          GstCapsFeatures *features =
              gst_caps_features_from_string ("memory:NVMM");

          caps = gst_caps_copy (caps);
          str = gst_caps_get_structure (caps, 0);
          gst_caps_set_features (caps, 0, features);

          gst_video_info_init (&pad_data->in_videoinfo);
          gst_video_info_from_caps (&pad_data->in_videoinfo, caps);
          GST_DEBUG_OBJECT (mux,"parse video info from caps %" GST_PTR_FORMAT, caps);

          if (gst_structure_has_field (str, "pixel-aspect-ratio"))
            gst_structure_remove_field (str, "pixel-aspect-ratio");
          if (gst_structure_has_field (str, "chroma-site"))
            gst_structure_remove_field (str, "chroma-site");
          if (gst_structure_has_field (str, "interlace-mode"))
            gst_structure_remove_field (str, "interlace-mode");
          if (gst_structure_has_field (str, "colorimetry"))
            gst_structure_remove_field (str, "colorimetry");


          if (mux->width && mux->height) {
            GstStructure *str = gst_caps_get_structure (caps, 0);
            gst_structure_set (str, "width", G_TYPE_INT, mux->width,
                "height", G_TYPE_INT, mux->height,
                "batch-size", G_TYPE_INT, mux->batch_size,
                "num-surfaces-per-frame", G_TYPE_INT, mux->num_surfaces_per_frame,
                NULL);
          } else if (mux->query_resolution) {
            GstQuery *query = gst_nvquery_resolution_new ();
            if (gst_pad_peer_query (mux->srcpad, query)) {
              GstStructure *str = gst_caps_get_structure (caps, 0);

              gst_nvquery_resolution_parse (query, &mux->width, &mux->height);
              gst_structure_set (str, "width", G_TYPE_INT, mux->width,
                  "height", G_TYPE_INT, mux->height, NULL);
            }
            gst_query_unref (query);
          }
          {
            GstCaps *peer_caps = gst_pad_peer_query_caps(mux->srcpad, NULL);
            GstStructure *peer_structure;
            int n=gst_caps_get_size(peer_caps);
            const gchar *out_mem_type_string = NULL;
            bool peer_has_gpu_id = false;
            if(n>0)
            {
                peer_caps = gst_caps_truncate(peer_caps);
                GST_DEBUG_OBJECT (mux, "peer caps %" GST_PTR_FORMAT, peer_caps);
                peer_structure = gst_caps_get_structure (peer_caps, 0);
                out_mem_type_string = gst_structure_get_string (peer_structure, "nvbuf-memory-type");
                peer_has_gpu_id = gst_structure_has_field(peer_structure, "gpu-id");
            }
            if(!out_mem_type_string)
            {
                // set "nvbuf-memory-type"
                int mem_type = mux->cuda_mem_type;
                if(mem_type == NVBUF_MEM_DEFAULT)
                    GET_DEFAULT_MEM_TYPE(mem_type);
                g_assert(mem_type != NVBUF_MEM_DEFAULT);
                gst_structure_set (str, "nvbuf-memory-type", G_TYPE_STRING , gst_nvbuf_memory_get_name(mem_type), NULL);
            }else
            {
                mux->cuda_mem_type = gst_nvbuf_memory_get_value(out_mem_type_string);
                if(mux->cuda_mem_type <= NVBUF_MEM_DEFAULT)
                    GST_ERROR_OBJECT(mux, "Incorrect nvbuf-memory-type set on src pad!!");
                GST_WARNING_OBJECT(mux, "nvbuf-memory-type property is set based on SRC caps. Property config setting (if any) is overridden!!");
                gst_structure_set (str, "nvbuf-memory-type", G_TYPE_STRING , out_mem_type_string, NULL);
            }
            if(peer_has_gpu_id)
            {
                gst_structure_get_int(peer_structure, "gpu-id", (int *)&mux->gpu_id);
                GST_WARNING_OBJECT(mux, "gpu-id property is set based on SRC caps. Property config setting (if any) is overridden!!");
            }
            gst_structure_set (str, "gpu-id", G_TYPE_INT , mux->gpu_id, NULL);
            gst_caps_unref(peer_caps);
        }
          gst_video_info_init (&mux->out_videoinfo);
          gst_video_info_from_caps (&mux->out_videoinfo, caps);

          if (mux->out_videoinfo.fps_d == 0)
            mux->out_videoinfo.fps_d = 1;
          if (mux->out_videoinfo.fps_n == 0)
            mux->out_videoinfo.fps_n = 30;

          if (!mux->sync_inputs)
          {
              mux->frame_duration_nsec =
                  1000000000UL * mux->out_videoinfo.fps_d /
                  mux->out_videoinfo.fps_n;
          }
          else
          {
              if (mux->timeout_usec >
                      (gint)(1000000UL * mux->out_videoinfo.fps_d / mux->out_videoinfo.fps_n))
              {
                  mux->timeout_usec =
                      (1000000UL * mux->out_videoinfo.fps_d / mux->out_videoinfo.fps_n);
                  printf ("Adjusting muxer's batch push timeout based on FPS of fastest source to %d\n",
                          mux->timeout_usec);
              }
              mux->frame_duration_nsec = mux->timeout_usec * 1000;
          }

          event = gst_event_new_caps (caps);
          gst_pad_push_event (mux->srcpad, event);

#if 0
          if (USE_CUDA_BATCH
              && !gst_nvstreammux_alloc_output_buffers (mux, caps)) {
            g_mutex_unlock (&mux->ctx_lock);
            return FALSE;
          }

  if (mux->num_surfaces_per_frame == 0) {
    GstQuery *query = gst_nvquery_num_surfaces_per_buffer_new ();

    if (!gst_pad_peer_query (GST_PAD (GST_ELEMENT(mux)->sinkpads->data), query) ||
        !gst_nvquery_num_surfaces_per_buffer_parse (query, &mux->num_surfaces_per_frame))
    {
      GST_DEBUG_OBJECT (mux, "*** MUX -- > num_surface_per_frame query failed... Using num_surface_per_frame = 1\n");
      mux->num_surfaces_per_frame = 1;
    }
    else
        GST_DEBUG_OBJECT (mux, "MUX -- > num_surface_per_frame after query is %d\n", mux->num_surfaces_per_frame);
  }

  if (mux->batch_size % mux->num_surfaces_per_frame != 0) {
    GST_ELEMENT_ERROR (mux, LIBRARY, SETTINGS,
        ("Muxer batch-size (%d) not a multiple of number of dewarped surfaces (%d)",
        mux->batch_size, mux->num_surfaces_per_frame), NULL);
    return FALSE;
  }
#endif
          if (!mux->stop_task) {
            if (!mux->output_buf_pool) {
              mux->output_buf_pool  = gst_nvstreammux_buffer_pool_new (mux->batch_size, caps, mux->buffer_pool_size);
              gst_buffer_pool_set_active (mux->output_buf_pool, TRUE);
              if (mux->buffer_cache && !mux->black_out_buf_pool) {
                mux->black_out_buf_pool = gst_nvds_buffer_pool_new();

                if (!gst_nvstreammux_alloc_output_buffers(mux, &mux->black_out_buf_pool, pad_data)) {
                  GST_ERROR_OBJECT (mux, "Failed to allocate black_out_buf_pool\n");
                  return GST_FLOW_ERROR;
                }
                GstFlowReturn ret_flow = create_black_frame(mux, &mux->gray_buffer);
                if (ret_flow != GST_FLOW_OK) {
                  GST_ERROR_OBJECT (mux, "Failed to create gray buffer\n");
                  return GST_STATE_CHANGE_FAILURE;
                }
              }
            }
            if (mux->batch_method != BATCH_METHOD_NONE) {
              mux->pad_task_created =
                  gst_pad_start_task (mux->srcpad,
                          gst_nvstreammux_src_push_loop, mux, NULL);
            }
          }
          gst_caps_unref (caps);
        }
      } else {
        gst_video_info_init (&pad_data->in_videoinfo);
        gst_video_info_from_caps (&pad_data->in_videoinfo, caps);

        if (pad_data->in_videoinfo.finfo->format != mux->out_videoinfo.finfo->format)
        {
          g_mutex_unlock (&mux->ctx_lock);
          GST_ELEMENT_ERROR (mux, STREAM, FORMAT, ("Input Output Color Format Mismatch"), (nullptr));
          return FALSE;
        }

        GST_DEBUG_OBJECT (mux,"parse video info from caps %" GST_PTR_FORMAT, caps);
        g_mutex_unlock (&mux->ctx_lock);
        g_mutex_lock (&pad_data->queue_lock);
        while(!g_queue_is_empty (&pad_data->buf_queue)) {
          g_cond_wait (&pad_data->queue_cond, &pad_data->queue_lock);
          //g_usleep(1000);
        }
        g_mutex_unlock (&pad_data->queue_lock);

        g_mutex_lock (&mux->ctx_lock);
        //g_mutex_unlock (&mux->ctx_lock);
      }

      if (pad_data->in_videoinfo.width != (gint) mux->width)
        needs_conversion = TRUE;
      if (pad_data->in_videoinfo.height != (gint) mux->height)
        needs_conversion = TRUE;
      if (pad_data->in_videoinfo.finfo->format != mux->out_videoinfo.finfo->format)
        needs_conversion = TRUE;

      if (needs_conversion) {
        GST_INFO_OBJECT (mux, "Conversion enabled for source %d", pad_data->pad_id);
      }

     /* if (needs_conversion && !pad_data->int_buf_pool) {
        if (!gst_nvstreammux_alloc_output_buffers(mux, &pad_data->int_buf_pool, pad_data)){
          g_mutex_unlock (&mux->ctx_lock);
          return FALSE;
        }
      }
      */
      if (gst_structure_has_field (caps_str, "framerate")) {
        GstStructure *str =
            gst_structure_new ("stream-frame-rate", "stream-id", G_TYPE_UINT,
            pad_data->pad_id, NULL);
        gst_structure_set_value (str, "frame-rate",
            gst_structure_get_value (caps_str, "framerate"));

        /* Store framerate for particular pad_id to hashtable and pass it downstream via CUSTOM_QUERY */
        /* This should be made thread-safe by using the lock associated with it */
        GValue *framerate = (GValue *) g_malloc0(sizeof(GValue));
        g_value_init (framerate, GST_TYPE_FRACTION);
        g_value_copy ((GValue *)gst_structure_get_value (caps_str, "framerate"), framerate);

        g_mutex_lock (&(mux->pad_framerates->read_write_lock));
        g_hash_table_insert (mux->pad_framerates->table, pad_data->pad_id + (char *)NULL, framerate);
        g_mutex_unlock (&(mux->pad_framerates->read_write_lock));

        gst_structure_set (str, "nvmultistream-pad-framerates", G_TYPE_POINTER, mux->pad_framerates, NULL);
        caps_query = gst_query_new_custom (GST_QUERY_CUSTOM, str);
        gst_pad_peer_query (mux->srcpad, caps_query);
        gst_query_unref (caps_query);
      }
      g_mutex_unlock (&mux->ctx_lock);

      gst_event_unref (event);

      return TRUE;
    }
      break;
    case GST_EVENT_EOS:
    {
      handle_eos_event(mux, pad_data);
      return ret;
    }
      break;
    case GST_EVENT_SEGMENT:
    {
      const GstSegment *segment = NULL;

      gst_event_parse_segment (event, &segment);
      GST_INFO_OBJECT (mux, "mux got segment from src %d %" GST_SEGMENT_FORMAT, pad_data->pad_id, segment);
      gst_segment_copy_into (segment, &pad_data->segment);

      g_mutex_lock (&pad_data->queue_lock);
      g_queue_push_tail (&pad_data->buf_queue, event);
      g_mutex_unlock (&pad_data->queue_lock);

      g_mutex_lock (&mux->ctx_lock);
      COND_BROADCAST (mux);
      g_mutex_unlock (&mux->ctx_lock);
      return ret;
    }

    case GST_EVENT_FLUSH_START:
      gst_event_unref (event);
      return TRUE;
    case GST_EVENT_FLUSH_STOP:
    {
      g_mutex_lock (&mux->ctx_lock);
      if (pad_data->got_eos) {
        mux->num_pads_eos--;
      }
      pad_data->got_eos = FALSE;
      mux->all_pads_eos = FALSE;
      mux->eos_sent = FALSE;

      if (pad_data->queue_empty) {
        mux->num_queues_empty--;
      }
      pad_data->queue_empty = FALSE;

      COND_BROADCAST (mux);
      g_mutex_unlock (&mux->ctx_lock);
      gst_event_unref (event);
    }
      return TRUE;
      break;
    default:
      break;
  }

  switch ((GstNvEventType)GST_EVENT_TYPE (event)) {
    case GST_NVEVENT_STREAM_RESET:
    {
      g_mutex_lock (&pad_data->queue_lock);
      g_queue_push_tail (&pad_data->buf_queue, gst_event_copy (event));
      g_mutex_unlock (&pad_data->queue_lock);

      g_mutex_lock (&mux->ctx_lock);
      COND_BROADCAST (mux);
      g_mutex_unlock (&mux->ctx_lock);
      gst_event_unref (event);
      return ret;
    }
    break;
    default:
      break;
  }
  ret = gst_pad_push_event (mux->srcpad, event);

  return ret;
}

static GstPad *
gst_nvstreammux_request_new_pad (GstElement * element, GstPadTemplate * templ,
    const gchar * name, const GstCaps * caps)
{
  GstNvStreamMux *mux = GST_NVSTREAMMUX (element);
  GstNvStreamMuxPadData *pad_data;
  GstPad *sinkpad = NULL;
  guint stream_index = 0;
  guint i, n;
  GList *iter;
  GstEvent *new_pad_event = NULL;
  gboolean ret=FALSE;

  GST_DEBUG_OBJECT (element, "Requesting new sink pad");

  if (!name || sscanf (name, "sink_%u", &stream_index) < 1) {
    GST_ERROR_OBJECT (element,
        "Pad should be named 'sink_%%u' when requesting a pad");
    return NULL;
  }
  g_mutex_lock (&mux->ctx_lock);

  if (g_hash_table_lookup (mux->pad_indexes, stream_index + (char *)NULL)) {
    GST_ERROR_OBJECT (element, "Pad named '%s' already requested", name);
    g_mutex_unlock(&mux->ctx_lock);
    return NULL;
  }

  sinkpad = GST_PAD_CAST (g_object_new (GST_TYPE_NVSTREAM_PAD,
          "name", name, "direction", templ->direction, "template", templ,
          NULL));

  g_hash_table_insert (mux->pad_indexes, stream_index + (char *)NULL, sinkpad);
  g_mutex_unlock(&mux->ctx_lock);

  pad_data = (GstNvStreamMuxPadData *)g_malloc0 (sizeof (GstNvStreamMuxPadData));
  pad_data->pad_id = stream_index;
  //pad_data->new_pad_added = TRUE;
  pad_data->sei_frame_id = pad_data->last_sei_frame_id = 0;
  pad_data->source_id = pad_data->pad_id;
  pad_data->pad_holding_iterations = 0;  // Initialize per-source failsafe counter
  pad_data->previous_sei_timestamp = GST_CLOCK_TIME_NONE;

  g_queue_init (&pad_data->buf_queue);
  g_queue_init (&pad_data->sync_queue);
  g_mutex_init (&pad_data->queue_lock);
  g_cond_init (&pad_data->queue_cond);
  g_object_set_data_full (G_OBJECT (sinkpad), PAD_DATA_KEY, pad_data,
      (GDestroyNotify) gst_nvstreammux_free_pad_data);

  gst_pad_set_chain_function (sinkpad,
      GST_DEBUG_FUNCPTR (gst_nvstreammux_chain));

  gst_pad_set_event_function (sinkpad,
      GST_DEBUG_FUNCPTR (gst_nvstreammux_sink_event));

  gst_pad_set_query_function (sinkpad,
      GST_DEBUG_FUNCPTR (gst_nvstreammux_sink_query));

  gst_pad_set_active (sinkpad, TRUE);

  g_mutex_lock (&mux->ctx_lock);
  gst_element_add_pad (element, sinkpad);
  n = element->numsinkpads;
  if (mux->sink_pads){
    g_free(mux->sink_pads);
    mux->sink_pads = NULL;
  }
  mux->sink_pads = (GstPad **)g_malloc0(n*sizeof(GstPad *));
  iter = element->sinkpads;
  for (i=0; i<n; i++){
    mux->sink_pads[i] = (GstPad*)iter->data;
    iter = iter->next;
  }
  mux->all_pads_eos = FALSE;
  COND_BROADCAST (mux);

  if (mux->enable_adaptive_batch_size)
    mux->current_batch_size = MIN (mux->batch_size, n - mux->num_pads_eos);

  GST_DEBUG_OBJECT (mux, "Pad added event sent %d\n", pad_data->pad_id);
  new_pad_event = gst_nvevent_new_pad_added (pad_data->pad_id);
  ret = gst_pad_push_event (mux->srcpad, new_pad_event);
  if(ret==FALSE){
     pad_data->new_pad_added = TRUE;
  }
  else {
     pad_data->new_pad_added = FALSE;
  }

  g_mutex_unlock (&mux->ctx_lock);

  return sinkpad;
}

static void
gst_nvstreammux_release_pad (GstElement * element, GstPad * pad)
{
  GstNvStreamMux *mux = GST_NVSTREAMMUX (element);
  GstNvStreamMuxPadData *data =
      (GstNvStreamMuxPadData *) g_object_get_data (G_OBJECT (pad),
      PAD_DATA_KEY);
  guint i, n;
  GList *iter;
  GstEvent *pad_removed = NULL;

  g_mutex_lock (&data->queue_lock);

  gst_buffer_replace(&data->cached_src_buffers, NULL);

  if(data->buffer_available_pad==FALSE){
    gst_pad_push_event (mux->srcpad, gst_nvevent_new_pad_deleted (data->pad_id));
    data->buffer_available_pad=TRUE;
  }
  else if (GST_STATE(element) >= GST_STATE_PAUSED) {
    pad_removed = gst_nvevent_new_pad_deleted (data->pad_id);
    g_queue_push_tail (&data->buf_queue, pad_removed);
  }
  g_mutex_unlock (&data->queue_lock);

  g_mutex_lock (&mux->ctx_lock);
  COND_BROADCAST (mux);
  g_mutex_unlock (&mux->ctx_lock);

  g_mutex_lock (&data->queue_lock);
  /* Do not wait if the pad task is not running since data from the queue will
   * never be dequeued leading to release_pad getting stuck. */
  while (mux->pad_task_created && g_queue_get_length (&data->buf_queue)) {
    g_cond_wait (&data->queue_cond, &data->queue_lock);
  }
  g_mutex_unlock (&data->queue_lock);

  gst_pad_set_active (pad, FALSE);

  g_mutex_lock (&mux->ctx_lock);
  if (mux->pad_indexes)
    g_hash_table_remove (mux->pad_indexes, data->pad_id + (char *) NULL);

  if (mux->pad_framerates) {
    if (mux->pad_framerates->table) {
      g_mutex_lock (&(mux->pad_framerates->read_write_lock));
      g_hash_table_remove (mux->pad_framerates->table, data->pad_id + (char *) NULL);
      g_mutex_unlock (&(mux->pad_framerates->read_write_lock));
    }
  }

  if (data->got_eos) {
    mux->num_pads_eos--;
    if (data->queue_empty) {
      mux->num_queues_empty--;
    }
  }

  GST_DEBUG_OBJECT (mux, "Pad deleted %d\n", data->pad_id);

  gst_element_remove_pad (GST_ELEMENT_CAST (mux), pad);
  mux->all_pads_eos = (mux->num_pads_eos == element->numsinkpads);
  n = element->numsinkpads;
  if (mux->sink_pads){
    g_free(mux->sink_pads);
    mux->sink_pads = NULL;
  }
  mux->sink_pads = (GstPad **)g_malloc0(n*sizeof(GstPad *));
  iter = element->sinkpads;
  for (i=0; i<n; i++){
    mux->sink_pads[i] = (GstPad*)iter->data;
    iter = iter->next;
  }
  COND_BROADCAST (mux);

  if (mux->enable_adaptive_batch_size)
    mux->current_batch_size = MAX(MIN (mux->batch_size, n - mux->num_pads_eos), 1);

  g_mutex_unlock (&mux->ctx_lock);
}
#ifdef DEBUG_STREAM_MUX
static guint pdest_index;
#endif

static gboolean
copy_data_cuda (GstNvStreamMux * mux, NvBufSurface *dest_surf,
    GstBuffer * src_buffer, GstVideoInfo * in_vinfo, GstNvStreamMuxPadData *pad_data)
{
  GstMapInfo info = GST_MAP_INFO_INIT;
  NvBufSurface *surf;
  guint i = 0;

  if (!gst_buffer_map (src_buffer, &info, GST_MAP_READ)) {
    return FALSE;
  }
  surf = (NvBufSurface *) info.data;
  gst_buffer_unmap (src_buffer, &info);

  if (CHECK_NVDS_MEMORY_AND_GPUID(mux, surf))
    return FALSE;

  for (i=0; i<surf->numFilled; i++)
  {
	  memcpy (dest_surf->surfaceList + dest_surf->numFilled + i, surf->surfaceList + i, sizeof (NvBufSurfaceParams));
  }
  dest_surf->numFilled += i;

#if 0
  gboolean ret = FALSE;
#ifndef DEBUG_STREAM_MUX
  gpointer out_base =
      dest_data + mux->num_bufs_in_current_batch * mux->out_videoinfo.size;
#else
  gpointer out_base = dest_data +pdest_index  * mux->out_videoinfo.size;
#endif
  gboolean is_nvbuf;
  gpointer in_comp_base;
  guint j;
  int cond = 0;


  is_nvbuf = (info.size == sizeof (NvBufSurface));

  if (is_nvbuf) {
    in_comp_base = surf->allocated_mem;

    if (CHECK_NVDS_MEMORY_AND_GPUID (mux, surf)) {
      gst_buffer_unmap (src_buffer, &info);
      return FALSE;
    }
  } else {
    in_comp_base = info.data;
  }

  float x_scale_factor = 1.0;
  float y_scale_factor = 1.0;
  int x_offset = 0;
  int y_offset = 0;

  if (FALSE == mux->enable_padding)
  {
    // Scale without any padding
    cond = 0;
  }
  else
  {
    guint out_width   = GST_VIDEO_INFO_COMP_WIDTH (&mux->out_videoinfo, 0);
    guint out_height  = GST_VIDEO_INFO_COMP_HEIGHT (&mux->out_videoinfo, 0);
    guint out_pstride = GST_VIDEO_FORMAT_INFO_PSTRIDE (mux->out_videoinfo.finfo, 0);

    cudaMemset (out_base, 0, out_width*out_pstride*out_height);
    cudaMemset (out_base +
                out_width*out_height*out_pstride,128,
                mux->out_videoinfo.size - out_width*out_height*out_pstride);

    if ( (GST_VIDEO_INFO_COMP_HEIGHT (in_vinfo, 0) < GST_VIDEO_INFO_COMP_HEIGHT (&mux->out_videoinfo, 0)) &&
        (GST_VIDEO_INFO_COMP_WIDTH (in_vinfo, 0) < GST_VIDEO_INFO_COMP_WIDTH (&mux->out_videoinfo, 0)) )
    {
      // Src W&H < Dst W&H, Case of horizontal and vertical padding
      cond = 3;
    }
    else if ( (GST_VIDEO_INFO_COMP_HEIGHT (in_vinfo, 0) >= GST_VIDEO_INFO_COMP_HEIGHT (&mux->out_videoinfo, 0)) &&
        (GST_VIDEO_INFO_COMP_WIDTH (in_vinfo, 0) >= GST_VIDEO_INFO_COMP_WIDTH (&mux->out_videoinfo, 0)) )
    {
      // Src W&H > Dst W&H, Scale without padding
      cond = 0;
    }
    else if (GST_VIDEO_INFO_COMP_HEIGHT (in_vinfo, 0) > GST_VIDEO_INFO_COMP_HEIGHT (&mux->out_videoinfo, 0))
    {
      // Horizontally centered, src.height > dst.height
      cond = 1;
    }
    else if (GST_VIDEO_INFO_COMP_WIDTH (in_vinfo, 0) > GST_VIDEO_INFO_COMP_WIDTH (&mux->out_videoinfo, 0))
    {
      // Vertically centered, src.width > dst.width
      cond = 2;
    }
  }

  int i = 0;
  int num_bufs_in_current_batch = mux->num_bufs_in_current_batch;

  if ((mux->width != surf->width) || (mux->height != surf->height)) {
    for(i = 0; i < mux->num_surfaces_per_frame; i++)
    {
      in_comp_base = surf->buf_data[i];
      out_base = dest_data + num_bufs_in_current_batch * mux->out_videoinfo.size;

      for (j = 0; j < mux->out_videoinfo.finfo->n_planes; j++) {
        gpointer out_comp_base = out_base + mux->out_videoinfo.offset[j];
        NppStatus status;
        NppiSize oSrcSize = { GST_VIDEO_INFO_COMP_WIDTH (in_vinfo, j),
          GST_VIDEO_INFO_COMP_HEIGHT (in_vinfo, j)
        };
        NppiRect oSrcROI = { 0, 0, GST_VIDEO_INFO_COMP_WIDTH (in_vinfo, j),
          GST_VIDEO_INFO_COMP_HEIGHT (in_vinfo, j)
        };

        NppiRect ooSrcROI =
        { 0, 0, GST_VIDEO_INFO_COMP_WIDTH (&mux->out_videoinfo, j),
          GST_VIDEO_INFO_COMP_HEIGHT (&mux->out_videoinfo, j)
        };

        if (cond == 0) {
          // Fill full output frame
          x_scale_factor = (float) GST_VIDEO_INFO_COMP_WIDTH (&mux->out_videoinfo, j) / GST_VIDEO_INFO_COMP_WIDTH (in_vinfo, j);
          y_scale_factor = (float) GST_VIDEO_INFO_COMP_HEIGHT (&mux->out_videoinfo, j) / GST_VIDEO_INFO_COMP_HEIGHT (in_vinfo, j);
        } else if (cond == 1) {
          // Horizontally centered, keep aspect ratio
          // src.height > dst.height
          x_scale_factor = (float) GST_VIDEO_INFO_COMP_HEIGHT (&mux->out_videoinfo, j) / GST_VIDEO_INFO_COMP_HEIGHT (in_vinfo, j);
          y_scale_factor = (float) GST_VIDEO_INFO_COMP_HEIGHT (&mux->out_videoinfo, j) / GST_VIDEO_INFO_COMP_HEIGHT (in_vinfo, j);
          x_offset = (ooSrcROI.width - (oSrcROI.width*x_scale_factor)) / 2 ;
        } else if (cond == 2) {
          // Vertically centered, keep aspect ratio
          // src.width > dst.width
          x_scale_factor = (float) GST_VIDEO_INFO_COMP_WIDTH (&mux->out_videoinfo, j) / GST_VIDEO_INFO_COMP_WIDTH (in_vinfo, j);
          y_scale_factor = (float) GST_VIDEO_INFO_COMP_WIDTH (&mux->out_videoinfo, j) / GST_VIDEO_INFO_COMP_WIDTH (in_vinfo, j);
          y_offset = (ooSrcROI.height - (oSrcROI.height*x_scale_factor)) / 2 ;
        } else if (cond == 3) {
          // Horizontally & Vertically centered
          x_scale_factor = 1.0;
          y_scale_factor = 1.0;
          x_offset = (ooSrcROI.width - (oSrcROI.width*x_scale_factor)) / 2 ;
          y_offset = (ooSrcROI.height - (oSrcROI.height*x_scale_factor)) / 2 ;
        }

        guint input_pitch = oSrcSize.width *
            GST_VIDEO_FORMAT_INFO_PSTRIDE (in_vinfo->finfo, j);

        if (mux->nppStream)
          nppSetStream (mux->nppStream);

        switch (GST_VIDEO_FORMAT_INFO_PSTRIDE (mux->out_videoinfo.finfo, j)) {
          case 1:
            status =
              nppiResizeSqrPixel_8u_C1R (in_comp_base, oSrcSize,
                  oSrcSize.width, oSrcROI,
                  out_comp_base,
                  ooSrcROI.width, ooSrcROI,
                  x_scale_factor,
                  y_scale_factor,
                  x_offset, y_offset,
                  //NPPI_INTER_LINEAR);
                                        NPPI_INTER_CUBIC);
            break;
          case 3:
            status =
              nppiResizeSqrPixel_8u_C3R (in_comp_base, oSrcSize,
                  oSrcSize.width * 3, oSrcROI,
                  out_comp_base,
                  ooSrcROI.width * 3, ooSrcROI,
                  x_scale_factor,
                  y_scale_factor, x_offset, y_offset, NPPI_INTER_CUBIC);
            break;
          case 2:
            oSrcSize.width /= 2;
            ooSrcROI.width /= 2;
            oSrcROI.width /= 2;
            x_offset /= 2;
          case 4:
            status =
              nppiResizeSqrPixel_8u_C4R (in_comp_base, oSrcSize,
                  oSrcSize.width * 4, oSrcROI,
                  out_comp_base,
                  ooSrcROI.width * 4, ooSrcROI,
                  x_scale_factor,
                  y_scale_factor, x_offset, y_offset, NPPI_INTER_CUBIC);
            break;
          default:
            break;
        }
        in_comp_base += input_pitch * oSrcSize.height;
      }
      num_bufs_in_current_batch++;
    }
#ifdef USE_NPPSTREAM
    if (mux->nppStream)
    {
      cudaStreamSynchronize (mux->nppStream);
    }
#endif
  } else {
    /* set num surfaces per stream to max surfaces , init = 1 */
    for(i = 0; i < mux->num_surfaces_per_frame; i++)
    {
      in_comp_base = surf->buf_data[i];
      out_base = dest_data + num_bufs_in_current_batch * mux->out_videoinfo.size;

      for (j = 0; j < mux->out_videoinfo.finfo->n_planes; j++) {
        guint comp_width = GST_VIDEO_INFO_COMP_WIDTH (&mux->out_videoinfo, j);
        guint comp_height = GST_VIDEO_INFO_COMP_HEIGHT (&mux->out_videoinfo, j);
        guint comp_pixel_stride =
          GST_VIDEO_FORMAT_INFO_PSTRIDE (mux->out_videoinfo.finfo,
              j);
        guint input_pitch = comp_width * comp_pixel_stride;

        gpointer out_comp_base = out_base + mux->out_videoinfo.offset[j];

        cudaMemcpy2DAsync (out_comp_base, comp_width * comp_pixel_stride,
            in_comp_base, input_pitch, comp_width * comp_pixel_stride,
            comp_height,
            (is_nvbuf) ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice,
            mux->stream);
        in_comp_base += input_pitch * comp_height;
      }
      num_bufs_in_current_batch++;
    }
  }

  gst_buffer_unmap (src_buffer, &info);
  return ret;
#endif
  return TRUE;
}
static
void get_sync_obj_and_wait(GstBuffer *buf, GQueue *p_sync_queue, uint32_t time_out)
{
  if (!g_queue_is_empty(p_sync_queue)){
    GstNvStreamMuxSyncInfo *sync_info = (GstNvStreamMuxSyncInfo *)g_queue_peek_head(p_sync_queue);
    // Check in the buffer in "buf_queue" is same as the buffer in "sync_queue"
    if (buf == sync_info->buf){
      //g_print("Popping from sync queue\n");
      sync_info = (GstNvStreamMuxSyncInfo *)g_queue_pop_head(p_sync_queue);
      g_assert(sync_info != NULL);
      g_assert(sync_info->sync_obj != NULL);
      NvBufSurfTransformSyncObjWait(sync_info->sync_obj, time_out);
      NvBufSurfTransformSyncObjDestroy(&(sync_info->sync_obj));
      gst_buffer_unref(sync_info->inp_buf);
      g_free(sync_info);
    }
  }
}

static gboolean gstnvstreammuxAddToBatch(GstNvStreamMux *mux,
      GstNvStreamMuxPadData *pad_data, GstBuffer *src_buffer,
      GstBuffer *out_buf, NvBufSurface *dest_surf,
      NvDsBatchMeta *dest_batch_meta, guint batch_size)
{
  gboolean ret = FALSE;
  NvDsMeta *meta = NULL;
  NvDewarperSurfaceMeta *src_surface_meta = NULL;
  guint surf_cnt = 0;
  NvDsUserMeta *user_latency_meta = NULL;
  NvDsMetaCompLatency *latency_metadata = NULL;
  GstElement *element = GST_ELEMENT (mux);
  guint n = element->numsinkpads;

#if 1
  gpointer state = NULL;
  GstMeta *gst_meta;
  NvDsBatchMeta *batch_meta = NULL;
  GstBufferCopyFlags flags = GST_BUFFER_COPY_META;
  pad_data->timestamp = 0;
  pad_data->sei_frame_id = 0;

  if (mux->extract_sei_type5_data == TRUE)
  {
      GstVideoSEIMeta *meta =
          (GstVideoSEIMeta *) gst_buffer_get_meta (src_buffer, GST_VIDEO_SEI_META_API_TYPE);
      if (!meta)
      {
          GST_DEBUG_OBJECT (mux, "no video SEI meta data retrieved in streammux\n");
          pad_data->sei_frame_id = 0;
          pad_data->timestamp = 0;
      }
      else
      {
          uint32_t total_metadata_size = meta->sei_metadata_size;
          if (meta->sei_metadata_type == (guint)GST_USER_SEI_META)
          {
              FrameInfoSeiPayload *ptr = (FrameInfoSeiPayload *)meta->sei_metadata_ptr;
              pad_data->sei_frame_id = ptr->frameId;
              pad_data->timestamp = ptr->timestamp;
              GST_DEBUG_OBJECT (mux, "nvstreammux : RETRIEVED VIDEO SEI META DATA frame_id = %d\n", pad_data->sei_frame_id);
              mux->sei_based_frameId = TRUE;
          }
      }
  }
  if (mux->extract_sei_sim_time == TRUE && !mux->is_current_buffer_cached)
  {
    GstVideoSEIMeta *meta =
          (GstVideoSEIMeta *) gst_buffer_get_meta (src_buffer, GST_VIDEO_SEI_META_API_TYPE);
    if (!meta)
    {
      GST_ERROR_OBJECT (mux, "no sim time metadata retrieved in streammux\n");
      pad_data->sim_time = 0;
      pad_data->latent_sim_time_s = 0;
    }
    else
    {
      if (meta->sei_metadata_type == (guint)GST_USER_SEI_META)
      {
        GstVideoSEIMeta* meta = (GstVideoSEIMeta*) gst_buffer_get_meta(src_buffer, GST_VIDEO_SEI_META_API_TYPE);
        if (!meta) {
             GST_DEBUG_OBJECT(mux, "no sim time metadata retrieved in streammux\n");
             pad_data->sim_time = 0;
             pad_data->latent_sim_time_s = 0;
        }
        else
        {
          if (meta->sei_metadata_type == (guint)GST_USER_SEI_META)
          {
            unsigned char* jsonContentPtr = (unsigned char*)meta->sei_metadata_ptr;
            size_t jsonContentSize = meta->sei_metadata_size;

            std::string jsonContent(reinterpret_cast<const char*>(jsonContentPtr), jsonContentSize);
            std::stringstream ss(jsonContent);
//          std::cout << "Raw JSON: " << jsonContentPtr << std::endl;

          try {
            boost::property_tree::ptree pt;
            boost::property_tree::read_json(ss, pt);

            pad_data->sim_time = pt.get<float>("sim_time", 0.0f);
            pad_data->latent_sim_time_s = pt.get<uint64_t>("latent_sim_time_s", 0);
            pad_data->sei_frame_id = pt.get<uint64_t>("frame_id", 0);

            // Parse timestamp_iso8601
            std::string timestamp_iso8601 = pt.get<std::string>("timestamp_iso8601", "");
            if (!timestamp_iso8601.empty()) {
              strncpy(pad_data->timestamp_iso8601, timestamp_iso8601.c_str(), sizeof(pad_data->timestamp_iso8601) - 1);
              pad_data->timestamp_iso8601[sizeof(pad_data->timestamp_iso8601) - 1] = '\0';  // Ensure null-termination
            } else {
              pad_data->timestamp_iso8601[0] = '\0';  // Empty string if not present in JSON
            }

            // Parse timestamp_mega
            pad_data->timestamp_mega = pt.get<guint64>("timestamp", 0);
            GST_DEBUG_OBJECT(mux,"RETRIEVED sim_time = %f, latent_sim_time_s = %lu, "
                     "timestamp_iso8601 = %s, timestamp_mega = %lu sei_frame_id=%d\n",
                     pad_data->sim_time, pad_data->latent_sim_time_s,
                     pad_data->timestamp_iso8601, pad_data->timestamp_mega, pad_data->sei_frame_id);
            } catch (const boost::property_tree::json_parser_error& e) {
              GST_DEBUG_OBJECT(mux, "JSON parsing error: %s\n", e.what());
              pad_data->sim_time = 0;
              pad_data->latent_sim_time_s = 0;
              pad_data->timestamp_iso8601[0] = '\0';
              pad_data->timestamp_mega = 0;
	      pad_data->sei_frame_id = 0;
            }
          }
        }
      }
    }
  }

  batch_meta = gst_buffer_get_nvds_batch_meta (src_buffer);

  if(batch_meta)
  {
    mux->prev_batch_meta = 1;
  }

  if(mux->prev_batch_meta == 0)
  {
  while ((gst_meta = gst_buffer_iterate_meta(src_buffer, &state))) {
    if (!gst_meta_api_type_has_tag (gst_meta->info->api, _dsmeta_quark)) {

     if ((!(flags & GST_BUFFER_COPY_MEMORY)
              || (flags & GST_BUFFER_COPY_MERGE))
          && gst_meta_api_type_has_tag (gst_meta->info->api, _gst_meta_tag_memory)) {
        //printf("don't copy memory meta %p of API type %s", gst_meta,
          //  g_type_name (gst_meta->info->api));
      }
      else if (gst_meta->info->transform_func)
      {
        GstMetaTransformCopy copy_data;

        copy_data.region = FALSE;
        copy_data.offset = 0;
        copy_data.size = gst_buffer_get_size (src_buffer);

        if (!gst_meta->info->transform_func (out_buf, gst_meta, src_buffer,
              _gst_meta_transform_copy, &copy_data)) {
          GST_ERROR ("failed to copy meta %p of API type %s", gst_meta,
              g_type_name (gst_meta->info->api));
        }
      }
    }
  }
  }
#else
      if (!gst_buffer_copy_into (out_buf, src_buffer, GST_BUFFER_COPY_META, 0, -1)) {
        GST_DEBUG ("Buffer metadata copy failed \n");
      }
#endif
      const GstMetaInfo *info = GST_REFERENCE_TIMESTAMP_META_INFO;
      GstReferenceTimestampMeta *dec_meta = NULL;
      GstCaps *reference_caps = NULL;
      const GstStructure *str;
      gint dec_frame_num = 0;
      gchar *dec_name = NULL;
      gdouble dec_in_timestamp = 0;
      gdouble dec_out_timestamp = 0;

#ifdef META_BEFORE_NVSTREAMMUX
      guint len = 0;
      GList * gst_meta_list = NULL;
#endif

      if (mux->prev_batch_meta == 0)
      {
      while ((gst_meta = gst_buffer_iterate_meta (src_buffer, &state)))
      {
#ifdef META_BEFORE_NVSTREAMMUX
        if (gst_meta_api_type_has_tag (gst_meta->info->api, _dsmeta_quark)) {
           gst_meta_list = g_list_append(gst_meta_list, gst_meta);
        }
#endif
        meta = (NvDsMeta *) gst_meta;
        if (gst_meta_api_type_has_tag (gst_meta->info->api, _dsmeta_quark)) {
          if (meta->meta_type == NVDS_DEWARPER_GST_META)
          {
            src_surface_meta = (NvDewarperSurfaceMeta *)(meta->meta_data);
            //dewarper_meta = meta;
          }
        }
        if (gst_meta->info->api == info->api)
        {
          dec_meta = (GstReferenceTimestampMeta *) gst_meta;
          reference_caps = dec_meta->reference;
          str = gst_caps_get_structure (reference_caps, 0);
          dec_name = (gchar *)gst_structure_get_string (str, "component_name");
          gst_structure_get_int (str, "frame_num", &dec_frame_num);
          gst_structure_get_double (str, "in_timestamp", &dec_in_timestamp);
          gst_structure_get_double (str, "out_timestamp", &dec_out_timestamp);
        }
      }
      }

      if (USE_CUDA_BATCH) {
        {
          GstEvent *new_pad_event = NULL;

          (gst_buffer_get_nvstream_memory (out_buf))->orig_buffer_ptrs[dest_surf->numFilled] = src_buffer;//gst_buffer_ref (src_buffer);
          //g_print ("TK: %s : %d GSTBuffer=%p out_buf=%p dest_surf->numFilled=%d\n", __func__, dest_surf->numFilled,
          //		  (gst_buffer_get_nvstream_memory (out_buf))->orig_buffer_ptrs[dest_surf->numFilled],
          //		  out_buf, dest_surf->numFilled);
          copy_data_cuda (mux, dest_surf, src_buffer, &pad_data->in_videoinfo, pad_data);
          GstClockTime ntp_ts = GST_CLOCK_TIME_NONE;

          if(mux->prev_batch_meta == 0)
          {
          if (!pad_data->ntp_calc)
            pad_data->ntp_calc =
                gst_nvds_ntp_calculator_new (mux->sys_ts ? GST_NVDS_NTP_CALC_MODE_SYSTEM_TIME :
                        GST_NVDS_NTP_CALC_MODE_RTCP, mux->frame_duration,
                        GST_ELEMENT (mux), pad_data->pad_id);

          if (!gst_nvds_ntp_calculator_have_ntp_sync_values (pad_data->ntp_calc)) {
            GstQuery *nvquery = gst_nvquery_ntp_sync_new ();
            GstClockTime ntp_time_epoch_ns = 0, ntp_frame_timestamp = 0;
            GstClockTime avg_frame_time = 0;

            gst_pad_peer_query (mux->sink_pads[mux->current_loc], nvquery);
            _NtpData ntpdata;
            memset(&ntpdata, 0, sizeof(_NtpData));
            gst_nvquery_ntp_sync_parse (nvquery, &ntpdata);

            if (ntpdata.ntp_time_epoch_ns > 0) {
              ntp_time_epoch_ns = ntpdata.ntp_time_epoch_ns;
              ntp_frame_timestamp = ntpdata.frame_timestamp;
              avg_frame_time = ntpdata.avg_frame_time;
            }
            gst_nvds_ntp_calculator_add_ntp_sync_values (pad_data->ntp_calc,
                ntp_time_epoch_ns, ntp_frame_timestamp, avg_frame_time);
            gst_query_unref (nvquery);
          }

           ntp_ts =
              gst_nvds_ntp_calculator_get_buffer_ntp (pad_data->ntp_calc,
                      GST_BUFFER_PTS (src_buffer));
          }

          if(mux->prev_batch_meta == 0)
          {
          for(surf_cnt = 0; surf_cnt < mux->num_surfaces_per_frame; surf_cnt++)
          {
            NvDsFrameMeta *frame_meta = NULL;
            frame_meta = nvds_acquire_frame_meta_from_pool(dest_batch_meta);
            frame_meta->pad_index = pad_data->pad_id;
            frame_meta->source_id = pad_data->source_id;

            frame_meta->buf_pts = GST_BUFFER_PTS (src_buffer);

            frame_meta->ntp_timestamp = ntp_ts;


            if (TRUE == mux->sei_based_frameId)
            {
//              pad_data->timestamp = GST_BUFFER_PTS (src_buffer);
                frame_meta->ntp_timestamp = pad_data->timestamp;
                frame_meta->frame_num = pad_data->sei_frame_id;
            }

            if (FALSE == mux->sei_based_frameId)
            {
                frame_meta->frame_num = pad_data->curr_frame_no++;
            }
            if(mux->extract_sei_sim_time == TRUE) {
              frame_meta->ntp_timestamp = pad_data->timestamp_mega;
              frame_meta->frame_num = pad_data->sei_frame_id;
            }

            frame_meta->batch_id = mux->num_bufs_in_current_batch;
            frame_meta->source_frame_width = pad_data->in_videoinfo.width;
            frame_meta->source_frame_height = pad_data->in_videoinfo.height;
            frame_meta->pipeline_width = mux->width;
            frame_meta->pipeline_height = mux->height;
            frame_meta->num_surfaces_per_frame = mux->num_surfaces_per_frame;

            if(src_surface_meta) {
                frame_meta->surface_type = src_surface_meta->type[surf_cnt];
                frame_meta->surface_index = src_surface_meta->index[surf_cnt];
                frame_meta->source_id = src_surface_meta->source_id;
            }

            /* Overlay clock skey between PTS attached by mux and incoming buffer */
            if (0)
            {
                NvDsDisplayMeta *dmeta = nvds_acquire_display_meta_from_pool (dest_batch_meta);
                char q[] = "Serif";
                dmeta->num_labels = 1;
                dmeta->text_params[0].display_text = g_strdup_printf("TS DIFF=%" GST_TIME_FORMAT, GST_TIME_ARGS(mux->cur_frame_pts - GST_BUFFER_PTS(src_buffer)));
                //dmeta->text_params[0].display_text = g_strdup_printf("queue length = %d", g_queue_get_length(&pad_data->buf_queue));
                dmeta->text_params[0].x_offset = 0;
                dmeta->text_params[0].y_offset = 0;
                dmeta->text_params[0].set_bg_clr = 1;
                dmeta->text_params[0].text_bg_clr = (NvOSD_ColorParams){0, 0, 0, 1};
                dmeta->text_params[0].font_params.font_size=16;
                dmeta->text_params[0].font_params.font_name =q;
                dmeta->text_params[0].font_params.font_color= (NvOSD_ColorParams){1, 1, 1, 1};
                nvds_add_display_meta_to_frame(frame_meta, dmeta);
            }

#ifdef META_BEFORE_NVSTREAMMUX
            if(surf_cnt == 0)
            {
              NvDsMetaList *l = NULL;
              for (l = gst_meta_list; l != NULL; l = l->next)
              {
                NvDsUserMeta *user_gst_meta = nvds_acquire_user_meta_from_pool (dest_batch_meta);
                meta = (NvDsMeta *)(l->data);
                user_gst_meta->user_meta_data = meta->copyfunc(meta->meta_data, NULL);
                user_gst_meta->base_meta.meta_type = static_cast<NvDsMetaType>(meta->meta_type);
                user_gst_meta->base_meta.copy_func = (NvDsMetaCopyFunc)meta->gst_to_nvds_meta_transform_func;
                user_gst_meta->base_meta.release_func = (NvDsMetaReleaseFunc)meta->gst_to_nvds_meta_release_func;
                nvds_add_user_meta_to_frame(frame_meta, user_gst_meta);
              }

              while(g_list_length(gst_meta_list))
              {
                  l = g_list_first(gst_meta_list);
                  gst_meta_list = g_list_remove_link(gst_meta_list, l);
                  g_list_free (l);
              }
            }
#endif

            if(nvds_enable_latency_measurement || nvds_latency_measurement_silent) {
              if(dec_meta)
              {
                frame_meta->frame_num = dec_frame_num;
                NvDsMetaCompLatency *dec_latency_metadata = NULL;
                NvDsUserMeta *user_meta = NULL;
                user_meta = nvds_set_input_system_timestamp
                  (out_buf, dec_name);
                dec_latency_metadata = (NvDsMetaCompLatency *)user_meta->user_meta_data;
                dec_latency_metadata->in_system_timestamp = dec_in_timestamp;
                dec_latency_metadata->out_system_timestamp = dec_out_timestamp;
                dec_latency_metadata->source_id = frame_meta->source_id;
                dec_latency_metadata->pad_index = frame_meta->pad_index;
              }

              Input_TS_Identification *buf_in_TS_ptr = (Input_TS_Identification *)gst_mini_object_get_qdata(
                  (GstMiniObject *)src_buffer,
                  g_quark_from_string(GST_ELEMENT_NAME(mux)));

              user_latency_meta = nvds_acquire_user_meta_from_pool (dest_batch_meta);
              user_latency_meta->user_meta_data = (void *)nvds_set_latency_metadata_ptr();
              user_latency_meta->base_meta.meta_type = NVDS_LATENCY_MEASUREMENT_META;
              user_latency_meta->base_meta.copy_func = (NvDsMetaCopyFunc)nvds_copy_latency_meta;
              user_latency_meta->base_meta.release_func = (NvDsMetaReleaseFunc)nvds_release_latency_meta;
              latency_metadata = (NvDsMetaCompLatency *)user_latency_meta->user_meta_data;
              g_snprintf(latency_metadata->component_name, MAX_COMPONENT_LEN,
                  "nvstreammux-%s", GST_ELEMENT_NAME(mux));
              if(buf_in_TS_ptr)
              {
                latency_metadata->in_system_timestamp = buf_in_TS_ptr->in_system_timestamp;
                latency_metadata->source_id = frame_meta->source_id;
                latency_metadata->frame_num = dec_frame_num;
                latency_metadata->pad_index = frame_meta->pad_index;
                //g_print("got timestamp*******************\n");
              }
              nvds_add_user_meta_to_batch(dest_batch_meta, user_latency_meta);
              mux->latency_metadata_list =
                g_list_append(mux->latency_metadata_list, latency_metadata);
            }

            nvds_add_frame_meta_to_batch(dest_batch_meta, frame_meta);
#ifndef NEW_METADATA
            if (frameinfo)
            {
              dest_meta->camera_id[dest_meta->num_filled] = frameinfo->camera_id;
              dest_meta->surface_type[dest_meta->num_filled] = frameinfo->surface_type[surf_cnt];
              dest_meta->surface_index[dest_meta->num_filled] = frameinfo->surface_index[surf_cnt];
              strncpy (dest_meta->input_pkt_pts[dest_meta->num_filled], frameinfo->input_pkt_pts, MAX_PKT_PTS_LEN);
            }
#endif
            if (pad_data->new_pad_added == TRUE) {
               GST_DEBUG_OBJECT (mux, "Pad added event sent %d\n", pad_data->pad_id);
               new_pad_event = gst_nvevent_new_pad_added (pad_data->pad_id);
               gst_pad_push_event (mux->srcpad, new_pad_event);
               pad_data->new_pad_added = FALSE;
            }
          }
          }
          else
          {
            guint num_frames_in_batch = 0;
            NvDsBatchMeta *src_batch_meta = gst_buffer_get_nvds_batch_meta(src_buffer);
            if(src_batch_meta == NULL)
            {
              GST_ERROR ("%s: Batch meta not found in nvstreammux\n", GST_ELEMENT_NAME(mux));
              return GST_FLOW_ERROR;
            }

            for (num_frames_in_batch = 0; num_frames_in_batch < src_batch_meta->num_frames_in_batch; num_frames_in_batch++)
            {
              NvDsFrameMeta *frame_meta = nvds_acquire_frame_meta_from_pool(dest_batch_meta);
              NvDsFrameMeta *src_frame_meta = nvds_get_nth_frame_meta(src_batch_meta->frame_meta_list, 0);
              nvds_copy_frame_meta(src_frame_meta, frame_meta);
              frame_meta->batch_id = mux->num_bufs_in_current_batch;
              frame_meta->pad_index = pad_data->pad_id;
              nvds_add_frame_meta_to_batch(dest_batch_meta, frame_meta);
            }
            if (pad_data->new_pad_added == TRUE) {
              GST_DEBUG_OBJECT (mux, "Pad added event sent %d\n", pad_data->pad_id);
              new_pad_event = gst_nvevent_new_pad_added (pad_data->pad_id);
              gst_pad_push_event (mux->srcpad, new_pad_event);
              pad_data->new_pad_added = FALSE;
            }
          }
        }
#if 1
        if(!mux->async_process)
            cudaStreamSynchronize (mux->stream);
        //gst_buffer_unref (src_buffer);
        //src_buffer = NULL;
#endif
      }

#if 0
      if (dewarper_meta)
        gst_buffer_remove_meta (out_buf, (GstMeta *)dewarper_meta);
#endif

//      if (src_buffer)
//        gst_buffer_list_add (buf_list, src_buffer);

      g_cond_broadcast (&pad_data->queue_cond);

      for(surf_cnt = 0; surf_cnt < mux->num_surfaces_per_frame; surf_cnt++)
      {
        if (pad_data->num_bufs_in_current_batch >=
            (batch_size / (n - mux->num_queues_empty))) {
          mux->num_extra_bufs_in_batch++;
        }

        pad_data->num_bufs_in_current_batch++;
        mux->num_bufs_in_current_batch++;
      }
      ret = TRUE;

		return ret;
}

// Supporting function to extract NTP timestamp from SEI metadata with race condition fix
static GstClockTime get_ntp_timestamp_from_sei_meta(GstBuffer *buffer) {
    const int MAX_RETRIES = 3;
    const int RETRY_DELAY_US = 100; // 100 microseconds

    for (int retry = 0; retry < MAX_RETRIES; retry++) {
        GstVideoSEIMeta *meta = (GstVideoSEIMeta *) gst_buffer_get_meta(buffer, GST_VIDEO_SEI_META_API_TYPE);

        if (meta && meta->sei_metadata_type == (guint)GST_USER_SEI_META && 
            meta->sei_metadata_ptr && meta->sei_metadata_size > 0) {

            try {
                std::string jsonContent(reinterpret_cast<const char*>(meta->sei_metadata_ptr), meta->sei_metadata_size);

                // Validate JSON content is not empty and contains expected data
                if (jsonContent.empty() || jsonContent.find("timestamp") == std::string::npos) {
                    if (retry < MAX_RETRIES - 1) {
                        g_usleep(RETRY_DELAY_US); // Small delay before retry
                        continue;
                    }
                    return GST_CLOCK_TIME_NONE;
                }

                std::stringstream ss(jsonContent);
                boost::property_tree::ptree pt;
                boost::property_tree::read_json(ss, pt);

                guint64 timestamp = pt.get<guint64>("timestamp", 0);
                if (timestamp > 0) {
                    return timestamp;
                }
            } catch (const std::exception& e) {
                // JSON parsing failed, retry if attempts remaining
                if (retry < MAX_RETRIES - 1) {
                    g_usleep(RETRY_DELAY_US);
                    continue;
                }
            }
        } else {
            // Metadata not ready yet, retry with delay
            if (retry < MAX_RETRIES - 1) {
                g_usleep(RETRY_DELAY_US);
                continue;
            }
        }
    }

    return GST_CLOCK_TIME_NONE;
}

static void adjust_base_ntp(GstNvStreamMux *mux) {
    GstClockTime frame_duration = gst_util_uint64_scale_int(GST_SECOND, 1, mux->fps);
        GstClockTime next_value = mux->base_ntp + frame_duration;
        GstClockTime remainder = next_value % GST_SECOND;
    GstClockTime old_base_ntp = mux->base_ntp;
    // Base NTP adjustment starting

        if (remainder < (GST_MSECOND * 1) || remainder > (GST_SECOND - GST_MSECOND * 1)) {
            // Close to a value divisible by 1 second, use adjusted increment
            mux->base_ntp += frame_duration + (GST_NSECOND * 1);
            // Adjusted increment for second alignment
        } else {
            // Not close, use normal increment
            mux->base_ntp += frame_duration;
            // Normal frame duration increment
    }

    GST_DEBUG_OBJECT(mux, "[ADJUST-BASE-NTP] Advanced base_ntp %ld -> %ld (delta=%ld ns, frame_duration=%ld, fps=%lu)",
        old_base_ntp, mux->base_ntp, (gint64)(mux->base_ntp - old_base_ntp), frame_duration, mux->fps);

    GST_DEBUG_OBJECT(mux, "adjust_base_ntp: Advanced base_ntp from %ld to %ld (delta=%ld ns)",
        old_base_ntp, mux->base_ntp, (gint64)(mux->base_ntp - old_base_ntp));

    // After advancing time, drop any held/stale buffers from all pad queues
    // that were ALREADY behind the OLD base_ntp (before adjustment)
    // Use trylock to avoid deadlock - if lock is already held by caller, skip that pad

    // Calculate cleanup threshold: use a more generous threshold than sync_inputs_ntp
    // to avoid dropping valid buffers. If sync_inputs_ntp is 0 (default), use 2x frame duration.
    GstClockTime frame_duration_cleanup = gst_util_uint64_scale_int(GST_SECOND, 1, mux->fps);
    GstClockTime cleanup_threshold = (mux->sync_inputs_ntp > 0) ?
        mux->sync_inputs_ntp : (frame_duration_cleanup * 2);

    GST_DEBUG_OBJECT(mux, "adjust_base_ntp: Cleaning buffers behind old_base_ntp=%ld (threshold=%ld ns, sync_inputs_ntp=%u, frame_duration=%ld)",
        old_base_ntp, cleanup_threshold, mux->sync_inputs_ntp, frame_duration_cleanup);

    gint n = GST_ELEMENT(mux)->numsinkpads;
    for (gint i = 0; i < n; i++) {
        GstNvStreamMuxPadData *pad_data = (GstNvStreamMuxPadData *)
            g_object_get_data(G_OBJECT(mux->sink_pads[i]), PAD_DATA_KEY);

        if (!pad_data) continue;

        // Try to lock - if already locked by caller's context, skip this pad
        // It will be cleaned up in the next iteration when lock is available
        if (!g_mutex_trylock(&pad_data->queue_lock)) {
            GST_DEBUG_OBJECT(mux, "adjust_base_ntp: Skipping pad %d cleanup (lock busy)", pad_data->pad_id);
            continue;
        }

        // Drop old buffers from this pad's queue that were behind OLD base_ntp
        gint dropped = 0;
        while (!g_queue_is_empty(&pad_data->buf_queue)) {
            GstMiniObject *check_obj = (GstMiniObject *)g_queue_peek_head(&pad_data->buf_queue);
            if (!GST_IS_BUFFER(check_obj)) {
                break;
            }

            GstBuffer *check_buf = GST_BUFFER(check_obj);
            GstClockTime check_ntp = get_ntp_timestamp_from_sei_meta(check_buf);

            if (check_ntp == GST_CLOCK_TIME_NONE) {
                // Can't determine age without metadata, leave it for normal processing
                break;
            }

            // Drop only if buffer was already behind the OLD base_ntp (before adjustment)
            // This ensures we only clean truly stale buffers, not ones that are valid for next batch
            if ((old_base_ntp > check_ntp) &&
                ((gint64)(old_base_ntp - check_ntp) > (gint64)cleanup_threshold)) {
                GstBuffer *old_buf = GST_BUFFER(g_queue_pop_head(&pad_data->buf_queue));
                dropped++;
                // Dropping stale buffer during cleanup
                GST_DEBUG_OBJECT(mux, "adjust_base_ntp: Dropping stale buffer with ntp=%ld (old_base=%ld, delta=%ld ns)",
                    check_ntp, old_base_ntp, (gint64)(old_base_ntp - check_ntp));
                get_sync_obj_and_wait(old_buf, &pad_data->sync_queue, -1);
                gst_buffer_unref(old_buf);
            } else {
                // This buffer is still valid (was not behind old base_ntp), stop checking
                break;
            }
        }

        if (dropped > 0) {
            // Stale buffer cleanup completed
            GST_DEBUG_OBJECT(mux, "adjust_base_ntp: Dropped %d stale buffers from pad %d (new_base=%ld)",
                dropped, pad_data->pad_id, mux->base_ntp);
        }

        g_mutex_unlock(&pad_data->queue_lock);
    }
}

static GstClockTime calculate_fps(GstClockTime duration_ns) {
    // Check if duration is valid
    if (duration_ns == GST_CLOCK_TIME_NONE || duration_ns == 0) {
        return GST_CLOCK_TIME_NONE;
    }

    // Convert nanoseconds to seconds
    gdouble duration_s = (gdouble)duration_ns / GST_SECOND;

    // Calculate fps
    gdouble fps = 1.0 / duration_s;

    // Round to two decimal places
    fps = round(fps * 100) / 100;

    // Convert fps to GstClockTime (without multiplying by GST_SECOND)
    return (GstClockTime)fps;
}

static gboolean
gst_nvstreammux_src_collect_buffers (GstNvStreamMux * mux, GstBuffer * out_buf,
    NvBufSurface *dest_surf, NvDsBatchMeta * dest_batch_meta,
    GstBufferList * buf_list, gboolean timedout, guint batch_size)
{
  gboolean ret = FALSE;
  GstElement *element = GST_ELEMENT (mux);
  //GList *iter = element->sinkpads;
  guint i, n;
  guint dec_meta_cnt = 0;
  GstMeta *gst_meta = NULL;
  gpointer state = NULL;
  //NvDsMeta *dewarper_meta = NULL;
  GstBuffer *temp_buf = NULL;
  GstBuffer *src_buffer = NULL;

  if (!mux->segment_sent) {
    gst_pad_push_event (mux->srcpad, gst_event_new_segment (&mux->segment));
    mux->segment_sent = TRUE;
  }

  n = element->numsinkpads;
#if 1
  cudaError_t CUerr = cudaSuccess;
  CUerr = cudaSetDevice(mux->gpu_id);
  if(CUerr != cudaSuccess)
  {
    g_print ("Unable to set device in %s\n", __func__);
    return FALSE;
  }
  //GST_TRACE_OBJECT (mux, "SETTING CUDA DEVICE = %d in nvstreammux  func=%s\n", mux->gpu_id, __func__);
#endif
//  mux->one_time_in_batch=TRUE;
  //g_print("[COLLECT-START] n_pads=%u, batch_size=%u, current_batch=%lu, timedout=%d\n",
          //n, batch_size, mux->num_bufs_in_current_batch, timedout);

  for (i = 0; i < n && mux->num_bufs_in_current_batch < batch_size; i++) {
    mux->current_loc = mux->current_loc % n;
    GstNvStreamMuxPadData *pad_data =
        (GstNvStreamMuxPadData *)
        g_object_get_data (G_OBJECT (mux->sink_pads[mux->current_loc]),
        PAD_DATA_KEY);

    guint queue_len = g_queue_get_length(&pad_data->buf_queue);
    gboolean is_empty = g_queue_is_empty(&pad_data->buf_queue);
    gpointer head = g_queue_peek_head(&pad_data->buf_queue);
    //g_print("[COLLECT-PAD] pad_id=%d: queue_len=%u, is_empty=%d, head=%p, got_eos=%d, stopping=%d\n",
            //pad_data->pad_id, queue_len, is_empty, head, pad_data->got_eos, pad_data->stopping);

    // VERIFICATION: Cross-check queue state consistency
    if ((queue_len == 0) != is_empty) {
      //g_print("[COLLECT-PAD-ERROR] pad_id=%d: INCONSISTENT QUEUE STATE! queue_len=%u but is_empty=%d\n",
              //pad_data->pad_id, queue_len, is_empty);
    }
    if ((queue_len > 0) && (head == NULL)) {
      //g_print("[COLLECT-PAD-ERROR] pad_id=%d: INCONSISTENT! queue_len=%u but head is NULL\n",
              //pad_data->pad_id, queue_len);
    }
    if ((queue_len == 0) && (head != NULL)) {
      //g_print("[COLLECT-PAD-ERROR] pad_id=%d: INCONSISTENT! queue_len=0 but head=%p\n",
              //pad_data->pad_id, head);
    }

    /*
     * Below booleans (buffer_available and buffer_available_pad) are used
     * to check if at least one buffer has arrived on pad after requesting a new pad.
     * If these values are default and EOS/Pad delete events are called ,
     * mux pushes out these events immidiately to the downstream components
     * Events are pushed out from handle_eos and release_pad respectively
	 */
    pad_data->buffer_available=TRUE;
    pad_data->buffer_available_pad=TRUE;

    if (pad_data->got_eos && !pad_data->queue_empty) {
      pad_data->queue_empty = g_queue_is_empty (&pad_data->buf_queue);
      if (pad_data->queue_empty) {
        GstMessage *msg = gst_nvmessage_new_stream_eos (GST_OBJECT (mux), pad_data->pad_id);
        GstEvent *event = gst_event_new_sink_message ("stream-eos", msg);
        //GstEvent *event_eos = gst_nvevent_new_stream_eos (pad_data->pad_id);
        mux->event_list = g_list_prepend (mux->event_list, event);
        //mux->event_list = g_list_prepend (mux->event_list, event);
        gst_message_unref(msg);

        mux->num_queues_empty++;
        ret = TRUE;
      }
    }

    g_mutex_lock (&pad_data->queue_lock);

    GstMiniObject *obj;
    while ((obj = (GstMiniObject *) g_queue_peek_head (&pad_data->buf_queue))) {
      if (GST_IS_EVENT (obj)) {
        GstEvent *event = GST_EVENT (g_queue_pop_head (&pad_data->buf_queue));
        g_cond_broadcast (&pad_data->queue_cond);
        switch ((guint32)GST_EVENT_TYPE (event)) {
          case GST_EVENT_SEGMENT:
          {
            const GstSegment *segment = NULL;
            GstEvent *new_event;
            gst_event_parse_segment (event, &segment);
            new_event = gst_nvevent_new_stream_segment (pad_data->pad_id,
                (GstSegment*) segment);
            gst_pad_push_event (mux->srcpad, new_event);
          }
            break;
          case GST_EVENT_EOS:
          {
            GstEvent *event_eos = gst_nvevent_new_stream_eos (pad_data->pad_id);
            mux->event_list = g_list_prepend (mux->event_list, event_eos);
            if (pad_data->ntp_calc)
              gst_nvds_ntp_calculator_reset (pad_data->ntp_calc);
            if (mux->frame_num_reset_on_eos)
              pad_data->curr_frame_no = 0;
          }
            break;
          case GST_NVEVENT_STREAM_RESET:
          {
            GstEvent *event_reset = gst_nvevent_new_stream_reset (pad_data->pad_id);
            if (pad_data->num_bufs_in_current_batch > 0 || mux->event_list) {
              mux->event_list = g_list_prepend (mux->event_list, event_reset);
            } else {
              gst_pad_push_event (mux->srcpad, event_reset);
            }
            if (mux->frame_num_reset_on_stream_reset)
              pad_data->curr_frame_no = 0;
          }
            break;
          case GST_NVEVENT_PAD_DELETED:
            {
              if (pad_data->num_bufs_in_current_batch > 0 || mux->event_list) {
                mux->event_list = g_list_prepend (mux->event_list, gst_nvevent_new_pad_deleted (pad_data->pad_id));
              } else {
                gst_pad_push_event (mux->srcpad, gst_nvevent_new_pad_deleted (pad_data->pad_id));
              }
            }
            break;
          default:
            break;
        }
        gst_event_unref (event);
      } else {
        break;
      }
    }

    if (n == mux->num_queues_empty) {
      g_mutex_unlock (&pad_data->queue_lock);
      break;
    }

    mux->num_extra_bufs = batch_size % (n - mux->num_queues_empty);

#define ALIGNMENT_THRESHOLD 33333333 // 2ms in nanoseconds
    // Since we are accessing streams in round robin way, do we
    // really need this logic?
    if (!g_queue_is_empty (&pad_data->buf_queue) &&
        (pad_data->num_bufs_in_current_batch <
            (batch_size / (n - mux->num_queues_empty)) ||
            mux->num_extra_bufs_in_batch < mux->num_extra_bufs)
        && mux->num_bufs_in_current_batch < batch_size) {

        if (!mux->sync_inputs && !mux->sync_inputs_ntp)
        {
            // No synchronization enabled
            // For live source we want one buffer from one source.
            if ((mux->live_source || mux->enable_adaptive_batch_size) &&
                    pad_data->num_bufs_in_current_batch == 1) {
                g_mutex_unlock (&pad_data->queue_lock);
                break;
            }
        }
        else if(!mux->sync_inputs_ntp)
        {

            GST_TRACE_OBJECT (mux, "buffs in queue = %d for pad_id = %d\n", g_queue_get_length(&pad_data->buf_queue), pad_data->pad_id);
            temp_buf = (GstBuffer *)obj;
            /* This frame is future frame, hold it and pick it later at its time*/
            if (BUF_PTS_TO_RUNNING_TIME (temp_buf) > mux->cur_frame_pts)
            {
                g_mutex_unlock(&pad_data->queue_lock);
                mux->current_loc++;
                continue;
            }
        }
        else if(!mux->first_batch)
        {
            static GstClockTime ntp_ts = GST_CLOCK_TIME_NONE;
            temp_buf = (GstBuffer *)obj;
            ntp_ts = get_ntp_timestamp_from_sei_meta(temp_buf);
            // Race condition fix: NTP extraction with retry logic applied

            if (ntp_ts == GST_CLOCK_TIME_NONE) {
              // RACE CONDITION FIX: Retry NTP extraction once more with delay
              g_usleep(500); // 500 microseconds delay
              ntp_ts = get_ntp_timestamp_from_sei_meta(temp_buf);

              if (ntp_ts == GST_CLOCK_TIME_NONE) {
                // Drop buffer immediately if sync_inputs_ntp is enabled to prevent buffer pool exhaustion
                if (mux->sync_inputs_ntp > 0) {
            GST_WARNING_OBJECT(mux, "DROPPING buffer with invalid NTP (ntp=-1) for pad %d (sync_inputs_ntp=%u enabled)",
                               pad_data->pad_id, mux->sync_inputs_ntp);
            GST_DEBUG_OBJECT(mux, "[NTP-DROP] pad_id=%d: Dropping buffer %p with ntp=-1 to prevent buffer pool exhaustion (sync_inputs_ntp=%u)",
                pad_data->pad_id, temp_buf, mux->sync_inputs_ntp);

                // Pop the buffer from queue and drop it
                GstBuffer *invalid_buf = GST_BUFFER(g_queue_pop_head(&pad_data->buf_queue));

                // Signal decoder to free buffer
                get_sync_obj_and_wait(invalid_buf, &pad_data->sync_queue, -1);
                gst_buffer_unref(invalid_buf);

                // Signal waiting threads
                g_cond_broadcast(&pad_data->queue_cond);
                g_mutex_unlock(&pad_data->queue_lock);

                // Move to next pad
                mux->current_loc++;
                continue;
              }
            }
          }

            /* This frame is future frame, hold it and pick it later at its time*/
            //if ((ABS((gint64)(ntp_ts - mux->base_ntp)) > mux->sync_inputs_ntp) && (ntp_ts > mux->base_ntp) )
            // CRITICAL: Skip holding logic if ntp_ts is invalid (-1), otherwise UINT64_MAX > base_ntp is always true!
            gint64 ntp_diff_for_holding = (gint64)(ntp_ts - mux->base_ntp);
            // Race condition fix: Check for massive time gaps

            // EMERGENCY FAILSAFE: If NTP difference is extremely large (>1 second), force base_ntp jump
            if (ntp_diff_for_holding > GST_SECOND) {
                GstClockTime old_base = mux->base_ntp;
                mux->base_ntp = ntp_ts - (mux->sync_inputs_ntp / 2);  // Jump to slightly before this buffer
                // Emergency jump due to massive time gap

                // Reset all holding counters after emergency jump
                if (mux->processed_pads) {
                    g_hash_table_remove_all(mux->processed_pads);
                }
                g_atomic_int_set((gint*)&mux->holding_counter, 0);

                // Don't hold this buffer, let it proceed normally
                g_mutex_unlock(&pad_data->queue_lock);
                mux->current_loc++;
                continue;
            }

            // FIX: Only hold buffers that are MORE than sync_inputs_ntp threshold ahead
            // Buffers within the sync window should be processed, not held
            gint64 future_diff = (ntp_ts > mux->base_ntp) ? (gint64)(ntp_ts - mux->base_ntp) : 0;
            gboolean is_too_future = (ntp_ts != GST_CLOCK_TIME_NONE && future_diff > (gint64)mux->sync_inputs_ntp);

            if (is_too_future)
            {
               g_mutex_unlock(&pad_data->queue_lock);

               // DISABLED: This was advancing base_ntp when buffer is way too future OR queue full
               // but with inference load, this causes base_ntp to run ahead without batching buffers
               // The failsafe mechanism will handle truly stuck situations more intelligently
               // if((((gint64)(ntp_ts - mux->base_ntp)) > ( mux->sync_inputs_ntp * 12 ) ) || ((g_queue_get_length(&pad_data->buf_queue)*4) >= mux->buffer_pool_size)) {
               //   adjust_base_ntp(mux);
               //   src_buffer = GST_BUFFER (g_queue_pop_head (&pad_data->buf_queue));
               //   GST_DEBUG_OBJECT(mux,"%d dropping too early buffer or queue is full for pad id = %d mux->base_ntp=%ld ntp_ts=%ld queue length=%d\n",__LINE__, pad_data->pad_id,mux->base_ntp, ntp_ts, g_queue_get_length(&pad_data->buf_queue));
               //   get_sync_obj_and_wait(src_buffer, &pad_data->sync_queue, -1);
               //   gst_buffer_unref (src_buffer);
               // }
               // else {
               {
                 // Create a hash table to store unique pad IDs if it doesn't exist
                 if (!mux->processed_pads) {
                   mux->processed_pads = g_hash_table_new(g_direct_hash, g_direct_equal);
                 }

                // Check if this pad_id has been processed in this batch
                if (!g_hash_table_contains(mux->processed_pads, GINT_TO_POINTER(pad_data->pad_id))) {
                  g_hash_table_add(mux->processed_pads, GINT_TO_POINTER(pad_data->pad_id));
                  g_atomic_int_inc((gint*)&mux->holding_counter); // Thread-safe increment

                  // Buffer held due to future timestamp
                  GST_DEBUG_OBJECT(mux,"%d Holding current frame ntp_ts=%ld mux->base_ntp=%ld for pad_id=%d holding_counter=%d batchsize=%d\n",
                  __LINE__, ntp_ts, mux->base_ntp, pad_data->pad_id, g_atomic_int_get((gint*)&mux->holding_counter), batch_size);
                }

              // PROACTIVE BASE_NTP ADVANCEMENT: If >= 3 sources are holding future buffers,
              // advance base_ntp immediately instead of waiting for failsafe (failsafe_flush_count)
              // This prevents buffer pool exhaustion and decoder backpressure
              guint current_counter = g_atomic_int_get((gint*)&mux->holding_counter);
              guint threshold = 3; // Fixed threshold to prevent buffer pool exhaustion

              GST_DEBUG_OBJECT(mux, "[HOLDING-CHECK] pad_id=%d: holding_counter=%u, batch_size=%u, threshold=%u",
                  pad_data->pad_id, current_counter, batch_size, threshold);

               if (current_counter >= threshold) {
                // Proactive base_ntp advancement due to multiple pads holding
                GST_DEBUG_OBJECT(mux, "[HOLDING-ADVANCE] holding_counter=%u >= threshold=%u, WILL JUMP base_ntp to min NTP across pads",
                    current_counter, threshold);
                GST_DEBUG_OBJECT(mux,"%d Holding current frame adjust NTP NOW since holding_counter=%u >= threshold=%u (batch_size=%u)\n",
                __LINE__, current_counter, threshold, batch_size);

                // Find minimum NTP across all pads with buffers (smart jump, not just frame_duration)
                GstClockTime min_ntp_across_pads = GST_CLOCK_TIME_NONE;
                guint n = GST_ELEMENT(mux)->numsinkpads;
                guint pads_with_buffers = 0;

                for (guint check_idx = 0; check_idx < n; check_idx++) {
                  GstNvStreamMuxPadData *check_pad =
                      (GstNvStreamMuxPadData *)g_object_get_data(G_OBJECT(mux->sink_pads[check_idx]), PAD_DATA_KEY);

                  if (check_pad && !g_queue_is_empty(&check_pad->buf_queue)) {
                    GstMiniObject *peek_obj = (GstMiniObject *)g_queue_peek_head(&check_pad->buf_queue);
                    if (GST_IS_BUFFER(peek_obj)) {
                      GstBuffer *peek_buf = GST_BUFFER(peek_obj);
                      GstClockTime peek_ntp = get_ntp_timestamp_from_sei_meta(peek_buf);

                      if (peek_ntp != GST_CLOCK_TIME_NONE) {
                        pads_with_buffers++;
                        if (min_ntp_across_pads == GST_CLOCK_TIME_NONE || peek_ntp < min_ntp_across_pads) {
                          min_ntp_across_pads = peek_ntp;
                        }
                      }
                    }
                  }
                }

                // Jump base_ntp to min NTP (smart jump, not incremental)
                if (min_ntp_across_pads != GST_CLOCK_TIME_NONE && min_ntp_across_pads > mux->base_ntp) {
                  GstClockTime old_base_ntp = mux->base_ntp;
                  mux->base_ntp = min_ntp_across_pads;
                  // Base NTP jumped to minimum across pads
                  GST_DEBUG_OBJECT(mux, "[HOLDING-ADVANCE] Jumped base_ntp %ld -> %ld (delta=%ld ms, pads=%u)",
                      old_base_ntp, mux->base_ntp, (gint64)((mux->base_ntp - old_base_ntp)/1000000), pads_with_buffers);
                  GST_WARNING_OBJECT(mux, "PROACTIVE: base_ntp advanced from %ld to %ld to unblock %u holding pads (threshold=%u/%u)",
                                     old_base_ntp, mux->base_ntp, current_counter, threshold, batch_size);
                } else {
                  // FALLBACK: No valid NTP timestamps found, use frame duration advancement
                  GST_DEBUG_OBJECT(mux, "[HOLDING-ADVANCE] No valid NTP timestamps (none=%d, pads=%u), falling back to frame_duration advancement",
                      min_ntp_across_pads == GST_CLOCK_TIME_NONE, pads_with_buffers);
                  adjust_base_ntp(mux);
                }

                // Reset global holding counter
                if (mux->processed_pads)
                  g_hash_table_remove_all(mux->processed_pads);
                g_atomic_int_set((gint*)&mux->holding_counter, 0);

                // Reset holding counters for ALL pads after advancing base_ntp
                for (guint reset_idx = 0; reset_idx < n; reset_idx++) {
                  GstNvStreamMuxPadData *reset_pad =
                      (GstNvStreamMuxPadData *)g_object_get_data(G_OBJECT(mux->sink_pads[reset_idx]), PAD_DATA_KEY);
                  if (reset_pad) {
                    reset_pad->pad_holding_iterations = 0;
                  }
                }
                GST_DEBUG_OBJECT(mux, "[HOLDING-ADVANCE] Reset all counters (holding_counter=0, all pad_holding_iterations=0) after base_ntp jump");
              }
              }
              GST_DEBUG_OBJECT(mux,"QUEUE LENGTH IN HOLDING STATE =%d\n",g_queue_get_length(&pad_data->buf_queue));

              // PER-SOURCE FAILSAFE: Increment THIS pad's holding counter
              // Each source tracks its own holding events independently
              pad_data->pad_holding_iterations++;

              // Log WHY this buffer is being held
              GstMiniObject *held_obj = (GstMiniObject *)g_queue_peek_head(&pad_data->buf_queue);
              if (GST_IS_BUFFER(held_obj)) {
                GstBuffer *held_buf = GST_BUFFER(held_obj);
                GstClockTime held_ntp = get_ntp_timestamp_from_sei_meta(held_buf);
                gint64 held_diff = held_ntp - mux->base_ntp;
                GST_DEBUG_OBJECT(mux, "[FAILSAFE-COUNT] pad_id=%d: holding_iterations=%u (max=%u), buffer_ntp=%ld, base_ntp=%ld, diff=%ld (buffer is %s)",
                    pad_data->pad_id, pad_data->pad_holding_iterations, mux->failsafe_flush_count,
                    held_ntp, mux->base_ntp, held_diff, 
                    held_ntp > mux->base_ntp ? "TOO FUTURE" : (held_ntp < mux->base_ntp ? "TOO LATE" : "IN SYNC"));
              } else {
                GST_DEBUG_OBJECT(mux, "[FAILSAFE-COUNT] pad_id=%d: holding_iterations=%u (max=%u)",
                    pad_data->pad_id, pad_data->pad_holding_iterations, mux->failsafe_flush_count);
              }

              // PER-SOURCE FAILSAFE: Check if THIS PAD has been holding too long
              // Drop ALL buffers from THIS pad if IT exceeds its own threshold
              if (pad_data->pad_holding_iterations >= mux->failsafe_flush_count) {
                GST_DEBUG_OBJECT(mux, "[FAILSAFE-TRIGGER] pad_id=%d: Holding iterations %u >= max %u, WILL DRAIN ALL BUFFERS",
                    pad_data->pad_id, pad_data->pad_holding_iterations, mux->failsafe_flush_count);

                GST_WARNING_OBJECT(mux, "FAILSAFE TRIGGERED: pad_id=%d has been holding buffers for %u iterations "
                                  "(>= failsafe-flush-count=%u). Draining all buffers from this pad to prevent buffer pool exhaustion.",
                                  pad_data->pad_id, pad_data->pad_holding_iterations, mux->failsafe_flush_count);

                // STEP 1: ALWAYS drain ALL buffers from THIS pad (regardless of timestamp)
                // After failsafe_flush_count iterations, the pad is clearly stuck - free up decoder pool!
                g_mutex_lock(&pad_data->queue_lock);

                guint dropped_count = 0;
                guint queue_length_before = g_queue_get_length(&pad_data->buf_queue);
                gboolean is_empty_before = g_queue_is_empty(&pad_data->buf_queue);
                GST_DEBUG_OBJECT(mux, "[FAILSAFE-DRAIN] pad_id=%d: BEFORE drain - queue_len=%u, is_empty=%d",
                    pad_data->pad_id, queue_length_before, is_empty_before);

                // Drain ALL buffers from this pad's queue
                while (!g_queue_is_empty(&pad_data->buf_queue)) {
                  GstBuffer *stuck_buf = GST_BUFFER(g_queue_pop_head(&pad_data->buf_queue));
                  if (stuck_buf) {
                    GstClockTime buf_ntp __attribute__((unused)) = get_ntp_timestamp_from_sei_meta(stuck_buf);
                    gint64 time_diff __attribute__((unused)) = 0;
                    if (buf_ntp != GST_CLOCK_TIME_NONE) {
                      time_diff = (gint64)(buf_ntp - mux->base_ntp);
                    }

                    GST_DEBUG_OBJECT(mux, "[FAILSAFE-UNREF] pad_id=%d: Unreffing stuck buffer %p, ntp=%ld (base=%ld, diff=%ld ms)",
                        pad_data->pad_id, stuck_buf, buf_ntp, mux->base_ntp, time_diff/1000000);

                    get_sync_obj_and_wait(stuck_buf, &pad_data->sync_queue, -1);
                    gst_buffer_unref(stuck_buf);
                    dropped_count++;
                  }
                }

                // VERIFICATION: Confirm queue is actually empty after draining
                guint queue_length_after = g_queue_get_length(&pad_data->buf_queue);
                gboolean is_empty_after = g_queue_is_empty(&pad_data->buf_queue);
                gpointer head_after = g_queue_peek_head(&pad_data->buf_queue);
                GST_DEBUG_OBJECT(mux, "[FAILSAFE-DRAIN] pad_id=%d: AFTER drain - queue_len=%u, is_empty=%d, head=%p, dropped=%u",
                    pad_data->pad_id, queue_length_after, is_empty_after, head_after, dropped_count);

                 if (queue_length_after != 0 || !is_empty_after || head_after != NULL) {
                  GST_DEBUG_OBJECT(mux, "[FAILSAFE-ERROR] pad_id=%d: Queue NOT fully drained! len=%u, empty=%d, head=%p after dropping %u",
                      pad_data->pad_id, queue_length_after, is_empty_after, head_after, dropped_count);
                }

                if (dropped_count > 0) {
                  GST_WARNING_OBJECT(mux, "FAILSAFE: Dropped %u stuck buffers from pad_id=%d after %u holding iterations "
                    "(failsafe_flush_count=%u) to prevent buffer leak and free up decoder pool.",
                    dropped_count, pad_data->pad_id, pad_data->pad_holding_iterations,
                    mux->failsafe_flush_count);
                  g_cond_broadcast(&pad_data->queue_cond); // Wake up blocked upstream for THIS pad
                }

                g_mutex_unlock(&pad_data->queue_lock);

                // STEP 2: Check if ALL pads have future buffers (base_ntp stuck in past)
                gboolean all_pads_future = TRUE;
                GstClockTime min_ntp_across_pads = GST_CLOCK_TIME_NONE;
                guint pads_with_buffers = 0;

                GST_DEBUG_OBJECT(mux, "[FAILSAFE-CHECK] Checking all %u pads to see if base_ntp=%ld is stuck in past",
                    n, mux->base_ntp);

                for (guint check_idx = 0; check_idx < n; check_idx++) {
                  GstNvStreamMuxPadData *check_pad =
                      (GstNvStreamMuxPadData *)g_object_get_data(G_OBJECT(mux->sink_pads[check_idx]), PAD_DATA_KEY);

                  if (check_pad && !g_queue_is_empty(&check_pad->buf_queue)) {
                    GstMiniObject *peek_obj = (GstMiniObject *)g_queue_peek_head(&check_pad->buf_queue);
                    if (GST_IS_BUFFER(peek_obj)) {
                      GstBuffer *peek_buf = GST_BUFFER(peek_obj);
                      GstClockTime peek_ntp = get_ntp_timestamp_from_sei_meta(peek_buf);

                      if (peek_ntp != GST_CLOCK_TIME_NONE) {
                        pads_with_buffers++;

                        // Track minimum NTP across all pads
                        if (min_ntp_across_pads == GST_CLOCK_TIME_NONE || peek_ntp < min_ntp_across_pads) {
                          min_ntp_across_pads = peek_ntp;
                        }

                        // If ANY pad has buffer with NTP <= base_ntp, not all are future
                        if (peek_ntp <= mux->base_ntp) {
                          all_pads_future = FALSE;
                        }

                        GST_DEBUG_OBJECT(mux, "[FAILSAFE-CHECK] pad_id=%d: has buffer with ntp=%ld (base=%ld, future=%d)",
                            check_pad->pad_id, peek_ntp, mux->base_ntp, peek_ntp > mux->base_ntp);
                      }
                    }
                  }
                }

                // If ALL pads with buffers have future NTPs, base_ntp is stuck - advance it!
                if (all_pads_future && pads_with_buffers > 0 && min_ntp_across_pads != GST_CLOCK_TIME_NONE) {
                  GST_DEBUG_OBJECT(mux, "[FAILSAFE-ADJUST-NTP] ALL %u pads have future buffers! Advancing base_ntp %ld -> %ld (min NTP across pads)",
                      pads_with_buffers, mux->base_ntp, min_ntp_across_pads);

                  GST_WARNING_OBJECT(mux, "FAILSAFE: base_ntp stuck in past, all %u pads holding future buffers. "
                                     "Advancing base_ntp from %ld to %ld to unblock pipeline.",
                                     pads_with_buffers, mux->base_ntp, min_ntp_across_pads);

                  mux->base_ntp = min_ntp_across_pads;

                  // Reset holding counters for ALL pads since we advanced time
                  for (guint reset_idx = 0; reset_idx < n; reset_idx++) {
                    GstNvStreamMuxPadData *reset_pad =
                        (GstNvStreamMuxPadData *)g_object_get_data(G_OBJECT(mux->sink_pads[reset_idx]), PAD_DATA_KEY);
                    if (reset_pad) {
                      reset_pad->pad_holding_iterations = 0;
                    }
                  }

                  GST_DEBUG_OBJECT(mux, "[FAILSAFE-RESET-ALL] Reset holding_iterations for all pads after base_ntp adjustment");

                  // Continue to next iteration - buffers should now be processable
              mux->current_loc++;
              continue;
            }

                // If we reach here, base_ntp is OK but pad was stuck
                // (Buffers already drained in STEP 1, so just reset counter and continue)
                GST_DEBUG_OBJECT(mux, "[FAILSAFE-DONE] pad_id=%d: Buffers drained, base_ntp OK, resetting counter",
                    pad_data->pad_id);

                // Reset ONLY THIS pad's holding counter (per-source failsafe)
                pad_data->pad_holding_iterations = 0;
                GST_DEBUG_OBJECT(mux, "[FAILSAFE-RESET] pad_id=%d: Reset pad_holding_iterations to 0", pad_data->pad_id);
              }

              mux->current_loc++;
              continue;
            }
        }

        guint queue_len_before_pop = g_queue_get_length(&pad_data->buf_queue);
        gboolean is_empty_before_pop = g_queue_is_empty(&pad_data->buf_queue);
        gpointer head_before_pop = g_queue_peek_head(&pad_data->buf_queue);

        //g_print("[COLLECT-POP] pad_id=%d: BEFORE pop - queue_len=%u, is_empty=%d, head=%p\n",
                //pad_data->pad_id, queue_len_before_pop, is_empty_before_pop, head_before_pop);

        // VERIFICATION: Should never try to pop from empty queue
        if (queue_len_before_pop == 0 || is_empty_before_pop || head_before_pop == NULL) {
          //g_print("[COLLECT-POP-ERROR] pad_id=%d: Trying to pop from EMPTY queue! len=%u, empty=%d, head=%p\n",
                  //pad_data->pad_id, queue_len_before_pop, is_empty_before_pop, head_before_pop);
        }

        src_buffer = GST_BUFFER (g_queue_pop_head (&pad_data->buf_queue));

        guint queue_len_after_pop = g_queue_get_length(&pad_data->buf_queue);
        gboolean is_empty_after_pop = g_queue_is_empty(&pad_data->buf_queue);
        //g_print("[COLLECT-POP] pad_id=%d: Popped buffer %p, queue_len %u->%u, is_empty %d->%d\n",
                //pad_data->pad_id, src_buffer, queue_len_before_pop, queue_len_after_pop,
                //is_empty_before_pop, is_empty_after_pop);

      if (mux->sync_inputs) {
          while (1)
          {
              obj = (GstMiniObject *)g_queue_peek_head(&pad_data->buf_queue);
              if (GST_IS_BUFFER(obj))
              {
                  temp_buf = (GstBuffer *) obj;
              }
              else
                  break;
              if (BUF_PTS_TO_RUNNING_TIME (temp_buf) < mux->cur_frame_pts)
              {
                  GST_WARNING_OBJECT (mux, "dropping late buffer pad id = %d window_start =%" GST_TIME_FORMAT "src_buffer pts=%" GST_TIME_FORMAT ,
                      pad_data->pad_id, GST_TIME_ARGS((mux->cur_frame_pts - mux->timeout_usec * 1000)), GST_TIME_ARGS(BUF_PTS_TO_RUNNING_TIME(src_buffer)));
                  //g_print("[COLLECT-DROP] pad_id=%d: Dropping LATE buffer %p in sync_inputs loop (UNREF #2)\n",
                          //pad_data->pad_id, src_buffer);
                  get_sync_obj_and_wait(src_buffer, &pad_data->sync_queue, -1);
                  gst_buffer_unref (src_buffer);
                  src_buffer =  GST_BUFFER (g_queue_pop_head (&pad_data->buf_queue));
                  //g_print("[COLLECT-REPOP] pad_id=%d: Popped next buffer %p after drop\n",
                          //pad_data->pad_id, src_buffer);
              }
              else
                  break;
          }
          /* This frame is outside our window,
           * window = (cur_frame_pts - timeout, cur_frame_pts) */
          if (BUF_PTS_TO_RUNNING_TIME(src_buffer) < (mux->cur_frame_pts - mux->timeout_usec * 1000))
          {
              g_mutex_unlock(&pad_data->queue_lock);
              GST_WARNING_OBJECT (mux, "dropping late buffer pad id = %d window_start =%" GST_TIME_FORMAT "src_buffer pts=%" GST_TIME_FORMAT ,
                      pad_data->pad_id, GST_TIME_ARGS((mux->cur_frame_pts - mux->timeout_usec * 1000)), GST_TIME_ARGS(BUF_PTS_TO_RUNNING_TIME(src_buffer)));
              //g_print("[COLLECT-DROP] pad_id=%d: Dropping LATE buffer %p outside window (UNREF #4)\n",
                      //pad_data->pad_id, src_buffer);
              get_sync_obj_and_wait(src_buffer, &pad_data->sync_queue, -1);
              gst_buffer_unref (src_buffer);
              //g_print("[COLLECT-SIGNAL] pad_id=%d: Broadcasting after late buffer drop\n", pad_data->pad_id);
              g_cond_broadcast(&pad_data->queue_cond);
              mux->current_loc++;
              continue;
          }
      }

#if 1
      if (mux->sync_inputs_ntp > 0) {
        static GstClockTime ntp_ts = GST_CLOCK_TIME_NONE;

        // NTP synchronization check starting

        while(mux->first_batch) {
            // For the first batch, use the first stream's NTP as base
          ntp_ts = get_ntp_timestamp_from_sei_meta(src_buffer);

          // First batch NTP processing

          if (ntp_ts == GST_CLOCK_TIME_NONE) {
            GST_WARNING_OBJECT(mux, "No sim time metadata in first batch for pad %d, dropping and trying next", pad_data->pad_id);
            //g_print("[COLLECT-DROP] pad_id=%d: No NTP in FIRST_BATCH, dropping buffer %p (UNREF #5)\n",
                    //pad_data->pad_id, src_buffer);
            get_sync_obj_and_wait(src_buffer, &pad_data->sync_queue, -1);
            gst_buffer_unref(src_buffer);
            src_buffer = GST_BUFFER(g_queue_pop_head(&pad_data->buf_queue));
            //g_print("[COLLECT-REPOP] pad_id=%d: Popped next buffer %p after no-NTP drop\n",
                    //pad_data->pad_id, src_buffer);
            if (!GST_IS_BUFFER(src_buffer)) {
              break;  // No more buffers, exit while loop
            }
            continue;  // Try next buffer in queue
          }
          if(mux->one_time_in_batch) {
            // Initialize base_ntp from first valid NTP timestamp

            mux->one_time_in_batch=FALSE;
            mux->base_ntp = ntp_ts;
          }

          obj = (GstMiniObject *)g_queue_peek_head(&pad_data->buf_queue);
          if (GST_IS_BUFFER(obj))
          {
            temp_buf = (GstBuffer *) obj;
          }
          else
            break;

          //g_print("%s:%d i=%d pad_id=%d ntp_ts=%ld mux->base_ntp=%ld\n",__func__, __LINE__, i, pad_data->pad_id, ntp_ts, mux->base_ntp);

          gint64 ntp_diff = mux->base_ntp - ntp_ts;
          gboolean is_late = (ntp_ts < mux->base_ntp);
          gboolean exceeds_threshold = (ABS(ntp_diff) > mux->sync_inputs_ntp);

          // First batch synchronization check

          if (exceeds_threshold && is_late) {
              // Dropping late buffer in first batch
              GST_DEBUG_OBJECT(mux,"%d dropping late buffer pad id = %d mux->base_ntp=%ld ntp_ts=%ld \n",__LINE__, pad_data->pad_id,mux->base_ntp, ntp_ts);
              get_sync_obj_and_wait(src_buffer, &pad_data->sync_queue, -1);
              gst_buffer_unref (src_buffer);
              src_buffer =  GST_BUFFER (g_queue_pop_head (&pad_data->buf_queue));
              //g_print("[COLLECT-REPOP] pad_id=%d: Popped next buffer %p after late drop in first_batch\n",
                      //pad_data->pad_id, src_buffer);
          }
          else {
            // Buffer accepted for first batch
            break;
          }
        }
        while(!mux->first_batch) {
          obj = (GstMiniObject *)g_queue_peek_head(&pad_data->buf_queue);
          if (GST_IS_BUFFER(obj))
          {
            temp_buf = (GstBuffer *) obj;
          }
          else
            break;

          ntp_ts = get_ntp_timestamp_from_sei_meta(src_buffer);

          // Regular batch NTP processing

          if (ntp_ts == GST_CLOCK_TIME_NONE) {
            GST_WARNING_OBJECT(mux, "No sim time metadata in buffer from pad %d, dropping and trying next", pad_data->pad_id);
             //g_print("[COLLECT-DROP] pad_id=%d: No NTP in regular batch, dropping buffer %p (UNREF #7)\n",
                     //pad_data->pad_id, src_buffer);
            get_sync_obj_and_wait(src_buffer, &pad_data->sync_queue, -1);
            gst_buffer_unref(src_buffer);
            src_buffer = GST_BUFFER(g_queue_pop_head(&pad_data->buf_queue));
             //g_print("[COLLECT-REPOP] pad_id=%d: Popped next buffer %p after no-NTP drop\n",
                     //pad_data->pad_id, src_buffer);
            if (!GST_IS_BUFFER(src_buffer)) {
              break;  // No more buffers, exit while loop
            }
            continue;  // Try next buffer in queue
          }

          gint64 ntp_diff = mux->base_ntp - ntp_ts;
          gboolean is_late = (ntp_ts < mux->base_ntp);
          gboolean exceeds_threshold = (ABS(ntp_diff) > mux->sync_inputs_ntp);

          // Regular batch synchronization check

          if (exceeds_threshold && is_late) {
              // Dropping late buffer in regular batch
              GST_DEBUG_OBJECT(mux,"%d dropping late buffer pad id = %d mux->base_ntp=%ld ntp_ts=%ld \n",__LINE__, pad_data->pad_id,mux->base_ntp, ntp_ts);
              get_sync_obj_and_wait(src_buffer, &pad_data->sync_queue, -1);
              gst_buffer_unref (src_buffer);
              src_buffer =  GST_BUFFER (g_queue_pop_head (&pad_data->buf_queue));
              //g_print("[COLLECT-REPOP] pad_id=%d: Popped next buffer %p after late drop in regular batch\n",
                      //pad_data->pad_id, src_buffer);
          }
          else {
            // Buffer accepted for regular batch
            break;
          }
         }

         ntp_ts = get_ntp_timestamp_from_sei_meta(src_buffer);

         // Final NTP check after sync loops

         if (ntp_ts == GST_CLOCK_TIME_NONE) {
           // No valid NTP after retries, dropping buffer
           GST_WARNING_OBJECT(mux, "No sim time metadata retrieved for pad %d after loop, dropping buffer", pad_data->pad_id);
             get_sync_obj_and_wait(src_buffer, &pad_data->sync_queue, -1);
             gst_buffer_unref(src_buffer);
            //g_print("[COLLECT-SIGNAL] pad_id=%d: Broadcasting after no-NTP drop\n", pad_data->pad_id);
            g_cond_broadcast(&pad_data->queue_cond);
           g_mutex_unlock(&pad_data->queue_lock);
           mux->current_loc++;
           continue;
         }

          GST_DEBUG_OBJECT(mux,"%s:%d i=%d pad_id=%d ntp_ts=%ld mux->base_ntp=%ld\n",__func__, __LINE__, i, pad_data->pad_id, ntp_ts, mux->base_ntp);

          gint64 ntp_diff = mux->base_ntp - ntp_ts;
          gboolean is_late = (ntp_ts < mux->base_ntp);
          gboolean exceeds_threshold = (ABS(ntp_diff) > mux->sync_inputs_ntp);

          // Final synchronization check

          if (exceeds_threshold && is_late) {
            // Dropping late buffer after final check
            g_mutex_unlock(&pad_data->queue_lock);
            GST_DEBUG_OBJECT(mux,"%d dropping late buffer pad id = %d mux->base_ntp=%ld ntp_ts=%ld \n",__LINE__, pad_data->pad_id,mux->base_ntp, ntp_ts);
            get_sync_obj_and_wait(src_buffer, &pad_data->sync_queue, -1);
            gst_buffer_unref (src_buffer);
             //g_print("[COLLECT-SIGNAL] pad_id=%d: Broadcasting after final late drop\n", pad_data->pad_id);
             g_cond_broadcast(&pad_data->queue_cond);
            mux->current_loc++;
            continue;
        }

        // Buffer passed all NTP synchronization checks
      }
#endif
#if 1
if (mux->align_first_buffer && !mux->first_batch_aligned && mux->extract_sei_sim_time == TRUE) {
   GstClockTime buffer_pts=0;
   GstClockTime frame_duration = 0; // 33ms in nanoseconds
   do{
      GstVideoSEIMeta *meta = (GstVideoSEIMeta *) gst_buffer_get_meta ((GstBuffer*)src_buffer, GST_VIDEO_SEI_META_API_TYPE);

      if (meta && meta->sei_metadata_type == (guint)GST_USER_SEI_META) {
         unsigned char* jsonContentPtr = (unsigned char*)meta->sei_metadata_ptr;
         size_t jsonContentSize = meta->sei_metadata_size;

         std::string jsonContent(reinterpret_cast<const char*>(jsonContentPtr), jsonContentSize);
         std::stringstream ss(jsonContent);
         boost::property_tree::ptree pt;
         boost::property_tree::read_json(ss, pt);

           buffer_pts = pt.get<guint64>("timestamp", 0);
         } else {
           GST_ERROR_OBJECT (mux, "no sim time metadata retrieved in streammux\n");
         }

         if (buffer_pts < (mux->highest_first_pts - frame_duration)/* || buffer_pts > (mux->highest_first_pts + frame_duration)*/) {
        // DISABLED: This was advancing base_ntp when dropping unaligned buffers
        // but this can cause base_ntp to run ahead without batching buffers
        // if (mux->sync_inputs_ntp) {
        //   adjust_base_ntp(mux);
        // }
        GST_DEBUG_OBJECT(mux,"dropping unaligned buffer pad id = %d highest_pts = %ld buffer_pts=%ld\n",pad_data->pad_id, mux->highest_first_pts, buffer_pts);
        //GST_WARNING_OBJECT(mux, "dropping unaligned buffer pad id = %d highest_pts = %" GST_TIME_FORMAT " buffer_pts = %" GST_TIME_FORMAT,
        //	pad_data->pad_id, GST_TIME_ARGS(mux->highest_first_pts), GST_TIME_ARGS(buffer_pts));
        get_sync_obj_and_wait(src_buffer, &pad_data->sync_queue, -1);
        gst_buffer_unref(src_buffer);
        src_buffer =  GST_BUFFER (g_queue_pop_head (&pad_data->buf_queue));
        while(!(GST_IS_BUFFER(src_buffer))){
          g_mutex_unlock (&mux->ctx_lock);
          g_mutex_unlock (&pad_data->queue_lock);
          g_usleep(100);
          g_mutex_lock (&pad_data->queue_lock);
          g_mutex_lock (&mux->ctx_lock);
          if(src_buffer!=NULL)
            gst_buffer_unref(src_buffer);
          src_buffer =  GST_BUFFER (g_queue_pop_head (&pad_data->buf_queue));
        }
         } else {
          break; // Found a buffer within the acceptable range
        }
     } while(TRUE);

   GST_INFO_OBJECT(mux, "First batch aligned. Switching to normal processing.");
}
#endif
      get_sync_obj_and_wait(src_buffer, &pad_data->sync_queue, -1);

      if (mux->buffer_cache) {
        gst_buffer_replace(&pad_data->cached_src_buffers, src_buffer);
        gettimeofday(&pad_data->cached_time, NULL);
      }
        //g_print("[COLLECT-ADD] pad_id=%d: Adding buffer %p to batch\n",
                //pad_data->pad_id, src_buffer);
      ret = gstnvstreammuxAddToBatch(mux, pad_data, src_buffer, out_buf, dest_surf, dest_batch_meta, batch_size);
        //g_print("[COLLECT-ADDED] pad_id=%d: Added buffer %p, ret=%d, batch_filled=%lu/%u\n",
                //pad_data->pad_id, src_buffer, ret, mux->num_bufs_in_current_batch, batch_size);

      // Reset per-source holding counter on successful batch addition
      // This pad is no longer stuck since it successfully added a buffer
        if (ret && pad_data->pad_holding_iterations > 0) {
          GST_DEBUG_OBJECT(mux, "[FAILSAFE-RESET] pad_id=%d: Reset pad_holding_iterations %u -> 0 after successful add",
              pad_data->pad_id, pad_data->pad_holding_iterations);
        pad_data->pad_holding_iterations = 0;
      }

    }

    //g_print("[COLLECT-SIGNAL] pad_id=%d: Broadcasting after processing\n", pad_data->pad_id);
    g_cond_broadcast(&pad_data->queue_cond);

    g_mutex_unlock (&pad_data->queue_lock);
   // iter = iter->next;
    mux->current_loc++;
  }

  //g_print("[COLLECT-END] Collected batch: %lu/%u buffers, ret=%d\n",
          //mux->num_bufs_in_current_batch, batch_size, ret);

#if 0
  while (0 && timedout
      && mux->num_bufs_in_current_batch < mux->batch_size && have_new_buffers) {
    have_new_buffers = FALSE;
    iter = element->sinkpads;
    for (i = 0; i < n && mux->num_bufs_in_current_batch < mux->batch_size; i++) {
      GstNvStreamMuxPadData *pad_data =
          (GstNvStreamMuxPadData *) g_object_get_data (G_OBJECT (iter->data),
          PAD_DATA_KEY);

      g_mutex_lock (&pad_data->queue_lock);

      if (!g_queue_is_empty (&pad_data->buf_queue)) {
        GstBuffer *src_buffer =
            GST_BUFFER (g_queue_pop_head (&pad_data->buf_queue));

        if (USE_CUDA_BATCH) {
          copy_data_cuda (mux, dest_data, src_buffer, &pad_data->in_videoinfo, pad_data);
          dest_meta->stream_id[dest_meta->num_filled] = pad_data->pad_id;
          dest_meta->is_valid[dest_meta->num_filled] = FALSE;
          dest_meta->buf_pts[dest_meta->num_filled] =
              GST_BUFFER_PTS (src_buffer);

          dest_meta->num_filled++;
#if 1
          cudaStreamSynchronize (mux->stream);
          gst_buffer_unref (src_buffer);
          src_buffer = NULL;
#endif
        }
        if (src_buffer)
          gst_buffer_list_add (buf_list, src_buffer);

        g_cond_broadcast (&pad_data->queue_cond);

        pad_data->num_bufs_in_current_batch++;
        mux->num_bufs_in_current_batch++;
        ret = TRUE;
        have_new_buffers = TRUE;
      }

      g_mutex_unlock (&pad_data->queue_lock);
      iter = iter->next;
    }
  }
#endif

  return ret;
}

GstClockTime
gst_get_current_clock_time (GstElement * element);
GstClockTime
gst_get_current_clock_time (GstElement * element)
{
  GstClock *clock = NULL;
  GstClockTime ret;

  g_return_val_if_fail (GST_IS_ELEMENT (element), GST_CLOCK_TIME_NONE);

  clock = gst_element_get_clock (element);

  if (!clock) {
    GST_DEBUG_OBJECT (element, "Element has no clock");
    return GST_CLOCK_TIME_NONE;
  }

  ret = gst_clock_get_time (clock);
  gst_object_unref (clock);

  return ret;
}

GstClockTime
gst_get_current_running_time (GstElement * element, GstNvStreamMux *mux);
GstClockTime
gst_get_current_running_time (GstElement * element, GstNvStreamMux *mux)
{
  GstClockTime base_time = GST_CLOCK_TIME_NONE, clock_time = GST_CLOCK_TIME_NONE;

  g_return_val_if_fail (GST_IS_ELEMENT (element), GST_CLOCK_TIME_NONE);

  base_time = gst_element_get_base_time (element);

  if (!GST_CLOCK_TIME_IS_VALID (base_time)) {
    GST_DEBUG_OBJECT (element, "Could not determine base time");
    return GST_CLOCK_TIME_NONE;
  }

  clock_time = gst_get_current_clock_time (element);

  if (!GST_CLOCK_TIME_IS_VALID (clock_time)) {
    return GST_CLOCK_TIME_NONE;
  }

  if (clock_time < base_time) {
    GST_DEBUG_OBJECT (element, "Got negative current running time");
    return GST_CLOCK_TIME_NONE;
  }

  /* To make live-source=1/sync=1 work for non-live sources. */
  if ((base_time == 0) && (element->current_state != GST_STATE_PLAYING)) {
    return 0;
  }

  return clock_time - base_time - mux->ts_latency_offset;
}

/* Wait for clock to reach the timeout for pushing the buffer based on its PTS
 * Should be called with mux->mutex held. */
static GstClockReturn
wait_for_buffers (GstNvStreamMux * mux, GstClockTime out_buf_pts)
{

  GstClockTime latency = gst_nvstreammux_get_latency_unlocked (mux);
  /* Time when element went to playing state + PTS of the output buffer +
   * Time after which mux recieves buffer after it has been generated by the source */
   GstClockTime time =
      GST_ELEMENT_CAST (mux)->base_time + out_buf_pts + mux->max_latency;
  if (latency != GST_CLOCK_TIME_NONE)
      time += latency;

  GstClockTimeDiff jitter = 0;
  GstClockReturn clk_return;
  GstClock *clock = GST_ELEMENT_CLOCK (mux);

  /* Don't have clock. Can't wait. */
  if (!clock) {
    return GST_CLOCK_UNSUPPORTED;
  }

  mux->timeout_clk_id = gst_clock_new_single_shot_id (clock, time);
  g_mutex_unlock (&mux->ctx_lock);

  clk_return = gst_clock_id_wait (mux->timeout_clk_id, &jitter);
  if (clk_return == GST_CLOCK_EARLY) {
    GST_WARNING_OBJECT (mux,
        "Output buffer %" GST_TIME_FORMAT " is late. Jitter: %" GST_TIME_FORMAT,
        GST_TIME_ARGS (out_buf_pts), GST_TIME_ARGS (jitter));
  } else if (clk_return == GST_CLOCK_DONE || clk_return == GST_CLOCK_OK) {
    GST_DEBUG_OBJECT (mux,
        "Output buffer %" GST_TIME_FORMAT ". Jitter: %" GST_TIME_FORMAT,
        GST_TIME_ARGS (out_buf_pts), GST_TIME_ARGS (jitter));
  }

  g_mutex_lock (&mux->ctx_lock);
  gst_clock_id_unref (mux->timeout_clk_id);
  mux->timeout_clk_id = NULL;

  return clk_return;
}

static gboolean check_for_batch_align (GstNvStreamMux *mux)
{
    unsigned int count = 0;
    int i, n, j;
    GstObject *obj;
    n = GST_ELEMENT(mux)->numsinkpads;
    GList *iter = GST_ELEMENT(mux)->sinkpads;
    for (i = 0; i < n; i++)
    {
        GstNvStreamMuxPadData *pad_data = (GstNvStreamMuxPadData *)
            g_object_get_data (G_OBJECT (iter->data), PAD_DATA_KEY);
        GList *tail = pad_data->buf_queue.tail;
        while (tail != NULL)
        {
            obj = (GstObject *) tail->data;
            if (GST_IS_BUFFER(obj))
            {
               count++;
               break;
            }
            tail = tail->prev;

        }
        iter = iter->next;
    }

    if (count >= MIN(mux->batch_size, (unsigned int)n))
        return true;
    else
        return false;
}

static gboolean check_for_batch (GstNvStreamMux *mux)
{
    unsigned int count = 0;
    int i, n, j;
    GstObject *obj;
    n = GST_ELEMENT(mux)->numsinkpads;
    GList *iter = GST_ELEMENT(mux)->sinkpads;
    for (i = 0; i < n; i++)
    {
        GstNvStreamMuxPadData *pad_data = (GstNvStreamMuxPadData *)
            g_object_get_data (G_OBJECT (iter->data), PAD_DATA_KEY);
        GList *tail = pad_data->buf_queue.tail;
        while (tail != NULL)
        {
            obj = (GstObject *) tail->data;
            if (GST_IS_BUFFER(obj))
            {
                if (BUF_PTS_TO_RUNNING_TIME (obj) >  mux->cur_frame_pts)
                {
                   GST_DEBUG_OBJECT (mux, "buffer pts = %" GST_TIME_FORMAT "cur frame pts = %" GST_TIME_FORMAT "\n",
                          GST_TIME_ARGS(BUF_PTS_TO_RUNNING_TIME(obj)), GST_TIME_ARGS(mux->cur_frame_pts));
                   count++;
                   break;
                }
            }
            tail = tail->prev;

        }
        iter = iter->next;
    }

    if (count >= MIN(mux->batch_size, n - mux->num_pads_eos))
        return true;
    else
        return false;
}

static gint
compare_frame_meta_by_source_id(gconstpointer a, gconstpointer b)
{
    NvDsFrameMeta *meta1 = (NvDsFrameMeta *)a;
    NvDsFrameMeta *meta2 = (NvDsFrameMeta *)b;

//	 g_print("Comparing source_id: %u and %u\n", meta1->source_id, meta2->source_id);
    // Sort in ascending order of source_id
    return meta1->source_id - meta2->source_id;
}

static
guint64 timeval_diff_microseconds(struct timeval *t1, struct timeval *t2) {
    guint64 seconds = t2->tv_sec - t1->tv_sec;
    guint64 microseconds = t2->tv_usec - t1->tv_usec;

    // Convert seconds to microseconds and add the microsecond difference
    return seconds * 1000000UL + microseconds;
}


static gboolean check_for_batch_and_buffers(GstNvStreamMux *mux)
{
    unsigned int count = 0;
    int i, n;
    GstMiniObject *obj;
    n = GST_ELEMENT(mux)->numsinkpads;
    GList *iter = GST_ELEMENT(mux)->sinkpads;
    gboolean all_pads_have_buffer = FALSE;

    while (!all_pads_have_buffer) {
        all_pads_have_buffer = TRUE;
        count = 0;

        for (i = 0; i < n; i++)
        {
            GstNvStreamMuxPadData *pad_data = (GstNvStreamMuxPadData *)
                g_object_get_data(G_OBJECT(iter->data), PAD_DATA_KEY);

            g_mutex_lock(&pad_data->queue_lock);
            GList *head = pad_data->buf_queue.head;
            gboolean found_buffer = FALSE;

            while (head != NULL && !found_buffer)
            {
                obj = (GstMiniObject *)head->data;
                if (GST_IS_BUFFER(obj))
                {
                    GST_DEBUG_OBJECT(mux, "Found buffer on pad %d\n", pad_data->pad_id);
                    count++;
                    found_buffer = TRUE;
                }
                // We're not processing events, just skipping them
                head = head->next;
            }

            if (!found_buffer)
            {
                all_pads_have_buffer = FALSE;
            }

            g_mutex_unlock(&pad_data->queue_lock);
            iter = iter->next;
        }

        if (!all_pads_have_buffer || count < MIN(mux->batch_size, n - mux->num_pads_eos))
        {
            g_mutex_unlock(&mux->ctx_lock);
            g_usleep(10000);
            g_mutex_lock(&mux->ctx_lock);
            iter = GST_ELEMENT(mux)->sinkpads; // Reset iterator for next loop
        }
        else
        {
            break; // We have enough buffers for a batch
        }
    }

    return true;
}

static gboolean get_pts (GstNvStreamMux *mux)
{
    unsigned int count = 0;
    int i, n;
    GstObject *obj;
    n = GST_ELEMENT(mux)->numsinkpads;
    GList *iter = GST_ELEMENT(mux)->sinkpads;
    GstClockTime buffer_pts = 0;
    gboolean all_pads_have_three_buffers = TRUE;

    for (i = 0; i < n; i++)
    {
        GstNvStreamMuxPadData *pad_data = (GstNvStreamMuxPadData *)
            g_object_get_data (G_OBJECT (iter->data), PAD_DATA_KEY);
        GList *head = pad_data->buf_queue.head;
        int valid_buffers = 0;

        while (head != NULL && valid_buffers < 2)
        {
            obj = (GstObject *) head->data;
            if (GST_IS_BUFFER(obj))
            {
                GstVideoSEIMeta *meta = (GstVideoSEIMeta *) gst_buffer_get_meta ((GstBuffer*)obj, GST_VIDEO_SEI_META_API_TYPE);
                if (meta && meta->sei_metadata_type == (guint)GST_USER_SEI_META) {
                    unsigned char* jsonContentPtr = (unsigned char*)meta->sei_metadata_ptr;
                    size_t jsonContentSize = meta->sei_metadata_size;

                    std::string jsonContent(reinterpret_cast<const char*>(jsonContentPtr), jsonContentSize);
                    std::stringstream ss(jsonContent);
                    boost::property_tree::ptree pt;
                    boost::property_tree::read_json(ss, pt);

                    buffer_pts = pt.get<guint64>("timestamp", 0);

                    if (buffer_pts > 0) {
                        valid_buffers++;
                        if (buffer_pts > mux->highest_first_pts) {
                            mux->highest_first_pts = buffer_pts;
                            GST_DEBUG_OBJECT(mux, "mux->highest_first_pts=%ld pad_id=%d, buffer_pts=%ld meta=%p\n",
                                         mux->highest_first_pts, pad_data->pad_id, buffer_pts, meta);
                          break;
                        }
                    } else {
                        GST_WARNING_OBJECT(mux, "Invalid timestamp (0) in buffer for pad %d\n", pad_data->pad_id);
                    }
                } else {
                    GST_ERROR_OBJECT(mux, "No valid SEI metadata in buffer for pad %d\n", pad_data->pad_id);
                }
            }
            head = head->next;
        }

        if (valid_buffers < 2) {
            all_pads_have_three_buffers = FALSE;
            GST_ERROR_OBJECT(mux, "Pad %d has only %d valid buffers\n", pad_data->pad_id, valid_buffers);
        }

        iter = iter->next;
    }

    return all_pads_have_three_buffers;
}


static void
gst_nvstreammux_src_push_loop (gpointer user_data)
{
  static struct timeval t1, t2;
  GstNvStreamMux *mux = GST_NVSTREAMMUX (user_data);
  GstBufferList *buf_list = NULL;//gst_buffer_list_new_sized (mux->batch_size);
  guint i = 0, n = 0;
  gint64 end_time = -1;
  gboolean timedout = FALSE;
  gboolean send_eos = FALSE;
  gboolean all_pads_eos = FALSE;
  GstBuffer *out_buf = NULL;
  GstFlowReturn ret = GST_FLOW_OK;
  gpointer dest_data;
  NvBufSurface *surf;
  GstMapInfo out_buf_map = GST_MAP_INFO_INIT;
  guint batch_size = mux->current_batch_size;

#ifdef NEW_METADATA
  NvDsMeta *meta = NULL;
  NvDsBatchMeta *batch_meta = NULL;
#else
  GstNvStreamMeta *meta;
#endif
  GList *iter = NULL;

  char context_name[100];
  snprintf(context_name, sizeof(context_name), "%s_acquireBufferFromPool(Batch=%u)",
      GST_ELEMENT_NAME(mux), mux->frame_num);
  if (USE_CUDA_BATCH) {
    nvtx_helper_push_pop (context_name);
    ret = gst_buffer_pool_acquire_buffer (mux->output_buf_pool, &out_buf, NULL);
    nvtx_helper_push_pop (NULL);
    if (ret != GST_FLOW_OK) {
      return;
    }
  }

#if 0
  /* This can happen when downstream is returning buffers late,
   * now we need to adjust cur_frame_pts to running time */
  if (mux->sync_inputs)
  {
      if (gst_get_current_running_time (GST_ELEMENT (mux), mux) >
              mux->cur_frame_pts + mux->max_latency + 2 * (mux->timeout_usec * 1000))
      {
          mux->cur_frame_pts = gst_get_current_running_time (GST_ELEMENT (mux), mux);
      }
  }
#endif

  snprintf(context_name, sizeof(context_name), "%s_collectingBuffers(Batch=%u)",
      GST_ELEMENT_NAME(mux), mux->frame_num);
  g_mutex_lock (&mux->ctx_lock);
  iter = GST_ELEMENT (mux)->sinkpads;
  n = GST_ELEMENT (mux)->numsinkpads;
  //g_assert (iter != NULL);

  nvtx_helper_push_pop (context_name);
  gettimeofday (&t1, NULL);

#ifdef NEW_METADATA
  batch_meta = nvds_create_batch_meta(mux->batch_size);
  if(batch_meta == NULL) {
    // g_print("line = %d file = %s func = %s\n",__LINE__,__FILE__,__func__);
    exit(-1);
  }
  meta = gst_buffer_add_nvds_meta (out_buf, batch_meta, NULL,
      nvds_batch_meta_copy_func, nvds_batch_meta_release_func);

  meta->meta_type = NVDS_BATCH_GST_META;

  batch_meta->base_meta.batch_meta = batch_meta;
  batch_meta->base_meta.copy_func = nvds_batch_meta_copy_func;
  batch_meta->base_meta.release_func = nvds_batch_meta_release_func;
  batch_meta->max_frames_in_batch = mux->batch_size;
#else
  meta = gst_buffer_add_nvstream_meta (out_buf, mux->batch_size);
  meta->num_surfaces_per_frame = mux->num_surfaces_per_frame;
  meta->num_filled = 0;
#endif
  mux->num_bufs_in_current_batch = 0;
  mux->num_extra_bufs_in_batch = 0;

  gboolean dummy_clear_buffer_sent = FALSE;
  while (iter == NULL) {
    GST_ELEMENT_WARNING(mux, RESOURCE, NOT_FOUND,
        ("No Sources found at the input of muxer. Waiting for sources."), (NULL));

    /* Push the idle/keepalive batch downstream EXACTLY ONCE. gst_pad_push()
     * transfers ownership of out_buf to the peer, which unrefs it. The stock code
     * re-pushed the SAME out_buf on every iteration of this 1 Hz "no sources" wait
     * loop, dereferencing an already-freed buffer -> a GstBuffer use-after-free that
     * shows up as repeated "gst_pad_push: assertion 'GST_IS_BUFFER (buffer)' failed"
     * CRITICALs and an eventual refcount-underflow crash whenever the muxer drains to
     * zero sources for >1s (e.g. removing all streams before re-adding). Sending it
     * once is sufficient to flush downstream; a fresh out_buf is built on the next
     * task invocation once a source returns. */
    if (dummy_clear_buffer_sent == FALSE) {
      mux->last_flow_ret = gst_pad_push (mux->srcpad, out_buf);
      dummy_clear_buffer_sent = TRUE;
    }

    gint64 end_time;
    end_time = g_get_monotonic_time () + (1000 * 1000 /** in microseconds */);


    g_cond_wait_until (&mux->ctx_cond, &mux->ctx_lock, end_time);
    if(mux->all_pads_eos && (mux->no_pipeline_eos == FALSE)) {
      if (!mux->eos_sent) {
        gst_pad_push_event (mux->srcpad, gst_event_new_eos ());
        mux->eos_sent = TRUE;
      }
      g_mutex_unlock (&mux->ctx_lock);
      return;
    }
    iter = GST_ELEMENT (mux)->sinkpads;
    if(iter) {
      //return to allow allocation of a new out_buf
      g_mutex_unlock (&mux->ctx_lock);
      return;
    }
  }

  for (i = 0; i < n; i++) {
    GstNvStreamMuxPadData *pad_data =
        (GstNvStreamMuxPadData *) g_object_get_data (G_OBJECT (iter->data),
        PAD_DATA_KEY);
    pad_data->num_bufs_in_current_batch = 0;
    iter = iter->next;
  }

  if (mux->no_pipeline_eos == FALSE){
    send_eos = all_pads_eos = mux->all_pads_eos;
  }

  if (!mux->stop_task && send_eos && mux->eos_sent) {
    gst_buffer_unref(out_buf);
    g_cond_wait (&mux->ctx_cond, &mux->ctx_lock);
    g_mutex_unlock (&mux->ctx_lock);
    return;
  }

  if (USE_CUDA_BATCH) {
    gst_buffer_map (out_buf, &out_buf_map, GST_MAP_READ);
    surf = (NvBufSurface *) out_buf_map.data;
    NvBufSurfaceParams *list = surf->surfaceList;
    memset (surf, 0 , sizeof(NvBufSurface));
    surf->surfaceList = list;
    surf->batchSize = mux->batch_size;
    surf->memType = static_cast<NvBufSurfaceMemType>(mux->cuda_mem_type);
    surf->gpuId  = mux->gpu_id;
  }

  if (mux->align_inputs) {
    while (!mux->all_pads_eos && !mux->stop_task)
    {
      GstClockReturn cret;

      if (check_for_batch_align (mux))
        break;

      g_cond_wait (&mux->ctx_cond, &mux->ctx_lock);
    }
  }

  if (mux->sync_inputs) {
      while (!mux->all_pads_eos && !mux->stop_task)
      {
          GstClockReturn cret;

          if (check_for_batch (mux))
              break;

          cret = wait_for_buffers (mux, mux->cur_frame_pts) ;
          if (cret == GST_CLOCK_DONE || cret == GST_CLOCK_OK || cret == GST_CLOCK_EARLY)
              break;

          if (cret == GST_CLOCK_UNSUPPORTED)
          {
              g_cond_wait (&mux->ctx_cond, &mux->ctx_lock);
          }
      }
  }

#if 1
  /*pads are connected and start producing data*/
if (mux->align_first_buffer && !mux->first_batch_aligned) {
    gboolean all_pads_have_buffer = FALSE;
    GstBuffer *src_buffer_temp;
    GstNvStreamMuxPadData *pad_datatemp;
    while ((GST_ELEMENT(mux)->numsinkpads < batch_size)) {
        g_mutex_unlock (&mux->ctx_lock);
        g_usleep(1000);
        g_mutex_lock (&mux->ctx_lock);
    }
    check_for_batch_and_buffers(mux);
    while(!get_pts(mux)) {
      GST_DEBUG_OBJECT(mux, "Now repeating while loop for proper PTS\n");
      g_mutex_unlock (&mux->ctx_lock);
      g_usleep(1000);
      g_mutex_lock (&mux->ctx_lock);
    }
}
#endif

  while (mux->num_bufs_in_current_batch < batch_size) {
    gboolean buffers_collected =
        gst_nvstreammux_src_collect_buffers (mux, out_buf, surf, batch_meta,
        buf_list, timedout, batch_size);

    if (USE_CUDA_BATCH) {
      if(!mux->async_process)
        cudaStreamSynchronize (mux->stream);
      //gst_buffer_list_unref (buf_list);
      //buf_list = gst_buffer_list_new_sized (batch_size);
    }

    if (buffers_collected) {
      send_eos = FALSE;
      if (end_time == -1) {
        end_time = g_get_monotonic_time () + mux->timeout_usec;
      } else if (g_get_monotonic_time() >= end_time  && mux->timeout_usec > -1) {
        timedout = TRUE;
      }
    }

    if (mux->num_bufs_in_current_batch == batch_size || timedout) {
      break;
    } else if (mux->live_source &&
        mux->num_bufs_in_current_batch == (batch_size - mux->num_queues_empty)) {
      break;
    }

    if (mux->all_pads_eos) {
      g_usleep(10);
      timedout = TRUE;
      continue;
    }

    if (buffers_collected) {
      continue;
    }

    if (!mux->sync_inputs ) {
        if (mux->num_bufs_in_current_batch == 0 || mux->timeout_usec == -1) {
            if (!mux->stop_task) {
                g_cond_wait (&mux->ctx_cond, &mux->ctx_lock);
            }
            if (mux->stop_task){
                timedout = TRUE;
			}
        } else {
            if (!g_cond_wait_until (&mux->ctx_cond, &mux->ctx_lock, end_time)) {
                timedout = TRUE;
            }
        }
    }
    else {
        timedout = TRUE;
    }
  }

mux->first_batch_aligned=TRUE;

  if (USE_CUDA_BATCH) {
    gst_buffer_unmap (out_buf, &out_buf_map);
  }

  if (USE_CUDA_BATCH) {
    if (mux->num_bufs_in_current_batch > 0) {
      if(!mux->async_process)
        cudaStreamSynchronize (mux->stream);

      if (mux->live_source && !mux->sync_inputs){
        NvDsFrameMeta *frame_meta= nvds_get_nth_frame_meta(batch_meta->frame_meta_list,
            (batch_meta->num_frames_in_batch - 1));

        GST_BUFFER_PTS (out_buf) = gst_get_current_running_time (GST_ELEMENT (mux), mux);
      }
      else{
        GST_BUFFER_PTS (out_buf) = mux->cur_frame_pts;
      }

      gettimeofday (&t2, NULL);
      nvtx_helper_push_pop (NULL);
      GST_DEBUG_OBJECT(mux, "Pushing buffer %p, batch size %lu, PTS %" GST_TIME_FORMAT,
          out_buf, mux->num_bufs_in_current_batch, GST_TIME_ARGS(GST_BUFFER_PTS(out_buf)));


      if(nvds_enable_latency_measurement || nvds_latency_measurement_silent) {
        NvDsMetaList *l = NULL;
        NvDsMetaCompLatency *latency_metadata = NULL;
        for (l = mux->latency_metadata_list; l != NULL; l = l->next)
        {
          latency_metadata = (NvDsMetaCompLatency *)(l->data);
          latency_metadata->out_system_timestamp = nvds_get_current_system_timestamp();
        }
        guint list_len = g_list_length(mux->latency_metadata_list);
        guint len = 0;
        for (len = 0; len < list_len; len++)
        {
          mux->latency_metadata_list  = g_list_remove_link(mux->latency_metadata_list,
              g_list_first(mux->latency_metadata_list));
        }
      }

      if (mux->buffer_cache) {
        GList *l;
        guint j;
        NvDsFrameMeta *frame_meta=NULL;
        struct timeval current_time;
        guint64 diff_microseconds;
        gettimeofday(&current_time, NULL);
        GArray *seen_sources = g_array_new(FALSE, FALSE, sizeof(guint));

        // Iterate through existing frame meta list and collect seen source IDs
        for (l = batch_meta->frame_meta_list; l != NULL; l = l->next) {
          frame_meta = (NvDsFrameMeta *)(l->data);
          guint sourceId = frame_meta->source_id;
          g_array_append_val(seen_sources, sourceId);
        }
        // Iterate through all pad data and add cached data for unseen sources
        GList *iter = GST_ELEMENT(mux)->sinkpads;
        for (; iter;  iter=iter->next)
        {
          GstNvStreamMuxPadData *pad_data = (GstNvStreamMuxPadData *)
            g_object_get_data (G_OBJECT (iter->data), PAD_DATA_KEY);
          gboolean source_seen = FALSE;

          // Check if the pad's source ID is in the seen_sources array
          for (j = 0; j < seen_sources->len; j++) {
            if (g_array_index(seen_sources, guint, j) == pad_data->source_id) {
                source_seen = TRUE;
                //g_print("PROPER DATA ----> pad_data->source_id=%d, frame_meta->source_id=%d, frame_meta->ntp_timestamp=%lu\n", pad_data->source_id, frame_meta->source_id, frame_meta->ntp_timestamp);
                break;
            }
          }
         diff_microseconds = timeval_diff_microseconds(&pad_data->cached_time, &current_time);
         if(diff_microseconds > (guint64)mux->buffer_cache_timeout && pad_data->cached_src_buffers) {
           gst_buffer_replace(&pad_data->cached_src_buffers, NULL);
         }

          // If the source wasn't seen, add cached meta and surface params
         if (!source_seen && (mux->num_bufs_in_current_batch < batch_size)) {
           g_mutex_lock (&pad_data->queue_lock);
           GST_DEBUG_OBJECT (mux,"%s %s:%d %p %d %d\n",__FILE__,__func__,__LINE__, pad_data->cached_src_buffers, GST_IS_BUFFER(pad_data->cached_src_buffers), GST_IS_BUFFER(mux->gray_buffer));
           pad_data->timestamp_mega = frame_meta ? frame_meta->ntp_timestamp : 0;
           mux->is_current_buffer_cached=TRUE;
           gstnvstreammuxAddToBatch(mux, pad_data, gst_buffer_ref(pad_data->cached_src_buffers ? pad_data->cached_src_buffers:mux->gray_buffer), out_buf, surf, batch_meta, batch_size);
           mux->is_current_buffer_cached=FALSE;
           //gstnvstreammuxAddToBatch(mux, pad_data, gst_buffer_ref(mux->gray_buffer), out_buf, surf, batch_meta, batch_size);
           g_mutex_unlock (&pad_data->queue_lock);
           //g_print("CACHED_FRAME_DATA----> pad_data->source_id=%d, frame_meta->source_id=%d, frame_meta->ntp_timestamp=%lu, timestamp_mega=%lu\n", pad_data->source_id, frame_meta->source_id, frame_meta->ntp_timestamp, pad_data->timestamp_mega);
         }
        }
        // Clean up
        g_array_free(seen_sources, TRUE);
      }
      //g_print("***********************BATCH_OVER*********************************#####################################\n");
      GST_DEBUG_OBJECT (mux,"%d surf->batchSize=%d, surf->numFilled=%d batch_meta->num_frames_in_batch=%d, mux->num_bufs_in_current_batch=%ld\n",__LINE__, surf->batchSize, surf->numFilled, batch_meta->num_frames_in_batch, mux->num_bufs_in_current_batch);

      if (mux->sort_batch){
        batch_meta->frame_meta_list = g_list_sort(batch_meta->frame_meta_list, compare_frame_meta_by_source_id);
        for (GList *l = batch_meta->frame_meta_list; l != NULL; l = g_list_next(l)) {
            NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l->data);
            GST_DEBUG_OBJECT (mux,"Sorted Frame Meta - source_id: %u\n", frame_meta->source_id);
        }

        NvBufSurfaceParams tmp_surf_list[batch_meta->num_frames_in_batch];
        GstBuffer **tmp_orig_buffer_ptrs = (GstBuffer **)g_malloc0(sizeof(GstBuffer *) * batch_meta->num_frames_in_batch);
        guint batch_size = batch_meta->num_frames_in_batch;

        // Reorder surfaces based on sorted frame_meta_list
        guint i = 0;
        for (GList *l = batch_meta->frame_meta_list; l != NULL; l = g_list_next(l), i++) {
            NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l->data);
            tmp_surf_list[i] = surf->surfaceList[frame_meta->batch_id];
            tmp_orig_buffer_ptrs[i] = (gst_buffer_get_nvstream_memory (out_buf))->orig_buffer_ptrs[frame_meta->batch_id];
            frame_meta->batch_id = i;  // Update batch_id to new position

            GST_DEBUG_OBJECT (mux,"Reordered: source_id %u, old batch_id %u, new batch_id %u\n",
                    frame_meta->source_id, frame_meta->batch_id, i);
        }
        // Copy the reordered surfaces back to the original surface
        memcpy(surf->surfaceList, tmp_surf_list,
               batch_size * sizeof(NvBufSurfaceParams));
        memcpy((gst_buffer_get_nvstream_memory (out_buf))->orig_buffer_ptrs, 
               tmp_orig_buffer_ptrs, sizeof(GstBuffer *) * batch_size);
        g_free(tmp_orig_buffer_ptrs);
      }

#if 0
		GstClockTime min_ntp_ts = GST_CLOCK_TIME_NONE;
GstClockTime max_ntp_ts = 0;

for (GList *l = batch_meta->frame_meta_list; l != NULL; l = g_list_next(l), i++) {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l->data);

    // Update min and max NTP timestamps
    min_ntp_ts = MIN(min_ntp_ts, frame_meta->ntp_timestamp);
    max_ntp_ts = MAX(max_ntp_ts, frame_meta->ntp_timestamp);
}

// Check if the difference is greater than sync_inputs_ntp
GstClockTime ntp_diff = max_ntp_ts-min_ntp_ts;
if (ntp_diff > mux->sync_inputs_ntp) {
    // g_print("############################################Error: NTP timestamp difference %ld is greater than sync_inputs_ntp \n",ntp_diff);
}
#endif
      GST_DEBUG_OBJECT (mux, "STREAMMUX OUT BUFFER attached timestamp %"
                  GST_TIME_FORMAT,
                  GST_TIME_ARGS (GST_BUFFER_PTS(out_buf)));
      //g_print("[BATCH-PUSH] Pushing batch with %lu buffers, PTS=%" GST_TIME_FORMAT "\n",
              //mux->num_bufs_in_current_batch, GST_TIME_ARGS(GST_BUFFER_PTS(out_buf)));
      mux->last_flow_ret = gst_pad_push (mux->srcpad, out_buf);
      //g_print("[BATCH-PUSH-DONE] Batch pushed, flow_ret=%d\n", mux->last_flow_ret);

      if (!mux->sync_inputs)
        mux->cur_frame_pts += mux->frame_duration_nsec;
    } else {
      //g_print("[BATCH-NOT-PUSHED] No buffers to push (%lu), unreffing out_buf\n",
              //mux->num_bufs_in_current_batch);
      gst_buffer_unref (out_buf);
    }
    if (mux->sync_inputs)
    {
        if (!mux->all_pads_eos)
            mux->cur_frame_pts += mux->frame_duration_nsec;
    }
    if (mux->sync_inputs_ntp)
    {
        // CRITICAL FIX: Only advance base_ntp if we actually pushed buffers in this batch
        // If batch had 0 buffers, base_ntp should NOT advance, otherwise it runs ahead
        // and all incoming buffers become "too late" and get dropped
        if (!mux->all_pads_eos && mux->num_bufs_in_current_batch > 0) {
          //mux->base_ntp += mux->sync_inputs_ntp;
          //g_print("[BATCH-NTP-ADVANCE] Batch had %lu buffers, advancing base_ntp\n", 
                  //mux->num_bufs_in_current_batch);
            adjust_base_ntp(mux);
        } else if (mux->num_bufs_in_current_batch == 0) {
          //g_print("[BATCH-NTP-SKIP] Batch had 0 buffers, NOT advancing base_ntp (current=%ld)\n", 
                  //mux->base_ntp);
        }
        if (mux->first_batch) {
          mux->first_batch = FALSE;
        }
        if (mux->processed_pads)
          g_hash_table_remove_all(mux->processed_pads);
        g_atomic_int_set((gint*)&mux->holding_counter, 0);    // Thread-safe reset
        // Note: pad_holding_iterations are NOT reset here - they reset per-source when buffer is added
		  GST_DEBUG_OBJECT (mux,"!!!!!!!!!!!!!!!!!!!!!!!BATCH OVER !!!!!!!!!!!!!!!!!!!!!!!!!\n");
    }
//    gst_buffer_list_unref (buf_list);
  } else {
    gettimeofday (&t2, NULL);
    if (gst_buffer_list_length (buf_list) > 0) {
      mux->last_flow_ret = gst_nvstreammux_push_buffers (mux, buf_list);
    } else {
      gst_buffer_list_unref (buf_list);
    }
  }

  if (0) {
    double elapsedTime = 0;

    elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;
    elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;

    g_print ("(%s): %s ElaspedTime=%f\n",
        GST_ELEMENT_NAME(mux),
        "nvstreammux", elapsedTime);
  }

  while (mux->event_list) {
    GstEvent *event = GST_EVENT (mux->event_list->data);
    iter = g_list_first(mux->event_list);
    gst_pad_push_event (mux->srcpad, event);
    mux->event_list = g_list_remove_link (mux->event_list, iter);
    g_list_free (iter);
  }

  if (send_eos && !mux->eos_sent) {
    gst_pad_push_event (mux->srcpad, gst_event_new_eos ());
    mux->eos_sent = TRUE;
  }
  g_mutex_unlock (&mux->ctx_lock);
}

static GstStateChangeReturn
gst_nvstreammux_change_state (GstElement * element, GstStateChange transition)
{
  GstNvStreamMux *mux = GST_NVSTREAMMUX (element);
  GstStateChangeReturn ret;
  GList *iter = element->sinkpads;
  guint i, n;


  cudaError_t CUerr = cudaSuccess;
  CUerr = cudaSetDevice(mux->gpu_id);

  mux->is_integrated = 0;
  NvBufSurfaceDeviceInfo dev_info{};
  if (NvBufSurfaceGetDeviceInfo (&dev_info) == 0) {
    if (dev_info.driverType == NVBUF_DRIVER_TYPE_NVGPU) {
      mux->is_integrated = 1;
    }
  }

  if(CUerr != cudaSuccess)
  {
    g_print ("Unable to set device in %s\n", __func__);
    return GST_STATE_CHANGE_FAILURE;
  }
  GST_TRACE_OBJECT (mux, "SETTING CUDA DEVICE = %d in nvstreammux func=%s\n", mux->gpu_id, __func__);

  switch (transition) {
    case GST_STATE_CHANGE_NULL_TO_READY:
    {
      if(mux->width == 0)
      {
        GST_ELEMENT_ERROR (mux, LIBRARY, SETTINGS,
            ("Output width not set"),
            (nullptr));
        return GST_STATE_CHANGE_FAILURE;
      }
      if(mux->height == 0)
      {
        GST_ELEMENT_ERROR (mux, LIBRARY, SETTINGS,
            ("Output height not set"),
            (nullptr));
        return GST_STATE_CHANGE_FAILURE;
      }



      if(mux->batch_size == 0)
      {
        GST_ELEMENT_ERROR (mux, LIBRARY, SETTINGS,
            ("Batch size not set"),
            (nullptr));
        return GST_STATE_CHANGE_FAILURE;
      }

      if (mux->sync_inputs && (mux->timeout_usec == DEFAULT_BATCHED_PUSH_TIMEOUT))
      {
          GST_ELEMENT_ERROR (mux, LIBRARY, SETTINGS,
            ("Batch push timeout not set"),
            (nullptr));
          return GST_STATE_CHANGE_FAILURE;
      }

      if(mux->cuda_mem_type == NVBUF_MEM_DEFAULT)
      {
        if(mux->is_integrated)
          mux->cuda_mem_type = NVBUF_MEM_SURFACE_ARRAY;
        else
          mux->cuda_mem_type = NVBUF_MEM_CUDA_DEVICE;
      }

      if (mux->stream == NULL)
      {
        cudaStreamCreate (&mux->stream);
      }
      mux->current_loc = 0;

#ifdef USE_NPPSTREAM
      if (mux->nppStream == NULL)
      {
        cudaStreamCreate (&mux->nppStream);
      }
#endif
    }
      break;
    case GST_STATE_CHANGE_READY_TO_NULL:
    {
      if (mux->stream)
      {
        cudaStreamDestroy (mux->stream);
        mux->stream = NULL;
      }
      if (mux->nppStream)
      {
        cudaStreamDestroy (mux->nppStream);
        mux->nppStream = NULL;
      }
      if (mux->pad_indexes) {
        g_hash_table_unref (mux->pad_indexes);
        mux->pad_indexes = NULL;
      }
      if (mux->pad_framerates) {
        if (mux->pad_framerates->table) {
          g_mutex_lock (&(mux->pad_framerates->read_write_lock));
          g_hash_table_destroy (mux->pad_framerates->table);
          mux->pad_framerates->table = NULL;
          g_mutex_unlock (&(mux->pad_framerates->read_write_lock));
        }
        g_free (mux->pad_framerates);
        mux->pad_framerates = NULL;
      }

    }
      break;
    case GST_STATE_CHANGE_READY_TO_PAUSED:
    {
      g_mutex_lock (&mux->ctx_lock);
      if (mux->enable_adaptive_batch_size)
        mux->current_batch_size = MAX(MIN (mux->batch_size, GST_ELEMENT(mux)->numsinkpads), 1);
      else
        mux->current_batch_size = mux->batch_size;
      g_mutex_unlock (&mux->ctx_lock);

      if(mux->current_batch_size % mux->num_surfaces_per_frame)
      {
        // g_print("***** Error: Batch size should be multiple of num-surfaces-per-frame. Please check the configuration file \n");
        return GST_STATE_CHANGE_FAILURE;
      }

      n = element->numsinkpads;
      GList *iter = element->sinkpads;
      for (i = 0; i < n; i++) {
        GstNvStreamMuxPadData *pad_data = (GstNvStreamMuxPadData *)
            g_object_get_data (G_OBJECT (iter->data),
            PAD_DATA_KEY);
        pad_data->stopping = FALSE;
        iter = iter->next;
      }
      mux->stop_task = FALSE;
    }
      break;
    case GST_STATE_CHANGE_PAUSED_TO_READY:
    {
      guint i;

#if 1
      if (USE_CUDA_BATCH && mux->output_buf_pool) {
        gst_buffer_pool_set_active (mux->output_buf_pool, FALSE);
        if (mux->black_out_buf_pool) {
          gst_buffer_pool_set_active (mux->black_out_buf_pool, FALSE);
        }
      }
#endif
      if (mux->batch_method != BATCH_METHOD_NONE) {
        g_mutex_lock (&mux->ctx_lock);
        mux->stop_task = TRUE;
        COND_BROADCAST (mux);
        g_mutex_unlock (&mux->ctx_lock);
        if (mux->pad_task_created) {
          gst_pad_stop_task (mux->srcpad);
        }
        mux->pad_task_created = FALSE;
      }
#if 1
      if (USE_CUDA_BATCH && mux->output_buf_pool) {
        gst_object_unref (mux->output_buf_pool);
        mux->output_buf_pool = NULL;
        if (mux->black_out_buf_pool) {
          gst_object_unref (mux->black_out_buf_pool);
          mux->black_out_buf_pool = NULL;
        }
      }
      gst_buffer_replace(&mux->gray_buffer, NULL);
#endif
      GList *iter = element->sinkpads;
      g_mutex_lock (&mux->ctx_lock);
      while(iter) {
        GstNvStreamMuxPadData *pad_data = (GstNvStreamMuxPadData *)
            g_object_get_data (G_OBJECT (iter->data),
            PAD_DATA_KEY);
        gst_nvstreammux_reset_pad_data (pad_data);
        iter = iter->next;
      }
      g_mutex_unlock (&mux->ctx_lock);
      mux->num_pads_eos = 0;
      mux->num_queues_empty = 0;
      mux->all_pads_eos = FALSE;
      mux->eos_sent = FALSE;
      mux->last_flow_ret = GST_FLOW_OK;
      mux->frame_num = 0;
      mux->segment_sent = FALSE;
      mux->cur_frame_pts = 0;
      mux->sync_inputs = FALSE;
      mux->align_inputs = FALSE;
      mux->current_loc = 0;
    }
      break;
    default:
      break;
  }
  ret = GST_ELEMENT_CLASS (parent_class)->change_state (element, transition);
  return ret;
}

static void
gst_nvstreammux_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstNvStreamMux *mux = GST_NVSTREAMMUX (object);
  switch (prop_id) {
    case PROP_PUSH_MODE_BATCHED:
      mux->batch_method = (GstNvStreamMuxBatchMethod)g_value_get_enum (value);
      break;
    case PROP_BATCH_SIZE:
      mux->batch_size = g_value_get_uint (value);
      break;
    case PROP_BATCHED_PUSH_TIMEOUT:
      mux->timeout_usec = g_value_get_int (value);
      break;
    case PROP_WIDTH:
      mux->width = g_value_get_uint (value);
      break;
    case PROP_HEIGHT:
      mux->height = g_value_get_uint (value);
      break;
    case PROP_GPU_DEVICE_ID:
      mux->gpu_id = g_value_get_uint (value);
      break;
    case PROP_NUM_SURFACES_PER_FRAME:
      mux->num_surfaces_per_frame = g_value_get_uint (value);
      break;
    case PROP_LIVE_SOURCE:
      mux->live_source = g_value_get_boolean(value);
      break;
    case PROP_SYNC_INPUTS:
      mux->sync_inputs = g_value_get_boolean(value);
      break;
    case PROP_ALIGN_INPUTS:
      mux->align_inputs = g_value_get_boolean(value);
      break;
    case PROP_ATTACH_SYS_TIME_STAMP:
      mux->sys_ts = g_value_get_boolean(value);
      break;
    case PROP_ADAPTIVE_BATCH_SIZE:
      mux->enable_adaptive_batch_size = g_value_get_boolean(value);
      break;
    case PROP_ENABLE_PADDING:
      mux->enable_padding = g_value_get_boolean (value);
      break;
    case PROP_QUERY_RESOLUTION:
      mux->query_resolution = g_value_get_boolean (value);
      break;
    case PROP_COMPUTE_HW:
      mux->compute_hw = g_value_get_enum (value);
      break;
    case PROP_NVBUF_MEMORY_TYPE:
      mux->cuda_mem_type = g_value_get_enum (value);
      break;
    case PROP_INTERPOLATION_METHOD:
      mux->interpolation_method = g_value_get_enum (value);
      break;
    case PROP_BUFFER_POOL_SIZE:
      mux->buffer_pool_size = g_value_get_uint (value);
      // Ensure failsafe_flush_count doesn't exceed new buffer_pool_size - 1
      if (mux->failsafe_flush_count >= mux->buffer_pool_size) {
        guint old_value = mux->failsafe_flush_count;
        mux->failsafe_flush_count = (mux->buffer_pool_size > 1) ? (mux->buffer_pool_size - 1) : 1;
        GST_WARNING_OBJECT(mux, "failsafe-flush-count (%u) exceeded buffer-pool-size (%u), capping to %u",
                          old_value, mux->buffer_pool_size, mux->failsafe_flush_count);
      }
      break;
    case PROP_FAILSAFE_FLUSH_COUNT:
      {
        guint requested_value = g_value_get_uint (value);
        // Cap to buffer_pool_size - 1
        if (requested_value >= mux->buffer_pool_size) {
          mux->failsafe_flush_count = (mux->buffer_pool_size > 1) ? (mux->buffer_pool_size - 1) : 1;
          GST_WARNING_OBJECT(mux, "failsafe-flush-count requested value (%u) exceeds buffer-pool-size (%u), "
                            "capping to %u", requested_value, mux->buffer_pool_size, mux->failsafe_flush_count);
        } else {
          mux->failsafe_flush_count = requested_value;
        }
      }
      break;
    case PROP_MAX_LATNECY:
      mux->max_latency = g_value_get_uint (value);
      break;
    case PROP_FRAME_NUM_RESET_ON_EOS:
      mux->frame_num_reset_on_eos = g_value_get_boolean (value);
      break;
    case PROP_FRAME_NUM_RESET_ON_STREAM_RESET:
      mux->frame_num_reset_on_stream_reset = g_value_get_boolean (value);
      break;
    case PROP_FRAME_DURATION:
      {
        guint64 ms_value = g_value_get_uint64 (value);
        if (ms_value != GST_CLOCK_TIME_NONE) {
          mux->frame_duration = (GstClockTime) ms_value * GST_MSECOND;
        }
	else
          mux->frame_duration = GST_CLOCK_TIME_NONE;
        break;
      }
    case PROP_ASYNC_PROCESS:
      mux->async_process = g_value_get_boolean(value);
      break;
    case PROP_NO_PIPELINE_EOS:
      mux->no_pipeline_eos = g_value_get_boolean(value);
      break;
    case PROP_EXTRACT_SEI_TYPE5_DATA:
      mux->extract_sei_type5_data = g_value_get_boolean (value);
      break;
    case PROP_EXTRACT_SIM_TIME:
      mux->extract_sei_sim_time = g_value_get_boolean (value);
      break;
    case PROP_SORT_BATCH_BUFFERS:
      mux->sort_batch = g_value_get_boolean(value);
      break;
    case PROP_CACHE_BUFFERS:
      mux->buffer_cache = g_value_get_boolean(value);
      break;
    case PROP_CACHE_BUFFERS_TIMEOUT:
      mux->buffer_cache_timeout = g_value_get_int (value);
      break;
    case PROP_ALIGN_FIRST_BUFFER:
      mux->align_first_buffer = g_value_get_boolean(value);
      break;
    case PROP_SYNC_INPUTS_NTP:
      mux->sync_inputs_ntp = g_value_get_int (value);
      // sync-inputs-ntp property configured
      if(mux->sync_inputs_ntp > 0) {
        mux->fps=calculate_fps(mux->sync_inputs_ntp);
        GST_DEBUG_OBJECT (mux, "FPS VALUE Calculated as %ld\n", mux->fps);
      }
      break;
    case PROP_DROP_BACKWARD_SEI:
      mux->drop_backward_sei = g_value_get_boolean (value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static void
gst_nvstreammux_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstNvStreamMux *mux = GST_NVSTREAMMUX (object);

  switch (prop_id) {
    case PROP_PUSH_MODE_BATCHED:
      g_value_set_enum (value, mux->batch_method);
      break;
    case PROP_BATCH_SIZE:
      g_value_set_uint (value, mux->batch_size);
      break;
    case PROP_BATCHED_PUSH_TIMEOUT:
      g_value_set_int (value, mux->timeout_usec);
      break;
    case PROP_WIDTH:
      g_value_set_uint (value, mux->width);
      break;
    case PROP_HEIGHT:
      g_value_set_uint (value, mux->height);
      break;
    case PROP_GPU_DEVICE_ID:
      g_value_set_uint (value, mux->gpu_id);
      break;
    case PROP_NUM_SURFACES_PER_FRAME:
      g_value_set_uint (value, mux->num_surfaces_per_frame);
      break;
    case PROP_ENABLE_PADDING:
      g_value_set_boolean (value, mux->enable_padding);
      break;
    case PROP_QUERY_RESOLUTION:
      g_value_set_boolean (value, mux->query_resolution);
      break;
    case PROP_LIVE_SOURCE:
      g_value_set_boolean (value, mux->live_source);
      break;
    case PROP_SYNC_INPUTS:
      g_value_set_boolean (value, mux->sync_inputs);
      break;
    case PROP_ALIGN_INPUTS:
      g_value_set_boolean (value, mux->align_inputs);
      break;
   case PROP_ATTACH_SYS_TIME_STAMP:
      g_value_set_boolean (value, mux->sys_ts);
      break;
    case PROP_ADAPTIVE_BATCH_SIZE:
      g_value_set_boolean (value, mux->enable_adaptive_batch_size);
      break;
    case PROP_COMPUTE_HW:
      g_value_set_enum (value, mux->compute_hw);
      break;
    case PROP_NVBUF_MEMORY_TYPE:
      g_value_set_enum (value, mux->cuda_mem_type);
      break;
    case PROP_INTERPOLATION_METHOD:
      g_value_set_enum (value, mux->interpolation_method);
      break;
    case PROP_BUFFER_POOL_SIZE:
      g_value_set_uint (value, mux->buffer_pool_size);
      break;
    case PROP_FAILSAFE_FLUSH_COUNT:
      g_value_set_uint (value, mux->failsafe_flush_count);
      break;
    case PROP_MAX_LATNECY:
      g_value_set_uint (value, mux->max_latency);
      break;
    case PROP_FRAME_NUM_RESET_ON_EOS:
      g_value_set_boolean (value, mux->frame_num_reset_on_eos);
      break;
    case PROP_FRAME_NUM_RESET_ON_STREAM_RESET:
      g_value_set_boolean (value, mux->frame_num_reset_on_stream_reset);
      break;
    case PROP_FRAME_DURATION:
      {
        guint64 ms_value = GST_CLOCK_TIME_NONE;
        if (mux->frame_duration >= 0) {
          ms_value = mux->frame_duration / GST_MSECOND;
        }
        g_value_set_uint64 (value, ms_value);
        break;
      }
    case PROP_ASYNC_PROCESS:
      g_value_set_boolean (value, mux->async_process);
      break;
    case PROP_NO_PIPELINE_EOS:
      g_value_set_boolean (value, mux->no_pipeline_eos);
      break;
    case PROP_EXTRACT_SEI_TYPE5_DATA:
      g_value_set_boolean (value, mux->extract_sei_type5_data);
      break;
    case PROP_EXTRACT_SIM_TIME:
      g_value_set_boolean (value, mux->extract_sei_sim_time);
      break;
    case PROP_SORT_BATCH_BUFFERS:
      g_value_set_boolean (value, mux->sort_batch);
      break;
   case PROP_CACHE_BUFFERS:
      g_value_set_boolean (value, mux->buffer_cache);
      break;
    case PROP_CACHE_BUFFERS_TIMEOUT:
      g_value_set_int (value, mux->buffer_cache_timeout);
      break;
    case PROP_ALIGN_FIRST_BUFFER:
      g_value_set_boolean (value, mux->align_first_buffer);
      break;
    case PROP_SYNC_INPUTS_NTP:
      g_value_set_int (value, mux->sync_inputs_ntp);
      break;
    case PROP_DROP_BACKWARD_SEI:
      g_value_set_boolean (value, mux->drop_backward_sei);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static void
gst_nvstreammux_class_init (GstNvStreamMuxClass * klass)
{
  GObjectClass *gobject_class = G_OBJECT_CLASS (klass);
  GstElementClass *gstelement_class = GST_ELEMENT_CLASS (klass);

  // Indicate the use of DS buf api version
  g_setenv ("DS_NEW_BUFAPI", "1", TRUE);

  gst_element_class_set_static_metadata (gstelement_class,
      "Stream multiplexer", "Generic", "N-to-1 pipe stream multiplexing",
      "NVIDIA Corporation. Post on Deepstream for Tesla forum for any queries "
      "@ https://devtalk.nvidia.com/default/board/209/");

  gobject_class->set_property = gst_nvstreammux_set_property;
  gobject_class->get_property = gst_nvstreammux_get_property;

//  g_object_class_install_property (gobject_class, PROP_PUSH_MODE_BATCHED,
//      g_param_spec_enum ("batching-method", "Batching Method",
//          "Method to form a batch of buffers received from multiple sources to be pushed together",
//          GST_TYPE_NVSTREAMMUX_BATCH_METHOD, DEFAULT_BATCH_METHOD,
//          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_BATCH_SIZE,
      g_param_spec_uint ("batch-size", "Batch Size",
          "Maximum number of buffers in a batch",
          0, MAX_NVBUFFERS, DEFAULT_BATCH_SIZE,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_BATCHED_PUSH_TIMEOUT,
      g_param_spec_int ("batched-push-timeout", "Batched Push Timeout",
          "Timeout in microseconds to wait after the first buffer is available\n"
          "\t\t\tto push the batch even if the complete batch is not formed.\n"
          "\t\t\tSet to -1 to wait infinitely",
          -1, G_MAXINT, DEFAULT_BATCHED_PUSH_TIMEOUT,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_WIDTH,
      g_param_spec_uint ("width", "Width",
          "Width of each frame in output batched buffer. This property MUST be set.",
          0, G_MAXUINT, DEFAULT_WIDTH,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_HEIGHT,
      g_param_spec_uint ("height", "Height",
          "Height of each frame in output batched buffer. This property MUST be set.",
          0, G_MAXUINT, DEFAULT_HEIGHT,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_NUM_SURFACES_PER_FRAME,
      g_param_spec_uint ("num-surfaces-per-frame", "Max number of surfaces per frame",
          "Max number of surfaces per frame",
          1, 4, 1,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property (gobject_class, PROP_ENABLE_PADDING,
      g_param_spec_boolean ("enable-padding", "Enable Padding",
        "Maintain input aspect ratio when scaling by padding with black bands.",
        FALSE, G_PARAM_READWRITE));

  if (0) {
    g_object_class_install_property (gobject_class, PROP_QUERY_RESOLUTION,
        g_param_spec_boolean ("query-resolution", "Query Resolution",
            "Query batched frame resolution from downstream element",
            DEFAULT_QUERY_RESOLUTION,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  }

  g_object_class_install_property (gobject_class, PROP_LIVE_SOURCE,
      g_param_spec_boolean ("live-source", "live source",
          "Boolean property to inform muxer that sources are live.",
          DEFAULT_LIVE_SOURCE,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_SYNC_INPUTS,
      g_param_spec_boolean ("sync-inputs", "Synchronize Inputs",
          "Boolean property to force sychronization of input frames.",
          0,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_ALIGN_INPUTS,
      g_param_spec_boolean ("align-inputs", "Align Inputs",
          "Boolean property to force timestamp align of input frames.",
          0,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  if (0) {
    g_object_class_install_property (gobject_class, PROP_ADAPTIVE_BATCH_SIZE,
        g_param_spec_boolean ("enable-adaptive-batch-size", "Enable Adaptive Batch Size",
            "Muxer will adjust the max batch size between [1, batch-size]\n"
            "\t\t\tcorresponding to the number of sources connected to the muxer",
            DEFAULT_ADAPTIVE_BATCH_SIZE,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  }

  g_object_class_install_property (gobject_class, PROP_GPU_DEVICE_ID,
      g_param_spec_uint ("gpu-id", "Set GPU Device ID",
          "Set GPU Device ID",
          0, G_MAXUINT, DEFAULT_GPU_DEVICE_ID,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_BUFFER_POOL_SIZE,
      g_param_spec_uint ("buffer-pool-size", "Buffer Pool Size",
          "Maximum number of buffers from muxer's output pool",
          DEFAULT_BUFFER_POOL_SIZE, MAX_POOL_BUFFERS, DEFAULT_BUFFER_POOL_SIZE,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_FAILSAFE_FLUSH_COUNT,
      g_param_spec_uint ("failsafe-flush-count", "Failsafe Flush Count",
          "Maximum iterations before forcing batch push to prevent buffer leaks (failsafe mechanism). "
          "Should be less than buffer-pool-size. If set higher, it will be capped at buffer-pool-size - 1.",
          1, MAX_POOL_BUFFERS, DEFAULT_FAILSAFE_FLUSH_COUNT,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_DROP_BACKWARD_SEI,
      g_param_spec_boolean ("drop-backward-sei", "Drop Backward SEI",
          "Drop incoming buffers whose SEI timestamp is behind the previous buffer's SEI timestamp (per sink pad)",
          DEFAULT_DROP_BACKWARD_SEI,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_MAX_LATNECY,
      g_param_spec_uint ("max-latency", "maximum lantency",
          "Additional latency in live mode to allow upstream to take longer to produce buffers for the current position (in nanoseconds)",
          0, G_MAXUINT, 0/*200000000*/,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_ATTACH_SYS_TIME_STAMP,
      g_param_spec_boolean ("attach-sys-ts", "Set system timestamp as ntp timestamp",
          "If set to TRUE, system timestamp will be attached as ntp timestamp.\n"
          "\t\t\tIf set to FALSE, ntp timestamp from rtspsrc, if available, will be attached.",
          DEFAULT_ATTACH_SYS_TIME_STAMP,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  //PROP_NVDS_MEMORY_TYPE_INSTALL (gobject_class);
  PROP_NVBUF_MEMORY_TYPE_INSTALL(gobject_class);
  PROP_COMPUTE_HW_INSTALL(gobject_class);

  g_object_class_install_property (gobject_class, PROP_INTERPOLATION_METHOD,
      g_param_spec_enum ("interpolation-method", "Interpolation-method",
          "Set interpolation methods",
          GST_TYPE_INTERPOLATION_METHOD, NvBufSurfTransformInter_Bilinear,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
          GST_PARAM_CONTROLLABLE | G_PARAM_CONSTRUCT)));

  g_object_class_install_property (gobject_class, PROP_FRAME_NUM_RESET_ON_EOS,
      g_param_spec_boolean ("frame-num-reset-on-eos", "Frame Number Reset on EOS",
          "Reset frame numbers to 0 for a source from which EOS is received (For debugging purpose only)",
          FALSE, (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_FRAME_NUM_RESET_ON_STREAM_RESET,
      g_param_spec_boolean ("frame-num-reset-on-stream-reset", "Frame Number Reset on stream reset",
          "Reset frame numbers to 0 for a source which needs to be reset. (For debugging purpose only)\n"
          "Needs to be paired with tracking-id-reset-mode=1 in the tracker config.",
          FALSE, (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_FRAME_DURATION,
      g_param_spec_uint64 ("frame-duration", "Frame duration",
          "Duration of input frames in milliseconds for use in NTP timestamp correction based on frame rate.\n"
          "\t\t\tIf set to 0, frame duration is inferred automatically from PTS values.\n"
          "\t\t\tIf set to -1, disables frame rate based NTP timestamp correction. (default)",
          0, G_MAXUINT64, DEFAULT_FRAME_DURATION,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property (gobject_class, PROP_ASYNC_PROCESS,
      g_param_spec_boolean ("async-process", "Asynchronous Processing",
          "Boolean property to enable/disable asynchronous processing of input frames for performance.",
          DEFAULT_ASYNC_PROCESS, (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property (gobject_class, PROP_NO_PIPELINE_EOS,
      g_param_spec_boolean ("drop-pipeline-eos", "No Pipeline EOS",
          "Boolean property so that EOS is not propagated downstream when all the sink pads are at EOS. (Experimental)",
          DEFAULT_NO_PIPELINE_EOS, (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_EXTRACT_SEI_TYPE5_DATA,
        g_param_spec_boolean ("extract-sei-type5-data",
            "extract-sei-type5-data",
            "Set to extract and attach SEI type5 unregistered data on output buffer, for debugging purpose",
            DEFAULT_SEI_EXTRACT_DATA,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_EXTRACT_SIM_TIME,
        g_param_spec_boolean ("extract-sei-sim-time",
            "extract-sei-sim-time",
            "Set to extract and attach simulation time on output buffer",
            DEFAULT_SEI_EXTRACT_SIM_TIME,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_SORT_BATCH_BUFFERS,
      g_param_spec_boolean ("sort-batch", "Sort Batch Buffer",
          "Sort Batch Buffer in ascending order of pad-ids",
          DEFAULT_SORT_BATCH_BUFFERS, (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property (gobject_class, PROP_CACHE_BUFFERS,
      g_param_spec_boolean ("cache-buffer", "Render cached frame",
          "Render previous frame in case new frame is delayed",
          DEFAULT_CACHE_BATCH_BUFFERS, (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_CACHE_BUFFERS_TIMEOUT,
      g_param_spec_int ("cache-buffer-timeout", "Cache Batch Buffer Timeout",
          "Timeout in microseconds to reuse previous rendered buffer\n"
          "\t\t\tPost timeout and when no initial buffer available , grey image to be rendered\n"
          "\t\t\tSet to -1 to wait infinitely",
          -1, G_MAXINT, DEFAULT_CACHED_BUFFER_TIMEOUT,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_ALIGN_FIRST_BUFFER,
      g_param_spec_boolean ("align-first-buffer", "Align First Buffer",
          "Align only the first buffer of all sources within frame rate range",
          DEFAULT_ALIGN_FIRST_BATCH, (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_SYNC_INPUTS_NTP,
      g_param_spec_int ("sync-inputs-ntp", "PTS difference between incoming NTP frames",
          "Time in nanoseconds to synchronize incoming inputs based on ntp",
          0, G_MAXINT, DEFAULT_SYNC_INPUTS_NTP,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  gst_element_class_add_static_pad_template (gstelement_class,
      &nvstreammux_sinkpad_template);
  gst_element_class_add_static_pad_template (gstelement_class,
      &nvstreammux_srcpad_template);

  gstelement_class->request_new_pad =
      GST_DEBUG_FUNCPTR (gst_nvstreammux_request_new_pad);
  gstelement_class->release_pad =
      GST_DEBUG_FUNCPTR (gst_nvstreammux_release_pad);
  gstelement_class->change_state =
      GST_DEBUG_FUNCPTR (gst_nvstreammux_change_state);
}

static void
gst_nvstreammux_init (GstNvStreamMux * mux)
{
  g_mutex_init (&mux->ctx_lock);
  g_cond_init (&mux->ctx_cond);

  REPEAT_MODE = (g_getenv ("NVSTREAMMUX_REPEAT_MODE") != NULL);
  if (REPEAT_MODE) {
    GST_INFO_OBJECT (mux, "Repeat mode enabled");
  }

  mux->batch_method = DEFAULT_BATCH_METHOD;
  mux->timeout_usec = DEFAULT_BATCHED_PUSH_TIMEOUT;
  mux->batch_size = DEFAULT_BATCH_SIZE;
  mux->width = DEFAULT_WIDTH;
  mux->height = DEFAULT_HEIGHT;
  mux->query_resolution = DEFAULT_QUERY_RESOLUTION;
  mux->sei_based_frameId = FALSE;
  mux->live_source = DEFAULT_LIVE_SOURCE;
  mux->sys_ts = DEFAULT_ATTACH_SYS_TIME_STAMP;
  mux->enable_adaptive_batch_size = DEFAULT_ADAPTIVE_BATCH_SIZE;
  mux->buffer_cache = DEFAULT_CACHE_BATCH_BUFFERS;
  mux->buffer_cache_timeout = DEFAULT_CACHED_BUFFER_TIMEOUT;
  mux->max_latency = 0;//200000000;
  if (!g_strcmp0(g_getenv("NVSTREAMMUX_ADAPTIVE_BATCHING"), "yes"))
    mux->enable_adaptive_batch_size = TRUE;
  mux->last_flow_ret = GST_FLOW_OK;
  mux->num_surfaces_per_frame = 1;
#ifdef __aarch64__
  mux->cuda_mem_type = NVBUF_MEM_DEFAULT;
#else
  mux->cuda_mem_type = NVBUF_MEM_CUDA_DEVICE;
#endif
  mux->compute_hw = NvBufSurfTransformCompute_Default;
  mux->interpolation_method = NvBufSurfTransformInter_Bilinear;
  mux->segment_sent = FALSE;
  mux->buffer_pool_size = DEFAULT_BUFFER_POOL_SIZE;
  mux->failsafe_flush_count = DEFAULT_FAILSAFE_FLUSH_COUNT;
  mux->frame_num_reset_on_eos = FALSE;
  mux->frame_num_reset_on_stream_reset = FALSE;
  mux->processed_pads = NULL;

  mux->prev_batch_meta = FALSE;

  mux->frame_duration = DEFAULT_FRAME_DURATION;
  mux->async_process = DEFAULT_ASYNC_PROCESS;
  mux->no_pipeline_eos = DEFAULT_NO_PIPELINE_EOS;
  mux->extract_sei_type5_data = FALSE;
  mux->extract_sei_sim_time = DEFAULT_SEI_EXTRACT_SIM_TIME;
  mux->is_current_buffer_cached = FALSE;
  mux->sort_batch = DEFAULT_SORT_BATCH_BUFFERS;
  mux->align_first_buffer = DEFAULT_ALIGN_FIRST_BATCH;
  mux->highest_first_pts = 0;
  mux->all_first_buffers_received = FALSE;
  mux->first_batch_aligned = FALSE;
  mux->first_batch = TRUE;
  mux->sync_inputs_ntp = DEFAULT_SYNC_INPUTS_NTP;
  mux->fps=0;
  mux->base_ntp = GST_CLOCK_TIME_NONE;
  mux->one_time_in_batch=TRUE;
  // Note: holding_iterations moved to per-pad (pad_holding_iterations) for per-source failsafe
  mux->drop_backward_sei = DEFAULT_DROP_BACKWARD_SEI;

  mux->srcpad =
      gst_pad_new_from_static_template (&nvstreammux_srcpad_template, "src");
  gst_pad_set_query_function (mux->srcpad,
      GST_DEBUG_FUNCPTR (gst_nvstreammux_src_query));
  gst_pad_use_fixed_caps (mux->srcpad);

  gst_pad_set_event_function (mux->srcpad,
    GST_DEBUG_FUNCPTR (gst_nvstreammux_src_event));

  gst_element_add_pad (GST_ELEMENT (mux), mux->srcpad);

  mux->pad_indexes = g_hash_table_new (NULL, NULL);
  mux->pad_framerates = (GstNvMultiStreamPadFrameRates*)g_malloc0(sizeof(GstNvMultiStreamPadFrameRates));
  g_mutex_init (&(mux->pad_framerates->read_write_lock));
  mux->pad_framerates->table = g_hash_table_new_full (NULL, NULL, NULL, g_free);

  gst_segment_init (&mux->segment, GST_FORMAT_TIME);

  _dsmeta_quark = g_quark_from_static_string (NVDS_META_STRING);
}


