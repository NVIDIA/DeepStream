/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "gstnvstreammux.h"
#include "gstnvstreampad.h"

#include "gstnvdsmeta.h"
#include "nvbufsurftransform.h"
#include "gst-nvquery.h"
#include "gst-nvquery-internal.h"
#include "gst-nvmessage.h"
#include "gst-nvevent.h"
#include <stdio.h>
#include <string.h>
#include <npp.h>
#include <sys/time.h>
#include <condition_variable>
#include <mutex>
#include "nvtx_helper.h"
#include "nvbufsurface.h"
#include "nvstreammux.h"
#include "nvstreammux_batch.h"
#include "nvstreammux_pads.h"
#include "gstnvstreammux_impl.h"
#include "gstnvstreammux_audio.h"
#include "MuxConfigParser.h"
#include "gstnvbufaudio.h"
#include <unistd.h>

#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wpointer-arith"
//#pragma GCC diagn nostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wuninitialized"

#define MAX_NVBUFFERS 1024
#define MAX_SURFACES 8
#define DEFAULT_NUM_SURFACES 1
#define DEFAULT_ATTACH_SYS_TIME_STAMP TRUE
GST_DEBUG_CATEGORY_STATIC (gst_nvstreammux_debug);
#define GST_CAT_DEFAULT gst_nvstreammux_debug

#define USE_CUDA_BATCH 1
#define PAD_DATA_KEY "pad-data"

#define MAX_BUFFERS_IN_QUEUE 10

#define MIN_POOL_BUFFERS 2
#define MAX_POOL_BUFFERS 4

#define DEFAULT_NO_PIPELINE_EOS FALSE
#define DEFAULT_FRAME_DURATION GST_CLOCK_TIME_NONE

#define CEIL(a,b) (((a) + (b) - 1) / (b))

#define _do_init \
    GST_DEBUG_CATEGORY_INIT (gst_nvstreammux_debug, "nvstreammux", 0, "nvstreammux element");
#define gst_nvstreammux_2_parent_class parent_class
G_DEFINE_TYPE_WITH_CODE (GstNvStreamMux, gst_nvstreammux_2,
    GST_TYPE_ELEMENT, _do_init);

/** Note: Keep the debug interface definition below the call to G_DEFINE_TYPE_WITH_CODE()
 * For the category level via plugin name viz nvstreammux:5 to work */
#include "gstnvstreammuxdebug.h"

class GstNvStreammuxDebug : public  INvStreammuxDebug
{
    public:

    GstNvStreammuxDebug(GstElement* aMux) : mux(aMux) {}

    void DebugPrint(const char* format, ...)
    {
        va_list args;
        va_start(args, format);
        gst_debug_log_valist (GST_CAT_DEFAULT, GST_LEVEL_DEBUG, "", "", 0,
                (GObject *) mux, format, args);
        va_end(args);
    }
    public:
    GstElement* mux;
};

static gboolean REPEAT_MODE = FALSE;

  //"channels = (int) [ 1, MAX ], "
#define COMMON_AUDIO_CAPS \
  "channels = " GST_AUDIO_CHANNELS_RANGE ", " \
  "rate = (int) [ 1, MAX ]"

#define GST_CAPS_FEATURE_MEMORY_NVMM "memory:NVMM"

//            "format = (string) { U8, S16BE, S16LE, S24BE, S24LE, S32BE, S32LE, F32LE, F64LE }, "


/**
 * Sink Pad templates
 * For video, we support only NVMM buffers (could be from components like
 * nvvideoconvert or another instance of nvstreammux)
 * For audio, we support both non NVMM (direct from components like audioconvert)
 * and NVMM data (from another nvstreammux instance)
 */
static GstStaticPadTemplate nvstreammux_sinkpad_template =
    GST_STATIC_PAD_TEMPLATE ("sink_%u",
    GST_PAD_SINK,
    GST_PAD_REQUEST,
    GST_STATIC_CAPS (
        GST_VIDEO_CAPS_MAKE_WITH_FEATURES ("memory:NVMM",
            "{ " "NV12, RGBA, I420 }") "; "
        GST_VIDEO_CAPS_MAKE ("{ " "NV12, RGBA }") "; "
            "audio/x-raw, "
            "format = { " "S16LE, F32LE }, "
            "layout = (string) interleaved, "
            COMMON_AUDIO_CAPS "; "
            "audio/x-raw(memory:NVMM), "
            "format = { " "S16LE, F32LE }, "
            "layout = (string) interleaved, "
            COMMON_AUDIO_CAPS "; "
            ));

static GstStaticPadTemplate nvstreammux_srcpad_template =
GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (
            GST_VIDEO_CAPS_MAKE_WITH_FEATURES ("memory:NVMM",
            "{ " "NV12, RGBA, I420 }") "; "
            "audio/x-raw(memory:NVMM), "
            "format = { " "S16LE, F32LE }, "
            "layout = (string) interleaved, "
            COMMON_AUDIO_CAPS
            ));

enum
{
  PROP_0,
  PROP_BATCH_SIZE,
  PROP_BATCHED_PUSH_TIMEOUT,
  PROP_CONFIG_FILE_PATH,
  PROP_NUM_SURFACES_PER_FRAME,
  PROP_ATTACH_SYS_TIME_STAMP,
  PROP_SYNC_INPUTS,
  PROP_MAX_LATENCY,
  PROP_FRAME_NUM_RESET_ON_EOS,
  PROP_FRAME_NUM_RESET_ON_STREAM_RESET,
  PROP_FRAME_DURATION,
  PROP_NO_PIPELINE_EOS
};

/* Filter signals and args */
enum {
    SIGNAL_BUFFER_DROPPED,
    SIGNAL_LAST_SIGNAL
};

static guint nvstreammux_signals[SIGNAL_LAST_SIGNAL] = { 0 };

#define DEFAULT_BATCH_METHOD BATCH_METHOD_ROUND_ROBIN
#define DEFAULT_BATCH_SIZE 0
#define DEFAULT_BATCHED_PUSH_TIMEOUT 33000
#define DEFAULT_WIDTH 1280
#define DEFAULT_HEIGHT 720
#define DEFAULT_QUERY_RESOLUTION FALSE
#define DEFAULT_GPU_DEVICE_ID 0
#define DEFAULT_LIVE_SOURCE FALSE

#define GST_TYPE_NVSTREAMMUX_BATCH_METHOD (gst_nvstreammux_batch_method_get_type())

static void gst_nvstreammux_src_push_loop (gpointer user_data);

static gboolean
gst_nvstreammux_query_latency_unlocked (GstNvStreamMux * self, GstQuery * query)
{
  gboolean query_ret = FALSE, live = FALSE;
  GstClockTime our_latency = 0, min = 0, max = 0;
  query_ret = gst_pad_query_default (self->srcpad, GST_OBJECT (self), query);
  if (!query_ret) {
    GST_INFO_OBJECT (self, "Latency query failed");
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
  our_latency = (GstClockTime)self->helper->get_min_fps_duration().count();
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
  GstClockTime latency;
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
  latency += (GstClockTime)self->helper->get_min_fps_duration().count();
  return latency;
}

static gboolean
gst_nvstreammux_src_event (GstPad * pad, GstObject * parent, GstEvent * event)
{
  GstNvStreamMux *mux = GST_NVSTREAMMUX (parent);
  GstElement *element = GST_ELEMENT (parent);
  gboolean ret = TRUE;
  GstClockTime latency = 0;

  if (GST_EVENT_TYPE (event) == GST_EVENT_LATENCY)
  {
      /** Notification of new latency adjustment.
       * Sinks will use the latency information to adjust their synchronisation
       * This event comes from the sink and include the whole pipeline latency
       */
      gst_event_parse_latency (event, &latency);

      GST_DEBUG_OBJECT (mux, "latency %" GST_TIME_FORMAT, GST_TIME_ARGS (latency));

      /** Adding user-configured max_latency to the pipelineLatency
       * The mux latency of helper->get_min_fps_duration() is
       * added by the ISynchronizeBuffer API
       */
      mux->synch_buffer->SetPipelineLatency(latency + mux->max_latency);
  }
  gst_event_unref (event);
  return ret;
}


static GstFlowReturn
gst_nvstreammux_chain (GstPad * pad, GstObject * parent, GstBuffer * buffer)
{
     GstFlowReturn ret = GST_FLOW_OK;
     guint stream_index = 0;
     gchar * name = gst_pad_get_name(pad);
     GstNvStreamMux *mux = GST_NVSTREAMMUX (parent);
     if( name == NULL )
     {
       return GST_FLOW_ERROR;
     }
     else if (sscanf (name, "sink_%u", &stream_index) < 1)
     {
       g_free(name);
       return GST_FLOW_ERROR;
     }
     if(!mux->isAudio)
     {
       GstMapInfo in_info = GST_MAP_INFO_INIT;
       NvBufSurface *in_surf;
       gst_buffer_map (buffer, &in_info, GST_MAP_READ);
       in_surf = (NvBufSurface *) in_info.data;
       (void)in_surf;
       LOGD("in_surf->numFilled=%d\n", in_surf->numFilled);
       gst_buffer_unmap (buffer, &in_info);
       if((int32_t)in_surf->numFilled <= 0)
       {
         GST_ELEMENT_ERROR (mux, STREAM, FAILED,
           ("input buffer [stream_id=%d] is not NVMM format; FATAL error", stream_index), (nullptr));
         /** ignore the buffer as its empty / corrupted / non NVMM */
         gst_buffer_unref(buffer);
         return GST_FLOW_ERROR;
       }
     }
     GstSinkPad* sinkPad = (GstSinkPad*) mux->helper->get_pad(stream_index);

     /** Add ntp timestamp*/
     GstNvDsNtpCalculator* ntp_calc = sinkPad->get_ntp_calc(
             mux->ntp_calc_mode, mux->frame_duration);

     if (ntp_calc && !gst_nvds_ntp_calculator_have_ntp_sync_values (ntp_calc)) {
        GstQuery *nvquery = gst_nvquery_ntp_sync_new ();
        GstClockTime ntp_time_epoch_ns = 0, ntp_frame_timestamp = 0;
        GstClockTime avg_frame_time = 0;

        gst_pad_peer_query ((GstPad*)(sinkPad->wrapped), nvquery);
        _NtpData ntpdata;
        memset(&ntpdata, 0, sizeof(_NtpData));
        gst_nvquery_ntp_sync_parse (nvquery, &ntpdata);

        if (ntpdata.ntp_time_epoch_ns > 0) {
          ntp_time_epoch_ns = ntpdata.ntp_time_epoch_ns;
          ntp_frame_timestamp = ntpdata.frame_timestamp;
          avg_frame_time = ntpdata.avg_frame_time;
        }
        gst_nvds_ntp_calculator_add_ntp_sync_values (ntp_calc,
            ntp_time_epoch_ns, ntp_frame_timestamp, avg_frame_time);
        gst_query_unref (nvquery);
      }

      GstClockTime ntp_ts = 0;
      if(ntp_calc) {
       ntp_ts =
          gst_nvds_ntp_calculator_get_buffer_ntp (ntp_calc,
                  GST_BUFFER_PTS (buffer));
     }


     LOGD("got a buffer\n");
     GstBufferWrapper *gst_buffer =
         new GstBufferWrapper(buffer, ENTRY_BUFFER, BATCH_SEQUENCE_IN_BATCH,
                              ntp_ts, GST_BUFFER_PTS(buffer), GST_BUFFER_DURATION(buffer),
                              stream_index);
     /** TODO: check gst_buffer == nullptr or catch possible exception from GstBufferWrapper constructor */
     gst_buffer->SetAudioParams(mux->muxCtx->GetAudioParams(stream_index));
     gst_buffer->SetMemTypeNVMM(mux->muxCtx->IsMemTypeNVMM(stream_index));

     if((mux->helper->get_pad_state(sinkPad) == SOURCE_STATE_IDLE) && (!mux->isAudio) )
     {
        if(mux->cur_frame_pts==0)
        {
          mux->pts_offset = 0;
          mux->cur_frame_pts=1;
          mux->helper->set_pts_offset(0);
        }
        else
        {
          mux->pts_offset = mux->synch_buffer->GetCurrentRunningTime();
          mux->helper->set_pts_offset(0);
        }
     }

     mux->helper->add_buffer(stream_index, gst_buffer);
     g_free(name);

     return GST_FLOW_OK;
}


static GstCaps *
remove_width_height (GstCaps * caps)
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
    gst_caps_append_structure_full (ret_caps, str, NULL);
    str = gst_structure_copy (str);
    gst_caps_append_structure_full (ret_caps, str,
        gst_caps_features_from_string ("memory:NVMM"));
  }

  return ret_caps;
}

static GstCaps *
remove_framerate_memory_feautures (GstCaps * caps)
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
        NULL);
    gst_structure_set (str, "framerate", GST_TYPE_FRACTION_RANGE, 0, G_MAXINT,
        G_MAXINT, 1, NULL);
    gst_caps_append_structure_full (ret_caps, str, NULL);
    str = gst_structure_copy (str);
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
    gst_nvquery_batch_size_set (query, mux->helper->get_config_batch_size());
    return TRUE;
  }

  if (gst_nvquery_is_stream_caps (query)) {
    guint nStreamId = 0;
    gst_nvquery_stream_caps_parse_streamid(query, &nStreamId);
    GstCaps* sinkCaps = GST_CAPS (g_hash_table_lookup (mux->sink_pad_caps,
            nStreamId + (char *)NULL));
    LOGD("sink caps to string = %s\n", gst_caps_to_string(sinkCaps));
    gst_nvquery_stream_caps_set(query, sinkCaps);
    return TRUE;
  }

  if (gst_nvquery_is_numStreams_size (query)) {

    gst_nvquery_numStreams_size_set (query, GST_ELEMENT (mux)->numsinkpads);
    return TRUE;
  }

  if (gst_nvquery_is_uri_from_streamid(query)) {
    guint streamid;
    if(!gst_nvquery_uri_from_streamid_parse_streamid(query, &streamid)) {
      return FALSE;
    }
    std::string pad_name = "sink_" + std::to_string(streamid);

    for (GList *iter = GST_ELEMENT(mux)->sinkpads; iter; iter = iter->next) {
      if (pad_name == GST_PAD_NAME(iter->data)) {
        return gst_pad_peer_query(GST_PAD(iter->data), query);
      }
    }
    return FALSE;
  }

  return gst_pad_query_default (pad, parent, query);
}

#if 0
/* Functions below print the Capabilities in a human-friendly format */
static gboolean print_field (GQuark field, const GValue * value, gpointer pfx) {
  gchar *str = gst_value_serialize (value);

  g_print ("%s  %15s: %s\n", (gchar *) pfx, g_quark_to_string (field), str);
  g_free (str);
  return TRUE;
}

static void print_caps (const GstCaps * caps, const gchar * pfx) {
  guint i;

  g_return_if_fail (caps != NULL);

  if (gst_caps_is_any (caps)) {
    g_print ("%sANY\n", pfx);
    return;
  }
  if (gst_caps_is_empty (caps)) {
    g_print ("%sEMPTY\n", pfx);
    return;
  }

  for (i = 0; i < gst_caps_get_size (caps); i++) {
    GstStructure *structure = gst_caps_get_structure (caps, i);

    g_print ("%s%s\n", pfx, gst_structure_get_name (structure));
    gst_structure_foreach (structure, print_field, (gpointer) pfx);
  }
}
#endif

static gboolean gst_nvstreammux_is_video(GstCaps* caps) {
  GstStructure* caps_str = gst_caps_get_structure (caps, 0);
  const gchar* mimetype = gst_structure_get_name (caps_str);

  /** If mimetype is audio/x-raw, extract useful information and return */
  if(strcmp(mimetype, "video/x-raw") == 0) {
    return TRUE;
  }
  return FALSE;
}

static GstCaps* gst_nvstreammux_remove_memory_feature(GstCaps* ret_caps)
{
  GstCapsFeatures *tft;
  gint n, i;

  if(!ret_caps || !GST_IS_CAPS(ret_caps)
     || gst_caps_is_empty(ret_caps) || gst_caps_is_any(ret_caps)) {
    return ret_caps;
  }

  n = gst_caps_get_size (ret_caps);
  for (i = 0; i < n; i++) {
    ret_caps = gst_caps_make_writable(ret_caps);
    tft = gst_caps_get_features (ret_caps, i);
    if (gst_caps_features_get_size (tft))
        gst_caps_features_remove (tft, GST_CAPS_FEATURE_MEMORY_NVMM);
  }

  return ret_caps;
}

/**
 * @brief making sure ret_caps is compatible with mux sink pad template
 * @param pad [IN] (transfer-none)      The sink pad
 * @param ret_caps [IN] (transfer-full) The caps to intersect with pad's template
 * @return (transfer-full) intersection of ret_caps and pad's template caps;
 */
static GstCaps*
gst_nvstreammux_intersect_with_sink_caps (GstPad* pad, GstCaps* ret_caps)
{
  GstCaps* caps = nullptr;
  GstCaps* caps_template = nullptr;
  if(!pad || !ret_caps || !GST_IS_CAPS(ret_caps)
     || gst_caps_is_empty(ret_caps) || gst_caps_is_any(ret_caps)) {
    return nullptr;
  }
  caps_template = gst_pad_get_pad_template_caps (pad);
  LOGD("caps_template=[%s]\n", gst_caps_to_string(caps_template));
  caps = gst_caps_intersect (ret_caps, caps_template);
  gst_caps_unref(ret_caps);
  gst_caps_unref(caps_template);
  ret_caps = caps;

  return ret_caps;
}

static gboolean
gst_nvstreammux_sink_query (GstPad * pad, GstObject * parent, GstQuery * query)
{
  GstNvStreamMux *mux = GST_NVSTREAMMUX (parent);
  gchar * name = gst_pad_get_name(pad);
  guint stream_index = 0;
  GstQuery *peer_query;
  GstCaps* sink_caps;

  if( name == NULL ) {
    return FALSE;
  }
  else if (sscanf (name, "sink_%u", &stream_index) < 1){
    g_free(name);
    return FALSE;
  }

  LOGD("DEBUGME\n");

  if (gst_nvquery_is_batch_size (query)) {
    gst_nvquery_batch_size_set (query, 1);
    return TRUE;
  }
  if (gst_nvquery_is_ntp_sync (query)) {
    _NtpData ntpdata;

    gst_nvquery_ntp_sync_parse (query, &ntpdata);
    GstSinkPad* sinkPad = (GstSinkPad*) mux->helper->get_pad(stream_index);
    GstNvDsNtpCalculator* ntp_calc = sinkPad->get_ntp_calc(mux->ntp_calc_mode,
            mux->frame_duration);
    if (ntp_calc)
      gst_nvds_ntp_calculator_add_ntp_sync_values (ntp_calc,
              ntpdata.ntp_time_epoch_ns, ntpdata.frame_timestamp,
              ntpdata.avg_frame_time);

    return TRUE;
  }

  // TBD code a lock function in helper class returning
  // unique_lock<std::mutex> g_mutex_lock (&mux->ctx_lock);
  if (GST_QUERY_TYPE (query) == GST_QUERY_CAPS) {
    GstCaps *ret_caps = NULL;
    GstCaps *caps = NULL;
    GstCaps *filter = NULL;

    LOGD("DEBUGME\n");
    gst_query_parse_caps (query, &filter);
    LOGD("filter caps=[%s]\n", gst_caps_to_string(filter));

    if (gst_pad_has_current_caps (mux->srcpad)) {
      LOGD("DEBUGME\n");
      ret_caps = gst_pad_get_current_caps (mux->srcpad);
    } else {
      LOGD("DEBUGME\n");
      /** try querying caps on src pad without filter and check */
      if ((ret_caps = gst_pad_peer_query_caps (mux->srcpad, NULL)) != nullptr) {
        LOGD("ret_caps from src_pad is [%s]\n", gst_caps_to_string(ret_caps));
        /** remove memory:NVMM feature as this is not supported upstream */
        ret_caps = gst_nvstreammux_remove_memory_feature(ret_caps);
        LOGD("ret_caps from src_pad is removed NVMM [%s]\n", gst_caps_to_string(ret_caps));
        /** making sure ret_caps is compatible with mux sink pad template */
        ret_caps = gst_nvstreammux_intersect_with_sink_caps (pad, ret_caps);
        LOGD("ret_caps from src_pad after intersect is [%s]\n", gst_caps_to_string(ret_caps));
        if(!ret_caps || gst_caps_is_empty (ret_caps) || gst_caps_is_any (ret_caps)) {
          LOGD("src not returning caps yet\n");
          /** try querying caps on src pad with filter and check */
          ret_caps = gst_pad_peer_query_caps (mux->srcpad, filter);
          /** remove memory:NVMM feature as this is not supported upstream */
          ret_caps = gst_nvstreammux_remove_memory_feature(ret_caps);
          LOGD("ret_caps from src_pad with filter after intersect is [%s]\n", gst_caps_to_string(ret_caps));
          /** making sure ret_caps is compatible with mux sink pad template */
          ret_caps = gst_nvstreammux_intersect_with_sink_caps (pad, ret_caps);
          LOGD("ret_caps from src_pad with filter after intersect is [%s]\n", gst_caps_to_string(ret_caps));
          LOGD("ret_caps=[%s]\n", gst_caps_to_string(ret_caps));
          if(!ret_caps || gst_caps_is_empty (ret_caps) || gst_caps_is_any (ret_caps)) {
            ret_caps = gst_pad_get_pad_template_caps (pad);
          }
        }
      }
      else {
        ret_caps = gst_pad_get_pad_template_caps (pad);
      }
    }

    if(gst_nvstreammux_is_video(ret_caps)) {
      caps = remove_framerate_memory_feautures (ret_caps);
      gst_caps_unref (ret_caps);
      ret_caps = caps;

      if (mux->width && mux->height) {
        caps = remove_width_height (ret_caps);
        gst_caps_unref (ret_caps);
        ret_caps = caps;
      }
    }
    else {
      GstCapsFeatures *tft;
      gint n, i;
      n = gst_caps_get_size (ret_caps);
      for (i = 0; i < n; i++) {
          ret_caps = gst_caps_make_writable(ret_caps);
          tft = gst_caps_get_features (ret_caps, i);
          if (gst_caps_features_get_size (tft))
              gst_caps_features_remove (tft, GST_CAPS_FEATURE_MEMORY_NVMM);
      }
    }

    if (mux->query_resolution) {
      GstQuery *query = gst_nvquery_resolution_new ();
      if (gst_pad_peer_query (mux->srcpad, query)) {
        if(gst_nvstreammux_is_video(ret_caps)) {
          caps = remove_width_height (ret_caps);
          gst_caps_unref (ret_caps);
          ret_caps = caps;
        }
      }
      gst_query_unref (query);
    }
    LOGD("caps before intersect on filter=[%s]\n", gst_caps_to_string(ret_caps));
    if (filter) {
      caps = gst_caps_intersect (ret_caps, filter);
      gst_caps_unref (ret_caps);
      ret_caps = caps;
    }

    LOGD("ret_capsss=[%s]\n", gst_caps_to_string(ret_caps));
    gst_query_set_caps_result (query, ret_caps);
    gst_caps_unref (ret_caps);
  // TBD code a lock function in helper class returning
  // unique_lock<std::mutex> g_mutex_unlock (&mux->ctx_lock);
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
    if(gst_nvstreammux_is_video(acaps)) {

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
    }

    LOGD("peer query accept caps=[%s]\n", gst_caps_to_string(acaps));
    peer_query = gst_query_new_accept_caps (acaps);//gst_caps_from_string("audio/x-raw, layout=interleaved, rate=48000, channels=1, format=S16LE")); //acaps
    if (gst_pad_peer_query (mux->srcpad, peer_query)) {
      gst_query_parse_accept_caps_result (peer_query, &result);
    }

    LOGD("result=%d\n", result);

    gst_query_unref (peer_query);
    gst_caps_unref (acaps);

    gst_query_set_accept_caps_result (query, result);
    return TRUE;

  }
  return gst_pad_query_default (pad, parent, query);
}


static bool handle_caps(unsigned int pad_id, GstPad * pad, GstObject * parent, GstEvent * event)
{
    GstNvStreamMux *mux = GST_NVSTREAMMUX (parent);
    GstElement *element = GST_ELEMENT (parent);
      GstCaps *caps = NULL;
      GstCaps *caps_copy = NULL;
      GstEvent *event_caps;
      GstEvent *new_event;
      GstStructure *caps_str;
      gboolean needs_conversion = FALSE;
      gboolean event_ret = FALSE;
      gint width_val=0, height_val=0;
      guint n,i;
      GValue *wd, *ht;

      gst_event_parse_caps (event, &caps);
      caps_copy = gst_caps_copy(caps);

      g_mutex_lock (&mux->ctx_lock);
      g_hash_table_insert (mux->sink_pad_caps, pad_id + (char *)NULL, caps_copy);
      caps_str = gst_caps_get_structure (caps, 0);
      const gchar* mimetype = gst_structure_get_name (caps_str);
      GstCapsFeatures* caps_ftr = gst_caps_get_features (caps, 0);

      if(gst_caps_features_contains(caps_ftr, GST_CAPS_FEATURE_MEMORY_NVMM)) {
        mux->muxCtx->SetMemTypeNVMM(pad_id, true);
      }
      else {
        GST_DEBUG_OBJECT (mux, "*** MUX -- > input caps with feature memory:NVMM; pad_id=%d\n", pad_id);
        mux->muxCtx->SetMemTypeNVMM(pad_id, false);
      }
      GST_DEBUG_OBJECT (mux, "*** MUX -- > input caps; feature memory:NVMM %d; pad_id=%d "
                "caps=[%s] mimetype=[%s]\n",
                mux->muxCtx->IsMemTypeNVMM(pad_id),
                pad_id,
                gst_caps_to_string(caps), mimetype);

      {
        GstQuery *query = gst_nvquery_sourceid_new();
        guint sourceid;
        if (gst_pad_peer_query(pad, query) &&
            gst_nvquery_sourceid_parse(query, &sourceid)) {
          mux->helper->get_pad(pad_id)->source_id = sourceid;
        }
        gst_query_unref(query);
      }

      /** If mimetype is audio/x-raw, extract useful information and return */
      if(strcmp(mimetype, "audio/x-raw") == 0) {

        /** update 0th stream ID to audio; we support only audio or only video mux */
        mux->helper->update_pad_mimetype(pad_id, PAD_MIME_TYPE_AUDIO);
        mux->isAudio = true;
        /** Fetch audio params */
        GstNvBufAudioCaps audioCapsToParams(caps);
        NvBufAudioParams audioParams;
        bool ok = audioCapsToParams.GetAudioParams(audioParams);
        if(!ok) {

            g_mutex_unlock (&mux->ctx_lock);
            return false;
        }
        uint32_t source_id = mux->helper->get_pad(pad_id)->source_id;
        mux->muxCtx->SaveAudioParams(pad_id, source_id, audioParams);
        if (!gst_pad_has_current_caps (mux->srcpad)) {
          if (caps) {
            GstEvent *event;
            GstStructure *str;
            GstCapsFeatures *features =
              gst_caps_features_from_string ("memory:NVMM");
            caps = gst_caps_copy (caps);

            gst_caps_set_features (caps, 0, features);

            str = gst_caps_get_structure (caps, 0);
            if(gst_structure_has_field (str, "framerate"))
              gst_structure_remove_field (str, "framerate");
            if (gst_structure_has_field (str, "width"))
              gst_structure_remove_field (str, "width");
            if (gst_structure_has_field (str, "height"))
              gst_structure_remove_field (str, "height");


            LOGD("caps to srcpad is [%s]\n", gst_caps_to_string(caps));
            event = gst_event_new_caps (caps);
            gboolean res = gst_pad_push_event (mux->srcpad, event);
            if(!res) {
              LOGD("failed to set caps [%s] on source pad\n", gst_caps_to_string(caps));
              GST_ERROR_OBJECT (element, "failed to set caps [%s] on source pad\n",
                gst_caps_to_string(caps));
              return FALSE;
            }
          }
          if(!mux->pad_task_created) {
            mux->pad_task_created =
            gst_pad_start_task (mux->srcpad,
                gst_nvstreammux_src_push_loop, mux, NULL);
          }
        }

        /** Send event_caps after the caps event is sent out of src pad
         * Note: This is required to properly order caps event and query
         * handling in nvstreamdemux
         */
        new_event = gst_nvevent_new_update_caps (pad_id, 0, 0, caps_str, gst_pad_get_stream_id(pad), 1);
        event_ret = gst_pad_push_event(mux->srcpad, new_event);
        if(!event_ret) {
           LOGD("failed to set updated caps [%s] on source pad\n", gst_caps_to_string(caps));
           GST_ERROR_OBJECT (element, "failed to set caps [%s] on source pad\n",
              gst_caps_to_string(caps));
           return FALSE;
        }

        g_mutex_unlock (&mux->ctx_lock);
        return TRUE;
      }

      LOGD("DEBUGME\n");
      GstCapsFeatures* featuresInputCaps = gst_caps_get_features(caps, 0);
      if(!featuresInputCaps || !gst_caps_features_contains(featuresInputCaps, "memory:NVMM"))
      {
        GST_ERROR_OBJECT (element,
            "Feature missing; input caps; memory:NVMM");

        g_mutex_unlock (&mux->ctx_lock);
        return FALSE;
      }

      if (!gst_pad_has_current_caps (mux->srcpad)) {
        if (caps) {
          GstEvent *event;
          GstStructure *str;
          GstCapsFeatures *features =
              gst_caps_features_from_string ("memory:NVMM");

          caps = gst_caps_copy (caps);
          str = gst_caps_get_structure (caps, 0);
          gst_caps_set_features (caps, 0, features);

          n = gst_caps_get_size (caps);
          for (i=0; i<n; i++)
          {
              if (gst_structure_has_field (str, "width"))
              {
                 wd = (GValue *)gst_structure_get_value(str, "width");
                 width_val = g_value_get_int(wd);
              }
              if (gst_structure_has_field (str, "height"))
              {
                 ht = (GValue *)gst_structure_get_value(str, "height");
                 height_val = g_value_get_int(ht);
              }
          }
	  
          if (gst_structure_has_field (str, "pixel-aspect-ratio"))
            gst_structure_remove_field (str, "pixel-aspect-ratio");
          if (gst_structure_has_field (str, "chroma-site"))
            gst_structure_remove_field (str, "chroma-site");
          if (gst_structure_has_field (str, "interlace-mode"))
            gst_structure_remove_field (str, "interlace-mode");
          if (gst_structure_has_field (str, "colorimetry"))
            gst_structure_remove_field (str, "colorimetry");

          //gst_video_info_init (&pad_data->in_videoinfo);
          //gst_video_info_from_caps (&pad_data->in_videoinfo, caps);

          gst_structure_set (str,
                "batch-size", G_TYPE_INT, mux->helper->get_config_batch_size(),
                "num-surfaces-per-frame", G_TYPE_INT, mux->helper->get_num_surfaces_per_frame(),
                NULL);

          gst_video_info_init (&mux->out_videoinfo);
          gst_video_info_from_caps (&mux->out_videoinfo, caps);

          if (mux->out_videoinfo.fps_d == 0)
            mux->out_videoinfo.fps_d = 1;
          if (mux->out_videoinfo.fps_n == 0)
            mux->out_videoinfo.fps_n = 30;

          //mux->frame_duration_nsec =
           //   1000000000UL * mux->out_videoinfo.fps_d /
            //  mux->out_videoinfo.fps_n;

          mux->helper->set_frame_duration(1000000000UL * ((double)mux->out_videoinfo.fps_d /(double) mux->out_videoinfo.fps_n));
          event = gst_event_new_caps (caps);

          LOGD("caps to srcpad is [%s]\n", gst_caps_to_string(caps));
          gst_pad_push_event (mux->srcpad, event);

  if (mux->helper->get_num_surfaces_per_frame() == 0) {
    GstQuery *query = gst_nvquery_num_surfaces_per_buffer_new ();

    uint32_t num_surfaces_per_frame = 1;
    if (!gst_pad_peer_query (GST_PAD (GST_ELEMENT(mux)->sinkpads->data), query) ||
        !gst_nvquery_num_surfaces_per_buffer_parse (query, &num_surfaces_per_frame))
    {
      GST_DEBUG_OBJECT (mux, "*** MUX -- > num_surface_per_frame query failed... Using num_surface_per_frame = 1\n");
      mux->helper->set_num_surfaces_per_frame(1);
    }
    else
    {
        GST_DEBUG_OBJECT (mux, "MUX -- > num_surface_per_frame after query is %d\n", num_surfaces_per_frame);
        mux->helper->set_num_surfaces_per_frame(num_surfaces_per_frame);
    }
  }

  if (mux->helper->get_config_batch_size() % mux->helper->get_num_surfaces_per_frame() != 0) {
    //GST_ELEMENT_ERROR (mux, LIBRARY, SETTINGS,
     //   ("Muxer batch-size (%d) not a multiple of number of dewarped surfaces (%d)",
      //  mux->helper->get_batch_size(), mux->helper->get_num_surfaces_per_frame()), NULL);
    GST_ERROR_OBJECT (mux, "returning incorrect batch_size\n");

    g_mutex_unlock (&mux->ctx_lock);
    return FALSE;
  }
          if(!mux->pad_task_created) {
            mux->pad_task_created =
            gst_pad_start_task (mux->srcpad,
                gst_nvstreammux_src_push_loop, mux, NULL);
          }
          //gst_caps_unref (caps);
        }
      } else {
        GstCaps *src_caps = gst_pad_get_current_caps (mux->srcpad);
        GstCaps *new_caps = gst_caps_copy (caps);
        GstStructure *str = gst_caps_get_structure (new_caps, 0);
        GstCapsFeatures *features =
            gst_caps_features_from_string ("memory:NVMM");

        n = gst_caps_get_size (caps);
        for (i=0; i<n; i++)
        {
           if (gst_structure_has_field (str, "width"))
           {
               wd = (GValue *)gst_structure_get_value(str, "width");
               width_val = g_value_get_int(wd);
           }
           if (gst_structure_has_field (str, "height"))
           {
               ht = (GValue *)gst_structure_get_value(str, "height");
               height_val = g_value_get_int(ht);
           }
        }
	
        gst_caps_set_features (new_caps, 0, features);
        if (gst_structure_has_field (str, "framerate"))
          gst_structure_remove_field (str, "framerate");

        if (mux->width && mux->height) {
          if (gst_structure_has_field (str, "width"))
            gst_structure_remove_field (str, "width");
          if (gst_structure_has_field (str, "height"))
            gst_structure_remove_field (str, "height");
          if (gst_structure_has_field (str, "pixel-aspect-ratio"))
            gst_structure_remove_field (str, "pixel-aspect-ratio");
          if (gst_structure_has_field (str, "chroma-site"))
            gst_structure_remove_field (str, "chroma-site");
          if (gst_structure_has_field (str, "interlace-mode"))
            gst_structure_remove_field (str, "interlace-mode");
          if (gst_structure_has_field (str, "colorimetry"))
            gst_structure_remove_field (str, "colorimetry");
        }

#if 0
        if (!gst_caps_can_intersect (src_caps, new_caps)) {
          g_mutex_unlock (&mux->ctx_lock);
          return FALSE;
        }
#endif
        gst_caps_unref (src_caps);
        gst_caps_unref (new_caps);
      }

      new_event = gst_nvevent_new_update_caps (pad_id, width_val, height_val, caps_str, gst_pad_get_stream_id(pad), 0);
      event_ret = gst_pad_push_event (mux->srcpad, new_event);
      if(!event_ret) {
         LOGD("failed to set updated caps [%s] on source pad\n", gst_caps_to_string(caps));
         GST_ERROR_OBJECT (element, "failed to set caps [%s] on source pad\n",
            gst_caps_to_string(caps));
         return FALSE;
      }

      g_mutex_unlock (&mux->ctx_lock);
      return TRUE;

}

static void
configure_module(GstNvStreamMux * mux)
{
  BatchPolicyConfig cfg;
  MuxConfigParser parsed_config;

  bool parsed = false;
  if(mux->config_file_available && mux->config_file_path){
      parsed = parsed_config.SetConfigFile(mux->config_file_path);
  }
  if(parsed)
  {
     /** If config file does not explicitly configure batch-size,
      * the non-zero value caller set before invoking ParseConfigs()
      * take precedence over the Parser's default batch-size config.
      */
     cfg.batch_size = mux->batch_size;
     parsed_config.ParseConfigs(&cfg);
     mux->helper->set_policy(cfg);
  }
  else
  {
     parsed_config.ParseConfigs(&cfg, true, mux->num_sink_pads);
     mux->helper->set_policy(cfg);
     /** override default with plugin props */
     mux->helper->set_batch_size(mux->batch_size);
     mux->helper->set_num_surfaces_per_frame(mux->num_surfaces_per_frame);
     GST_WARNING_OBJECT(mux, "No config-file provided; falling back to default streammux config %d\n", mux->num_sink_pads);
  }
  mux->module_initialized = true;
  mux->config_file_available = FALSE;
}

static gboolean
gst_nvstreammux_sink_event (GstPad * pad, GstObject * parent, GstEvent * event)
{
    GstNvStreamMux *mux = GST_NVSTREAMMUX (parent);
    GstElement *element = GST_ELEMENT (parent);
    guint stream_index = 0;
    gchar * name = gst_pad_get_name(pad);
    gboolean ret = TRUE;
    GstSinkPad* sinkPad;
    GstNvDsNtpCalculator* ntp_calc;
    const GstSegment *segment = NULL;
    GstQuery *peer_query;
    gboolean result = FALSE;
    GstCaps* ret_caps = NULL;
    GstCaps* peercaps = NULL;
    if( name == NULL )
    {
        GST_DEBUG_OBJECT(mux, "Streammux sink event name null\n");
    }
    else if (sscanf (name, "sink_%u", &stream_index) < 1)
    {
        GST_DEBUG_OBJECT(mux, "Streammux sink event name invalid\n");
        g_free(name);
    }
    else
    {
        LOGD("event at sink pad(%s) [%s]\n", name, GST_EVENT_TYPE_NAME(event));
        g_free(name);

        GstEvent * ev = gst_event_copy(event);
        EventWrapper * entry;
        switch (GST_EVENT_TYPE (event))
        {
             case GST_EVENT_CAPS:
                 GST_OBJECT_LOCK(mux);
                 if(!mux->module_initialized)
                 {
                     configure_module(mux);
                 }
                 GST_OBJECT_UNLOCK(mux);
                 LOGD("DEBUGME\n");
                 ret = handle_caps( stream_index, pad, parent, ev);
                 LOGD("DEBUGME ret=%d\n", ret);
                 gst_event_unref (ev);
                 /** new caps negotiation on the pad; invalidate
                  * the record on sending EOS downstream */
                 mux->eos_sent = FALSE;
                 mux->helper->set_all_pads_eos(false);
                 sinkPad = (GstSinkPad*) mux->helper->get_pad(stream_index);
                 if(sinkPad && sinkPad->get_eos())
                 {
                     /** If this Pad had already received EOS, reset! */
                     LOGD("reset_pad\n");
                     mux->helper->reset_pad((SinkPad*)sinkPad);
                 }
                 return TRUE;
             case GST_EVENT_EOS:
                 GST_DEBUG_OBJECT (mux, "Received EOS for stream %d [%s]\n", stream_index, mux->isAudio ? "audiomux" : "videomux");
                 g_print ("nvstreammux: Successfully handled EOS for source_id=%d\n", stream_index);
                 LOGD ("Received EOS for stream %d [%s]\n", stream_index, mux->isAudio ? "audiomux" : "videomux");
                 entry = new EventWrapper((void *)ev, ENTRY_EVENT, BATCH_SEQUENCE_POST_BATCH);
                 ret = mux->helper->handle_eos(SINK_EVENT_EOS, stream_index, entry);
                 sinkPad = (GstSinkPad*) mux->helper->get_pad(stream_index);
                 sinkPad->set_eos(true);
                 ntp_calc = sinkPad->get_ntp_calc(mux->ntp_calc_mode,
                         mux->frame_duration);
                 if(ntp_calc) {
                   gst_nvds_ntp_calculator_reset (ntp_calc);
                 }
                 return TRUE;
             case GST_EVENT_SEGMENT:
                 LOGD("Got SEGMENT\n");

                 gst_event_parse_segment (event, &segment);
                 mux->synch_buffer->SetSegment(stream_index, segment);
                 mux->synch_buffer->SetOperatingMinFpsDuration(mux->helper->get_min_fps_duration());

                 entry = new EventWrapper((void *)ev, ENTRY_EVENT, BATCH_SEQUENCE_IN_BATCH);
                 ret = mux->helper->handle_segment(SINK_EVENT_SEGMENT, stream_index, entry);
                 mux->helper->push_events();
                 return TRUE;
             case GST_EVENT_FLUSH_START:
                 gst_event_unref (ev);
                 return TRUE;
             case GST_EVENT_FLUSH_STOP:
                 entry = new EventWrapper((void *)ev, ENTRY_EVENT, BATCH_SEQUENCE_POST_BATCH);
                 ret = mux->helper->handle_flush_stop(SINK_EVENT_FLUSH_STOP, stream_index, entry);
                 mux->eos_sent = FALSE;
                 gst_event_unref (ev);
                 return TRUE;
             case GST_EVENT_STREAM_START:
                 /** A new stream added; mark eos_sent as FALSE;
                  * We could have marked EOS (all streams) before */
                 mux->eos_sent = FALSE;
                 LOGD("Got STREAM_START %d\n", stream_index);
                 entry = new EventWrapper((void *)ev, ENTRY_EVENT, BATCH_SEQUENCE_IN_BATCH);
                 ret = mux->helper->handle_segment(SINK_EVENT_STREAM_START, stream_index, entry);
                 mux->helper->push_events();
                 if (mux->pushed_stream_start_once == false)
                 {
                     mux->pushed_stream_start_once = true;
                     ret = gst_pad_push_event (mux->srcpad, event);
                     return ret;
                 }
                 else
                     return TRUE;
                 return TRUE;
             default:
                 break;
        }
        if ((GstNvEventType)GST_EVENT_TYPE(event) == GST_NVEVENT_STREAM_EOS)
        {
                GST_DEBUG_OBJECT (mux, "Received NVEVENT_EOS for stream %d [%s]\n", stream_index, mux->isAudio ? "audiomux" : "videomux");
                entry = new EventWrapper((void *)ev, ENTRY_EVENT, BATCH_SEQUENCE_POST_BATCH);
                ret = mux->helper->handle_eos_cascaded(SINK_EVENT_EOS, stream_index, entry);
                return TRUE;
        }

        switch ((GstNvEventType)GST_EVENT_TYPE (event)) {
         case GST_NVEVENT_STREAM_RESET:
         {
           entry = new EventWrapper((void *)ev, ENTRY_EVENT, BATCH_SEQUENCE_IN_BATCH);
           ret = mux->helper->handle_stream_reset(SINK_EVENT_STREAM_RESET, stream_index, entry);
           mux->helper->push_events();
           return TRUE;
         }
         break;
         default:
           break;
        }
    }
    LOGD("event pushed to src pad [%s]\n", GST_EVENT_TYPE_NAME(event));
    ret = gst_pad_push_event (mux->srcpad, event);
    return ret;
}


static GstPad *
gst_nvstreammux_request_new_pad (GstElement * element, GstPadTemplate * templ,
    const gchar * name, const GstCaps * caps)
{
    GstNvStreamMux *mux = GST_NVSTREAMMUX (element);
    GstPad *sinkpad = NULL;
    guint stream_index = 0;
    guint i, n;
    GList *iter;


    LOGD("DEBUGME\n");

    GST_DEBUG_OBJECT (element, "Requesting new sink pad");

    if (!name || sscanf (name, "sink_%u", &stream_index) < 1)
    {
        GST_ERROR_OBJECT (element,
            "Pad should be named 'sink_%%u' when requesting a pad");
        return NULL;
    }


    g_mutex_lock (&mux->ctx_lock);
    SinkPad * wrapper_pad;
    if((wrapper_pad = (mux->helper->get_pad(stream_index))) != NULL)
    {
        sinkpad = GST_PAD_CAST (wrapper_pad->wrapped);
    }
    else
    {
        sinkpad = GST_PAD_CAST (g_object_new (GST_TYPE_NVSTREAM_PAD,
              "name", name, "direction", templ->direction, "template", templ,
              NULL));
        /** TODO remove duplicate GstSinkPad creation */
        wrapper_pad = new GstSinkPad(mux, stream_index, sinkpad);
        LOGD("DEBUGME\n");
        gst_pad_set_chain_function (sinkpad,
            GST_DEBUG_FUNCPTR (gst_nvstreammux_chain));

        gst_pad_set_event_function (sinkpad,
            GST_DEBUG_FUNCPTR (gst_nvstreammux_sink_event));

        gst_pad_set_query_function (sinkpad,
            GST_DEBUG_FUNCPTR (gst_nvstreammux_sink_query));

        gst_pad_set_active (sinkpad, TRUE);

        gst_element_add_pad (element, sinkpad);
        mux->helper->add_pad(stream_index, new GstSinkPad(mux, stream_index, sinkpad));
        delete(wrapper_pad);
    }
    wrapper_pad = NULL;
    mux->helper->notify_all();
    mux->helper->set_all_pads_eos(false);

    mux->num_sink_pads++;

    g_mutex_unlock (&mux->ctx_lock);
    return sinkpad;
}


static void
gst_nvstreammux_release_pad (GstElement * element, GstPad * pad)
{
    GstNvStreamMux *mux = GST_NVSTREAMMUX (element);
    guint i, n;
    GList *iter;
    GstEvent *pad_removed = NULL;
    SinkPad * wrapper_pad;
    guint stream_index = 0;
    gchar *name;

    if (pad != NULL)
    {
      name = gst_pad_get_name(pad);
      if (name !=NULL)
      {
        if (sscanf (name, "sink_%u", &stream_index) < 1)
        {
          g_free(name);
        }
        else if((wrapper_pad = mux->helper->get_pad(stream_index)) != NULL)
        {
          g_free(name);
          //wrapper_pad->wait_till_empty();
          //mux->helper>lock();
          gst_pad_set_active (GST_PAD_CAST (wrapper_pad->wrapped), FALSE);
          pad_removed = gst_nvevent_new_pad_deleted (wrapper_pad->id);
          GST_DEBUG_OBJECT (mux, "Pad deleted %d\n", wrapper_pad->id);
          gst_pad_push_event (mux->srcpad, pad_removed);
          mux->helper->remove_pad((wrapper_pad->id));
          //mux->helper->notify_all();
        }
      }
      //This lock is not needed as gst_element_remove_pad is MT SAFE.
      //Has been added to fix Maxine multithreaded and dynamic pad add/remove usecase issue.
      g_mutex_lock (&mux->ctx_lock);
      gst_element_remove_pad(GST_ELEMENT(mux), pad);
      g_mutex_unlock (&mux->ctx_lock);
    }
}

static void
gst_nvstreammux_src_push_loop (gpointer user_data)
{
    static struct timeval t1, t2;
    GstNvStreamMux *mux = GST_NVSTREAMMUX (user_data);
    guint i;
    gint64 end_time = -1;
    gboolean timedout = FALSE;
    gboolean send_eos = FALSE;
    gboolean all_pads_eos = FALSE;
    GstFlowReturn ret;
    gpointer dest_data;
    NvBufSurface *surf;
    NvDsMeta *meta = NULL;
    NvDsBatchMeta *batch_meta = NULL;

    GList *iter = NULL;

    iter = GST_ELEMENT (mux)->sinkpads;

    LOGD("DEBUGME\n");
    if (iter == NULL || mux->eos_sent)
    {
        if (mux->no_pipeline_eos == FALSE){
            send_eos = mux->helper->get_all_pads_eos();
        }
        if (send_eos && !mux->eos_sent)
        {
             GST_DEBUG_OBJECT (mux, "Sending EOS downstream [%s:%d]\n", mux->isAudio ? "audiomux" : "videomux", __LINE__);
             mux->eos_sent = gst_pad_push_event (mux->srcpad, gst_event_new_eos ());
             /** If new streams are added after we push out EOS, we need to send STREAM_START again */
             mux->pushed_stream_start_once = false;
        }
        GST_DEBUG_OBJECT (mux, "No Sources found at the input of muxer [%s:%d]", mux->isAudio ? "audiomux" : "videomux", __LINE__ );
        /** to avoid tight loop in this thread, sleep */
        usleep(5 * 1000);
        return;
    }

    if (!mux->segment_sent) {
      gst_pad_push_event (mux->srcpad, gst_event_new_segment (&mux->segment));
      mux->segment_sent = TRUE;
    }

    gettimeofday (&t1, NULL);

    LOGD("mimetype=%d \n", mux->helper->get_pad_mimetype(0));
    if(mux->helper->get_batch_size() == 0)
    {
        /** Make sure we push any pending events */
        LOGD("batch-size=0\n");
        mux->helper->push_loop(nullptr, nullptr);

        if (mux->no_pipeline_eos == FALSE){
            send_eos = mux->helper->get_all_pads_eos();
            LOGD("send_eos=%d\n", send_eos);
        }
        if (send_eos && !mux->eos_sent)
        {
          /** Sending EOS; shall not print an error here as its expected the pad_push
           * might fail and we tight loop here when there are no more streams
           * and we are tearing down the pipeline */
          mux->eos_sent = gst_pad_push_event (mux->srcpad, gst_event_new_eos ());
          /** If new streams are added after we push out EOS, we need to send STREAM_START again */
          mux->pushed_stream_start_once = false;
          LOGD("DEBUGME eos_sent=%d\n", mux->eos_sent);
        }
        usleep(5 * 1000);
        return;
    }

    bool batch_formed = false;

    if(mux->sync_inputs) {
      GstClockTime latency_up = gst_nvstreammux_get_latency_unlocked (mux);
      LOGD("latency_up=%lu has_peer_latency=%d peer_latency_min=%lu max=%lu\n",
        latency_up, mux->has_peer_latency, mux->peer_latency_min, mux->peer_latency_max);
      if(latency_up != GST_CLOCK_TIME_NONE) {
        mux->synch_buffer->SetUpstreamLatency(mux->max_latency + latency_up);
      }
    }

    //printf("bs=%d c_bs=%d\n", mux->helper->get_batch_size(), mux->helper->get_config_batch_size());
    if(mux->isAudio) {
      GstAudioBatchBufferWrapper * out_buf = new GstAudioBatchBufferWrapper(mux, mux->helper->get_batch_size(), false);
      batch_meta = nvds_create_audio_batch_meta(mux->helper->get_config_batch_size());
      if(batch_meta == NULL) {
        g_print("line = %d file = %s func = %s\n",__LINE__,__FILE__,__func__);
        exit(-1);
      }
      meta = gst_buffer_add_nvds_meta ((GstBuffer *) (out_buf->gst_buffer), batch_meta, NULL,
        nvds_audio_batch_meta_copy_func, nvds_audio_batch_meta_release_func);
      if(meta == NULL)
      {
          g_print("line = %d file = %s func = %s\n",__LINE__,__FILE__,__func__);
          exit(-1);
      }

      meta->meta_type = NVDS_BATCH_GST_META;

      batch_meta->base_meta.batch_meta = batch_meta;
      batch_meta->base_meta.copy_func = nvds_audio_batch_meta_copy_func;
      batch_meta->base_meta.release_func = nvds_audio_batch_meta_release_func;
      batch_meta->max_frames_in_batch = mux->helper->get_config_batch_size();

      LOGD("DEBUGME\n");
      batch_formed = mux->helper->push_loop(out_buf, batch_meta);
      LOGD("DEBUGME\n");
    }
    else {
      GstBatchBufferWrapper * out_buf = new GstBatchBufferWrapper(mux, mux->helper->get_batch_size(), false);
      batch_meta = nvds_create_batch_meta(mux->helper->get_config_batch_size());
      if(batch_meta == NULL)
      {
          g_print("line = %d file = %s func = %s\n",__LINE__,__FILE__,__func__);
          exit(-1);
      }
      meta = gst_buffer_add_nvds_meta ((GstBuffer *) (out_buf->gst_buffer), batch_meta, NULL,
        nvds_batch_meta_copy_func, nvds_batch_meta_release_func);
      if(meta == NULL)
      {
          g_print("line = %d file = %s func = %s\n",__LINE__,__FILE__,__func__);
          exit(-1);
      }

      meta->meta_type = NVDS_BATCH_GST_META;

      batch_meta->base_meta.batch_meta = batch_meta;
      batch_meta->base_meta.copy_func = nvds_batch_meta_copy_func;
      batch_meta->base_meta.release_func = nvds_batch_meta_release_func;
      //batch_meta->base_meta.copy_func = batch_meta_copy_func;
      //batch_meta->base_meta.release_func = batch_meta_release_func;
      batch_meta->max_frames_in_batch = mux->helper->get_config_batch_size();
      //batch_meta->unique_id = 0; /*TODO*/



      LOGD("out_buf=%p batch_meta=%p name=%s %p\n", out_buf, batch_meta, GST_ELEMENT_NAME(mux),
                batch_meta->meta_mutex.p);
      /** TODO: Add helper->pre_loop() to check if we have enough buffers to form batch
       * and then only create out_buf */
      batch_formed = mux->helper->push_loop(out_buf, batch_meta);
      LOGD("DEBUGME\n");
    }
    (void)batch_formed;

    if (mux->no_pipeline_eos == FALSE){
        send_eos = mux->helper->get_all_pads_eos();
        LOGD("send_eos=%d\n", send_eos);
    }
    if (send_eos && !mux->eos_sent)
    {
        /** Sending EOS */
        GST_DEBUG_OBJECT (mux, "Sending EOS downstream [%s:%d]\n", mux->isAudio ? "audiomux" : "videomux", __LINE__);
        mux->eos_sent = gst_pad_push_event (mux->srcpad, gst_event_new_eos ());
        /** If new streams are added after we push out EOS, we need to send STREAM_START again */
        mux->pushed_stream_start_once = false;
        LOGD("DEBUGME eos_sent=%d\n", mux->eos_sent);
    }

    if(mux->config_file_available) {
        GST_DEBUG_OBJECT(mux, "configuring nvstreammux with new config-file-path\n");
        configure_module(mux);
    }

}
static GstStateChangeReturn
gst_nvstreammux_change_state (GstElement * element, GstStateChange transition)
{
  GstNvStreamMux *mux = GST_NVSTREAMMUX (element);
  GstStateChangeReturn ret;
  GList *iter = element->sinkpads;
  guint i = 0, n = 0;

  switch (transition) {
    case GST_STATE_CHANGE_NULL_TO_READY:
      if(mux->batch_size <= 0)
      {
        GST_ELEMENT_ERROR (mux, LIBRARY, SETTINGS,
            ("Batch size not set"),
            (nullptr));
        return GST_STATE_CHANGE_FAILURE;
      }
      break;
    case GST_STATE_CHANGE_READY_TO_NULL:
      break;
    case GST_STATE_CHANGE_READY_TO_PAUSED:
      mux->helper->handle_ready_pause();

      break;
    case GST_STATE_CHANGE_PAUSED_TO_PLAYING:
      mux->helper->handle_pause_play();
      break;
    case GST_STATE_CHANGE_PLAYING_TO_PAUSED:
      mux->helper->handle_play_pause();
      break;
    case GST_STATE_CHANGE_PAUSED_TO_READY:
    {
        GHashTableIter iter;
        gpointer value;
        gint key = 0;
        g_mutex_lock (&mux->ctx_lock);
        g_hash_table_iter_init(&iter, mux->sink_pad_caps);
        while (g_hash_table_iter_next(&iter, (gpointer*)&key, &value)){
          SinkPad * wrapper_pad;
          GstPad *sinkpad = NULL;
          if((wrapper_pad = (mux->helper->get_pad(key))) != NULL){
             sinkpad = GST_PAD_CAST (wrapper_pad->wrapped);
             gst_pad_set_active(sinkpad, FALSE);
          }
        }
        g_mutex_unlock (&mux->ctx_lock);
        mux->helper->handle_stop();
        if (mux->pad_task_created)
        {
          gst_pad_stop_task (mux->srcpad);
        }
        mux->helper->reset_stop();
        mux->helper->set_all_pads_eos(false);
        mux->all_pads_eos = FALSE;
        mux->pad_task_created = FALSE;
        mux->eos_sent = FALSE;
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

    case PROP_BATCH_SIZE:
      mux->batch_size = g_value_get_uint (value);
      mux->helper->set_batch_size(mux->batch_size);
      break;

    case PROP_BATCHED_PUSH_TIMEOUT:
      mux->timeout_usec = g_value_get_int (value);
      mux->helper->set_batch_push_timeout(mux->timeout_usec);
      break;

    case PROP_CONFIG_FILE_PATH:
      if(mux->config_file_path) {
        g_free (mux->config_file_path);
      }
      mux->config_file_path = g_value_dup_string (value);
      mux->config_file_available = TRUE;
      break;

    case PROP_NUM_SURFACES_PER_FRAME:
      mux->num_surfaces_per_frame = g_value_get_uint (value);
      mux->helper->set_num_surfaces_per_frame(mux->num_surfaces_per_frame);
      break;

    case PROP_ATTACH_SYS_TIME_STAMP:
      mux->sys_ts = g_value_get_boolean(value);
      mux->ntp_calc_mode = mux->sys_ts
                        ? GST_NVDS_NTP_CALC_MODE_SYSTEM_TIME :
                        GST_NVDS_NTP_CALC_MODE_RTCP;
      break;

    case PROP_MAX_LATENCY:
      mux->max_latency = g_value_get_uint64 (value);
      mux->synch_buffer->SetUpstreamLatency(mux->max_latency);
      break;

    case PROP_SYNC_INPUTS:
      mux->sync_inputs = g_value_get_boolean(value);
      if(mux->sync_inputs) {
        mux->helper->set_synch_buffer_iface(mux->synch_buffer);
      }
      else {
        mux->helper->set_synch_buffer_iface(NULL);
      }
      break;
    case PROP_FRAME_NUM_RESET_ON_EOS:
      mux->frame_num_reset_on_eos = g_value_get_boolean (value);
      mux->helper->set_frame_num_reset_on_eos(mux->frame_num_reset_on_eos);
      break;
    case PROP_FRAME_NUM_RESET_ON_STREAM_RESET:
      mux->frame_num_reset_on_stream_reset = g_value_get_boolean (value);
      mux->helper->set_frame_num_reset_on_stream_reset(mux->frame_num_reset_on_stream_reset);
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
    case PROP_NO_PIPELINE_EOS:
      mux->no_pipeline_eos = g_value_get_boolean(value);
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

    case PROP_BATCH_SIZE:
      g_value_set_uint (value, mux->helper->get_config_batch_size());
      break;
    case PROP_BATCHED_PUSH_TIMEOUT:
      g_value_set_int (value, mux->timeout_usec);
      break;
    case PROP_CONFIG_FILE_PATH:
      g_value_set_string (value, mux->config_file_path);
      break;
    case PROP_NUM_SURFACES_PER_FRAME:
      g_value_set_uint (value, mux->helper->get_num_surfaces_per_frame());
      break;
   case PROP_ATTACH_SYS_TIME_STAMP:
      g_value_set_boolean (value, mux->sys_ts);
      break;
   case PROP_MAX_LATENCY:
      g_value_set_uint64 (value, mux->max_latency);
      break;
    case PROP_SYNC_INPUTS:
      g_value_set_boolean (value, mux->sync_inputs);
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
    case PROP_NO_PIPELINE_EOS:
      g_value_set_boolean (value, mux->no_pipeline_eos);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static void
gst_nvstreammux_2_class_init (GstNvStreamMuxClass * klass)
{
  GObjectClass *gobject_class = G_OBJECT_CLASS (klass);
  GstElementClass *gstelement_class = GST_ELEMENT_CLASS (klass);

  // Indicate the use of DS buf api version
  g_setenv ("DS_NEW_BUFAPI", "1", TRUE);

  gst_element_class_set_static_metadata (gstelement_class,
      "Stream multiplexer 2", "Generic", "N-to-1 pipe stream multiplexing",
      "NVIDIA Corporation. Post on Deepstream for Tesla forum for any queries "
      "@ https://devtalk.nvidia.com/default/board/209/");

  gobject_class->set_property = gst_nvstreammux_set_property;
  gobject_class->get_property = gst_nvstreammux_get_property;


  g_object_class_install_property (gobject_class, PROP_BATCH_SIZE,
      g_param_spec_uint ("batch-size", "Batch Size",
          "Maximum number of buffers in a batch",
          0, MAX_NVBUFFERS, DEFAULT_BATCH_SIZE,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_BATCHED_PUSH_TIMEOUT,
      g_param_spec_int ("batched-push-timeout", "Batched Push Timeout",
          "Timeout in microseconds to wait for batch formation i.e minimum FPS\n"
          "\t\t\tto push the batch even if the complete batch is not formed.\n"
          "\t\t\tSet to 33ms (33fps) by default",
          1, G_MAXINT, DEFAULT_BATCHED_PUSH_TIMEOUT,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_NUM_SURFACES_PER_FRAME,
      g_param_spec_uint ("num-surfaces-per-frame", "Num Surfaces Per Frame",
          "Number of Surfaces in a frame",
          0, MAX_SURFACES, DEFAULT_NUM_SURFACES,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_CONFIG_FILE_PATH,
      g_param_spec_string ("config-file-path", "Config File Path",
          "Path to the configuration file for this instance of nvmultistream",
          "NULL",
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_ATTACH_SYS_TIME_STAMP,
      g_param_spec_boolean ("attach-sys-ts", "Set system timestamp as ntp timestamp",
          "If set to TRUE, system timestamp will be attached as ntp timestamp.\n"
          "\t\t\tIf set to FALSE, ntp timestamp from rtspsrc, if available, will be attached.",
          DEFAULT_ATTACH_SYS_TIME_STAMP,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_MAX_LATENCY,
      g_param_spec_uint64 ("max-latency", "maximum lantency",
          "Additional latency in live mode to allow upstream to take longer to produce buffers for the current position (in nanoseconds)",
          0, G_MAXUINT64, 0,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_SYNC_INPUTS,
      g_param_spec_boolean ("sync-inputs", "Synchronize Inputs",
          "Boolean property to force sychronization of input frames.",
          FALSE,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_FRAME_NUM_RESET_ON_EOS,
      g_param_spec_boolean ("frame-num-reset-on-eos", "Frame Number Reset on EOS",
          "Reset frame numbers to 0 for a source from which EOS is received. (For debugging purpose only)",
          FALSE, (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_FRAME_NUM_RESET_ON_STREAM_RESET,
      g_param_spec_boolean ("frame-num-reset-on-stream-reset", "Frame Number Reset on stream reset",
          "Reset frame numbers to 0 for a source which needs to be reset. (For debugging purpose only)\n"
          "Needs to be paired with tracking-id-reset-mode=1 in the tracker config.",
          FALSE, (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_FRAME_DURATION,
      g_param_spec_uint64 ("frame-duration", "Frame duration",
          "Duration of input frames in milliseconds for use in NTP timestamp correction based on frame rate.\n"
          "\t\t\tIf set to 0, frame duration is inferred automatically from PTS values.\n"
          "\t\t\tIf set to -1, disables frame rate based NTP timestamp correction (default).",
          0, G_MAXUINT64, DEFAULT_FRAME_DURATION,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_NO_PIPELINE_EOS,
      g_param_spec_boolean ("drop-pipeline-eos", "No Pipeline EOS",
          "Boolean property so that EOS is not propagated downstream when all sink pads are at EOS. (Experimental)",
          DEFAULT_NO_PIPELINE_EOS, (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  nvstreammux_signals[SIGNAL_BUFFER_DROPPED] =
                 g_signal_new("dropped", G_TYPE_FROM_CLASS (klass), G_SIGNAL_RUN_LAST,
                 0,  NULL, NULL,
                 g_cclosure_marshal_generic, G_TYPE_NONE, 1, G_TYPE_POINTER);


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
gst_nvstreammux_2_init (GstNvStreamMux * mux)
{

  g_mutex_init (&mux->ctx_lock);
  mux->num_sink_pads = 0;
  mux->synch_buffer = new NvTimeSync((GstElement*)mux);
  mux->srcpad =
      gst_pad_new_from_static_template (&nvstreammux_srcpad_template, "src");
  gst_pad_set_query_function (mux->srcpad,
      GST_DEBUG_FUNCPTR (gst_nvstreammux_src_query));
  gst_pad_set_event_function (mux->srcpad,
    GST_DEBUG_FUNCPTR (gst_nvstreammux_src_event));
  gst_pad_use_fixed_caps (mux->srcpad);

  gboolean ret_add_pad = gst_element_add_pad (GST_ELEMENT (mux), mux->srcpad);

  gst_segment_init (&mux->segment, GST_FORMAT_TIME);
  mux->width = DEFAULT_WIDTH;
  mux->height = DEFAULT_HEIGHT;
  mux->batch_size = 0;
  mux->module_initialized = false;
  mux->config_file_available = FALSE;
  mux->config_file_path = nullptr;
  mux->pushed_stream_start_once = false;
  BatchPolicyConfig cfg;
  MuxConfigParser default_config;
  default_config.ParseConfigs(&cfg, 1);
  mux->debug_iface = new GstNvStreammuxDebug((GstElement*)mux);
  mux->helper = new NvStreamMux(new SourcePad(0,(void *)mux->srcpad), mux->debug_iface);
  mux->helper->set_policy(cfg);
  mux->query_resolution = DEFAULT_QUERY_RESOLUTION;
  mux->helper->set_num_surfaces_per_frame(1);
  //mux->last_flow_ret = GST_FLOW_OK;
  mux->segment_sent = FALSE;
  mux->num_surfaces_per_frame = 1;
  mux->muxCtx = new GstNvStreamMuxCtx();
  mux->sys_ts = DEFAULT_ATTACH_SYS_TIME_STAMP;
  mux->ntp_calc_mode = GST_NVDS_NTP_CALC_MODE_SYSTEM_TIME;
  mux->pad_task_created = FALSE;
  mux->sink_pad_caps = g_hash_table_new (NULL, NULL);
  mux->sync_inputs = FALSE;
  mux->max_latency = 0;
  mux->has_peer_latency = FALSE;
  mux->peer_latency_min = 0;
  mux->peer_latency_max = 0;
  mux->peer_latency_live = FALSE;
  mux->eos_sent = FALSE;
  mux->frame_num_reset_on_eos = FALSE;
  mux->frame_num_reset_on_stream_reset = FALSE;
  mux->frame_duration = DEFAULT_FRAME_DURATION;
  mux->no_pipeline_eos = DEFAULT_NO_PIPELINE_EOS;
  mux->cur_frame_pts = 0;
  mux->timeout_usec = DEFAULT_BATCHED_PUSH_TIMEOUT;
}
