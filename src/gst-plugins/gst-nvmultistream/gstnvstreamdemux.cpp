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

#include "gstnvstreamdemux.h"
#include "nvbufsurface.h"
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include "gst-nvquery.h"
#include "gst-nvquery-internal.h"
#include "gstnvdsmeta.h"
#include "nvdsmeta.h"
#include "nvdsmeta_internal.h"
#include "gst-nvevent.h"
#include "gstnvstreammux.h"
#include "gst-nvmessage.h"

GST_DEBUG_CATEGORY_STATIC (gst_nvstreamdemux_debug);
#define GST_CAT_DEFAULT gst_nvstreamdemux_debug

#define _do_init \
    GST_DEBUG_CATEGORY_INIT (gst_nvstreamdemux_debug, "nvstreamdemux", 0, "nvstreamdemux element");
#define gst_nvstreamdemux_parent_class parent_class
G_DEFINE_TYPE_WITH_CODE (GstNvStreamDemux, gst_nvstreamdemux,
    GST_TYPE_ELEMENT, _do_init);

#define USE_CUDA_BATCH 1
#define DEFAULT_PER_STREAM_EOS             FALSE

#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wpointer-arith"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wswitch"
#pragma GCC diagnostic ignored "-Wunused-label"
#pragma GCC diagnostic ignored "-Wenum-compare"

static GstStaticPadTemplate nvstreamdemux_sinkpad_template =
GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES ("memory:NVMM",
            "{ " "NV12, RGBA, I420 }")));

static GstStaticPadTemplate nvstreamdemux_srcpad_template =
GST_STATIC_PAD_TEMPLATE ("src_%u",
    GST_PAD_SRC,
    GST_PAD_REQUEST,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES ("memory:NVMM",
            "{ " "NV12, RGBA, I420 }")));

enum
{
  PROP_0,
  PROP_PER_STREAM_EOS
};

static GQuark dsmeta_quark = 0;

static void
gst_nvstreamdemux_finalize (GObject * object);

static void gst_nvdsstreamdemux_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_nvdsstreamdemux_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);

static void
gst_nvdsstreamdemux_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstNvStreamDemux *nvstreamdemux;

  nvstreamdemux = GST_NVSTREAMDEMUX (object);

  switch (prop_id) {
    case PROP_PER_STREAM_EOS:
      nvstreamdemux->per_stream_eos = g_value_get_boolean (value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static void
gst_nvdsstreamdemux_get_property (GObject * object, guint prop_id, GValue * value,
    GParamSpec * pspec)
{
  GstNvStreamDemux *nvstreamdemux;

  nvstreamdemux = GST_NVSTREAMDEMUX (object);

  switch (prop_id) {
    case PROP_PER_STREAM_EOS:
      g_value_set_boolean (value, nvstreamdemux->per_stream_eos);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static gboolean
gst_nvstreamdemux_src_query (GstPad * pad, GstObject * parent, GstQuery * query)
{
  if (gst_nvquery_is_batch_size (query)) {
    gst_nvquery_batch_size_set (query, 1);
    return TRUE;
  }

  return gst_pad_query_default (pad, parent, query);
}

static GstPad *
gst_nvstreamdemux_request_new_pad (GstElement * element, GstPadTemplate * templ,
    const gchar * name, const GstCaps * caps)
{
  GstNvStreamDemux *nvstreamdemux = GST_NVSTREAMDEMUX (element);
  GstPad *srcpad = NULL;
  guint stream_index = 0;

  GST_DEBUG_OBJECT (element, "Requesting new src pad");
  if (!name || sscanf (name, "src_%u", &stream_index) < 1) {
    GST_ERROR_OBJECT (element,
        "Pad should be named 'src_%%u' when requesting a pad");
    return NULL;
  }

  g_mutex_lock (&nvstreamdemux->ctx_lock);
  if (g_hash_table_contains (nvstreamdemux->pad_indexes, stream_index + (char *)NULL)) {
    GST_ERROR_OBJECT (element, "Pad named '%s' already requested", name);
    g_mutex_unlock (&nvstreamdemux->ctx_lock);   /* was leaking ctx_lock on this path */
    return NULL;
  }

  srcpad = GST_PAD_CAST (g_object_new (GST_TYPE_PAD,
          "name", name, "direction", templ->direction, "template", templ,
          NULL));
  g_mutex_unlock (&nvstreamdemux->ctx_lock);
  GST_DEBUG_OBJECT (element, "Requesting new src pad");

  gst_pad_activate_mode (srcpad, GST_PAD_MODE_PUSH, TRUE);
  gst_element_add_pad (element, srcpad);

  gst_pad_set_query_function (srcpad,
      GST_DEBUG_FUNCPTR (gst_nvstreamdemux_src_query));

  gst_pad_use_fixed_caps (srcpad);

  /* Seed stream-start on the new pad BEFORE it becomes routable (i.e. before it
   * is inserted into pad_indexes, which is what the chain function and the
   * STREAM_SEGMENT handler use to deliver buffers + per-source segments to it).
   *
   * A src pad requested AFTER the batched stream-start has already passed the
   * sink would otherwise never receive stream-start: this element forwards
   * stream-start to its src pads only via gst_pad_event_default at the instant
   * the event arrives (to the pads existing then), and never seeds it on a
   * later-requested pad. Such a pad then receives its first segment/caps with no
   * preceding stream-start -> GStreamer logs "sticky event misordering, got
   * 'segment' before 'stream-start'", the pad's sticky-event set is invalid, and
   * that source is silently starved (0 fps / blank tile). This is an intermittent
   * race on the FIRST dynamically-added source (its pad request vs. the very
   * first stream-start).
   *
   * Replay the sink's own stream-start (same stream-id) so the new pad starts in
   * the correct order. If the sink has not seen stream-start yet, the pad already
   * exists and the normal forwarding path will deliver it -> nothing to do.
   * Done before the pad_indexes insert so no buffer/segment can reach the pad
   * ahead of stream-start. */
  {
    GstEvent *ss = gst_pad_get_sticky_event (nvstreamdemux->sinkpad,
        GST_EVENT_STREAM_START, 0);
    if (ss)
      gst_pad_push_event (srcpad, ss);   /* transfer-full: consumes our ref */
  }

  g_mutex_lock (&nvstreamdemux->ctx_lock);
  g_hash_table_insert (nvstreamdemux->pad_indexes, stream_index + (char *)NULL, srcpad);
  g_mutex_unlock (&nvstreamdemux->ctx_lock);

  return srcpad;
#if 0
  GST_OBJECT_FLAG_SET (sinkpad, GST_PAD_FLAG_PROXY_CAPS);
  GST_OBJECT_FLAG_SET (sinkpad, GST_PAD_FLAG_PROXY_ALLOCATION);

  GST_DEBUG_OBJECT (element, "requested pad %s:%s",
      GST_DEBUG_PAD_NAME (sinkpad));
#endif
}

static void
gst_nvstreamdemux_release_pad (GstElement * element, GstPad * pad)
{
  GstNvStreamDemux *nvstreamdemux = GST_NVSTREAMDEMUX (element);
  gchar *name = gst_pad_get_name (pad);
  guint stream_index = 0;

  if (!name || sscanf (name, "src_%u", &stream_index) < 1) {
    return;
  }

  g_free (name);

  g_hash_table_remove (nvstreamdemux->pad_indexes, stream_index + (char *)NULL);

  gst_pad_set_active (pad, FALSE);
  gst_element_remove_pad (GST_ELEMENT_CAST (nvstreamdemux), pad);
}

typedef struct
{
  NvBufSurface surf;
  GstBuffer *src_buffer;
} GstNvStreamDemuxMemory;

static void
shared_mem_buf_unref_callback (gpointer data)
{
  GstNvStreamDemuxMemory *mem = (GstNvStreamDemuxMemory *) data;

  gst_buffer_unref (mem->src_buffer);
  g_free(mem->surf.surfaceList);
  g_slice_free (GstNvStreamDemuxMemory, mem);
}

static GstMeta *rem_meta_list[1024];
static GstBuffer *
create_shared_mem_buf (GstNvStreamDemux *demux, GstBuffer * src_buffer,
    NvDsBatchMeta *src_batch_meta,
    guint index, guint demuxIndex)
{
  GstNvStreamDemuxMemory *mem;
  GstNvStreamMemory *mux_mem = gst_buffer_get_nvstream_memory (src_buffer);
  NvBufSurface *src_surf;
  GstMapInfo info = GST_MAP_INFO_INIT;
  GstBuffer *out_buf = NULL;
  guint num_rem_meta = 0;
  guint k;

  if (!gst_buffer_map (src_buffer, &info, GST_MAP_READ)) {
    return NULL;
  }

  mem = g_slice_new0 (GstNvStreamDemuxMemory);

  if (mux_mem)
    mem->src_buffer = mux_mem->orig_buffer_ptrs[index];
  else
    mem->src_buffer = src_buffer;

  src_surf = (NvBufSurface *) info.data;
  /* src_surf and hence src_surf->surfaceList is owned by src_buffer. Hence copy
   * src_surf contents and alloc new memory for surfaceList. */
  memcpy (&mem->surf, src_surf, sizeof (NvBufSurface));
  mem->surf.numFilled = mem->surf.batchSize = demux->num_surfaces_per_frame;
  mem->surf.surfaceList = (NvBufSurfaceParams *)g_memdup2(src_surf->surfaceList + index,
      mem->surf.numFilled * sizeof(NvBufSurfaceParams));

  gst_buffer_unmap (src_buffer, &info);

  gst_buffer_ref (mem->src_buffer);

    out_buf =
        gst_buffer_new_wrapped_full (GST_MEMORY_FLAG_READONLY, &mem->surf, sizeof (NvBufSurface),
        0, sizeof (NvBufSurface), mem, shared_mem_buf_unref_callback);

  NvDsBatchMeta *batch_meta = nvds_create_batch_meta(demux->num_surfaces_per_frame);
  if(batch_meta == NULL) {
    return NULL;
  }
  NvDsMeta *meta = gst_buffer_add_nvds_meta (out_buf, batch_meta, NULL,
      nvds_batch_meta_copy_func, nvds_batch_meta_release_func);

  meta->meta_type = NVDS_BATCH_GST_META;

  batch_meta->base_meta.batch_meta = batch_meta;
  batch_meta->base_meta.copy_func = nvds_batch_meta_copy_func;
  batch_meta->base_meta.release_func = nvds_batch_meta_release_func;
  batch_meta->max_frames_in_batch = demux->num_surfaces_per_frame;

  for (k = 0; k < demux->num_surfaces_per_frame; k++) {
    NvDsFrameMeta *frame_meta = nvds_acquire_frame_meta_from_pool(batch_meta);
    NvDsFrameMeta *src_frame_meta = nvds_get_nth_frame_meta(src_batch_meta->frame_meta_list, index + k);
    nvds_copy_frame_meta(src_frame_meta, frame_meta);
    frame_meta->batch_id = k;
    nvds_add_frame_meta_to_batch(batch_meta, frame_meta);
  }

  return out_buf;
}

/* forward decl: defined below; the chain uses it to deliver caps to a
 * linked-but-capless src pad (late-added source recovery). */
static gboolean set_src_pad_caps (GstNvStreamDemux * nvstreamdemux, gint index);

static GstFlowReturn
gst_nvstreamdemux_sink_chain_cuda_batch (GstPad * pad, GstObject * parent,
    GstBuffer * buffer)
{
  GstNvStreamDemux *nvstreamdemux = GST_NVSTREAMDEMUX (parent);
  GstPad *src_pad;
  GstFlowReturn ret = GST_FLOW_OK;
  guint i;
  NvDsBatchMeta *batch_meta = NULL;
  GstMeta *gst_meta = NULL;
  gpointer state = NULL;
  NvDsMeta *dsmeta = NULL;
  struct timeval in_time;
  gdouble in_time_f;

  gettimeofday(&in_time, NULL);
  in_time_f = in_time.tv_sec * 1000.0 + in_time.tv_usec / 1000.0;

  while ((gst_meta = gst_buffer_iterate_meta (buffer, &state)) != NULL) {
    if (!gst_meta_api_type_has_tag (gst_meta->info->api, dsmeta_quark)) {
      continue;
    }

    dsmeta = (NvDsMeta *) gst_meta;
    /* Check if the metadata of NvDsMeta contains object bounding boxes. */
    if (dsmeta->meta_type == NVDS_BATCH_GST_META) {
      batch_meta = (NvDsBatchMeta *) dsmeta->meta_data;
      break;
    }
  }

  if (batch_meta == NULL) {
    GST_WARNING_OBJECT (nvstreamdemux, "NvDsBatchMeta not found for input buffer.");
    return GST_FLOW_ERROR;
  }

  for (i = 0; i < batch_meta->num_frames_in_batch; i+= nvstreamdemux->num_surfaces_per_frame) {
    NvDsFrameMeta *frame_meta = nvds_get_nth_frame_meta (batch_meta->frame_meta_list, i);
    guint stream_id = frame_meta->pad_index;
    GstBuffer *buf;
    gchar source_name[32] = {0};
    g_snprintf (source_name, 32, "nvv4l2decoder%u", stream_id);

    src_pad =
        GST_PAD (g_hash_table_lookup (nvstreamdemux->pad_indexes,
            stream_id + (char *)NULL));
    /* A src pad that is linked but still has NO caps would have every buffer
     * skipped below and be starved forever (0 fps, blank tile). That happens on
     * a pad requested AFTER this element already delivered caps to the
     * then-existing pads: set_src_pad_caps() only (re)pushes caps to pads present
     * in pad_indexes when a caps event / stream-frame-rate query fires, so a
     * later-added pad (race on the first dynamically-added source) never receives
     * caps. Deliver caps to it here, now that data is flowing (sink_caps and the
     * per-stream framerate are known by this point). Idempotent: once the pad has
     * caps this is skipped. stream-start is already seeded on the pad in
     * request_new_pad, so the order stays stream-start -> caps -> buffer. */
    if (src_pad && gst_pad_is_linked (src_pad)
        && !gst_pad_has_current_caps (src_pad)) {
      g_mutex_lock (&nvstreamdemux->ctx_lock);
      set_src_pad_caps (nvstreamdemux, stream_id);
      g_mutex_unlock (&nvstreamdemux->ctx_lock);
    }
    if (!src_pad || !gst_pad_has_current_caps (src_pad) || !gst_pad_is_linked (src_pad)) {
      continue;
    }

    buf =
        create_shared_mem_buf (nvstreamdemux, buffer, batch_meta, i, stream_id);


    GST_BUFFER_PTS (buf) = frame_meta->buf_pts;
    //g_print("DEMUX : Pushing buffer with PTS %" GST_TIME_FORMAT "\n", GST_TIME_ARGS(GST_BUFFER_PTS(buf)));

    if(nvds_enable_latency_measurement || nvds_latency_measurement_silent) {
        NvDsMetaList *l = NULL;
        for (l = batch_meta->batch_user_meta_list; l != NULL; l = l->next)
        {
          NvDsUserMeta *in_user_meta = (NvDsUserMeta *)(l->data);
          if (in_user_meta->base_meta.meta_type == NVDS_LATENCY_MEASUREMENT_META)
          {
            NvDsMetaCompLatency *latency_metadata = (NvDsMetaCompLatency *)in_user_meta->user_meta_data;

            if (!strncmp(latency_metadata->component_name, "nvv4l2decoder", strlen("nvv4l2decoder")))
            {
              if (latency_metadata->pad_index != stream_id)
              {
                continue;
              }
            }

            if (!strncmp(latency_metadata->component_name, "nvstreammux-", strlen("nvstreammux-")))
            {
              if (latency_metadata->pad_index != stream_id)
              {
                continue;
              }
            }

            NvDsBatchMeta *out_batch_meta = gst_buffer_get_nvds_batch_meta(buf);

            /* Acquire NvDsUserMeta user meta from pool */
            NvDsUserMeta *user_meta = nvds_acquire_user_meta_from_pool(out_batch_meta);

            /* Set NvDsUserMeta below */
            user_meta->user_meta_data = (void *)in_user_meta->base_meta.copy_func(in_user_meta, NULL);
            user_meta->base_meta.meta_type = NVDS_LATENCY_MEASUREMENT_META;
            user_meta->base_meta.copy_func = (NvDsMetaCopyFunc)in_user_meta->base_meta.copy_func;
            user_meta->base_meta.release_func = (NvDsMetaReleaseFunc)in_user_meta->base_meta.release_func;

            /* We want to add NvDsUserMeta to frame level */
            //g_print("Latency Meta = %s %d\n", latency_metadata->component_name, latency_metadata->source_id);
            nvds_add_user_meta_to_batch(out_batch_meta, user_meta);
          }
        }
    }

    if(nvds_enable_latency_measurement || nvds_latency_measurement_silent) {
      NvDsUserMeta *user_meta = nvds_set_input_system_timestamp(buf,
          GST_ELEMENT_NAME(nvstreamdemux));
      NvDsMetaCompLatency *latency_meta =
          (NvDsMetaCompLatency *) user_meta->user_meta_data;

      latency_meta->in_system_timestamp = in_time_f;
      nvds_set_output_system_timestamp(buf, GST_ELEMENT_NAME(nvstreamdemux));
    }

    ret = gst_pad_push (src_pad, buf);
    if (ret == GST_FLOW_NOT_LINKED)
      ret = GST_FLOW_OK;
  }

  gst_buffer_unref (buffer);

  return ret;
}

static gboolean
set_src_pad_caps (GstNvStreamDemux * nvstreamdemux, gint index)
{
  GList *keys, *keys_org = NULL;
  GList key = { index + (char *)NULL, NULL, NULL };
  gboolean ret = TRUE;
  gint batch_size = 0;
  GValue batchsize = G_VALUE_INIT;
  GstStructure *new_caps_s;
  gchar *caps_str;
  GstCapsFeatures *features = gst_caps_features_from_string ("memory:NVMM");

  if (!nvstreamdemux->sink_caps) {
    gst_caps_features_free (features);
    return TRUE;
  }

  if (index > -1)
    keys = &key;
  else {
    keys = g_hash_table_get_keys (nvstreamdemux->pad_indexes);
    keys_org = keys;
  }

  while (keys && ret) {
    gpointer stream_ptr = keys->data;
    GValue *frame_rate = NULL;
    GstPad *pad =
        GST_PAD (g_hash_table_lookup (nvstreamdemux->pad_indexes, stream_ptr));
    if (nvstreamdemux->pad_framerates) {
      g_mutex_lock (&(nvstreamdemux->pad_framerates->read_write_lock));
      frame_rate =  (GValue *) g_hash_table_lookup (nvstreamdemux->pad_framerates->table, stream_ptr);
      g_mutex_unlock (&(nvstreamdemux->pad_framerates->read_write_lock));
    }

    GstCaps *new_caps;
    GstStructure *new_caps_str;
    GstEvent *event;

    keys = keys->next;

    if (!pad || !frame_rate)
      continue;

    new_caps = gst_caps_copy (nvstreamdemux->sink_caps);
    new_caps_str = gst_caps_get_structure (new_caps, 0);
    if (frame_rate) {
      gint numerator = gst_value_get_fraction_numerator(frame_rate);
      gint denominator = gst_value_get_fraction_denominator(frame_rate);
      GST_DEBUG_OBJECT (nvstreamdemux, "Setting framerate::%d/%d\n", numerator, denominator);
      gst_structure_set_value (new_caps_str, "framerate", frame_rate);
    }
    new_caps_str = gst_structure_copy (new_caps_str);
    gst_caps_append_structure_full (new_caps, new_caps_str, NULL);

    new_caps_s = gst_caps_get_structure (new_caps, 0);
    GST_DEBUG_OBJECT (nvstreamdemux, "caps before = %s\n", gst_caps_to_string (new_caps));
    if (gst_structure_get_int (new_caps_s, "batch-size", &batch_size)) {
        if (batch_size > 1) {
            g_value_init (&batchsize, G_TYPE_INT);
            g_value_set_int (&batchsize, nvstreamdemux->num_surfaces_per_frame);
            new_caps = gst_caps_make_writable (new_caps);
            gst_caps_set_value (new_caps, "batch-size", &batchsize);
            g_value_unset (&batchsize);
        }
    }
    GST_DEBUG_OBJECT (nvstreamdemux, "caps after = %s\n", gst_caps_to_string (new_caps));

    new_caps = gst_caps_fixate (new_caps);

    event = gst_event_new_caps (new_caps);
    ret = gst_pad_push_event (pad, event);

    gst_caps_unref (new_caps);
  }

  if (keys_org)
    g_list_free (keys_org);

  gst_caps_features_free (features);

  return ret;
}

static gboolean
gst_nvstreamdemux_sink_event (GstPad * pad, GstObject * parent,
    GstEvent * event)
{
  GstNvStreamDemux *nvstreamdemux = GST_NVSTREAMDEMUX (parent);
  GstElement *element = GST_ELEMENT (parent);

  if (GST_EVENT_TYPE (event) == GST_EVENT_CAPS) {
    GstCaps *caps = NULL;
    GstQuery *query;
    gboolean ret;

    query = gst_nvquery_num_surfaces_per_buffer_new ();
    if (gst_pad_peer_query (nvstreamdemux->sinkpad, query)) {
      gboolean ok = gst_nvquery_num_surfaces_per_buffer_parse (query, &nvstreamdemux->num_surfaces_per_frame);
      if(!ok) {
        GST_WARNING_OBJECT (element, "num_surfaces_per_frame query failed; assuming == 1");
        nvstreamdemux->num_surfaces_per_frame = 1;
      }
    } else {
      nvstreamdemux->num_surfaces_per_frame = 1;
    }
    gst_query_unref (query);

    gst_event_parse_caps (event, &caps);
    if (caps) {
      gst_caps_replace (&nvstreamdemux->sink_caps, caps);
      GstStructure *str = gst_caps_get_structure (caps, 0);
      gint num_surfaces_per_frame;
      if (gst_structure_get_int (str, "num-surfaces-per-frame", &num_surfaces_per_frame)) {
        nvstreamdemux->num_surfaces_per_frame = num_surfaces_per_frame;
      }

    }

    g_mutex_lock (&nvstreamdemux->ctx_lock);
    ret = set_src_pad_caps(nvstreamdemux, -1);
    g_mutex_unlock(&nvstreamdemux->ctx_lock);
    gst_event_unref (event);

    return ret;
  }

  if (GST_EVENT_TYPE (event) == GST_EVENT_EOS) {
    GList *srcpads = element->srcpads;
    while (srcpads) {
      GstPad* srcpad = (GstPad *)(srcpads->data);
      if (!gst_pad_push_event (srcpad, gst_event_new_eos ())) {
        GST_WARNING_OBJECT (element,
          "nvstreamdemux srcpad[%s] EOS pad_push failed; "
          "the pad might not be active; "
          "EOS will be handled for default behaviour inside nvstreamdemux\n",
          GST_PAD_NAME(srcpad));
      }
      srcpads = srcpads->next;
    }
    return TRUE;
  }

  if (GST_EVENT_TYPE (event) == GST_NVEVENT_STREAM_RESET) {
    GstPad *src_pad = NULL;
    guint source_id = 0;
    gst_nvevent_parse_stream_reset (event, &source_id);
    gst_event_unref (event);
    src_pad =
      GST_PAD (g_hash_table_lookup (nvstreamdemux->pad_indexes,
            source_id + (char *)NULL));
    if (!src_pad) {
      return TRUE;
    }

    return gst_pad_push_event (src_pad, gst_nvevent_new_stream_reset (0));
  }

  if(GST_EVENT_TYPE (event) == GST_EVENT_SEGMENT) {
    /** ignore plain segment event as we do get segment event with
     * source ID associated as GST_NVEVENT_STREAM_SEGMENT from mux*/
    return TRUE;
  }

  if (nvstreamdemux->per_stream_eos == TRUE)
  {
      if (GST_EVENT_TYPE (event) ==  GST_NVEVENT_STREAM_EOS) {
          guint source_id = 0;
          GstPad *src_pad = NULL;
          gst_nvevent_parse_stream_eos (event, &source_id);
          src_pad =
              GST_PAD (g_hash_table_lookup (nvstreamdemux->pad_indexes,
                          source_id + (char *)NULL));
          if (!src_pad) {
              return TRUE;
          }
          gboolean ret = gst_pad_push_event (src_pad, gst_event_new_eos ());
          return ret;
      }
  }

  if (GST_EVENT_TYPE (event) == GST_EVENT_SINK_MESSAGE) {
    GstMessage *msg = NULL;
    GstPad *src_pad = NULL;
    gst_event_parse_sink_message (event, &msg);
    gboolean ret;
    if (msg && gst_nvmessage_is_stream_eos (msg))
    {
      guint stream_id;
      if (gst_nvmessage_parse_stream_eos(msg, &stream_id)) {
                    GST_ERROR_OBJECT (element,"Got EOS from stream %d\n", stream_id);
      }

      src_pad = GST_PAD (g_hash_table_lookup (nvstreamdemux->pad_indexes,
            stream_id + (char *)NULL));
      if (!src_pad)
      {
        return TRUE;
      }
      ret = gst_pad_push_event (src_pad, event);
      gst_message_unref(msg);
      return ret;
    }
    gst_message_unref(msg);
  }

  if (GST_EVENT_TYPE (event) == GST_NVEVENT_PAD_ADDED) {
    gst_event_unref (event);
    return TRUE;
  }
  if (GST_EVENT_TYPE (event) == GST_NVEVENT_PAD_DELETED) {
    guint source_id = 0;
    GstPad *src_pad = NULL;
    gboolean ret;
    gst_nvevent_parse_pad_deleted (event, &source_id);
    GST_DEBUG_OBJECT (nvstreamdemux, "Pad deleted %d\n", source_id);

    src_pad = GST_PAD (g_hash_table_lookup (nvstreamdemux->pad_indexes,
          source_id + (char *)NULL));
    if (!src_pad)
    {
      gst_event_unref (event);
      return TRUE;
    }
    ret = gst_pad_push_event (src_pad, event);
    return ret;
  }

  if (GST_EVENT_TYPE (event) == GST_NVEVENT_STREAM_SEGMENT) {
    GstSegment *segment = NULL;
    GstPad *src_pad = NULL;
    guint source_id = 0;
    gst_nvevent_parse_stream_segment (event, &source_id,
        &segment);
    src_pad =
      GST_PAD (g_hash_table_lookup (nvstreamdemux->pad_indexes,
            source_id + (char *)NULL));
    gst_event_unref (event);
    if (!src_pad) {
      return TRUE;
    }

    return gst_pad_push_event (src_pad, gst_event_new_segment (segment));
  }

  return gst_pad_event_default (pad, parent, event);
}

static gboolean
gst_nvstreamdemux_sink_query (GstPad * pad, GstObject * parent,
    GstQuery * query)
{
  GstNvStreamDemux *nvstreamdemux = GST_NVSTREAMDEMUX (parent);
  GstElement *element = GST_ELEMENT (parent);
  if (gst_nvquery_is_batch_size (query)) {
    return FALSE;
  }

  if (GST_QUERY_TYPE (query) == GST_QUERY_CUSTOM) {
    const GstStructure *str = gst_query_get_structure (query);
    if (str && gst_structure_has_name (str, "stream-frame-rate")) {
      GstCaps *new_caps;
      GstStructure *new_caps_str;
      guint stream_index = 0;
      gboolean ret;

      gst_structure_get_uint (str, "stream-id", &stream_index);
      gst_structure_get (str, "nvmultistream-pad-framerates", G_TYPE_POINTER, &(nvstreamdemux->pad_framerates), NULL);

      //GST_OBJECT_LOCK (nvstreamdemux);
      g_mutex_lock (&nvstreamdemux->ctx_lock);
      ret = set_src_pad_caps (nvstreamdemux, stream_index);
      g_mutex_unlock (&nvstreamdemux->ctx_lock);
      //GST_OBJECT_UNLOCK (nvstreamdemux);

      return ret;
    }
  }
  return gst_pad_query_default (pad, parent, query);
}

static void
gst_nvstreamdemux_class_init (GstNvStreamDemuxClass * klass)
{
  GObjectClass *gobject_class = G_OBJECT_CLASS (klass);
  GstElementClass *gstelement_class = GST_ELEMENT_CLASS (klass);

  gst_element_class_set_static_metadata (gstelement_class,
      "Stream demultiplexer", "Generic", "1-to-N pipes stream demultiplexing",
      "NVIDIA Corporation. Post on Deepstream for Tesla forum for any queries "
      "@ https://devtalk.nvidia.com/default/board/209/");

  gobject_class->finalize = GST_DEBUG_FUNCPTR (gst_nvstreamdemux_finalize);

  gobject_class->set_property = gst_nvdsstreamdemux_set_property;
  gobject_class->get_property = gst_nvdsstreamdemux_get_property;

  gst_element_class_add_static_pad_template (gstelement_class,
      &nvstreamdemux_sinkpad_template);
  gst_element_class_add_static_pad_template (gstelement_class,
      &nvstreamdemux_srcpad_template);

  gstelement_class->request_new_pad =
      GST_DEBUG_FUNCPTR (gst_nvstreamdemux_request_new_pad);
  gstelement_class->release_pad =
      GST_DEBUG_FUNCPTR (gst_nvstreamdemux_release_pad);

  g_object_class_install_property (gobject_class, PROP_PER_STREAM_EOS,
      g_param_spec_boolean ("per-stream-eos", "per-stream-eos",
          "Send EOS for individual streams as soon as EOS is received for corresponding source.\n"
          "\t\t\tBy default EOS is sent when EOS from all streams is received \n"
          "\t\t\tConnect all the SRC pads of demux to separate sink components when this is set to TRUE",
          DEFAULT_PER_STREAM_EOS,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

}

static void
gst_nvstreamdemux_init (GstNvStreamDemux * nvstreamdemux)
{
  nvstreamdemux->sinkpad =
      gst_pad_new_from_static_template (&nvstreamdemux_sinkpad_template,
      "sink");

  gst_pad_set_chain_function (nvstreamdemux->sinkpad,
      GST_DEBUG_FUNCPTR (gst_nvstreamdemux_sink_chain_cuda_batch));

  gst_pad_set_event_function (nvstreamdemux->sinkpad,
      GST_DEBUG_FUNCPTR (gst_nvstreamdemux_sink_event));

  gst_pad_set_query_function (nvstreamdemux->sinkpad,
      GST_DEBUG_FUNCPTR (gst_nvstreamdemux_sink_query));

  gst_element_add_pad (GST_ELEMENT (nvstreamdemux), nvstreamdemux->sinkpad);

  nvstreamdemux->pad_indexes = g_hash_table_new (NULL, NULL);
  nvstreamdemux->num_surfaces_per_frame = 1; /** default */
  g_mutex_init(&nvstreamdemux->ctx_lock);

  if (!dsmeta_quark)
    dsmeta_quark = g_quark_from_static_string (NVDS_META_STRING);

  nvstreamdemux->per_stream_eos = FALSE;
}

static void
gst_nvstreamdemux_finalize (GObject * object)
{
  GstNvStreamDemux *nvstreamdemux = (GstNvStreamDemux *) (object);

  if (nvstreamdemux->sink_caps) {
    gst_caps_replace (&nvstreamdemux->sink_caps, NULL);
    nvstreamdemux->sink_caps = NULL;
  }

  if (nvstreamdemux->pad_indexes) {
    g_hash_table_unref (nvstreamdemux->pad_indexes);
    nvstreamdemux->pad_indexes = NULL;
  }

  G_OBJECT_CLASS (parent_class)->finalize (object);
}
