/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * SECTION:element-nvdsmetamux
 *
 * nvdsmetamux muxes multiple batch_meta into one batch_meta.
 *
 * Application need set the same source id if want to merge meta data into
 * one source. Don't use the same source id if don't want to merge meta
 * data.
 *
 * Application need select one active sink pad which gstbuffer will be
 * passed to src pad. The default active sink pad is the sink pad 0.
 *
 * nvdsmetamux will find the nearest video frame meta based on the active
 * sink pad video frame PTS and merge the meta data into output buffer
 * batch_meta.
 * Application can set the PTS tolerance threshold, if the diff of frame
 * PTS for the same source id is bigger than the threshold, The meta data
 * merge will be skipped. The default of the PTS match threshold is 60ms.
 *
 * Input buffer will be return if all video frame PTS are less than current
 * selected PTS for all source id in the batch.
 *
 * Application can select meta data based on source id and model id.
 *
 */

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <list>
#include "nvbufsurface.h"
#include "gst-nvevent.h"
#include "gst-nvquery.h"
#include "gstnvdsinfer.h"
#include "gstnvdsmetamux.h"
#include "nvdsmetamux_property_parser.h"

GST_DEBUG_CATEGORY_STATIC (gst_nvdsmetamux_debug);
#define GST_CAT_DEFAULT gst_nvdsmetamux_debug

G_DEFINE_TYPE (GstNvDsMetaMuxPad, gst_nvdsmetamux_pad, GST_TYPE_AGGREGATOR_PAD);
#define _do_init \
    GST_DEBUG_CATEGORY_INIT (gst_nvdsmetamux_debug, "nvdsmetamux", 0, "nvdsmetamux element");
#define gst_nvdsmetamux_parent_class parent_class
#define gst_nvdsmetamux_pad_parent_class parent_class
G_DEFINE_TYPE_WITH_CODE (GstNvDsMetaMux, gst_nvdsmetamux,
    GST_TYPE_AGGREGATOR, _do_init);

#define COMMON_AUDIO_CAPS \
  "channels = (int) 1, " \
  "rate = (int) [ 1, MAX ]"

static GstStaticPadTemplate sinkpad_template =
    GST_STATIC_PAD_TEMPLATE ("sink_%u",
    GST_PAD_SINK,
    GST_PAD_REQUEST,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES ("memory:NVMM",
            "{ " "NV12, RGBA, I420 }") "; "
        "audio/x-raw(memory:NVMM), "
        "format = { " "S16LE, F32LE }, "
        "layout = (string) interleaved, " COMMON_AUDIO_CAPS));

static GstStaticPadTemplate srcpad_template = GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES ("memory:NVMM",
            "{ " "NV12, RGBA, I420 }") "; "
        "audio/x-raw, "
        "format = { " "S16LE, F32LE }, "
        "layout = (string) interleaved, " COMMON_AUDIO_CAPS));

enum
{
  PROP_0,
  PROP_ACTIVE_PAD,
  PROP_PTS_TOLERANCE,
  PROP_CONFIG_FILE
};

#define DEFAULT_TS_TOLERANCE 60000
#define MAX_BUF_CNT 4

static GstFlowReturn
gst_nvdsmetamux_aggregate (GstAggregator * aggregator, gboolean timeout);
static gboolean
gst_nvdsmetamux_sink_event (GstAggregator * aggregator, GstAggregatorPad * pad,
    GstEvent * event);

static GstAggregatorPad *gst_nvdsmetamux_create_new_pad (GstAggregator * agg,
    GstPadTemplate * templ, const gchar * req_name, const GstCaps * caps);
static void gst_nvdsmetamux_release_pad (GstElement * element, GstPad * pad);

static void gst_nvdsmetamux_get_property (GObject * object,
    guint prop_id, GValue * value, GParamSpec * pspec);
static void gst_nvdsmetamux_set_property (GObject * object,
    guint prop_id, const GValue * value, GParamSpec * pspec);
static void gst_nvdsmetamux_finalize (GObject * object);

static void gst_nvdsmetamux_reset (GstElement * element);
static void gst_nvdsmetamux_reset_pad (GstNvDsMetaMuxPad * pad);

static void gst_nvdsmetamux_pad_finalize (GObject * object);

static gboolean gst_nvdsmetamux_start (GstAggregator * aggregator);
static gboolean gst_nvdsmetamux_stop (GstAggregator * aggregator);
static GstFlowReturn gst_nvdsmetamux_flush (GstAggregator * aggregator);
static gboolean gst_nvdsmetamux_are_all_pads_eos (GstNvDsMetaMux * mux);
static GstClockTime gst_nvdsmetamux_get_next_time (GstAggregator * aggregator);
static GstFlowReturn gst_nvdsmetamux_update_src_caps (GstAggregator *
    aggregator, GstCaps * caps, GstCaps ** ret);
static GstFlowReturn gst_nvdsmetamux_clear_buflist (GstAggregator * aggregator);
static GstFlowReturn
gst_nvdsmetamux_pad_flush (GstAggregatorPad * pad, GstAggregator * aggregator)
{
  //GstNvDsMetaMuxPad *mpad = GST_NVDSMETAMUX_PAD (pad);

  return GST_FLOW_OK;
}

static void
gst_nvdsmetamux_pad_class_init (GstNvDsMetaMuxPadClass * klass)
{
  GstAggregatorPadClass *aggregatorpad_class = (GstAggregatorPadClass *) klass;
  GObjectClass *gobject_class = (GObjectClass *) klass;

  gobject_class->finalize = gst_nvdsmetamux_pad_finalize;

  aggregatorpad_class->flush = GST_DEBUG_FUNCPTR (gst_nvdsmetamux_pad_flush);
}

static void
gst_nvdsmetamux_pad_init (GstNvDsMetaMuxPad * pad)
{
  gst_nvdsmetamux_reset_pad (pad);
  pad->buf_list = gst_buffer_list_new ();
}

static void
gst_nvdsmetamux_class_init (GstNvDsMetaMuxClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;
  GstAggregatorClass *gstaggregator_class;

  gobject_class = (GObjectClass *) klass;
  gstelement_class = (GstElementClass *) klass;
  gstaggregator_class = (GstAggregatorClass *) klass;

  gobject_class->get_property = gst_nvdsmetamux_get_property;
  gobject_class->set_property = gst_nvdsmetamux_set_property;
  gobject_class->finalize = gst_nvdsmetamux_finalize;

  g_object_class_install_property (gobject_class, PROP_ACTIVE_PAD,
      g_param_spec_string ("active-pad", "Active Sink Pad",
          "Active sink pad which buffer will transfer to src pad",
          NULL, (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_PTS_TOLERANCE,
      g_param_spec_int64 ("pts-tolerance", "Time Diff Tolerance",
          "Time diff tolerance when search the same frame of the same source id in microseconds",
          G_MININT64, G_MAXINT64, DEFAULT_TS_TOLERANCE,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_CONFIG_FILE,
      g_param_spec_string ("config-file", "Preprocess Config File",
          "Preprocess Config File",
          NULL, (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  gstaggregator_class->create_new_pad =
      GST_DEBUG_FUNCPTR (gst_nvdsmetamux_create_new_pad);
  gstelement_class->release_pad =
      GST_DEBUG_FUNCPTR (gst_nvdsmetamux_release_pad);

  gstaggregator_class->start = GST_DEBUG_FUNCPTR (gst_nvdsmetamux_start);
  gstaggregator_class->stop = GST_DEBUG_FUNCPTR (gst_nvdsmetamux_stop);
  gstaggregator_class->aggregate =
      GST_DEBUG_FUNCPTR (gst_nvdsmetamux_aggregate);
  gstaggregator_class->sink_event =
      GST_DEBUG_FUNCPTR (gst_nvdsmetamux_sink_event);
  gstaggregator_class->flush = GST_DEBUG_FUNCPTR (gst_nvdsmetamux_flush);
  gstaggregator_class->get_next_time =
      GST_DEBUG_FUNCPTR (gst_nvdsmetamux_get_next_time);
  gstaggregator_class->update_src_caps =
      GST_DEBUG_FUNCPTR (gst_nvdsmetamux_update_src_caps);

  gst_element_class_add_static_pad_template_with_gtype (gstelement_class,
      &sinkpad_template, GST_TYPE_NVDSMETAMUX_PAD);
  gst_element_class_add_static_pad_template_with_gtype (gstelement_class,
      &srcpad_template, GST_TYPE_AGGREGATOR_PAD);
  gst_element_class_set_static_metadata (gstelement_class, "meta muxer",
      "Codec/Muxer",
      "Muxes batch_meta from different model into one batch_meta",
      "NVIDIA Corporation. Post on Deepstream for Tesla forum for any queries "
      "@ https://devtalk.nvidia.com/default/board/209/");
}

static void
gst_nvdsmetamux_init (GstNvDsMetaMux * mux)
{
  mux->srcpad = GST_AGGREGATOR_CAST (mux)->srcpad;

  /* property */
  mux->active_sink_pad = NULL;
  mux->pts_tolerance = DEFAULT_TS_TOLERANCE;
  mux->config_file_path = NULL;

  mux->pts_table = g_hash_table_new (NULL, NULL);

  gst_nvdsmetamux_reset (GST_ELEMENT (mux));
}

static void
gst_nvdsmetamux_finalize (GObject * object)
{
  GstNvDsMetaMux *mux = GST_NVDSMETAMUX (object);

  gst_nvdsmetamux_reset (GST_ELEMENT (object));
  g_hash_table_destroy (mux->pts_table);

  G_OBJECT_CLASS (gst_nvdsmetamux_parent_class)->finalize (object);
}

static void
gst_nvdsmetamux_pad_finalize (GObject * object)
{
  GstNvDsMetaMuxPad *pad = GST_NVDSMETAMUX_PAD (object);

  gst_buffer_list_unref (pad->buf_list);
  gst_nvdsmetamux_reset_pad (pad);

  G_OBJECT_CLASS (gst_nvdsmetamux_pad_parent_class)->finalize (object);
}

static GstFlowReturn
gst_nvdsmetamux_flush (GstAggregator * aggregator)
{
  gst_nvdsmetamux_reset (GST_ELEMENT (aggregator));
  return GST_FLOW_OK;
}

static gboolean
gst_nvdsmetamux_start (GstAggregator * aggregator)
{
  gst_nvdsmetamux_reset (GST_ELEMENT (aggregator));
  return TRUE;
}

static gboolean
gst_nvdsmetamux_stop (GstAggregator * aggregator)
{
  gst_nvdsmetamux_clear_buflist(aggregator);
  gst_nvdsmetamux_reset (GST_ELEMENT (aggregator));
  return TRUE;
}

static gboolean
remove_yes (gpointer key, gpointer value, gpointer user_data)
{
  return TRUE;
}

static void
gst_nvdsmetamux_reset (GstElement * element)
{
  GstNvDsMetaMux *mux = GST_NVDSMETAMUX (element);

  while (g_hash_table_size (mux->pts_table)) {
    g_hash_table_foreach_remove (mux->pts_table, remove_yes, NULL);
  }
}

static gboolean
gst_nvdsmetamux_sink_event (GstAggregator * aggregator, GstAggregatorPad * apad,
    GstEvent * event)
{
  GstNvDsMetaMux *mux = GST_NVDSMETAMUX (aggregator);
  GstNvDsMetaMuxPad *mpad = (GstNvDsMetaMuxPad *) apad;
  guint source_id = 0;
  gboolean ret = TRUE;

  GST_DEBUG_OBJECT (mpad, "Got event: %" GST_PTR_FORMAT, event);

  switch (GST_EVENT_TYPE (event)) {
    case GST_EVENT_CAPS:
    {
      GstCaps *caps;

      gst_event_parse_caps (event, &caps);

      break;
    }
    default:
      break;
  }

  switch (static_cast < GstNvEventType > GST_EVENT_TYPE (event)) {
    case GST_NVEVENT_STREAM_EOS:
    {
      gst_nvevent_parse_stream_eos (event, &source_id);
      GST_DEBUG_OBJECT (mux, "metamux stream eos: %d\n", source_id);
      g_hash_table_insert (mux->pts_table, source_id + (char *) NULL,
          (gpointer) GST_CLOCK_TIME_NONE);
      break;
    }
    case GST_NVEVENT_PAD_ADDED:
    {
      gst_nvevent_parse_pad_added (event, &source_id);
      GST_DEBUG_OBJECT (mux, "Pad added %d\n", source_id);
      mpad->source_ids.push_back (source_id);
      break;
    }
    default:
      break;
  }

  if (!ret)
    return FALSE;

  if (mux->active_pad
      && apad != (GstAggregatorPad *) mux->active_pad
      && GST_EVENT_TYPE (event) != GST_EVENT_EOS) {
    if (event)
      gst_event_unref (event);

    return TRUE;
  }

  return GST_AGGREGATOR_CLASS (parent_class)->sink_event (aggregator, apad,
      event);;
}

static void
gst_nvdsmetamux_reset_pad (GstNvDsMetaMuxPad * pad)
{
  GST_DEBUG_OBJECT (pad, "resetting pad");

  gst_nvdsmetamux_pad_flush (GST_AGGREGATOR_PAD_CAST (pad), NULL);
}

static GstAggregatorPad *
gst_nvdsmetamux_create_new_pad (GstAggregator * agg,
    GstPadTemplate * templ, const gchar * req_name, const GstCaps * caps)
{
  GstAggregatorPad *aggpad;
  GstNvDsMetaMuxPad *pad = NULL;
  const gchar *name = "sink_%u";

  aggpad =
      GST_AGGREGATOR_CLASS (gst_nvdsmetamux_parent_class)->create_new_pad (agg,
      templ, name, caps);
  if (aggpad == NULL)
    return NULL;

  pad = GST_NVDSMETAMUX_PAD (aggpad);

  gst_nvdsmetamux_reset_pad (pad);

  return aggpad;
}

static void
gst_nvdsmetamux_release_pad (GstElement * element, GstPad * pad)
{
  GstNvDsMetaMuxPad *mpad = GST_NVDSMETAMUX_PAD (gst_object_ref (pad));

  gst_nvdsmetamux_reset_pad (mpad);

  GST_ELEMENT_CLASS (gst_nvdsmetamux_parent_class)->release_pad (element, pad);
}

static GstFlowReturn
gst_nvdsmetamux_push (GstNvDsMetaMux * mux, GstBuffer * buf)
{
  return gst_aggregator_finish_buffer (GST_AGGREGATOR_CAST (mux), buf);
}

static gboolean
gst_nvdsmetamux_are_all_pads_eos (GstNvDsMetaMux * mux)
{
  GList *l;

  GST_OBJECT_LOCK (mux);
  for (l = GST_ELEMENT_CAST (mux)->sinkpads; l; l = l->next) {
    GstNvDsMetaMuxPad *pad = GST_NVDSMETAMUX_PAD (l->data);

    if (!gst_aggregator_pad_is_eos (GST_AGGREGATOR_PAD (pad))) {
      GST_OBJECT_UNLOCK (mux);
      return FALSE;
    }
  }
  GST_OBJECT_UNLOCK (mux);
  return TRUE;
}

typedef struct
{
  NvBufSurface surf;
  GstBuffer *src_buf;
} GstNvStreamMemory;

static void
shared_mem_buf_unref_callback (gpointer data)
{
  GstNvStreamMemory *mem = (GstNvStreamMemory *) data;

  GST_LOG ("unref active pad buffer: %p refcount: %d", mem->src_buf,
      GST_OBJECT_REFCOUNT_VALUE (mem->src_buf));
  gst_buffer_unref (mem->src_buf);
  g_slice_free (GstNvStreamMemory, mem);
}

static GstBuffer *
gst_nvdsmetamux_get_buffer (GstAggregatorPad * apad, guint idx)
{
  GstNvDsMetaMuxPad *mpad = GST_NVDSMETAMUX_PAD (apad);
  GstBuffer *buf;

  if (idx < gst_buffer_list_length (mpad->buf_list)) {
    return gst_buffer_list_get (mpad->buf_list, idx);
  }

  if (idx == gst_buffer_list_length (mpad->buf_list)) {
    buf = gst_aggregator_pad_pop_buffer (apad);
    if (buf) {
      gst_buffer_ref (buf);
      gst_buffer_list_insert (mpad->buf_list, -1, buf);
      GST_DEBUG ("get buffer from pad: %s buffer: %p refcount: %d",
          gst_pad_get_name (apad), buf, GST_OBJECT_REFCOUNT_VALUE (buf));
    }
    return buf;
  }

  return NULL;
}

static gboolean
gst_nvdsmetamux_search_nearest (GstNvDsMetaMux * mux, GstBuffer * buf,
    guint source_id, GstClockTime buf_pts, GstClockTime * nearest_pts)
{
  NvDsBatchMeta *batch_meta;
  NvDsMetaList *l_frame;

  batch_meta = gst_buffer_get_nvds_batch_meta (buf);
  if (batch_meta == NULL) {
    GST_WARNING_OBJECT (mux, "NvDsBatchMeta not found for input buffer.");
    return FALSE;
  }

  for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next) {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);

    if (frame_meta->source_id == source_id) {
      if (*nearest_pts == GST_CLOCK_TIME_NONE) {
        *nearest_pts = frame_meta->buf_pts;
      }
      if (ABS (GST_CLOCK_DIFF (frame_meta->buf_pts,
                  buf_pts)) < ABS (GST_CLOCK_DIFF (*nearest_pts, buf_pts))) {
        *nearest_pts = frame_meta->buf_pts;
      }
      if (frame_meta->buf_pts >= buf_pts) {
        GST_DEBUG_OBJECT (mux,
            "nearest PTS %" GST_TIME_FORMAT " Buffer PTS %" GST_TIME_FORMAT ".",
            GST_TIME_ARGS (*nearest_pts), GST_TIME_ARGS (buf_pts));
        if (ABS (GST_CLOCK_DIFF (*nearest_pts,
                    buf_pts)) / 1000 > mux->pts_tolerance) {
          *nearest_pts = GST_CLOCK_TIME_NONE;
        }
        return TRUE;
      }
    }
  }

  return FALSE;
}

static NvDsFrameMeta *
gst_nvdsmetamux_search_frame (GstNvDsMetaMux * mux, GstBuffer * buf,
    guint source_id, GstClockTime buf_pts)
{
  NvDsBatchMeta *batch_meta;
  NvDsMetaList *l_frame;

  batch_meta = gst_buffer_get_nvds_batch_meta (buf);
  if (batch_meta == NULL) {
    GST_WARNING_OBJECT (mux, "NvDsBatchMeta not found for input buffer.");
    return NULL;
  }

  for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next) {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);

    if (frame_meta->source_id == source_id && frame_meta->buf_pts == buf_pts) {
      if (!GST_CLOCK_TIME_IS_VALID (g_hash_table_lookup (mux->pts_table,
                  source_id + (char *) NULL))) {
        GST_DEBUG_OBJECT (mux, "source id: %d last PTS not valid, seems eos",
            source_id);
        continue;
      }

      GST_DEBUG_OBJECT (mux, "update source id: %d last PTS %" GST_TIME_FORMAT,
          frame_meta->source_id, GST_TIME_ARGS (buf_pts));

      g_hash_table_insert (mux->pts_table,
          frame_meta->source_id + (char *) NULL, (gpointer) buf_pts);
      return frame_meta;
    }
  }

  return NULL;
}

static gboolean
gst_nvdsmetamux_check_needed (GstNvDsMetaMux * mux, gint unique_component_id,
    guint source_id)
{
  std::vector < gint > src_ids;

  if (!mux->config_file_parse_successful)
    return TRUE;

  auto model_source = mux->model_source_map.find (unique_component_id);

  if (model_source == mux->model_source_map.end ()) {
    GST_DEBUG_OBJECT (mux,
        "Model Unique ID: %d config not found. Keep the meta data",
        unique_component_id);
    return TRUE;
  } else
    src_ids = model_source->second.source_id_vector;

  if (std::find (src_ids.begin (), src_ids.end (), source_id) == src_ids.end ()
      && src_ids[0] != -1) {
    GST_DEBUG_OBJECT (mux, "Model Unique ID: %d disenabled source id: %d",
        unique_component_id, source_id);
    return FALSE;
  }

  GST_DEBUG_OBJECT (mux, "Model Unique ID: %d enabled source id: %d",
      unique_component_id, source_id);

  return TRUE;
}

static GstFlowReturn
gst_nvdsmetamux_filter_metadata (GstNvDsMetaMux * mux, GstBuffer * buf)
{
  GstFlowReturn ret = GST_FLOW_OK;
  NvDsObjectMeta *obj_meta = NULL;
  NvDsMetaList *l_frame = NULL;
  NvDsMetaList *l_obj = NULL;
  NvDsMetaList *l_user = NULL;
  std::list < NvDsObjectMeta * >ometa_free_list;
  std::list < NvDsClassifierMeta * >cmeta_free_list;
  std::list < NvDsUserMeta * >umeta_free_list;

  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);
  if (batch_meta == NULL) {
    GST_WARNING_OBJECT (mux, "NvDsBatchMeta not found for input buffer.");
    return GST_FLOW_ERROR;
  }

  for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next) {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);
    for (l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) {
      obj_meta = (NvDsObjectMeta *) (l_obj->data);
      if (!gst_nvdsmetamux_check_needed (mux, obj_meta->unique_component_id,
              frame_meta->source_id)) {
        ometa_free_list.push_back (obj_meta);
      }

      for (NvDsMetaList * l_class = obj_meta->classifier_meta_list;
          l_class != NULL; l_class = l_class->next) {
        NvDsClassifierMeta *cmeta = (NvDsClassifierMeta *) l_class->data;
        if (!gst_nvdsmetamux_check_needed (mux, cmeta->unique_component_id,
                frame_meta->source_id)) {
          cmeta_free_list.push_back (cmeta);
        }
      }

      //filter the classifier
      std::list < NvDsClassifierMeta * >::iterator it, it_temp;
      for (it = cmeta_free_list.begin (); it != cmeta_free_list.end ();) {
        it_temp = it;
        nvds_remove_classifier_meta_from_obj (obj_meta,
            (NvDsClassifierMeta *) (*it_temp));
        it++;
      }
      cmeta_free_list.clear ();
    }

    //filter the objects
    std::list < NvDsObjectMeta * >::iterator it, it_temp;
    for (it = ometa_free_list.begin (); it != ometa_free_list.end ();) {
      it_temp = it;
      nvds_remove_obj_meta_from_frame (frame_meta,
          (NvDsObjectMeta *) (*it_temp));
      it++;
    }
    ometa_free_list.clear ();

    for (l_user = frame_meta->frame_user_meta_list; l_user != NULL;
        l_user = l_user->next) {
      NvDsUserMeta *user_meta = (NvDsUserMeta *) l_user->data;
      if (user_meta->base_meta.meta_type == NVDSINFER_TENSOR_OUTPUT_META) {
        NvDsInferTensorMeta *tensor_meta =
            (NvDsInferTensorMeta *) user_meta->user_meta_data;
        if (!gst_nvdsmetamux_check_needed (mux, tensor_meta->unique_id,
                frame_meta->source_id)) {
          umeta_free_list.push_back (user_meta);
        }
      }
    }
    //filter the tensor meta
    {
      std::list < NvDsUserMeta * >::iterator it, it_temp;
      for (it = umeta_free_list.begin (); it != umeta_free_list.end ();) {
        it_temp = it;
        nvds_remove_user_meta_from_frame (frame_meta,
            (NvDsUserMeta *) (*it_temp));
        it++;
      }
      umeta_free_list.clear ();
    }
  }

  return ret;
}


static GstFlowReturn
gst_nvdsmetamux_copy_meta (GstNvDsMetaMux * mux,
    NvDsFrameMeta * matched_frame_meta, NvDsFrameMeta * frame_meta)
{
  GstFlowReturn ret = GST_FLOW_OK;

  nvds_copy_obj_meta_list (matched_frame_meta->obj_meta_list, frame_meta);
  //can't draw crossing lines if no display meta.
  nvds_copy_display_meta_list (matched_frame_meta->display_meta_list,
      frame_meta);
  nvds_copy_frame_user_meta_list (matched_frame_meta->frame_user_meta_list,
      frame_meta);

  return ret;
}

static GstFlowReturn
gst_nvdsmetamux_merge_metadata (GstNvDsMetaMux * mux, GstBuffer * buf)
{
  NvDsMetaList *l_frame = NULL;
  GstIterator *pads;
  GValue padptr = { 0, };
  GstFlowReturn ret = GST_FLOW_OK;

  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);
  if (batch_meta == NULL) {
    GST_WARNING_OBJECT (mux, "NvDsBatchMeta not found for input buffer.");
    return GST_FLOW_ERROR;
  }

  for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next) {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);
    GST_DEBUG_OBJECT (mux, "merge source id: %d PTS %" GST_TIME_FORMAT,
        frame_meta->source_id, GST_TIME_ARGS (frame_meta->buf_pts));
  }

  for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next) {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);
    NvDsFrameMeta *matched_frame_meta;

    pads = gst_element_iterate_sink_pads (GST_ELEMENT (mux));

    gboolean done = FALSE;
    while (!done) {
      switch (gst_iterator_next (pads, &padptr)) {
        case GST_ITERATOR_OK:
        {
          GstAggregatorPad *apad =
              (GstAggregatorPad *) g_value_get_object (&padptr);
          GstNvDsMetaMuxPad *mpad = (GstNvDsMetaMuxPad *) apad;
          std::vector < gint > src_ids = mpad->source_ids;
          if (apad == (GstAggregatorPad *) mux->active_pad
              || std::find (src_ids.begin (), src_ids.end (),
                  frame_meta->source_id) == src_ids.end ())
            continue;

          GST_DEBUG_OBJECT (mux, "search pad: %s for source id: %d",
              gst_pad_get_name (apad), frame_meta->source_id);

          guint index = 0;
          GstClockTime nearest_pts = GST_CLOCK_TIME_NONE;
          while (1) {
            buf = gst_nvdsmetamux_get_buffer (apad, index);
            if (!buf) {
              break;
            }
            if (gst_nvdsmetamux_search_nearest (mux, buf, frame_meta->source_id,
                    frame_meta->buf_pts, &nearest_pts)) {
              break;
            }
            index++;
          }
          if (nearest_pts == GST_CLOCK_TIME_NONE)
            continue;

          index = 0;
          while (1) {
            buf = gst_nvdsmetamux_get_buffer (apad, index);
            if (!buf) {
              break;
            }

            matched_frame_meta = gst_nvdsmetamux_search_frame (mux, buf,
                frame_meta->source_id, nearest_pts);
            if (matched_frame_meta) {
              ret =
                  gst_nvdsmetamux_copy_meta (mux, matched_frame_meta,
                  frame_meta);
              break;
            }
            index++;
          }
          break;
        }
        case GST_ITERATOR_DONE:
          done = TRUE;
          break;
        case GST_ITERATOR_RESYNC:
          gst_iterator_resync (pads);
          break;
        case GST_ITERATOR_ERROR:
          g_assert_not_reached ();
          break;
      }
      g_value_reset (&padptr);
    }
    g_value_unset (&padptr);
    gst_iterator_free (pads);
  }

  return ret;
}

static gboolean
gst_nvdsmetamux_check_past_frame (GstNvDsMetaMux * mux, GstBuffer * buf)
{
  NvDsMetaList *l_frame;

  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);
  if (batch_meta == NULL) {
    GST_WARNING_OBJECT (mux, "NvDsBatchMeta not found for input buffer.");
    return FALSE;
  }

  for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next) {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);
    guint source_id = frame_meta->source_id;
    GstClockTime current_pts = 0;

    if (!g_hash_table_lookup_extended (mux->pts_table,
            source_id + (char *) NULL, NULL, (gpointer *) & current_pts)) {
      GST_DEBUG_OBJECT (mux, "source id: %d last PTS not found", source_id);
      continue;
    }

    if (!GST_CLOCK_TIME_IS_VALID (current_pts)) {
      GST_DEBUG_OBJECT (mux, "source id: %d last PTS not valid, seems eos",
          source_id);
      continue;
    }

    GST_DEBUG_OBJECT (mux,
        "source id: %d last PTS %" GST_TIME_FORMAT " Buffer PTS %"
        GST_TIME_FORMAT ".", source_id, GST_TIME_ARGS (current_pts),
        GST_TIME_ARGS (frame_meta->buf_pts));

    if (frame_meta->buf_pts >= current_pts)
      return FALSE;
  }

  return TRUE;
}

/**
 * clear all buffer list if EOS.
 */
static GstFlowReturn
gst_nvdsmetamux_clear_buflist (GstAggregator * aggregator)
{
  GstIterator *pads;
  GValue padptr = { 0, };
  gboolean done = FALSE;
  GstBuffer *buf;
  GstFlowReturn ret = GST_FLOW_OK;
  GstNvDsMetaMux *mux;

  mux = GST_NVDSMETAMUX (aggregator);
  pads = gst_element_iterate_sink_pads (GST_ELEMENT (mux));

  while (!done) {
    switch (gst_iterator_next (pads, &padptr)) {
      case GST_ITERATOR_OK:
      {
        GstAggregatorPad *apad =
            (GstAggregatorPad *) g_value_get_object (&padptr);
        while (1) {
          buf = gst_nvdsmetamux_get_buffer (apad, 0);
          if (!buf) {
            break;
          }
          gst_buffer_list_remove (((GstNvDsMetaMuxPad *) apad)->buf_list, 0, 1);
          GST_DEBUG ("unref pad: %s buffer: %p refcount: %d",
            gst_pad_get_name (apad), buf, GST_OBJECT_REFCOUNT_VALUE (buf));
          gst_buffer_unref (buf);
        }
        break;
      }
      case GST_ITERATOR_DONE:
        done = TRUE;
        break;
      case GST_ITERATOR_RESYNC:
        gst_iterator_resync (pads);
        break;
      case GST_ITERATOR_ERROR:
        g_assert_not_reached ();
        break;
    }
    g_value_reset (&padptr);
  }
  g_value_unset (&padptr);
  gst_iterator_free (pads);

  return ret;
}

static GstFlowReturn
gst_nvdsmetamux_return_buffer (GstNvDsMetaMux * mux)
{
  GstIterator *pads;
  GValue padptr = { 0, };
  gboolean done = FALSE;
  GstBuffer *buf;
  GstFlowReturn ret = GST_FLOW_OK;

  pads = gst_element_iterate_sink_pads (GST_ELEMENT (mux));

  while (!done) {
    switch (gst_iterator_next (pads, &padptr)) {
      case GST_ITERATOR_OK:
      {
        GstAggregatorPad *apad =
            (GstAggregatorPad *) g_value_get_object (&padptr);
        GstNvDsMetaMuxPad *mpad = GST_NVDSMETAMUX_PAD (apad);
        GST_DEBUG_OBJECT (mux, "pad: %s buffer list len: %d",
            gst_pad_get_name (apad), gst_buffer_list_length (mpad->buf_list));
        if (apad == (GstAggregatorPad *) mux->active_pad)
          continue;

        //guint index = 0;
        while (1) {
          buf = gst_nvdsmetamux_get_buffer (apad, 0);
          if (!buf) {
            break;
          }
          if (gst_nvdsmetamux_check_past_frame (mux, buf)
              || gst_buffer_list_length (mpad->buf_list) >= MAX_BUF_CNT) {
            gst_buffer_list_remove (((GstNvDsMetaMuxPad *) apad)->buf_list, 0,
                1);
            GST_DEBUG ("unref pad: %s buffer: %p refcount: %d",
                gst_pad_get_name (apad), buf, GST_OBJECT_REFCOUNT_VALUE (buf));
            gst_buffer_unref (buf);
          } else
            break;
          //index ++;
        }
        break;
      }
      case GST_ITERATOR_DONE:
        done = TRUE;
        break;
      case GST_ITERATOR_RESYNC:
        gst_iterator_resync (pads);
        break;
      case GST_ITERATOR_ERROR:
        g_assert_not_reached ();
        break;
    }
    g_value_reset (&padptr);
  }
  g_value_unset (&padptr);
  gst_iterator_free (pads);

  return ret;
}

static GstBuffer *
gst_nvdsmetamux_create_output (GstNvDsMetaMux * mux)
{
  GstNvStreamMemory *mem = nullptr;
  NvBufSurface *src_surf;
  GstMapInfo info = GST_MAP_INFO_INIT;
  GstBuffer *src_buf;
  GstBuffer *out_buf;

  src_buf =
      gst_aggregator_pad_pop_buffer (GST_AGGREGATOR_PAD (mux->active_pad));
  if (!src_buf) {
    GST_DEBUG_OBJECT (mux, "get active pad input buffer failed");
    return NULL;
  }

  nvds_set_input_system_timestamp (src_buf, GST_ELEMENT_NAME (mux));

  if (!gst_buffer_map (src_buf, &info, GST_MAP_READ)) {
    GST_ERROR_OBJECT (mux, "map input buffer failed");
    return NULL;
  }

  mem = g_slice_new0 (GstNvStreamMemory);

  mem->src_buf = src_buf;

  src_surf = (NvBufSurface *) info.data;
  memcpy (&mem->surf, src_surf, sizeof (NvBufSurface));

  gst_buffer_unmap (src_buf, &info);

  out_buf =
      gst_buffer_new_wrapped_full (GST_MEMORY_FLAG_READONLY, &mem->surf,
      sizeof (NvBufSurface), 0, sizeof (NvBufSurface), mem,
      shared_mem_buf_unref_callback);

  if (!gst_buffer_copy_into (out_buf, src_buf, GST_BUFFER_COPY_META, 0, -1)) {
    GST_DEBUG ("Buffer metadata copy failed \n");
  }

  GST_BUFFER_PTS (out_buf) = GST_BUFFER_PTS (src_buf);
  return out_buf;
}

static GstFlowReturn
gst_nvdsmetamux_aggregate (GstAggregator * aggregator, gboolean timeout)
{
  GstNvDsMetaMux *mux = GST_NVDSMETAMUX (aggregator);
  GstFlowReturn ret = GST_FLOW_OK;
  GstBuffer *buf = NULL;


  /* 1. Get output buffer which wrapped batched buffer from active pad.
   * 2. Search all meta data from all sink pads and add frame meta data for
   * the same frame.
   * 3. Filter meta data based on source ids and model ids.
   * 4. Push the batched buffer.
   * 5. Return gstbuffer if all frame PTS older than current PTS of the source.
   */

  GST_DEBUG_OBJECT (mux, "nvdsmetamux aggregate begin");

  if (gst_nvdsmetamux_are_all_pads_eos (mux)) {
    GST_DEBUG_OBJECT (mux, "nvdsmetamux aggregate send src pad eos event");
    gst_pad_push_event (mux->srcpad, gst_event_new_eos ());
  }

  buf = gst_nvdsmetamux_create_output (mux);
  if (!buf) {
    GST_DEBUG_OBJECT (mux, "create output buffer failed");
    if (gst_aggregator_pad_is_eos (GST_AGGREGATOR_PAD (mux->active_pad))) {
      gst_nvdsmetamux_clear_buflist(aggregator);
    }
    g_usleep (GST_MSECOND / 1000);
    return GST_AGGREGATOR_FLOW_NEED_DATA;
  }

  ret = gst_nvdsmetamux_merge_metadata (mux, buf);
  if (ret != GST_FLOW_OK) {
    GST_ERROR_OBJECT (mux, "merge meta data error");
    return ret;
  }

  ret = gst_nvdsmetamux_filter_metadata (mux, buf);
  if (ret != GST_FLOW_OK) {
    GST_ERROR_OBJECT (mux, "merge meta data error");
    return ret;
  }

  nvds_set_output_system_timestamp (buf, GST_ELEMENT_NAME (mux));

  GST_DEBUG_OBJECT (mux, "nvdsmetamux push output");
  ret = gst_nvdsmetamux_push (mux, buf);
  if (ret != GST_FLOW_OK) {
    GST_ERROR_OBJECT (mux, "push error");
    return ret;
  }

  GST_DEBUG_OBJECT (mux, "nvdsmetamux return buffers");
  ret = gst_nvdsmetamux_return_buffer (mux);
  if (ret != GST_FLOW_OK) {
    GST_ERROR_OBJECT (mux, "return error");
    return ret;
  }

  return GST_FLOW_OK;
}

static void
gst_nvdsmetamux_get_property (GObject * object,
    guint prop_id, GValue * value, GParamSpec * pspec)
{
  GstNvDsMetaMux *mux = GST_NVDSMETAMUX (object);

  switch (prop_id) {
    case PROP_ACTIVE_PAD:
      GST_OBJECT_LOCK (mux);
      g_value_set_string (value, mux->active_sink_pad);
      GST_OBJECT_UNLOCK (mux);
      break;
    case PROP_PTS_TOLERANCE:
      GST_OBJECT_LOCK (mux);
      g_value_set_int64 (value, mux->pts_tolerance);
      GST_OBJECT_UNLOCK (mux);
      break;
    case PROP_CONFIG_FILE:
      GST_OBJECT_LOCK (mux);
      g_value_set_string (value, mux->config_file_path);
      GST_OBJECT_UNLOCK (mux);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static void
gst_nvdsmetamux_set_property (GObject * object,
    guint prop_id, const GValue * value, GParamSpec * pspec)
{
  GstNvDsMetaMux *mux = GST_NVDSMETAMUX (object);

  switch (prop_id) {
    case PROP_ACTIVE_PAD:
      GST_OBJECT_LOCK (mux);
      g_free (mux->active_sink_pad);
      mux->active_sink_pad = g_value_dup_string (value);
      GST_OBJECT_UNLOCK (mux);
      break;
    case PROP_PTS_TOLERANCE:
      GST_OBJECT_LOCK (mux);
      mux->pts_tolerance = g_value_get_int64 (value);
      GST_OBJECT_UNLOCK (mux);
      break;
    case PROP_CONFIG_FILE:
    {
      GST_OBJECT_LOCK (mux);
      g_free (mux->config_file_path);
      mux->config_file_path = g_value_dup_string (value);
      mux->config_file_parse_successful =
          nvdsmetamux_parse_config_file (mux, mux->config_file_path);
      if (mux->config_file_parse_successful) {
        GST_DEBUG_OBJECT (mux, "Successfully Parsed Config file\n");
      }
      GST_OBJECT_UNLOCK (mux);
    }
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static GstClockTime
gst_nvdsmetamux_get_next_time (GstAggregator * aggregator)
{
  //GstNvDsMetaMux *mux = GST_NVDSMETAMUX (aggregator);

  //return gst_aggregator_simple_get_next_time (aggregator);
  return GST_CLOCK_TIME_NONE;
}

static GstFlowReturn
gst_nvdsmetamux_update_src_caps (GstAggregator * aggregator,
    GstCaps * caps, GstCaps ** ret)
{
  GstNvDsMetaMux *mux = GST_NVDSMETAMUX (aggregator);
  GstCaps *sinkcaps = NULL;
  GList *l;

  GST_OBJECT_LOCK (mux);
  for (l = GST_ELEMENT_CAST (mux)->sinkpads; l != NULL; l = l->next) {
    GstNvDsMetaMuxPad *pad = GST_NVDSMETAMUX_PAD (l->data);

    if (!mux->active_sink_pad
        || !strcmp (gst_pad_get_name (pad), mux->active_sink_pad)) {
      mux->active_pad = pad;
      sinkcaps = gst_pad_get_current_caps ((GstPad *) pad);
      break;
    }
  }
  GST_OBJECT_UNLOCK (mux);

  *ret = sinkcaps;

  return GST_FLOW_OK;
}

/**
 * Boiler plate for registering a plugin and an element.
 */
static gboolean
nvdsmetamux_plugin_init (GstPlugin * plugin)
{
  GST_DEBUG_CATEGORY_INIT (gst_nvdsmetamux_debug, "nvdsmetamux", 0,
      "metamux plugin");

  return gst_element_register (plugin, "nvdsmetamux", GST_RANK_PRIMARY,
      GST_TYPE_NVDSMETAMUX);
}

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    nvdsgst_metamux,
    DESCRIPTION, nvdsmetamux_plugin_init, "9.0", LICENSE, BINARY_PACKAGE,
    URL)
