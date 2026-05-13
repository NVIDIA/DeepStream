/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "gstnvstreamdemux.h"
#include "nvbufsurface.h"
#include <stdio.h>
#include <string.h>
#include <vector>
#include <tuple>
#include <iostream>
#include <chrono>
#include "gst-nvquery.h"
#include "gst-nvquery-internal.h"
#include "gstnvdsmeta.h"
#include "nvdsmeta.h"
#include "gst-nvevent.h"
#include "gstnvstreammuxdebug.h"
#include "gst/audio/audio-format.h"
#include "gst-nvmessage.h"

GST_DEBUG_CATEGORY_STATIC (gst_nvstreamdemux_debug);
#define GST_CAT_DEFAULT gst_nvstreamdemux_debug

#define _do_init \
    GST_DEBUG_CATEGORY_INIT (gst_nvstreamdemux_debug, "nvstreamdemux", 0, "nvstreamdemux element");
#define gst_nvstreamdemux_2_parent_class parent_class
G_DEFINE_TYPE_WITH_CODE (GstNvStreamDemux, gst_nvstreamdemux_2,
    GST_TYPE_ELEMENT, _do_init);

#define USE_CUDA_BATCH 1

#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wpointer-arith"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wswitch"
#pragma GCC diagnostic ignored "-Wunused-label"
#pragma GCC diagnostic ignored "-Wenum-compare"

#define COMMON_AUDIO_CAPS \
  "channels = " GST_AUDIO_CHANNELS_RANGE ", " \
  "rate = (int) [ 1, MAX ]"

static GstStaticPadTemplate nvstreamdemux_sinkpad_template =
GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES ("memory:NVMM",
            "{ " "NV12, RGBA, I420 }") "; "
            "audio/x-raw(memory:NVMM), "
            "format = { " "S16LE, F32LE }, "
            "layout = (string) interleaved, "
            COMMON_AUDIO_CAPS
        ));

static GstStaticPadTemplate nvstreamdemux_srcpad_template =
GST_STATIC_PAD_TEMPLATE ("src_%u",
    GST_PAD_SRC,
    GST_PAD_REQUEST,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES ("memory:NVMM",
            "{ " "NV12, RGBA, I420 }") "; "
            "audio/x-raw, "
            "format = { " "S16LE, F32LE }, "
            "layout = (string) interleaved, "
            COMMON_AUDIO_CAPS
        ));

static GQuark dsmeta_quark = 0;

static void
send_stream_start_if_not_already_sent(GstNvStreamDemux * nvstreamdemux,
        gint stream_index, gchar* stream_id, GstPad *src_pad);

static gboolean
is_stream_start_sent(GstNvStreamDemux * nvstreamdemux,
        gint stream_index);

static gboolean gst_nvstreamdemux_is_video(GstCaps* caps) {
  GstStructure* caps_str = gst_caps_get_structure (caps, 0);
  const gchar* mimetype = gst_structure_get_name (caps_str);

  /** If mimetype is audio/x-raw, extract useful information and return */
  if(strcmp(mimetype, "video/x-raw") == 0) {
    return TRUE;
  }
  return FALSE;
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


  if (!name || sscanf (name, "src_%u", &stream_index) < 1) {
    GST_ERROR_OBJECT (element,
        "Pad should be named 'src_%%u' when requesting a pad");
    return NULL;
  }

  LOGD("requesting demux src_%d\n", stream_index);
  GST_DEBUG_OBJECT (element, "requesting demux src_%d\n", stream_index);

  g_mutex_lock (&nvstreamdemux->ctx_lock);
  if (g_hash_table_contains (nvstreamdemux->pad_indexes, stream_index + (char *)NULL)) {
    GST_ERROR_OBJECT (element, "Pad named '%s' already requested", name);
    return NULL;
  }

  srcpad = GST_PAD_CAST (g_object_new (GST_TYPE_PAD,
          "name", name, "direction", templ->direction, "template", templ,
          NULL));

  LOGD("adding demux src_%d\n", stream_index);
  GST_DEBUG_OBJECT (element, "adding demux src_%d\n", stream_index);
  g_hash_table_insert (nvstreamdemux->pad_indexes, stream_index + (char *)NULL, srcpad);

  gst_pad_activate_mode (srcpad, GST_PAD_MODE_PUSH, TRUE);
  gst_element_add_pad (element, srcpad);

  gst_pad_set_query_function (srcpad,
      GST_DEBUG_FUNCPTR (gst_nvstreamdemux_src_query));

  gst_pad_use_fixed_caps (srcpad);

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
  if( GPOINTER_TO_INT(g_hash_table_lookup (nvstreamdemux->eos_flag,stream_index + (char *)NULL)) != 1 )
  {
    GST_ERROR_OBJECT (nvstreamdemux,"Demuxer EOS not received, release_pad cannot be called\n");
  }

  GST_DEBUG_OBJECT (element, "removing demux src_%d\n", stream_index);
  LOGD("removing demux src_%d\n", stream_index);
  g_mutex_lock (&nvstreamdemux->ctx_lock);
  g_hash_table_remove (nvstreamdemux->pad_indexes, stream_index + (char *)NULL);
  g_hash_table_remove (nvstreamdemux->eos_flag, stream_index + (char *)NULL);

  gst_pad_set_active (pad, FALSE);
  gst_element_remove_pad (GST_ELEMENT_CAST (nvstreamdemux), pad);
  g_mutex_unlock (&nvstreamdemux->ctx_lock);
}

typedef struct
{
  NvBufSurface surf;
  GstBuffer *src_buffer;
} GstNvStreamMemory;

static void
shared_mem_buf_unref_callback (gpointer data)
{
  GstNvStreamMemory *mem = (GstNvStreamMemory *) data;

  gst_buffer_unref (mem->src_buffer);
  g_slice_free (GstNvStreamMemory, mem);
}

static void
shared_audio_buf_unref_callback (gpointer data)
{
  GstBuffer* buffer = (GstBuffer *) data;
  gst_buffer_unref (buffer);
}

static void copy_user_meta(NvDsUserMeta *src_user_meta, NvDsUserMeta *dst_user_meta)
{
  NvDsBaseMeta *src_base_meta = (NvDsBaseMeta *)src_user_meta;
  NvDsBaseMeta *dst_base_meta = (NvDsBaseMeta *)dst_user_meta;
  dst_base_meta->meta_type = src_base_meta->meta_type;
  dst_base_meta->copy_func = src_base_meta->copy_func;
  dst_base_meta->release_func = src_base_meta->release_func;
  dst_base_meta->uContext = src_base_meta->uContext;
  dst_user_meta->user_meta_data = src_base_meta->copy_func(src_user_meta, NULL);
}

static gboolean
move_gst_meta(GstBuffer *out_buf, NvDsBatchMeta* batch_meta, gpointer frame_meta, gboolean is_audio)
{
    /** check if the frame_meta->user_meta_list have NVDS_BUFFER_GST_AS_FRAME_USER_META
     * If so, copy it to the output GstBuffer and delete the entry from user_meta_list
     */
    NvDsMetaList* l_user_meta = nullptr;
    NvDsUserMeta* user_meta_to_delete = nullptr;

    NvDsUserMetaList* frame_user_meta_list = nullptr;

    if(!is_audio)
    {
        frame_user_meta_list = ((NvDsFrameMeta*)frame_meta)->frame_user_meta_list;
    }
    else
    {
        frame_user_meta_list = ((NvDsAudioFrameMeta*)frame_meta)->frame_user_meta_list;
    }

    for (l_user_meta = frame_user_meta_list; l_user_meta != NULL;
      l_user_meta = l_user_meta->next) {
      NvDsUserMeta* user_meta = (NvDsUserMeta *) (l_user_meta->data);

      if(user_meta->base_meta.meta_type == NVDS_BUFFER_GST_AS_FRAME_USER_META)
      {
          GstBuffer* meta_buffer = (GstBuffer*)user_meta->user_meta_data;
          gpointer state = NULL;
          GstMeta *gst_meta = NULL;

          /** copy all the propagated Gst Meta to out_buf */
          while ((gst_meta = gst_buffer_iterate_meta (meta_buffer, &state)))
          {
            GstMetaTransformCopy copy_data = { FALSE, 0, (gsize)-1 };
            /* simply copy *meta from inbuf into outbuf */
            gst_meta->info->transform_func (out_buf, gst_meta, meta_buffer,
                _gst_meta_transform_copy, &copy_data);

          }
          /** break here to delete the user_meta entry;
           * caller may invoke this API again to remove next instance */
          user_meta_to_delete = user_meta;
          break;
      }
    }
    if(user_meta_to_delete)
    {
      /** Now delete the l_user_meta as its no longer required */
      if(!is_audio)
      {
        nvds_remove_user_meta_from_frame((NvDsFrameMeta*)frame_meta, user_meta_to_delete);
      }
      else
      {
        nvds_remove_user_meta_from_audio_frame((NvDsAudioFrameMeta*)frame_meta, user_meta_to_delete);
      }
      /** deleted one NVDS_BUFFER_GST_AS_FRAME_USER_META; there might be more */
      return TRUE;
    }
    /** no more NVDS_BUFFER_GST_AS_FRAME_USER_META in frame_meta->user_meta_list */
    return FALSE;
}

static GstBuffer *
create_shared_mem_buf (GstNvStreamDemux *demux, GstBuffer * src_buffer,
    NvDsBatchMeta *src_batch_meta,
    guint index, guint demuxIndex, gboolean is_raw)
{
  GstNvStreamMemory *mem;
  NvBufSurface *src_surf;
  GstMapInfo info = GST_MAP_INFO_INIT;
  GstBuffer *out_buf;
  GstMeta *rem_meta_list[128];
  guint num_rem_meta = 0;
  guint k, frame_number=0;
  NvDsFrameMeta *frame_meta = nvds_get_nth_frame_meta (src_batch_meta->frame_meta_list, index);
  guint stream_id = frame_meta->pad_index;

  if (!gst_buffer_map (src_buffer, &info, GST_MAP_READ)) {
    return NULL;
  }

  mem = g_slice_new0 (GstNvStreamMemory);

  mem->src_buffer = src_buffer;

  src_surf = (NvBufSurface *) info.data;
  memcpy (&mem->surf, src_surf, sizeof (NvBufSurface));
  mem->surf.numFilled = mem->surf.batchSize = demux->num_surfaces_per_frame;
  mem->surf.surfaceList = src_surf->surfaceList + index;

  gst_buffer_unmap (src_buffer, &info);

  gst_buffer_ref (src_buffer);

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
  //batch_meta->unique_id = 0; /*TODO*/


  for (k = 0; k < demux->num_surfaces_per_frame; k++) {
    NvDsFrameMeta *frame_meta = nvds_acquire_frame_meta_from_pool(batch_meta);
    NvDsFrameMeta *src_frame_meta = nvds_get_nth_frame_meta(src_batch_meta->frame_meta_list, index + k);
    nvds_copy_frame_meta(src_frame_meta, frame_meta);
    frame_meta->batch_id = k;
    nvds_add_frame_meta_to_batch(batch_meta, frame_meta);

    gboolean have_more_user_meta = FALSE;
    do {
        have_more_user_meta = move_gst_meta(out_buf, batch_meta, (gpointer)frame_meta, FALSE);
    } while(have_more_user_meta);

    if(frame_number==0) {
     /* for dewarper case, there will be only one latency meta for all surfaces */
     frame_number=src_frame_meta->frame_num;
    }
  }

  for(GList* nodeUserMeta = src_batch_meta->batch_user_meta_list; nodeUserMeta; nodeUserMeta = g_list_next(nodeUserMeta))
  {
        NvDsUserMeta* src_user_meta = static_cast<NvDsUserMeta*>(nodeUserMeta->data);
        if(src_user_meta->base_meta.meta_type == NVDS_LATENCY_MEASUREMENT_META)
        {
           NvDsMetaCompLatency* latency_metadata = (NvDsMetaCompLatency *)(src_user_meta->user_meta_data);
           if(latency_metadata->source_id == stream_id)
           {
               if(latency_metadata->frame_num==frame_number)
               {
                 NvDsUserMeta* dst_user_meta = nvds_acquire_user_meta_from_pool (batch_meta);
                 copy_user_meta(src_user_meta, dst_user_meta);
                 nvds_add_user_meta_to_batch(batch_meta, dst_user_meta);
               }
           }
        }
  }

  return out_buf;
}


static GstFlowReturn
gst_nvstreamdemux_sink_chain_audio_batch (GstNvStreamDemux *nvstreamdemux,
        GstBuffer * buffer)
{
  GstMapInfo info = GST_MAP_INFO_INIT;
  NvBufAudio *src_surf;
  std::vector <std::tuple<GstPad*, GstBuffer*, gint>> data ;
  NvDsBatchMeta* src_batch_meta = gst_buffer_get_nvds_batch_meta(buffer);

  GstMeta *gst_meta = NULL;
  gpointer state = NULL;
  NvDsMeta *dsmeta = NULL;
  guint k, frame_number=0;

  if(!src_batch_meta) {
    GST_WARNING_OBJECT (nvstreamdemux, "received input audio batch without NvDsBatchMeta; "
       "a plugin post nvstreammux and before nvstreamdemux might be stripping "
       "this meta off; FATAL\n");
    return GST_FLOW_ERROR;
  }

  if (!gst_buffer_map (buffer, &info, GST_MAP_READ)) {
    return GST_FLOW_ERROR;
  }

  NvBufAudio* audioBatch = (NvBufAudio*) info.data;

  gst_buffer_unmap (buffer, &info);

  for(uint32_t i = 0; i < audioBatch->numFilled; i++) {
    guint stream_id = audioBatch->audioBuffers[i].sourceId;
    GstBuffer *buf;
    LOGD("buffer sourceId=%d\n", audioBatch->audioBuffers[i].sourceId);

    g_mutex_lock (&nvstreamdemux->ctx_lock);
    GstPad* src_pad =
        GST_PAD (g_hash_table_lookup (nvstreamdemux->pad_indexes,
            stream_id + (char *)NULL));
    g_mutex_unlock (&nvstreamdemux->ctx_lock);

    if (G_UNLIKELY(!src_pad || !gst_pad_is_linked (src_pad))) {
      GST_WARNING_OBJECT (nvstreamdemux, "Pushing of buffer skipped for stream %d; "
        "caps negotiation appear unsuccessful; src_pad=%p has_current_caps=%d is_linked=%d\n",
        audioBatch->audioBuffers[i].sourceId, src_pad,
        src_pad ? gst_pad_has_current_caps (src_pad) : 0, src_pad ? gst_pad_is_linked (src_pad) : 0);
      LOGD("has_current_caps=%d is_linked=%d\n",
        src_pad ? gst_pad_has_current_caps (src_pad) : 0, src_pad ? gst_pad_is_linked (src_pad) : 0);
      continue;
    }

    gst_buffer_ref (buffer);

    LOGD("pushing buffer sourceId=%d\n", stream_id);
    /** assuming NvBufAudio is always with memType SYS_MEM;
     * TODO: Add and use NvBufAudio utility APIs to copy */
    GstBuffer* out_buf =
        gst_buffer_new_wrapped_full (GST_MEMORY_FLAG_READONLY, audioBatch->audioBuffers[i].dataPtr,  audioBatch->audioBuffers[i].dataSize,
        0,  audioBatch->audioBuffers[i].dataSize, buffer, shared_audio_buf_unref_callback);
    GST_BUFFER_PTS(out_buf) = audioBatch->audioBuffers[i].bufPts;
    GST_BUFFER_DURATION(out_buf) = audioBatch->audioBuffers[i].duration;

    /** attach NvDsBatchMeta */
    NvDsBatchMeta *batch_meta = nvds_create_audio_batch_meta(nvstreamdemux->num_surfaces_per_frame);
    if(batch_meta == NULL) {
      return GST_FLOW_ERROR;
    }
    NvDsMeta *meta = gst_buffer_add_nvds_meta (out_buf, batch_meta, NULL,
      nvds_audio_batch_meta_copy_func, nvds_audio_batch_meta_release_func);

    meta->meta_type = NVDS_BATCH_GST_META;

    batch_meta->base_meta.batch_meta = batch_meta;
    batch_meta->base_meta.copy_func = nvds_audio_batch_meta_copy_func;
    batch_meta->base_meta.release_func = nvds_audio_batch_meta_release_func;
    batch_meta->max_frames_in_batch = nvstreamdemux->num_surfaces_per_frame;
    //batch_meta->unique_id = 0; /*TODO*/

    for (int k = 0; k < nvstreamdemux->num_surfaces_per_frame; k++) {
      NvDsAudioFrameMeta *frame_meta = nvds_acquire_audio_frame_meta_from_pool(batch_meta);
      NvDsAudioFrameMeta *src_frame_meta = nvds_get_nth_audio_frame_meta(src_batch_meta->frame_meta_list, i + k);
      nvds_copy_audio_frame_meta(src_frame_meta, frame_meta);
      frame_meta->batch_id = k;
      nvds_add_audio_frame_meta_to_audio_batch(batch_meta, frame_meta);
      gboolean have_more_user_meta = FALSE;
      do {
        have_more_user_meta = move_gst_meta(out_buf, batch_meta, (gpointer)frame_meta, TRUE);
      } while(have_more_user_meta);
      if(frame_number==0) {
        /* there will be only one latency meta for all surfaces */
        frame_number=src_frame_meta->frame_num;
      }
    }

    for(GList* nodeUserMeta = src_batch_meta->batch_user_meta_list; nodeUserMeta; nodeUserMeta = g_list_next(nodeUserMeta))
    {
      NvDsUserMeta* src_user_meta = static_cast<NvDsUserMeta*>(nodeUserMeta->data);
      if(src_user_meta->base_meta.meta_type == NVDS_LATENCY_MEASUREMENT_META)
      {
         NvDsMetaCompLatency* latency_metadata = (NvDsMetaCompLatency *)(src_user_meta->user_meta_data);
         if(latency_metadata->source_id == stream_id)
         {
             if(latency_metadata->frame_num==frame_number)
             {
                NvDsUserMeta* dst_user_meta = nvds_acquire_user_meta_from_pool (batch_meta);
                copy_user_meta(src_user_meta, dst_user_meta);
                nvds_add_user_meta_to_batch(batch_meta, dst_user_meta);
             }
         }
      }
    }
    data.emplace_back(src_pad, out_buf, stream_id);
  }


  for (const auto &item : data) {
    GstFlowReturn ret = gst_pad_push (std::get<0>(item), std::get<1>(item));
    LOGD("pushed buffer sourceId=%d ret=%d\n", std::get<2>(item), ret);
    if(G_UNLIKELY(ret != GST_FLOW_OK)) {
       GST_WARNING_OBJECT (nvstreamdemux, "Pushing of buffer failed for stream %d with error %d",std::get<2>(item), ret);
       if (ret == GST_FLOW_ERROR){
         GST_ERROR_OBJECT (nvstreamdemux, "Pushing of buffer failed for stream %d with error %d",std::get<2>(item), ret);
         return ret;
       }
    }
  }

  gst_buffer_unref (buffer);

  return GST_FLOW_OK;
}

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

  if(nvstreamdemux->isAudio) {
    return gst_nvstreamdemux_sink_chain_audio_batch(nvstreamdemux, buffer);
  }

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

    g_mutex_lock (&nvstreamdemux->ctx_lock);
    src_pad =
        GST_PAD (g_hash_table_lookup (nvstreamdemux->pad_indexes,
            stream_id + (char *)NULL));
    g_mutex_unlock (&nvstreamdemux->ctx_lock);
    if (!src_pad || !gst_pad_has_current_caps (src_pad) || !gst_pad_is_linked (src_pad)) {
      continue;
    }

    buf =
        create_shared_mem_buf (nvstreamdemux, buffer, batch_meta, i, stream_id,
        g_hash_table_lookup (nvstreamdemux->pad_caps_is_raw, stream_id + (char *)NULL) == NULL);
    GST_BUFFER_PTS (buf) = frame_meta->buf_pts;

    ret = gst_pad_push (src_pad, buf);
    if (ret == GST_FLOW_NOT_LINKED)
      ret = GST_FLOW_OK;
  }

  gst_buffer_unref (buffer);

  return ret;
}

static gboolean
set_src_pad_caps (GstNvStreamDemux * nvstreamdemux, gint index, gint width_val, gint height_val, gchar* stream_id = NULL)
{
  GList *keys;
  GList key = { index + (char *)NULL, NULL, NULL };
  gboolean ret = TRUE;
  gint batch_size = 0;
  GValue batchsize = G_VALUE_INIT;
  GstStructure *new_caps_s;
  gchar *caps_str;

  if (index < 0 && nvstreamdemux->isAudio != TRUE)
    return TRUE;
  
  if (index > -1)
    keys = &key;
  else
    keys = g_hash_table_get_keys (nvstreamdemux->pad_indexes);

  LOGD("index=%d keys=%p\n", index, keys);
  while (keys && ret) {
    gpointer stream_ptr = keys->data;
    GstPad *pad =
        GST_PAD (g_hash_table_lookup (nvstreamdemux->pad_indexes, stream_ptr));
    GValue *frame_rate =
        (GValue *) g_hash_table_lookup (nvstreamdemux->pad_framerates,
        stream_ptr);
    GstCaps *new_caps = NULL;
    GstEvent *event;
    GstCaps *other_caps = NULL;
    GstCapsFeatures *features = gst_caps_features_from_string ("memory:NVMM");

    keys = keys->next;

    LOGD("pad=%p\n", pad);
    if (!pad)
      continue;

    /** query the muxer for this stream's caps and use that if available */
    if(GST_ELEMENT_CAST (nvstreamdemux)->sinkpads) {

      if (nvstreamdemux->sink_caps) {
        new_caps = gst_caps_copy (nvstreamdemux->sink_caps);
        if(!gst_nvstreamdemux_is_video(new_caps)) {
          nvstreamdemux->isAudio = TRUE;
        }

        new_caps_s = gst_caps_get_structure (new_caps, 0);
        GST_DEBUG_OBJECT (nvstreamdemux, "caps before = %s\n", gst_caps_to_string (new_caps));
        if (nvstreamdemux->isAudio == TRUE)
        {
            GstCapsFeatures* ft;
            guint n,i;
            n = gst_caps_get_size (new_caps);
            for (i=0; i<n; i++)
            {
                ft = gst_caps_get_features (new_caps, i);
                if (gst_caps_features_get_size (ft))
                {
#define GST_CAPS_FEATURE_MEMORY_NVMM "memory:NVMM"
                    gst_caps_features_remove (ft, GST_CAPS_FEATURE_MEMORY_NVMM);
                }
            }
        }
       else if(index > -1)
       {
            GstCapsFeatures* ft;
            guint n,i;
            GValue wd=G_VALUE_INIT, ht=G_VALUE_INIT;
           GValue batchsize = G_VALUE_INIT;

            g_value_init (&wd, G_TYPE_INT);
            g_value_init (&ht, G_TYPE_INT);
            g_value_init (&batchsize, G_TYPE_UINT);

            g_value_set_int (&wd, width_val);
            g_value_set_int (&ht, height_val);
            g_value_set_uint (&batchsize, (uint)1);


            n = gst_caps_get_size (new_caps);
            for (i=0; i<n; i++)
            {
		if (gst_structure_has_field (new_caps_s, "width"))
		{
		  gst_structure_set_value(new_caps_s, "width", (GValue *)&wd);
		}
		if (gst_structure_has_field (new_caps_s, "height"))
		{
		  gst_structure_set_value(new_caps_s, "height",(GValue *)&ht);
		}
		if (gst_structure_has_field (new_caps_s, "batch-size"))
		{
		  gst_structure_set_value(new_caps_s, "batch-size",(GValue *)&batchsize);
		}
            }
        }

        GST_DEBUG_OBJECT (nvstreamdemux, "caps after = %s\n", gst_caps_to_string (new_caps));

        if (gst_caps_get_size(new_caps) > 0 && gst_caps_get_features (new_caps, 0) &&
          gst_caps_features_is_equal (gst_caps_get_features (new_caps, 0),
            features)) {

          g_hash_table_insert (nvstreamdemux->pad_caps_is_raw, stream_ptr,
            NULL + (char *)1);
        }
        LOGD("procured caps from upstream mux for stream %llu; caps=[%s]\n",
                (guint64)stream_ptr,
                gst_caps_to_string(new_caps));

        /** push STREAM_START event before caps if STREAM_START was not
         * already sent */
        if(stream_id) {
          LOGD("sending stream-start event on pad %d stream_id=%s\n", index, stream_id);
          send_stream_start_if_not_already_sent(nvstreamdemux, index, stream_id, pad);
        }
        LOGD("DEBUGME stream_id=%s new_caps=%s\n", stream_id, gst_caps_to_string(new_caps));
        event = gst_event_new_caps (new_caps);
        /** Push the caps event even when the index is == -1
         * with STREAM_START (unique ID generated in nvstreamdemux) going before caps
         * This allows for us to use muxer's output caps
         * as demuxer's src caps (ALL streams)
         * Note: This puts a restriction that all incoming caps at nvstreammux
         * must be the same. Known limitation: JIRA DSNEX-1043.
         */
        {
            /** Documentation from STREAM_START:
             * A new stream-id should only be created for a stream if the upstream
             * stream is split into (potentially) multiple new streams, e.g. in a demuxer,
             * but not for every single element in the pipeline.
             * We try in nvstreamdemux to reuse the stream-id originally set by
             * the source plugins - however, in certain cases,
             * caps event from nvstreammux flows into demux first and then
             * STREAM_START comes in (in which case the later will be discarded
             * and former will be handled for caps and a stream-start event will
             * be sent downstream before caps with new stream-id as done here.
             * nvstreamdemux here, is responsible for creating unique stream-id)
             */
            char unique_stream_id[64];
            int n = snprintf(unique_stream_id, 64, "unique-stream-id-%ld", (gint64)stream_ptr);
            unique_stream_id[n] = '\0';
            LOGD("pushing stream start event downstream [unique_stream_id=%s]\n", unique_stream_id);
            send_stream_start_if_not_already_sent(nvstreamdemux, (gint64)stream_ptr, unique_stream_id, pad);
            LOGD("pushing caps event downstream\n");
            ret = gst_pad_push_event (pad, event);
        }
        gst_caps_unref (new_caps);

      }
    }
  }

  return ret;
}

static gboolean
gst_nvstreamdemux_sink_event (GstPad * pad, GstObject * parent,
    GstEvent * event)
{
  GstNvStreamDemux *nvstreamdemux = GST_NVSTREAMDEMUX (parent);
  GstElement *element = GST_ELEMENT (parent);

  if (GST_EVENT_TYPE (event) == GST_EVENT_CAPS) {
    LOGD("incoming caps\n");
    GstCaps *caps = NULL;
    GstQuery *query;
    gboolean ret;

    query = gst_nvquery_num_surfaces_per_buffer_new ();
    if (gst_pad_peer_query (nvstreamdemux->sinkpad, query)) {
      gst_nvquery_num_surfaces_per_buffer_parse (query, &nvstreamdemux->num_surfaces_per_frame);
    } else {
      nvstreamdemux->num_surfaces_per_frame = 1;
    }
    gst_query_unref (query);

    gst_event_parse_caps (event, &caps);
    if (caps) {
      if(!gst_nvstreamdemux_is_video(caps))
          nvstreamdemux->isAudio = TRUE;
      LOGD("incoming sink caps = [%s]\n", gst_caps_to_string(caps));
      gst_caps_replace (&nvstreamdemux->sink_caps, caps);
      GstStructure *str = gst_caps_get_structure (caps, 0);
      gint num_surfaces_per_frame;
      if (gst_structure_get_int (str, "num-surfaces-per-frame", &num_surfaces_per_frame)) {
        nvstreamdemux->num_surfaces_per_frame = num_surfaces_per_frame;
      }
    }
    gst_event_unref (event);
    g_mutex_lock (&nvstreamdemux->ctx_lock);
    ret = set_src_pad_caps (nvstreamdemux, -1, 0, 0);
    g_mutex_unlock (&nvstreamdemux->ctx_lock);

    return ret;
  }

  if (GST_EVENT_TYPE (event) == GST_NVEVENT_UPDATE_CAPS) {
	  const GstStructure *const str = gst_event_get_structure(event);
	  LOGD("Got update-caps event\n");
	  GstCaps *new_caps;
	  GstStructure *new_caps_str;
	  guint stream_index = 0;
	  guint width_val=0;
	  guint height_val=0;

	  gchar* stream_id = NULL;
	  const GValue *frame_rate = NULL;
	  GValue *fr = (GValue *)g_malloc0 (sizeof (GValue));
	  gboolean ret;

	  gst_structure_get_uint (str, "stream-id", &stream_index);
	  stream_id = (gchar*)gst_structure_get_string (str, "stream-id-str");
	  gst_structure_get_uint (str, "width-val", &width_val);
	  gst_structure_get_uint (str, "height-val", &height_val);

	  frame_rate = gst_structure_get_value (str, "frame-rate");
	  if(frame_rate) {
		  g_value_init (fr, GST_TYPE_FRACTION);
		  g_value_copy (frame_rate, fr);
	  }

	  //GST_OBJECT_LOCK (nvstreamdemux);

	  g_mutex_lock (&nvstreamdemux->ctx_lock);
	  g_hash_table_insert (nvstreamdemux->pad_framerates, stream_index + (char *)NULL,
			  fr);
	  ret = set_src_pad_caps (nvstreamdemux, stream_index, width_val, height_val, stream_id);
	  g_mutex_unlock (&nvstreamdemux->ctx_lock);
	  //GST_OBJECT_UNLOCK (nvstreamdemux);

	  return ret;
  }

  if (GST_EVENT_TYPE (event) ==  GST_NVEVENT_STREAM_EOS) {
    guint source_id = 0;
    GstPad *src_pad = NULL;
    gst_nvevent_parse_stream_eos (event, &source_id);
    LOGD("sending eos event on pad %d\n", source_id);
    g_mutex_lock (&nvstreamdemux->ctx_lock);
    g_hash_table_insert (nvstreamdemux->eos_flag, source_id + (char *)NULL, GINT_TO_POINTER(1));
    g_mutex_unlock (&nvstreamdemux->ctx_lock);
    src_pad =
      GST_PAD (g_hash_table_lookup (nvstreamdemux->pad_indexes,
            source_id + (char *)NULL));
    if (!src_pad) {
      return TRUE;
    }
    LOGD("sending eos event on pad %d\n", source_id);
    gboolean ret = gst_pad_push_event (src_pad, gst_event_new_eos ());
    LOGD("eos send ret=%d\n", ret);
    return ret;
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
                    LOGD ("Got EOS from stream %d\n", stream_id);
      }

      src_pad = GST_PAD (g_hash_table_lookup (nvstreamdemux->pad_indexes,
            stream_id + (char *)NULL));
      if (!src_pad)
      {
        return TRUE;
      }
      ret = gst_pad_push_event (src_pad, event);
      LOGD ("return value from pad push event %d\n", ret);
      gst_message_unref(msg);
      return ret;
    }
    gst_message_unref(msg);
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
    if (!src_pad) {
      return TRUE;
    }

    LOGD("sending segment event on pad %d\n", source_id);
    return gst_pad_push_event (src_pad, gst_event_new_segment (segment));
  }

  if ((GST_EVENT_TYPE (event) == GST_NVEVENT_PAD_ADDED) ||
      (GST_EVENT_TYPE (event) == GST_NVEVENT_PAD_DELETED)) {
      gst_event_unref (event);
      return TRUE;
  }

  if (GST_EVENT_TYPE (event) == GST_NVEVENT_STREAM_START) {
    LOGD("got STREAM_START in demux\n");
    GstPad *src_pad = NULL;
    gchar* stream_id = NULL;
    guint source_id = 0;
    gst_nvevent_parse_stream_start (event, &source_id, &stream_id);
    LOGD("sending stream-start event on pad %d stream_id=%s\n", source_id, stream_id);
    gst_event_unref (event);
    src_pad =
      GST_PAD (g_hash_table_lookup (nvstreamdemux->pad_indexes,
            source_id + (char *)NULL));
    if (!src_pad) {
      return TRUE;
    }

    LOGD("sending stream-start event on pad %d stream_id=%s\n", source_id, stream_id);
    send_stream_start_if_not_already_sent(nvstreamdemux, source_id, stream_id, src_pad);
    return TRUE;
  }

  if(GST_EVENT_TYPE (event) == GST_EVENT_SEGMENT
         || GST_EVENT_TYPE (event) == GST_EVENT_STREAM_START
         || GST_EVENT_TYPE (event) == GST_EVENT_EOS
        ) {
    /**
     * SEGMENT:
     *   ignore plain segment event as we do get segment event with
     *   source ID associated as GST_NVEVENT_STREAM_SEGMENT from mux
     * STREAM_START: handling STREAM_START with GST_NVEVENT_STREAM_START
     *  (have source_id info) alone and hence ignore here.
     * EOS:
     *   We can only handle EOS in a stream-specific manner.
     *   handling EOS with GST_NVEVENT_STREAM_EOS (have source_id info) alone
     *   and hence ignoring EVENT_EOS here.
     */
    LOGD("ignoring redundant event\n");
    gst_event_unref (event);
    return TRUE;
  }

  LOGD("handling event type=%d(%s)\n",
        GST_EVENT_TYPE(event), gst_event_type_get_name(GST_EVENT_TYPE(event)));

  if (GST_EVENT_TYPE (event) == GST_EVENT_TAG) {
      gst_event_unref (event);
      return TRUE;
  } else {
      g_mutex_lock (&nvstreamdemux->ctx_lock);
      gboolean ret =  gst_pad_event_default (pad, parent, event);
      g_mutex_unlock (&nvstreamdemux->ctx_lock);
      return ret;
  }
}

static gboolean
gst_nvstreamdemux_sink_query (GstPad * pad, GstObject * parent,
    GstQuery * query)
{
  LOGD("DEBUGME\n");
  GstNvStreamDemux *nvstreamdemux = GST_NVSTREAMDEMUX (parent);
  GstElement *element = GST_ELEMENT (parent);

  if (gst_nvquery_is_batch_size (query)) {
    return FALSE;
  }

  return gst_pad_query_default (pad, parent, query);
}

static void
gst_nvstreamdemux_2_class_init (GstNvStreamDemuxClass * klass)
{
  GstElementClass *gstelement_class = GST_ELEMENT_CLASS (klass);

  gst_element_class_set_static_metadata (gstelement_class,
      "Stream demultiplexer 2", "Generic", "1-to-N pipes stream demultiplexing",
      "NVIDIA Corporation. Post on Deepstream for Tesla forum for any queries "
      "@ https://devtalk.nvidia.com/default/board/209/");

  gst_element_class_add_static_pad_template (gstelement_class,
      &nvstreamdemux_sinkpad_template);
  gst_element_class_add_static_pad_template (gstelement_class,
      &nvstreamdemux_srcpad_template);

  gstelement_class->request_new_pad =
      GST_DEBUG_FUNCPTR (gst_nvstreamdemux_request_new_pad);
  gstelement_class->release_pad =
      GST_DEBUG_FUNCPTR (gst_nvstreamdemux_release_pad);
}

static void
gst_nvstreamdemux_2_init (GstNvStreamDemux * nvstreamdemux)
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
  nvstreamdemux->pad_framerates = g_hash_table_new (NULL, NULL);
  nvstreamdemux->pad_caps_is_raw = g_hash_table_new (NULL, NULL);
  nvstreamdemux->pad_stream_start_sent = g_hash_table_new (NULL, NULL);
  nvstreamdemux->eos_flag = g_hash_table_new (NULL, NULL);
  nvstreamdemux->isAudio = FALSE;
  nvstreamdemux->sink_caps = NULL;
  nvstreamdemux->num_surfaces_per_frame = 1;

  g_mutex_init (&nvstreamdemux->ctx_lock);
  if (!dsmeta_quark)
    dsmeta_quark = g_quark_from_static_string (NVDS_META_STRING);
}

static void
send_stream_start_if_not_already_sent(GstNvStreamDemux * nvstreamdemux,
        gint stream_index, gchar* stream_id, GstPad *src_pad) {
  gboolean ret = TRUE;
  if (is_stream_start_sent(nvstreamdemux, stream_index)) {
    GST_DEBUG_OBJECT (nvstreamdemux, "STREAM_START already sent; ignoring new request\n");
    return;
  }
  LOGD("stream-start was not sent; sending now [%d]\n", stream_index);
  ret = gst_pad_push_event (src_pad, gst_event_new_stream_start (stream_id));
  if(ret == TRUE) {
    g_hash_table_insert (nvstreamdemux->pad_stream_start_sent, stream_index + (char *)NULL, (void*)1);
  }
}


static gboolean
is_stream_start_sent(GstNvStreamDemux * nvstreamdemux,
        gint stream_index)
{
  if (g_hash_table_contains (nvstreamdemux->pad_stream_start_sent, stream_index + (char *)NULL)) {
    GST_DEBUG_OBJECT (nvstreamdemux, "STREAM_START already sent; ignoring new request\n");
    return TRUE;
  }
  return FALSE;
}
