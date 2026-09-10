/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "gstnvtiler.h"
#include "gstnvdsbufferpool.h"
#include <stdio.h>
#include <string.h>
#include <npp.h>
#include <math.h>
#include "gst-nvevent.h"
#include "gst-nvquery.h"
#include "gst-nvcommon.h"
#include "nvtx_helper.h"
#include "NvTiler.h"
#include "nvdsmeta_internal.h"

#include "nvdsmeta_internal.h"

GST_DEBUG_CATEGORY_STATIC (gst_nvmultistreamtiler_debug);
#define GST_CAT_DEFAULT gst_nvmultistreamtiler_debug

#define _do_init \
    GST_DEBUG_CATEGORY_INIT (gst_nvmultistreamtiler_debug, "nvmultistreamtiler", 0, "nvmultistreamtiler element");
#define gst_nvmultistreamtiler_parent_class parent_class
G_DEFINE_TYPE_WITH_CODE (GstNvMultiStreamTiler, gst_nvmultistreamtiler,
    GST_TYPE_BASE_TRANSFORM, _do_init);

#define ENABLE_LAST_FRAME_BACKUP 1

static GstStaticPadTemplate nvmultistreamtiler_sinkpad_template =
GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES ("memory:NVMM",
            "{ " "NV12, RGBA, I420 }")));

static GstStaticPadTemplate nvmultistreamtiler_srcpad_template =
GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES ("memory:NVMM",
            "{ " "NV12, RGBA, I420 }")));

static GQuark dsmeta_quark = 0;

//#define LOGD_ENABLE
#ifdef LOGD_ENABLE
#define LOGD(...) printf(__VA_ARGS__)
#else
#define LOGD(...)
#endif

enum
{
  PROP_0,
  PROP_COLMUNS,
  PROP_ROWS,
  PROP_WIDTH,
  PROP_HEIGHT,
  PROP_GPU_DEVICE_ID,
  PROP_SHOW_SOURCE,
  PROP_NVBUF_MEMORY_TYPE,
  PROP_COMPUTE_HW,
  PROP_INTERPOLATION_METHOD,
  PROP_CUSTOM_TILE_CONFIG,
  PROP_BUFFER_POOL_SIZE,
  PROP_SQUARE_SEQ_GRID
};


#define DEFAULT_ROWS 1
#define DEFAULT_COLUMNS 1
#define DEFAULT_WIDTH 1920
#define DEFAULT_HEIGHT 1080
#define DEFAULT_BUFFER_POOL_SIZE 5
#define DEFAULT_GPU_DEVICE_ID 0
#define DEFAULT_SHOW_SOURCE -1
#define DEFAULT_MAX_TILES 128

static gboolean nvmultistreamtiler_clear_scratch(GstNvMultiStreamTiler* nvmultistreamtiler);

static CustomTileConfig* nvmultistreamtiler_dup_custom_tile_config(CustomTileConfig* tileC);

static void nvmultistreamtiler_set_single_source_mode_if_required(GstNvMultiStreamTiler* nvmultistreamtiler);

static gboolean handle_eos (GstNvMultiStreamTiler* nvmultistreamtiler, guint source_id) {
    guint var = 0;

    nvmultistreamtiler->cur_sources--;
    if(nvmultistreamtiler->square_grid) {
      /* Write/Read on the map should be under a lock. */
      g_mutex_lock(&nvmultistreamtiler->tiler_map_lock);
      auto it = nvmultistreamtiler->tiler_map->find(source_id);
     if (it != nvmultistreamtiler->tiler_map->end()) {
        nvmultistreamtiler->tiler_map->erase(source_id);
        int j = 0;
        for (auto it = nvmultistreamtiler->tiler_map->begin(); it != nvmultistreamtiler->tiler_map->end(); it++) {
          it->second = j;
          j++;
        }
     }
      g_mutex_unlock(&nvmultistreamtiler->tiler_map_lock);
    }

    var = sqrt(nvmultistreamtiler->cur_sources);
    if(nvmultistreamtiler->tilerIface) {
      /** GST_NVEVENT_STREAM_EOS may arrive before the decide_allocation()
       * call if Gst pipeline faces an error */
      nvmultistreamtiler->tilerIface->DeleteSource(source_id);
      nvmultistreamtiler_clear_scratch(nvmultistreamtiler);
      if (var && nvmultistreamtiler->square_grid && (var*var >= nvmultistreamtiler->cur_sources)) {
        nvmultistreamtiler->tilerConfig.rows = nvmultistreamtiler->tilerConfig.columns = var;

        GST_ELEMENT_INFO (nvmultistreamtiler, LIBRARY, SETTINGS,
          ("EOS received, reconfigured nvmultistreamtiler"),
          ("Configuration %dx%d\n", nvmultistreamtiler->tilerConfig.rows, nvmultistreamtiler->tilerConfig.columns));

        /** Re-init the tiler low-level API with new config */
        bool ok = nvmultistreamtiler->tilerIface->Init(nvmultistreamtiler->tilerConfig);
        if(!ok)
        {
            g_mutex_unlock(&nvmultistreamtiler->tilerIfaceLock);
            return FALSE;
        }
      }
    }
    return TRUE;

}
static gboolean gst_nvmultistreamtiler_sink_event (GstBaseTransform *trans, GstEvent * event)
{
  GstNvMultiStreamTiler *nvmultistreamtiler = GST_NVMULTISTREAMTILER (trans);
  guint total_tiles = nvmultistreamtiler->tilerConfig.rows*nvmultistreamtiler->tilerConfig.columns;
  guint tiler_rows, tiler_columns;
  guint source_id = 0;

  switch (static_cast<GstNvEventType>(event->type)) {
    case GST_NVEVENT_PAD_ADDED:
      gst_nvevent_parse_pad_added (event, &source_id);
      GST_DEBUG_OBJECT (nvmultistreamtiler, "Pad added %d\n", source_id);
      LOGD("Pad added %d\n", source_id);

      if (nvmultistreamtiler->square_grid) {
        g_mutex_lock(&nvmultistreamtiler->tilerIfaceLock);

        auto it = nvmultistreamtiler->tiler_map->find(source_id);
        if (it == nvmultistreamtiler->tiler_map->end()) {
          nvmultistreamtiler->cur_sources++;
        }

        /* Write/Read on the map should be under a lock. */
        g_mutex_lock(&nvmultistreamtiler->tiler_map_lock);
        if (nvmultistreamtiler->tiler_map->empty()) {
          nvmultistreamtiler->tiler_map->insert({source_id, 0});
        } else {
          auto last_pair = std::prev(nvmultistreamtiler->tiler_map->end());
          nvmultistreamtiler->tiler_map->insert({source_id, last_pair->second + 1});
          /* When a previous source_id is used again. We will re-assign the tiler
           * windows.
           */
          if (source_id < last_pair->first) {
            int j = 0;
            for (auto it = nvmultistreamtiler->tiler_map->begin(); it != nvmultistreamtiler->tiler_map->end(); it++) {
              it->second = j;
              j++;
            }
          }
        }
        g_mutex_unlock(&nvmultistreamtiler->tiler_map_lock);

        total_tiles = nvmultistreamtiler->tilerConfig.rows*nvmultistreamtiler->tilerConfig.columns;

        if (nvmultistreamtiler->cur_sources > total_tiles) {
          if (nvmultistreamtiler->cur_sources > DEFAULT_MAX_TILES) {
            GST_WARNING_OBJECT (nvmultistreamtiler,"Warning no of sources exceed"\
                " total num of tiles unable to display%d\n", source_id);
            g_mutex_unlock(&nvmultistreamtiler->tilerIfaceLock);
            return FALSE;
          }

          nvmultistreamtiler->tilerConfig.rows++;
          nvmultistreamtiler->tilerConfig.columns++;

          GST_ELEMENT_INFO (nvmultistreamtiler, LIBRARY, SETTINGS,
              ("Source added, reconfigured nvmultistreamtiler"),
              ("Configuration %dx%d\n", nvmultistreamtiler->tilerConfig.rows, nvmultistreamtiler->tilerConfig.columns));

          /** GST_NVEVENT_PAD_ADDED from nvstreammux could come in
          * even before tilerIface interface initialization
          * in which case we need not clear the scratch
          * nor re-init (without init) with new tiler info as that's
          * also part of the tilerIface initialization
          * at gst_nvmultistreamtiler_decide_allocation
          */
          if(nvmultistreamtiler->tilerIface)
          {
            /** clear tiler-scratch to black before accomodating
            * new source
            */
            nvmultistreamtiler_clear_scratch(nvmultistreamtiler);
            /** Re-init the tiler low-level API with new config */
            bool ok = nvmultistreamtiler->tilerIface->Init(nvmultistreamtiler->tilerConfig);
            if(!ok)
            {
                g_mutex_unlock(&nvmultistreamtiler->tilerIfaceLock);
                return FALSE;
            }
          }
          g_mutex_unlock(&nvmultistreamtiler->tilerIfaceLock);
          nvmultistreamtiler_set_single_source_mode_if_required(nvmultistreamtiler);
        } else {
          g_mutex_unlock(&nvmultistreamtiler->tilerIfaceLock);
        }
      } else {
        if (source_id >= total_tiles) {

          if (source_id >= DEFAULT_MAX_TILES) {
            GST_WARNING_OBJECT (nvmultistreamtiler,"Warning Source id exceeds"\
                " total num of tiles unable to display%d\n", source_id);
            return FALSE;
          }
          /**
          * Strategy:
          * A new source_id came in- we cannot accomodate in current canvas.
          * ALL WE DO:
          * 1) Set rows to new tiler_rows (discussed below)
          * 2) Find the new columns count
          *
          * More info:
          * now with the new source_id, our goal is to accomplish:
          * (source_id + 1 >= rows * columns)
          * That means - if we fix tiler_rows to any new value,
          * New tiler_columns = (source_id+1) / tiler_rows
          * NOTE on fixing tiler_rows:
          * a) We fix tiler_rows in a way that if we have
          * perfect square when:
          * (source_id+1) == (normal case stream count)
          * is a perfect square-root (like 4, 16, etc)
          */
          g_mutex_lock(&nvmultistreamtiler->tilerIfaceLock);

          tiler_rows = sqrt(source_id+1);
          tiler_columns = (guint) ceil (1.0 * (source_id+1) / tiler_rows);
          GST_ELEMENT_INFO (nvmultistreamtiler, LIBRARY, SETTINGS,
              ("Source added, reconfigured nvmultistreamtiler"),
              ("Configuration %dx%d\n", tiler_rows, tiler_columns));

          nvmultistreamtiler->tilerConfig.rows = tiler_rows;
          nvmultistreamtiler->tilerConfig.columns = tiler_columns;

          /** GST_NVEVENT_PAD_ADDED from nvstreammux could come in
          * even before tilerIface interface initialization
          * in which case we need not clear the scratch
          * nor re-init (without init) with new tiler info as that's
          * also part of the tilerIface initialization
          * at gst_nvmultistreamtiler_decide_allocation
          */
          if(nvmultistreamtiler->tilerIface)
          {
            /** clear tiler-scratch to black before accomodating
            * new source
            */
            nvmultistreamtiler_clear_scratch(nvmultistreamtiler);
            /** Re-init the tiler low-level API with new config */
            bool ok = nvmultistreamtiler->tilerIface->Init(nvmultistreamtiler->tilerConfig);
            if(!ok)
            {
                g_mutex_unlock(&nvmultistreamtiler->tilerIfaceLock);
                return FALSE;
            }
          }
          g_mutex_unlock(&nvmultistreamtiler->tilerIfaceLock);
          nvmultistreamtiler_set_single_source_mode_if_required(nvmultistreamtiler);
        }
      }
      break;
    case GST_NVEVENT_STREAM_EOS:
      gst_nvevent_parse_stream_eos (event, &source_id);
      GST_DEBUG_OBJECT (nvmultistreamtiler, "stream eos %d\n", source_id);
      g_mutex_lock(&nvmultistreamtiler->tilerIfaceLock);
      handle_eos(nvmultistreamtiler, source_id);
      g_mutex_unlock(&nvmultistreamtiler->tilerIfaceLock);
      break;
    case GST_NVEVENT_STREAM_SEGMENT:
      {
        GstSegment *segment;
        gst_nvevent_parse_stream_segment (event, &source_id,
            &segment);
        GST_DEBUG_OBJECT (nvmultistreamtiler, "new segment %d\n", source_id);
        gst_segment_free (segment);
      }
      break;
    case GST_NVEVENT_PAD_DELETED:
      gst_nvevent_parse_pad_deleted (event, &source_id);
      GST_DEBUG_OBJECT (nvmultistreamtiler, "Pad deleted %d\n", source_id);
      g_mutex_lock(&nvmultistreamtiler->tilerIfaceLock);
      handle_eos(nvmultistreamtiler, source_id);
      g_mutex_unlock(&nvmultistreamtiler->tilerIfaceLock);

      if (source_id < total_tiles) {
        /**
        * ALL WE DO:
        * Black out the particular tile that this
        * source_id belonged to in the tilerScratchBuffer
        */
        /** clear tiler-scratch to black before accomodating
        * new source
        */
        g_mutex_lock(&nvmultistreamtiler->tilerIfaceLock);
        if(nvmultistreamtiler->tilerIface) {
          nvmultistreamtiler->tilerIface->DeleteSource(source_id);
          nvmultistreamtiler_clear_scratch(nvmultistreamtiler);
        }
        g_mutex_unlock(&nvmultistreamtiler->tilerIfaceLock);
      }
      break;

    default:
      break;
  }

  return GST_BASE_TRANSFORM_CLASS (parent_class)->sink_event(trans, event);
}

static GstCaps *
gst_nvmultistreamtiler_transform_caps (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, GstCaps * filter)
{
  GstNvMultiStreamTiler *nvmultistreamtiler = GST_NVMULTISTREAMTILER (trans);
  GstCaps *ret = gst_caps_copy (caps);
  guint i = 0, n = 0;

  GST_DEBUG_OBJECT (trans,
      "Transforming caps %" GST_PTR_FORMAT " in direction %s", caps,
      (direction == GST_PAD_SINK) ? "sink" : "src");

  if (!ret)
    return nullptr;
  n = gst_caps_get_size (ret);
  for (i = 0; i < n; i++) {
    GstStructure *str = gst_caps_get_structure (ret, i);

    if (gst_structure_has_field_typed (str, "width", G_TYPE_INT)) {
      gint width = 0;
      gst_structure_get_int (str, "width", &width);

      if (nvmultistreamtiler->tilerConfig.width) {
        if (direction == GST_PAD_SRC) {
          gst_structure_set (str, "width", GST_TYPE_INT_RANGE, 1, G_MAXINT,
              nullptr);
        } else {
          gst_structure_set (str, "width", G_TYPE_INT,
              nvmultistreamtiler->tilerConfig.width, nullptr);
        }
      } else {
        if (direction == GST_PAD_SRC) {
          width = width / nvmultistreamtiler->tilerConfig.columns;
        } else {
          width = width * nvmultistreamtiler->tilerConfig.columns;
        }
        gst_structure_set (str, "width", G_TYPE_INT, width, nullptr);
      }
    }

    if (gst_structure_has_field (str, "batch-size")) {
      if (direction == GST_PAD_SRC) {
        gst_structure_remove_field (str, "batch-size");
      }
      else if (direction == GST_PAD_SINK) {
        gst_structure_set (str, "batch-size", G_TYPE_INT, 1, nullptr);
      }
    }

    if (gst_structure_has_field_typed (str, "height", G_TYPE_INT)) {
      gint height = 0;
      gst_structure_get_int (str, "height", &height);

      if (nvmultistreamtiler->tilerConfig.height) {
        if (direction == GST_PAD_SRC) {
          gst_structure_set (str, "height", GST_TYPE_INT_RANGE, 1, G_MAXINT,
              nullptr);
        } else {
          gst_structure_set (str, "height", G_TYPE_INT,
              nvmultistreamtiler->tilerConfig.height, nullptr);
        }
      } else {
        if (direction == GST_PAD_SRC) {
          height = height / nvmultistreamtiler->tilerConfig.rows;
        } else {
          height = height * nvmultistreamtiler->tilerConfig.rows;
        }
        gst_structure_set (str, "height", G_TYPE_INT, height, nullptr);
      }
    }
    gst_structure_remove_fields (str,  "nvbuf-memory-type", "gpu-id", NULL);
  }

  if (filter) {
    GstCaps *tmp = gst_caps_intersect (ret, filter);
    gst_caps_unref (ret);
    ret = tmp;
  }

  GST_DEBUG_OBJECT (trans, "transformed %" GST_PTR_FORMAT " into %"
      GST_PTR_FORMAT, caps, ret);

  return ret;
}


/* fixate the caps on the other side */
static GstCaps* gst_nvmultistreamtiler_fixate_caps(GstBaseTransform* btrans,
    GstPadDirection direction, GstCaps* caps, GstCaps* othercaps)
{
  GstNvMultiStreamTiler *nvmultistreamtiler = GST_NVMULTISTREAMTILER (btrans);
  GstStructure *out_struct;

  othercaps = gst_caps_truncate(othercaps);
  othercaps = gst_caps_make_writable(othercaps);

  GST_DEBUG_OBJECT (nvmultistreamtiler, "trying to fixate othercaps %" GST_PTR_FORMAT
      " based on caps %" GST_PTR_FORMAT, othercaps, caps);

  out_struct = gst_caps_get_structure (othercaps, 0);

  if (direction == GST_PAD_SINK)
  {
    GstCaps *peer_caps = gst_pad_peer_query_caps(GST_BASE_TRANSFORM_SRC_PAD(btrans), NULL);
    GstStructure *peer_structure;
    const gchar *out_mem_type_string = NULL;
    int n=gst_caps_get_size(peer_caps);
    bool peer_has_gpu_id = false;
    if(n>0)
    {
        peer_caps = gst_caps_truncate(peer_caps);
        GST_DEBUG_OBJECT (btrans, "peer caps %" GST_PTR_FORMAT, peer_caps);
        peer_structure = gst_caps_get_structure (peer_caps, 0);
        out_mem_type_string = gst_structure_get_string (peer_structure, "nvbuf-memory-type");
        peer_has_gpu_id = gst_structure_has_field(peer_structure, "gpu-id");
    }
    if(!out_mem_type_string)
    {
        int mem_type = nvmultistreamtiler->cuda_mem_type;
        if(mem_type == NVBUF_MEM_DEFAULT)
            GET_DEFAULT_MEM_TYPE(mem_type);
        g_assert(mem_type != NVBUF_MEM_DEFAULT);
        gst_structure_set (out_struct, "nvbuf-memory-type", G_TYPE_STRING , gst_nvbuf_memory_get_name(mem_type), NULL);
    }
    else
    {
        nvmultistreamtiler->cuda_mem_type = gst_nvbuf_memory_get_value(out_mem_type_string);
        if(nvmultistreamtiler->cuda_mem_type <= NVBUF_MEM_DEFAULT)
            GST_ERROR_OBJECT(btrans, "Incorrect nvbuf-memory-type set on src pad!!");
        GST_WARNING_OBJECT(btrans, "nvbuf-memory-type property is set based on SRC caps. Property config setting (if any) is overridden!!");
        gst_structure_set (out_struct, "nvbuf-memory-type", G_TYPE_STRING , out_mem_type_string, NULL);
    }
    if(peer_has_gpu_id)
    {
        gst_structure_get_int(peer_structure, "gpu-id", (int *)&nvmultistreamtiler->gpu_id);
        GST_WARNING_OBJECT(btrans, "gpu-id property is set based on SRC caps. Property config setting (if any) is overridden!!");
    }
    gst_structure_set (out_struct, "gpu-id", G_TYPE_INT , nvmultistreamtiler->gpu_id, NULL);
    gst_caps_unref(peer_caps);
    }


  othercaps = GST_BASE_TRANSFORM_CLASS (parent_class)->fixate_caps(btrans, direction, caps, othercaps);

  GST_DEBUG_OBJECT (nvmultistreamtiler, "fixated othercaps to %" GST_PTR_FORMAT, othercaps);
  return othercaps;

}

static gboolean
gst_nvmultistreamtiler_transform_meta (GstBaseTransform * trans,
    GstBuffer * outbuf, GstMeta * meta, GstBuffer * inbuf)
{
  /** Shall not copy any meta;
   * gst_nvmultistreamtiler_transform() shall add a new NvDsBatchMeta
   * with all relevant metadata sync'd from inbuf->NvDsBatchMeta
   */
  return FALSE;
}

static gboolean nvmultistreamtiler_clear_scratch(GstNvMultiStreamTiler* nvmultistreamtiler)
{
    if(!nvmultistreamtiler->tilerScratchBuffer)
    {
        return FALSE;
    }
    cudaError_t CUerr = cudaSuccess;
    CUerr = cudaSetDevice(nvmultistreamtiler->gpu_id);
    if(CUerr != cudaSuccess)
    {
      GST_ELEMENT_ERROR (nvmultistreamtiler, RESOURCE, FAILED,
        ("Failed to set cuda device %d", nvmultistreamtiler->gpu_id),
        ("cudaSetDevice failed with error %d(%s)", CUerr,
         cudaGetErrorName(CUerr)));
       return FALSE;
    }
    for (guint32 i = 0; i < nvmultistreamtiler->tilerScratchBuffer->batchSize; i++)
    {
        for (guint32 j = 0; j < nvmultistreamtiler->tilerScratchBuffer->surfaceList[i].planeParams.num_planes; j++)
        {
            /* Memset 0th plane to 0 and other planes to 128 for black color */
            int status = NvBufSurfaceMemSet (nvmultistreamtiler->tilerScratchBuffer, i, j, ((j == 0) ? 0 : 128));
            if (status < 0)
            {
                GST_ELEMENT_ERROR (nvmultistreamtiler, STREAM, FAILED,
                       ("GstNvTiler: FATAL; Error(%d) in buffer memset\n", status),
                       (nullptr));
                return FALSE;
            }
        }
    }
    return TRUE;
}

static gboolean
gst_nvmultistreamtiler_decide_allocation (GstBaseTransform * btrans,
    GstQuery * query)
{
  GstNvMultiStreamTiler *nvmultistreamtiler = GST_NVMULTISTREAMTILER (btrans);
  guint j, metas_no;
  GstCaps *outcaps = nullptr;
  GstBufferPool *cur_pool;

  GstQuery *nsquery = gst_nvquery_numStreams_size_new ();
  guint num_streams = 0;
  if (gst_pad_peer_query (GST_BASE_TRANSFORM_SINK_PAD (btrans), nsquery)) {
    gst_nvquery_numStreams_size_parse (nsquery, &num_streams);
    GST_DEBUG_OBJECT (nvmultistreamtiler, "nvmultistreamtiler: num_streams set as %d...\n", 0);
  } else {
    GST_DEBUG_OBJECT (nvmultistreamtiler, "nvmultistreamtiler: num_streams not set.\n");
  }
  gst_query_unref (nsquery);

  metas_no = gst_query_get_n_allocation_metas (query);
  for (j = 0; j < metas_no; j++) {
    gboolean remove_meta;
    GType meta_api;
    const GstStructure *param_str = nullptr;

    meta_api = gst_query_parse_nth_allocation_meta (query, j, &param_str);

    if (gst_meta_api_type_has_tag (meta_api, GST_META_TAG_MEMORY)) {
      /* Different memory will get allocated for input and output.
         remove all memory dependent metadata */
      remove_meta = TRUE;
    } else {
      /* Default remove all metadata */
      remove_meta = TRUE;
    }

    if (remove_meta) {
      gst_query_remove_nth_allocation_meta (query, j);
      j--;
      metas_no--;
    }
  }

  gst_query_parse_allocation (query, &outcaps, nullptr);
  if (outcaps == nullptr)
  {
    GST_ERROR ("no caps specified");
    return FALSE;
  }

  cur_pool = gst_base_transform_get_buffer_pool (btrans);
  if (cur_pool) {
    GstStructure *config = gst_buffer_pool_get_config (cur_pool);
    GstCaps *myoutcaps;

    gst_buffer_pool_config_get_params (config, &myoutcaps, nullptr, nullptr, nullptr);

    if (!gst_caps_is_equal (outcaps, myoutcaps)) {
      /* different caps, we can't use current pool */
      gst_object_unref (cur_pool);
      cur_pool = nullptr;
    }
    gst_structure_free (config);
  }

  if (cur_pool == nullptr) {
    GstVideoInfo info;
    gst_video_info_init (&info);
    GstStructure *config;

    if (!gst_video_info_from_caps (&info, outcaps))
    {
        GST_ERROR ("invalid caps specified");
        return FALSE;
    }

    cur_pool = gst_nvds_buffer_pool_new ();
    config = gst_buffer_pool_get_config (cur_pool);
    gst_buffer_pool_config_set_params (config, outcaps, sizeof(NvBufSurface),
            nvmultistreamtiler->buffer_pool_size, nvmultistreamtiler->buffer_pool_size);

    gst_structure_set (config,
          "memtype", G_TYPE_UINT, nvmultistreamtiler->cuda_mem_type,
          "gpu-id", G_TYPE_UINT, nvmultistreamtiler->gpu_id,
          "batch-size", G_TYPE_UINT, 1, NULL);
    if (!gst_buffer_pool_set_config (cur_pool, config)) {
      GST_WARNING ("Nvstreammux bufferpool configuration failed");
      return FALSE;
    }

    gboolean is_active = gst_buffer_pool_set_active (cur_pool, TRUE);
    if (!is_active) {

      gst_object_unref (cur_pool);
      GST_ELEMENT_ERROR (nvmultistreamtiler, RESOURCE, FAILED,
          ("GstNvTiler: FATAL; Failed to allocate buffers\n"),
          (nullptr));


      return FALSE;
    } else {
      GST_DEBUG (" Output buffer pool (%p) successfully created with 4 buffers", cur_pool);
    }
  }

  {
    GstStructure *config;

    config = gst_buffer_pool_get_config (cur_pool);

    /* Set allocation pool */
    if (gst_query_get_n_allocation_pools (query) > 0) {
      gst_query_set_nth_allocation_pool (query, 0, cur_pool,
          sizeof (NvBufSurface), 4, 4);
    } else {
      gst_query_add_allocation_pool (query, cur_pool, sizeof (NvBufSurface),
          4, 4);
    }
    gst_structure_free (config);
  }

  if (nvmultistreamtiler->square_grid) {
    if (num_streams) {
      guint var = sqrt(num_streams);
      if (var*var == num_streams) {
        nvmultistreamtiler->tilerConfig.rows = nvmultistreamtiler->tilerConfig.columns = var;
      } else {
        nvmultistreamtiler->tilerConfig.rows = nvmultistreamtiler->tilerConfig.columns = var + 1;
      }
    } else {
      nvmultistreamtiler->tilerConfig.rows = nvmultistreamtiler->tilerConfig.columns = 1;
    }
    GST_ELEMENT_INFO (nvmultistreamtiler, LIBRARY, SETTINGS,
    ("Square Grid property enabled, ignoring nvmultistreamtiler rows/columns"),
    ("Configuration %dx%d\n", nvmultistreamtiler->tilerConfig.rows, nvmultistreamtiler->tilerConfig.columns));
  }

  if(!nvmultistreamtiler->tilerIface)
  {
      nvmultistreamtiler->tilerIface = new NvTiler();
      LOGD("initializing tiler %d %d\n", nvmultistreamtiler->tilerConfig.width, nvmultistreamtiler->tilerConfig.height);
      g_mutex_lock(&nvmultistreamtiler->tilerIfaceLock);
      bool ok = nvmultistreamtiler->tilerIface->Init(nvmultistreamtiler->tilerConfig);
      g_mutex_unlock(&nvmultistreamtiler->tilerIfaceLock);
      if(!ok)
      {
          return FALSE;
      }
      ok = nvmultistreamtiler->tilerIface->SetSquareGridMode(nvmultistreamtiler->square_grid);
      if(!ok)
      {
          return FALSE;
      }
      ok = nvmultistreamtiler->tilerIface->SetTilerMap(nvmultistreamtiler->tiler_map, &nvmultistreamtiler->tiler_map_lock);
      if(!ok)
      {
          return FALSE;
      }
      /** Disabled now - TODO Debug why a slice from pool doesnt work */
      GstFlowReturn okGst = gst_buffer_pool_acquire_buffer(cur_pool, &nvmultistreamtiler->tilerScratchBufferGst, nullptr);
      if(okGst == GST_FLOW_OK)
      {
         GstMapInfo mapInfo;
         memset(&mapInfo, 0, sizeof(GstMapInfo));
         if(TRUE == gst_buffer_map (nvmultistreamtiler->tilerScratchBufferGst, &mapInfo, GST_MAP_WRITE))
         {
             nvmultistreamtiler->tilerScratchBuffer = reinterpret_cast<NvBufSurface*>(mapInfo.data);
             gst_buffer_unmap(nvmultistreamtiler->tilerScratchBufferGst, &mapInfo);
             LOGD("surface create success %p colorFormat=%d (%d X %d) %p\n",
                nvmultistreamtiler->tilerScratchBuffer,
                nvmultistreamtiler->tilerScratchBuffer->surfaceList[0].colorFormat,
                nvmultistreamtiler->tilerScratchBuffer->surfaceList[0].width,
                nvmultistreamtiler->tilerScratchBuffer->surfaceList[0].height,
                nvmultistreamtiler->tilerScratchBuffer->surfaceList[0].dataPtr
                );
             g_mutex_lock(&nvmultistreamtiler->tilerIfaceLock);
             nvmultistreamtiler_clear_scratch(nvmultistreamtiler);
             g_mutex_unlock(&nvmultistreamtiler->tilerIfaceLock);
         }
         else
         {
            GST_ELEMENT_ERROR (nvmultistreamtiler, STREAM, FAILED,
                ("GstNvTiler: FATAL; Failed to acquire tiler-scratch buffer from the pool\n"),
                (nullptr));
            return FALSE;
         }
      }
      else
      {
          GST_ELEMENT_ERROR (nvmultistreamtiler, STREAM, FAILED,
              ("GstNvTiler: FATAL; Failed to acquire tiler-scratch buffer from the pool\n"),
              (nullptr));
          return FALSE;
      }
      nvmultistreamtiler_set_single_source_mode_if_required(nvmultistreamtiler);
  }

  gst_object_unref (cur_pool);

  return TRUE;
}

static gboolean
gst_nvmultistreamtiler_set_caps (GstBaseTransform * trans, GstCaps * incaps,
    GstCaps * outcaps)
{
  GstNvMultiStreamTiler *nvmultistreamtiler = GST_NVMULTISTREAMTILER (trans);
  GstStructure *s1 = NULL;
  gint val = 1;

  s1 = gst_caps_get_structure(incaps, 0);
  gst_structure_get_int (s1, "num-surfaces-per-frame", &val);
  nvmultistreamtiler->num_surfaces_per_frame = val;

  nvmultistreamtiler->frame_num = 0;
  gst_video_info_init (&nvmultistreamtiler->invideoinfo);
  gst_video_info_from_caps (&nvmultistreamtiler->invideoinfo, incaps);
  gst_video_info_init (&nvmultistreamtiler->outvideoinfo);
  gst_video_info_from_caps (&nvmultistreamtiler->outvideoinfo, outcaps);

  return TRUE;
}

static void nvmultistreamtiler_unmap(GstBuffer* inbuf, GstMapInfo* inMap, GstBuffer* outbuf, GstMapInfo* outMap)
{
    if(inbuf && inMap)
    {
        gst_buffer_unmap(inbuf, inMap);
    }
    if(outbuf && outMap)
    {
        gst_buffer_unmap(outbuf, outMap);
    }
}

static NvDsBatchMeta* nvmultistreamtiler_get_batch_meta(GstBuffer* buffer)
{
  NvDsMeta*      dsMeta          = nullptr;
  NvDsBatchMeta* bufferBatchMeta = nullptr;
  GstMeta*       meta;
  gpointer state = NULL;

  while ((meta = gst_buffer_iterate_meta (buffer, &state)))
  {
      if (gst_meta_api_type_has_tag(meta->info->api, dsmeta_quark))
      {
          dsMeta = reinterpret_cast<NvDsMeta*>(meta);
          /** Extract NvDsBatchMeta */
          if(dsMeta->meta_type == NVDS_BATCH_GST_META)
          {
              bufferBatchMeta = static_cast<NvDsBatchMeta*>(dsMeta->meta_data);
              break;
          }
      }
  }

  return bufferBatchMeta;
}

static void nvmultistreamtiler_set_single_source_mode_if_required(GstNvMultiStreamTiler* nvmultistreamtiler)
{
  if(nvmultistreamtiler->tilerIface)
  {
      g_mutex_lock(&nvmultistreamtiler->tilerIfaceLock);
      bool ok = nvmultistreamtiler->tilerIface->SetSingleSourceMode(
                  (nvmultistreamtiler->show_source != -1), nvmultistreamtiler->show_source);
      if(ok)
      {
          nvmultistreamtiler_clear_scratch(nvmultistreamtiler);
      }
      g_mutex_unlock(&nvmultistreamtiler->tilerIfaceLock);
      if(!ok)
      {
          GST_ELEMENT_ERROR (nvmultistreamtiler, STREAM, FAILED,
                ("GstNvTiler: ERROR; SetSingleSourceMode failed"),
                (nullptr));
      }
  }
}

unsigned int frame_num = 0;

static GstFlowReturn
gst_nvmultistreamtiler_transform (GstBaseTransform * trans, GstBuffer * inbuf,
    GstBuffer * outbuf)
{
  GstNvMultiStreamTiler *nvmultistreamtiler = GST_NVMULTISTREAMTILER (trans);
  GstMapInfo in_map;
  GstMapInfo out_map;
  cudaError_t CUerr = cudaSuccess;
  memset(&in_map, 0, sizeof(GstMapInfo));
  memset(&out_map, 0, sizeof(GstMapInfo));

  nvds_set_input_system_timestamp(inbuf, GST_ELEMENT_NAME(nvmultistreamtiler));

  NvDsBatchMeta* inputBufferBatchMeta = nvmultistreamtiler_get_batch_meta(inbuf);

  CUerr = cudaSetDevice(nvmultistreamtiler->gpu_id);
  if(CUerr != cudaSuccess)
  {
    GST_ELEMENT_ERROR (nvmultistreamtiler, RESOURCE, FAILED,
        ("Failed to set cuda device %d", nvmultistreamtiler->gpu_id),
        ("cudaSetDevice failed with error %d(%s)", CUerr,
         cudaGetErrorName(CUerr)));
    return GST_FLOW_ERROR;
  }


  if(!inputBufferBatchMeta)
  {
      GST_ELEMENT_ERROR (nvmultistreamtiler, STREAM, FAILED,
            ("GstNvTiler: FATAL ERROR; NvDsMeta->NvDsBatchMeta missing in the input buffer"),
            (nullptr));
      return GST_FLOW_ERROR;
  }

  NvDsBatchMeta* outputBufferBatchMeta = nvds_create_batch_meta(1);
  if(!outputBufferBatchMeta)
  {
      GST_ELEMENT_ERROR (nvmultistreamtiler, STREAM, FAILED,
            ("GstNvTiler: FATAL ERROR; output NvDsMeta creation failed"),
            (nullptr));
      return GST_FLOW_ERROR;
  }
  outputBufferBatchMeta->base_meta.batch_meta = outputBufferBatchMeta;
  outputBufferBatchMeta->base_meta.copy_func = nvds_batch_meta_copy_func;
  outputBufferBatchMeta->base_meta.release_func = nvds_batch_meta_release_func;
  outputBufferBatchMeta->max_frames_in_batch = 1;

  /** Fetch the NvBufSurface from inbuf and Composite! */
  gst_buffer_map (inbuf, &in_map, GST_MAP_READ);
  gst_buffer_map (outbuf, &out_map, GST_MAP_WRITE);

  if(!in_map.data || !out_map.data)
  {
      nvmultistreamtiler_unmap(inbuf, &in_map, outbuf, &out_map);
      GST_ELEMENT_ERROR (nvmultistreamtiler, STREAM, FAILED,
        ("GstNvTiler: FATAL ERROR; Failed to map buffer"),
        ("inputBuffer=%p outputBuffer=%p", in_map.data, out_map.data));
      return GST_FLOW_ERROR;
  }

  NvBufSurface* inputBuffer  = reinterpret_cast<NvBufSurface*>(in_map.data);
  NvBufSurface* outputBuffer = reinterpret_cast<NvBufSurface*>(out_map.data);

  bool inputBufCheck  = CHECK_NVDS_MEMORY_AND_GPUID(nvmultistreamtiler, inputBuffer);
  bool outputBufCheck = CHECK_NVDS_MEMORY_AND_GPUID(nvmultistreamtiler, outputBuffer);
  if (inputBufCheck || outputBufCheck)  {
    nvmultistreamtiler_unmap(inbuf, &in_map, outbuf, &out_map);
    GST_ELEMENT_ERROR (nvmultistreamtiler, STREAM, FAILED,
        ("GstNvTiler: Incompatible input / output buffer"),
        ("inputBuffer=%d outputBuffer=%d", inputBuffer->memType, outputBuffer->memType));
    return GST_FLOW_ERROR;
  }

  NvDsFrameMeta *frame_meta = NULL;
  frame_meta = nvds_acquire_frame_meta_from_pool(outputBufferBatchMeta);
  frame_meta->buf_pts = GST_BUFFER_PTS (outbuf);
  frame_meta->frame_num = nvmultistreamtiler->frame_num;
  frame_meta->batch_id = 0;
  frame_meta->source_frame_width = nvmultistreamtiler->tilerConfig.width;
  frame_meta->source_frame_height = nvmultistreamtiler->tilerConfig.height;
  frame_meta->num_surfaces_per_frame = 1;
  nvds_add_frame_meta_to_batch(outputBufferBatchMeta, frame_meta);

  char context_name[100];
  snprintf(context_name, sizeof(context_name), "%s_(Frame=%u)",
        GST_ELEMENT_NAME(nvmultistreamtiler), frame_num++);
  nvtx_helper_push_pop(context_name);

  g_mutex_lock(&nvmultistreamtiler->tilerIfaceLock);
  bool ok = nvmultistreamtiler->tilerIface->Composite(inputBuffer,
    inputBufferBatchMeta, outputBuffer, nvmultistreamtiler->tilerScratchBuffer,
    nvmultistreamtiler->gpu_id, nvmultistreamtiler->compute_hw,
    nvmultistreamtiler->interpolation_method);
  nvmultistreamtiler_unmap(inbuf, &in_map, outbuf, &out_map);
  if(!ok)
  {
      GST_ELEMENT_ERROR (nvmultistreamtiler, RESOURCE, FAILED,
        ("GstNvTiler: FATAL ERROR; NvTiler::Composite failed"),
        (nullptr));
      g_mutex_unlock(&nvmultistreamtiler->tilerIfaceLock);
      return GST_FLOW_ERROR;
  }

  ok = nvmultistreamtiler->tilerIface->AdjustOutputMeta(outputBufferBatchMeta);

  g_mutex_unlock(&nvmultistreamtiler->tilerIfaceLock);
  if(!ok)
  {
    GST_ELEMENT_ERROR (nvmultistreamtiler, RESOURCE, FAILED,
        ("GstNvTiler: FATAL ERROR; NvTiler::AdjustOutputMeta failed"),
        (nullptr));
      return GST_FLOW_ERROR;
  }

  /** NOTE: This NvDsBatchMeta shall be the first and only
   * meta-data in outbuf as gst_nvmultistreamtiler_transform_meta()
   * does not allow meta-data-copy from inbuf to outbuf
   */
  NvDsMeta *dsmeta = gst_buffer_add_nvds_meta (outbuf, outputBufferBatchMeta, NULL,
                (NvDsMetaCopyFunc) nvds_batch_meta_copy_func,
                (NvDsMetaReleaseFunc) nvds_batch_meta_release_func);
  if (dsmeta == nullptr)
  {
      GST_ELEMENT_ERROR (nvmultistreamtiler, STREAM, FAILED,
            ("GstNvTiler: FATAL ERROR; output NvDsMeta add failed"),
            (nullptr));
      return GST_FLOW_ERROR;
  }
  dsmeta->meta_type = NVDS_BATCH_GST_META;

  // Here wait for the asyncronous processing by "NvBufSurfTransformAsync" to finish
  g_mutex_lock(&nvmultistreamtiler->tilerIfaceLock);
  nvmultistreamtiler->tilerIface->SyncObjWait();
  g_mutex_unlock(&nvmultistreamtiler->tilerIfaceLock);
  nvtx_helper_push_pop (NULL);

  /** outbuf shall have input metadata copied as is
   * But, we need to :
   * 1) Have meta-data for surfaces which were not present in
   * this particular batch - copied from previous and latest
   * corresponding metadata
   * 2) Update the outbuf->NvDsMeta->NvDsBatchMeta to
   * reflect (1)
   */

  /** Push output buffer */

  nvds_set_output_system_timestamp(inbuf, GST_ELEMENT_NAME(nvmultistreamtiler));

  nvds_copy_batch_user_meta_list( inputBufferBatchMeta->batch_user_meta_list,
      outputBufferBatchMeta);

  nvmultistreamtiler->frame_num++;
  return GST_FLOW_OK;
}

static gboolean
gst_nvmultistreamtiler_start (GstBaseTransform * trans)
{
  GstNvMultiStreamTiler *nvmultistreamtiler = GST_NVMULTISTREAMTILER (trans);

  cudaError_t CUerr = cudaSuccess;
  CUerr = cudaSetDevice(nvmultistreamtiler->gpu_id);
  if(CUerr != cudaSuccess)
  {
    GST_ELEMENT_ERROR (nvmultistreamtiler, RESOURCE, FAILED,
        ("Failed to set cuda device %d", nvmultistreamtiler->gpu_id),
        ("cudaSetDevice failed with error %d(%s)", CUerr,
         cudaGetErrorName(CUerr)));
    return FALSE;
  }

  return TRUE;
}

static gboolean
gst_nvmultistreamtiler_stop (GstBaseTransform * trans)
{
  GstNvMultiStreamTiler *nvmultistreamtiler = GST_NVMULTISTREAMTILER (trans);

  if(nvmultistreamtiler->tilerScratchBufferGst)
  {
      gst_buffer_unref(nvmultistreamtiler->tilerScratchBufferGst);
      nvmultistreamtiler->tilerScratchBufferGst = nullptr;
      nvmultistreamtiler->tilerScratchBuffer = nullptr;
  }

  g_mutex_lock(&nvmultistreamtiler->tilerIfaceLock);
  if(nvmultistreamtiler->tilerIface)
  {
      nvmultistreamtiler->tilerIface->Deinit();
      delete nvmultistreamtiler->tilerIface;
      nvmultistreamtiler->tilerIface = nullptr;
  }

  g_mutex_unlock(&nvmultistreamtiler->tilerIfaceLock);

  return TRUE;
}

static gboolean
gst_nvmultistreamtiler_query (GstBaseTransform * trans,
    GstPadDirection direction, GstQuery * query)
{
  if (gst_nvquery_is_batch_size (query)) {
    if (direction == GST_PAD_SINK)
      return FALSE;
    gst_nvquery_batch_size_set (query, 1);
    return TRUE;
  }
  return GST_BASE_TRANSFORM_CLASS (parent_class)->query (trans, direction,
      query);
}

static void
gst_nvmultistreamtiler_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstNvMultiStreamTiler *nvmultistreamtiler = GST_NVMULTISTREAMTILER (object);
  CustomTileConfig* customTileConfig = nullptr;

  switch (prop_id) {
    case PROP_COLMUNS:
      nvmultistreamtiler->tilerConfig.columns = g_value_get_uint (value);
      break;
    case PROP_ROWS:
      nvmultistreamtiler->tilerConfig.rows = g_value_get_uint (value);
      break;
    case PROP_WIDTH:
      nvmultistreamtiler->tilerConfig.width = g_value_get_uint (value);
      nvmultistreamtiler->tilerConfig.width = GST_ROUND_UP_4(nvmultistreamtiler->tilerConfig.width);
      break;
    case PROP_HEIGHT:
      nvmultistreamtiler->tilerConfig.height = g_value_get_uint (value);
      nvmultistreamtiler->tilerConfig.height = GST_ROUND_UP_4(nvmultistreamtiler->tilerConfig.height);
      break;
    case PROP_BUFFER_POOL_SIZE:
      nvmultistreamtiler->buffer_pool_size = g_value_get_uint (value);
      break;
    case PROP_SQUARE_SEQ_GRID:
      nvmultistreamtiler->square_grid = g_value_get_boolean (value);
      break;
    case PROP_GPU_DEVICE_ID:
      nvmultistreamtiler->gpu_id = g_value_get_uint (value);
      nvmultistreamtiler->tilerConfig.gpuId = nvmultistreamtiler->gpu_id;
      break;
    case PROP_SHOW_SOURCE:
      if(nvmultistreamtiler->show_source == g_value_get_int (value))
      {
        break;
      }
      nvmultistreamtiler->show_source = g_value_get_int (value);
      nvmultistreamtiler_set_single_source_mode_if_required(nvmultistreamtiler);
      break;
    case PROP_COMPUTE_HW:
      nvmultistreamtiler->compute_hw = g_value_get_enum (value);
      break;
    case PROP_NVBUF_MEMORY_TYPE:
      nvmultistreamtiler->cuda_mem_type = g_value_get_enum (value);
      break;
    case PROP_INTERPOLATION_METHOD:
      nvmultistreamtiler->interpolation_method = g_value_get_enum (value);
      break;
    case PROP_CUSTOM_TILE_CONFIG:
      customTileConfig = nvmultistreamtiler_dup_custom_tile_config(static_cast<CustomTileConfig*>(g_value_get_pointer (value)));
      if(customTileConfig)
      {
          nvmultistreamtiler->tilerConfig.customTileConfig = *customTileConfig;
      }
      g_mutex_lock(&nvmultistreamtiler->tilerIfaceLock);
      if (nvmultistreamtiler->tilerIface)
      {
        nvmultistreamtiler_clear_scratch(nvmultistreamtiler);
        nvmultistreamtiler->tilerIface->Init(nvmultistreamtiler->tilerConfig);
      }
      g_mutex_unlock(&nvmultistreamtiler->tilerIfaceLock);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static void
gst_nvmultistreamtiler_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstNvMultiStreamTiler *nvmultistreamtiler = GST_NVMULTISTREAMTILER (object);

  switch (prop_id) {
    case PROP_COLMUNS:
      g_value_set_uint (value, nvmultistreamtiler->tilerConfig.columns);
      break;
    case PROP_ROWS:
      g_value_set_uint (value, nvmultistreamtiler->tilerConfig.rows);
      break;
    case PROP_WIDTH:
      g_value_set_uint (value, nvmultistreamtiler->tilerConfig.width);
      break;
    case PROP_HEIGHT:
      g_value_set_uint (value, nvmultistreamtiler->tilerConfig.height);
      break;
    case PROP_BUFFER_POOL_SIZE:
      g_value_set_uint (value, nvmultistreamtiler->buffer_pool_size);
      break;
    case PROP_SQUARE_SEQ_GRID:
      g_value_set_boolean (value, nvmultistreamtiler->square_grid);
      break;
    case PROP_GPU_DEVICE_ID:
      g_value_set_uint (value, nvmultistreamtiler->gpu_id);
      break;
    case PROP_SHOW_SOURCE:
      g_value_set_int (value, nvmultistreamtiler->show_source);
      break;
    case PROP_COMPUTE_HW:
      g_value_set_enum (value, nvmultistreamtiler->compute_hw);
      break;
    case PROP_NVBUF_MEMORY_TYPE:
      g_value_set_enum (value, nvmultistreamtiler->cuda_mem_type);
      break;
    case PROP_INTERPOLATION_METHOD:
      g_value_set_enum (value, nvmultistreamtiler->interpolation_method);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}


static void
gst_nvmultistreamtiler_class_init (GstNvMultiStreamTilerClass * klass)
{
  GObjectClass *gobject_class = G_OBJECT_CLASS (klass);
  GstElementClass *gstelement_class = GST_ELEMENT_CLASS (klass);
  GstBaseTransformClass *basetrans_class = GST_BASE_TRANSFORM_CLASS (klass);

  gst_element_class_set_static_metadata (gstelement_class,
      "Stream Tiler DS", "Generic",
      "Tile input multistream buffer into a 2D array",
      "NVIDIA Corporation. Post on Deepstream for Tesla forum for any queries "
      "@ https://devtalk.nvidia.com/default/board/209/");

  gobject_class->set_property = gst_nvmultistreamtiler_set_property;
  gobject_class->get_property = gst_nvmultistreamtiler_get_property;

  g_object_class_install_property (gobject_class, PROP_COLMUNS,
      g_param_spec_uint ("columns", "Columns",
          "Number of columns in the Tiled 2D output",
          1, G_MAXUINT, DEFAULT_COLUMNS,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_ROWS,
      g_param_spec_uint ("rows", "Rows",
          "Number of rows in the Tiled 2D output",
          1, G_MAXUINT, DEFAULT_COLUMNS,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_WIDTH,
      g_param_spec_uint ("width", "Width",
          "Width of the tiled output in pixels",
          16, G_MAXUINT, DEFAULT_WIDTH,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_HEIGHT,
      g_param_spec_uint ("height", "Height",
          "Height of the tiled output in pixels",
          16, G_MAXUINT, DEFAULT_HEIGHT,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_BUFFER_POOL_SIZE,
      g_param_spec_uint ("buffer-pool-size", "buffer-pool-size",
          "nvtiler output buffer pool size",
          4, 10, DEFAULT_BUFFER_POOL_SIZE,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_GPU_DEVICE_ID,
      g_param_spec_uint ("gpu-id", "Set GPU Device ID",
          "Set GPU Device ID",
          0, G_MAXUINT, DEFAULT_GPU_DEVICE_ID,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_SQUARE_SEQ_GRID,
      g_param_spec_boolean ("square-seq-grid", "Enable Square Sequential Grid",
          "Enable automatic square tiling according to number of sources. "
          "The tiles are placed sequentially on the grid with empty tiles at the end.\n"
          "The plugin will automatically increase/decrease tiling configuration. "
          "Rows and Columns field in the tiler is ignored if this property is enabled.",
          FALSE, (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_SHOW_SOURCE,
      g_param_spec_int ("show-source", "Show Source",
          "ID of the source to be shown. If -1 all the sources will be tiled else only a single source will be scaled into the output buffer.",
          -1, G_MAXINT, DEFAULT_SHOW_SOURCE,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_CUSTOM_TILE_CONFIG,
      g_param_spec_pointer ("custom-tile-config", "Custom Tile Configuration Structure",
          "Specifies individual tile resolution for all involved sources",
          (GParamFlags)(G_PARAM_WRITABLE | G_PARAM_STATIC_STRINGS)));

  PROP_NVBUF_MEMORY_TYPE_INSTALL (gobject_class);
  PROP_COMPUTE_HW_INSTALL(gobject_class);
  PROP_INTERPOLATION_METHOD_INSTALL(gobject_class);

  gst_element_class_add_static_pad_template (gstelement_class,
      &nvmultistreamtiler_sinkpad_template);
  gst_element_class_add_static_pad_template (gstelement_class,
      &nvmultistreamtiler_srcpad_template);

  basetrans_class->query = GST_DEBUG_FUNCPTR (gst_nvmultistreamtiler_query);
  basetrans_class->transform_caps =
      GST_DEBUG_FUNCPTR (gst_nvmultistreamtiler_transform_caps);
  basetrans_class->transform_meta =
      GST_DEBUG_FUNCPTR (gst_nvmultistreamtiler_transform_meta);
  basetrans_class->transform =
      GST_DEBUG_FUNCPTR (gst_nvmultistreamtiler_transform);
  basetrans_class->set_caps =
      GST_DEBUG_FUNCPTR (gst_nvmultistreamtiler_set_caps);
    basetrans_class->start = GST_DEBUG_FUNCPTR (gst_nvmultistreamtiler_start);
    basetrans_class->stop = GST_DEBUG_FUNCPTR (gst_nvmultistreamtiler_stop);
  basetrans_class->decide_allocation =
      GST_DEBUG_FUNCPTR (gst_nvmultistreamtiler_decide_allocation);
  basetrans_class->fixate_caps =
      GST_DEBUG_FUNCPTR (gst_nvmultistreamtiler_fixate_caps);

  basetrans_class->sink_event =
      GST_DEBUG_FUNCPTR (gst_nvmultistreamtiler_sink_event);
  basetrans_class->transform_ip_on_passthrough = FALSE;

  if (!dsmeta_quark)
    dsmeta_quark = g_quark_from_static_string (NVDS_META_STRING);
}

static void
gst_nvmultistreamtiler_init (GstNvMultiStreamTiler * nvmultistreamtiler)
{
  memset(&nvmultistreamtiler->tilerConfig, 0, sizeof(TilerConfig));
  nvmultistreamtiler->tilerConfig.rows = DEFAULT_ROWS;
  nvmultistreamtiler->tilerConfig.columns = DEFAULT_COLUMNS;
  nvmultistreamtiler->tilerConfig.width = DEFAULT_WIDTH;
  nvmultistreamtiler->tilerConfig.height = DEFAULT_HEIGHT;
  nvmultistreamtiler->cuda_mem_type = NVBUF_MEM_CUDA_DEVICE;
  nvmultistreamtiler->compute_hw = NvBufSurfTransformCompute_Default;
  nvmultistreamtiler->interpolation_method = NvBufSurfTransformInter_Nearest;
  nvmultistreamtiler->show_source = DEFAULT_SHOW_SOURCE;
  nvmultistreamtiler->tilerIface = nullptr;
  nvmultistreamtiler->tilerScratchBuffer = nullptr;
  nvmultistreamtiler->buffer_pool_size = DEFAULT_BUFFER_POOL_SIZE;
  nvmultistreamtiler->square_grid = FALSE;
  g_mutex_init(&nvmultistreamtiler->tilerIfaceLock);
  g_mutex_init(&nvmultistreamtiler->tiler_map_lock);
  nvmultistreamtiler->tiler_map = new std::map<uint32_t, uint32_t>;

  NvBufSurfaceDeviceInfo dev_info{};
  if (NvBufSurfaceGetDeviceInfo(&dev_info) == 0) {
    if (dev_info.driverType == NVBUF_DRIVER_TYPE_NVGPU) {
      nvmultistreamtiler->cuda_mem_type = NVBUF_MEM_SURFACE_ARRAY;
    }
  }
}

#ifndef PACKAGE
#define PACKAGE "nvmultistreamTiler"
#endif

#define VERSION "1.0"
#define LICENSE "Proprietary"
#define DESCRIPTION "NVIDIA Multistream Tiler plugin"
#define BINARY_PACKAGE "NVIDIA Multistream Plugins"
#define URL "http://nvidia.com/"


static gboolean
plugin_init (GstPlugin * plugin)
{
  if (!gst_element_register (plugin, "nvmultistreamtiler", GST_RANK_PRIMARY,
          GST_TYPE_NVMULTISTREAMTILER))
    return FALSE;

  return TRUE;
}

#ifdef ENABLE_GST_NVTILER_UNIT_TESTS
gboolean gGstNvTilerStaticInit()
{
  return gst_plugin_register_static(GST_VERSION_MAJOR, GST_VERSION_MINOR,
                 "nvmultistreamtiler",
                 DESCRIPTION, plugin_init, DS_VERSION, LICENSE, BINARY_PACKAGE, PACKAGE, URL);
}
#else
GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    nvdsgst_multistreamtiler,
    DESCRIPTION, plugin_init, DS_VERSION, LICENSE, BINARY_PACKAGE, URL)
#endif

CustomTileConfig* nvmultistreamtiler_dup_custom_tile_config(CustomTileConfig* tileC)
{
    if(!tileC || !tileC->tiles)
    {
        return nullptr;
    }
    CustomTileConfig* customTileConfig = static_cast<CustomTileConfig*>(g_memdup2(tileC, sizeof(CustomTileConfig)));
    if(customTileConfig)
    {
        customTileConfig->tiles = static_cast<CustomTile*>(g_memdup2(tileC->tiles, tileC->length * sizeof(CustomTile)));
    }
    return customTileConfig;
}
