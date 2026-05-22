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

#include "gst-nvmultiurisrcbincreator.h"
#include "gst-nvdscustommessage.h"
#include "nvds_rest_server.h"
#include "gst-nvquery.h"
#include "gst-nvquery-internal.h"
#include "nvdsmeta.h"
#include "gstnvdsmeta.h"
#include <thread>

//#define NVMULTIURISRCBIN_CREATOR_DEBUG
#ifndef NVMULTIURISRCBIN_CREATOR_DEBUG
#define LOGD(...)
#else
#define LOGD(fmt, ...) g_print("[DEBUG %s %d] " fmt, __func__, __LINE__, ## __VA_ARGS__)
#endif

typedef struct
{
  GstDsNvUriSrcConfig *config;
  GstElement *uribin;
  NvDst_Handle_NvMultiUriSrcCreator apiHandle;
  GstPad *muxSinkPad;
  GstPad *uribin_src_pad;
  gulong probe_eos_handling;
  gulong newPadAddedHandler;
  gulong elemRemovedHandler;
  gboolean is_removing;  /* Flag to indicate source is being removed, prevents race conditions in callbacks */
} NvDsUriSourceInfo;

typedef struct
{
  //List if N X nvurisrcbin (NvDsUriSourceInfo) instances
  GList *sourceInfoList;
  //Hash of sourceBin instances from sourceId
  GHashTable *sourceInfoHash;
  GstElement *streammux;
  //Parent bin for N X sources -> nvstreammux
  GstElement *nvmultiurisrcbin;

  //mode can either be video or audio
  NvDsMultiUriMode mode;
  guint numOfActiveSources;
  gboolean using_new_mux;

  GstDsNvStreammuxConfig *muxConfig;
  /*lock to protect book keeping data structures:
   * sourceInfoHash, sourceInfoList, numOfActiveSources
   * from possible data race with add/remove/probe_eos_handling API calls
   */
  GMutex lock;

  /* Thread to handle uribin , sourceinfo removal
   * and release nvstreammux sinkpad on EOS.
   */
  GThread *uribin_removal_thread;
  //Queue to hold sourceinfo which would be released on EOS.
  GQueue *remove_uribin_queue;
  GCond remove_uribin_cond;
  gboolean uribin_removal_thread_stop;
  GMutex uribin_removal_lock;
  guint base_index;

} NvMultiUriSrcBinCreator;

#define NVGSTDS_BIN_ADD_GHOST_PAD_NAMED(bin, elem, pad, ghost_pad_name) \
  do { \
    GstPad *gstpad = gst_element_get_static_pad (elem, pad); \
    if (gstpad) { \
      gst_element_add_pad (bin, gst_ghost_pad_new (ghost_pad_name, gstpad)); \
      gst_object_unref (gstpad); \
    } \
  } while (0)

#define NVGSTDS_BIN_ADD_GHOST_PAD(bin, elem, pad) \
      NVGSTDS_BIN_ADD_GHOST_PAD_NAMED (bin, elem, pad, pad)

#define NVGSTDS_ELEM_ADD_PROBE(parent_elem, elem, pad, probe_func, probe_type, probe_data) \
    ({ \
      gulong probe_id = 0; \
      GstPad *gstpad = gst_element_get_static_pad (elem, pad); \
      if (!gstpad) { \
        GST_ELEMENT_ERROR(parent_elem, RESOURCE, FAILED, \
            ("Could not find '%s' in '%s'", pad, \
            GST_ELEMENT_NAME(elem)), (NULL)); \
      } else { \
        probe_id = gst_pad_add_probe(gstpad, (GstPadProbeType) (probe_type), probe_func, probe_data, NULL); \
        gst_object_unref (gstpad); \
      } \
      probe_id; \
    })

static gboolean
s_nvmultiurisrcbincreator_link_element_to_streammux_sink_pad (NvDsUriSourceInfo
    * sourceInfo, GstElement * streammux, GstPad * src_pad, gint index);

static void s_nvmultiurisrcbincreator_set_properties_nvuribin (GstElement *
    element_, GstDsNvUriSrcConfig const *config);

static GstDsNvStreammuxConfig
    * s_nvmultiurisrcbincreator_create_mux_config (GstDsNvStreammuxConfig *
    muxConfig);

static void s_nvmultiurisrcbincreator_destroy_mux_config (GstDsNvStreammuxConfig
    * muxConfig);

static GstPadProbeReturn
s_nvmultiurisrcbincreator_probe_func_eos_handling (GstPad * pad,
    GstPadProbeInfo * info, gpointer u_data);

static GstPadProbeReturn
s_nvmultiurisrcbincreator_probe_func_add_sensorInfo (GstPad * pad,
    GstPadProbeInfo * info, gpointer u_data);

static void
s_nvmultiurisrcbincreator_remove_source_info (NvMultiUriSrcBinCreator *
    nvmultiurisrcbinCreator, NvDsUriSourceInfo * sourceInfo);

static void s_nvmultiurisrcbincreator_add_source_info_handlers
    (NvDsUriSourceInfo * sourceInfo);

static void s_nvmultiurisrcbincreator_remove_source_info_handlers
    (NvDsUriSourceInfo * sourceInfo);

static gboolean
s_nvmultiurisrcbincreator_remove_source_impl (NvDst_Handle_NvMultiUriSrcCreator
    apiHandle, guint sourceId, gboolean forceSourceStateChange);

static gpointer s_uribin_removal_thread (gpointer data);

gint s_get_source_id (NvMultiUriSrcBinCreator * nvmultiurisrcbinCreator, GstDsNvUriSrcConfig * sourceConfig);

static GstBus *
s_nvmultiurisrcbincreator_get_bus_from_parent (NvMultiUriSrcBinCreator *
    nvmultiurisrcbinCreator)
{
  GstObject *parent = NULL;
  GstBus *bus = NULL;
  if (!GST_IS_BIN (nvmultiurisrcbinCreator->nvmultiurisrcbin)) {
    return bus;
  }
  if (!(parent =
          gst_object_get_parent (GST_OBJECT
              (nvmultiurisrcbinCreator->nvmultiurisrcbin)))) {
  } else {
    bus = gst_element_get_bus ((GstElement *) parent);
    gst_object_unref (parent);
  }
  return bus;
}

gboolean
find_source (NvDst_Handle_NvMultiUriSrcCreator apiHandle, guint sourceId)
{
  NvMultiUriSrcBinCreator *nvmultiurisrcbinCreator =
      (NvMultiUriSrcBinCreator *) apiHandle;

  NvDsUriSourceInfo *sourceInfo =
      (NvDsUriSourceInfo *)
      g_hash_table_lookup (nvmultiurisrcbinCreator->sourceInfoHash,
      sourceId + (gchar *) NULL);

  if (!sourceInfo) {
    //No source found
    g_print ("[WARN] No source found .. !! \n");
    return FALSE;
  }
  return TRUE;
}

guint get_gpu_id(NvDst_Handle_NvMultiUriSrcCreator apiHandle)
{
  NvMultiUriSrcBinCreator *nvmultiurisrcbinCreator =
      (NvMultiUriSrcBinCreator *) apiHandle;

  if (nvmultiurisrcbinCreator->muxConfig) {
    return nvmultiurisrcbinCreator->muxConfig->gpu_id;
  }
  return 0;
}

gchar *gst_nvmultiurisrcbincreator_get_sensor_id_from_source_id(NvDst_Handle_NvMultiUriSrcCreator apiHandle, guint sourceId)
{
  NvMultiUriSrcBinCreator *nvmultiurisrcbinCreator =
      (NvMultiUriSrcBinCreator *) apiHandle;

  g_mutex_lock (&nvmultiurisrcbinCreator->lock);
  NvDsUriSourceInfo *sourceInfo =
      (NvDsUriSourceInfo *)
      g_hash_table_lookup (nvmultiurisrcbinCreator->sourceInfoHash,
      sourceId + (gchar *) NULL);

  if (!sourceInfo) {
    //No source found
    g_print ("[WARN] No source found .. !! \n");
    g_mutex_unlock (&nvmultiurisrcbinCreator->lock);
    return NULL;
  }

  gchar *sensor_id = sourceInfo->config->sensorId;
  g_mutex_unlock (&nvmultiurisrcbinCreator->lock);
  return sensor_id;
}

gboolean
s_force_eos_handle (NvDst_Handle_NvMultiUriSrcCreator apiHandle,
    NvDsServerAppInstanceInfo * appinstance_info)
{

  NvMultiUriSrcBinCreator *nvmultiurisrcbinCreator =
      (NvMultiUriSrcBinCreator *) apiHandle;

  g_object_set (G_OBJECT (nvmultiurisrcbinCreator->streammux),
      "drop-pipeline-eos", FALSE, NULL);
  gboolean force_eos;

  force_eos = appinstance_info->app_quit;

  if (force_eos) {
    GstBus *bus =
        s_nvmultiurisrcbincreator_get_bus_from_parent (nvmultiurisrcbinCreator);
    if (bus) {
      gst_bus_post (bus,
          gst_nvmessage_force_pipeline_eos (GST_OBJECT
              (gst_nvmultiurisrcbincreator_get_bin (nvmultiurisrcbinCreator)),
              force_eos));
      gst_object_unref (bus);
    }
  }

  return TRUE;
}

gboolean
set_nvuribin_mux_prop (NvDst_Handle_NvMultiUriSrcCreator apiHandle,
    NvDsServerMuxInfo * mux_info)
{

  NvMultiUriSrcBinCreator *nvmultiurisrcbinCreator =
      (NvMultiUriSrcBinCreator *) apiHandle;

  const gchar *new_mux_str = g_getenv ("USE_NEW_NVSTREAMMUX");
  gboolean use_new_mux = !g_strcmp0 (new_mux_str, "yes");

  if (mux_info->mux_flag == BATCHED_PUSH_TIMEOUT && !use_new_mux) {
    g_object_set (G_OBJECT (nvmultiurisrcbinCreator->streammux),
        "batched-push-timeout", mux_info->batched_push_timeout, NULL);
  }
  if (mux_info->mux_flag == MAX_LATENCY) {
    g_object_set (G_OBJECT (nvmultiurisrcbinCreator->streammux), "max-latency",
        mux_info->max_latency, NULL);
  }

  return TRUE;
}

gboolean
set_nvuribin_conv_prop (NvDst_Handle_NvMultiUriSrcCreator apiHandle,
    guint sourceId, NvDsServerConvInfo * conv_info)
{
  NvMultiUriSrcBinCreator *nvmultiurisrcbinCreator =
      (NvMultiUriSrcBinCreator *) apiHandle;

  NvDsUriSourceInfo *sourceInfo =
      (NvDsUriSourceInfo *)
      g_hash_table_lookup (nvmultiurisrcbinCreator->sourceInfoHash,
      sourceId + (gchar *) NULL);

  if (!sourceInfo) {
    //No source found
    g_print ("[WARN] No source found .. !! \n");
    return FALSE;
  }

  GstElement *uribin = sourceInfo->uribin;
  gboolean ret = TRUE;

  GstIterator *iter = NULL;
  GValue value = { 0 };
  GstElement *elem = NULL;
  gboolean done = FALSE;
  iter = gst_bin_iterate_elements (GST_BIN (uribin));
  GstElementFactory *factory = NULL;

  while (!done) {
    switch (gst_iterator_next (iter, &value)) {
      case GST_ITERATOR_OK:
        elem = (GstElement *) g_value_get_object (&value);
        factory = GST_ELEMENT_GET_CLASS (elem)->elementfactory;

        if (!g_strcmp0 (GST_OBJECT_NAME (factory), "nvvideoconvert")) {
          if (conv_info->conv_flag == SRC_CROP) {
            g_object_set (G_OBJECT (elem), "src-crop",
                conv_info->src_crop.c_str (), NULL);
          } else if (conv_info->conv_flag == DEST_CROP) {
            g_object_set (G_OBJECT (elem), "dest-crop",
                conv_info->dest_crop.c_str (), NULL);
          } else if (conv_info->conv_flag == FLIP_METHOD) {
            g_object_set (G_OBJECT (elem), "flip-method",
                conv_info->flip_method, NULL);
          } else if (conv_info->conv_flag == INTERPOLATION_METHOD) {
            g_object_set (G_OBJECT (elem), "interpolation-method",
                conv_info->interpolation_method, NULL);
          }
        }
        g_value_unset (&value);
        break;
      case GST_ITERATOR_RESYNC:
        gst_iterator_resync (iter);
        break;
      case GST_ITERATOR_ERROR:
        GST_WARNING_OBJECT (GST_BIN (uribin), "error in iterating elements");
        done = TRUE;
        ret = FALSE;
        break;
      case GST_ITERATOR_DONE:
        done = TRUE;
        break;
    }
  }
  return ret;
}

gboolean
set_nvuribin_dec_prop (NvDst_Handle_NvMultiUriSrcCreator apiHandle,
    guint sourceId, NvDsServerDecInfo * dec_info)
{
  NvMultiUriSrcBinCreator *nvmultiurisrcbinCreator =
      (NvMultiUriSrcBinCreator *) apiHandle;

  NvDsUriSourceInfo *sourceInfo =
      (NvDsUriSourceInfo *)
      g_hash_table_lookup (nvmultiurisrcbinCreator->sourceInfoHash,
      sourceId + (gchar *) NULL);

  if (!sourceInfo) {
    //No source found
    g_print ("[WARN] No source found .. !! \n");
    return FALSE;
  }

  GstElement *uribin = sourceInfo->uribin;
  gboolean ret = TRUE;

  GstIterator *iter = NULL;
  GValue value = { 0 };
  GstElement *elem = NULL;
  gboolean done = FALSE;
  iter = gst_bin_iterate_elements (GST_BIN (uribin));

  GstIterator *itr2;
  GValue value2 = { 0 };
  GstElementFactory *factory = NULL;
  while (!done) {
    switch (gst_iterator_next (iter, &value)) {
      case GST_ITERATOR_OK:
        elem = (GstElement *) g_value_get_object (&value);
        factory = GST_ELEMENT_GET_CLASS (elem)->elementfactory;
        if (!g_strcmp0 (GST_OBJECT_NAME (factory), "uridecodebin")) {
          /* Elements inside  uridecodebin */
          for (itr2 = gst_bin_iterate_elements (GST_BIN ((elem)));
              gst_iterator_next (itr2, &value2) == GST_ITERATOR_OK;
              g_value_reset (&value2)) {
            GstElement *elem2 = (GstElement *) g_value_get_object (&value2);
            GstElementFactory *factory2 =
                GST_ELEMENT_GET_CLASS (elem2)->elementfactory;

            GstIterator *itr3;
            GValue value3 = { 0 };

            if (!g_strcmp0 (GST_OBJECT_NAME (factory2), "decodebin")) {
              for (itr3 = gst_bin_iterate_elements (GST_BIN ((elem2)));
                  gst_iterator_next (itr3, &value3) == GST_ITERATOR_OK;
                  g_value_reset (&value3)) {
                GstElement *elem3 = (GstElement *) g_value_get_object (&value3);
                GstElementFactory *factory3 =
                    GST_ELEMENT_GET_CLASS (elem3)->elementfactory;

                if (!g_strcmp0 (GST_OBJECT_NAME (factory3), "nvv4l2decoder")) {
                  if (dec_info->dec_flag == DROP_FRAME_INTERVAL) {
                    g_object_set (G_OBJECT (elem3), "drop-frame-interval",
                        dec_info->drop_frame_interval, NULL);
                  } else if (dec_info->dec_flag == SKIP_FRAMES) {
                    /* TODO: Define skip-frame enum in rest server header */
                    g_object_set (G_OBJECT (elem3), "skip-frames",
                        dec_info->skip_frames, NULL);
                  } else if (dec_info->dec_flag == LOW_LATENCY_MODE) {
                    g_object_set (G_OBJECT (elem3), "low-latency-mode",
                        dec_info->low_latency_mode, NULL);
                  }
                }
              }
            }
          }
        } else if (!g_strcmp0 (GST_OBJECT_NAME (factory), "decodebin")) {
          /* Elements inside  decodebin */
          for (itr2 = gst_bin_iterate_elements (GST_BIN ((elem)));
              gst_iterator_next (itr2, &value2) == GST_ITERATOR_OK;
              g_value_reset (&value2)) {
            GstElement *elem2 = (GstElement *) g_value_get_object (&value2);
            GstElementFactory *factory2 =
                GST_ELEMENT_GET_CLASS (elem2)->elementfactory;

            if (!g_strcmp0 (GST_OBJECT_NAME (factory2), "nvv4l2decoder")) {
              if (dec_info->dec_flag == DROP_FRAME_INTERVAL) {
                g_object_set (G_OBJECT (elem2), "drop-frame-interval",
                    dec_info->drop_frame_interval, NULL);
              } else if (dec_info->dec_flag == SKIP_FRAMES) {
                /* TODO: Define skip-frame enum in rest server header */
                g_object_set (G_OBJECT (elem2), "skip-frames",
                    dec_info->skip_frames, NULL);
              } else if (dec_info->dec_flag == LOW_LATENCY_MODE) {
                g_object_set (G_OBJECT (elem2), "low-latency-mode",
                    dec_info->low_latency_mode, NULL);
              }
            }
          }
        }
        g_value_unset (&value);
        break;
      case GST_ITERATOR_RESYNC:
        gst_iterator_resync (iter);
        break;
      case GST_ITERATOR_ERROR:
        GST_WARNING_OBJECT (GST_BIN (uribin), "error in iterating elements");
        done = TRUE;
        ret = FALSE;
        break;
      case GST_ITERATOR_DONE:
        done = TRUE;
        break;
    }
  }
  return ret;
}

static GstPadProbeReturn
src_pad_query_probe (GstPad * pad, GstPadProbeInfo * info, gpointer data)
{
  NvMultiUriSrcBinCreator *nvmultiurisrcbinCreator =
      (NvMultiUriSrcBinCreator *) data;

  if (info->type & GST_PAD_PROBE_TYPE_QUERY_UPSTREAM) {
    GstQuery *query = GST_QUERY (info->data);
    if (gst_nvquery_is_batch_size (query) && nvmultiurisrcbinCreator->muxConfig) {
      gst_nvquery_batch_size_set (query,
          nvmultiurisrcbinCreator->muxConfig->maxBatchSize);
      return GST_PAD_PROBE_HANDLED;
    }
  }
  return GST_PAD_PROBE_OK;
}

NvDst_Handle_NvMultiUriSrcCreator
gst_nvmultiurisrcbincreator_init (gchar * binName, NvDsMultiUriMode mode,
    GstDsNvStreammuxConfig * muxConfig)
{
  NvMultiUriSrcBinCreator *nvmultiurisrcbinCreator =
      (NvMultiUriSrcBinCreator *) g_malloc0 (sizeof (NvMultiUriSrcBinCreator));
  nvmultiurisrcbinCreator->mode = mode;
  nvmultiurisrcbinCreator->numOfActiveSources = 0;
  nvmultiurisrcbinCreator->base_index = 0;
  if (muxConfig) {
    nvmultiurisrcbinCreator->muxConfig =
        s_nvmultiurisrcbincreator_create_mux_config (muxConfig);
  }

  const gchar *new_mux_str = g_getenv ("USE_NEW_NVSTREAMMUX");
  nvmultiurisrcbinCreator->using_new_mux = !g_strcmp0 (new_mux_str, "yes");

  /*
   * Create nvstreammux instance
   * Note: nvmultiurisrcbin support only new nvstreammux to take advantage
   * of adaptive batching required for dynamic sensor provisioning
   */
  nvmultiurisrcbinCreator->streammux =
      gst_element_factory_make ("nvstreammux", "src_bin_muxer");

  if (nvmultiurisrcbinCreator->muxConfig) {
    if (!nvmultiurisrcbinCreator->using_new_mux) {
      //set properties exclusive for legacy nvstreammux
      if (!muxConfig->pipeline_width || !muxConfig->pipeline_height) {
        GST_ERROR_OBJECT (nvmultiurisrcbinCreator->streammux,
            "[FATAL ERROR] Mandatory fields not set; pipeline will fail; "
            "width=%d height=%d\n",
            muxConfig->pipeline_width, muxConfig->pipeline_height);
      }
      g_object_set (nvmultiurisrcbinCreator->streammux, "width",
          muxConfig->pipeline_width, "height", muxConfig->pipeline_height,
          "batched-push-timeout", muxConfig->batched_push_timeout, "batch-size",
          muxConfig->maxBatchSize,
          /* Note: set batch-size to 1 and let add/remove APIs update
           * This logic cause pipeline slowdown and hence the maxBatchSize
           * usage.
           * TODO; Tracked
           */
          NULL);
      if (nvmultiurisrcbinCreator->muxConfig->buffer_pool_size) {
        g_object_set (nvmultiurisrcbinCreator->streammux,
            "buffer-pool-size",
            nvmultiurisrcbinCreator->muxConfig->buffer_pool_size, NULL);
      }
      if (nvmultiurisrcbinCreator->muxConfig->extract_sei_type5_data) {
        g_object_set (nvmultiurisrcbinCreator->streammux,
            "extract-sei-type5-data",
            nvmultiurisrcbinCreator->muxConfig->extract_sei_type5_data, NULL);
      }
      if (nvmultiurisrcbinCreator->muxConfig->compute_hw) {
        g_object_set (nvmultiurisrcbinCreator->streammux,
            "compute-hw", nvmultiurisrcbinCreator->muxConfig->compute_hw, NULL);
      }
      if (nvmultiurisrcbinCreator->muxConfig->interpolation_method) {
        g_object_set (nvmultiurisrcbinCreator->streammux,
            "interpolation-method",
            nvmultiurisrcbinCreator->muxConfig->interpolation_method, NULL);
      }
      if (nvmultiurisrcbinCreator->muxConfig->gpu_id) {
        g_object_set (nvmultiurisrcbinCreator->streammux,
            "gpu-id", nvmultiurisrcbinCreator->muxConfig->gpu_id, NULL);
      }
      g_object_set (nvmultiurisrcbinCreator->streammux,
          "nvbuf-memory-type",
          nvmultiurisrcbinCreator->muxConfig->nvbuf_memory_type, NULL);
      if (nvmultiurisrcbinCreator->muxConfig->live_source) {
        g_object_set (nvmultiurisrcbinCreator->streammux,
            "live-source", nvmultiurisrcbinCreator->muxConfig->live_source,
            NULL);
      }
      if (nvmultiurisrcbinCreator->muxConfig->enable_padding) {
        g_object_set (nvmultiurisrcbinCreator->streammux,
            "enable-padding",
            nvmultiurisrcbinCreator->muxConfig->enable_padding, NULL);
      }

      if (nvmultiurisrcbinCreator->muxConfig->async_process) {
        g_object_set (nvmultiurisrcbinCreator->streammux,
            "async-process", nvmultiurisrcbinCreator->muxConfig->async_process,
            NULL);
      }
      if (nvmultiurisrcbinCreator->muxConfig->sort_batch) {
        g_object_set (nvmultiurisrcbinCreator->streammux,
            "sort-batch", nvmultiurisrcbinCreator->muxConfig->sort_batch,
            NULL);
      }
      if (nvmultiurisrcbinCreator->muxConfig->buffer_cache) {
        g_object_set (nvmultiurisrcbinCreator->streammux,
            "cache-buffer", nvmultiurisrcbinCreator->muxConfig->buffer_cache,
            NULL);
      }
      if (nvmultiurisrcbinCreator->muxConfig->buffer_cache_timeout) {
        g_object_set (nvmultiurisrcbinCreator->streammux,
            "cache-buffer-timeout", nvmultiurisrcbinCreator->muxConfig->buffer_cache_timeout,
            NULL);
      }
      if (nvmultiurisrcbinCreator->muxConfig->extract_sei_sim_time) {
        g_object_set (nvmultiurisrcbinCreator->streammux,
            "extract-sei-sim-time", nvmultiurisrcbinCreator->muxConfig->extract_sei_sim_time,
            NULL);
      }
		if (nvmultiurisrcbinCreator->muxConfig->align_first_buffer) {
        g_object_set (nvmultiurisrcbinCreator->streammux,
            "align-first-buffer", nvmultiurisrcbinCreator->muxConfig->align_first_buffer,
            NULL);
      }
		if (nvmultiurisrcbinCreator->muxConfig->sync_inputs_ntp) {
        g_object_set (nvmultiurisrcbinCreator->streammux,
            "sync-inputs-ntp", nvmultiurisrcbinCreator->muxConfig->sync_inputs_ntp,
            NULL);
      }
      if (nvmultiurisrcbinCreator->muxConfig->drop_backward_sei) {
        g_object_set (nvmultiurisrcbinCreator->streammux,
            "drop-backward-sei", nvmultiurisrcbinCreator->muxConfig->drop_backward_sei,
            NULL);
      }
    } else {
      //using new nvstreammux
      if (nvmultiurisrcbinCreator->muxConfig->config_file_path) {
        g_object_set (nvmultiurisrcbinCreator->streammux,
            "config-file-path",
            nvmultiurisrcbinCreator->muxConfig->config_file_path, NULL);
      }
      g_object_set (nvmultiurisrcbinCreator->streammux,
          "batch-size",
          nvmultiurisrcbinCreator->muxConfig->maxBatchSize, NULL);
    }

    //common to both

    if (nvmultiurisrcbinCreator->muxConfig->num_surfaces_per_frame) {
      g_object_set (nvmultiurisrcbinCreator->streammux,
          "num-surfaces-per-frame",
          nvmultiurisrcbinCreator->muxConfig->num_surfaces_per_frame, NULL);
    }

    g_object_set (nvmultiurisrcbinCreator->streammux,
        "attach-sys-ts",
        nvmultiurisrcbinCreator->muxConfig->attach_sys_ts_as_ntp, NULL);

    if (nvmultiurisrcbinCreator->muxConfig->sync_inputs) {
      g_object_set (nvmultiurisrcbinCreator->streammux,
          "sync-inputs", nvmultiurisrcbinCreator->muxConfig->sync_inputs, NULL);
    }
    if (nvmultiurisrcbinCreator->muxConfig->max_latency) {
      g_object_set (nvmultiurisrcbinCreator->streammux,
          "max-latency", nvmultiurisrcbinCreator->muxConfig->max_latency, NULL);
    }
    if (nvmultiurisrcbinCreator->muxConfig->frame_num_reset_on_eos) {
      g_object_set (nvmultiurisrcbinCreator->streammux,
          "frame-num-reset-on-eos",
          nvmultiurisrcbinCreator->muxConfig->frame_num_reset_on_eos, NULL);
    }
    if (nvmultiurisrcbinCreator->muxConfig->frame_num_reset_on_stream_reset) {
      g_object_set (nvmultiurisrcbinCreator->streammux,
          "frame-num-reset-on-stream-reset",
          nvmultiurisrcbinCreator->muxConfig->frame_num_reset_on_stream_reset,
          NULL);
    }
    if (nvmultiurisrcbinCreator->muxConfig->frame_duration) {
      g_object_set (nvmultiurisrcbinCreator->streammux,
          "frame-duration", nvmultiurisrcbinCreator->muxConfig->frame_duration,
          NULL);
    }

    g_object_set (nvmultiurisrcbinCreator->streammux,
        "drop-pipeline-eos",
        nvmultiurisrcbinCreator->muxConfig->no_pipeline_eos, NULL);
  }

  nvmultiurisrcbinCreator->sourceInfoHash = g_hash_table_new (NULL, NULL);

  //Create parent bin
  nvmultiurisrcbinCreator->nvmultiurisrcbin =
      gst_element_factory_make ("bin", binName);

  if (!nvmultiurisrcbinCreator->nvmultiurisrcbin) {
    GST_WARNING_OBJECT (nvmultiurisrcbinCreator->nvmultiurisrcbin,
        "bin creation failed\n");
    g_free(nvmultiurisrcbinCreator);
    return (NvDst_Handle_NvMultiUriSrcCreator) NULL;
  }

  g_object_set (nvmultiurisrcbinCreator->nvmultiurisrcbin,
      "async-handling", TRUE, NULL);

  //Add nvstreammux to this bin
  gst_bin_add (GST_BIN (nvmultiurisrcbinCreator->nvmultiurisrcbin),
      nvmultiurisrcbinCreator->streammux);

  //Add ghost src pad for the bin
  NVGSTDS_BIN_ADD_GHOST_PAD (nvmultiurisrcbinCreator->nvmultiurisrcbin,
      nvmultiurisrcbinCreator->streammux, "src");

  NVGSTDS_ELEM_ADD_PROBE (nvmultiurisrcbinCreator->nvmultiurisrcbin,
      nvmultiurisrcbinCreator->streammux, "src", src_pad_query_probe,
      GST_PAD_PROBE_TYPE_QUERY_BOTH, nvmultiurisrcbinCreator);

  NVGSTDS_ELEM_ADD_PROBE (nvmultiurisrcbinCreator->nvmultiurisrcbin,
      nvmultiurisrcbinCreator->streammux, "src", s_nvmultiurisrcbincreator_probe_func_add_sensorInfo,
      GST_PAD_PROBE_TYPE_BUFFER, nvmultiurisrcbinCreator);
  nvmultiurisrcbinCreator->remove_uribin_queue = g_queue_new ();

  g_cond_init (&nvmultiurisrcbinCreator->remove_uribin_cond);
  nvmultiurisrcbinCreator->uribin_removal_thread_stop = FALSE;

  g_mutex_init (&nvmultiurisrcbinCreator->lock);
  g_mutex_init (&nvmultiurisrcbinCreator->uribin_removal_lock);

  nvmultiurisrcbinCreator->uribin_removal_thread =
      g_thread_new ("nvmultiurisrcbincreator-uribin-removal-thread",
      s_uribin_removal_thread, nvmultiurisrcbinCreator);

  return (NvDst_Handle_NvMultiUriSrcCreator) nvmultiurisrcbinCreator;
}

void
gst_nvmultiurisrcbincreator_deinit (NvDst_Handle_NvMultiUriSrcCreator apiHandle)
{
  NvMultiUriSrcBinCreator *nvmultiurisrcbinCreator =
      (NvMultiUriSrcBinCreator *) apiHandle;

  //Remove all remaining sources
  g_mutex_lock (&nvmultiurisrcbinCreator->lock);
  GList *sourceIds = NULL;
  for (GList *node = nvmultiurisrcbinCreator->sourceInfoList; node; node = g_list_next(node)) {
    NvDsUriSourceInfo *sourceInfo = (NvDsUriSourceInfo *) (node->data);
    /* Skip sources that are already being removed */
    if (sourceInfo && !sourceInfo->is_removing) {
      sourceIds = g_list_append(sourceIds, GUINT_TO_POINTER(sourceInfo->config->source_id));
    }
  }
  g_mutex_unlock (&nvmultiurisrcbinCreator->lock);

  // Now remove sources using the collected IDs
  for (GList *node = sourceIds; node; node = g_list_next(node)) {
    guint sourceId = GPOINTER_TO_UINT(node->data);
    gst_nvmultiurisrcbincreator_remove_source (apiHandle, sourceId);
  }

  g_list_free(sourceIds);

  /* Stop the uribin removal thread before acquiring lock to avoid deadlock.
   * The removal thread may be blocked trying to acquire lock for uribin_removal_lock
   * Signal it to stop and wait for it to exit before holding lock to prevent deadlock.
   */
  if (nvmultiurisrcbinCreator->uribin_removal_thread) {
    g_mutex_lock (&nvmultiurisrcbinCreator->uribin_removal_lock);
    nvmultiurisrcbinCreator->uribin_removal_thread_stop = TRUE;
    g_cond_broadcast (&nvmultiurisrcbinCreator->remove_uribin_cond);
    g_mutex_unlock (&nvmultiurisrcbinCreator->uribin_removal_lock);
    g_thread_join (nvmultiurisrcbinCreator->uribin_removal_thread);
  }

  /* Now it's safe to acquire lock for final cleanup */
  g_mutex_lock (&nvmultiurisrcbinCreator->lock);
  /* Cannot unref the bin: nvmultiurisrcbinCreator->nvmultiurisrcbin;
   * The floating ref on this bin will be unref'd only when parent
   * pipeline is unref'd
   */

  s_nvmultiurisrcbincreator_destroy_mux_config
      (nvmultiurisrcbinCreator->muxConfig);

  g_cond_clear (&nvmultiurisrcbinCreator->remove_uribin_cond);
  g_queue_free (nvmultiurisrcbinCreator->remove_uribin_queue);
  g_mutex_unlock (&nvmultiurisrcbinCreator->lock);
  g_mutex_clear (&nvmultiurisrcbinCreator->lock);
  g_mutex_clear (&nvmultiurisrcbinCreator->uribin_removal_lock);
  g_free (nvmultiurisrcbinCreator);
}

static gpointer
s_uribin_removal_thread (gpointer data)
{

  NvMultiUriSrcBinCreator *nvmultiurisrcbinCreator =
      (NvMultiUriSrcBinCreator *) data;

  g_mutex_lock (&nvmultiurisrcbinCreator->uribin_removal_lock);

  while (!nvmultiurisrcbinCreator->uribin_removal_thread_stop) {

    if (g_queue_is_empty (nvmultiurisrcbinCreator->remove_uribin_queue))
      g_cond_wait (&nvmultiurisrcbinCreator->remove_uribin_cond,
          &nvmultiurisrcbinCreator->uribin_removal_lock);

    while (!g_queue_is_empty (nvmultiurisrcbinCreator->remove_uribin_queue)) {

      NvDsUriSourceInfo *sourceInfo =
          (NvDsUriSourceInfo *) g_queue_pop_head (nvmultiurisrcbinCreator->
          remove_uribin_queue);
      g_mutex_unlock (&nvmultiurisrcbinCreator->uribin_removal_lock);

      if (GST_IS_PAD (sourceInfo->muxSinkPad)) {
        gst_pad_send_event (sourceInfo->muxSinkPad,
            gst_event_new_flush_stop (FALSE));
        gst_element_release_request_pad (nvmultiurisrcbinCreator->streammux,
            sourceInfo->muxSinkPad);
        gst_object_unref (sourceInfo->muxSinkPad);
      }

      GstElement *uribin = sourceInfo->uribin;
      g_object_ref (uribin);
      if ((!gst_bin_remove (GST_BIN (nvmultiurisrcbinCreator->nvmultiurisrcbin),
                  uribin))) {
        GST_WARNING_OBJECT (nvmultiurisrcbinCreator->nvmultiurisrcbin,
            "Failed to set remove source-id:%u", sourceInfo->config->source_id);
        return NULL;
      }

      GstStateChangeReturn state_return = GST_STATE_CHANGE_FAILURE;
      if (GST_IS_BIN (uribin)
          && (state_return =
              gst_element_set_state (GST_ELEMENT (uribin),
                  GST_STATE_NULL)) == GST_STATE_CHANGE_FAILURE) {
        GST_WARNING_OBJECT (nvmultiurisrcbinCreator->nvmultiurisrcbin,
            "Failed to set stop source-id:%u", sourceInfo->config->source_id);
        return FALSE;
      }
      gst_object_unref (uribin);

      /* Remove sourceInfo from the hash map and the list, free sourceInfo.
         This ensure the sourceInfo is removed from the list and the hash map for EOS usecases.
         Hence if we use a REST API call to get the sourceInfo list for specific endpoint
        implementation, it will be updated correctly
      */
      g_mutex_lock (&nvmultiurisrcbinCreator->lock);
      s_nvmultiurisrcbincreator_remove_source_info (nvmultiurisrcbinCreator, sourceInfo);
      g_mutex_unlock (&nvmultiurisrcbinCreator->lock);

      g_mutex_lock (&nvmultiurisrcbinCreator->uribin_removal_lock);
    }
  }

  g_mutex_unlock (&nvmultiurisrcbinCreator->uribin_removal_lock);
  return NULL;
}

static GstPadProbeReturn
s_nvmultiurisrcbincreator_probe_func_eos_handling (GstPad * pad,
    GstPadProbeInfo * info, gpointer u_data)
{
  NvDsUriSourceInfo *sourceInfo = (NvDsUriSourceInfo *) u_data;
  NvMultiUriSrcBinCreator *nvmultiurisrcbinCreator =
      (NvMultiUriSrcBinCreator *) sourceInfo->apiHandle;

  g_mutex_lock (&nvmultiurisrcbinCreator->lock);

  /* Check if source is already being removed to prevent race conditions */
  if (sourceInfo->is_removing) {
    GST_DEBUG ("EOS probe: source already being removed, skipping");
    g_mutex_unlock (&nvmultiurisrcbinCreator->lock);
    return GST_PAD_PROBE_REMOVE;
  }

  if (info->type & GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM) {
    if (GST_EVENT_TYPE (info->data) == GST_EVENT_EOS) {
      GST_INFO_OBJECT (nvmultiurisrcbinCreator->nvmultiurisrcbin,
          "EOS for source_id=%d\n", sourceInfo->config->source_id);

      /* Set flag BEFORE queuing to prevent other callbacks from accessing */
      sourceInfo->is_removing = TRUE;
      g_mutex_unlock (&nvmultiurisrcbinCreator->lock);

      g_mutex_lock (&nvmultiurisrcbinCreator->uribin_removal_lock);
      g_queue_push_tail (nvmultiurisrcbinCreator->remove_uribin_queue,
          sourceInfo);
      g_cond_broadcast (&nvmultiurisrcbinCreator->remove_uribin_cond);
      g_mutex_unlock (&nvmultiurisrcbinCreator->uribin_removal_lock);
      sourceInfo->probe_eos_handling = 0;

      return GST_PAD_PROBE_REMOVE;
    }
  }

  g_mutex_unlock (&nvmultiurisrcbinCreator->lock);
  return GST_PAD_PROBE_OK;
}

static GstPadProbeReturn
s_nvmultiurisrcbincreator_probe_func_add_sensorInfo (GstPad * pad,
    GstPadProbeInfo * info, gpointer u_data)
{

  NvMultiUriSrcBinCreator *nvmultiurisrcbinCreator = (NvMultiUriSrcBinCreator *) u_data;
  GstBuffer *buf = (GstBuffer *) info->data;
  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);

  if (!batch_meta){
    return GST_PAD_PROBE_OK;
  }

  NvDsMetaList * l_frame = NULL;
  for (l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);
    NvDsUriSourceInfo *srcInfo =
      (NvDsUriSourceInfo *) g_hash_table_lookup (nvmultiurisrcbinCreator->sourceInfoHash, frame_meta->source_id + (gchar *) NULL);

    /* Skip if source not found, being removed, or pads not properly linked */
    if (srcInfo == NULL || srcInfo->is_removing || !gst_pad_is_linked(srcInfo->uribin_src_pad) || !srcInfo->muxSinkPad || !srcInfo->uribin_src_pad || !srcInfo->uribin)  {
        continue;
    }

    frame_meta->sensorInfo_meta.source_id = srcInfo->config->source_id;
    frame_meta->sensorInfo_meta.sensor_id = srcInfo->config->sensorId ? (gchar const*)g_strdup(srcInfo->config->sensorId) : (gchar const*)g_strdup("");
    frame_meta->sensorInfo_meta.sensor_name = srcInfo->config->sensorName ? (gchar const*)g_strdup(srcInfo->config->sensorName) : (gchar const*)g_strdup("");
    frame_meta->sensorInfo_meta.uri = srcInfo->config->uri ? (gchar const*)g_strdup(srcInfo->config->uri) : (gchar const*)g_strdup("");

  }

  return GST_PAD_PROBE_OK;
}

#ifdef PLATFORM_TEGRA
static GstPadProbeReturn
streammux_processing_done_buf_prob (GstPad * pad, GstPadProbeInfo * info,
    gpointer u_data)
{
  NvDsUriSourceInfo *sourceInfo = (NvDsUriSourceInfo *) u_data;
	  NvMultiUriSrcBinCreator *nvmultiurisrcbinCreator =
      (NvMultiUriSrcBinCreator *) sourceInfo->apiHandle;
  GstBuffer *buf = (GstBuffer *) info->data;
  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);

  if (!batch_meta) {
    GST_ERROR_OBJECT (nvmultiurisrcbinCreator->nvmultiurisrcbin, "Batch meta not found for buffer %p", buf);
    return GST_PAD_PROBE_OK;
  } else {
    for (NvDsMetaList * l_frame = batch_meta->frame_meta_list; l_frame != NULL;
         l_frame = l_frame->next) {
      NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) l_frame->data;
      frame_meta->ntp_timestamp = frame_meta->buf_pts;
    }
  }
  return GST_PAD_PROBE_OK;
}
#endif

static void
s_nvmultiurisrcbincreator_cb_newpad (GstElement * decodebin, GstPad * pad,
    gpointer data)
{
  GstCaps *caps = gst_pad_query_caps (pad, NULL);
  const GstStructure *str = gst_caps_get_structure (caps, 0);
  const gchar *name = gst_structure_get_name (str);
  NvDsUriSourceInfo *sourceInfo = (NvDsUriSourceInfo *) data;
  NvMultiUriSrcBinCreator *nvmultiurisrcbinCreator =
      (NvMultiUriSrcBinCreator *) sourceInfo->apiHandle;

  GST_DEBUG_OBJECT (nvmultiurisrcbinCreator->nvmultiurisrcbin,
      "added pad %s -- %s\n", name, GST_PAD_NAME (pad));

  /** Link the pad with nvstreammux sink pad according to the current mode of support */
  if ((nvmultiurisrcbinCreator->mode == NVDS_MULTIURISRCBIN_MODE_VIDEO
          && !strncmp (GST_PAD_NAME (pad), "vsrc_", 5))
      || (nvmultiurisrcbinCreator->mode == NVDS_MULTIURISRCBIN_MODE_AUDIO
          && !strncmp (GST_PAD_NAME (pad), "asrc_", 5))) {
    //Get sink_%d pad from nvstreammux and link to it
    s_nvmultiurisrcbincreator_link_element_to_streammux_sink_pad (sourceInfo,
        nvmultiurisrcbinCreator->streammux, pad, sourceInfo->config->source_id);
    //Attach a probe for EOS handling
    sourceInfo->uribin_src_pad = pad;

    sourceInfo->probe_eos_handling = gst_pad_add_probe (pad,
        GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM,
        s_nvmultiurisrcbincreator_probe_func_eos_handling, data, NULL);

    /* Send NTP sync query if creation_time_ns is set */
    if (sourceInfo->config->creation_time_ns > 0 && sourceInfo->muxSinkPad) {
      GstQuery *ntp_query = gst_nvquery_ntp_sync_new ();
      if (ntp_query) {
        _NtpData ntp_data = { 0, 0, 0 };
        ntp_data.ntp_time_epoch_ns = sourceInfo->config->creation_time_ns;
        gst_nvquery_ntp_sync_set (ntp_query, &ntp_data);

        gboolean query_ret = gst_pad_query (sourceInfo->muxSinkPad, ntp_query);
        GST_DEBUG_OBJECT (nvmultiurisrcbinCreator->nvmultiurisrcbin,
            "Sent NTP sync query with creation_time %" G_GUINT64_FORMAT " ns to sink pad, result: %d\n",
            sourceInfo->config->creation_time_ns, query_ret);
        gst_query_unref (ntp_query);
      }
    }
#ifdef PLATFORM_TEGRA
    gboolean is_ipc = sourceInfo->config->uri
        && g_str_has_prefix (sourceInfo->config->uri, "ipc://");
    if (is_ipc) {
      GstPad *src_pad = gst_nvmultiurisrcbincreator_get_source_pad (nvmultiurisrcbinCreator);
      gst_pad_add_probe (src_pad,
          GST_PAD_PROBE_TYPE_BUFFER,
          streammux_processing_done_buf_prob,
          data,
          NULL);
    }
#endif
  }
#if 0
  /** Link the unnecessary pad with fakesink according to the current mode of support
   * TODO; Leaving this hanging to avoid hang in pipeline teardown
   * Test Case: test_nvdshelper_multiurisrcbin_dynamic_add_remove_video_uri
   */
  if ((nvmultiurisrcbinCreator->mode == NVDS_MULTIURISRCBIN_MODE_VIDEO
          && !strncmp (GST_PAD_NAME (pad), "asrc_", 5))
      || (nvmultiurisrcbinCreator->mode == NVDS_MULTIURISRCBIN_MODE_AUDIO
          && !strncmp (GST_PAD_NAME (pad), "vsrc_", 5))) {
    //Create fakesink, add to bin and link to it
    GstElement *queue = gst_element_factory_make ("queue", NULL);
    GstElement *fakesink = gst_element_factory_make ("fakesink", NULL);
    g_object_set (fakesink, "async", FALSE, "enable-last-sample", FALSE, NULL);
    GstPad *fakesinkSinkPad = gst_element_get_static_pad (queue, "sink");
    gst_bin_add_many (GST_BIN (nvmultiurisrcbinCreator->nvmultiurisrcbin),
        queue, fakesink, NULL);
    gst_pad_link (pad, fakesinkSinkPad);
    gst_object_unref (fakesinkSinkPad);
    gst_element_link (queue, fakesink);
    gst_element_sync_state_with_parent (queue);
    gst_element_sync_state_with_parent (fakesink);
  }
#endif
  gst_nvmultiurisrcbincreator_sync_children_states (sourceInfo->apiHandle);
}

static void
s_nvmultiurisrcbincreator_cb_removeelem (GstElement * decodebin,
    GstElement * elem, gpointer data)
{
  NvDsUriSourceInfo *sourceInfo = (NvDsUriSourceInfo *) data;
  NvMultiUriSrcBinCreator *nvmultiurisrcbinCreator =
      (NvMultiUriSrcBinCreator *) sourceInfo->apiHandle;
  GST_DEBUG_OBJECT (nvmultiurisrcbinCreator->nvmultiurisrcbin,
      "removed element %s\n", GST_ELEMENT_NAME (elem));
  if (strcmp (GST_ELEMENT_NAME (elem), "src") == 0
      || strcmp (GST_ELEMENT_NAME (elem), "source") == 0) {
    /** source element is now removed; we expect no more callbacks for this source
     * from GStreamer; also the user did not call the API: gst_nvmultiurisrcbincreator_remove_source()
     * If user did call, this remove-elem callback would have been disconnected
     * So, remove the source info from internal book-keeping here:
     */

    g_mutex_lock (&nvmultiurisrcbinCreator->lock);
    /* Check if source is already being removed to prevent race conditions */
    if (sourceInfo->is_removing) {
      GST_DEBUG_OBJECT (nvmultiurisrcbinCreator->nvmultiurisrcbin,
          "element-removed: source already being removed, skipping");
      g_mutex_unlock (&nvmultiurisrcbinCreator->lock);
      return;
    }

    /* Set flag to prevent other callbacks from accessing this source */
    sourceInfo->is_removing = TRUE;

    //remove handlers to ensure no more callbacks to handle remove stream
    s_nvmultiurisrcbincreator_remove_source_info_handlers (sourceInfo);
    s_nvmultiurisrcbincreator_remove_source_info (nvmultiurisrcbinCreator,
        sourceInfo);
    g_mutex_unlock (&nvmultiurisrcbinCreator->lock);
  }
}

GstDsNvUriSrcConfig *
gst_nvmultiurisrcbincreator_src_config_dup (GstDsNvUriSrcConfig * sourceConfig)
{
  GstDsNvUriSrcConfig *config =
      (GstDsNvUriSrcConfig *) g_malloc0 (sizeof (GstDsNvUriSrcConfig));
  *config = *sourceConfig;
 /** allocate memory for pointers and copy them over */
  config->uri = g_strdup (sourceConfig->uri);
  config->sensorId =
      sourceConfig->sensorId ? g_strdup (sourceConfig->sensorId) : NULL;
  config->sensorName =
      sourceConfig->sensorName ? g_strdup (sourceConfig->sensorName) : NULL;
  config->smart_rec_dir_path = g_strdup (sourceConfig->smart_rec_dir_path);
  config->smart_rec_file_prefix =
      g_strdup (sourceConfig->smart_rec_file_prefix);
  return config;
}

void
gst_nvmultiurisrcbincreator_src_config_free (GstDsNvUriSrcConfig * config)
{
  if (config->uri) {
    g_free (config->uri);
  }
  if (config->sensorId) {
    g_free (config->sensorId);
  }
  if (config->sensorName) {
    g_free (config->sensorName);
  }
  if (config->smart_rec_dir_path) {
    g_free (config->smart_rec_dir_path);
  }
  if (config->smart_rec_file_prefix) {
    g_free (config->smart_rec_file_prefix);
  }
  if (config) {
    g_free (config);
  }
}

static NvDsUriSourceInfo *
s_nvmultiurisrcbincreator_create_source_info (GstDsNvUriSrcConfig *
    sourceConfig, NvDst_Handle_NvMultiUriSrcCreator apiHandle)
{

  NvDsUriSourceInfo *sourceInfo =
      (NvDsUriSourceInfo *) g_malloc0 (sizeof (NvDsUriSourceInfo));

    /** Copy the source config */
  sourceInfo->config =
      gst_nvmultiurisrcbincreator_src_config_dup (sourceConfig);

  sourceInfo->apiHandle = apiHandle;
  sourceInfo->is_removing = FALSE;  /* Initialize removal flag */
  return sourceInfo;
}

static void
s_nvmultiurisrcbincreator_destroy_source_info (NvDsUriSourceInfo * sourceInfo)
{
  gst_nvmultiurisrcbincreator_src_config_free (sourceInfo->config);
  g_free (sourceInfo);
}

static GstDsNvStreammuxConfig *
s_nvmultiurisrcbincreator_create_mux_config (GstDsNvStreammuxConfig * muxConfig)
{
  GstDsNvStreammuxConfig *config =
      (GstDsNvStreammuxConfig *) g_malloc0 (sizeof (GstDsNvStreammuxConfig));
  *config = *muxConfig;
  config->config_file_path = g_strdup (muxConfig->config_file_path);

  return config;
}

static void
s_nvmultiurisrcbincreator_destroy_mux_config (GstDsNvStreammuxConfig *
    muxConfig)
{
  if (muxConfig->config_file_path) {
    g_free (muxConfig->config_file_path);
  }
  if (muxConfig) {
    g_free (muxConfig);
  }


}

static guint
extract_pad_index(const gchar *str) {
    while (*str && !isdigit(*str)) str++;
    return atoi(str);
}

gint
s_get_source_id (NvMultiUriSrcBinCreator * nvmultiurisrcbinCreator, GstDsNvUriSrcConfig * sourceConfig)
{

  gchar pad_name[64];
  gint max_batch_size = 0;
  guint pad_indx = 0;

  if ( nvmultiurisrcbinCreator && nvmultiurisrcbinCreator->muxConfig){
    max_batch_size = nvmultiurisrcbinCreator->muxConfig->maxBatchSize;
  }

  if(sourceConfig->sensorIdToPadIdMapping == TRUE)
  {
    pad_indx = extract_pad_index(sourceConfig->sensorId);
    GstPad *test_pad =
    gst_element_get_static_pad (nvmultiurisrcbinCreator->streammux, pad_name);
    if (!test_pad) {
      return pad_indx;
    }
    gst_object_unref (test_pad);
  }
  else
  {
    for (int i = 0; i < max_batch_size; i++) {
      pad_indx = (nvmultiurisrcbinCreator->base_index + i) % max_batch_size;
      g_snprintf (pad_name, sizeof (pad_name), "sink_%u", pad_indx);
      GstPad *test_pad =
        gst_element_get_static_pad (nvmultiurisrcbinCreator->streammux,
        pad_name);
      if (!test_pad) {
        nvmultiurisrcbinCreator->base_index = pad_indx + 1;
        return pad_indx;
      }
      gst_object_unref (test_pad);
    }
  }
  return -1;
}

gboolean
gst_nvmultiurisrcbincreator_add_source (NvDst_Handle_NvMultiUriSrcCreator
    apiHandle, GstDsNvUriSrcConfig * sourceConfig)
{
  NvMultiUriSrcBinCreator *nvmultiurisrcbinCreator =
      (NvMultiUriSrcBinCreator *) apiHandle;
  g_mutex_lock (&nvmultiurisrcbinCreator->lock);
  GstElement *uribin = gst_element_factory_make ("nvurisrcbin", NULL);
  if (!uribin) {
    GST_WARNING_OBJECT (nvmultiurisrcbinCreator->nvmultiurisrcbin,
        "Could not create element 'nvurisrcbin'");
    g_mutex_unlock (&nvmultiurisrcbinCreator->lock);
    return FALSE;
  }

  NvDsUriSourceInfo *sourceInfo =
      s_nvmultiurisrcbincreator_create_source_info (sourceConfig, apiHandle);
  sourceInfo->uribin = (GstElement *) gst_object_ref (uribin);
  sourceConfig->source_id = s_get_source_id (nvmultiurisrcbinCreator,sourceConfig);
  sourceInfo->config->source_id = sourceConfig->source_id;
  //set nvurisrcbin properties
  s_nvmultiurisrcbincreator_set_properties_nvuribin (GST_ELEMENT (uribin),
      sourceConfig);

  //Add sourceInfo to the list and the hashMap
  nvmultiurisrcbinCreator->sourceInfoList =
      g_list_prepend (nvmultiurisrcbinCreator->sourceInfoList, sourceInfo);
  g_hash_table_insert (nvmultiurisrcbinCreator->sourceInfoHash,
      sourceConfig->source_id + (char *) NULL, sourceInfo);

  //Add nvurisrcbin instance to the bin
  gst_bin_add (GST_BIN (nvmultiurisrcbinCreator->nvmultiurisrcbin), uribin);

  //Do all necessary calls to g_signal_connect()
  s_nvmultiurisrcbincreator_add_source_info_handlers (sourceInfo);

  nvmultiurisrcbinCreator->numOfActiveSources++;

  /** POST nvmessage stream added on the bus */
  {
    NvDsSensorInfo sensorInfo;
    sensorInfo.source_id = sourceConfig->source_id;
    sensorInfo.sensor_id = sourceConfig->sensorId;
    sensorInfo.sensor_name = sourceConfig->sensorName;
    sensorInfo.uri = sourceConfig->uri;
    GstBus *bus =
        s_nvmultiurisrcbincreator_get_bus_from_parent (nvmultiurisrcbinCreator);
    if (bus) {
      gst_bus_post (bus,
          gst_nvmessage_new_stream_add (GST_OBJECT
              (nvmultiurisrcbinCreator->nvmultiurisrcbin), &sensorInfo));
      gst_object_unref (bus);
    }
  }

  g_mutex_unlock (&nvmultiurisrcbinCreator->lock);

  return TRUE;
}

gboolean
    gst_nvmultiurisrcbincreator_sync_children_states
    (NvDst_Handle_NvMultiUriSrcCreator apiHandle) {
  NvMultiUriSrcBinCreator *nvmultiurisrcbinCreator =
      (NvMultiUriSrcBinCreator *) apiHandle;
#if 0
  /* Note: set batch-size to 1 and let add/remove APIs update
   * this logic cause pipeline slowdown and hence the maxBatchSize usage
   * TODO; Tracked and hence code commented.
   */
  if (!nvmultiurisrcbinCreator->using_new_mux) {
    g_object_set (nvmultiurisrcbinCreator->streammux, "batch-size",
        nvmultiurisrcbinCreator->numOfActiveSources, NULL);
  }
#endif
  /*
   * Synchronize the state of every child of bin with the state of bin from a
   * different thread:
   */
  gst_element_call_async (nvmultiurisrcbinCreator->nvmultiurisrcbin,
      (GstElementCallAsyncFunc) gst_bin_sync_children_states, NULL, NULL);
  return TRUE;
}

static void
s_nvmultiurisrcbincreator_remove_source_info (NvMultiUriSrcBinCreator *
    nvmultiurisrcbinCreator, NvDsUriSourceInfo * sourceInfo)
{

  NvDsSensorInfo sensorInfoM;
  sensorInfoM.source_id = sourceInfo->config->source_id;
  sensorInfoM.sensor_id = sourceInfo->config->sensorId;
  sensorInfoM.sensor_name = sourceInfo->config->sensorName;
  sensorInfoM.uri = sourceInfo->config->uri;

  /** POST nvmessage stream removed on the bus */
  if (GST_IS_BIN (nvmultiurisrcbinCreator->nvmultiurisrcbin)) {
    GstBus *bus =
        s_nvmultiurisrcbincreator_get_bus_from_parent (nvmultiurisrcbinCreator);
    if (bus) {
      GstMessage *streamRemoveMsg =
          gst_nvmessage_new_stream_remove (GST_OBJECT
          (nvmultiurisrcbinCreator->nvmultiurisrcbin), &sensorInfoM);
      if (streamRemoveMsg) {
        gst_bus_post (bus, streamRemoveMsg);
        gst_object_unref (bus);
      }
    }
  }
  //remove sourceInfo from the hash map and the list
  g_hash_table_remove (nvmultiurisrcbinCreator->sourceInfoHash,
      sourceInfo->config->source_id + (gchar *) NULL);
  nvmultiurisrcbinCreator->sourceInfoList =
      g_list_remove (nvmultiurisrcbinCreator->sourceInfoList, sourceInfo);
  //free sourceInfo
  s_nvmultiurisrcbincreator_destroy_source_info (sourceInfo);
  nvmultiurisrcbinCreator->numOfActiveSources--;
}

void s_nvmultiurisrcbincreator_add_source_info_handlers
    (NvDsUriSourceInfo * sourceInfo)
{
  //connect necessary signals on uribin for linking to nvstreammux
  sourceInfo->newPadAddedHandler =
      g_signal_connect (G_OBJECT (sourceInfo->uribin), "pad-added",
      G_CALLBACK (s_nvmultiurisrcbincreator_cb_newpad), sourceInfo);
  /** Note: The callback pad-added will add: sourceInfo->probe_eos_handling probe */

  //connect element-removed signal to remove the source in case
  //the stream got removed before an API call to gst_nvmultiurisrcbincreator_remove_source or EOS
  sourceInfo->elemRemovedHandler =
      g_signal_connect (G_OBJECT (sourceInfo->uribin), "element-removed",
      G_CALLBACK (s_nvmultiurisrcbincreator_cb_removeelem), sourceInfo);
}

void s_nvmultiurisrcbincreator_remove_source_info_handlers
    (NvDsUriSourceInfo * sourceInfo)
{
  //Remove all handlers to which we gave sourceInfo pointer as callback param
  //Remove Error handling probe; stream is removed over the API call than an error
  if (sourceInfo->elemRemovedHandler) {
    g_signal_handler_disconnect (G_OBJECT (sourceInfo->uribin),
        sourceInfo->elemRemovedHandler);
  }
  sourceInfo->elemRemovedHandler = 0;

  if (sourceInfo->newPadAddedHandler) {
    g_signal_handler_disconnect (G_OBJECT (sourceInfo->uribin),
        sourceInfo->newPadAddedHandler);
  }
  sourceInfo->newPadAddedHandler = 0;
}

gboolean
gst_nvmultiurisrcbincreator_remove_source (NvDst_Handle_NvMultiUriSrcCreator
    apiHandle, guint sourceId)
{
  return s_nvmultiurisrcbincreator_remove_source_impl (apiHandle, sourceId,
      TRUE);
}

gboolean
    gst_nvmultiurisrcbincreator_remove_source_without_forced_state_change
    (NvDst_Handle_NvMultiUriSrcCreator apiHandle, guint sourceId) {
  return s_nvmultiurisrcbincreator_remove_source_impl (apiHandle, sourceId,
      FALSE);
}

gboolean
s_nvmultiurisrcbincreator_remove_source_impl (NvDst_Handle_NvMultiUriSrcCreator
    apiHandle, guint sourceId, gboolean forceSourceStateChange)
{
  GstStateChangeReturn state_return = GST_STATE_CHANGE_FAILURE;
  NvMultiUriSrcBinCreator *nvmultiurisrcbinCreator =
      (NvMultiUriSrcBinCreator *) apiHandle;
  g_mutex_lock (&nvmultiurisrcbinCreator->lock);
  NvDsUriSourceInfo *sourceInfo =
      (NvDsUriSourceInfo *)
      g_hash_table_lookup (nvmultiurisrcbinCreator->sourceInfoHash,
      sourceId + (gchar *) NULL);

  if (!sourceInfo) {
    //No source found
    g_mutex_unlock (&nvmultiurisrcbinCreator->lock);
    return FALSE;
  }

  /* Check if source is already being removed to prevent double-removal */
  if (sourceInfo->is_removing) {
    GST_INFO_OBJECT (nvmultiurisrcbinCreator->nvmultiurisrcbin,
        "API removal: source %d already being removed, skipping", sourceId);
    g_mutex_unlock (&nvmultiurisrcbinCreator->lock);
    return FALSE;
  }

  /* Set flag to prevent other callbacks from accessing this source */
  sourceInfo->is_removing = TRUE;

  LOGD ("removing source %d\n", sourceId);
  //Removing handlers from g_signal_connect() calls
  s_nvmultiurisrcbincreator_remove_source_info_handlers (sourceInfo);

  //Removing probes that were added:
  //Remove the EOS handling probe; stream is removed over the API call than EOS
  if (sourceInfo->uribin_src_pad && GST_IS_PAD (sourceInfo->uribin_src_pad)) {
    /** Note: Removing this probe will ensure the forced EOS
     * to not trigger a call to: s_nvmultiurisrcbincreator_probe_func_eos_handling
     */
    if (sourceInfo->probe_eos_handling) {
      gst_pad_remove_probe (sourceInfo->uribin_src_pad,
          sourceInfo->probe_eos_handling);
    }
    sourceInfo->probe_eos_handling = 0;
    /* Note: we shall not gst_object_unref(sourceInfo->uribin_src_pad);
     * GStreamer warns:
     * Trying to dispose object "a/vsrc_0", but it still has a parent
     * "dsnvurisrcbin1".
     * You need to let the parent manage the object instead of unreffing
     * the object directly.
     */
  }

  if (forceSourceStateChange) {

      if (GST_IS_PAD (sourceInfo->muxSinkPad)) {
      gst_pad_send_event (sourceInfo->muxSinkPad,
                          gst_event_new_flush_start());
      gst_element_send_event (GST_ELEMENT (sourceInfo->uribin),
                              gst_event_new_flush_start());
      gst_pad_send_event (sourceInfo->muxSinkPad,
                          gst_event_new_flush_stop(FALSE));
      gst_element_send_event (GST_ELEMENT (sourceInfo->uribin),
                              gst_event_new_flush_stop(FALSE));

      // If the pad is linked to another pad, first unlink it
      if (gst_pad_is_linked(sourceInfo->muxSinkPad)) {
        GstPad *peer_pad = gst_pad_get_peer(sourceInfo->muxSinkPad);

        if (peer_pad) {
           if (GST_PAD_IS_SINK(sourceInfo->muxSinkPad) && GST_PAD_IS_SRC(peer_pad)) {
              gst_pad_unlink(peer_pad, sourceInfo->muxSinkPad);
           } else {
                g_warning("Cannot unlink pads: pad direction mismatch");
           }
           gst_object_unref(peer_pad);
        }
      }
    }

    g_object_ref (sourceInfo->uribin);
    if ((!gst_bin_remove (GST_BIN (nvmultiurisrcbinCreator->nvmultiurisrcbin),
            sourceInfo->uribin))) {
       GST_WARNING_OBJECT (nvmultiurisrcbinCreator->nvmultiurisrcbin,
         "Failed to set remove source-id:%u", sourceInfo->config->source_id);
       g_mutex_unlock (&nvmultiurisrcbinCreator->lock);
       return FALSE;
    }

    gst_element_send_event (GST_ELEMENT (sourceInfo->uribin),
               gst_event_new_flush_start());
    gst_element_send_event (GST_ELEMENT (sourceInfo->uribin),
               gst_event_new_flush_stop(FALSE));
    if (GST_IS_BIN (sourceInfo->uribin)
        && (state_return =
           gst_element_set_state (GST_ELEMENT (sourceInfo->uribin),
              GST_STATE_NULL)) == GST_STATE_CHANGE_FAILURE) {
       GST_WARNING_OBJECT (nvmultiurisrcbinCreator->nvmultiurisrcbin,
         "Failed to set stop source-id:%u", sourceInfo->config->source_id);
       g_mutex_unlock (&nvmultiurisrcbinCreator->lock);
       return FALSE;
    }

    if (state_return == GST_STATE_CHANGE_ASYNC) {
      gst_element_get_state(sourceInfo->uribin, NULL, NULL, GST_CLOCK_TIME_NONE);
    }

    gst_object_unref (sourceInfo->uribin);

    if (GST_IS_PAD (sourceInfo->muxSinkPad)) {
      gst_pad_send_event (sourceInfo->muxSinkPad,
        gst_event_new_flush_stop (FALSE));
      gst_element_release_request_pad (nvmultiurisrcbinCreator->streammux,
        sourceInfo->muxSinkPad);
      gst_object_unref (sourceInfo->muxSinkPad);
    }

  }

  //remove sourceInfo from the hash map and the list, free sourceInfo
  s_nvmultiurisrcbincreator_remove_source_info (nvmultiurisrcbinCreator,
      sourceInfo);
  LOGD ("removed source %d\n", sourceId);
  g_mutex_unlock (&nvmultiurisrcbinCreator->lock);

  return TRUE;
}

GstPad *
gst_nvmultiurisrcbincreator_get_source_pad (NvDst_Handle_NvMultiUriSrcCreator
    apiHandle)
{
  NvMultiUriSrcBinCreator *nvmultiurisrcbinCreator =
      (NvMultiUriSrcBinCreator *) apiHandle;
  GstPad *srcpad =
      gst_element_get_static_pad (nvmultiurisrcbinCreator->nvmultiurisrcbin,
      "src");
  if (!srcpad) {
    GST_WARNING ("nvstreammux request src pad failed. Exiting\n");
    return NULL;
  }
  return srcpad;
}

GstElement *
gst_nvmultiurisrcbincreator_get_bin (NvDst_Handle_NvMultiUriSrcCreator
    apiHandle)
{
  NvMultiUriSrcBinCreator *nvmultiurisrcbinCreator =
      (NvMultiUriSrcBinCreator *) apiHandle;
  return nvmultiurisrcbinCreator->nvmultiurisrcbin;
}

static gboolean
s_nvmultiurisrcbincreator_link_element_to_streammux_sink_pad (NvDsUriSourceInfo
    * sourceInfo, GstElement * streammux, GstPad * src_pad, gint index)
{
  gchar pad_name[16];

  if (index >= 0) {
    g_snprintf (pad_name, 16, "sink_%u", index);
    pad_name[15] = '\0';
  } else {
    strcpy (pad_name, "sink_%u");
  }

  sourceInfo->muxSinkPad = gst_element_request_pad_simple (streammux, pad_name);
  if (!sourceInfo->muxSinkPad) {
    GST_WARNING ("Failed to get sink pad (%d) from streammux\n",
        sourceInfo->config->source_id);
    return FALSE;
  }

  if (gst_pad_link (src_pad, sourceInfo->muxSinkPad) != GST_PAD_LINK_OK) {
    GST_WARNING ("Failed to link '%s' -> '%s'\n",
        GST_PAD_NAME (src_pad), GST_ELEMENT_NAME (streammux));
    return FALSE;
  }

  return TRUE;
}

GstDsNvUriSrcConfig *
gst_nvmultiurisrcbincreator_get_source_config (NvDst_Handle_NvMultiUriSrcCreator
    apiHandle, gchar const *uri, gchar const *sensorId)
{
  NvMultiUriSrcBinCreator *nvmultiurisrcbinCreator =
      (NvMultiUriSrcBinCreator *) apiHandle;
  GstDsNvUriSrcConfig *config = NULL;
  g_mutex_lock (&nvmultiurisrcbinCreator->lock);
  /** Go through the list and find it */
  for (GList * node = nvmultiurisrcbinCreator->sourceInfoList; node;
      node = g_list_next (node)) {
    NvDsUriSourceInfo *sourceInfo = (NvDsUriSourceInfo *) (node->data);
    /* Skip sources that are being removed */
    if (sourceInfo && !sourceInfo->is_removing
        && sourceInfo->config->uri
        && sourceInfo->config->sensorId
        && (g_strcmp0 (sourceInfo->config->uri, uri) == 0)
        && (g_strcmp0 (sourceInfo->config->sensorId, sensorId) == 0)) {
      config = gst_nvmultiurisrcbincreator_src_config_dup (sourceInfo->config);
      break;
    }
  }
  g_mutex_unlock (&nvmultiurisrcbinCreator->lock);
  return config;
}

GstDsNvUriSrcConfig *gst_nvmultiurisrcbincreator_get_source_config_by_sensorid
    (NvDst_Handle_NvMultiUriSrcCreator apiHandle, gchar const *sensorId)
{
  NvMultiUriSrcBinCreator *nvmultiurisrcbinCreator =
      (NvMultiUriSrcBinCreator *) apiHandle;
  GstDsNvUriSrcConfig *config = NULL;
  g_mutex_lock (&nvmultiurisrcbinCreator->lock);
  /** Go through the list and find it */
  for (GList * node = nvmultiurisrcbinCreator->sourceInfoList; node;
      node = g_list_next (node)) {
    NvDsUriSourceInfo *sourceInfo = (NvDsUriSourceInfo *) (node->data);
    /* Skip sources that are being removed */
    if (sourceInfo && !sourceInfo->is_removing
        && sourceInfo->config->sensorId
        && (g_strcmp0 (sourceInfo->config->sensorId, sensorId) == 0)) {
      config = gst_nvmultiurisrcbincreator_src_config_dup (sourceInfo->config);
      break;
    }
  }
  g_mutex_unlock (&nvmultiurisrcbinCreator->lock);
  return config;
}

static void
s_nvmultiurisrcbincreator_set_properties_nvuribin (GstElement * element_,
    GstDsNvUriSrcConfig const *config)
{
  g_object_set (element_, "uri", config->uri, NULL);
  g_object_set (element_, "source-id", (guint) config->source_id, NULL);
  if (config->num_extra_surfaces)
    g_object_set (element_, "num-extra-surfaces", config->num_extra_surfaces,
        NULL);
  if (config->gpu_id)
    g_object_set (element_, "gpu-id", config->gpu_id, NULL);
  if (config->skip_frames_type)
    g_object_set (element_, "dec-skip-frames", config->skip_frames_type, NULL);
  g_object_set (element_, "low-latency-mode", config->low_latency_mode, NULL);
  g_object_set (element_, "type", SOURCE_TYPE_AUTO, NULL);      //always set to auto
  g_object_set (element_, "cudadec-memtype", config->cuda_memory_type, NULL);
  if (config->drop_frame_interval)
    g_object_set (element_, "drop-frame-interval", config->drop_frame_interval,
        NULL);
  g_object_set (element_, "drop-on-latency", config->drop_on_latency, NULL);
  if (config->extract_sei_type5_data) {
    g_object_set (element_, "extract-sei-type5-data", config->extract_sei_type5_data,
        NULL);
  }
  if (config->buffer_mode){
    g_object_set (element_, "buffer-mode", config->buffer_mode, NULL);
  }
  g_object_set (element_, "sei-uuid", config->sei_uuid,
        NULL);
  if (config->rtp_protocol)
    g_object_set (element_, "select-rtp-protocol", config->rtp_protocol, NULL);
  if (config->leaky)
    g_object_set (element_, "leaky", config->leaky, NULL);
  if (config->max_size_buffers)
    g_object_set (element_, "max-size-buffers", config->max_size_buffers, NULL);
  if (config->loop)
    g_object_set (element_, "file-loop", config->loop, NULL);
  if (config->smart_record)
    g_object_set (element_, "smart-record", config->smart_record, NULL);
  if (config->smart_rec_dir_path)
    g_object_set (element_, "smart-rec-dir-path", config->smart_rec_dir_path,
        NULL);
  if (config->smart_rec_file_prefix)
    g_object_set (element_, "smart-rec-file-prefix",
        config->smart_rec_file_prefix, NULL);
  if (config->smart_rec_cache_size)
    g_object_set (element_, "smart-rec-cache", config->smart_rec_cache_size,
        NULL);
  if (config->smart_rec_mode)
    g_object_set (element_, "smart-rec-mode", config->smart_rec_mode, NULL);
  if (config->smart_rec_container)
    g_object_set (element_, "smart-rec-container", config->smart_rec_container,
        NULL);
  if (config->smart_rec_def_duration)
    g_object_set (element_, "smart-rec-default-duration",
        config->smart_rec_def_duration, NULL);
  //g_object_set(element_, "smart-rec-status", config->, NULL); //Not supported
  if (config->rtsp_reconnect_interval_sec)
    g_object_set (element_, "rtsp-reconnect-interval",
        config->rtsp_reconnect_interval_sec, NULL);
  if (config->init_rtsp_reconnect_interval_sec)
    g_object_set (element_, "init-rtsp-reconnect-interval",
        config->init_rtsp_reconnect_interval_sec, NULL);
  if (config->latency)
    g_object_set (element_, "latency", config->latency, NULL);
  if (config->udp_buffer_size)
    g_object_set (element_, "udp-buffer-size", config->udp_buffer_size, NULL);
  g_object_set (element_, "disable-passthrough", config->disable_passthrough,
      NULL);
  g_object_set (element_, "rtsp-reconnect-attempts", config->rtsp_reconnect_attempts, NULL);
  g_object_set (element_, "disable-audio", config->disable_audio,
      NULL);
  if (config->ipc_socket_path)
    g_object_set (element_, "ipc-socket-path", config->ipc_socket_path, NULL);
  g_object_set (element_, "ipc-buffer-timestamp-copy", config->ipc_buffer_timestamp_copy, NULL);
  g_object_set (element_, "ipc-connection-attempts", config->ipc_connection_attempts, NULL);
  g_object_set (element_, "ipc-connection-interval", config->ipc_connection_interval, NULL);
  g_object_set (element_, "simulate-fps-interval-ms", config->simulate_fps_interval_ms, NULL);
}

gboolean
gst_nvmultiurisrcbincreator_get_source_info_list
(NvDst_Handle_NvMultiUriSrcCreator apiHandle, GList **stream_info_list)
{
  NvMultiUriSrcBinCreator *nvmultiurisrcbinCreator =
      (NvMultiUriSrcBinCreator *) apiHandle;

  g_mutex_lock (&nvmultiurisrcbinCreator->lock);

  if (nvmultiurisrcbinCreator->sourceInfoList) {

    for (GList *node = nvmultiurisrcbinCreator->sourceInfoList; node;
         node = g_list_next (node)) {
      NvDsUriSourceInfo *sourceInfo = (NvDsUriSourceInfo *) (node->data);

      /* Skip sources that are being removed */
      if (!sourceInfo || sourceInfo->is_removing) {
        continue;
      }

      NvDsSensorInfo *sensorInfo = g_new0 (NvDsSensorInfo, 1);

      sensorInfo->source_id = sourceInfo->config->source_id;
      sensorInfo->sensor_id = sourceInfo->config->sensorId;
      sensorInfo->sensor_name = g_strdup (sourceInfo->config->sensorName);
      sensorInfo->uri = g_strdup (sourceInfo->config->uri);

      *stream_info_list = g_list_append (*stream_info_list, sensorInfo);
    }
  }

  g_mutex_unlock (&nvmultiurisrcbinCreator->lock);

  return TRUE;
}

gboolean
    gst_nvmultiurisrcbincreator_get_active_sources_list
    (NvDst_Handle_NvMultiUriSrcCreator apiHandle, guint * count,
    GstDsNvUriSrcConfig *** configs) {
  NvMultiUriSrcBinCreator *nvmultiurisrcbinCreator =
      (NvMultiUriSrcBinCreator *) apiHandle;

  if (!nvmultiurisrcbinCreator || !count || !configs) {
    return FALSE;
  }

  *count = 0;
  *configs = NULL;

  g_mutex_lock (&nvmultiurisrcbinCreator->lock);

  if (nvmultiurisrcbinCreator->sourceInfoList) {
    *count = g_list_length (nvmultiurisrcbinCreator->sourceInfoList);
    *configs =
        (GstDsNvUriSrcConfig **) g_malloc0 (sizeof (GstDsNvUriSrcConfig *) *
        (*count));
    guint i = 0;
    for (GList * node = nvmultiurisrcbinCreator->sourceInfoList; node;
        node = g_list_next (node)) {
      NvDsUriSourceInfo *sourceInfo = (NvDsUriSourceInfo *) (node->data);
      (*configs)[i] =
          gst_nvmultiurisrcbincreator_src_config_dup (sourceInfo->config);
      i++;
    }
  }

  g_mutex_unlock (&nvmultiurisrcbinCreator->lock);

  return TRUE;
}

void gst_nvmultiurisrcbincreator_src_config_list_free
    (NvDst_Handle_NvMultiUriSrcCreator apiHandle, guint count,
    GstDsNvUriSrcConfig ** configs)
{
  for (guint i = 0; i < count; i++) {
    gst_nvmultiurisrcbincreator_src_config_free (configs[i]);
  }
  g_free (configs);
}
