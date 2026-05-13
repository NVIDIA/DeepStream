/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <linux/limits.h>       /* For PATH_MAX */
#include <stdio.h>
#include <string.h>

#include "deepstream_common.h"
#include "deepstream_secondary_preprocess.h"


/**
 * Wait for all secondary preprocess to complete the processing and then send
 * the processed buffer to downstream.
 * This is way of synchronization between all secondary preprocess and sending
 * buffer once meta data from all secondary infer components got attached.
 * This is needed because all secondary preprocess process same buffer in parallel.
 */
static GstPadProbeReturn
wait_queue_buf_probe (GstPad * pad, GstPadProbeInfo * info, gpointer u_data)
{
  NvDsSecondaryPreProcessBin *bin = (NvDsSecondaryPreProcessBin *) u_data;
  if (info->type & GST_PAD_PROBE_TYPE_EVENT_BOTH) {
    GstEvent *event = (GstEvent *) info->data;
    if (event->type == GST_EVENT_EOS) {
      return GST_PAD_PROBE_OK;
    }
  }

  if (info->type & GST_PAD_PROBE_TYPE_BUFFER) {
    g_mutex_lock (&bin->wait_lock);
    while (GST_OBJECT_REFCOUNT_VALUE (GST_BUFFER (info->data)) > 1
        && !bin->stop && !bin->flush) {
      gint64 end_time;
      end_time = g_get_monotonic_time () + G_TIME_SPAN_SECOND / 1000;
      g_cond_wait_until (&bin->wait_cond, &bin->wait_lock, end_time);
    }
    g_mutex_unlock (&bin->wait_lock);
  }

  return GST_PAD_PROBE_OK;
}

/**
 * Probe function on sink pad of tee element. It is being used to
 * capture EOS event. So that wait for all secondary to finish can be stopped.
 * see ::wait_queue_buf_probe
 */
static GstPadProbeReturn
wait_queue_buf_probe1 (GstPad * pad, GstPadProbeInfo * info, gpointer u_data)
{
  NvDsSecondaryPreProcessBin *bin = (NvDsSecondaryPreProcessBin *) u_data;
  if (info->type & GST_PAD_PROBE_TYPE_EVENT_BOTH) {
    GstEvent *event = (GstEvent *) info->data;
    if (event->type == GST_EVENT_EOS) {
      bin->stop = TRUE;
    }
  }

  return GST_PAD_PROBE_OK;
}

/**
 * Create secondary preprocess sub bin and sets properties mentioned
 * in configuration file.
 */
static gboolean
create_secondary_preprocess (NvDsPreProcessConfig * configs1,
    NvDsSecondaryPreProcessBinSubBin * subbins1, GstBin * bin, guint index)
{
  gboolean ret = FALSE;
  gchar elem_name[50];
  NvDsPreProcessConfig *config = &configs1[index];
  NvDsSecondaryPreProcessBinSubBin *subbin = &subbins1[index];

  if (!subbin->create) {
    return TRUE;
  }

  g_snprintf (elem_name, sizeof (elem_name), "secondary_preprocess_%d_queue",
      index);

  if (subbin->parent_index == -1 ||
      subbins1[subbin->parent_index].num_children > 1) {
    subbin->queue = gst_element_factory_make (NVDS_ELEM_QUEUE, elem_name);
    if (!subbin->queue) {
      NVGSTDS_ERR_MSG_V ("Failed to create '%s'", elem_name);
      goto done;
    }
    gst_bin_add (bin, subbin->queue);
  }

  g_snprintf (elem_name, sizeof (elem_name), "secondary_preprocess_%d", index);
  subbin->secondary_preprocess =
      gst_element_factory_make (NVDS_ELEM_SECONDARY_PREPROCESS, elem_name);

  if (!subbin->secondary_preprocess) {
    NVGSTDS_ERR_MSG_V ("Failed to create '%s'", elem_name);
    goto done;
  }

  g_object_set (G_OBJECT (subbin->secondary_preprocess),
      "config-file", config->config_file_path, NULL);

  if (config->is_operate_on_gie_id_set) {
    g_object_set (G_OBJECT (subbin->secondary_preprocess), "operate-on-gie-id",
        config->operate_on_gie_id, NULL);
  }


  gst_bin_add (bin, subbin->secondary_preprocess);

  if (subbin->num_children == 0) {
    g_snprintf (elem_name, sizeof (elem_name), "secondary_preprocess_%d_sink",
        index);
    subbin->sink =
        gst_element_factory_make (NVDS_ELEM_SINK_FAKESINK, elem_name);
    if (!subbin->sink) {
      NVGSTDS_ERR_MSG_V ("Failed to create '%s'", elem_name);
      goto done;
    }
    gst_bin_add (bin, subbin->sink);
    g_object_set (G_OBJECT (subbin->sink), "async", FALSE, "sync", FALSE,
        "enable-last-sample", FALSE, NULL);
  }

  if (subbin->num_children > 1) {
    g_snprintf (elem_name, sizeof (elem_name), "secondary_preprocess_%d_tee",
        index);
    subbin->tee = gst_element_factory_make (NVDS_ELEM_TEE, elem_name);
    if (!subbin->tee) {
      NVGSTDS_ERR_MSG_V ("Failed to create '%s'", elem_name);
      goto done;
    }
    gst_bin_add (bin, subbin->tee);
  }

  if (subbin->queue) {
    NVGSTDS_LINK_ELEMENT (subbin->queue, subbin->secondary_preprocess);
  }
  if (subbin->sink) {
    NVGSTDS_LINK_ELEMENT (subbin->secondary_preprocess, subbin->sink);
  }
  if (subbin->tee) {
    NVGSTDS_LINK_ELEMENT (subbin->secondary_preprocess, subbin->tee);
  }

  ret = TRUE;

done:
  if (!ret) {
    NVGSTDS_ERR_MSG_V ("%s failed", __func__);
  }
  return ret;
}

/**
 * This decides if secondary infer sub bin should be created or not.
 * Decision is based on following criteria.
 * 1) It is enabled in configuration file.
 * 2) operate_on_gie_id should match the provided unique id of primary infer.
 * 3) If third or more level of inference is created then operate_on_gie_id
 *    and unique_id should be created in such a way that third infer operate
 *    on second's output and second operate on primary's output and so on.
 */
static gboolean
should_create_secondary_preprocess (NvDsPreProcessConfig * config_array,
    guint num_configs, NvDsSecondaryPreProcessBinSubBin * bins, guint index,
    gint primary_gie_id)
{
  NvDsPreProcessConfig *config = &config_array[index];

  if (!config->enable) {
    return FALSE;
  }

  if (bins[index].create) {
    return TRUE;
  }

  if (config->operate_on_gie_id == primary_gie_id) {
    bins[index].create = TRUE;
    bins[index].parent_index = -1;
    return TRUE;
  }

  return FALSE;
}


// Create bin, add queue and the element, link all elements and ghost pads,
// Set the element properties from the parsed config
gboolean
create_secondary_preprocess_bin (guint num_secondary_preprocess,
    guint primary_gie_unique_id, NvDsPreProcessConfig * config_array,
    NvDsSecondaryPreProcessBin * bin)
{
  gboolean ret = FALSE;
  guint i;
  GstPad *pad;

  bin->bin = gst_bin_new ("secondary_preprocess_bin");
  if (!bin->bin) {
    NVGSTDS_ERR_MSG_V ("Failed to create 'secondary_preprocess_bin'");
    goto done;
  }

  bin->tee =
      gst_element_factory_make (NVDS_ELEM_TEE, "secondary_preprocess_bin_tee");
  if (!bin->tee) {
    NVGSTDS_ERR_MSG_V
        ("Failed to create element 'secondary_preprocess_bin_tee'");
    goto done;
  }

  gst_bin_add (GST_BIN (bin->bin), bin->tee);

  bin->queue =
      gst_element_factory_make (NVDS_ELEM_QUEUE, "secondary_preprocess_queue");
  if (!bin->queue) {
    NVGSTDS_ERR_MSG_V ("Failed to create 'secondary_preprocess_queue'");
    goto done;
  }

  gst_bin_add (GST_BIN (bin->bin), bin->queue);

  pad = gst_element_get_static_pad (bin->queue, "src");
  bin->wait_for_secondary_preprocess_process_buf_probe_id =
      gst_pad_add_probe (pad,
      (GstPadProbeType) (GST_PAD_PROBE_TYPE_BUFFER |
          GST_PAD_PROBE_TYPE_EVENT_BOTH), wait_queue_buf_probe, bin, NULL);
  gst_object_unref (pad);
  pad = gst_element_get_static_pad (bin->tee, "sink");
  gst_pad_add_probe (pad,
      GST_PAD_PROBE_TYPE_EVENT_BOTH, wait_queue_buf_probe1, bin, NULL);
  gst_object_unref (pad);

  NVGSTDS_BIN_ADD_GHOST_PAD (bin->bin, bin->tee, "sink");
  NVGSTDS_BIN_ADD_GHOST_PAD (bin->bin, bin->queue, "src");

  if (!link_element_to_tee_src_pad (bin->tee, bin->queue)) {
    goto done;
  }

  for (i = 0; i < num_secondary_preprocess; i++) {
    should_create_secondary_preprocess (config_array, num_secondary_preprocess,
        bin->sub_bins, i, primary_gie_unique_id);
  }

  for (i = 0; i < num_secondary_preprocess; i++) {
    if (bin->sub_bins[i].create) {
      if (!create_secondary_preprocess (config_array, bin->sub_bins,
              GST_BIN (bin->bin), i)) {
        goto done;
      }
    }
  }

  for (i = 0; i < num_secondary_preprocess; i++) {
    if (bin->sub_bins[i].create) {
      if (bin->sub_bins[i].parent_index == -1) {
        link_element_to_tee_src_pad (bin->tee, bin->sub_bins[i].queue);
      } else {
        if (bin->sub_bins[bin->sub_bins[i].parent_index].tee) {
          link_element_to_tee_src_pad (bin->sub_bins[bin->sub_bins[i].
                  parent_index].tee, bin->sub_bins[i].queue);
        } else {
          NVGSTDS_LINK_ELEMENT (bin->sub_bins[bin->sub_bins[i].parent_index].
              secondary_preprocess, bin->sub_bins[i].secondary_preprocess);
        }
      }
    }
  }

  g_mutex_init (&bin->wait_lock);
  g_cond_init (&bin->wait_cond);

  ret = TRUE;

done:
  if (!ret) {
    NVGSTDS_ERR_MSG_V ("%s failed", __func__);
  }

  return ret;
}

void
destroy_secondary_preprocess_bin (NvDsSecondaryPreProcessBin * bin)
{
  if (bin->queue && bin->wait_for_secondary_preprocess_process_buf_probe_id) {
    GstPad *pad = gst_element_get_static_pad (bin->queue, "src");
    gst_pad_remove_probe (pad,
        bin->wait_for_secondary_preprocess_process_buf_probe_id);
    gst_object_unref (pad);
  }
}
