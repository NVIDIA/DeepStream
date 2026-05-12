/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "deepstream_secondary_gie.h"

#define SECONDARY_GIE_CONFIG_KEY "secondary-gie-config"
#define SECONDARY_GIE_BIN_KEY "secondary-gie-bin"

#define GET_FILE_PATH(path) ((path) + (((path) && strstr ((path), "file://")) ? 7 : 0))

/**
 * Wait for all secondary inferences to complete the processing and then send
 * the processed buffer to downstream.
 * This is way of synchronization between all secondary infers and sending
 * buffer once meta data from all secondary infer components got attached.
 * This is needed because all secondary infers process same buffer in parallel.
 */
static GstPadProbeReturn
wait_queue_buf_probe (GstPad * pad, GstPadProbeInfo * info, gpointer u_data)
{
  NvDsSecondaryGieBin *bin = (NvDsSecondaryGieBin *) u_data;
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
  NvDsSecondaryGieBin *bin = (NvDsSecondaryGieBin *) u_data;
  if (info->type & GST_PAD_PROBE_TYPE_EVENT_BOTH) {
    GstEvent *event = (GstEvent *) info->data;
    if (event->type == GST_EVENT_EOS) {
      bin->stop = TRUE;
    }
  }

  return GST_PAD_PROBE_OK;
}

static void
write_infer_output_to_file (GstBuffer * buf,
    NvDsInferNetworkInfo * network_info, NvDsInferLayerInfo * layers_info,
    guint num_layers, guint batch_size, gpointer user_data)
{
  NvDsGieConfig *config = (NvDsGieConfig *) user_data;
  guint i;

  /* "gst_buffer_get_nvstream_meta()" can be called on the GstBuffer to get more
   * information about the buffer.*/

  for (i = 0; i < num_layers; i++) {
    NvDsInferLayerInfo *info = &layers_info[i];
    guint element_size = 0;
    FILE *file;
    gchar file_name[PATH_MAX];
    gchar layer_name[128];
    guint j;

    switch (info->dataType) {
      case FLOAT:
        element_size = 4;
        break;
      case HALF:
        element_size = 2;
        break;
      case INT32:
        element_size = 4;
        break;
      case INT8:
      case UINT8:
        element_size = 1;
        break;
      case INT64:
        element_size = 8;
        break;
    }

    g_strlcpy (layer_name, info->layerName, 128);
    for (j = 0; layer_name[j] != '\0'; j++) {
      layer_name[j] = (layer_name[j] == '/') ? '_' : layer_name[j];
    }

    g_snprintf (file_name, PATH_MAX,
        "%s/%s_batch%010lu_batchsize%02d.bin",
        config->raw_output_directory, layer_name,
        config->file_write_frame_num, batch_size);
    file_name[PATH_MAX - 1] = '\0';

    file = fopen (file_name, "w");
    if (!file) {
      g_printerr ("Could not open file '%s' for writing:%s\n",
          file_name, strerror (errno));
      continue;
    }
    fwrite (info->buffer, element_size,
        info->inferDims.numElements * batch_size, file);
    fclose (file);
  }
  config->file_write_frame_num++;
}


/**
 * Create secondary infer sub bin and sets properties mentioned
 * in configuration file.
 */
static gboolean
create_secondary_gie (NvDsGieConfig * configs1,
    NvDsSecondaryGieBinSubBin * subbins1, GstBin * bin, guint index)
{
  gboolean ret = FALSE;
  gchar elem_name[50];
  gchar operate_on_class_str[128];
  gint i;
  NvDsGieConfig *config = &configs1[index];
  NvDsSecondaryGieBinSubBin *subbin = &subbins1[index];
  gst_nvinfer_raw_output_generated_callback out_callback =
      write_infer_output_to_file;

  if (!subbin->create) {
    return TRUE;
  }

  g_snprintf (elem_name, sizeof (elem_name), "secondary_gie_%d_queue", index);

  if (subbin->parent_index == -1 ||
      subbins1[subbin->parent_index].num_children > 1) {
    subbin->queue = gst_element_factory_make (NVDS_ELEM_QUEUE, elem_name);
    if (!subbin->queue) {
      NVGSTDS_ERR_MSG_V ("Failed to create '%s'", elem_name);
      goto done;
    }
    gst_bin_add (bin, subbin->queue);
  }

  g_snprintf (elem_name, sizeof (elem_name), "secondary_gie_%d", index);
  subbin->secondary_gie = gst_element_factory_make (NVDS_ELEM_SGIE, elem_name);
  switch (config->plugin_type) {
    case NV_DS_GIE_PLUGIN_INFER:
      subbin->secondary_gie =
          gst_element_factory_make (NVDS_ELEM_SGIE, elem_name);
      break;
    case NV_DS_GIE_PLUGIN_INFER_SERVER:
      subbin->secondary_gie =
          gst_element_factory_make (NVDS_ELEM_INFER_SERVER, elem_name);
      break;
    case NV_DS_GIE_PLUGIN_VISION_ENCODER:
      subbin->secondary_gie =
          gst_element_factory_make ("nvdsvisionencoder", elem_name);
      break;
    default:
      NVGSTDS_ERR_MSG_V ("Failed to create %s on unknown plugin_type",
          elem_name);
      goto done;
  }

  if (!subbin->secondary_gie) {
    NVGSTDS_ERR_MSG_V ("Failed to create '%s'", elem_name);
    goto done;
  }

  operate_on_class_str[0] = '\0';
  for (i = 0; i < config->num_operate_on_class_ids; i++) {
    g_snprintf (operate_on_class_str + strlen (operate_on_class_str),
        sizeof (operate_on_class_str) -
        strlen (operate_on_class_str) - 1, "%d:",
        config->list_operate_on_class_ids[i]);
  }

  g_object_set (G_OBJECT (subbin->secondary_gie),
      "config-file-path", GET_FILE_PATH (config->config_file_path),
      "process-mode", 2, NULL);

  g_object_set (G_OBJECT (subbin->secondary_gie),
      "input-tensor-meta", config->input_tensor_meta, NULL);

  if (config->num_operate_on_class_ids != 0)
    g_object_set (G_OBJECT (subbin->secondary_gie),
        "infer-on-class-ids", operate_on_class_str, NULL);

  if (config->is_batch_size_set)
    g_object_set (G_OBJECT (subbin->secondary_gie),
        "batch-size", config->batch_size, NULL);

  if (config->is_interval_set)
    g_object_set (G_OBJECT (subbin->secondary_gie),
        "interval", config->interval, NULL);

  if (config->is_unique_id_set)
    g_object_set (G_OBJECT (subbin->secondary_gie),
        "unique-id", config->unique_id, NULL);

  if (config->is_operate_on_gie_id_set)
    g_object_set (G_OBJECT (subbin->secondary_gie),
        "infer-on-gie-id", config->operate_on_gie_id, NULL);

  if (config->raw_output_directory) {
    g_object_set (G_OBJECT (subbin->secondary_gie),
        "raw-output-generated-callback", out_callback,
        "raw-output-generated-userdata", config, NULL);
  }

  if (config->is_gpu_id_set &&
      NV_DS_GIE_PLUGIN_INFER_SERVER == config->plugin_type) {
    NVGSTDS_WARN_MSG_V
        ("gpu-id: %u in sgie: %s is ignored, only accept nvinferserver config",
        config->gpu_id, elem_name);
  }

  if (NV_DS_GIE_PLUGIN_INFER == config->plugin_type) {
    if (config->model_engine_file_path) {
      g_object_set (G_OBJECT (subbin->secondary_gie), "model-engine-file",
          GET_FILE_PATH (config->model_engine_file_path), NULL);
    }

    if (config->is_gpu_id_set)
      g_object_set (G_OBJECT (subbin->secondary_gie),
          "gpu-id", config->gpu_id, NULL);
  }

  if (NV_DS_GIE_PLUGIN_VISION_ENCODER == config->plugin_type) {
    if (config->model_variant)
      g_object_set (G_OBJECT (subbin->secondary_gie),
          "model-variant", config->model_variant, NULL);

    if (config->device)
      g_object_set (G_OBJECT (subbin->secondary_gie),
          "device", config->device, NULL);

    if (config->is_min_crop_size_set)
      g_object_set (G_OBJECT (subbin->secondary_gie),
          "min-crop-size", config->min_crop_size, NULL);

    g_object_set (G_OBJECT (subbin->secondary_gie),
        "verbose", config->verbose, NULL);

    if (config->backend)
      g_object_set (G_OBJECT (subbin->secondary_gie),
          "backend", config->backend, NULL);

    if (config->triton_url)
      g_object_set (G_OBJECT (subbin->secondary_gie),
          "triton-url", config->triton_url, NULL);

    if (config->triton_model)
      g_object_set (G_OBJECT (subbin->secondary_gie),
          "triton-model", config->triton_model, NULL);

    if (config->is_gpu_id_set)
      g_object_set (G_OBJECT (subbin->secondary_gie),
          "gpu-id", config->gpu_id, NULL);
  }

  gst_bin_add (bin, subbin->secondary_gie);

  if (subbin->num_children == 0) {
    g_snprintf (elem_name, sizeof (elem_name), "secondary_gie_%d_sink", index);
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
    g_snprintf (elem_name, sizeof (elem_name), "secondary_gie_%d_tee", index);
    subbin->tee = gst_element_factory_make (NVDS_ELEM_TEE, elem_name);
    if (!subbin->tee) {
      NVGSTDS_ERR_MSG_V ("Failed to create '%s'", elem_name);
      goto done;
    }
    gst_bin_add (bin, subbin->tee);
  }

  if (subbin->queue) {
    NVGSTDS_LINK_ELEMENT (subbin->queue, subbin->secondary_gie);
  }
  if (subbin->sink) {
    NVGSTDS_LINK_ELEMENT (subbin->secondary_gie, subbin->sink);
  }
  if (subbin->tee) {
    NVGSTDS_LINK_ELEMENT (subbin->secondary_gie, subbin->tee);
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
should_create_secondary_gie (NvDsGieConfig * config_array, guint num_configs,
    NvDsSecondaryGieBinSubBin * bins, guint index, gint primary_gie_id)
{
  NvDsGieConfig *config = &config_array[index];
  guint i;

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

  for (i = 0; i < num_configs; i++) {
    if (config->operate_on_gie_id == (gint) config_array[i].unique_id) {
      if (should_create_secondary_gie (config_array, num_configs, bins, i,
              primary_gie_id)) {
        bins[index].create = TRUE;
        bins[index].parent_index = i;
        bins[i].num_children++;
        return TRUE;
      }
      break;
    }
  }
  return FALSE;
}

gboolean
create_secondary_gie_bin (guint num_secondary_gie, guint primary_gie_unique_id,
    NvDsGieConfig * config_array, NvDsSecondaryGieBin * bin)
{
  gboolean ret = FALSE;
  guint i;
  GstPad *pad;

  bin->bin = gst_bin_new ("secondary_gie_bin");
  if (!bin->bin) {
    NVGSTDS_ERR_MSG_V ("Failed to create element 'secondary_gie_bin'");
    goto done;
  }

  bin->tee = gst_element_factory_make (NVDS_ELEM_TEE, "secondary_gie_bin_tee");
  if (!bin->tee) {
    NVGSTDS_ERR_MSG_V ("Failed to create element 'secondary_gie_bin_tee'");
    goto done;
  }

  gst_bin_add (GST_BIN (bin->bin), bin->tee);

  bin->queue =
      gst_element_factory_make (NVDS_ELEM_QUEUE, "secondary_gie_bin_queue");
  if (!bin->queue) {
    NVGSTDS_ERR_MSG_V ("Failed to create element 'secondary_gie_bin_queue'");
    goto done;
  }

  gst_bin_add (GST_BIN (bin->bin), bin->queue);

  pad = gst_element_get_static_pad (bin->queue, "src");
  bin->wait_for_sgie_process_buf_probe_id = gst_pad_add_probe (pad,
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

  for (i = 0; i < num_secondary_gie; i++) {
    should_create_secondary_gie (config_array, num_secondary_gie,
        bin->sub_bins, i, primary_gie_unique_id);
  }

  for (i = 0; i < num_secondary_gie; i++) {
    if (bin->sub_bins[i].create) {
      if (!create_secondary_gie (config_array, bin->sub_bins,
              GST_BIN (bin->bin), i)) {
        goto done;
      }
    }
  }

  for (i = 0; i < num_secondary_gie; i++) {
    if (bin->sub_bins[i].create) {
      if (bin->sub_bins[i].parent_index == -1) {
        link_element_to_tee_src_pad (bin->tee, bin->sub_bins[i].queue);
      } else {
        if (bin->sub_bins[bin->sub_bins[i].parent_index].tee) {
          link_element_to_tee_src_pad (bin->sub_bins[bin->sub_bins[i].
                  parent_index].tee, bin->sub_bins[i].queue);
        } else {
          NVGSTDS_LINK_ELEMENT (bin->sub_bins[bin->sub_bins[i].parent_index].
              secondary_gie, bin->sub_bins[i].secondary_gie);
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
destroy_secondary_gie_bin (NvDsSecondaryGieBin * bin)
{
  if (bin->queue && bin->wait_for_sgie_process_buf_probe_id) {
    GstPad *pad = gst_element_get_static_pad (bin->queue, "src");
    gst_pad_remove_probe (pad, bin->wait_for_sgie_process_buf_probe_id);
    gst_object_unref (pad);
  }
}
