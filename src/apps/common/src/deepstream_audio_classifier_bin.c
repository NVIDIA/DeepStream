/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "deepstream_audio_classifier.h"

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
    guint j;
    FILE *file;
    gchar file_name[PATH_MAX];
    gchar layer_name[128];

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

gboolean
create_audio_classifier_bin (NvDsGieConfig * config,
    NvDsAudioClassifierBin * bin)
{
  gboolean ret = FALSE;
  gst_nvinfer_raw_output_generated_callback out_callback =
      write_infer_output_to_file;

  bin->bin = gst_bin_new ("audio_classifier_bin");
  if (!bin->bin) {
    NVGSTDS_ERR_MSG_V ("Failed to create 'audio_classifier_bin'");
    goto done;
  }

  bin->queue = gst_element_factory_make (NVDS_ELEM_QUEUE, "classifier_queue");
  if (!bin->queue) {
    NVGSTDS_ERR_MSG_V ("Failed to create 'classifier_queue'");
    goto done;
  }

  bin->classifier =
      gst_element_factory_make (NVDS_ELEM_INFER_AUDIO, "audio_classifier");
  if (!bin->classifier) {
    NVGSTDS_ERR_MSG_V ("Failed to create 'audio_classifier'");
    goto done;
  }

  g_object_set (G_OBJECT (bin->classifier),
      "config-file-path", GET_FILE_PATH (config->config_file_path), NULL);

  if (config->is_batch_size_set)
    g_object_set (G_OBJECT (bin->classifier),
        "batch-size", config->batch_size, NULL);

  if (config->is_unique_id_set)
    g_object_set (G_OBJECT (bin->classifier),
        "unique-id", config->unique_id, NULL);

  if (config->is_gpu_id_set)
    g_object_set (G_OBJECT (bin->classifier), "gpu-id", config->gpu_id, NULL);

  if (config->model_engine_file_path)
    g_object_set (G_OBJECT (bin->classifier), "model-engine-file",
        GET_FILE_PATH (config->model_engine_file_path), NULL);

  if (config->is_frame_size_set)
    g_object_set (G_OBJECT (bin->classifier),
        "audio-framesize", config->frame_size, NULL);

  if (config->is_hop_size_set)
    g_object_set (G_OBJECT (bin->classifier),
        "audio-hopsize", config->hop_size, NULL);

  if (config->audio_transform) {
    GstStructure *p;

    p = gst_structure_from_string (config->audio_transform, NULL);
    g_object_set (G_OBJECT (bin->classifier), "audio-transform", p, NULL);
  }

  if (config->raw_output_directory) {
    g_object_set (G_OBJECT (bin->classifier),
        "raw-output-generated-callback", out_callback,
        "raw-output-generated-userdata", config, NULL);
  }

  gst_bin_add_many (GST_BIN (bin->bin), bin->queue, bin->classifier, NULL);

  NVGSTDS_LINK_ELEMENT (bin->queue, bin->classifier);

  NVGSTDS_BIN_ADD_GHOST_PAD (bin->bin, bin->classifier, "src");

  NVGSTDS_BIN_ADD_GHOST_PAD (bin->bin, bin->queue, "sink");

  ret = TRUE;
done:
  if (!ret) {
    NVGSTDS_ERR_MSG_V ("%s failed", __func__);
  }
  return ret;
}
