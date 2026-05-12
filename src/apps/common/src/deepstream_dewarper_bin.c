/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "deepstream_common.h"
#include "deepstream_dewarper.h"

gboolean
create_dewarper_bin (NvDsDewarperConfig * config, NvDsDewarperBin * bin)
{
  GstCaps *caps = NULL;
  gboolean ret = FALSE;
  GstCapsFeatures *feature = NULL;

  bin->bin = gst_bin_new ("dewarper_bin");
  if (!bin->bin) {
    NVGSTDS_ERR_MSG_V ("Failed to create 'dewarper_bin'");
    goto done;
  }

  bin->nvvidconv =
      gst_element_factory_make (NVDS_ELEM_VIDEO_CONV, "dewarper_conv");

  if (!bin->nvvidconv) {
    NVGSTDS_ERR_MSG_V ("Failed to create 'dewarper_conv'");
    goto done;
  }

  bin->queue = gst_element_factory_make (NVDS_ELEM_QUEUE, "dewarper_queue");
  if (!bin->queue) {
    NVGSTDS_ERR_MSG_V ("Failed to create 'dewarper_queue'");
    goto done;
  }

  bin->src_queue =
      gst_element_factory_make (NVDS_ELEM_QUEUE, "dewarper_src_queue");
  if (!bin->src_queue) {
    NVGSTDS_ERR_MSG_V ("Failed to create 'dewarper_src_queue'");
    goto done;
  }

  bin->conv_queue =
      gst_element_factory_make (NVDS_ELEM_QUEUE, "dewarper_conv_queue");
  if (!bin->conv_queue) {
    NVGSTDS_ERR_MSG_V ("Failed to create 'dewarper_conv_queue'");
    goto done;
  }

  bin->cap_filter =
      gst_element_factory_make (NVDS_ELEM_CAPS_FILTER, "dewarper_caps");
  if (!bin->cap_filter) {
    NVGSTDS_ERR_MSG_V ("Failed to create 'dewarper_caps'");
    goto done;
  }

  caps =
      gst_caps_new_simple ("video/x-raw", "format", G_TYPE_STRING, "RGBA",
      NULL);

  feature = gst_caps_features_new (MEMORY_FEATURES, NULL);
  gst_caps_set_features (caps, 0, feature);
  g_object_set (G_OBJECT (bin->cap_filter), "caps", caps, NULL);
  gst_caps_unref (caps);

  bin->nvdewarper = gst_element_factory_make (NVDS_ELEM_DEWARPER, NULL);
  if (!bin->nvdewarper) {
    NVGSTDS_ERR_MSG_V ("Failed to create 'nvdewarper'");
    goto done;
  }

  bin->dewarper_caps_filter =
      gst_element_factory_make (NVDS_ELEM_CAPS_FILTER, "dewarper_caps_filter");
  if (!bin->dewarper_caps_filter) {
    NVGSTDS_ERR_MSG_V ("Could not create 'dewarper_caps_filter'");
    goto done;
  }

  caps = gst_caps_new_simple ("video/x-raw", "format", G_TYPE_STRING,
      "RGBA", "width", GST_TYPE_INT_RANGE, 1, G_MAXINT,
      "height", GST_TYPE_INT_RANGE, 1, G_MAXINT, NULL);

  feature = gst_caps_features_new ("memory:NVMM", NULL);
  gst_caps_set_features (caps, 0, feature);

  g_object_set (G_OBJECT (bin->dewarper_caps_filter), "caps", caps, NULL);


  gst_bin_add_many (GST_BIN (bin->bin), bin->queue, bin->src_queue, bin->conv_queue,
      bin->nvvidconv, bin->cap_filter,
      bin->nvdewarper, bin->dewarper_caps_filter, NULL);

  g_object_set (G_OBJECT (bin->nvvidconv), "gpu-id", config->gpu_id, NULL);
  g_object_set (G_OBJECT (bin->nvvidconv), "nvbuf-memory-type",
      config->nvbuf_memory_type, NULL);

  g_object_set (G_OBJECT (bin->nvdewarper), "gpu-id", config->gpu_id, NULL);
  g_object_set (G_OBJECT (bin->nvdewarper), "config-file",
      config->config_file, NULL);

  g_object_set (G_OBJECT (bin->nvdewarper), "source-id", config->source_id,
      NULL);
  g_object_set (G_OBJECT (bin->nvdewarper), "nvbuf-memory-type",
      config->nvbuf_memory_type, NULL);
  g_object_set (G_OBJECT (bin->nvdewarper), "num-output-buffers",
      config->num_out_buffers, NULL);

  NVGSTDS_LINK_ELEMENT (bin->queue, bin->nvvidconv);

  NVGSTDS_LINK_ELEMENT (bin->nvvidconv, bin->cap_filter);
  NVGSTDS_LINK_ELEMENT (bin->cap_filter, bin->conv_queue);

  NVGSTDS_LINK_ELEMENT (bin->conv_queue, bin->nvdewarper);

  NVGSTDS_LINK_ELEMENT (bin->nvdewarper, bin->dewarper_caps_filter);
  NVGSTDS_LINK_ELEMENT (bin->dewarper_caps_filter, bin->src_queue);

  NVGSTDS_BIN_ADD_GHOST_PAD (bin->bin, bin->queue, "sink");

  NVGSTDS_BIN_ADD_GHOST_PAD (bin->bin, bin->src_queue, "src");

  ret = TRUE;
done:
  if (caps) {
    gst_caps_unref (caps);
  }

  if (!ret) {
    NVGSTDS_ERR_MSG_V ("%s failed", __func__);
  }
  return ret;
}
