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

#include "deepstream_common.h"
#include "deepstream_dsexample.h"


// Create bin, add queue and the element, link all elements and ghost pads,
// Set the element properties from the parsed config
gboolean
create_dsexample_bin (NvDsDsExampleConfig * config, NvDsDsExampleBin * bin)
{
  GstCaps *caps = NULL;
  gboolean ret = FALSE;

  bin->bin = gst_bin_new ("dsexample_bin");
  if (!bin->bin) {
    NVGSTDS_ERR_MSG_V ("Failed to create 'dsexample_bin'");
    goto done;
  }

  bin->queue = gst_element_factory_make (NVDS_ELEM_QUEUE, "dsexample_queue");
  if (!bin->queue) {
    NVGSTDS_ERR_MSG_V ("Failed to create 'dsexample_queue'");
    goto done;
  }

  bin->elem_dsexample =
      gst_element_factory_make (NVDS_ELEM_DSEXAMPLE_ELEMENT, "dsexample0");
  if (!bin->elem_dsexample) {
    NVGSTDS_ERR_MSG_V ("Failed to create 'dsexample0'");
    goto done;
  }

  bin->pre_conv =
      gst_element_factory_make (NVDS_ELEM_VIDEO_CONV, "dsexample_conv0");
  if (!bin->pre_conv) {
    NVGSTDS_ERR_MSG_V ("Failed to create 'dsexample_conv0'");
    goto done;
  }

  bin->cap_filter =
      gst_element_factory_make (NVDS_ELEM_CAPS_FILTER, "dsexample_caps");
  if (!bin->cap_filter) {
    NVGSTDS_ERR_MSG_V ("Failed to create 'dsexample_caps'");
    goto done;
  }

  if (config->blur_objects) {
    caps =
        gst_caps_new_simple ("video/x-raw", "format", G_TYPE_STRING,
        "RGBA", NULL);

    GstCapsFeatures *feature = NULL;
    feature = gst_caps_features_new (MEMORY_FEATURES, NULL);
    gst_caps_set_features (caps, 0, feature);

    g_object_set (G_OBJECT (bin->cap_filter), "caps", caps, NULL);

    gst_caps_unref (caps);
  }

  gst_bin_add_many (GST_BIN (bin->bin), bin->queue,
      bin->pre_conv, bin->cap_filter, bin->elem_dsexample, NULL);

  NVGSTDS_LINK_ELEMENT (bin->queue, bin->pre_conv);
  NVGSTDS_LINK_ELEMENT (bin->pre_conv, bin->cap_filter);
  NVGSTDS_LINK_ELEMENT (bin->cap_filter, bin->elem_dsexample);

  NVGSTDS_BIN_ADD_GHOST_PAD (bin->bin, bin->queue, "sink");

  NVGSTDS_BIN_ADD_GHOST_PAD (bin->bin, bin->elem_dsexample, "src");

  g_object_set (G_OBJECT (bin->elem_dsexample),
      "full-frame", config->full_frame,
      "processing-width", config->processing_width,
      "processing-height", config->processing_height,
      "unique-id", config->unique_id,
      "gpu-id", config->gpu_id,
      "blur-objects", config->blur_objects, NULL);
  if (config->batch_size) {
    g_object_set (G_OBJECT (bin->elem_dsexample), "batch-size", config->batch_size, NULL);
  }
  g_object_set (G_OBJECT (bin->pre_conv), "gpu-id", config->gpu_id, NULL);

  g_object_set (G_OBJECT (bin->pre_conv), "nvbuf-memory-type",
      config->nvbuf_memory_type, NULL);

  ret = TRUE;

done:
  if (!ret) {
    NVGSTDS_ERR_MSG_V ("%s failed", __func__);
  }

  return ret;
}
