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
#include "deepstream_dsanalytics.h"


// Create bin, add queue and the element, link all elements and ghost pads,
// Set the element properties from the parsed config
gboolean
create_dsanalytics_bin (NvDsDsAnalyticsConfig * config,
    NvDsDsAnalyticsBin * bin)
{
  gboolean ret = FALSE;

  bin->bin = gst_bin_new ("dsanalytics_bin");
  if (!bin->bin) {
    NVGSTDS_ERR_MSG_V ("Failed to create 'dsanalytics_bin'");
    goto done;
  }

  bin->queue = gst_element_factory_make (NVDS_ELEM_QUEUE, "dsanalytics_queue");
  if (!bin->queue) {
    NVGSTDS_ERR_MSG_V ("Failed to create 'dsanalytics_queue'");
    goto done;
  }

  bin->elem_dsanalytics =
      gst_element_factory_make (NVDS_ELEM_DSANALYTICS_ELEMENT, "dsanalytics0");
  if (!bin->elem_dsanalytics) {
    NVGSTDS_ERR_MSG_V ("Failed to create 'dsanalytics0'");
    goto done;
  }

  gst_bin_add_many (GST_BIN (bin->bin), bin->queue,
      bin->elem_dsanalytics, NULL);

  NVGSTDS_LINK_ELEMENT (bin->queue, bin->elem_dsanalytics);

  NVGSTDS_BIN_ADD_GHOST_PAD (bin->bin, bin->queue, "sink");

  NVGSTDS_BIN_ADD_GHOST_PAD (bin->bin, bin->elem_dsanalytics, "src");

  g_object_set (G_OBJECT (bin->elem_dsanalytics),
      "config-file", config->config_file_path, NULL);

  ret = TRUE;

done:
  if (!ret) {
    NVGSTDS_ERR_MSG_V ("%s failed", __func__);
  }

  return ret;
}
