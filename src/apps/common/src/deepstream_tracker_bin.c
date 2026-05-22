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
#include "deepstream_tracker.h"

GST_DEBUG_CATEGORY_EXTERN (NVDS_APP);

gboolean
create_tracking_bin (NvDsTrackerConfig * config, NvDsTrackerBin * bin)
{
  gboolean ret = FALSE;

  bin->bin = gst_bin_new ("tracking_bin");
  if (!bin->bin) {
    NVGSTDS_ERR_MSG_V ("Failed to create 'tracking_bin'");
    goto done;
  }

  bin->tracker =
      gst_element_factory_make (NVDS_ELEM_TRACKER, "tracking_tracker");
  if (!bin->tracker) {
    NVGSTDS_ERR_MSG_V ("Failed to create 'tracking_tracker'");
    goto done;
  }

  g_object_set (G_OBJECT (bin->tracker), "tracker-width", config->width,
      "tracker-height", config->height,
      "gpu-id", config->gpu_id,
      "ll-config-file", config->ll_config_file,
      "ll-lib-file", config->ll_lib_file, NULL);

  g_object_set (G_OBJECT (bin->tracker), "display-tracking-id",
      config->display_tracking_id, NULL);

  g_object_set (G_OBJECT (bin->tracker), "tracking-id-reset-mode",
      config->tracking_id_reset_mode, NULL);

  g_object_set (G_OBJECT (bin->tracker), "tracking-surface-type",
      config->tracking_surface_type, NULL);

  g_object_set (G_OBJECT (bin->tracker), "input-tensor-meta",
      config->input_tensor_meta, NULL);

  g_object_set (G_OBJECT (bin->tracker), "tensor-meta-gie-id",
      config->input_tensor_gie_id, NULL);

  g_object_set (G_OBJECT (bin->tracker), "compute-hw",
      config->compute_hw, NULL);

  g_object_set (G_OBJECT (bin->tracker), "user-meta-pool-size",
      config->user_meta_pool_size, NULL);

  g_object_set (G_OBJECT (bin->tracker), "sub-batches",
      config->sub_batches, NULL);

  g_object_set (G_OBJECT (bin->tracker), "sub-batch-err-recovery-trial-cnt",
      config->sub_batch_err_recovery_trial_cnt, NULL);

  g_object_set (G_OBJECT (bin->tracker), "operate-on-class-ids",
    config->operate_on_class_ids, NULL);


  gst_bin_add_many (GST_BIN (bin->bin), bin->tracker, NULL);

  NVGSTDS_BIN_ADD_GHOST_PAD (bin->bin, bin->tracker, "sink");

  NVGSTDS_BIN_ADD_GHOST_PAD (bin->bin, bin->tracker, "src");

  ret = TRUE;

  GST_CAT_DEBUG (NVDS_APP, "Tracker bin created successfully");

done:
  if (!ret) {
    NVGSTDS_ERR_MSG_V ("%s failed", __func__);
  }
  return ret;
}
