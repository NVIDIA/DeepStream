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
