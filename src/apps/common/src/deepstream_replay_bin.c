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

#include "deepstream_common.h"
#include "deepstream_replay.h"

GST_DEBUG_CATEGORY_EXTERN (NVDS_APP);

gboolean
create_replay_bin (NvDsReplayConfig *config, NvDsReplayBin *bin)
{
  gboolean ret = FALSE;

  bin->bin = gst_bin_new ("replay_bin");
  if (!bin->bin) {
    NVGSTDS_ERR_MSG_V ("Failed to create 'replay_bin'");
    goto done;
  }

  bin->replay =
      gst_element_factory_make (NVDS_ELEM_REPLAY, "replay");
  if (!bin->replay) {
    NVGSTDS_ERR_MSG_V ("Failed to create 'replay'");
    goto done;
  }

  g_object_set (G_OBJECT (bin->replay), "label-dir", config->label_dir,
      "file-names", config->file_names, "max-frame-nums", config->max_frame_nums,
	  "label-width", config->label_width, "label-height", config->label_height,
	  "interval", config->interval, NULL);

  gst_bin_add_many (GST_BIN (bin->bin), bin->replay, NULL);

  NVGSTDS_BIN_ADD_GHOST_PAD (bin->bin, bin->replay, "sink");

  NVGSTDS_BIN_ADD_GHOST_PAD (bin->bin, bin->replay, "src");

  ret = TRUE;

  GST_CAT_DEBUG (NVDS_APP, "Replay bin created successfully");

done:
  if (!ret) {
    NVGSTDS_ERR_MSG_V ("%s failed", __func__);
  }
  return ret;
}
