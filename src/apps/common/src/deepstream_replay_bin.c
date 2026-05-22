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
