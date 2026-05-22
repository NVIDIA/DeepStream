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
#include "deepstream_tiled_display.h"

gboolean
create_tiled_display_bin (NvDsTiledDisplayConfig * config,
    NvDsTiledDisplayBin * bin)
{
  gboolean ret = FALSE;

  bin->bin = gst_bin_new ("tiled_display_bin");
  if (!bin->bin) {
    NVGSTDS_ERR_MSG_V ("Failed to create 'tiled_display_bin'");
    goto done;
  }

  bin->queue =
      gst_element_factory_make (NVDS_ELEM_QUEUE, "tiled_display_queue");
  if (!bin->queue) {
    NVGSTDS_ERR_MSG_V ("Failed to create 'tiled_display_queue'");
    goto done;
  }

  if ((config->enable == 1) || (config->enable == 2))
    bin->tiler =
        gst_element_factory_make (NVDS_ELEM_TILER, "tiled_display_tiler");
  else if (config->enable == 3)
    bin->tiler =
        gst_element_factory_make ("identity", "tiled_display_identity");

  if (!bin->tiler) {
    NVGSTDS_ERR_MSG_V ("Failed to create 'tiled_display_tiler'");
    goto done;
  }

  /* Only set tiler-specific properties when using nvmultistreamtiler (enable == 1 or 2)
   * Not when using identity element (enable == 3) */
  if ((config->enable == 1) || (config->enable == 2)) {
    if (config->width)
      g_object_set (G_OBJECT (bin->tiler), "width", config->width, NULL);

    if (config->height)
      g_object_set (G_OBJECT (bin->tiler), "height", config->height, NULL);

    if (config->rows)
      g_object_set (G_OBJECT (bin->tiler), "rows", config->rows, NULL);

    if (config->columns)
      g_object_set (G_OBJECT (bin->tiler), "columns", config->columns, NULL);

    if (config->buffer_pool_size)
      g_object_set (G_OBJECT (bin->tiler), "buffer-pool-size",
          config->buffer_pool_size, NULL);

    if (config->square_seq_grid)
      g_object_set (G_OBJECT (bin->tiler), "square-seq-grid", config->square_seq_grid, NULL);

#ifdef IS_TEGRA
    if (config->compute_hw)
      g_object_set (G_OBJECT (bin->tiler), "compute-hw", config->compute_hw,
          NULL);
#endif

    g_object_set (G_OBJECT (bin->tiler), "gpu-id", config->gpu_id,
        "nvbuf-memory-type", config->nvbuf_memory_type, NULL);
  }

  gst_bin_add_many (GST_BIN (bin->bin), bin->queue, bin->tiler, NULL);

  NVGSTDS_LINK_ELEMENT (bin->queue, bin->tiler);

  NVGSTDS_BIN_ADD_GHOST_PAD (bin->bin, bin->queue, "sink");

  NVGSTDS_BIN_ADD_GHOST_PAD (bin->bin, bin->tiler, "src");

  ret = TRUE;
done:
  if (!ret) {
    NVGSTDS_ERR_MSG_V ("%s failed", __func__);
  }
  return ret;
}
