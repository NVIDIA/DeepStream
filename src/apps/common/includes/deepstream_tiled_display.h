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

#ifndef __NVGSTDS_TILED_DISPLAY_H__
#define __NVGSTDS_TILED_DISPLAY_H__

#include <gst/gst.h>
#include "nvll_osd_struct.h"

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct
{
  GstElement *bin;
  GstElement *queue;
  GstElement *tiler;
} NvDsTiledDisplayBin;

typedef enum
{
  NV_DS_TILED_DISPLAY_DISABLE = 0,
  NV_DS_TILED_DISPLAY_ENABLE = 1,
  /** When user sets tiler group enable=2,
   * all sinks with the key: link-only-to-demux=1
   * shall be linked to demuxer's src_[source_id] pad
   * where source_id is the key set in this
   * corresponding [sink] group
   */
  NV_DS_TILED_DISPLAY_ENABLE_WITH_PARALLEL_DEMUX = 2
} NvDsTiledDisplayEnable;

typedef struct
{
  NvDsTiledDisplayEnable enable;
  guint rows;
  guint columns;
  guint width;
  guint height;
  guint gpu_id;
  guint nvbuf_memory_type;
  /**Compute Scaling HW to use
   * Applicable only for Jetson; x86 uses GPU by default
   * (0): Default          - Default, GPU for Tesla, VIC for Jetson
   * (1): GPU              - GPU
   * (2): VIC              - VIC
   *  */
  guint compute_hw;
  guint buffer_pool_size;
  guint square_seq_grid;
} NvDsTiledDisplayConfig;

/**
 * Initialize @ref NvDsTiledDisplayBin. It creates and adds tiling and
 * other elements needed for processing to the bin.
 * It also sets properties mentioned in the configuration file under
 * group @ref CONFIG_GROUP_TILED_DISPLAY
 *
 * @param[in] config pointer of type @ref NvDsTiledDisplayConfig
 *            parsed from configuration file.
 * @param[in] bin pointer to @ref NvDsTiledDisplayBin to be filled.
 *
 * @return true if bin created successfully.
 */
gboolean
create_tiled_display_bin (NvDsTiledDisplayConfig * config,
                          NvDsTiledDisplayBin * bin);

#ifdef __cplusplus
}
#endif

#endif
