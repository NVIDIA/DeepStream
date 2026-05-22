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

#ifndef __NVGSTDS_REPLAY_H__
#define __NVGSTDS_REPLAY_H__


#ifdef __cplusplus
extern "C"
{
#endif

#include <gst/gst.h>
#include <stdint.h>

typedef struct
{
  gboolean enable;
  gchar* label_dir;
  gchar* file_names;
  gchar* max_frame_nums;
  guint label_width;
  guint label_height;
  guint interval;
} NvDsReplayConfig;

typedef struct
{
  GstElement *bin;
  GstElement *replay;
} NvDsReplayBin;

/**
 * Initialize @ref NvDsReplayBin. It creates and adds replay and
 * other elements needed for processing to the bin.
 * It also sets properties mentioned in the configuration file under
 * group @ref CONFIG_GROUP_REPLAY
 *
 * @param[in] config pointer of type @ref NvDsReplayConfig
 *            parsed from configuration file.
 * @param[in] bin pointer of type @ref NvDsReplayBin to be filled.
 *
 * @return true if bin created successfully.
 */
gboolean
create_replay_bin (NvDsReplayConfig* config, NvDsReplayBin * bin);

#ifdef __cplusplus
}
#endif

#endif
