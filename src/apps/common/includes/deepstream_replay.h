/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
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
