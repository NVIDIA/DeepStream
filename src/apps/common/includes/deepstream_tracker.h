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

#ifndef __NVGSTDS_TRACKER_H__
#define __NVGSTDS_TRACKER_H__

#include <gst/gst.h>
#include <stdint.h>
#include "nvds_tracker_meta.h"

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct
{
  gboolean enable;
  gint width;
  gint height;
  guint gpu_id;
  guint tracking_surf_type;
  gchar* ll_config_file;
  gchar* ll_lib_file;
  guint tracking_surface_type;
  gboolean display_tracking_id;
  guint tracking_id_reset_mode;
  gboolean input_tensor_meta;
  guint input_tensor_gie_id;
  guint compute_hw;
  guint user_meta_pool_size;
  gchar* sub_batches;
  gint sub_batch_err_recovery_trial_cnt;
  gchar* operate_on_class_ids;
} NvDsTrackerConfig;

typedef struct
{
  GstElement *bin;
  GstElement *tracker;
} NvDsTrackerBin;

typedef uint64_t NvDsTrackerStreamId;

/**
 * Initialize @ref NvDsTrackerBin. It creates and adds tracker and
 * other elements needed for processing to the bin.
 * It also sets properties mentioned in the configuration file under
 * group @ref CONFIG_GROUP_TRACKER
 *
 * @param[in] config pointer of type @ref NvDsTrackerConfig
 *            parsed from configuration file.
 * @param[in] bin pointer of type @ref NvDsTrackerBin to be filled.
 *
 * @return true if bin created successfully.
 */
gboolean
create_tracking_bin (NvDsTrackerConfig * config, NvDsTrackerBin * bin);

#ifdef __cplusplus
}
#endif

#endif
