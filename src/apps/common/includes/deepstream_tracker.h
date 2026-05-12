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
