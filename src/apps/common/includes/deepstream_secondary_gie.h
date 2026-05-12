/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2019 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef __NVGSTDS_SECONDARY_GIE_H__
#define __NVGSTDS_SECONDARY_GIE_H__

#include "deepstream_gie.h"

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct
{
  GstElement *queue;
  GstElement *secondary_gie;
  GstElement *tee;
  GstElement *sink;
  gboolean create;
  guint num_children;
  gint parent_index;
} NvDsSecondaryGieBinSubBin;

typedef struct
{
  GstElement *bin;
  GstElement *tee;
  GstElement *queue;
  gulong wait_for_sgie_process_buf_probe_id;
  gboolean stop;
  gboolean flush;
  NvDsSecondaryGieBinSubBin sub_bins[MAX_SECONDARY_GIE_BINS];
  GMutex wait_lock;
  GCond wait_cond;
} NvDsSecondaryGieBin;

/**
 * Initialize @ref NvDsSecondaryGieBin. It creates and adds secondary infer and
 * other elements needed for processing to the bin.
 * It also sets properties mentioned in the configuration file under
 * group @ref CONFIG_GROUP_SECONDARY_GIE
 *
 * @param[in] num_secondary_gie number of secondary infers.
 * @param[in] primary_gie_unique_id Unique id of primary infer to work on.
 * @param[in] config_array array of pointers of type @ref NvDsGieConfig
 *            parsed from configuration file.
 * @param[in] bin pointer to @ref NvDsSecondaryGieBin to be filled.
 *
 * @return true if bin created successfully.
 */
gboolean create_secondary_gie_bin (guint num_secondary_gie,
    guint primary_gie_unique_id, NvDsGieConfig *config_array,
    NvDsSecondaryGieBin *bin);

/**
 * Release the resources.
 */
void destroy_secondary_gie_bin (NvDsSecondaryGieBin *bin);

#ifdef __cplusplus
}
#endif

#endif
