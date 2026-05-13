/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef __NVGSTDS_SECONDARY_PREPROCESS_H__
#define __NVGSTDS_SECONDARY_PREPROCESS_H__

#include <gst/gst.h>
#include "deepstream_preprocess.h"

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct
{
  GstElement *queue;
  GstElement *secondary_preprocess;
  GstElement *tee;
  GstElement *sink;
  gboolean create;
  guint num_children;
  gint parent_index;
} NvDsSecondaryPreProcessBinSubBin;

typedef struct
{
  GstElement *bin;
  GstElement *tee;
  GstElement *queue;
  gulong wait_for_secondary_preprocess_process_buf_probe_id;
  gboolean stop;
  gboolean flush;
  NvDsSecondaryPreProcessBinSubBin sub_bins[MAX_SECONDARY_GIE_BINS];
  GMutex wait_lock;
  GCond wait_cond;
} NvDsSecondaryPreProcessBin;

/**
 * Initialize @ref NvDsSecondaryPreProcessBin. It creates and adds secondary preprocess and
 * other elements needed for processing to the bin.
 * It also sets properties mentioned in the configuration file under
 * group @ref CONFIG_GROUP_SECONDARY_PREPROCESS
 *
 * @param[in] num_secondary_gie number of secondary preprocess.
 * @param[in] config_array array of pointers of type @ref NvDsPreProcessConfig
 *            parsed from configuration file.
 * @param[in] bin pointer to @ref NvDsSecondaryPreProcessBin to be filled.
 *
 * @return true if bin created successfully.
 */
gboolean create_secondary_preprocess_bin (guint num_secondary_preprocess,
    guint primary_gie_unique_id,
    NvDsPreProcessConfig *config_array,
    NvDsSecondaryPreProcessBin *bin);

/**
 * Release the resources.
 */
void destroy_secondary_preprocess_bin (NvDsSecondaryPreProcessBin *bin);

#ifdef __cplusplus
}
#endif

#endif