/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef _NVGSTDS_DSANALYTICS_H_
#define _NVGSTDS_DSANALYTICS_H_

#include <gst/gst.h>

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct
{
  // Create a bin for the element only if enabled
  gboolean enable;
  guint unique_id;
  // Config file path having properties for the element
  gchar *config_file_path;
} NvDsDsAnalyticsConfig;

// Struct to store references to the bin and elements
typedef struct
{
  GstElement *bin;
  GstElement *queue;
  GstElement *elem_dsanalytics;
} NvDsDsAnalyticsBin;

// Function to create the bin and set properties
gboolean
create_dsanalytics_bin (NvDsDsAnalyticsConfig *config, NvDsDsAnalyticsBin *bin);

#ifdef __cplusplus
}
#endif

#endif /* _NVGSTDS_DSANALYTICS_H_ */
