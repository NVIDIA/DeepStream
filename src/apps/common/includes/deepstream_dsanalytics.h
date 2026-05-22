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
