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

#ifndef __NVGSTDS_PREPROCESS_H__
#define __NVGSTDS_PREPROCESS_H__

#include <gst/gst.h>

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct
{
  /** create a bin for the element only if enabled */
  gboolean enable;
  /*gie id on which preprocessing is to be done*/
  gint operate_on_gie_id;
  gboolean is_operate_on_gie_id_set;
  /** config file path having properties for preprocess */
  gchar *config_file_path;
} NvDsPreProcessConfig;

typedef struct
{
  GstElement *bin;
  GstElement *queue;
  GstElement *preprocess;
} NvDsPreProcessBin;

/**
 * Initialize @ref NvDsPreProcessBin. It creates and adds preprocess and
 * other elements needed for processing to the bin.
 * It also sets properties mentioned in the configuration file under
 * group @ref CONFIG_GROUP_PREPROCESS
 *
 * @param[in] config pointer to infer @ref NvDsPreProcessConfig parsed from
 *            configuration file.
 * @param[in] bin pointer to @ref NvDsPreProcessBin to be filled.
 *
 * @return true if bin created successfully.
 */
gboolean create_preprocess_bin (NvDsPreProcessConfig *config,
    NvDsPreProcessBin *bin);

#ifdef __cplusplus
}
#endif

#endif