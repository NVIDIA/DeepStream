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

#ifndef __NVGSTDS_PRIMARY_GIE_H__
#define __NVGSTDS_PRIMARY_GIE_H__

#ifdef __cplusplus
extern "C"
{
#endif

#include "deepstream_gie.h"

typedef struct
{
  GstElement *bin;
  GstElement *queue;
  GstElement *nvvidconv;
  GstElement *primary_gie;
} NvDsPrimaryGieBin;

/**
 * Initialize @ref NvDsPrimaryGieBin. It creates and adds primary infer and
 * other elements needed for processing to the bin.
 * It also sets properties mentioned in the configuration file under
 * group @ref CONFIG_GROUP_PRIMARY_GIE
 *
 * @param[in] config pointer to infer @ref NvDsGieConfig parsed from
 *            configuration file.
 * @param[in] bin pointer to @ref NvDsPrimaryGieBin to be filled.
 *
 * @return true if bin created successfully.
 */
gboolean create_primary_gie_bin (NvDsGieConfig *config,
    NvDsPrimaryGieBin *bin);

#ifdef __cplusplus
}
#endif

#endif
