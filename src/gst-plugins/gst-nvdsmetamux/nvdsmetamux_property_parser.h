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

#ifndef NVDSMETAMUX_PROPERTY_FILE_PARSER_H_
#define NVDSMETAMUX_PROPERTY_FILE_PARSER_H_

#include <gst/gst.h>
#include "gstnvdsmetamux.h"

/**
 * This file describes the Macro defined for config file property parser.
 */

/** max string length */
#define _PATH_MAX 4096

#define NVDSMETAMUX_PROPERTY "property"
#define NVDSMETAMUX_PROPERTY_ENABLE "enable"
#define NVDSMETAMUX_PROPERTY_ACTIVE_PAD "active-pad"
#define NVDSMETAMUX_PROPERTY_PTS_TOLERANCE "pts-tolerance"

#define NVDSMETAMUX_USER_CONFIGS "user-configs"

#define NVDSMETAMUX_GROUP "group-"
#define NVDSMETAMUX_GROUP_SRC_IDS_MODEL "src-ids-model"

/**
 * Get GstNvDsMetaMuxMemory structure associated with buffer allocated using
 * GstNvDsMetaMuxAllocator.
 *
 * @param nvdsmetamux pointer to GstNvDsMetaMux structure
 *
 * @param cfg_file_path config file path
 *
 * @return boolean denoting if successfully parsed config file
 */
gboolean
nvdsmetamux_parse_config_file (GstNvDsMetaMux *nvdsmetamux, gchar *cfg_file_path);

#endif /* NVDSMETAMUX_PROPERTY_FILE_PARSER_H_ */
