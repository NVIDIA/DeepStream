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

#ifndef __YAML_PARSER_H__
#define __YAML_PARSER_H__
#include <glib.h>
#include <nvdsinfer_context.h>
 
 /*
* parameters of calibrator
*/
struct cfg_params{
  /* model tensor width*/
  int m_tensor_width;
  /*sr model tensor height*/
  int m_tensor_height;
};

gboolean gst_parse_context_params_yaml (const gchar * cfg_file_path, cfg_params& cal_params);

#endif