/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <vector>
#include <string>
#include <cstring>
#include <iostream>
#include <gst/gst.h>

#define _PATH_MAX 1024

/**
* Function definition to split semicolon separated string value and store tokens
* in a vector
*
* @param[in]  input A semicolon separated string.
* @return Vector containing tokens (i.e. uri(s))
*/
std::vector<std::string>
split_string (std::string input);

/**
* Function definition to get the absolute path of a file.
*
* @param[in]  cfg_file_path YAML config file name/path.
* @param[in]  file_path File name of whose absolute path is to be obtained.
* @param[in]  abs_path_str An empty char pointer. At the end of function call,
*             it contains the full path of the file.
* @return Boolean value on the basis of absolute path value.
*/
gboolean
get_absolute_file_path_yaml (
    const gchar * cfg_file_path, const gchar * file_path,
    char *abs_path_str);
/** @} */

