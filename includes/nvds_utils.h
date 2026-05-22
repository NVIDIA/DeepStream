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

#ifndef __NVDS_COMMON_UTILS_H__
#define __NVDS_COMMON_UTILS_H__

#ifdef __cplusplus
extern "C"
{
#endif

/*
 * Release memory allocated for gobjects
 */
void free_gobjs(GKeyFile *gcfg_file, GError *error, gchar **keys,  gchar *key_name);

/*
 * Internal function to verify if the config string value is a quoted string
 * config key = "value"
 * If so, strip beginning and ending quote
 */
NvDsMsgApiErrorType strip_quote(char *cfg_value, const char *cfg_key, const char *log_cat);

/*
 *Function to open a file and search config value for a config key
 *If found, place the result in cfg_val
 */
NvDsMsgApiErrorType fetch_config_value(char *config_path, const char *cfg_key, char *cfg_val, int len, const char *log_category);

/*
 * Get hostname with IP address or Kubernetes pod information
 * Returns a newly allocated string containing either:
 *   - Kubernetes: "pod-name (pod-dns)"
 *   - Regular: "hostname (IP)"
 * Caller must free with g_free().
 */
char* nvds_get_hostname_with_ip(void);

#ifdef __cplusplus
}

#include <string>

/*
 *Generate hash value of a string using SHA256
 */
std::string generate_sha256_hash(std::string str);

/*
 * remove leading & trailing whitespaces from a string
 */
std::string trim(std::string str);

/*
 * Sort string of format key=value;key1=value1
 * ex: str = "pq=89;xyz=33;abc=123;"
 * output  = "abc=123;pq=89;xyz=33"
 */
std::string sort_key_value_pairs(std::string str);

#endif

#endif
