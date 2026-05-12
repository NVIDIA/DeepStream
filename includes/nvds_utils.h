/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
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
