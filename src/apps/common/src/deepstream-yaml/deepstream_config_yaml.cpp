/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "deepstream_common.h"
#include "deepstream_config_yaml.h"
#include <string>
#include <cstring>
#include <iostream>

#define _PATH_MAX 1024

/* Separate a config file entry with delimiters
 * into strings. */
std::vector<std::string> split_string (std::string input) {
  std::vector<int> positions;
  for (unsigned int i = 0; i < input.size(); i++) {
    if (input[i] == ';')
      positions.push_back(i);
  }
  std::vector<std::string> ret;
  int prev = 0;
  for (auto &j: positions) {
    std::string temp = input.substr(prev, j - prev);
    ret.push_back(temp);
    prev = j + 1;
  }
  ret.push_back(input.substr(prev, input.size() - prev));
  return ret;
}

/* Get the absolute path of a file mentioned in the config given a
 * file path absolute/relative to the config file. */
gboolean
get_absolute_file_path_yaml (
    const gchar * cfg_file_path, const gchar * file_path,
    char *abs_path_str)
{
  gchar abs_cfg_path[PATH_MAX + 1];
  gchar abs_real_file_path[PATH_MAX + 1];
  gchar *abs_file_path;
  gchar *delim;

  /* Absolute path. No need to resolve further. */
  if (file_path[0] == '/') {
    /* Check if the file exists, return error if not. */
    if (!realpath (file_path, abs_real_file_path)) {
      /* Ignore error if file does not exist and use the unresolved path. */
      if (errno != ENOENT)
        return FALSE;
    }
    g_strlcpy (abs_path_str, abs_real_file_path, _PATH_MAX);
    return TRUE;
  }

  /* Get the absolute path of the config file. */
  if (!realpath (cfg_file_path, abs_cfg_path)) {
    return FALSE;
  }

  /* Remove the file name from the absolute path to get the directory of the
   * config file. */
  delim = g_strrstr (abs_cfg_path, "/");
  *(delim + 1) = '\0';

  /* Get the absolute file path from the config file's directory path and
   * relative file path. */
  abs_file_path = g_strconcat (abs_cfg_path, file_path, nullptr);

  /* Resolve the path.*/
  if (realpath (abs_file_path, abs_real_file_path) == nullptr) {
    /* Ignore error if file does not exist and use the unresolved path. */
    if (errno == ENOENT)
      g_strlcpy (abs_real_file_path, abs_file_path, _PATH_MAX);
    else {
      if (abs_file_path)
        g_free (abs_file_path);
      return FALSE;
    }
  }

  g_free (abs_file_path);

  g_strlcpy (abs_path_str, abs_real_file_path, _PATH_MAX);
  return TRUE;
}
