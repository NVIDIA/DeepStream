/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

using std::cout;
using std::endl;

gboolean
parse_dsanalytics_yaml (NvDsDsAnalyticsConfig *config, gchar* cfg_file_path)
{
  gboolean ret = FALSE;

  YAML::Node configyml = YAML::LoadFile(cfg_file_path);
  for(YAML::const_iterator itr = configyml["nvds-analytics"].begin();
     itr != configyml["nvds-analytics"].end(); ++itr)
  {
    std::string paramKey = itr->first.as<std::string>();
    if (paramKey == "enable") {
      config->enable = itr->second.as<gboolean>();
    } else if (paramKey == "config-file") {
      std::string temp = itr->second.as<std::string>();
      char* str = (char*) malloc(sizeof(char) * 1024);
      std::strncpy (str, temp.c_str(), 1023);
      config->config_file_path = (char*) malloc(sizeof(char) * 1024);
      if (!get_absolute_file_path_yaml (cfg_file_path, str,
            config->config_file_path)) {
        g_printerr ("Error: Could not parse config-file in dsanalytics.\n");
        g_free (str);
        goto done;
      }
      g_free (str);
    } else {
      cout << "[WARNING] Unknown param found in nvds-analytics: " << paramKey << endl;
    }
  }

  ret = TRUE;
done:
  if (!ret) {
    cout <<  __func__ << " failed" << endl;
  }
  return ret;
}