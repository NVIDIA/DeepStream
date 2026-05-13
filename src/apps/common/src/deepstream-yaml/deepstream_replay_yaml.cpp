/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
parse_replay_yaml (NvDsReplayConfig *config, gchar *cfg_file_path)
{
  gboolean ret = FALSE;
  YAML::Node configyml = YAML::LoadFile(cfg_file_path);

  // Initialize default values
  config->label_width = 0;
  config->label_height = 0;
  config->interval = 0;

  for(YAML::const_iterator itr = configyml["replay"].begin();
     itr != configyml["replay"].end(); ++itr)
  {
    std::string paramKey = itr->first.as<std::string>();
    if (paramKey == "enable") {
      config->enable = itr->second.as<gboolean>();
    } else if (paramKey == "label-dir") {
      std::string temp = itr->second.as<std::string>();
      char* str = (char*) malloc(sizeof(char) * 1024);
      std::strncpy (str, temp.c_str(), 1023);
      config->label_dir = (char*) malloc(sizeof(char) * 1024);
      if (!get_absolute_file_path_yaml (cfg_file_path, str,
              config->label_dir)) {
        g_printerr ("Error: Could not parse label-dir in replay.\n");
        g_free (str);
        goto done;
      }
      g_free (str);
    } else if (paramKey == "file-names") {
      std::string temp = itr->second.as<std::string>();
      config->file_names = g_strdup(temp.c_str());
    } else if (paramKey == "max-frame-nums") {
      std::string temp = itr->second.as<std::string>();
      config->max_frame_nums = g_strdup(temp.c_str());
    } else if (paramKey == "label-width") {
      config->label_width = itr->second.as<guint>();
    } else if (paramKey == "label-height") {
      config->label_height = itr->second.as<guint>();
    } else if (paramKey == "interval") {
      config->interval = itr->second.as<guint>();
    } else {
      cout << "Unknown key " << paramKey << " for replay" << endl;
    }
  }

  ret = TRUE;
done:
  if (!ret) {
    cout <<  __func__ << " failed" << endl;
  }
  return ret;
}
