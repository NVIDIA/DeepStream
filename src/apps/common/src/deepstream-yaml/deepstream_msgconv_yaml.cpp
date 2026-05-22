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

#include "deepstream_common.h"
#include "deepstream_config_yaml.h"
#include <string>
#include <cstring>
#include <iostream>

using std::cout;
using std::endl;

gboolean
parse_msgconv_yaml (NvDsSinkMsgConvBrokerConfig *config, std::string group_str, gchar *cfg_file_path)
{
  gboolean ret = FALSE;
  YAML::Node configyml = YAML::LoadFile(cfg_file_path);
  char *group = (char*) malloc(sizeof(char) * 1024);
  std::strncpy (group, group_str.c_str(), 1023);

  for(YAML::const_iterator itr = configyml[group_str].begin();
     itr != configyml[group_str].end(); ++itr)
  {
    std::string paramKey = itr->first.as<std::string>();

    if (paramKey == "enable") {
      config->enable = itr->second.as<gboolean>();
    } else if (paramKey == "msg-conv-config") {
      std::string temp = itr->second.as<std::string>();
      char* str = (char*) malloc(sizeof(char) * 1024);
      std::strncpy (str, temp.c_str(), 1023);
      config->config_file_path = (char*) malloc(sizeof(char) * 1024);
      if (!get_absolute_file_path_yaml (cfg_file_path, str,
              config->config_file_path)) {
        g_printerr ("Error: Could not parse msg-conv-config in %s.\n", group);
        g_free (str);
        goto done;
      }
      g_free (str);
    } else if (paramKey == "msg-conv-payload-type") {
      config->conv_payload_type = itr->second.as<guint>();
    } else if (paramKey == "msg-conv-msg2p-lib") {
      std::string temp = itr->second.as<std::string>();
      char* str = (char*) malloc(sizeof(char) * 1024);
      std::strncpy (str, temp.c_str(), 1023);
      config->conv_msg2p_lib = (char*) malloc(sizeof(char) * 1024);
      if (!get_absolute_file_path_yaml (cfg_file_path, str,
              config->conv_msg2p_lib)) {
        g_printerr ("Error: Could not parse msg-conv-msg2p-lib in %s.\n", group);
        g_free (str);
        goto done;
      }
      g_free (str);
    } else if (paramKey == "msg-conv-comp-id") {
      config->conv_comp_id = itr->second.as<guint>();
    } else if (paramKey == "debug-payload-dir") {
      std::string temp = itr->second.as<std::string>();
      char* str = (char*) malloc(sizeof(char) * 1024);
      std::strncpy (str, temp.c_str(), 1023);
      config->debug_payload_dir = (char*) malloc(sizeof(char) * 1024);
      if (!get_absolute_file_path_yaml (cfg_file_path, str,
              config->debug_payload_dir)) {
        g_printerr ("Error: Could not parse debug-payload-dir in %s.\n", group);
        g_free (str);
        goto done;
      }
      g_free (str);
    } else if (paramKey == "multiple-payloads") {
      config->multiple_payloads = itr->second.as<gboolean>();
    } else if (paramKey == "msg-conv-msg2p-new-api") {
      config->conv_msg2p_new_api = itr->second.as<gboolean>();
    } else if (paramKey == "msg-conv-frame-interval") {
      config->conv_frame_interval = itr->second.as<guint>();
    } else if (paramKey == "msg-conv-dummy-payload") {
      config->conv_dummy_payload = itr->second.as<gboolean>();
    }
  }

  ret = TRUE;
done:
  if (!ret) {
    cout <<  __func__ << " failed" << endl;
  }
  if (group) {
    g_free (group);
  }
  return ret;
}