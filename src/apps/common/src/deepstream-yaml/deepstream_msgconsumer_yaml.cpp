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
parse_msgconsumer_yaml (NvDsMsgConsumerConfig *config, std::string group, gchar *cfg_file_path)
{
  gboolean ret = FALSE;
  YAML::Node configyml = YAML::LoadFile(cfg_file_path);
  for(YAML::const_iterator itr = configyml[group].begin();
     itr != configyml[group].end(); ++itr)
  {
    std::string paramKey = itr->first.as<std::string>();
    if (paramKey == "enable") {
      config->enable = itr->second.as<gboolean>();
    } else if (paramKey == "config-file") {
      std::string temp = itr->second.as<std::string>();
      config->config_file_path = (char*) malloc(sizeof(char) * 1024);
      std::strncpy (config->config_file_path, temp.c_str(), 1023);
    } else if (paramKey == "proto-lib") {
      std::string temp = itr->second.as<std::string>();
      config->proto_lib = (char*) malloc(sizeof(char) * 1024);
      std::strncpy (config->proto_lib, temp.c_str(), 1023);
    } else if (paramKey == "conn-str") {
      std::string temp = itr->second.as<std::string>();
      config->conn_str = (char*) malloc(sizeof(char) * 1024);
      std::strncpy (config->conn_str, temp.c_str(), 1023);
    } else if (paramKey == "sensor-list-file") {
      std::string temp = itr->second.as<std::string>();
      char* str = (char*) malloc(sizeof(char) * 1024);
      std::strncpy (str, temp.c_str(), 1023);
      config->sensor_list_file = (char*) malloc(sizeof(char) * 1024);
      if (!get_absolute_file_path_yaml (cfg_file_path, str,
              config->sensor_list_file)) {
        g_printerr ("Error: Could not parse labels file path\n");
        g_free (str);
        goto done;
      }
      g_free (str);
    } else if (paramKey == "subscribe-topic-list") {
      gchar **topicList;
      std::string temp = itr->second.as<std::string>();
      std::vector<std::string> vec = split_string (temp);
      int length = (int) vec.size();
      topicList = g_new (gchar *, length + 1);

      for (int i = 0; i < length; i++) {
        char* str2 = (char*) malloc(sizeof(char) * _MAX_STR_LENGTH);
        std::strncpy (str2, vec[i].c_str(), _MAX_STR_LENGTH - 1);
        topicList[i] = str2;
      }
      topicList[length] = NULL;

      if (length < 1) {
        NVGSTDS_ERR_MSG_V ("%s at least one topic must be provided", __func__);
        g_strfreev (topicList);
        goto done;
      }
      if (config->topicList)
        g_ptr_array_unref (config->topicList);

      config->topicList = g_ptr_array_new_full (length, g_free);
      for (int i = 0; i < length; i++) {
        g_ptr_array_add (config->topicList, g_strdup (topicList[i]));
      }
      g_strfreev (topicList);
    } else {
      cout << "[WARNING] Unknown param found in msgconsumer: " << paramKey << endl;
    }
  }

  ret = TRUE;
  done:
  if (!ret) {
    cout <<  __func__ << " failed" << endl;
  }
  return ret;
}
