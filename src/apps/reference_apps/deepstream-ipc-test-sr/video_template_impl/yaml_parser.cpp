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

#include "yaml_parser.h"
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <string>
#include <assert.h>
#include "cuda_runtime_api.h"
 #include <cstring>

using std::endl;
using std::cout;
 
static gboolean
gst_parse_props_yaml (const gchar * cfg_file_path, cfg_params& cfg_params)
{
  gboolean ret = FALSE;
  YAML::Node configyml = YAML::LoadFile(cfg_file_path);
  if(!(configyml.size() > 0))  {
  	cout << "Can't open config file (" << cfg_file_path << ")" << endl;
  }
  for(YAML::const_iterator itr = configyml["property"].begin(); itr != configyml["property"].end(); ++itr)
  {
    std::string paramKey = itr->first.as<std::string>();
    if (paramKey == "width") {
        cfg_params.m_tensor_width = itr->second.as<unsigned int>();
    } else if (paramKey == "height") {
        cfg_params.m_tensor_height = itr->second.as<unsigned int>();
    } else {
      std::string paramVal = itr->second.as<std::string>();
      printf("not need %s\n", paramVal.c_str());
    }
  }

  ret = TRUE;
done:
  return ret;
}

/* Parse nvinfer config file for context params. Returns FALSE in case of an error. */
gboolean
gst_parse_context_params_yaml (const gchar * cfg_file_path, cfg_params& cfg_params)
{
  gboolean ret = FALSE;

  YAML::Node configyml = YAML::LoadFile(cfg_file_path);
  if(!(configyml.size() > 0))  {
  	cout << "Can't open config file (" << cfg_file_path << ")" << endl;
  }
  /* 'property' group is mandatory. */
  if(configyml["property"]) {
    if (!gst_parse_props_yaml (cfg_file_path, cfg_params)) {
      g_printerr ("Failed to parse group property\n");
      goto done;
    }
  }
  else  {
    g_printerr ("Could not find group property\n");
    goto done;
  }
  ret = TRUE;

done:
  if (!ret) {
    g_printerr ("** ERROR: <%s:%d>: failed\n", __func__, __LINE__);
  }
  return ret;
}
