/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "nvds_yml_parser.h"

#include <yaml-cpp/yaml.h>
#include <string>
#include <iostream>
#include <unordered_map>

/** Function to set properties at nvdsanalytics element using YAML config file.*/
NvDsYamlParserStatus
nvds_parse_nvdsanalytics(GstElement *element, gchar *cfg_file_path, const char* group)
{
  NvDsYamlParserStatus ret = NVDS_YAML_PARSER_SUCCESS;

  GstElementFactory *factory = GST_ELEMENT_GET_CLASS(element)->elementfactory;

  if (g_strcmp0(GST_OBJECT_NAME(factory), "nvdsanalytics")) {
    std::cerr << "[ERROR] Passed element is not nvdsanalytics" << std::endl;
    ret = NVDS_YAML_PARSER_ERROR;
    return ret;
  }

  std::string paramKey = "";

  auto docs = YAML::LoadAllFromFile(cfg_file_path);

  std::vector<int> docs_indx_vec;
  std::unordered_map<std::string, int> docs_indx_umap;

  int total_docs = docs.size();

  for (int i =0; i < total_docs;i++)
  {
    if (docs[i][group].Type() != YAML::NodeType::Null) {

      if (docs[i][group]["enable"]) {
        gboolean val= docs[i][group]["enable"].as<gboolean>();
        if(val == FALSE) {
          std::cerr << "!! [WARNING]  \"analytics\" group not enabled. Use \"enable: 1\" to set properties." << std::endl;
          ret = NVDS_YAML_PARSER_DISABLED;
          return ret;
        }
      }

      YAML::const_iterator itr = docs[i].begin();
      std::string group_name = itr->first.as<std::string>();
      docs_indx_umap[group_name] = i;
      docs_indx_vec.push_back(i);

    }
  }

  int docs_indx_vec_size = docs_indx_vec.size();

  int docs_indx_umap_size = docs_indx_umap.size();

  if (docs_indx_umap_size != docs_indx_vec_size) {
    std::cerr << "[ERROR] Duplicate group names in the config file : " << group << std::endl;
    ret = NVDS_YAML_PARSER_ERROR;
    return ret;
  }

  for (int i = 0; i< docs_indx_vec_size; i++)
  {
    int indx = docs_indx_vec [i];
    for(YAML::const_iterator itr = docs[indx][group].begin(); itr != docs[indx][group].end(); ++itr)
    {
      paramKey = itr->first.as<std::string>();

      if (paramKey == "enable" && docs[indx][group]["enable"].as<gboolean>() == TRUE) {
        continue;
      }
      else if(paramKey == "config-file") {

        std::string temp = itr->second.as<std::string>();
        gchar *str = (char *) temp.c_str();
        g_object_set(G_OBJECT(element), "config-file",
                     str, NULL);
      }
      else {
        std::cerr << "!! [WARNING] Unknown param found for nvdsanalytics: " << paramKey << std::endl;
      }
    }
  }

  return ret;
}

/** Function to check if the rest server is enabled in the YAML config file.*/
gboolean
check_enable_status (gchar * cfg_file_path, const char* group)
{
  std::vector<int> docs_indx_vec;
  std::unordered_map<std::string, int> docs_indx_umap;
  auto docs = YAML::LoadAllFromFile(cfg_file_path);

  int total_docs = docs.size();
  for (int i =0; i < total_docs;i++)
  {
    if (docs[i][group].Type() != YAML::NodeType::Null) {

      if (docs[i][group]["enable"]) {
        gboolean val= docs[i][group]["enable"].as<gboolean>();
        if(val == FALSE) {
          return FALSE;
        }
      }

      YAML::const_iterator itr = docs[i].begin();
      std::string group_name = itr->first.as<std::string>();
      docs_indx_umap[group_name] = i;
      docs_indx_vec.push_back(i);

    }
  }
  return TRUE;
}
