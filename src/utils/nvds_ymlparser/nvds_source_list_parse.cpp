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

#include "nvds_yml_parser.h"

#include <yaml-cpp/yaml.h>
#include <string>
#include <iostream>
#include <unordered_map>

/** Function to store the uri(s) in the source list using YAML config file.*/
NvDsYamlParserStatus
nvds_parse_source_list(GList **src_list, gchar *cfg_file_path, const char* group)
{
  NvDsYamlParserStatus ret = NVDS_YAML_PARSER_SUCCESS;
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
          std::cerr << "!! [WARNING]  \"infer\" group not enabled. Use \"enable: 1\" to set properties." << std::endl;
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
      else if(paramKey == "list") {

        std::string temp = itr->second.as<std::string>();
        std::vector<std::string> token_vec;
        std::string token;
        std::stringstream ss(temp);

        while(std::getline(ss, token, ';')){
          gchar *str = (char *) token.c_str();
          token_vec.push_back(str);
        }
        int  vec_size = token_vec.size();
        for (int i = 0; i < vec_size; ++i)
        {
          *src_list = g_list_append(*src_list,g_strdup(token_vec[i].c_str()));
        }

        token_vec.clear();

      }
      else {
        std::cerr << "!! [WARNING] Unknown param found for source list: " << paramKey << std::endl;
      }
    }
  }

  return ret;
}
