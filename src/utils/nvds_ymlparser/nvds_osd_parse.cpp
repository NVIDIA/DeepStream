/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/** Function to set properties at nvdsosd element using YAML config file.*/
NvDsYamlParserStatus
nvds_parse_osd(GstElement *element, gchar *cfg_file_path, const char* group)
{
  NvDsYamlParserStatus ret = NVDS_YAML_PARSER_SUCCESS;

  GstElementFactory *factory = GST_ELEMENT_GET_CLASS(element)->elementfactory;

  if (g_strcmp0(GST_OBJECT_NAME(factory), "nvdsosd")) {
    std::cerr << "[ERROR] Passed element is not  nvdsosd" << std::endl;
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
      else if(paramKey == "gpu-id") {
        g_object_set(G_OBJECT(element), "gpu-id",
                   itr->second.as<guint>(), NULL);
      }
      else if(paramKey == "display-clock") {
        g_object_set(G_OBJECT(element), "display-clock",
                   itr->second.as<gboolean>(), NULL);
      }
      else if(paramKey == "display-text") {
        g_object_set(G_OBJECT(element), "display-text",
                   itr->second.as<gboolean>(), NULL);
      }
      else if(paramKey == "clock-font") {
        g_object_set(G_OBJECT(element), "clock-font",
                   itr->second.as<std::string>(), NULL);
      }
      else if(paramKey == "clock-font-size") {
        g_object_set(G_OBJECT(element), "clock-font-size",
                   itr->second.as<guint>(), NULL);
      }
      else if(paramKey == "x-clock-offset") {
        g_object_set(G_OBJECT(element), "x-clock-offset",
                   itr->second.as<guint>(), NULL);
      }
      else if(paramKey == "y-clock-offset") {
        g_object_set(G_OBJECT(element), "y-clock-offset",
                   itr->second.as<guint>(), NULL);
      }
      else if(paramKey == "clock-color") {
        g_object_set(G_OBJECT(element), "clock-color",
                   itr->second.as<guint>(), NULL);
      }
      else if(paramKey == "process-mode") {
        g_object_set(G_OBJECT(element), "process-mode",
                   itr->second.as<guint>(), NULL);
      }
      else if(paramKey == "hw-blend-color-attr") {
        g_object_set(G_OBJECT(element), "hw-blend-color-attr",
                   itr->second.as<std::string>(), NULL);
      }
      else if(paramKey == "display-bbox") {
        g_object_set(G_OBJECT(element), "display-bbox",
                   itr->second.as<gboolean>(), NULL);
      }
      else if(paramKey == "display-mask") {
        g_object_set(G_OBJECT(element), "display-mask",
                   itr->second.as<gboolean>(), NULL);
      }
      else if(paramKey == "blur-bbox") {
        g_object_set(G_OBJECT(element), "blur-bbox",
                   itr->second.as<gboolean>(), NULL);
      }
      else if(paramKey == "blur-on-gie-class-ids") {
        g_object_set(G_OBJECT(element), "blur-on-gie-class-ids",
                   itr->second.as<std::string>().c_str(), NULL);
      }
      else {
        std::cerr << "!! [WARNING] Unknown param found for nvdsosd: " << paramKey << std::endl;
      }
    }
  }

  return ret;
}
