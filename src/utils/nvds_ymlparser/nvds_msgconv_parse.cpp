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
#include "nvds_common_parser.h"

#include <yaml-cpp/yaml.h>
#include <string>
#include <iostream>
#include <unordered_map>

/** Function to set properties at nvmsgconv element using YAML config file.*/
NvDsYamlParserStatus
nvds_parse_msgconv (GstElement *element, gchar *cfg_file_path, const char* group)
{
  NvDsYamlParserStatus ret = NVDS_YAML_PARSER_SUCCESS;

  GstElementFactory *factory = GST_ELEMENT_GET_CLASS(element)->elementfactory;

  if (g_strcmp0(GST_OBJECT_NAME(factory), "nvmsgconv")) {
    std::cerr << "[ERROR] Passed element is not nvmsgconv" << std::endl;
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
      else if (paramKey == "config") {
        std::string temp = itr->second.as<std::string>();
        char* str = (char*) malloc(sizeof(char) * 1024);
        std::strcpy (str, temp.c_str());
        char *config_file_path = (char*) malloc(sizeof(char) * 1024);
        if (!get_absolute_file_path_yaml (cfg_file_path, str,
                config_file_path)) {
          g_printerr ("Error: Could not parse config in %s.\n", group);
          g_free (str);
          g_free(config_file_path);
          ret = NVDS_YAML_PARSER_ERROR;
          return ret;
        }
        g_object_set(G_OBJECT(element), "config",
                   config_file_path, NULL);
        g_free (str);
        g_free(config_file_path);
      }
      else if (paramKey == "msg2p-lib") {
        std::string temp = itr->second.as<std::string>();
        char* str = (char*) malloc(sizeof(char) * 1024);
        std::strcpy (str, temp.c_str());
        char *conv_msg2p_lib = (char*) malloc(sizeof(char) * 1024);
        if (!get_absolute_file_path_yaml (cfg_file_path, str,
                conv_msg2p_lib)) {
          g_printerr ("Error: Could not parse msg2p-lib in %s.\n", group);
          g_free (str);
          g_free (conv_msg2p_lib);
          ret = NVDS_YAML_PARSER_ERROR;
          return ret;
        }
        g_object_set(G_OBJECT(element), "msg2p-lib",
                   conv_msg2p_lib, NULL);
        g_free (str);
        g_free(conv_msg2p_lib);
      }
      else if (paramKey == "payload-type") {
        g_object_set(G_OBJECT(element), "payload-type",
                   itr->second.as<guint>(), NULL);
      }
      else if (paramKey == "comp-id") {
        g_object_set(G_OBJECT(element), "comp-id",
                   itr->second.as<guint>(), NULL);
      }
      else if (paramKey == "debug-payload-dir") {
        std::string temp = itr->second.as<std::string>();
        char* str = (char*) malloc(sizeof(char) * 1024);
        std::strcpy (str, temp.c_str());
        char *debug_payload_dir = (char*) malloc(sizeof(char) * 1024);
        if (!get_absolute_file_path_yaml (cfg_file_path, str,
                debug_payload_dir)) {
          g_printerr ("Error: Could not parse debug-payload-dir in %s.\n", group);
          g_free (str);
          g_free(debug_payload_dir);
          ret = NVDS_YAML_PARSER_ERROR;
          return ret;
        }
        g_object_set(G_OBJECT(element), "debug-payload-dir",
                   debug_payload_dir, NULL);
        g_free (str);
        g_free(debug_payload_dir);
      }
      else if (paramKey == "multiple-payloads") {
        g_object_set(G_OBJECT(element), "multiple-payloads",
                   itr->second.as<gboolean>(), NULL);
      }
      else if (paramKey == "msg2p-newapi") {
        g_object_set(G_OBJECT(element), "msg2p-newapi",
                   itr->second.as<gboolean>(), NULL);
      }
      else if (paramKey == "frame-interval") {
        g_object_set(G_OBJECT(element), "frame-interval",
                   itr->second.as<guint>(), NULL);
      }
      else {
        std::cerr << "!! [WARNING] Unknown param found for msgconverter: " << paramKey << std::endl;
      }
    }
  }

  return ret;
}