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

/** Function to set properties at nvmsgbroker element using YAML config file.*/
NvDsYamlParserStatus
nvds_parse_msgbroker (GstElement *element, gchar *cfg_file_path, const char* group)
{
  NvDsYamlParserStatus ret = NVDS_YAML_PARSER_SUCCESS;

  GstElementFactory *factory = GST_ELEMENT_GET_CLASS(element)->elementfactory;

  if (g_strcmp0(GST_OBJECT_NAME(factory), "nvmsgbroker")) {
    std::cerr << "[ERROR] Passed element is not nvmsgbroker" << std::endl;
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
        char *broker_config_file_path = (char*) malloc(sizeof(char) * 1024);
        if (!get_absolute_file_path_yaml (cfg_file_path, str,
                broker_config_file_path)) {
          g_printerr ("Error: Could not parse msgbroker config in sink.\n");
          g_free (str);
          g_free(broker_config_file_path);
          ret = NVDS_YAML_PARSER_ERROR ;
          return ret;
        }
        g_object_set(G_OBJECT(element), "config",
                   broker_config_file_path, NULL);
        g_free (str);
        g_free(broker_config_file_path);
      }
      else if (paramKey == "conn-str") {
        std::string temp = itr->second.as<std::string>();
        char *conn_str = (char*) malloc(sizeof(char) * 1024);
        std::strcpy (conn_str, temp.c_str());
        g_object_set(G_OBJECT(element), "conn-str",
                   conn_str, NULL);
        g_free(conn_str);
      }
      else if (paramKey == "proto-lib") {
        std::string temp = itr->second.as<std::string>();
        char *proto_lib = (char*) malloc(sizeof(char) * 1024);
        std::strcpy (proto_lib, temp.c_str());
        g_object_set(G_OBJECT(element), "proto-lib",
                   proto_lib, NULL);
        g_free(proto_lib);
      }
      else if (paramKey == "comp-id") {
        g_object_set(G_OBJECT(element), "comp-id",
                   itr->second.as<guint>(), NULL);
      }
      else if (paramKey == "topic") {
        std::string temp = itr->second.as<std::string>();
        char *topic = (char*) malloc(sizeof(char) * 1024);
        std::strcpy (topic, temp.c_str());
        g_object_set(G_OBJECT(element), "topic",
                   topic, NULL);
        g_free(topic);
      }
      else if (paramKey == "subscribe-topic-list") {
        std::string temp = itr->second.as<std::string>();
        char *topic_list = (char*) malloc(sizeof(char) * 1024);
        std::strcpy (topic_list, temp.c_str());
        g_object_set(G_OBJECT(element), "subscribe-topic-list",
                   topic_list, NULL);
        g_free(topic_list);
      }
      else if (paramKey == "new-api") {
        g_object_set(G_OBJECT(element), "new-api",
                   itr->second.as<gboolean>(), NULL);
      }
      else if (paramKey == "sync") {
        g_object_set(G_OBJECT(element), "sync",
                   itr->second.as<gboolean>(), NULL);
      }
      else if (paramKey == "async") {
        g_object_set(G_OBJECT(element), "async",
                   itr->second.as<gboolean>(), NULL);
      }
      else {
        std::cerr << "!! [WARNING] Unknown param found for msgbroker: " << paramKey << std::endl;
      }
    }
  }

  return ret;
}