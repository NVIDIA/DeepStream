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


/** Function to set properties of nvinfer element using YAML node.*/
static NvDsYamlParserStatus
nvds_set_nvinfer_properties_from_yaml(GstElement* element, YAML::Node node,
    gboolean* config_file_set, gchar* app_cfg_file_path, const char* group)
{
  std::string paramKey = "";

  for(YAML::const_iterator itr = node.begin(); itr != node.end(); ++itr)
  {
    paramKey = itr->first.as<std::string>();

    if (paramKey == "enable" && node["enable"].as<gboolean>() == TRUE) {
      continue;
    }
    else if (paramKey == "plugin-type" &&
        node["plugin-type"].as<guint>() == NVDS_GIE_PLUGIN_INFER) {
      continue;
    }
    else if (paramKey == "config-file-path") {
      std::string temp = itr->second.as<std::string>();
      char* str = (char*) malloc(sizeof(char) * 1024);
      std::strcpy (str, temp.c_str());
      char *config_file_path = (char*) malloc(sizeof(char) * 1024);
      if (!get_absolute_file_path_yaml (app_cfg_file_path, str,
            config_file_path)) {
        g_printerr ("Error: Could not parse config-file-path in %s.\n", group);
        g_free (str);
        g_free(config_file_path);
        return NVDS_YAML_PARSER_ERROR;
      }
      g_object_set(G_OBJECT(element), "config-file-path",
          config_file_path, NULL);
      *config_file_set = TRUE;
      g_free (str);
      g_free(config_file_path);
    }
    else if (paramKey == "process-mode") {
      g_object_set(G_OBJECT(element), "process-mode",
          itr->second.as<guint>(), NULL);
    }
    else if (paramKey == "unique-id") {
      g_object_set(G_OBJECT(element), "unique-id",
          itr->second.as<guint>(), NULL);
    }
    else if (paramKey == "infer-on-gie-id") {
      g_object_set(G_OBJECT(element), "infer-on-gie-id",
          itr->second.as<gint>(), NULL);
    }
    else if (paramKey == "operate-on-class-ids") {
      g_object_set(G_OBJECT(element), "operate-on-class-ids",
          itr->second.as<std::string>(), NULL);
    }
    else if (paramKey == "filter-out-class-ids") {
      g_object_set(G_OBJECT(element), "filter-out-class-ids",
          itr->second.as<std::string>(), NULL);
    }
    else if (paramKey == "infer-on-class-ids") {
      g_object_set(G_OBJECT(element), "infer-on-class-ids",
          itr->second.as<std::string>(), NULL);
    }
    else if (paramKey == "model-engine-file") {
      std::string temp = itr->second.as<std::string>();
      char* str = (char*) malloc(sizeof(char) * 1024);
      std::strcpy (str, temp.c_str());
      char *model_engine_file_path = (char*) malloc(sizeof(char) * 1024);
      if (!get_absolute_file_path_yaml (app_cfg_file_path, str,
            model_engine_file_path)) {
        g_printerr ("Error: Could not parse model-engine-file in %s.\n", group);
        g_free (str);
        g_free(model_engine_file_path);
        return  NVDS_YAML_PARSER_ERROR;
      }
      g_object_set(G_OBJECT(element), "model-engine-file",
          model_engine_file_path, NULL);
      g_free (str);
      g_free(model_engine_file_path);
    }
    else if (paramKey == "batch-size") {
      g_object_set(G_OBJECT(element), "batch-size",
          itr->second.as<guint>(), NULL);
    }
    else if (paramKey == "interval") {
      g_object_set(G_OBJECT(element), "interval",
          itr->second.as<guint>(), NULL);
    }
    else if (paramKey == "gpu-id") {
      g_object_set(G_OBJECT(element), "gpu-id",
          itr->second.as<guint>(), NULL);
    }
    else if (paramKey == "raw-output-file-write") {
      g_object_set(G_OBJECT(element), "raw-output-file-write",
          itr->second.as<gboolean>(), NULL);
    }
    else if (paramKey == "output-tensor-meta") {
      g_object_set(G_OBJECT(element), "output-tensor-meta",
          itr->second.as<gboolean>(), NULL);
    }
    else if (paramKey == "output-instance-mask") {
      g_object_set(G_OBJECT(element), "output-instance-mask",
          itr->second.as<gboolean>(), NULL);
    }
    else if (paramKey == "input-tensor-meta") {
      g_object_set(G_OBJECT(element), "input-tensor-meta",
          itr->second.as<gboolean>(), NULL);
    }
    else {
      std::cerr << "[WARNING] Unknown param found in gie: " << paramKey << std::endl;
    }
  }
  return NVDS_YAML_PARSER_SUCCESS;
}

/** Function to set properties of nvinferserver element using YAML node.*/
static NvDsYamlParserStatus
nvds_set_nvinferserver_properties_from_yaml(GstElement* element, YAML::Node node,
    gboolean* config_file_set, gchar* app_cfg_file_path, const char* group)
{
  std::string paramKey = "";

  for(YAML::const_iterator itr = node.begin(); itr != node.end(); ++itr)
  {
    paramKey = itr->first.as<std::string>();

    if (paramKey == "enable" && node["enable"].as<gboolean>() == TRUE) {
      continue;
    }
    else if (paramKey == "plugin-type" &&
        node["plugin-type"].as<guint>() == NVDS_GIE_PLUGIN_INFER_SERVER) {
      continue;
    }
    else if (paramKey == "config-file-path") {
      std::string temp = itr->second.as<std::string>();
      char* str = (char*) malloc(sizeof(char) * 1024);
      std::strcpy (str, temp.c_str());
      char *config_file_path = (char*) malloc(sizeof(char) * 1024);
      if (!get_absolute_file_path_yaml (app_cfg_file_path, str,
            config_file_path)) {
        g_printerr ("Error: Could not parse config-file-path in %s.\n", group);
        g_free (str);
        g_free(config_file_path);
        return NVDS_YAML_PARSER_ERROR;
      }
      g_object_set(G_OBJECT(element), "config-file-path",
          config_file_path, NULL);
      *config_file_set = TRUE;
      g_free (str);
      g_free(config_file_path);
    }
    else if (paramKey == "process-mode") {
      g_object_set(G_OBJECT(element), "process-mode",
          itr->second.as<guint>(), NULL);
    }
    else if (paramKey == "unique-id") {
      g_object_set(G_OBJECT(element), "unique-id",
          itr->second.as<guint>(), NULL);
    }
    else if (paramKey == "infer-on-gie-id") {
      g_object_set(G_OBJECT(element), "infer-on-gie-id",
          itr->second.as<gint>(), NULL);
    }
    else if (paramKey == "operate-on-class-ids") {
      g_object_set(G_OBJECT(element), "operate-on-class-ids",
          itr->second.as<std::string>(), NULL);
    }
    else if (paramKey == "batch-size") {
      g_object_set(G_OBJECT(element), "batch-size",
          itr->second.as<guint>(), NULL);
    }
    else if (paramKey == "interval") {
      g_object_set(G_OBJECT(element), "interval",
          itr->second.as<guint>(), NULL);
    }
    else {
      std::cerr << "[WARNING] Unknown param found in gie: " << paramKey << std::endl;
    }
  }
  return NVDS_YAML_PARSER_SUCCESS;
}

/** Function to set properties at nvinfer element using YAML config file.*/
NvDsYamlParserStatus
nvds_parse_gie (GstElement *element, gchar *cfg_file_path, const char* group)
{
  NvDsYamlParserStatus ret = NVDS_YAML_PARSER_SUCCESS;
  gboolean config_file_set = FALSE;

  GstElementFactory *factory = GST_ELEMENT_GET_CLASS(element)->elementfactory;

  if (!(!g_strcmp0(GST_OBJECT_NAME(factory), "nvinfer") ||
       !g_strcmp0(GST_OBJECT_NAME(factory), "nvinferserver")   )) {
    std::cerr << "[ERROR] Passed element is not nvinfer or nvinfersever" << std::endl;
    return NVDS_YAML_PARSER_ERROR;
  }

  if (!cfg_file_path) {
    std::cout << "Config file not provided for group " << group << std::endl;
    return NVDS_YAML_PARSER_ERROR;
  }

  auto docs = YAML::LoadAllFromFile(cfg_file_path);

  std::vector<int> docs_indx_vec;
  std::unordered_map<std::string, int> docs_indx_umap;

  int total_docs = docs.size();

  for (int i = 0; i < total_docs; i++)
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

    if (!g_strcmp0(GST_OBJECT_NAME(factory), "nvinfer")) {
      ret = nvds_set_nvinfer_properties_from_yaml(element, docs[indx][group],
              &config_file_set, cfg_file_path, group);
    } else {
      ret = nvds_set_nvinferserver_properties_from_yaml(element, docs[indx][group],
              &config_file_set, cfg_file_path, group);
    }

    if (ret != NVDS_YAML_PARSER_SUCCESS) {
      return ret;
    }
  }

  if (!config_file_set) {
    std::cout << "Config file not provided for group " << group << std::endl;
    ret = NVDS_YAML_PARSER_ERROR;
    return ret;
  }

  return ret;
}

NvDsYamlParserStatus
nvds_parse_gie_type (NvDsGieType* gie_type, gchar* cfg_file_path, const char* group)
{
  NvDsYamlParserStatus ret = NVDS_YAML_PARSER_SUCCESS;

  if (!cfg_file_path) {
    std::cout << " Config file not provided." << std::endl;
    return NVDS_YAML_PARSER_ERROR;
  }

  std::string paramKey = "";

  auto docs = YAML::LoadAllFromFile(cfg_file_path);

  std::vector<int> docs_indx_vec;
  std::unordered_map<std::string, int> docs_indx_umap;

  int total_docs = docs.size();

  for (int i = 0; i < total_docs; i++)
  {
    if (docs[i][group].Type() != YAML::NodeType::Null) {
      if (docs[i][group]["enable"]) {
        gboolean val= docs[i][group]["enable"].as<gboolean>();
        if(val == FALSE) {
          std::cerr << "!! [WARNING]  \"infer\" group not enabled. Use \"enable: 1\" to set properties." << std::endl;
          return NVDS_YAML_PARSER_DISABLED;
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
    return NVDS_YAML_PARSER_ERROR;
  }

  for (int i = 0; i< docs_indx_vec_size; i++)
  {
    int indx = docs_indx_vec [i];
    for(YAML::const_iterator itr = docs[indx][group].begin(); itr != docs[indx][group].end(); ++itr) {
      paramKey = itr->first.as<std::string>();
      if (paramKey == "plugin-type") {
        guint infer_type = itr->second.as<guint>();
        if ((NVDS_GIE_PLUGIN_INFER != infer_type) &&
            (NVDS_GIE_PLUGIN_INFER_SERVER != infer_type)) {
          std::cerr << "[ERROR] Incorrect GIE type in the config file: " << group << std::endl;
          return NVDS_YAML_PARSER_ERROR;
        } else {
          *gie_type = static_cast<NvDsGieType>(infer_type);
        }
      }
    }
  }

  return ret;
}
