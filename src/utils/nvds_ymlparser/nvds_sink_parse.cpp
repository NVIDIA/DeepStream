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
#include <cstring>
#include <iostream>
#include <unordered_map>

/** Function to set properties at nveglglessink element using YAML config file.*/
NvDsYamlParserStatus
nvds_parse_egl_sink(GstElement *element, gchar *cfg_file_path, const char* group)
{
  NvDsYamlParserStatus ret = NVDS_YAML_PARSER_SUCCESS;

  GstElementFactory *factory = GST_ELEMENT_GET_CLASS(element)->elementfactory;

  if (g_strcmp0(GST_OBJECT_NAME(factory), "nveglglessink")) {
    std::cerr << "[ERROR] Passed element is not nveglglessink" << std::endl;
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
          std::cerr << "!! [WARNING]  \"sink\" group not enabled. Use \"enable: 1\" to set properties." << std::endl;
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
      else if(paramKey == "async") {
        g_object_set(G_OBJECT(element), "async",
                     itr->second.as<gboolean>(), NULL);
      }
      else if(paramKey == "columns") {
        g_object_set(G_OBJECT(element), "columns",
                     itr->second.as<guint>(), NULL);
      }
      else if(paramKey == "create-window") {
        g_object_set(G_OBJECT(element), "create-window",
                     itr->second.as<gboolean>(), NULL);
      }
      else if(paramKey == "enable-last-sample") {
        g_object_set(G_OBJECT(element), "enable-last-sample",
                     itr->second.as<gboolean>(), NULL);
      }
      else if(paramKey == "force-aspect-ratio") {
        g_object_set(G_OBJECT(element), "force-aspect-ratio",
                     itr->second.as<gboolean>(), NULL);
      }
      else if(paramKey == "gpu-id") {
        g_object_set(G_OBJECT(element), "gpu-id",
                     itr->second.as<guint>(), NULL);
      }
      else if(paramKey == "max-bitrate") {
        g_object_set(G_OBJECT(element), "max-bitrate",
                     itr->second.as<guint64>(), NULL);
      }
      else if(paramKey == "max-lateness") {
        g_object_set(G_OBJECT(element), "max-lateness",
                     itr->second.as<gint64>(), NULL);
      }
      else if(paramKey == "processing-deadline") {
        g_object_set(G_OBJECT(element), "processing-deadline",
                     itr->second.as<guint64>(), NULL);
      }
      else if(paramKey == "profile") {
        g_object_set(G_OBJECT(element), "profile",
                     itr->second.as<guint>(), NULL);
      }
      else if(paramKey == "qos") {
        g_object_set(G_OBJECT(element), "qos",
                     itr->second.as<gboolean>(), NULL);
      }
      else if(paramKey == "rows") {
        g_object_set(G_OBJECT(element), "rows",
                     itr->second.as<guint>(), NULL);
      }
      else if(paramKey == "sync") {
        g_object_set(G_OBJECT(element), "sync",
                     itr->second.as<gboolean>(), NULL);
      }
      else if(paramKey == "window-height") {
        g_object_set(G_OBJECT(element), "window-height",
                     itr->second.as<guint>(), NULL);
      }
      else if(paramKey == "window-width") {
        g_object_set(G_OBJECT(element), "window-width",
                     itr->second.as<guint>(), NULL);
      }
      else if(paramKey == "window-x") {
        g_object_set(G_OBJECT(element), "window-x",
                     itr->second.as<guint>(), NULL);
      }
      else if(paramKey == "window-y") {
        g_object_set(G_OBJECT(element), "window-y",
                     itr->second.as<guint>(), NULL);
      }
      else {
        std::cerr << "!! [WARNING] Unknown param found for nveglglessink : " << paramKey << std::endl;
      }
    }
  }

  return ret;
}

/** Function to set properties at nv3dsink element using YAML config file.*/
NvDsYamlParserStatus
nvds_parse_3d_sink(GstElement *element, gchar *cfg_file_path, const char* group)
{
  NvDsYamlParserStatus ret = NVDS_YAML_PARSER_SUCCESS;

  GstElementFactory *factory = GST_ELEMENT_GET_CLASS(element)->elementfactory;

  if (g_strcmp0(GST_OBJECT_NAME(factory), "nv3dsink")) {
    std::cerr << "[ERROR] Passed element is not nv3dsink" << std::endl;
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
          std::cerr << "!! [WARNING]  \"sink\" group not enabled. Use \"enable: 1\" to set properties." << std::endl;
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
      else if(paramKey == "async") {
        g_object_set(G_OBJECT(element), "async",
                     itr->second.as<gboolean>(), NULL);
      }
      else if(paramKey == "enable-last-sample") {
        g_object_set(G_OBJECT(element), "enable-last-sample",
                     itr->second.as<gboolean>(), NULL);
      }
      else if(paramKey == "max-bitrate") {
        g_object_set(G_OBJECT(element), "max-bitrate",
                     itr->second.as<guint64>(), NULL);
      }
      else if(paramKey == "max-lateness") {
        g_object_set(G_OBJECT(element), "max-lateness",
                     itr->second.as<gint64>(), NULL);
      }
      else if(paramKey == "processing-deadline") {
        g_object_set(G_OBJECT(element), "processing-deadline",
                     itr->second.as<guint64>(), NULL);
      }
      else if(paramKey == "qos") {
        g_object_set(G_OBJECT(element), "qos",
                     itr->second.as<gboolean>(), NULL);
      }
      else if(paramKey == "sync") {
        g_object_set(G_OBJECT(element), "sync",
                     itr->second.as<gboolean>(), NULL);
      }
      else if(paramKey == "window-height") {
        g_object_set(G_OBJECT(element), "window-height",
                     itr->second.as<guint>(), NULL);
      }
      else if(paramKey == "window-width") {
        g_object_set(G_OBJECT(element), "window-width",
                     itr->second.as<guint>(), NULL);
      }
      else if(paramKey == "window-x") {
        g_object_set(G_OBJECT(element), "window-x",
                     itr->second.as<guint>(), NULL);
      }
      else if(paramKey == "window-y") {
        g_object_set(G_OBJECT(element), "window-y",
                     itr->second.as<guint>(), NULL);
      }
      else {
        std::cerr << "!! [WARNING] Unknown param found for nv3dsink : " << paramKey << std::endl;
      }
    }
  }

  return ret;
}

/** Function to set properties at filesink element using YAML config file.*/
NvDsYamlParserStatus
nvds_parse_file_sink(GstElement *element, gchar *cfg_file_path, const char* group)
{
  NvDsYamlParserStatus ret = NVDS_YAML_PARSER_SUCCESS;

  GstElementFactory *factory = GST_ELEMENT_GET_CLASS(element)->elementfactory;

  if (g_strcmp0(GST_OBJECT_NAME(factory), "filesink")) {
    std::cerr << "[ERROR] Passed element is not filesink" << std::endl;
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
          std::cerr << "!! [WARNING]  \"sink\" group not enabled. Use \"enable: 1\" to set properties." << std::endl;
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
      else if(paramKey == "async") {
        g_object_set(G_OBJECT(element), "async",
                     itr->second.as<gboolean>(), NULL);
      }
      else if(paramKey == "enable-last-sample") {
        g_object_set(G_OBJECT(element), "enable-last-sample",
                     itr->second.as<gboolean>(), NULL);
      }
      else if(paramKey == "location") {

        std::string temp = itr->second.as<std::string>();
        gchar *str = (char *) temp.c_str();
        g_object_set(G_OBJECT(element), "location",
                     str, NULL);
      }
      else if(paramKey == "qos") {
        g_object_set(G_OBJECT(element), "qos",
                     itr->second.as<gboolean>(), NULL);
      }
      else if(paramKey == "sync") {
        g_object_set(G_OBJECT(element), "sync",
                     itr->second.as<gboolean>(), NULL);
      }
      else {
        std::cerr << "!! [WARNING] Unknown param found for filesink : " << paramKey << std::endl;
      }
    }
  }

  return ret;
}

/** Function to set properties at fakesink element using YAML config file.*/
NvDsYamlParserStatus
nvds_parse_fake_sink(GstElement *element, gchar *cfg_file_path, const char* group)
{
  NvDsYamlParserStatus ret = NVDS_YAML_PARSER_SUCCESS;

  GstElementFactory *factory = GST_ELEMENT_GET_CLASS(element)->elementfactory;

  if (g_strcmp0(GST_OBJECT_NAME(factory), "fakesink")) {
    std::cerr << "[ERROR] Passed element is not fakesink : " << group << std::endl;
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
          std::cerr << "!! [WARNING]  \"sink\" group not enabled. Use \"enable: 1\" to set properties." << std::endl;
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
      else if(paramKey == "async") {
        g_object_set(G_OBJECT(element), "async",
                     itr->second.as<gboolean>(), NULL);
      }
      else if(paramKey == "enable-last-sample") {
        g_object_set(G_OBJECT(element), "enable-last-sample",
                     itr->second.as<gboolean>(), NULL);
      }
      else if(paramKey == "qos") {
        g_object_set(G_OBJECT(element), "qos",
                     itr->second.as<gboolean>(), NULL);
      }
      else if(paramKey == "sync") {
        g_object_set(G_OBJECT(element), "sync",
                     itr->second.as<gboolean>(), NULL);
      }
      else {
        std::cerr << "!! [WARNING] Unknown param found for fakesink : " << paramKey << std::endl;
      }
    }
  }

  return ret;
}
