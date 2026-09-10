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
#include <unistd.h>
#include <unordered_map>

/** Function to set properties at filesrc element using YAML config file.*/
NvDsYamlParserStatus
nvds_parse_file_source(GstElement *element, gchar *cfg_file_path, const char* group)
{
  NvDsYamlParserStatus ret = NVDS_YAML_PARSER_SUCCESS;

  GstElementFactory *factory = GST_ELEMENT_GET_CLASS(element)->elementfactory;

  if (g_strcmp0(GST_OBJECT_NAME(factory), "filesrc")) {
    std::cerr << "[ERROR] Passed element is not filesrc" << std::endl;
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
      else if(paramKey == "location") {

        std::string temp = itr->second.as<std::string>();
        gchar *str = (char *) temp.c_str();
        g_object_set(G_OBJECT(element), "location",
                     str, NULL);
      }
      else {
        std::cerr << "!! [WARNING] Unknown param found for file source: " << paramKey << std::endl;
      }
    }
  }

  return ret;
}

/** Function to set properties at uridecodebin element using YAML config file.*/
NvDsYamlParserStatus
nvds_parse_uridecodebin(GstElement *element, gchar *cfg_file_path, const char* group)
{
  NvDsYamlParserStatus ret = NVDS_YAML_PARSER_SUCCESS;

  GstElementFactory *factory = GST_ELEMENT_GET_CLASS(element)->elementfactory;

  if (g_strcmp0(GST_OBJECT_NAME(factory), "uridecodebin")) {
    std::cerr << "[ERROR] Passed element is not uridecodebin" << std::endl;
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
      else if(paramKey == "async-handling") {
        g_object_set(G_OBJECT(element), "async-handling",
                     itr->second.as<gboolean>(), NULL);
      }
      else if (paramKey == "uri") {
        std::string temp = itr->second.as<std::string>();
        char* uri = (char*) malloc(sizeof(char) * 1024);
        std::strcpy (uri, temp.c_str());
        char *str;
        if (g_str_has_prefix (uri, "file://")) {
          str = g_strdup (uri + 7);
          char *file_uri = (char*) malloc(sizeof(char) * 1024);
          get_absolute_file_path_yaml (cfg_file_path, str, file_uri);
          char *final_uri = g_strdup_printf ("file://%s", file_uri);
          g_object_set(G_OBJECT(element), "uri",
                     final_uri, NULL);
          g_free (uri);
          g_free (str);
          g_free (file_uri);
          g_free (final_uri);
        } else {
          g_object_set(G_OBJECT(element), "uri",
                     itr->second.as<std::string>(), NULL);
          g_free(uri);
        }
      }
      else if(paramKey == "use-buffering") {
        g_object_set(G_OBJECT(element), "use-buffering",
                     itr->second.as<gboolean>(), NULL);
      }
      else {
        std::cerr << "!! [WARNING] Unknown param found for uridecodebin: " << paramKey << std::endl;
      }
    }
  }

  return ret;
}

/** Function to set properties at rtspsrc element using YAML config file.*/
NvDsYamlParserStatus
nvds_parse_rtsp_source(GstElement *element, gchar *cfg_file_path, const char* group)
{
  NvDsYamlParserStatus ret = NVDS_YAML_PARSER_SUCCESS;

  GstElementFactory *factory = GST_ELEMENT_GET_CLASS(element)->elementfactory;

  if (g_strcmp0(GST_OBJECT_NAME(factory), "rtspsrc")) {
    std::cerr << "[ERROR] Passed element is not rtspsrc" << std::endl;
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
      else if(paramKey == "async-handling") {
        g_object_set(G_OBJECT(element), "async-handling",
                     itr->second.as<gboolean>(), NULL);
      }
      else if(paramKey == "do-retransmission") {
        g_object_set(G_OBJECT(element), "do-retransmission",
                     itr->second.as<gboolean>(), NULL);
      }
      else if(paramKey == "do-rtcp") {
        g_object_set(G_OBJECT(element), "do-rtcp",
                     itr->second.as<gboolean>(), NULL);
      }
      else if(paramKey == "do-rtsp-keep-alive") {
        g_object_set(G_OBJECT(element), "do-rtsp-keep-alive",
                     itr->second.as<gboolean>(), NULL);
      }
      else if(paramKey == "drop-on-latency") {
        g_object_set(G_OBJECT(element), "drop-on-latency",
                     itr->second.as<gboolean>(), NULL);
      }
      else if(paramKey == "latency") {
        g_object_set(G_OBJECT(element), "latency",
                     itr->second.as<guint>(), NULL);
      }
      else if(paramKey == "location") {

        std::string temp = itr->second.as<std::string>();
        gchar *str = (char *) temp.c_str();
        g_object_set(G_OBJECT(element), "location",
                     str, NULL);
      }
      else if(paramKey == "max-rtcp-rtp-time-diff") {
        g_object_set(G_OBJECT(element), "max-rtcp-rtp-time-diff",
                     itr->second.as<gint>(), NULL);
      }
      else if(paramKey == "max-ts-offset") {
        g_object_set(G_OBJECT(element), "max-ts-offset",
                     itr->second.as<gint64>(), NULL);
      }
      else if(paramKey == "max-ts-offset-adjustment") {
        g_object_set(G_OBJECT(element), "max-ts-offset-adjustment",
                     itr->second.as<guint64>(), NULL);
      }
      else if(paramKey == "message-forward") {
        g_object_set(G_OBJECT(element), "message-forward",
                     itr->second.as<gboolean>(), NULL);
      }
      else if(paramKey == "multicast-iface") {
        g_object_set(G_OBJECT(element), "multicast-iface",
                     itr->second.as<std::string>(), NULL);
      }
      else if(paramKey == "nat-method") {
        g_object_set(G_OBJECT(element), "nat-method",
                     itr->second.as<gboolean>(), NULL);
      }
      else if(paramKey == "ntp-sync") {
        g_object_set(G_OBJECT(element), "ntp-sync",
                     itr->second.as<gboolean>(), NULL);
      }
      else if(paramKey == "ntp-time-source") {
        g_object_set(G_OBJECT(element), "ntp-time-source",
                     itr->second.as<gboolean>(), NULL);
      }
      else if(paramKey == "port-range") {
        g_object_set(G_OBJECT(element), "port-range",
                     itr->second.as<std::string>(), NULL);
      }
      else if(paramKey == "probation") {
        g_object_set(G_OBJECT(element), "probation",
                     itr->second.as<guint>(), NULL);
      }
      else if(paramKey == "proxy") {
        g_object_set(G_OBJECT(element), "proxy",
                     itr->second.as<std::string>(), NULL);
      }
      else if(paramKey == "proxy-id") {
        g_object_set(G_OBJECT(element), "proxy-id",
                     itr->second.as<std::string>(), NULL);
      }
      else if(paramKey == "proxy-pw") {
        g_object_set(G_OBJECT(element), "proxy-pw",
                     itr->second.as<std::string>(), NULL);
      }
      else if(paramKey == "retry") {
        g_object_set(G_OBJECT(element), "retry",
                     itr->second.as<guint>(), NULL);
      }
      else if(paramKey == "rfc7273-sync") {
        g_object_set(G_OBJECT(element), "rfc7273-sync",
                     itr->second.as<gboolean>(), NULL);
      }
      else if(paramKey == "rtp-blocksize") {
        g_object_set(G_OBJECT(element), "rtp-blocksize",
                     itr->second.as<guint>(), NULL);
      }
      else if(paramKey == "short-header") {
        g_object_set(G_OBJECT(element), "short-header",
                     itr->second.as<gboolean>(), NULL);
      }
      else if(paramKey == "tcp-timeout") {
        g_object_set(G_OBJECT(element), "tcp-timeout",
                     itr->second.as<guint64>(), NULL);
      }
      else if(paramKey == "teardown-timeout") {
        g_object_set(G_OBJECT(element), "teardown-timeout",
                     itr->second.as<guint64>(), NULL);
      }
      else if(paramKey == "timeout") {
        g_object_set(G_OBJECT(element), "timeout",
                     itr->second.as<guint64>(), NULL);
      }
      else if(paramKey == "udp-buffer-size") {
        g_object_set(G_OBJECT(element), "udp-buffer-size",
                     itr->second.as<gint>(), NULL);
      }
      else if(paramKey == "udp-reconnect") {
        g_object_set(G_OBJECT(element), "udp-reconnect",
                     itr->second.as<gboolean>(), NULL);
      }
      else if(paramKey == "use-pipeline-clock") {
        g_object_set(G_OBJECT(element), "use-pipeline-clock",
                     itr->second.as<gboolean>(), NULL);
      }
      else if(paramKey == "user-agent") {
        g_object_set(G_OBJECT(element), "user-agent",
                     itr->second.as<std::string>(), NULL);
      }
      else if(paramKey == "user-id") {
        g_object_set(G_OBJECT(element), "user-id",
                     itr->second.as<std::string>(), NULL);
      }
      else if(paramKey == "user-pw") {
        g_object_set(G_OBJECT(element), "user-pw",
                     itr->second.as<std::string>(), NULL);
      }
      else {
        std::cerr << "!! [WARNING] Unknown param found for rtsp source: " << paramKey << std::endl;
      }
    }
  }

  return ret;
}

/** Function to set properties at nvarguscamerasrc element using YAML config file.*/
NvDsYamlParserStatus
nvds_parse_nvarguscamerasrc(GstElement *element, gchar *cfg_file_path, const char* group)
{
  NvDsYamlParserStatus ret = NVDS_YAML_PARSER_SUCCESS;
  GstElementFactory *factory = GST_ELEMENT_GET_CLASS(element)->elementfactory;

  if (g_strcmp0(GST_OBJECT_NAME(factory), "nvarguscamerasrc")) {
    std::cerr << "[ERROR] Passed element is not nvarguscamerasrc" << std::endl;
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
      else if(paramKey == "blocksize") {
        g_object_set(G_OBJECT(element), "blocksize",
                     itr->second.as<guint>(), NULL);
      }
      else if(paramKey == "num-buffers") {
        g_object_set(G_OBJECT(element), "num-buffers",
                     itr->second.as<gint>(), NULL);
      }
      else if(paramKey == "typefind") {
        g_object_set(G_OBJECT(element), "typefind",
                     itr->second.as<gboolean>(), NULL);
      }
      else if(paramKey == "do-timestamp") {
        g_object_set(G_OBJECT(element), "do-timestamp",
                     itr->second.as<gboolean>(), NULL);
      }
      else if(paramKey == "silent") {
        g_object_set(G_OBJECT(element), "silent",
                     itr->second.as<gboolean>(), NULL);
      }
      else if(paramKey == "timeout") {
        g_object_set(G_OBJECT(element), "timeout",
                     itr->second.as<guint>(), NULL);
      }
      else if(paramKey == "saturation") {
        g_object_set(G_OBJECT(element), "saturation",
                     itr->second.as<gfloat>(), NULL);
      }
      else if(paramKey == "sensor-id") {
        g_object_set(G_OBJECT(element), "sensor-id",
                     itr->second.as<gint>(), NULL);
      }
      else if(paramKey == "sensor-mode") {
        g_object_set(G_OBJECT(element), "sensor-mode",
                     itr->second.as<gint>(), NULL);
      }
      else if(paramKey == "total-sensor-modes") {
        g_object_set(G_OBJECT(element), "total-sensor-modes",
                     itr->second.as<gint>(), NULL);
      }
      else if(paramKey == "exposuretimerange") {
        g_object_set(G_OBJECT(element), "exposuretimerange",
                     itr->second.as<std::string>(), NULL);
      }
      else if(paramKey == "gainrange") {
        g_object_set(G_OBJECT(element), "gainrange",
                     itr->second.as<std::string>(), NULL);
      }
      else if(paramKey == "ispdigitalgainrange") {
        g_object_set(G_OBJECT(element), "ispdigitalgainrange",
                     itr->second.as<std::string>(), NULL);
      }
      else if(paramKey == "tnr-strength") {
        g_object_set(G_OBJECT(element), "tnr-strength",
                     itr->second.as<gfloat>(), NULL);
      }
      else if(paramKey == "tnr-mode") {
        g_object_set(G_OBJECT(element), "tnr-mode",
                     itr->second.as<guint>(), NULL);
      }
      else if(paramKey == "ee-mode") {
        g_object_set(G_OBJECT(element), "ee-mode",
                     itr->second.as<guint>(), NULL);
      }
      else if(paramKey == "ee-strength") {
        g_object_set(G_OBJECT(element), "ee-strength",
                     itr->second.as<gfloat>(), NULL);
      }
      else if(paramKey == "aeantibanding") {
        g_object_set(G_OBJECT(element), "aeantibanding",
                     itr->second.as<guint>(), NULL);
      }
      else if(paramKey == "exposurecompensation") {
        g_object_set(G_OBJECT(element), "exposurecompensation",
                     itr->second.as<gfloat>(), NULL);
      }
      else if(paramKey == "aelock") {
        g_object_set(G_OBJECT(element), "aelock",
                     itr->second.as<gboolean>(), NULL);
      }
      else if(paramKey == "awblock") {
        g_object_set(G_OBJECT(element), "awblock",
                     itr->second.as<gboolean>(), NULL);
      }
      else if(paramKey == "bufapi-version") {
        g_object_set(G_OBJECT(element), "bufapi-version",
                     itr->second.as<gboolean>(), NULL);
      }
      else {
        std::cerr << "!! [WARNING] Unknown param found for nvarguscamerasrc source: " << paramKey << std::endl;
      }
    }
  }

  return ret;
}

/** Function to set properties at v4l2src element using YAML config file.*/
NvDsYamlParserStatus
nvds_parse_v4l2src(GstElement *element, gchar *cfg_file_path, const char* group)
{
  NvDsYamlParserStatus ret = NVDS_YAML_PARSER_SUCCESS;

  GstElementFactory *factory = GST_ELEMENT_GET_CLASS(element)->elementfactory;

  if (g_strcmp0(GST_OBJECT_NAME(factory), "v4l2src")) {
    std::cerr << "[ERROR] Passed element is not v4l2src" << std::endl;
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
      else if(paramKey == "blocksize") {
        g_object_set(G_OBJECT(element), "blocksize",
                     itr->second.as<guint>(), NULL);
      }
      else if(paramKey == "brightness") {
        g_object_set(G_OBJECT(element), "brightness",
                     itr->second.as<gint>(), NULL);
      }
      else if(paramKey == "contrast") {
        g_object_set(G_OBJECT(element), "contrast",
                     itr->second.as<gint>(), NULL);
      }
      else if(paramKey == "device") {
        std::string temp = itr->second.as<std::string>();
        gchar *str = (char *) temp.c_str();
        g_object_set(G_OBJECT(element), "device",
                     str, NULL);
      }
      else if(paramKey == "device-fd") {
        g_object_set(G_OBJECT(element), "device-fd",
                     itr->second.as<gint>(), NULL);
      }
      else if(paramKey == "device-name") {
        g_object_set(G_OBJECT(element), "device-name",
                     itr->second.as<gint>(), NULL);
      }
      else if(paramKey == "do-timestamp") {
        g_object_set(G_OBJECT(element), "do-timestamp",
                     itr->second.as<gboolean>(), NULL);
      }
      else if(paramKey == "force-aspect-ratio") {
        g_object_set(G_OBJECT(element), "force-aspect-ratio",
                     itr->second.as<std::string>(), NULL);
      }
      else if(paramKey == "hue") {
        g_object_set(G_OBJECT(element), "hue",
                     itr->second.as<gint>(), NULL);
      }
      else if(paramKey == "io-mode") {
        g_object_set(G_OBJECT(element), "io-mode",
                     itr->second.as<guint>(), NULL);
      }
      else if(paramKey == "norm") {
        g_object_set(G_OBJECT(element), "norm",
                     itr->second.as<guint>(), NULL);
      }
      else if(paramKey == "num-buffers") {
        g_object_set(G_OBJECT(element), "num-buffers",
                     itr->second.as<gint>(), NULL);
      }
      else if(paramKey == "pixel-aspect-ratio") {
        g_object_set(G_OBJECT(element), "pixel-aspect-ratio",
                     itr->second.as<std::string>(), NULL);
      }
      else if(paramKey == "saturation") {
        g_object_set(G_OBJECT(element), "saturation",
                     itr->second.as<gint>(), NULL);
      }
      else if(paramKey == "typefind") {
        g_object_set(G_OBJECT(element), "typefind",
                     itr->second.as<gboolean>(), NULL);
      }
      else {
        std::cerr << "!! [WARNING] Unknown param found for v4l2src: " << paramKey << std::endl;
      }
    }
  }

  return ret;
}

/** Function to set properties at multifilesrc element using YAML config file.*/
NvDsYamlParserStatus
nvds_parse_multifilesrc(GstElement *element, gchar *cfg_file_path, const char* group)
{
  NvDsYamlParserStatus ret = NVDS_YAML_PARSER_SUCCESS;

  GstElementFactory *factory = GST_ELEMENT_GET_CLASS(element)->elementfactory;

  if (g_strcmp0(GST_OBJECT_NAME(factory), "multifilesrc")) {
    std::cerr << "[ERROR] Passed element is not multifilesrc" << std::endl;
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
      else if(paramKey == "blocksize") {
        g_object_set(G_OBJECT(element), "blocksize",
                     itr->second.as<guint>(), NULL);
      }
      else if(paramKey == "do-timestamp") {
        g_object_set(G_OBJECT(element), "do-timestamp",
                     itr->second.as<gboolean>(), NULL);
      }
      else if(paramKey == "index") {
        g_object_set(G_OBJECT(element), "index",
                     itr->second.as<gint>(), NULL);
      }
      else if(paramKey == "location") {

        std::string temp = itr->second.as<std::string>();
        gchar *str = (char *) temp.c_str();
        g_object_set(G_OBJECT(element), "location",
                     str, NULL);
      }
      else if(paramKey == "loop") {
        g_object_set(G_OBJECT(element), "loop",
                     itr->second.as<gboolean>(), NULL);
      }
      else if(paramKey == "num-buffers") {
        g_object_set(G_OBJECT(element), "num-buffers",
                     itr->second.as<gint>(), NULL);
      }
      else if(paramKey == "start-index") {
        g_object_set(G_OBJECT(element), "start-index",
                     itr->second.as<gint>(), NULL);
      }
      else if(paramKey == "stop-index") {
        g_object_set(G_OBJECT(element), "stop-index",
                     itr->second.as<gint>(), NULL);
      }
      else if(paramKey == "typefind") {
        g_object_set(G_OBJECT(element), "typefind",
                     itr->second.as<gboolean>(), NULL);
      }
      else {
        std::cerr << "!! [WARNING] Unknown param found for multifilesrc: " << paramKey << std::endl;
      }
    }
  }

  return ret;
}

/** Function to set properties at alsasrc element using YAML config file.*/
NvDsYamlParserStatus
nvds_parse_alsasrc(GstElement *element, gchar *cfg_file_path, const char* group)
{
  NvDsYamlParserStatus ret = NVDS_YAML_PARSER_SUCCESS;

  GstElementFactory *factory = GST_ELEMENT_GET_CLASS(element)->elementfactory;

  if (g_strcmp0(GST_OBJECT_NAME(factory), "alsasrc")) {
    std::cerr << "[ERROR] Passed element is not alsasrc" << std::endl;
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
      else if(paramKey == "actual-buffer-time") {
        g_object_set(G_OBJECT(element), "actual-buffer-time",
                     itr->second.as<gint64>(), NULL);
      }
      else if(paramKey == "actual-latency-time") {
        g_object_set(G_OBJECT(element), "actual-latency-time",
                     itr->second.as<gint64>(), NULL);
      }
      else if(paramKey == "blocksize") {
        g_object_set(G_OBJECT(element), "blocksize",
                     itr->second.as<guint>(), NULL);
      }
      else if(paramKey == "buffer-time") {
        g_object_set(G_OBJECT(element), "buffer-time",
                     itr->second.as<gint64>(), NULL);
      }
      else if(paramKey == "card-name") {
        g_object_set(G_OBJECT(element), "card-name",
                     itr->second.as<std::string>(), NULL);
      }
      else if(paramKey == "device") {
        g_object_set(G_OBJECT(element), "device",
                     itr->second.as<std::string>(), NULL);
      }
      else if(paramKey == "device-name") {
        g_object_set(G_OBJECT(element), "device-name",
                     itr->second.as<std::string>(), NULL);
      }
      else if(paramKey == "do-timestamp") {
        g_object_set(G_OBJECT(element), "do-timestamp",
                     itr->second.as<gboolean>(), NULL);
      }
      else if(paramKey == "latency-time") {
        g_object_set(G_OBJECT(element), "latency-time",
                     itr->second.as<gint64>(), NULL);
      }
      else if(paramKey == "num-buffers") {
        g_object_set(G_OBJECT(element), "num-buffers",
                     itr->second.as<gint>(), NULL);
      }
      else if(paramKey == "provide-clock") {
        g_object_set(G_OBJECT(element), "provide-clock",
                     itr->second.as<gboolean>(), NULL);
      }
      else if(paramKey == "slave-method") {
        g_object_set(G_OBJECT(element), "slave-method",
                     itr->second.as<guint>(), NULL);
      }
      else if(paramKey == "typefind") {
        g_object_set(G_OBJECT(element), "typefind",
                     itr->second.as<gboolean>(), NULL);
      }
      else {
        std::cerr << "!! [WARNING] Unknown param found for alsasrc: " << paramKey << std::endl;
      }
    }
  }

  return ret;
}