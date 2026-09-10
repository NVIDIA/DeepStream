/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/** Function to set properties at nvmultiurisrcbin element using YAML config file.*/
NvDsYamlParserStatus
nvds_parse_multiurisrcbin (GstElement * element, gchar * cfg_file_path,
    const char *group)
{

  NvDsYamlParserStatus ret = NVDS_YAML_PARSER_SUCCESS;

  GstElementFactory *factory = GST_ELEMENT_GET_CLASS (element)->elementfactory;

  if (g_strcmp0 (GST_OBJECT_NAME (factory), "nvmultiurisrcbin")) {
    std::cerr << "[ERROR] Passed element is not  nvmultiurisrcbin" << std::endl;
    ret = NVDS_YAML_PARSER_ERROR;
    return ret;
  }

  const gchar *new_mux_str = g_getenv ("USE_NEW_NVSTREAMMUX");
  gboolean use_new_mux = !g_strcmp0 (new_mux_str, "yes");

  std::string paramKey = "";

  auto docs = YAML::LoadAllFromFile (cfg_file_path);

  std::vector < int >docs_indx_vec;
  std::unordered_map < std::string, int >docs_indx_umap;

  int total_docs = docs.size ();

  for (int i = 0; i < total_docs; i++) {
    if (docs[i][group].Type() != YAML::NodeType::Null) {

      if (docs[i][group]["enable"]) {
        gboolean val = docs[i][group]["enable"].as < gboolean > ();
        if (val == FALSE) {
          std::
              cerr <<
              "!! [WARNING]  \"nvmultiurisrcbin\" group not enabled. Use \"enable: 1\" to set properties."
              << std::endl;
          ret = NVDS_YAML_PARSER_DISABLED;
          return ret;
        }
      }

      YAML::const_iterator itr = docs[i].begin ();
      std::string group_name = itr->first.as < std::string > ();
      docs_indx_umap[group_name] = i;
      docs_indx_vec.push_back (i);

    }
  }

  int docs_indx_vec_size = docs_indx_vec.size ();

  int docs_indx_umap_size = docs_indx_umap.size ();

  if (docs_indx_umap_size != docs_indx_vec_size) {
    std::
        cerr << "[ERROR] Duplicate group names in the config file : " << group
        << std::endl;
    ret = NVDS_YAML_PARSER_ERROR;
    return ret;
  }

  for (int i = 0; i < docs_indx_vec_size; i++) {
    int indx = docs_indx_vec[i];
    for (YAML::const_iterator itr = docs[indx][group].begin ();
        itr != docs[indx][group].end (); ++itr) {
      paramKey = itr->first.as < std::string > ();

      if (paramKey == "enable"
          && docs[indx][group]["enable"].as < gboolean > () == TRUE) {
        continue;
      } else if (paramKey == "ip-address") {
        std::string temp = itr->second.as < std::string > ();
        char *str = (char *) malloc (sizeof (char) * 1024);
        strcpy (str, temp.c_str ());
        g_object_set (G_OBJECT (element), "ip-address", str, NULL);
        g_free (str);
      } else if (paramKey == "port") {
        std::string temp = itr->second.as < std::string > ();
        char *str = (char *) malloc (sizeof (char) * 1024);
        strcpy (str, temp.c_str ());
        g_object_set (G_OBJECT (element), "port", str, NULL);

        g_free (str);
      } else if (paramKey == "uri-list") {
        std::string temp = itr->second.as < std::string > ();
        char *str = (char *) malloc (sizeof (char) * 1024);
        strcpy (str, temp.c_str ());
        g_object_set (G_OBJECT (element), "uri-list", str, NULL);
        g_free (str);
      } else if (paramKey == "live-source") {
        g_object_set (G_OBJECT (element), "live-source",
            itr->second.as < gboolean > (), NULL);
      } else if (paramKey == "width") {
        if (!use_new_mux){
          g_object_set (G_OBJECT (element), "width",
              itr->second.as < guint > (), NULL);
        }
      } else if (paramKey == "height") {
        if (!use_new_mux){
          g_object_set (G_OBJECT (element), "height",
              itr->second.as < guint > (), NULL);
        }
      } else if (paramKey == "max-batch-size") {
        g_object_set (G_OBJECT (element), "max-batch-size",
            itr->second.as < guint > (), NULL);
      } else if (paramKey == "batched-push-timeout") {
        if (!use_new_mux){
          g_object_set (G_OBJECT (element), "batched-push-timeout",
              itr->second.as < guint > (), NULL);
        }
      } else if (paramKey == "disable-passthrough") {
        g_object_set (G_OBJECT (element), "disable-passthrough",
            itr->second.as < gboolean > (), NULL);
      } else if (paramKey == "rtsp-reconnect-interval") {
        g_object_set (G_OBJECT (element), "rtsp-reconnect-interval",
            itr->second.as < guint > (), NULL);
      } else if (paramKey == "init-rtsp-reconnect-interval") {
        g_object_set (G_OBJECT (element), "init-rtsp-reconnect-interval",
            itr->second.as < guint > (), NULL);
      } else if (paramKey == "rtsp-reconnect-attempts") {
        g_object_set (G_OBJECT (element), "rtsp-reconnect-attempts",
            itr->second.as < gint > (), NULL);
      } else if (paramKey == "disable-audio") {
        g_object_set (G_OBJECT (element), "disable-audio",
            itr->second.as < gboolean > (), NULL);
      } else if (paramKey == "drop-pipeline-eos") {
        g_object_set (G_OBJECT (element), "drop-pipeline-eos",
            itr->second.as < gboolean > (), NULL);
      } else if (paramKey == "max-latency") {
        g_object_set (G_OBJECT (element), "max-latency",
            itr->second.as < guint > (), NULL);
      } else if (paramKey == "config-file-path") {
        if (use_new_mux){
          std::string temp = itr->second.as < std::string > ();
          char *str = (char *) malloc (sizeof (char) * 1024);
          strcpy (str, temp.c_str ());
          g_object_set (G_OBJECT (element), "config-file-path",
              str, NULL);
          g_free (str);
        }
      } else if (paramKey == "enable-error-propagation")
      {
        g_object_set (G_OBJECT (element), "enable-error-propagation",
            itr->second.as < gboolean > (), NULL);
      } else if (paramKey == "proto-lib") {
        std::string temp = itr->second.as < std::string > ();
        char *str = (char *) malloc (sizeof (char) * 1024);
        strcpy (str, temp.c_str ());
        g_object_set (G_OBJECT (element), "proto-lib",
            str, NULL);
        g_free (str);
      } else if (paramKey == "conn-str") {
        std::string temp = itr->second.as < std::string > ();
        char *str = (char *) malloc (sizeof (char) * 1024);
        strcpy (str, temp.c_str ());
        g_object_set (G_OBJECT (element), "conn-str",
            str, NULL);
        g_free (str);
      } else if (paramKey == "topic") {
        std::string temp = itr->second.as < std::string > ();
        char *str = (char *) malloc (sizeof (char) * 1024);
        strcpy (str, temp.c_str ());
        g_object_set (G_OBJECT (element), "topic",
            str, NULL);
        g_free (str);
      } else {
        std::
            cerr << "!! [WARNING] Unknown param found for nvmultiurisrcbin: " <<
            paramKey << std::endl;
      }
    }
  }

  return ret;
}
