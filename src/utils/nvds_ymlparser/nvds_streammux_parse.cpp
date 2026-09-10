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
#include<unordered_map>

/** Function to set properties at nvstreammux element using YAML config file.*/
NvDsYamlParserStatus
nvds_parse_streammux(GstElement *element, gchar *cfg_file_path, const char* group)
{
  NvDsYamlParserStatus ret = NVDS_YAML_PARSER_SUCCESS;

  GstElementFactory *factory = GST_ELEMENT_GET_CLASS(element)->elementfactory;

  if (g_strcmp0(GST_OBJECT_NAME(factory), "nvstreammux")) {
    std::cerr << "[ERROR] Passed element is not nvstreammux" << std::endl;
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
      if (paramKey == "batch-size")  {
        g_object_set(G_OBJECT(element), "batch-size",
                   itr->second.as<guint>(), NULL);
      }
      else if (paramKey == "batched-push-timeout")  {
        g_object_set(G_OBJECT(element), "batched-push-timeout",
                   itr->second.as<gint>(), NULL);
      }
      else if(paramKey == "width") {
        g_object_set(G_OBJECT(element), "width",
                   itr->second.as<guint>(), NULL);
      }
      else if (paramKey == "height")  {
        g_object_set(G_OBJECT(element), "height",
                   itr->second.as<guint>(), NULL);
      }
      else if (paramKey == "enable-padding")  {
        g_object_set(G_OBJECT(element), "enable-padding",
                   itr->second.as<gboolean>(), NULL);
      }
      else if (paramKey == "gpu-id")  {
        g_object_set(G_OBJECT(element), "gpu-id",
                   itr->second.as<guint>(), NULL);
      }
      else if (paramKey == "live-source")  {
        g_object_set(G_OBJECT(element), "live-source",
                   itr->second.as<gboolean>(), NULL);
      }
      else if (paramKey == "nvbuf-memory-type")  {
        g_object_set(G_OBJECT(element), "nvbuf-memory-type",
                   itr->second.as<guint>(), NULL);
      }
      else if (paramKey == "num-surfaces-per-frame")  {
        g_object_set(G_OBJECT(element), "num-surfaces-per-frame",
                   itr->second.as<guint>(), NULL);
      }
      else if (paramKey == "buffer-pool-size")  {
        g_object_set(G_OBJECT(element), "buffer-pool-size",
                   itr->second.as<guint>(), NULL);
      }
      else if (paramKey == "attach-sys-ts")  {
        g_object_set(G_OBJECT(element), "attach-sys-ts",
                   itr->second.as<gboolean>(), NULL);
      }
      else if (paramKey == "compute-hw")  {
        g_object_set(G_OBJECT(element), "compute-hw",
                   itr->second.as<guint>(), NULL);
      }
      else if (paramKey == "interpolation-method")  {
        g_object_set(G_OBJECT(element), "interpolation-method",
                   itr->second.as<guint>(), NULL);
      }
      else if (paramKey == "sync-inputs")  {
        g_object_set(G_OBJECT(element), "sync-inputs",
                   itr->second.as<gboolean>(), NULL);
      }
      else if (paramKey == "frame-num-reset-on-eos")  {
        g_object_set(G_OBJECT(element), "frame-num-reset-on-eos",
                   itr->second.as<gboolean>(), NULL);
      }
      else if (paramKey == "max-latency")  {
        g_object_set(G_OBJECT(element), "max-latency",
                   itr->second.as<guint>(), NULL);
      }
      else if (paramKey == "async-process")  {
        g_object_set(G_OBJECT(element), "async-process",
                   itr->second.as<gboolean>(), NULL);
      }
      else if (paramKey == "drop-pipeline-eos")  {
        g_object_set(G_OBJECT(element), "drop-pipeline-eos",
                   itr->second.as<gboolean>(), NULL);
      }
      else if (paramKey == "failsafe-flush-count")  {
        g_object_set(G_OBJECT(element), "failsafe-flush-count",
                   itr->second.as<guint>(), NULL);
      }
      else {
        std::cerr << "!! [WARNING] Unknown param found for nvstreammux: " << paramKey << std::endl;
      }

    }
  }

  return ret;
}
