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
#include "nvds_appctx_server.h"

#include <yaml-cpp/yaml.h>
#include <string>
#include <iostream>
#include <unordered_map>

NvDsYamlParserStatus
nvds_parse_check_rest_server_with_app (gchar * cfg_file_path,  const char* group,
  gboolean  *within_multiurisrcbin)
{

  NvDsYamlParserStatus ret = NVDS_YAML_PARSER_SUCCESS;

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
              "!! [WARNING]  \"rest-server\" group not enabled. Use \"enable: 1\" to set properties."
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
      }
      else if (paramKey == "within_multiurisrcbin") {
        *within_multiurisrcbin = itr->second.as < gboolean > ();
      }
      else {
        std::
            cerr << "!! [WARNING] Unknown param found for rest-server: " <<
            paramKey << std::endl;
      }
    }
  }

  return ret;
}

/** Function to set properties at nvmultiurisrcbin element using YAML config file.*/
NvDsYamlParserStatus
nvds_parse_server_appctx (gchar * cfg_file_path,  const char* group,
    AppCtx *ctx)
{

  NvDsYamlParserStatus ret = NVDS_YAML_PARSER_SUCCESS;

  AppCtx *appctx = (AppCtx*) ctx;

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
              "!! [WARNING]  \"appctx\" group not enabled. Use \"enable: 1\" to set properties."
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
      }
      else if (paramKey == "httpIp") {
        std::string temp = itr->second.as < std::string > ();
        appctx->httpIp = (char *) malloc (sizeof (char) * 1024);
        strcpy (appctx->httpIp, temp.c_str ());
      } else if (paramKey == "httpPort") {
        std::string temp = itr->second.as < std::string > ();
        appctx->httpPort = (char *) malloc (sizeof (char) * 1024);
        strcpy (appctx->httpPort, temp.c_str ());
      } else if (paramKey == "uri_list") {
        std::string temp = itr->second.as < std::string > ();
        appctx->uri_list = (char *) malloc (sizeof (char) * 1024);
        strcpy (appctx->uri_list, temp.c_str ());
      } else if (paramKey == "pipeline_width") {
        (appctx->muxConfig).pipeline_width = itr->second.as < gint > ();
      } else if (paramKey == "pipeline_height") {
        (appctx->muxConfig).pipeline_height = itr->second.as < gint > ();
      } else if (paramKey == "batched_push_timeout") {
        (appctx->muxConfig).batched_push_timeout = itr->second.as < gint > ();
      } else if (paramKey == "batch_size") {
        (appctx->muxConfig).batch_size = itr->second.as < gint > ();
      } else if (paramKey == "buffer_pool_size") {
        (appctx->muxConfig).buffer_pool_size = itr->second.as < gint > ();
      } else if (paramKey == "compute_hw") {
        (appctx->muxConfig).compute_hw = itr->second.as < gint > ();
      } else if (paramKey == "num_surfaces_per_frame") {
        (appctx->muxConfig).num_surfaces_per_frame = itr->second.as < gint > ();
      } else if (paramKey == "interpolation_method") {
        (appctx->muxConfig).interpolation_method = itr->second.as < gint > ();
      } else if (paramKey == "gpu_id") {
        (appctx->muxConfig).gpu_id = itr->second.as < gint > ();
      } else if (paramKey == "nvbuf_memory_type") {
        (appctx->muxConfig).nvbuf_memory_type = itr->second.as < guint > ();
      } else if (paramKey == "live_source") {
        (appctx->muxConfig).live_source = itr->second.as < gboolean > ();
      } else if (paramKey == "enable_padding") {
        (appctx->muxConfig).enable_padding = itr->second.as < gboolean > ();
      } else if (paramKey == "attach_sys_ts_as_ntp") {
        (appctx->muxConfig).attach_sys_ts_as_ntp = itr->second.as < gboolean > ();
      } else if (paramKey == "config_file_path") {
        std::string temp = itr->second.as < std::string > ();
        (appctx->muxConfig).config_file_path = (char *) malloc (sizeof (char) * 1024);
        strcpy ((appctx->muxConfig).config_file_path, temp.c_str ());
      } else if (paramKey == "sync_inputs") {
        (appctx->muxConfig).attach_sys_ts_as_ntp = itr->second.as < gboolean > ();
      } else if (paramKey == "max_latency") {
        (appctx->muxConfig).max_latency = itr->second.as < guint64 > ();
      } else if (paramKey == "frame_num_reset_on_eos") {
        (appctx->muxConfig).frame_num_reset_on_eos = itr->second.as < gboolean > ();
      } else if (paramKey == "frame_num_reset_on_stream_reset") {
        (appctx->muxConfig).frame_num_reset_on_stream_reset = itr->second.as < gboolean > ();
      } else if (paramKey == "frame_duration") {
        (appctx->muxConfig).frame_duration = itr->second.as < guint64 > ();
      } else if (paramKey == "maxBatchSize") {
        (appctx->muxConfig).maxBatchSize = itr->second.as < guint > ();
      } else if (paramKey == "async_process") {
        (appctx->muxConfig).async_process = itr->second.as < gboolean > ();
      } else if (paramKey == "drop_pipeline_eos") {
        (appctx->muxConfig).no_pipeline_eos = itr->second.as < gboolean > ();
      }
      else {
        std::
            cerr << "!! [WARNING] Unknown param found for appctx: " <<
            paramKey << std::endl;
      }
    }
  }

  return ret;
}
