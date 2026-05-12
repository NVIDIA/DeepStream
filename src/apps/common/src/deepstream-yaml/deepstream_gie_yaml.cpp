/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "deepstream_common.h"
#include "deepstream_config_file_parser.h"
#include "deepstream_config_yaml.h"
#include <string>
#include <cstring>
#include <iostream>

using std::cout;
using std::endl;

gboolean
parse_gie_yaml (NvDsGieConfig *config, std::string group_str, gchar *cfg_file_path)
{
  gboolean ret = FALSE;
  YAML::Node configyml = YAML::LoadFile(cfg_file_path);
  char *group = (char*) malloc(sizeof(char) * 1024);
  std::strncpy (group, group_str.c_str(), 1023);

  if (configyml[group_str]["enable"]) {
    gboolean val = configyml[group_str]["enable"].as<gboolean>();
    if(val == FALSE) {
      g_free(group);
      return TRUE;
    }
  }

  config->bbox_border_color_table = g_hash_table_new (NULL, NULL);
  config->bbox_bg_color_table = g_hash_table_new (NULL, NULL);
  config->bbox_border_color = (NvOSD_ColorParams) {1, 0, 0, 1};
  std::string border_str = "bbox-border-color";
  std::string bg_str = "bbox-bg-color";

 for(YAML::const_iterator itr = configyml[group_str].begin();
     itr != configyml[group_str].end(); ++itr)
  {
    std::string paramKey = itr->first.as<std::string>();

    if (paramKey == "enable") {
      config->enable = itr->second.as<gboolean>();
    } else if (paramKey == "input-tensor-meta") {
      config->input_tensor_meta = itr->second.as<gboolean>();
    } else if (paramKey == "operate-on-class-ids") {
      std::string str = itr->second.as<std::string>();
      std::vector<std::string> vec = split_string (str);
      int length = vec.size();
      int *arr = (int *) malloc(length * sizeof(int));
      for (unsigned int i = 0; i < vec.size(); i++) {
        arr[i] = std::stoi(vec[i]);
      }
      config->list_operate_on_class_ids = arr;
      config->num_operate_on_class_ids = length;
    } else if (paramKey == "batch-size") {
      config->batch_size = itr->second.as<guint>();
      config->is_batch_size_set = TRUE;
    } else if (paramKey == "model-engine-file") {
      std::string temp = itr->second.as<std::string>();
      char* str = (char*) malloc(sizeof(char) * 1024);
      std::strncpy (str, temp.c_str(), 1023);
      config->model_engine_file_path = (char*) malloc(sizeof(char) * 1024);
      if (!get_absolute_file_path_yaml (cfg_file_path, str,
            config->model_engine_file_path)) {
        g_printerr ("Error: Could not parse model-engine-file in %s.\n", group);
        g_free (str);
        goto done;
      }
      g_free (str);
    } else if (paramKey == "plugin-type") {
      config->plugin_type = (NvDsGiePluginType) itr->second.as<guint>();
    } else if (paramKey == "audio-transform") {
      std::string temp = itr->second.as<std::string>();
      config->audio_transform = (char*) malloc(sizeof(char) * 1024);
      std::strncpy (config->audio_transform, temp.c_str(), 1023);
    } else if (paramKey == "audio-framesize") {
      config->frame_size = itr->second.as<guint>();
      config->is_frame_size_set = TRUE;
    } else if (paramKey == "audio-hopsize") {
      config->hop_size = itr->second.as<guint>();
      config->is_hop_size_set = TRUE;
    } else if (paramKey == "audio-input-rate") {
      config->input_audio_rate = itr->second.as<guint>();
      config->is_hop_size_set = TRUE;
    } else if (paramKey == "labelfile-path") {
      std::string temp = itr->second.as<std::string>();
      char* str = (char*) malloc(sizeof(char) * 1024);
      std::strncpy (str, temp.c_str(), 1023);
      config->label_file_path = (char*) malloc(sizeof(char) * 1024);
      if (!get_absolute_file_path_yaml (cfg_file_path, str,
            config->label_file_path)) {
        g_printerr ("Error: Could not parse labelfile-path in %s.\n", group);
        g_free (str);
        goto done;
      }
      g_free (str);
    } else if (paramKey == "config-file") {
      std::string temp = itr->second.as<std::string>();
      char* str = (char*) malloc(sizeof(char) * 1024);
      std::strncpy (str, temp.c_str(), 1023);
      config->config_file_path = (char*) malloc(sizeof(char) * 1024);
      if (!get_absolute_file_path_yaml (cfg_file_path, str,
            config->config_file_path)) {
        g_printerr ("Error: Could not parse config-file in %s.\n", group);
        g_free (str);
        goto done;
      }
      g_free (str);
    } else if (paramKey == "interval") {
      config->interval = itr->second.as<guint>();
      config->is_interval_set = TRUE;
    } else if (paramKey == "gie-unique-id") {
      config->unique_id = itr->second.as<guint>();
      config->is_unique_id_set = TRUE;
    } else if (paramKey == "operate-on-gie-id") {
      config->operate_on_gie_id = itr->second.as<gint>();
      config->is_operate_on_gie_id_set = TRUE;
    } else if (paramKey.compare(0, border_str.size(), border_str) == 0) {
      NvOSD_ColorParams *clr_params;
      std::string str = itr->second.as<std::string>();
      std::vector<std::string> vec = split_string (str);
      if (vec.size() != 4) {
        NVGSTDS_ERR_MSG_V
            ("Number of Color params should be exactly 4 "
            "floats {r, g, b, a} between 0 and 1");
        goto done;
      }

      gint64 class_index = -1;
      if(paramKey != border_str) {
        class_index = std::stoi(paramKey.substr(border_str.size()));
      }

      gdouble list[4];
      for(unsigned int i = 0; i < 4; i++) {
        list[i] = std::stod(vec[i]);
      }

      if (class_index == -1) {
        clr_params = &config->bbox_border_color;
      } else {
        clr_params = (NvOSD_ColorParams*) g_malloc (sizeof (NvOSD_ColorParams));
        g_hash_table_insert (config->bbox_border_color_table,
            class_index + (gchar *) NULL, clr_params);
      }

      clr_params->red = list[0];
      clr_params->green = list[1];
      clr_params->blue = list[2];
      clr_params->alpha = list[3];
    } else if (paramKey.compare(0, bg_str.size(), bg_str) == 0) {
      NvOSD_ColorParams *clr_params;
      std::string str = itr->second.as<std::string>();
      std::vector<std::string> vec = split_string (str);
      if (vec.size() != 4) {
        NVGSTDS_ERR_MSG_V
            ("Number of Color params should be exactly 4 "
            "floats {r, g, b, a} between 0 and 1");
        goto done;
      }

      gint64 class_index = -1;
      if(paramKey != bg_str) {
        class_index = std::stoi(paramKey.substr(bg_str.size()));
      }

      gdouble list[4];
      for(unsigned int i = 0; i < 4; i++) {
        list[i] = std::stod(vec[i]);
      }

      if (class_index == -1) {
        clr_params = &config->bbox_bg_color;
        config->have_bg_color = TRUE;
      } else {
        clr_params = (NvOSD_ColorParams*) g_malloc (sizeof (NvOSD_ColorParams));
        g_hash_table_insert (config->bbox_bg_color_table, class_index + (gchar *) NULL,
            clr_params);
      }

      clr_params->red = list[0];
      clr_params->green = list[1];
      clr_params->blue = list[2];
      clr_params->alpha = list[3];
    } else if (paramKey == "infer-raw-output-dir") {
      std::string temp = itr->second.as<std::string>();
      char* str = (char*) malloc(sizeof(char) * 1024);
      std::strncpy (str, temp.c_str(), 1023);
      config->raw_output_directory = (char*) malloc(sizeof(char) * 1024);
      if (!get_absolute_file_path_yaml (cfg_file_path, str,
            config->raw_output_directory)) {
        g_printerr ("Error: Could not parse infer-raw-output-dir in %s.\n", group);
        g_free (str);
        goto done;
      }
      g_free (str);
    } else if (paramKey == "gpu-id") {
      config->gpu_id = itr->second.as<guint>();
      config->is_gpu_id_set = TRUE;
    } else if (paramKey == "nvbuf-memory-type") {
      config->nvbuf_memory_type = itr->second.as<guint>();
    } else {
      cout << "[WARNING] Unknown param found in gie: " << paramKey << endl;
    }
  }

  if (config->enable && config->label_file_path && !parse_labels_file (config)) {
    cout << "Failed while parsing label file " << config->label_file_path << endl;
    goto done;
  }
  if (!config->config_file_path) {
    cout << "Config file not provided for group " << group_str << endl;
    goto done;
  }

  ret = TRUE;

  if (group) {
    g_free (group);
    group = NULL;
  }
done:
  if (!ret) {
    cout <<  __func__ << " failed" << endl;
  }
  if (group) {
    g_free (group);
    group = NULL;
  }
  return ret;
}
