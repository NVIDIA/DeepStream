/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <string>
#include <cstring>
#include "deepstream_app.h"
#include "deepstream_config_yaml.h"
#include <iostream>

#include <stdlib.h>
#include <fstream>

using std::cout;
using std::endl;

static gboolean
parse_tests_yaml (NvDsConfig *config, gchar *cfg_file_path)
{
  gboolean ret = FALSE;
  YAML::Node configyml = YAML::LoadFile(cfg_file_path);

  for(YAML::const_iterator itr = configyml["tests"].begin();
     itr != configyml["tests"].end(); ++itr)
  {
    std::string paramKey = itr->first.as<std::string>();
    if (paramKey == "file-loop") {
      config->file_loop = itr->second.as<gint>();
    } else {
      cout << "Unknown key " << paramKey << " for group tests" << endl;
    }
  }

  ret = TRUE;

  if (!ret) {
    cout <<  __func__ << " failed" << endl;
  }
  return ret;
}

static gboolean
parse_app_yaml (NvDsConfig *config, gchar *cfg_file_path)
{
  gboolean ret = FALSE;
  YAML::Node configyml = YAML::LoadFile(cfg_file_path);

  for(YAML::const_iterator itr = configyml["application"].begin();
     itr != configyml["application"].end(); ++itr)
  {
    std::string paramKey = itr->first.as<std::string>();
    if (paramKey == "enable-perf-measurement") {
      config->enable_perf_measurement =
          itr->second.as<gboolean>();
    } else if (paramKey == "perf-measurement-interval-sec") {
      config->perf_measurement_interval_sec =
          itr->second.as<guint>();
    } else if (paramKey == "gie-kitti-output-dir") {
      std::string temp = itr->second.as<std::string>();
      char* str = (char*) malloc(sizeof(char) * 1024);
      std::strncpy (str, temp.c_str(), 1023);
      config->bbox_dir_path = (char*) malloc(sizeof(char) * 1024);
      get_absolute_file_path_yaml (cfg_file_path, str, config->bbox_dir_path);
      g_free(str);
    } else if (paramKey == "kitti-track-output-dir") {
      std::string temp = itr->second.as<std::string>();
      char* str = (char*) malloc(sizeof(char) * 1024);
      std::strncpy (str, temp.c_str(), 1023);
      config->kitti_track_dir_path = (char*) malloc(sizeof(char) * 1024);
      get_absolute_file_path_yaml (cfg_file_path, str, config->kitti_track_dir_path);
      g_free(str);
    } else if (paramKey == "reid-track-output-dir") {
      std::string temp = itr->second.as<std::string>();
      char* str = (char*) malloc(sizeof(char) * 1024);
      std::strncpy (str, temp.c_str(), 1023);
      config->reid_track_dir_path = (char*) malloc(sizeof(char) * 1024);
      get_absolute_file_path_yaml (cfg_file_path, str, config->reid_track_dir_path);
      g_free(str);
    } else if (paramKey == "global-gpu-id") {
      /** App Level GPU ID is set here if it is present in APP LEVEL config group
       * if gpu_id prop is not set for any component, this global_gpu_id will be used */
      config->global_gpu_id = itr->second.as<guint>();
    } else if (paramKey == "terminated-track-output-dir") {
      std::string temp = itr->second.as<std::string>();
      char* str = (char*) malloc(sizeof(char) * 1024);
      std::strncpy (str, temp.c_str(), 1023);
      config->terminated_track_output_path = (char*) malloc(sizeof(char) * 1024);
      get_absolute_file_path_yaml (cfg_file_path, str, config->terminated_track_output_path);
      g_free(str);
    } else if (paramKey == "shadow-track-output-dir") {
      std::string temp = itr->second.as<std::string>();
      char* str = (char*) malloc(sizeof(char) * 1024);
      std::strncpy (str, temp.c_str(), 1023);
      config->shadow_track_output_path = (char*) malloc(sizeof(char) * 1024);
      get_absolute_file_path_yaml (cfg_file_path, str, config->shadow_track_output_path);
      g_free(str);
    }
    else {
      cout << "Unknown key " << paramKey << " for group application" << endl;
    }
  }

  ret = TRUE;

  if (!ret) {
    cout <<  __func__ << " failed" << endl;
  }
  return ret;
}

static std::vector<std::string>
split_csv_entries (std::string input) {
  std::vector<int> positions;
  for (unsigned int i = 0; i < input.size(); i++) {
    if (input[i] == ',')
      positions.push_back(i);
  }
  std::vector<std::string> ret;
  int prev = 0;
  for (auto &j: positions) {
    std::string temp = input.substr(prev, j - prev);
    ret.push_back(temp);
    prev = j + 1;
  }
  ret.push_back(input.substr(prev, input.size() - prev));
  return ret;
}

gboolean
parse_config_file_yaml (NvDsConfig *config, gchar *cfg_file_path)
{
  gboolean parse_err = false;
  gboolean ret = FALSE;
  YAML::Node configyml = YAML::LoadFile(cfg_file_path);
  std::string source_str = "source";
  std::string sink_str = "sink";
  std::string sgie_str = "secondary-gie";
  std::string msgcons_str = "message-consumer";
  std::string dewarper_str = "dewarper";

  config->source_list_enabled = FALSE;

  /** Initialize global gpu id to -1 */
  config->global_gpu_id = -1;

  /** App group parsing at top level to set global_gpu_id (if available)
   * before any other group parsing */
  if (configyml["application"]) {
    parse_err = !parse_app_yaml (config, cfg_file_path);
  }

  for(YAML::const_iterator itr = configyml.begin();
    itr != configyml.end(); ++itr) {
    std::string paramKey = itr->first.as<std::string>();
    if (paramKey == "source") {
      if(configyml["source"]["csv-file-path"]) {
        std::string csv_file_path = configyml["source"]["csv-file-path"].as<std::string>();
        char* str = (char*) malloc(sizeof(char) * 1024);
        std::strncpy (str, csv_file_path.c_str(), 1023);
        char *abs_csv_path = (char*) malloc(sizeof(char) * 1024);
        get_absolute_file_path_yaml (cfg_file_path, str, abs_csv_path);
        g_free(str);

        std::ifstream inputFile (abs_csv_path);
        if (!inputFile.is_open()) {
          cout << "Couldn't open CSV file " << abs_csv_path << endl;
        }
        std::string line, temp;
        /* Separating header field and inserting as strings into the vector.
        */
        while(getline(inputFile, line)){
          gboolean is_comment = false;
          size_t space_count = 0;
          for (char c : line) {
            if (c != ' ' && c!='\t') {
              if (c != '#') {
                is_comment = false;
              }
              else
              {
                is_comment = true;
              }
              break;
            }
            else {
              space_count++;
            }
          }
          if(!is_comment && space_count<line.length())
              break;
        }
        std::vector<std::string> headers = split_csv_entries(line);
        /*Parsing each csv entry as an input source */
        while(getline(inputFile, line)) {
          std::vector<std::string> source_values = split_csv_entries(line);
          if (config->num_source_sub_bins == MAX_SOURCE_BINS) {
            NVGSTDS_ERR_MSG_V ("App supports max %d sources", MAX_SOURCE_BINS);
            ret = FALSE;
            goto done;
          }
          guint source_id = 0;
          source_id = config->num_source_sub_bins;
          /** set gpu_id for source component using global_gpu_id(if available) */
          if (config->global_gpu_id != -1) {
            config->multi_source_config[source_id].gpu_id = config->global_gpu_id;
          }
          /** if gpu_id for source component is present,
           * it will override the value set using global_gpu_id in parse_source_yaml function */
          parse_err = !parse_source_yaml (&config->multi_source_config[source_id], headers, source_values, cfg_file_path);
          if (config->multi_source_config[source_id].enable)
            config->num_source_sub_bins++;
        }
      } else {
        NVGSTDS_ERR_MSG_V ("CSV file not specified\n");
        ret = FALSE;
        goto done;
      }
    }
    else if (paramKey == "streammux") {
      /** set gpu_id for streammux component using global_gpu_id(if available) */
      if (config->global_gpu_id != -1) {
        config->streammux_config.gpu_id = config->global_gpu_id;
      }
      /** if gpu_id for streammux component is present,
       * it will override the value set using global_gpu_id in parse_streammux_yaml function */
      parse_err = !parse_streammux_yaml(&config->streammux_config, cfg_file_path);
    }
    else if (paramKey == "osd") {
      /** set gpu_id for osd component using global_gpu_id(if available) */
      if (config->global_gpu_id != -1) {
        config->osd_config.gpu_id = config->global_gpu_id;
      }
      /** if gpu_id for osd component is present,
       * it will override the value set using global_gpu_id in parse_osd_yaml function */
      parse_err = !parse_osd_yaml(&config->osd_config, cfg_file_path);
    }
    else if (paramKey == "segvisual") {
      /**  set gpu_id for segvisual component using global_gpu_id(if available) */
      if (config->global_gpu_id != -1) {
        config->segvisual_config.gpu_id = config->global_gpu_id;
      }
      /** if gpu_id for segvisual component is present,
       * it will override the value set using global_gpu_id in parse_segvisual_yaml function */
      parse_err = !parse_segvisual_yaml(&config->segvisual_config, cfg_file_path);
    }
    else if (paramKey == "pre-process") {
      parse_err = !parse_preprocess_yaml(&config->preprocess_config, cfg_file_path);
    }
    else if (paramKey == "primary-gie") {
      /** set gpu_id for primary gie component using global_gpu_id(if available) */
      if (config->global_gpu_id != -1) {
        config->primary_gie_config.gpu_id = config->global_gpu_id;
        config->primary_gie_config.is_gpu_id_set = TRUE;
      }
      /** if gpu_id for primary gie component is present,
       * it will override the value set using global_gpu_id in parse_gie_yaml function */
      parse_err = !parse_gie_yaml(&config->primary_gie_config, paramKey, cfg_file_path);
    }
    else if (paramKey == "tracker") {
      /** set gpu_id for tracker component using global_gpu_id(if available) */
      if (config->global_gpu_id != -1) {
        config->tracker_config.gpu_id = config->global_gpu_id;
      }
      /** if gpu_id for tracker component is present,
       * it will override the value set using global_gpu_id in parse_tracker_yaml function */
      parse_err = !parse_tracker_yaml(&config->tracker_config, cfg_file_path);
    }
    else if (paramKey == "replay") {
      parse_err = !parse_replay_yaml(&config->replay_config, cfg_file_path);
    }
    else if (paramKey.compare(0, sgie_str.size(), sgie_str) == 0) {
      if (config->num_secondary_gie_sub_bins == MAX_SECONDARY_GIE_BINS) {
        NVGSTDS_ERR_MSG_V ("App supports max %d secondary GIEs", MAX_SECONDARY_GIE_BINS);
        ret = FALSE;
        goto done;
      }
      /* set gpu_id for secondary gie component using global_gpu_id(if available) */
      if (config->global_gpu_id != -1) {
        config->secondary_gie_sub_bin_config[config->num_secondary_gie_sub_bins].gpu_id = config->global_gpu_id;
        config->secondary_gie_sub_bin_config[config->num_secondary_gie_sub_bins].is_gpu_id_set = TRUE;
      }
      /** if gpu_id for secondary gie component is present,
       * it will override the value set using global_gpu_id in parse_gie_yaml function */
      parse_err =
          !parse_gie_yaml (&config->secondary_gie_sub_bin_config[config->
                                  num_secondary_gie_sub_bins],
                                  paramKey, cfg_file_path);
      if (config->secondary_gie_sub_bin_config[config->num_secondary_gie_sub_bins].enable){
        config->num_secondary_gie_sub_bins++;
      }
    }
    else if (paramKey.compare(0, sink_str.size(), sink_str) == 0) {
      if (config->num_sink_sub_bins == MAX_SINK_BINS) {
        NVGSTDS_ERR_MSG_V ("App supports max %d sinks", MAX_SINK_BINS);
        ret = FALSE;
        goto done;
      }

      /* set gpu_id for sink component using global_gpu_id(if available) */
      if (config->global_gpu_id != -1 && configyml[paramKey]["enable"].as<gboolean>()) {
        config->sink_bin_sub_bin_config[config->num_sink_sub_bins].encoder_config.gpu_id = config->sink_bin_sub_bin_config[config->num_sink_sub_bins].render_config.gpu_id = config->global_gpu_id;
      }
      /**  if gpu_id for sink component is present,
       * it will override the value set using global_gpu_id in parse_sink_yaml function */
      parse_err =
          !parse_sink_yaml (&config->
          sink_bin_sub_bin_config[config->num_sink_sub_bins], paramKey, cfg_file_path);
      if (config->
          sink_bin_sub_bin_config[config->num_sink_sub_bins].enable) {
        config->num_sink_sub_bins++;
      }
    }
    else if (paramKey.compare(0, msgcons_str.size(), msgcons_str) == 0) {
      if (config->num_message_consumers == MAX_MESSAGE_CONSUMERS) {
        NVGSTDS_ERR_MSG_V ("App supports max %d consumers", MAX_MESSAGE_CONSUMERS);
        ret = FALSE;
        goto done;
      }
      parse_err = !parse_msgconsumer_yaml (
                    &config->message_consumer_config[config->num_message_consumers],
                    paramKey, cfg_file_path);

      if (config->message_consumer_config[config->num_message_consumers].enable) {
        config->num_message_consumers++;
      }
    }
    else if (paramKey == "tiled-display") {
      /* set gpu_id for tiled display component using global_gpu_id(if available) */
      if (config->global_gpu_id != -1) {
        config->tiled_display_config.gpu_id = config->global_gpu_id;
      }
      /** if gpu_id for tiled display component is present,
       * it will override the value set using global_gpu_id in parse_tiled_display_yaml function */
      parse_err = !parse_tiled_display_yaml (&config->tiled_display_config, cfg_file_path);
    }
    else if (paramKey == "img-save") {
      /** set gpu_id for image save component using global_gpu_id(if available) */
      if (config->global_gpu_id != -1) {
        config->image_save_config.gpu_id = config->global_gpu_id;
      }
      /** if gpu_id for image save component is present,
       * it will override the value set using global_gpu_id in parse_image_save_yaml function */
      parse_err = !parse_image_save_yaml (&config->image_save_config , cfg_file_path);
    }
    else if (paramKey == "nvds-analytics") {
      parse_err = !parse_dsanalytics_yaml (&config->dsanalytics_config, cfg_file_path);
    }
    else if (paramKey == "ds-example") {
      /** set gpu_id for dsexample component using global_gpu_id(if available) */
      if (config->global_gpu_id != -1) {
        config->dsexample_config.gpu_id = config->global_gpu_id;
      }
      /** if gpu_id for dsexample component is present,
       * it will override the value set using global_gpu_id in parse_dsexample_yaml function */
      parse_err = !parse_dsexample_yaml (&config->dsexample_config, cfg_file_path);
    }
    else if (paramKey == "message-converter") {
      parse_err = !parse_msgconv_yaml (&config->msg_conv_config, paramKey, cfg_file_path);
    }
    else if (paramKey == "tests") {
      parse_err = !parse_tests_yaml (config, cfg_file_path);
    }
    else if (paramKey.compare(0, dewarper_str.size(), dewarper_str) == 0) {
      size_t start = paramKey.find(dewarper_str);
      int source_id = 0;
      if(start != std::string::npos) {
        std::string index_str = paramKey.substr(start+dewarper_str.length(), paramKey.length()-start-dewarper_str.length());
        source_id = std::stoi(index_str);
        parse_dewarper_yaml (&config->multi_source_config[source_id].dewarper_config, paramKey, cfg_file_path);
      } else {
        NVGSTDS_ERR_MSG_V ("Dewarper key is wrong ! ");
        parse_err = true;
      }
    }

    if (parse_err) {
      cout << "failed parsing" << endl;
      goto done;
    }
  }
  /* Updating batch size when source list is enabled */
  /* if (config->source_list_enabled == TRUE) {
      // For streammux and pgie, batch size is set to number of sources
      config->streammux_config.batch_size = config->num_source_sub_bins;
      config->primary_gie_config.batch_size = config->num_source_sub_bins;
      if (config->sgie_batch_size != 0) {
          for (i = 0; i < config->num_secondary_gie_sub_bins; i++) {
              config->secondary_gie_sub_bin_config[i].batch_size = config->sgie_batch_size;
          }
      }
  } */
  unsigned int i, j;
  for (i = 0; i < config->num_secondary_gie_sub_bins; i++) {
    if (config->secondary_gie_sub_bin_config[i].unique_id ==
        config->primary_gie_config.unique_id) {
      NVGSTDS_ERR_MSG_V ("Non unique gie ids found");
      ret = FALSE;
      goto done;
    }
  }

  for (i = 0; i < config->num_secondary_gie_sub_bins; i++) {
    for (j = i + 1; j < config->num_secondary_gie_sub_bins; j++) {
      if (config->secondary_gie_sub_bin_config[i].unique_id ==
          config->secondary_gie_sub_bin_config[j].unique_id) {
        NVGSTDS_ERR_MSG_V ("Non unique gie id %d found",
                            config->secondary_gie_sub_bin_config[i].unique_id);
        ret = FALSE;
        goto done;
      }
    }
  }

  for (i = 0; i < config->num_source_sub_bins; i++) {
    if (config->multi_source_config[i].type == NV_DS_SOURCE_URI_MULTIPLE) {
      if (config->multi_source_config[i].num_sources < 1) {
        config->multi_source_config[i].num_sources = 1;
      }
      for (j = 1; j < config->multi_source_config[i].num_sources; j++) {
        if (config->num_source_sub_bins == MAX_SOURCE_BINS) {
          NVGSTDS_ERR_MSG_V ("App supports max %d sources", MAX_SOURCE_BINS);
          ret = FALSE;
          goto done;
        }
        memcpy (&config->multi_source_config[config->num_source_sub_bins],
            &config->multi_source_config[i],
            sizeof (config->multi_source_config[i]));
        config->multi_source_config[config->num_source_sub_bins].type =
            NV_DS_SOURCE_URI;
        config->multi_source_config[config->num_source_sub_bins].uri =
            g_strdup_printf (config->multi_source_config[config->
                num_source_sub_bins].uri, j);
        config->num_source_sub_bins++;
      }
      config->multi_source_config[i].type = NV_DS_SOURCE_URI;
      config->multi_source_config[i].uri =
          g_strdup_printf (config->multi_source_config[i].uri, 0);
    }
  }

  ret = TRUE;
done:
  if (!ret) {
    cout <<  __func__ << " failed" << endl;
  }
  return ret;
}
