/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <iostream>
#include <string>
#include <cstring>
#include <cmath>
#include <algorithm>
#include "nvdsanalytics_property_yaml_parser.h"
#include <yaml-cpp/yaml.h>

GST_DEBUG_CATEGORY (NVDSANALYTICS_CFG_PARSER_YAML_CAT);

#define EXTRACT_STREAM_ID(for_group){\
      gchar *key1 = (gchar *)group_name.c_str() + sizeof (for_group) - 1; \
      gchar *endptr; \
      stream_index = g_ascii_strtoull (key1, &endptr, 10); \
}

#define PARSE_ERROR(details_fmt,...) \
  G_STMT_START { \
    GST_CAT_ERROR (NVDSANALYTICS_CFG_PARSER_YAML_CAT, \
        "Failed to parse config file %s: " details_fmt, \
        cfg_file_path, ##__VA_ARGS__); \
    GST_ELEMENT_ERROR (nvdsanalytics, LIBRARY, SETTINGS, \
        ("Failed to parse config file:%s", cfg_file_path), \
        (details_fmt, ##__VA_ARGS__)); \
    goto done; \
  } G_STMT_END

#define DSANALYTICS_PROPERTY "property"
#define DSANALYTICS_PROPERTY_ENABLE        "enable"
#define DSANALYTICS_PROPERTY_CONFIG_WIDTH  "config-width"
#define DSANALYTICS_PROPERTY_CONFIG_HEIGHT "config-height"
#define DSANALYTICS_PROPERTY_FONT_SIZE "display-font-size"
#define DSANALYTICS_PROPERTY_CLASS_ID         "class-id"
#define DSANALYTICS_PROPERTY_ROI              "roi-"
#define DSANALYTICS_PROPERTY_TIME_THRESHOLD   "time-threshold"
#define DSANALYTICS_PROPERTY_OBJECT_THRESHOLD "object-threshold"
#define DSANALYTICS_PROPERTY_MODE "mode"
#define DSANALYTICS_PROPERTY_OSD_MODE "osd-mode"
#define DSANALYTICS_PROPERTY_OBJ_CNT_WIN_MS "obj-cnt-win-in-ms"
#define DSANALYTICS_PROPERTY_DISPLAY_OBJ_CNT "display-obj-cnt"


#define DSANALYTICS_PROPERTY_GROUP_ROI_FILTERING       "roi-filtering-stream-"
#define DSANALYTICS_PROPERTY_GROUP_ROI_FILTERING_INVERSE_ROI "inverse-roi"
#define DSANALYTICS_PROPERTY_GROUP_OVERCROWDING        "overcrowding-stream-"
#define DSANALYTICS_PROPERTY_GROUP_LINE_CROSSING       "line-crossing-stream-"
#define DSANALYTICS_PROPERTY_GROUP_LINE_CROSSING_LC    "line-crossing-"

#define DSANALYTICS_PROPERTY_GROUP_LINE_CROSSING_EXTENDED "extended"
#define DSANALYTICS_PROPERTY_GROUP_DIRECTION_DETECTION "direction-detection-stream-"
#define DSANALYTICS_PROPERTY_GROUP_DIRECTION_DETECTION_DIRECTION "direction-"

static std::vector<std::string>
split_string (std::string input) {
  std::vector<int> positions;
  for(unsigned int i=0; i<input.size(); i++) {
    if(input[i] == ';')
      positions.push_back(i);
  }
  std::vector<std::string> ret;
  int prev = 0;
  for(auto &j: positions) {
    std::string temp = input.substr(prev,j - prev);
    ret.push_back(temp);
    prev = j + 1;
  }
  ret.push_back(input.substr(prev, input.size() - prev));
  return ret;
}

static gboolean
nvdsanalytics_parse_yaml_property_group (GstNvDsAnalytics *nvdsanalytics,
    gchar *cfg_file_path)
{
  g_autoptr(GError)error = nullptr;
  gboolean ret = FALSE;
  guint font_size = 12;
  guint osd_mode = 2;
  guint obj_cnt_win_in_ms = 0;
  YAML::Node config = YAML::LoadFile(cfg_file_path);

  if (config["property"]) {
    for(YAML::const_iterator itr = config["property"].begin();
      itr != config["property"].end(); ++itr) {
      std::string paramKey = itr->first.as<std::string>();
      if (paramKey == DSANALYTICS_PROPERTY_CONFIG_WIDTH) {
          itr->second.as<unsigned int>();
          nvdsanalytics->configuration_width =
             itr->second.as<unsigned int>();
      } else if (paramKey == DSANALYTICS_PROPERTY_ENABLE) {
          nvdsanalytics->enable = itr->second.as<gboolean>();
      } else if (paramKey == DSANALYTICS_PROPERTY_CONFIG_HEIGHT) {
          nvdsanalytics->configuration_height =
              itr->second.as<unsigned int>();
      } else if (paramKey == DSANALYTICS_PROPERTY_FONT_SIZE) {
          font_size = itr->second.as<unsigned int>();
          if (font_size){
              nvdsanalytics->font_size = font_size;
          }
      } else if (paramKey == DSANALYTICS_PROPERTY_OSD_MODE) {
          osd_mode = itr->second.as<unsigned int>();
          if (osd_mode > 2)
              osd_mode = 2;
      } else if (paramKey == DSANALYTICS_PROPERTY_OBJ_CNT_WIN_MS) {
          obj_cnt_win_in_ms = itr->second.as<unsigned int>();
          if ( obj_cnt_win_in_ms <1 || obj_cnt_win_in_ms > 1000000000)
              ret = FALSE;
      } else if (paramKey == DSANALYTICS_PROPERTY_DISPLAY_OBJ_CNT) {
          nvdsanalytics->display_obj_cnt = itr->second.as<gboolean>();
      }

      GST_CAT_INFO (NVDSANALYTICS_CFG_PARSER_YAML_CAT, "Parsed %s=%d, %s=%d, %s=%d in group '%s'\n",
          DSANALYTICS_PROPERTY_ENABLE, nvdsanalytics->enable,
          DSANALYTICS_PROPERTY_CONFIG_WIDTH, nvdsanalytics->configuration_width,
          DSANALYTICS_PROPERTY_CONFIG_HEIGHT, nvdsanalytics->configuration_height,
          DSANALYTICS_PROPERTY);

      ret = TRUE;
      nvdsanalytics->osd_mode = osd_mode;
      nvdsanalytics->obj_cnt_win_in_ms = obj_cnt_win_in_ms;
    }
  }
  return ret;
}

static gboolean
nvdsanalytics_parse_yaml_roi_filtering_group (GstNvDsAnalytics *nvdsanalytics,
    gchar *cfg_file_path, gchar *group, guint64 stream_id)
{
  gboolean ret = FALSE;
  gboolean enable = FALSE;
  //gint operate_on_class = -1;
  std::vector<gint> operate_on_class_vec;
  gboolean inverse_roi = FALSE;
  ROIInfo roi_info = {
    .enable = false,
    .roi_pts = {},
    .roi_label = "",
    .inverse_roi = false,
    .operate_on_class = {},
    .stream_id = 0
  };
  std::vector<ROIInfo> roi_vec;
  gsize list_len = 0;
  std::unordered_map<int, StreamInfo> *stream_analytics_info =
          (nvdsanalytics->stream_analytics_info);
  roi_info.stream_id = stream_id;

  YAML::Node config = YAML::LoadFile(cfg_file_path);

  std::string group_string = group;

  if (config[group_string]) {
    for(YAML::const_iterator itr = config[group_string].begin();
      itr != config[group_string].end(); ++itr) {
      std::string paramKey = itr->first.as<std::string>();
      if (!g_strcmp0(paramKey.c_str(), DSANALYTICS_PROPERTY_ENABLE)){
        if (config[group_string]["enable"])
         enable = itr->second.as<gboolean>();
        GST_CAT_INFO (NVDSANALYTICS_CFG_PARSER_YAML_CAT, "Parsed '%s=%d' in group '%s'\n",
        paramKey.c_str(), enable, group);
      }
      else if  (!g_strcmp0(paramKey.c_str(), DSANALYTICS_PROPERTY_CLASS_ID)){
        std::string values = itr->second.as<std::string>();
        std::vector<std::string> vec = split_string(std::move(values));
        list_len = vec.size();
        operate_on_class_vec.clear();
        for (gsize icnt = 0; icnt < list_len; icnt++){
         operate_on_class_vec.push_back(std::stoul(vec[icnt]));
           GST_CAT_INFO (NVDSANALYTICS_CFG_PARSER_YAML_CAT, "Parsed '%s=%s' in group '%s'\n",
           paramKey.c_str(), vec[icnt].c_str(), group);
        }
      }
      else if  (!g_strcmp0(paramKey.c_str(), DSANALYTICS_PROPERTY_GROUP_ROI_FILTERING_INVERSE_ROI)){
        if (config[group_string]["inverse-roi"])
          inverse_roi= itr->second.as<gboolean>();
        GST_CAT_INFO (NVDSANALYTICS_CFG_PARSER_YAML_CAT, "Parsed '%s=%d' in group '%s'\n",
          paramKey.c_str(), inverse_roi, group);
      }
      else if (!strncmp(paramKey.c_str(), DSANALYTICS_PROPERTY_ROI,
                                          sizeof(DSANALYTICS_PROPERTY_ROI)-1)){
        gchar * keywords = (gchar *)paramKey.c_str();
        size_t str_len = strlen(keywords);
        size_t prefix_len = sizeof(DSANALYTICS_PROPERTY_ROI)-1;
        size_t label_len = str_len - prefix_len + 1; // +1 for null terminator
        gchar* label = (gchar *)g_malloc(sizeof(gchar)*label_len);
        g_strlcpy(label, &keywords[prefix_len], label_len);
        roi_info.roi_label = label;
        std::string values = itr->second.as<std::string>();
        std::vector<std::string> vec = split_string(std::move(values));
        list_len = vec.size();
        //Check if the list is populated correctly
        if (list_len%2 !=0){
          ret = FALSE;
          g_free (label);
          goto done;
        }
        GST_CAT_INFO (NVDSANALYTICS_CFG_PARSER_YAML_CAT, "Parsed '%s' in group '%s'\n",
        keywords,  group);
        //FIXME: Handle multiple ROIs
        for (gsize icnt = 0; icnt < list_len; icnt+=2){
          GST_CAT_INFO (NVDSANALYTICS_CFG_PARSER_YAML_CAT, "pt-%lu x=%s y=%s' \n",
            icnt/2, vec[icnt].c_str(), vec[icnt+1].c_str());
          int x_value = std::stoi(vec[icnt]);
          int y_value = std::stoi(vec[icnt+1]);
          if (x_value < 0 || y_value < 0) {
            GST_CAT_INFO (NVDSANALYTICS_CFG_PARSER_YAML_CAT, "Parsed '%s' in group '%s'\n",
                keywords,  group);
            ret = FALSE;
            g_free (label);
            goto done;
          }
          roi_info.roi_pts.push_back(std::make_pair(x_value, y_value));
        }
        roi_vec.push_back(roi_info);
        roi_info.roi_pts.clear();
        g_free(label);
      }
      else {
      g_print ("Unknown key '%s' in group in '%s'\n", paramKey.c_str(), group);
      }
    }
  }
  if (roi_vec.size() == 0){
    PARSE_ERROR ("ROI not specified in group");

  }

  for (ROIInfo &roi: roi_vec){
    roi.enable = enable;
    roi.inverse_roi = inverse_roi;
    roi.operate_on_class = operate_on_class_vec;
  }
  if (stream_analytics_info->count(stream_id)==0){
    StreamInfo stream_specific_info = {
      .roi_info = {},
      .overcrowding_info = {},
      .linecrossing_info = {},
      .direction_info = {},
      .config_width = nvdsanalytics->configuration_width,
      .config_height = nvdsanalytics->configuration_height,
    };
    for (ROIInfo &roi: roi_vec){
      stream_specific_info.roi_info.push_back(roi);
    }
    //stream_analytics_info->insert(std::make_pair(stream_id, stream_specific_info));
    (*stream_analytics_info)[stream_id] = stream_specific_info;
  }
  else {
    StreamInfo &stream_specific_info = (*stream_analytics_info)[stream_id];
    for (ROIInfo &roi: roi_vec){
      stream_specific_info.roi_info.push_back(roi);
    }
  }

  ret = TRUE;

done:
  return ret;
}

static gboolean
nvdsanalytics_parse_yaml_overcrowding_group (GstNvDsAnalytics *nvdsanalytics,
    gchar *cfg_file_path, gchar *group, guint64 stream_id)
{
  gboolean ret = FALSE;
  gboolean enable = FALSE;
  std::vector<gint> operate_on_class_vec;
  gint object_threshold = 1;
  gint time_threshold_in_ms = 2000;
  OverCrowdingInfo oc_info = {
    .enable = false,
    .roi_pts = {},
    .oc_label = "",
    .operate_on_class = {},
    .stream_id = 0,
    .time_threshold_in_ms = 2000,
    .object_threshold = 1
  };
  std::vector<OverCrowdingInfo> oc_vec;
  gsize list_len = 0;
  std::unordered_map<int, StreamInfo> *stream_analytics_info =
          (nvdsanalytics->stream_analytics_info);
  oc_info.stream_id = stream_id;
  YAML::Node config = YAML::LoadFile(cfg_file_path);

  std::string group_string = group;

  if (config[group_string]) {
    for(YAML::const_iterator itr = config[group_string].begin();
      itr != config[group_string].end(); ++itr) {
      std::string paramKey = itr->first.as<std::string>();
      if (!g_strcmp0(paramKey.c_str(), DSANALYTICS_PROPERTY_ENABLE)){
        if (config[group_string]["enable"])
          enable = itr->second.as<gboolean>();
        GST_CAT_INFO (NVDSANALYTICS_CFG_PARSER_YAML_CAT, "Parsed '%s=%d' in group '%s'\n",
          paramKey.c_str(), enable, group);
      }
      else if  (!g_strcmp0(paramKey.c_str(), DSANALYTICS_PROPERTY_CLASS_ID)){
        std::string values = itr->second.as<std::string>();
        std::vector<std::string> vec = split_string(std::move(values));
        list_len = vec.size();
        operate_on_class_vec.clear();
        for (gsize icnt = 0; icnt < list_len; icnt++){
          operate_on_class_vec.push_back(std::stoul(vec[icnt]));
          GST_CAT_INFO (NVDSANALYTICS_CFG_PARSER_YAML_CAT, "Parsed '%s=%s' in group '%s'\n",
            paramKey.c_str(), vec[icnt].c_str(), group);
        }
      }
      else if  (!g_strcmp0(paramKey.c_str(), DSANALYTICS_PROPERTY_OBJECT_THRESHOLD)){
        object_threshold = itr->second.as<guint>();
        GST_CAT_INFO (NVDSANALYTICS_CFG_PARSER_YAML_CAT, "Parsed '%s=%d' in group '%s'\n",
          paramKey.c_str(), object_threshold, group);
      }
      else if  (!g_strcmp0(paramKey.c_str(), DSANALYTICS_PROPERTY_TIME_THRESHOLD)){
        time_threshold_in_ms = itr->second.as<guint>();
        GST_CAT_INFO (NVDSANALYTICS_CFG_PARSER_YAML_CAT, "Parsed '%s=%d' in group '%s'\n",
           paramKey.c_str(), time_threshold_in_ms, group);
      }
      else if (!strncmp(paramKey.c_str(), DSANALYTICS_PROPERTY_ROI,
                                          sizeof(DSANALYTICS_PROPERTY_ROI)-1)){
        gchar * keywords = (gchar *)paramKey.c_str();
        size_t str_len = strlen(keywords);
        size_t prefix_len = sizeof(DSANALYTICS_PROPERTY_ROI)-1;
        size_t label_len = str_len - prefix_len + 1; // +1 for null terminator
        gchar* label = (gchar *)g_malloc(sizeof(gchar)*label_len);
        g_strlcpy(label, &keywords[prefix_len], label_len);
        oc_info.oc_label = label;
        std::string values = itr->second.as<std::string>();
        std::vector<std::string> vec = split_string(std::move(values));
        list_len = vec.size();
        //FIXME: Handle multiple ROIs
        GST_CAT_INFO (NVDSANALYTICS_CFG_PARSER_YAML_CAT, "Parsed '%s' in group '%s'\n",
          paramKey.c_str(),  group);
        for (gsize icnt = 0; icnt < list_len; icnt+=2){
          int x_value = std::stoi(vec[icnt]);
          int y_value = std::stoi(vec[icnt+1]);
          GST_CAT_INFO (NVDSANALYTICS_CFG_PARSER_YAML_CAT, "%s-%lu x=%d y=%d' \n",
            paramKey.c_str(), icnt/2, x_value, y_value);
          if (x_value < 0 || y_value < 0) {
            GST_CAT_INFO (NVDSANALYTICS_CFG_PARSER_YAML_CAT, "Parsed '%s' in group '%s' fail\n",
                keywords,  group);
            ret = FALSE;
            g_free (label);
            goto done;
          }
          oc_info.roi_pts.push_back(std::make_pair(x_value, y_value));
        }
        oc_vec.push_back(oc_info);
        oc_info.roi_pts.clear();
        g_free (label);
      }
      else {
        g_print ("Unknown key '%s' in group in '%s'\n", paramKey.c_str(), group);
      }
    }
  }
  if (oc_vec.size() == 0){
    PARSE_ERROR ("ROI not specified in group");
  }

  for (OverCrowdingInfo &oc: oc_vec){
    oc.enable = enable;
    oc.object_threshold = object_threshold;
    oc.time_threshold_in_ms = time_threshold_in_ms;
    oc.operate_on_class = operate_on_class_vec;
  }
  if (stream_analytics_info->count(stream_id)==0){
    StreamInfo stream_specific_info = {
      .roi_info = {},
      .overcrowding_info = {},
      .linecrossing_info = {},
      .direction_info = {},
      .config_width = nvdsanalytics->configuration_width,
      .config_height = nvdsanalytics->configuration_height,
    };
    for (OverCrowdingInfo &oc: oc_vec){
      stream_specific_info.overcrowding_info.push_back(oc);
    }
    //stream_analytics_info->insert(std::make_pair(stream_id, stream_specific_info));
    (*stream_analytics_info)[stream_id] = stream_specific_info;
  }
  else {
    StreamInfo &stream_specific_info = (*stream_analytics_info)[stream_id];
    for (OverCrowdingInfo &oc: oc_vec){
      stream_specific_info.overcrowding_info.push_back(oc);
    }
  }


  ret = TRUE;

done:
  return ret;
}

static gboolean
nvdsanalytics_parse_yaml_direction_detection_group (GstNvDsAnalytics *nvdsanalytics,
    gchar *cfg_file_path, gchar *group, guint64 stream_id)
{
  gboolean ret = FALSE;
  gboolean enable = FALSE;
  std::vector<gint> operate_on_class_vec;
  DirectionInfo dir_info;
  std::vector<DirectionInfo> dir_vec;
  gchar *mode = nullptr;
  gsize list_len = 0;
  std::unordered_map<int, StreamInfo> *stream_analytics_info =
          (nvdsanalytics->stream_analytics_info);
  dir_info.stream_id = stream_id;
  dir_info.mode = eMode::balanced;
  YAML::Node config = YAML::LoadFile(cfg_file_path);

  std::string group_string = group;

  if (config[group_string]) {
    for(YAML::const_iterator itr = config[group_string].begin();
      itr != config[group_string].end(); ++itr) {
      std::string paramKey = itr->first.as<std::string>();
      if (!g_strcmp0(paramKey.c_str(), DSANALYTICS_PROPERTY_ENABLE)){
        if (config[group_string]["enable"])
          enable = itr->second.as<gboolean>();
        GST_CAT_INFO (NVDSANALYTICS_CFG_PARSER_YAML_CAT, "Parsed %s=%u in group '%s'\n",
          paramKey.c_str(), enable, group);
      }
      else if (!g_strcmp0(paramKey.c_str(), DSANALYTICS_PROPERTY_CLASS_ID)){
        std::string values = itr->second.as<std::string>();
        std::vector<std::string> vec = split_string(std::move(values));
        list_len = vec.size();
        operate_on_class_vec.clear();

        for (gsize icnt = 0; icnt < list_len; icnt++){
          operate_on_class_vec.push_back(std::stoul(vec[icnt]));
          GST_CAT_INFO(NVDSANALYTICS_CFG_PARSER_YAML_CAT, "Parsed '%s=%s' in group '%s'\n",
                     paramKey.c_str(), vec[icnt].c_str(), group);
        }
      }
      else if (!strncmp(paramKey.c_str(), DSANALYTICS_PROPERTY_GROUP_DIRECTION_DETECTION_DIRECTION,
                            sizeof(DSANALYTICS_PROPERTY_GROUP_DIRECTION_DETECTION_DIRECTION)-1)){
        gchar * keywords = (gchar *)paramKey.c_str();
        size_t str_len = strlen(keywords);
        size_t prefix_len = sizeof(DSANALYTICS_PROPERTY_GROUP_DIRECTION_DETECTION_DIRECTION)-1;
        size_t label_len = str_len - prefix_len + 1; // +1 for null terminator
        gchar* label = (gchar *)g_malloc(sizeof(gchar)*label_len);
        std::string values = itr->second.as<std::string>();
        std::vector<std::string> vec = split_string(std::move(values));
        list_len = vec.size();
        g_strlcpy(label, &keywords[prefix_len], label_len);
        gint x1,y1,x2,y2;
        //gfloat magxy=0.0f, xval=0.0f, yval=0.0f;

        dir_info.dir_label = label;
        //Check if the list is populated correctly
        if (list_len != 4 ){
            ret = FALSE;
            g_free (label);
            goto done;
        }

        for (gsize icnt = 0; icnt < list_len; icnt++){
          if (std::stoi(vec[icnt]) < 0) {
              ret = FALSE;
              g_free (label);
              goto done;
          }
        }
        // Direction vector
        x1 = std::stoi(vec[0]); y1 = std::stoi(vec[1]);
        x2 = std::stoi(vec[2]); y2 = std::stoi(vec[3]);
        GST_CAT_INFO (NVDSANALYTICS_CFG_PARSER_YAML_CAT, "Parsed '%s' in group '%s'\n",
          keywords,group);
        GST_CAT_INFO (NVDSANALYTICS_CFG_PARSER_YAML_CAT, "dir (x1=%d y1=%d) (x2=%d y2=%d) \n",
          x1, y1, x2, y2);
        dir_info.x1y1 = std::make_pair(x1,y1);
        dir_info.x2y2 = std::make_pair(x2,y2);
        std::vector<DirectionInfo>::iterator it = std::find_if(begin(dir_vec), end(dir_vec),
          [&dir_info](DirectionInfo &dir){if (dir.dir_label == dir_info.dir_label){return true;}
                                          return false;});
        if (it == dir_vec.end())
          dir_vec.push_back(dir_info);
        else
          *it = dir_info;
        g_free(label);
      }
      else if  (!g_strcmp0(paramKey.c_str(), DSANALYTICS_PROPERTY_MODE)){
        std::string str = itr->second.as<std::string>();
        mode = (gchar *) str.c_str();

        if (!strcmp(mode, "strict")){
          dir_info.mode = eMode::strict;
        }
        else if (!strcmp (mode, "balanced")){
          dir_info.mode = eMode::balanced;
        }
        else if (!strcmp (mode, "loose")){
          dir_info.mode = eMode::loose;
        }
        else {
          g_print("Unknown value '%s' in for key '%s' using 'balanced'\n",
            mode, paramKey.c_str());
          dir_info.mode = eMode::balanced;
        }
        mode=nullptr;
      }
      else {
        g_print ("Unknown key '%s' in group in '%s'\n", paramKey.c_str(), group);
      }
    }
  }
  if (dir_vec.size() == 0){
    g_print ("'%s' not specified in group '%s'",
        DSANALYTICS_PROPERTY_GROUP_DIRECTION_DETECTION_DIRECTION, group);
    goto done;
  }

  std::for_each(begin(dir_vec), end(dir_vec),
   [&](DirectionInfo &dir){dir.enable = enable; dir.operate_on_class = operate_on_class_vec;});

  if (stream_analytics_info->count(stream_id)==0){
    StreamInfo stream_specific_info = {
      .roi_info = {},
      .overcrowding_info = {},
      .linecrossing_info = {},
      .direction_info = {},
      .config_width = nvdsanalytics->configuration_width,
      .config_height = nvdsanalytics->configuration_height,
    };
    for (DirectionInfo &dir: dir_vec)
      stream_specific_info.direction_info.push_back(dir);

    //stream_analytics_info->insert(std::make_pair(stream_id, stream_specific_info));
    (*stream_analytics_info)[stream_id] = stream_specific_info;
  }
  else {
    StreamInfo &stream_specific_info = (*stream_analytics_info)[stream_id];
    for (DirectionInfo &dir: dir_vec)
      stream_specific_info.direction_info.push_back(dir);
  }


  ret = TRUE;

done:
  return ret;
}

static gboolean
nvdsanalytics_parse_yaml_linecrossing_group (GstNvDsAnalytics *nvdsanalytics,
    gchar *cfg_file_path, gchar *group, guint64 stream_id)
{
  g_autoptr(GError)error = nullptr;
  gboolean ret = FALSE;
  gboolean enable = FALSE;
  gboolean extended = TRUE;
  std::vector <gint>  operate_on_class_vec;
  LineCrossingInfo lc_info = {
    .enable = false,
    .extended = true,
    .lc_label = "",
    .lc_dir = std::make_pair(0.0, 0.0),
    .lc_info = {},
    .lcdir_pts = {},
    .operate_on_class = {},
    .stream_id = 0,
    .mode = eMode::loose,
    .mode_dir = eModeDir::use_dir
  };
  std::vector<LineCrossingInfo> lc_vec;
  gint *lc_list = nullptr;
  gsize list_len = 0;
  gchar *mode=nullptr;
  enum::eMode eMd = eMode::loose;
  std::unordered_map<int, StreamInfo> *stream_analytics_info =
          (nvdsanalytics->stream_analytics_info);
  lc_info.stream_id = stream_id;
  YAML::Node config = YAML::LoadFile(cfg_file_path);

  std::string group_string = group;

  if (config[group_string]) {
    for(YAML::const_iterator itr = config[group_string].begin();
      itr != config[group_string].end(); ++itr) {
      std::string paramKey = itr->first.as<std::string>();
      if (!g_strcmp0(paramKey.c_str(), DSANALYTICS_PROPERTY_ENABLE)){
        if (config[group_string]["enable"])
          enable = itr->second.as<gboolean>();
        GST_CAT_INFO (NVDSANALYTICS_CFG_PARSER_YAML_CAT, "Parsed '%s=%d' in group '%s'\n",
          paramKey.c_str(), enable, group);
      }
      else if (!g_strcmp0(paramKey.c_str(), DSANALYTICS_PROPERTY_GROUP_LINE_CROSSING_EXTENDED)){
        if (config[group_string]["extended"])
          extended = itr->second.as<gboolean>();
        GST_CAT_INFO (NVDSANALYTICS_CFG_PARSER_YAML_CAT, "Parsed '%s=%d' in group '%s'\n",
          paramKey.c_str(), extended, group);
      }
      else if  (!g_strcmp0(paramKey.c_str(), DSANALYTICS_PROPERTY_CLASS_ID)){
        std::string values = itr->second.as<std::string>();
        std::vector<std::string> vec = split_string(std::move(values));
        list_len = vec.size();

        operate_on_class_vec.clear();
        if (list_len) {
          for (gsize icnt = 0; icnt < list_len; icnt++){
            operate_on_class_vec.push_back(std::stoul(vec[icnt]));
            GST_CAT_INFO(NVDSANALYTICS_CFG_PARSER_YAML_CAT, "Parsed '%s=%s' in group '%s'\n",
              paramKey.c_str(), vec[icnt].c_str(), group);
          }
        }
      }
      else if (!strncmp(paramKey.c_str(), DSANALYTICS_PROPERTY_GROUP_LINE_CROSSING_LC,
        sizeof(DSANALYTICS_PROPERTY_GROUP_LINE_CROSSING_LC)-1)){
        gchar * keywords = (gchar *)paramKey.c_str();
        gint x1,y1,x2,y2;
        size_t str_len = strlen(keywords);
        size_t prefix_len = sizeof(DSANALYTICS_PROPERTY_GROUP_LINE_CROSSING_LC)-1;
        size_t label_len = str_len - prefix_len + 1; // +1 for null terminator
        gchar* label = (gchar *)g_malloc(sizeof(gchar)*label_len);
        std::string values = itr->second.as<std::string>();
        std::vector<std::string> vec = split_string(std::move(values));
        list_len = vec.size();
        g_strlcpy(label, &keywords[prefix_len], label_len);

        bool replace_old = false;
       // gfloat magxy=0.0f, xval=0.0f, yval=0.0f;

        lc_info.lc_label = label;
        //Check if the list is populated correctly
        if (list_len != 8){
          ret = FALSE;
          g_free (label);
          goto done;
        }
        for (gsize icnt = 0; icnt < list_len; icnt++){
          if (std::stoi(vec[icnt]) < 0) {
            ret = FALSE;
            g_free (label);
            goto done;
          }
        }
        GST_CAT_INFO (NVDSANALYTICS_CFG_PARSER_YAML_CAT, "Parsed '%s' in group '%s'\n",
          paramKey.c_str(),  group);
        // Direction vector
        x1 = std::stoi(vec[0]); y1 = std::stoi(vec[1]);
        x2 = std::stoi(vec[2]); y2 = std::stoi(vec[3]);
        GST_CAT_INFO (NVDSANALYTICS_CFG_PARSER_YAML_CAT, "dir (x1=%d y1=%d) (x2=%d y2=%d) \n",
          x1, y1, x2, y2);
        lc_info.lcdir_pts.push_back(std::make_pair(x1, y1));
        lc_info.lcdir_pts.push_back(std::make_pair(x2, y2));
        // LC vector
        x1 = std::stoi(vec[4]); y1 = std::stoi(vec[5]);
        x2 = std::stoi(vec[6]); y2 = std::stoi(vec[7]);
        lc_info.lcdir_pts.push_back(std::make_pair(x1,y1));
        lc_info.lcdir_pts.push_back(std::make_pair(x2,y2));

        GST_CAT_INFO (NVDSANALYTICS_CFG_PARSER_YAML_CAT, "lc (x1=%d y1=%d) (x2=%d y2=%d) \n",
          x1, y1, x2, y2);
        replace_old = false;
        //find_if iterator instead
        for (auto &lc: lc_vec){
          if (lc.lc_label == lc_info.lc_label){
            replace_old = true;
            lc = lc_info;
            break;
          }
        }
        if (false == replace_old){
         lc_vec.push_back(lc_info);
        }
        lc_info.lc_info.clear();
        lc_info.lcdir_pts.clear();
        g_free(label);
      }
      else if (!g_strcmp0(paramKey.c_str(), DSANALYTICS_PROPERTY_MODE))    {
        std::string str = itr->second.as<std::string>();
        mode = (gchar *) str.c_str();

        if (!strcmp(mode, "strict"))
          eMd = eMode::strict;
        else if (!strcmp (mode, "balanced"))
          eMd = eMode::balanced;
        else if (!strcmp (mode, "loose"))
          eMd = eMode::loose;
        else {
          g_print("Unknown value '%s' in for key '%s' using 'loose'\n", mode, paramKey.c_str());
          eMd = eMode::loose;
        }
        mode = nullptr;
      }
      else
        g_print ("Unknown key '%s' in group in '%s'\n", paramKey.c_str(), group);
      }
      if (lc_vec.size() == 0){
        PARSE_ERROR ("%s not specified in group '%s'", DSANALYTICS_PROPERTY_GROUP_LINE_CROSSING_LC, group);
      }

      for (LineCrossingInfo &lc: lc_vec){
        lc.enable = enable;
        lc.extended = extended;
        lc.operate_on_class = operate_on_class_vec;
        lc.mode = eMd;
      }

      if (stream_analytics_info->count(stream_id)==0){
        StreamInfo stream_specific_info = {
          .roi_info = {},
          .overcrowding_info = {},
          .linecrossing_info = {},
          .direction_info = {},
          .config_width = nvdsanalytics->configuration_width,
          .config_height = nvdsanalytics->configuration_height,
        };
      for (LineCrossingInfo &lc: lc_vec)
        stream_specific_info.linecrossing_info.push_back(lc);

      //stream_analytics_info->insert(std::make_pair(stream_id, stream_specific_info));
      (*stream_analytics_info)[stream_id] = stream_specific_info;
    }
    else {
      StreamInfo &stream_specific_info = (*stream_analytics_info)[stream_id];
      for (LineCrossingInfo &lc: lc_vec)
        stream_specific_info.linecrossing_info.push_back(lc);
    }
    ret = TRUE;
  }

done:
    g_free(lc_list);
    return ret;
}

//G_DEFINE_AUTO_CLEANUP_FREE_FUNC(GStrv, g_strfreev, nullptr);
/* Parse the nvdsanalytics config file. Returns FALSE in case of an error. */
gboolean
nvdsanalytics_parse_yaml_config_file (GstNvDsAnalytics * nvdsanalytics, gchar * cfg_file_path)
{
  gboolean ret = TRUE;
  std::string paramKey = "";
  int total_size = 0;
  guint64 stream_index = 0;
  gboolean property_present = FALSE;

  YAML::Node configyml = YAML::LoadFile(cfg_file_path);
  total_size = configyml.size();
  if(!(total_size > 0))  {
  	std::cout << "Can't open config file (" << cfg_file_path << ")" << std::endl;
    return FALSE;
  }

  if(!nvdsanalytics)
      goto done;

  nvdsanalytics->stream_analytics_info->clear();
  if (!NVDSANALYTICS_CFG_PARSER_YAML_CAT) {
    GstDebugLevel  level;
    GST_DEBUG_CATEGORY_INIT (NVDSANALYTICS_CFG_PARSER_YAML_CAT, "nvdsanalytics", 0,
        NULL);
    level = gst_debug_category_get_threshold (NVDSANALYTICS_CFG_PARSER_YAML_CAT);
    if (level < GST_LEVEL_ERROR )
      gst_debug_category_set_threshold (NVDSANALYTICS_CFG_PARSER_YAML_CAT, GST_LEVEL_ERROR);
  }

  for(YAML::const_iterator itr = configyml.begin(); itr != configyml.end(); ++itr) {
      std::string group_name = itr->first.as<std::string>();

      GST_CAT_INFO (NVDSANALYTICS_CFG_PARSER_YAML_CAT, "Group found %s \n", group_name.c_str());
      if (!strcmp(group_name.c_str(), DSANALYTICS_PROPERTY)){
        property_present = nvdsanalytics_parse_yaml_property_group(nvdsanalytics,
             cfg_file_path);
      }else
      if (!strncmp(group_name.c_str(), DSANALYTICS_PROPERTY_GROUP_ROI_FILTERING,
          sizeof(DSANALYTICS_PROPERTY_GROUP_ROI_FILTERING)-1)){

          EXTRACT_STREAM_ID(DSANALYTICS_PROPERTY_GROUP_ROI_FILTERING);
          if (!nvdsanalytics_parse_yaml_roi_filtering_group (nvdsanalytics,
                 cfg_file_path, (gchar *)group_name.c_str(), stream_index)){
              goto done;
          }
      }else
      if (!strncmp(group_name.c_str(), DSANALYTICS_PROPERTY_GROUP_OVERCROWDING,
            sizeof(DSANALYTICS_PROPERTY_GROUP_OVERCROWDING)-1)){
        EXTRACT_STREAM_ID(DSANALYTICS_PROPERTY_GROUP_OVERCROWDING);
        if (!nvdsanalytics_parse_yaml_overcrowding_group (nvdsanalytics,
              cfg_file_path, (gchar *)group_name.c_str(), stream_index)){
          goto done;
        }

      }else
    if (!strncmp(group_name.c_str(), DSANALYTICS_PROPERTY_GROUP_DIRECTION_DETECTION,
          sizeof(DSANALYTICS_PROPERTY_GROUP_DIRECTION_DETECTION)-1)){
      EXTRACT_STREAM_ID(DSANALYTICS_PROPERTY_GROUP_DIRECTION_DETECTION);
      if (!nvdsanalytics_parse_yaml_direction_detection_group (nvdsanalytics,
            cfg_file_path, (gchar *)group_name.c_str(), stream_index)){
        goto done;
      }
    }else
    if (!strncmp(group_name.c_str(), DSANALYTICS_PROPERTY_GROUP_LINE_CROSSING,
          sizeof(DSANALYTICS_PROPERTY_GROUP_LINE_CROSSING)-1)){
      EXTRACT_STREAM_ID(DSANALYTICS_PROPERTY_GROUP_LINE_CROSSING);

      if (!nvdsanalytics_parse_yaml_linecrossing_group (nvdsanalytics,
            cfg_file_path, (gchar *)group_name.c_str(), stream_index)){
        goto done;
      }
    }
    else {
      g_print("NVDSANALYTICS_CFG_PARSER: Group '%s' ignored\n", group_name.c_str());

    }
  }
  if (FALSE == property_present){
    ret = FALSE;
  }
  else {
    ret = TRUE;
  }

done:
  return ret;
}
