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
#include "nvdsanalytics_property_parser.h"

GST_DEBUG_CATEGORY (NVDSANALYTICS_CFG_PARSER_CAT);

#define PARSE_ERROR(details_fmt,...) \
  G_STMT_START { \
    GST_CAT_ERROR (NVDSANALYTICS_CFG_PARSER_CAT, \
        "Failed to parse config file %s: " details_fmt, \
        cfg_file_path, ##__VA_ARGS__); \
    GST_ELEMENT_ERROR (nvdsanalytics, LIBRARY, SETTINGS, \
        ("Failed to parse config file:%s", cfg_file_path), \
        (details_fmt, ##__VA_ARGS__)); \
    goto done; \
  } G_STMT_END

#define CHECK_IF_PRESENT(error, custom_err) \
  G_STMT_START { \
    if (error && error->code != G_KEY_FILE_ERROR_KEY_NOT_FOUND) { \
      std::string errvalue = "Error while setting property, in group ";  \
      errvalue.append(custom_err); \
      PARSE_ERROR ("%s %s", errvalue.c_str(), error->message); \
    } \
  } G_STMT_END

#define CHECK_ERROR(error, custom_err) \
  G_STMT_START { \
    if (error) { \
      std::string errvalue = "Error while setting property, in group ";  \
      errvalue.append(custom_err); \
      PARSE_ERROR ("%s %s", errvalue.c_str(), error->message); \
    } \
  } G_STMT_END

#define CHECK_BOOLEAN_VALUE(prop_name,value) \
  G_STMT_START { \
    if ((gint) value < 0 || value > 1) { \
      PARSE_ERROR ("Boolean property '%s' can have values 0 or 1", prop_name); \
    } \
  } G_STMT_END

#define CHECK_INT_VALUE_NON_NEGATIVE(prop_name,value, group) \
  G_STMT_START { \
    if ((gint) value < 0) { \
      PARSE_ERROR ("Integer property '%s' in group '%s' can have value >=0", prop_name, group); \
    } \
  } G_STMT_END

#define CHECK_INT_VALUE_RANGE(prop_name,value, group, min, max) \
  G_STMT_START { \
    if ((gint) value < min || (gint)value > max) { \
      PARSE_ERROR ("Integer property '%s' in group '%s' can have value >=%d and <=%d", \
      prop_name, group, min, max); \
    } \
  } G_STMT_END


#define GET_BOOLEAN_PROPERTY(group, property, field) {\
  field = g_key_file_get_boolean(key_file, group, property, &error); \
  CHECK_ERROR(error, group); \
}

#define GET_UINT_PROPERTY(group, property, field) {\
  field = g_key_file_get_integer(key_file, group, property, &error); \
  CHECK_ERROR(error, group); \
  CHECK_INT_VALUE_NON_NEGATIVE(property,\
                               field, group);\
}

#define READ_UINT_PROPERTY(group, property, field) {\
  field = g_key_file_get_integer(key_file, group, property, &error); \
  CHECK_ERROR(error, group); \
  CHECK_INT_VALUE_NON_NEGATIVE(property,\
                               field, group);\
}


#define GET_INT_PROPERTY(group, property, field) {\
  field = g_key_file_get_integer(key_file, group, property, &error); \
  CHECK_ERROR(error, group); \
}

#define GET_AND_PARSE_STRING_PROPERTY(group, property, field) {\
  field = g_key_file_get_integer(key_file, group, property, &error); \
  CHECK_ERROR(error, group); \
}



#define EXTRACT_STREAM_ID(for_group){\
      gchar *key1 = *group + sizeof (for_group) - 1; \
      gchar *endptr; \
      stream_index = g_ascii_strtoull (key1, &endptr, 10); \
}



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

static gboolean
nvdsanalytics_parse_overcrowding_group (GstNvDsAnalytics * nvdsanalytics,
    gchar * cfg_file_path, GKeyFile * key_file, gchar * group,
    guint64 stream_id);

static gboolean
nvdsanalytics_parse_property_group (GstNvDsAnalytics * nvdsanalytics,
    gchar * cfg_file_path, GKeyFile * key_file);

static gboolean
nvdsanalytics_parse_direction_detection_group (GstNvDsAnalytics * nvdsanalytics,
    gchar * cfg_file_path, GKeyFile * key_file, gchar * group,
    guint64 stream_id);

static gboolean
nvdsanalytics_parse_linecrossing_group (GstNvDsAnalytics * nvdsanalytics,
    gchar * cfg_file_path, GKeyFile * key_file, gchar * group,
    guint64 stream_id);

static gboolean
nvdsanalytics_parse_roi_filtering_group (GstNvDsAnalytics * nvdsanalytics,
    gchar * cfg_file_path, GKeyFile * key_file, gchar * group,
    guint64 stream_id);

static gboolean
nvdsanalytics_parse_property_group (GstNvDsAnalytics * nvdsanalytics,
    gchar * cfg_file_path, GKeyFile * key_file)
{
  g_autoptr (GError) error = nullptr;
  gboolean ret = FALSE;
  guint font_size = 12;
  guint osd_mode = 2;
  guint obj_cnt_win_in_ms = 0;

  GET_UINT_PROPERTY (DSANALYTICS_PROPERTY,
      DSANALYTICS_PROPERTY_CONFIG_WIDTH, nvdsanalytics->configuration_width)

      GET_BOOLEAN_PROPERTY (DSANALYTICS_PROPERTY,
      DSANALYTICS_PROPERTY_ENABLE, nvdsanalytics->enable);

  GET_UINT_PROPERTY (DSANALYTICS_PROPERTY,
      DSANALYTICS_PROPERTY_CONFIG_HEIGHT, nvdsanalytics->configuration_height)

      font_size = g_key_file_get_integer (key_file, DSANALYTICS_PROPERTY,
      DSANALYTICS_PROPERTY_FONT_SIZE, &error);
  CHECK_IF_PRESENT (error, DSANALYTICS_PROPERTY);
  if (error) {
    g_error_free (error);
    error = nullptr;
  }
  if (font_size) {
    nvdsanalytics->font_size = font_size;
  }
  osd_mode = g_key_file_get_integer (key_file, DSANALYTICS_PROPERTY,
      DSANALYTICS_PROPERTY_OSD_MODE, &error);
  CHECK_IF_PRESENT (error, DSANALYTICS_PROPERTY);
  if (!error) {
    CHECK_INT_VALUE_RANGE (DSANALYTICS_PROPERTY_OSD_MODE, osd_mode,
        DSANALYTICS_PROPERTY, 0, 2);
  } else {
    osd_mode = 2;
  }
  if (error) {
    g_error_free (error);
    error = nullptr;
  }
  obj_cnt_win_in_ms = g_key_file_get_integer (key_file, DSANALYTICS_PROPERTY,
      DSANALYTICS_PROPERTY_OBJ_CNT_WIN_MS, &error);
  CHECK_IF_PRESENT (error, DSANALYTICS_PROPERTY);
  if (!error) {
    CHECK_INT_VALUE_RANGE (DSANALYTICS_PROPERTY_OBJ_CNT_WIN_MS, obj_cnt_win_in_ms,
        DSANALYTICS_PROPERTY, 1, 1000000000);
  }

  if (error) {
    g_error_free (error);
    error = nullptr;
  }
  nvdsanalytics->display_obj_cnt =
      g_key_file_get_boolean (key_file, DSANALYTICS_PROPERTY,
      DSANALYTICS_PROPERTY_DISPLAY_OBJ_CNT, &error);
  CHECK_IF_PRESENT (error, DSANALYTICS_PROPERTY);
  if (error) {
    g_error_free (error);
    error = nullptr;
  }

  GST_CAT_INFO (NVDSANALYTICS_CFG_PARSER_CAT,
      "Parsed %s=%d, %s=%d, %s=%d in group '%s'\n", DSANALYTICS_PROPERTY_ENABLE,
      nvdsanalytics->enable, DSANALYTICS_PROPERTY_CONFIG_WIDTH,
      nvdsanalytics->configuration_width, DSANALYTICS_PROPERTY_CONFIG_HEIGHT,
      nvdsanalytics->configuration_height, DSANALYTICS_PROPERTY);

  ret = TRUE;
  nvdsanalytics->osd_mode = osd_mode;
  nvdsanalytics->obj_cnt_win_in_ms = obj_cnt_win_in_ms;

done:
  return ret;
}

static gboolean
nvdsanalytics_parse_direction_detection_group (GstNvDsAnalytics * nvdsanalytics,
    gchar * cfg_file_path, GKeyFile * key_file, gchar * group,
    guint64 stream_id)
{
  g_autoptr (GError) error = nullptr;
  gboolean ret = FALSE;
  gboolean enable = FALSE;
  std::vector < gint > operate_on_class_vec;
  DirectionInfo dir_info;
  std::vector < DirectionInfo > dir_vec;
  gint *dir_list = nullptr;
  gchar *mode = nullptr;
  gsize list_len = 0;
  g_auto (GStrv) keys = nullptr;
  GStrv key = nullptr;
  std::unordered_map < int, StreamInfo > *stream_analytics_info =
      (nvdsanalytics->stream_analytics_info);
  dir_info.stream_id = stream_id;
  dir_info.mode = eMode::balanced;

  keys = g_key_file_get_keys (key_file, group, nullptr, &error);
  CHECK_ERROR (error, group);

  for (key = keys; *key; key++) {
    if (!g_strcmp0 (*key, DSANALYTICS_PROPERTY_ENABLE)) {
      enable = g_key_file_get_boolean (key_file, group, *key, &error);
      CHECK_ERROR (error, group);
      GST_CAT_INFO (NVDSANALYTICS_CFG_PARSER_CAT,
          "Parsed %s=%u in group '%s'\n", *key, enable, group);
    } else if (!g_strcmp0 (*key, DSANALYTICS_PROPERTY_CLASS_ID)) {
      dir_list = g_key_file_get_integer_list (key_file, group, *key,
          &list_len, &error);
      //Check if the list is populated correctly
      if ((dir_list == nullptr)) {
        CHECK_ERROR (error, group);
      }

      operate_on_class_vec.clear ();
      for (gsize icnt = 0; icnt < list_len; icnt++) {
        operate_on_class_vec.push_back (dir_list[icnt]);
        GST_CAT_INFO (NVDSANALYTICS_CFG_PARSER_CAT,
            "Parsed '%s=%d' in group '%s'\n", *key, dir_list[icnt], group);
      }
      g_free (dir_list);
      dir_list = nullptr;

    } else if (!strncmp (*key,
            DSANALYTICS_PROPERTY_GROUP_DIRECTION_DETECTION_DIRECTION,
            sizeof (DSANALYTICS_PROPERTY_GROUP_DIRECTION_DETECTION_DIRECTION) -
            1)) {
      gchar *label =
          &key[0][sizeof
          (DSANALYTICS_PROPERTY_GROUP_DIRECTION_DETECTION_DIRECTION) - 1];
      gint x1, y1, x2, y2;
      //gfloat magxy=0.0f, xval=0.0f, yval=0.0f;

      dir_info.dir_label = label;
      dir_list = g_key_file_get_integer_list (key_file, group, *key,
          &list_len, &error);
      //Check if the list is populated correctly
      if ((dir_list == nullptr) || (list_len != 8)) {
        CHECK_ERROR (error, group);
      }
      for (gsize icnt = 0; icnt < list_len; icnt++) {
        CHECK_INT_VALUE_NON_NEGATIVE (*key, dir_list[icnt], group);
      }
      // Direction vector
      x1 = dir_list[0];
      y1 = dir_list[1];
      x2 = dir_list[2];
      y2 = dir_list[3];
      GST_CAT_INFO (NVDSANALYTICS_CFG_PARSER_CAT, "Parsed '%s' in group '%s'\n",
          *key, group);
      GST_CAT_INFO (NVDSANALYTICS_CFG_PARSER_CAT,
          "dir (x1=%d y1=%d) (x2=%d y2=%d) \n", x1, y1, x2, y2);
      dir_info.x1y1 = std::make_pair (x1, y1);
      dir_info.x2y2 = std::make_pair (x2, y2);
      std::vector < DirectionInfo >::iterator it =
          std::find_if (begin (dir_vec), end (dir_vec),
          [&dir_info] (DirectionInfo & dir) {
            if (dir.dir_label == dir_info.dir_label) {
            return true;}
          return false;}
      );
      if (it == dir_vec.end ())
        dir_vec.push_back (dir_info);
      else
        *it = dir_info;

      g_free (dir_list);
      dir_list = nullptr;
    } else if (!g_strcmp0 (*key, DSANALYTICS_PROPERTY_MODE)) {
      mode = g_key_file_get_string (key_file, group, *key, &error);

      if (!strcmp (mode, "strict")) {
        dir_info.mode = eMode::strict;
      } else if (!strcmp (mode, "balanced")) {
        dir_info.mode = eMode::balanced;
      } else if (!strcmp (mode, "loose")) {
        dir_info.mode = eMode::loose;
      } else {
        g_print ("Unknown value '%s' in for key '%s' using 'balanced'\n", mode,
            *key);
        dir_info.mode = eMode::balanced;
      }
      g_free (mode);
      mode = nullptr;
      CHECK_ERROR (error, group);
    } else {
      g_print ("Unknown key '%s' in group in '%s'\n", *key, group);
    }
  }
  if (dir_vec.size () == 0) {
    PARSE_ERROR ("'%s' not specified in group '%s'",
        DSANALYTICS_PROPERTY_GROUP_DIRECTION_DETECTION_DIRECTION, group);
  }

  std::for_each (begin (dir_vec), end (dir_vec),[&](DirectionInfo & dir) {
        dir.enable = enable;
        dir.operate_on_class = operate_on_class_vec;
      }
  );

  if (stream_analytics_info->count (stream_id) == 0) {
    StreamInfo stream_specific_info = {
      .roi_info = {},
      .overcrowding_info = {},
      .linecrossing_info = {},
      .direction_info = {},
      .config_width = nvdsanalytics->configuration_width,
      .config_height = nvdsanalytics->configuration_height,
    };
  for (DirectionInfo & dir:dir_vec)
      stream_specific_info.direction_info.push_back (dir);

    //stream_analytics_info->insert(std::make_pair(stream_id, stream_specific_info));
    (*stream_analytics_info)[stream_id] = stream_specific_info;
  } else {
    StreamInfo & stream_specific_info = (*stream_analytics_info)[stream_id];
  for (DirectionInfo & dir:dir_vec)
      stream_specific_info.direction_info.push_back (dir);
  }


  ret = TRUE;

done:
  g_free (dir_list);
  return ret;
}


static gboolean
nvdsanalytics_parse_linecrossing_group (GstNvDsAnalytics * nvdsanalytics,
    gchar * cfg_file_path, GKeyFile * key_file, gchar * group,
    guint64 stream_id)
{
  g_autoptr (GError) error = nullptr;
  gboolean ret = FALSE;
  gboolean enable = FALSE;
  gboolean extended = TRUE;
  std::vector < gint > operate_on_class_vec;
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
  std::vector < LineCrossingInfo > lc_vec;
  gint *lc_list = nullptr;
  gsize list_len = 0;
  g_auto (GStrv) keys = nullptr;
  GStrv key = nullptr;
  gchar *mode = nullptr;
  enum::eMode eMd = eMode::loose;
  std::unordered_map < int, StreamInfo > *stream_analytics_info =
      (nvdsanalytics->stream_analytics_info);
  lc_info.stream_id = stream_id;

  keys = g_key_file_get_keys (key_file, group, nullptr, &error);
  CHECK_ERROR (error, group);

  for (key = keys; *key; key++) {
    if (!g_strcmp0 (*key, DSANALYTICS_PROPERTY_ENABLE)) {
      enable = g_key_file_get_boolean (key_file, group, *key, &error);
      CHECK_ERROR (error, group);
      GST_CAT_INFO (NVDSANALYTICS_CFG_PARSER_CAT,
          "Parsed '%s=%d' in group '%s'\n", *key, enable, group);
    } else if (!g_strcmp0 (*key,
            DSANALYTICS_PROPERTY_GROUP_LINE_CROSSING_EXTENDED)) {
      extended = g_key_file_get_boolean (key_file, group, *key, &error);
      CHECK_ERROR (error, group);
      GST_CAT_INFO (NVDSANALYTICS_CFG_PARSER_CAT,
          "Parsed '%s=%d' in group '%s'\n", *key, extended, group);
    } else if (!g_strcmp0 (*key, DSANALYTICS_PROPERTY_CLASS_ID)) {
      lc_list = g_key_file_get_integer_list (key_file, group, *key,
          &list_len, &error);
      //Check if the list is populated correctly
      if ((lc_list == nullptr)) {
        CHECK_ERROR (error, group);
      }

      operate_on_class_vec.clear ();
      for (gsize icnt = 0; icnt < list_len; icnt++) {
        operate_on_class_vec.push_back (lc_list[icnt]);
        GST_CAT_INFO (NVDSANALYTICS_CFG_PARSER_CAT,
            "Parsed '%s=%d' in group '%s'\n", *key, lc_list[icnt], group);
      }
      g_free (lc_list);
      lc_list = nullptr;
    } else if (!strncmp (*key, DSANALYTICS_PROPERTY_GROUP_LINE_CROSSING_LC,
            sizeof (DSANALYTICS_PROPERTY_GROUP_LINE_CROSSING_LC) - 1)) {
      gchar *label =
          &key[0][sizeof (DSANALYTICS_PROPERTY_GROUP_LINE_CROSSING_LC) - 1];
      gint x1, y1, x2, y2;
      bool replace_old = false;
      // gfloat magxy=0.0f, xval=0.0f, yval=0.0f;

      lc_info.lc_label = label;
      lc_list = g_key_file_get_integer_list (key_file, group, *key,
          &list_len, &error);
      //Check if the list is populated correctly
      if ((lc_list == nullptr) || (list_len != 8)) {
        CHECK_ERROR (error, group);
      }
      for (gsize icnt = 0; icnt < list_len; icnt++) {
        CHECK_INT_VALUE_NON_NEGATIVE (*key, lc_list[icnt], group);
      }
      GST_CAT_INFO (NVDSANALYTICS_CFG_PARSER_CAT, "Parsed '%s' in group '%s'\n",
          *key, group);
      // Direction vector
      x1 = lc_list[0];
      y1 = lc_list[1];
      x2 = lc_list[2];
      y2 = lc_list[3];
      GST_CAT_INFO (NVDSANALYTICS_CFG_PARSER_CAT,
          "dir (x1=%d y1=%d) (x2=%d y2=%d) \n", x1, y1, x2, y2);
      lc_info.lcdir_pts.push_back (std::make_pair (x1, y1));
      lc_info.lcdir_pts.push_back (std::make_pair (x2, y2));
      // LC vector
      x1 = lc_list[4];
      y1 = lc_list[5];
      x2 = lc_list[6];
      y2 = lc_list[7];
      lc_info.lcdir_pts.push_back (std::make_pair (x1, y1));
      lc_info.lcdir_pts.push_back (std::make_pair (x2, y2));

      GST_CAT_INFO (NVDSANALYTICS_CFG_PARSER_CAT,
          "lc (x1=%d y1=%d) (x2=%d y2=%d) \n", x1, y1, x2, y2);
      replace_old = false;
      //find_if iterator instead
    for (auto & lc:lc_vec) {
        if (lc.lc_label == lc_info.lc_label) {
          replace_old = true;
          lc = lc_info;
          break;
        }
      }
      if (false == replace_old) {
        lc_vec.push_back (lc_info);
      }
      lc_info.lc_info.clear ();
      lc_info.lcdir_pts.clear ();
      g_free (lc_list);
      lc_list = nullptr;
    } else if (!g_strcmp0 (*key, DSANALYTICS_PROPERTY_MODE)) {
      mode = g_key_file_get_string (key_file, group, *key, &error);

      if (!strcmp (mode, "strict"))
        eMd = eMode::strict;
      else if (!strcmp (mode, "balanced"))
        eMd = eMode::balanced;
      else if (!strcmp (mode, "loose"))
        eMd = eMode::loose;
      else {
        g_print ("Unknown value '%s' in for key '%s' using 'loose'\n", mode,
            *key);
        eMd = eMode::loose;
      }
      g_free (mode);
      mode = nullptr;
      CHECK_ERROR (error, group);
    } else
      g_print ("Unknown key '%s' in group in '%s'\n", *key, group);
  }
  if (lc_vec.size () == 0) {
    PARSE_ERROR ("%s not specified in group '%s'",
        DSANALYTICS_PROPERTY_GROUP_LINE_CROSSING_LC, group);
  }

for (LineCrossingInfo & lc:lc_vec) {
    lc.enable = enable;
    lc.extended = extended;
    lc.operate_on_class = operate_on_class_vec;
    lc.mode = eMd;
  }

  if (stream_analytics_info->count (stream_id) == 0) {
    StreamInfo stream_specific_info = {
      .roi_info = {},
      .overcrowding_info = {},
      .linecrossing_info = {},
      .direction_info = {},
      .config_width = nvdsanalytics->configuration_width,
      .config_height = nvdsanalytics->configuration_height,
    };
  for (LineCrossingInfo & lc:lc_vec)
      stream_specific_info.linecrossing_info.push_back (lc);

    //stream_analytics_info->insert(std::make_pair(stream_id, stream_specific_info));
    (*stream_analytics_info)[stream_id] = stream_specific_info;
  } else {
    StreamInfo & stream_specific_info = (*stream_analytics_info)[stream_id];
  for (LineCrossingInfo & lc:lc_vec)
      stream_specific_info.linecrossing_info.push_back (lc);
  }
  ret = TRUE;

done:
  g_free (lc_list);
  return ret;
}

static gboolean
nvdsanalytics_parse_overcrowding_group (GstNvDsAnalytics * nvdsanalytics,
    gchar * cfg_file_path, GKeyFile * key_file, gchar * group,
    guint64 stream_id)
{
  g_autoptr (GError) error = nullptr;
  gboolean ret = FALSE;
  gboolean enable = FALSE;
  std::vector < gint > operate_on_class_vec;
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
  std::vector < OverCrowdingInfo > oc_vec;
  gint *roi_list = nullptr;
  gsize list_len = 0;
  g_auto (GStrv) keys = nullptr;
  GStrv key = nullptr;
  std::unordered_map < int, StreamInfo > *stream_analytics_info =
      (nvdsanalytics->stream_analytics_info);
  oc_info.stream_id = stream_id;

  keys = g_key_file_get_keys (key_file, group, nullptr, &error);
  CHECK_ERROR (error, group);

  for (key = keys; *key; key++) {
    if (!g_strcmp0 (*key, DSANALYTICS_PROPERTY_ENABLE)) {
      enable = g_key_file_get_boolean (key_file, group, *key, &error);
      CHECK_ERROR (error, group);
      GST_CAT_INFO (NVDSANALYTICS_CFG_PARSER_CAT,
          "Parsed '%s=%d' in group '%s'\n", *key, enable, group);
    } else if (!g_strcmp0 (*key, DSANALYTICS_PROPERTY_CLASS_ID)) {
      roi_list = g_key_file_get_integer_list (key_file, group, *key,
          &list_len, &error);
      //Check if the list is populated correctly
      if ((roi_list == nullptr)) {
        CHECK_ERROR (error, group);
      }
      operate_on_class_vec.clear ();
      for (gsize icnt = 0; icnt < list_len; icnt++) {
        operate_on_class_vec.push_back (roi_list[icnt]);
        GST_CAT_INFO (NVDSANALYTICS_CFG_PARSER_CAT,
            "Parsed '%s=%d' in group '%s'\n", *key, roi_list[icnt], group);
      }
      g_free (roi_list);
      roi_list = nullptr;
    } else if (!g_strcmp0 (*key, DSANALYTICS_PROPERTY_OBJECT_THRESHOLD)) {
      READ_UINT_PROPERTY (group, *key, object_threshold);
      GST_CAT_INFO (NVDSANALYTICS_CFG_PARSER_CAT,
          "Parsed '%s=%d' in group '%s'\n", *key, object_threshold, group);
    } else if (!g_strcmp0 (*key, DSANALYTICS_PROPERTY_TIME_THRESHOLD)) {
      READ_UINT_PROPERTY (group, *key, time_threshold_in_ms);
      GST_CAT_INFO (NVDSANALYTICS_CFG_PARSER_CAT,
          "Parsed '%s=%d' in group '%s'\n", *key, time_threshold_in_ms, group);
    } else if (!strncmp (*key, DSANALYTICS_PROPERTY_ROI,
            sizeof (DSANALYTICS_PROPERTY_ROI) - 1)) {
      gchar *label = &key[0][sizeof (DSANALYTICS_PROPERTY_ROI) - 1];
      oc_info.oc_label = label;
      roi_list = g_key_file_get_integer_list (key_file, group, *key,
          &list_len, &error);
      //Check if the list is populated correctly
      if ((roi_list == nullptr) || (list_len % 2 != 0)) {
        CHECK_ERROR (error, group);
      }
      //FIXME: Handle multiple ROIs
      GST_CAT_INFO (NVDSANALYTICS_CFG_PARSER_CAT, "Parsed '%s' in group '%s'\n",
          *key, group);
      for (gsize icnt = 0; icnt < list_len; icnt += 2) {
        GST_CAT_INFO (NVDSANALYTICS_CFG_PARSER_CAT, "pt-%lu x=%d y=%d' \n",
            icnt / 2, roi_list[icnt], roi_list[icnt + 1]);
        oc_info.roi_pts.push_back (std::make_pair (roi_list[icnt],
                roi_list[icnt + 1]));
        CHECK_INT_VALUE_NON_NEGATIVE (*key, roi_list[icnt], group);
        CHECK_INT_VALUE_NON_NEGATIVE (*key, roi_list[icnt + 1], group);

      }
      oc_vec.push_back (oc_info);
      oc_info.roi_pts.clear ();
      g_free (roi_list);
      roi_list = nullptr;
    } else {
      g_print ("Unknown key '%s' in group in '%s'\n", *key, group);
    }
  }
  if (oc_vec.size () == 0) {
    PARSE_ERROR ("ROI not specified in group");
  }

for (OverCrowdingInfo & oc:oc_vec) {
    oc.enable = enable;
    oc.object_threshold = object_threshold;
    oc.time_threshold_in_ms = time_threshold_in_ms;
    oc.operate_on_class = operate_on_class_vec;
  }
  if (stream_analytics_info->count (stream_id) == 0) {
    StreamInfo stream_specific_info = {
      .roi_info = {},
      .overcrowding_info = {},
      .linecrossing_info = {},
      .direction_info = {},
      .config_width = nvdsanalytics->configuration_width,
      .config_height = nvdsanalytics->configuration_height,
    };
  for (OverCrowdingInfo & oc:oc_vec) {
      stream_specific_info.overcrowding_info.push_back (oc);
    }
    //stream_analytics_info->insert(std::make_pair(stream_id, stream_specific_info));
    (*stream_analytics_info)[stream_id] = stream_specific_info;
  } else {
    StreamInfo & stream_specific_info = (*stream_analytics_info)[stream_id];
  for (OverCrowdingInfo & oc:oc_vec) {
      stream_specific_info.overcrowding_info.push_back (oc);
    }
  }


  ret = TRUE;

done:
  g_free (roi_list);
  return ret;
}

static gboolean
nvdsanalytics_parse_roi_filtering_group (GstNvDsAnalytics * nvdsanalytics,
    gchar * cfg_file_path, GKeyFile * key_file, gchar * group,
    guint64 stream_id)
{
  g_autoptr (GError) error = nullptr;
  gboolean ret = FALSE;
  gboolean enable = FALSE;
  //gint operate_on_class = -1;
  std::vector < gint > operate_on_class_vec;
  gboolean inverse_roi = FALSE;
  ROIInfo roi_info = {
    .enable = false,
    .roi_pts = {},
    .roi_label = "",
    .inverse_roi = false,
    .operate_on_class = {},
    .stream_id = 0
  };
  std::vector < ROIInfo > roi_vec;
  gint *roi_list = nullptr;
  gsize list_len = 0;
  g_auto (GStrv) keys = nullptr;
  GStrv key = nullptr;
  std::unordered_map < int, StreamInfo > *stream_analytics_info =
      (nvdsanalytics->stream_analytics_info);
  roi_info.stream_id = stream_id;

  keys = g_key_file_get_keys (key_file, group, nullptr, &error);
  CHECK_ERROR (error, group);

  for (key = keys; *key; key++) {

    if (!g_strcmp0 (*key, DSANALYTICS_PROPERTY_ENABLE)) {
      enable = g_key_file_get_boolean (key_file, group, *key, &error);
      CHECK_ERROR (error, group);
      GST_CAT_INFO (NVDSANALYTICS_CFG_PARSER_CAT,
          "Parsed '%s=%d' in group '%s'\n", *key, enable, group);
    } else if (!g_strcmp0 (*key, DSANALYTICS_PROPERTY_CLASS_ID)) {
      roi_list = g_key_file_get_integer_list (key_file, group, *key,
          &list_len, &error);
      //Check if the list is populated correctly
      if ((roi_list == nullptr)) {
        CHECK_ERROR (error, group);
      }

      operate_on_class_vec.clear ();
      for (gsize icnt = 0; icnt < list_len; icnt++) {
        operate_on_class_vec.push_back (roi_list[icnt]);
        GST_CAT_INFO (NVDSANALYTICS_CFG_PARSER_CAT,
            "Parsed '%s=%d' in group '%s'\n", *key, roi_list[icnt], group);
      }
      g_free (roi_list);
      roi_list = nullptr;

    } else if (!g_strcmp0 (*key,
            DSANALYTICS_PROPERTY_GROUP_ROI_FILTERING_INVERSE_ROI)) {
      inverse_roi = g_key_file_get_boolean (key_file, group, *key, &error);
      CHECK_ERROR (error, group);
      GST_CAT_INFO (NVDSANALYTICS_CFG_PARSER_CAT,
          "Parsed '%s=%d' in group '%s'\n", *key, inverse_roi, group);
    } else if (!strncmp (*key, DSANALYTICS_PROPERTY_ROI,
            sizeof (DSANALYTICS_PROPERTY_ROI) - 1)) {
      gchar *label = &key[0][sizeof (DSANALYTICS_PROPERTY_ROI) - 1];
      roi_info.roi_label = label;
      roi_list = g_key_file_get_integer_list (key_file, group, *key,
          &list_len, &error);
      //Check if the list is populated correctly
      if ((roi_list == nullptr) || (list_len % 2 != 0)) {
        CHECK_ERROR (error, group);
      }
      GST_CAT_INFO (NVDSANALYTICS_CFG_PARSER_CAT, "Parsed '%s' in group '%s'\n",
          *key, group);
      //FIXME: Handle multiple ROIs
      for (gsize icnt = 0; icnt < list_len; icnt += 2) {
        GST_CAT_INFO (NVDSANALYTICS_CFG_PARSER_CAT, "pt-%lu x=%d y=%d' \n",
            icnt / 2, roi_list[icnt], roi_list[icnt + 1]);

        roi_info.roi_pts.push_back (std::make_pair (roi_list[icnt],
                roi_list[icnt + 1]));
        CHECK_INT_VALUE_NON_NEGATIVE (*key, roi_list[icnt], group);
        CHECK_INT_VALUE_NON_NEGATIVE (*key, roi_list[icnt + 1], group);
      }
      roi_vec.push_back (roi_info);
      roi_info.roi_pts.clear ();
      g_free (roi_list);
      roi_list = nullptr;
    } else {
      g_print ("Unknown key '%s' in group in '%s'\n", *key, group);
    }
  }
  if (roi_vec.size () == 0) {
    PARSE_ERROR ("ROI not specified in group");

  }

for (ROIInfo & roi:roi_vec) {
    roi.enable = enable;
    roi.inverse_roi = inverse_roi;
    roi.operate_on_class = operate_on_class_vec;
  }
  if (stream_analytics_info->count (stream_id) == 0) {
    StreamInfo stream_specific_info = {
      .roi_info = {},
      .overcrowding_info = {},
      .linecrossing_info = {},
      .direction_info = {},
      .config_width = nvdsanalytics->configuration_width,
      .config_height = nvdsanalytics->configuration_height,
    };
  for (ROIInfo & roi:roi_vec) {
      stream_specific_info.roi_info.push_back (roi);
    }
    //stream_analytics_info->insert(std::make_pair(stream_id, stream_specific_info));
    (*stream_analytics_info)[stream_id] = stream_specific_info;
  } else {
    StreamInfo & stream_specific_info = (*stream_analytics_info)[stream_id];
  for (ROIInfo & roi:roi_vec) {
      stream_specific_info.roi_info.push_back (roi);
    }
  }

  ret = TRUE;

done:
  g_free (roi_list);
  return ret;
}



//G_DEFINE_AUTO_CLEANUP_FREE_FUNC(GStrv, g_strfreev, nullptr);
/* Parse the nvdsanalytics config file. Returns FALSE in case of an error. */
gboolean
nvdsanalytics_parse_config_file (GstNvDsAnalytics * nvdsanalytics,
    gchar * cfg_file_path)
{
  g_autoptr (GError) error = nullptr;
  gboolean ret = FALSE;
  g_auto (GStrv) groups = nullptr;
  gboolean property_present = FALSE;
  GStrv group;
  g_autoptr (GKeyFile) cfg_file = g_key_file_new ();
  guint64 stream_index = 0;
  //FIXME: Clear only if successful?
  nvdsanalytics->stream_analytics_info->clear ();
  if (!NVDSANALYTICS_CFG_PARSER_CAT) {
    GstDebugLevel level;
    GST_DEBUG_CATEGORY_INIT (NVDSANALYTICS_CFG_PARSER_CAT, "nvdsanalytics", 0,
        NULL);
    level = gst_debug_category_get_threshold (NVDSANALYTICS_CFG_PARSER_CAT);
    if (level < GST_LEVEL_ERROR)
      gst_debug_category_set_threshold (NVDSANALYTICS_CFG_PARSER_CAT,
          GST_LEVEL_ERROR);
  }

  if (!g_key_file_load_from_file (cfg_file, cfg_file_path, G_KEY_FILE_NONE,
          &error)) {
    PARSE_ERROR ("%s", error->message);
  }
  // Check if 'property' group present
  if (!g_key_file_has_group (cfg_file, DSANALYTICS_PROPERTY)) {
    PARSE_ERROR ("Group 'property' not specified");
  }

  g_key_file_set_list_separator (cfg_file, ';');

  groups = g_key_file_get_groups (cfg_file, nullptr);
  // iterate over all groups
  for (group = groups; *group; group++) {

    GST_CAT_INFO (NVDSANALYTICS_CFG_PARSER_CAT, "Group found %s \n", *group);
    if (!strcmp (*group, DSANALYTICS_PROPERTY)) {
      property_present = nvdsanalytics_parse_property_group (nvdsanalytics,
          cfg_file_path, cfg_file);
    } else
        if (!strncmp (*group, DSANALYTICS_PROPERTY_GROUP_ROI_FILTERING,
            sizeof (DSANALYTICS_PROPERTY_GROUP_ROI_FILTERING) - 1)) {

      EXTRACT_STREAM_ID (DSANALYTICS_PROPERTY_GROUP_ROI_FILTERING);
      if (!nvdsanalytics_parse_roi_filtering_group (nvdsanalytics,
              cfg_file_path, cfg_file, *group, stream_index)) {
        goto done;
      }
    } else
        if (!strncmp (*group, DSANALYTICS_PROPERTY_GROUP_OVERCROWDING,
            sizeof (DSANALYTICS_PROPERTY_GROUP_OVERCROWDING) - 1)) {
      EXTRACT_STREAM_ID (DSANALYTICS_PROPERTY_GROUP_OVERCROWDING);
      if (!nvdsanalytics_parse_overcrowding_group (nvdsanalytics,
              cfg_file_path, cfg_file, *group, stream_index)) {
        goto done;
      }

    } else
        if (!strncmp (*group, DSANALYTICS_PROPERTY_GROUP_DIRECTION_DETECTION,
            sizeof (DSANALYTICS_PROPERTY_GROUP_DIRECTION_DETECTION) - 1)) {
      EXTRACT_STREAM_ID (DSANALYTICS_PROPERTY_GROUP_DIRECTION_DETECTION);
      if (!nvdsanalytics_parse_direction_detection_group (nvdsanalytics,
              cfg_file_path, cfg_file, *group, stream_index)) {
        goto done;
      }
    } else
        if (!strncmp (*group, DSANALYTICS_PROPERTY_GROUP_LINE_CROSSING,
            sizeof (DSANALYTICS_PROPERTY_GROUP_LINE_CROSSING) - 1)) {
      EXTRACT_STREAM_ID (DSANALYTICS_PROPERTY_GROUP_LINE_CROSSING);

      if (!nvdsanalytics_parse_linecrossing_group (nvdsanalytics,
              cfg_file_path, cfg_file, *group, stream_index)) {
        goto done;
      }
    } else {
      g_print ("NVDSANALYTICS_CFG_PARSER: Group '%s' ignored\n", *group);

    }

  }
  if (FALSE == property_present) {
    ret = FALSE;
  } else {
    ret = TRUE;
  }

for (auto & info:nvdsanalytics->stream_analytics_info[0]) {
    info.second.config_width = nvdsanalytics->configuration_width;
    info.second.config_height = nvdsanalytics->configuration_height;
  }

done:

  return ret;
}
