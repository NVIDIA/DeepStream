/**
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
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
#include <filesystem>
#include "nvdsmetamux_property_parser.h"

GST_DEBUG_CATEGORY (NVDSMETAMUX_CFG_PARSER_CAT);

#define PARSE_ERROR(details_fmt,...) \
  G_STMT_START { \
    GST_CAT_ERROR (NVDSMETAMUX_CFG_PARSER_CAT, \
        "Failed to parse config file %s: " details_fmt, \
        cfg_file_path, ##__VA_ARGS__); \
    GST_ELEMENT_ERROR (nvdsmetamux, LIBRARY, SETTINGS, \
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

#define GET_STRING_PROPERTY(group, property, field) {\
  field = g_key_file_get_string(key_file, group, property, &error); \
  CHECK_ERROR(error, group); \
}

#define READ_UINT_PROPERTY(group, property, field) {\
  field = g_key_file_get_integer(key_file, group, property, &error); \
  CHECK_ERROR(error, group); \
  CHECK_INT_VALUE_NON_NEGATIVE(property,\
                               field, group);\
}

#define EXTRACT_STREAM_ID(for_key){\
      gchar **tmp; \
      gchar *endptr1; \
      model_id = 0; \
      tmp = g_strsplit(*for_key, "-", 5); \
      /*g_print("**** %s &&&&&&\n", tmp[g_strv_length(tmp)-1]);*/ \
      model_id = g_ascii_strtoull(tmp[g_strv_length(tmp)-1], &endptr1, 10); \
}

#define EXTRACT_GROUP_ID(for_group){\
      gchar *group1 = *group + sizeof (for_group) - 1; \
      gchar *endptr; \
      group_index = g_ascii_strtoull (group1, &endptr, 10); \
}

static gboolean
nvdsmetamux_parse_property_group (GstNvDsMetaMux * nvdsmetamux,
    gchar * cfg_file_path, GKeyFile * key_file, gchar * group)
{
  g_autoptr (GError) error = nullptr;
  gboolean ret = FALSE;
  g_auto (GStrv) keys = nullptr;
  GStrv key = nullptr;

  keys = g_key_file_get_keys (key_file, group, nullptr, &error);
  CHECK_ERROR (error, group);

  for (key = keys; *key; key++) {
    if (!g_strcmp0 (*key, NVDSMETAMUX_PROPERTY_ENABLE)) {
      gboolean val = g_key_file_get_boolean (key_file, group,
          NVDSMETAMUX_PROPERTY_ENABLE, &error);
      CHECK_ERROR (error, group);
      nvdsmetamux->enable = val;
    } else if (!g_strcmp0 (*key, NVDSMETAMUX_PROPERTY_ACTIVE_PAD)) {
      GET_STRING_PROPERTY (group, *key, nvdsmetamux->active_sink_pad);
      GST_CAT_INFO (NVDSMETAMUX_CFG_PARSER_CAT, "Parsed %s=%s in group '%s'\n",
          *key, nvdsmetamux->active_sink_pad, group);
    } else if (!g_strcmp0 (*key, NVDSMETAMUX_PROPERTY_PTS_TOLERANCE)) {
      guint val = g_key_file_get_integer (key_file, group, *key, &error);
      CHECK_ERROR (error, group);
      nvdsmetamux->pts_tolerance = val;
      GST_CAT_INFO (NVDSMETAMUX_CFG_PARSER_CAT, "Parsed %s=%ld in group '%s'\n",
          *key, nvdsmetamux->pts_tolerance, group);
    }
  }

  ret = TRUE;

done:
  return ret;
}

static gboolean
nvdsmetamux_parse_common_group (GstNvDsMetaMux * nvdsmetamux,
    gchar * cfg_file_path, GKeyFile * key_file, gchar * group, guint64 group_id)
{
  g_autoptr (GError) error = nullptr;
  gboolean ret = FALSE;
  g_auto (GStrv) keys = nullptr;
  GStrv key = nullptr;
  guint64 model_id = 0;
  gint *source_id_list = nullptr;
  std::vector < gint > src_ids;
  gsize source_id_list_len = 0;
  std::unordered_map < gint, GstNvDsMetaMuxModelSource > umap;

  keys = g_key_file_get_keys (key_file, group, nullptr, &error);
  CHECK_ERROR (error, group);

  for (key = keys; *key; key++) {
    if (!strncmp (*key, NVDSMETAMUX_GROUP_SRC_IDS_MODEL,
            sizeof (NVDSMETAMUX_GROUP_SRC_IDS_MODEL) - 1)) {
      EXTRACT_STREAM_ID (key);
      source_id_list =
          g_key_file_get_integer_list (key_file, group, *key,
          &source_id_list_len, &error);
      if (source_id_list == nullptr) {
        CHECK_ERROR (error, group);
      }
      GstNvDsMetaMuxModelSource model_source;

      GST_CAT_INFO (NVDSMETAMUX_CFG_PARSER_CAT,
          "Parsing model_id = %ld source_id listlen = %ld\n", model_id,
          source_id_list_len);

      for (guint i = 0; i < source_id_list_len; i++) {
        model_source.source_id_vector.push_back (source_id_list[i]);
      }

      umap.emplace (model_id, model_source);

      GST_CAT_INFO (NVDSMETAMUX_CFG_PARSER_CAT, "Parsed '%s' in group '%s'\n",
          NVDSMETAMUX_GROUP_SRC_IDS_MODEL, group);
      g_free (source_id_list);
      source_id_list = nullptr;
    }
  }

  nvdsmetamux->model_source_map = umap;
  ret = TRUE;

done:
  return ret;
}

/* Parse the nvdsmetamux config file. Returns FALSE in case of an error. */
gboolean
nvdsmetamux_parse_config_file (GstNvDsMetaMux * nvdsmetamux,
    gchar * cfg_file_path)
{
  g_autoptr (GError) error = nullptr;
  gboolean ret = FALSE;
  g_auto (GStrv) groups = nullptr;
  GStrv group;
  g_autoptr (GKeyFile) cfg_file = g_key_file_new ();
  guint64 group_index = 0;

  if (!NVDSMETAMUX_CFG_PARSER_CAT) {
    GstDebugLevel level;
    GST_DEBUG_CATEGORY_INIT (NVDSMETAMUX_CFG_PARSER_CAT, "nvdsmetamux", 0,
        NULL);
    level = gst_debug_category_get_threshold (NVDSMETAMUX_CFG_PARSER_CAT);
    if (level < GST_LEVEL_ERROR)
      gst_debug_category_set_threshold (NVDSMETAMUX_CFG_PARSER_CAT,
          GST_LEVEL_ERROR);
  }

  if (!g_key_file_load_from_file (cfg_file, cfg_file_path, G_KEY_FILE_NONE,
          &error)) {
    PARSE_ERROR ("%s", error->message);
  }
  // Check if 'property' group present
  if (!g_key_file_has_group (cfg_file, NVDSMETAMUX_PROPERTY)) {
    PARSE_ERROR ("Group 'property' not specified");
  }

  g_key_file_set_list_separator (cfg_file, ';');

  groups = g_key_file_get_groups (cfg_file, nullptr);

  for (group = groups; *group; group++) {
    GST_CAT_INFO (NVDSMETAMUX_CFG_PARSER_CAT, "Group found %s \n", *group);
    if (!strcmp (*group, NVDSMETAMUX_PROPERTY)) {
      ret = nvdsmetamux_parse_property_group (nvdsmetamux,
          cfg_file_path, cfg_file, *group);
      if (!ret) {
        g_print ("NVDSMETAMUX_CFG_PARSER: Group '%s' parse failed\n", *group);
        goto done;
      }
    } else if (!strncmp (*group, NVDSMETAMUX_GROUP,
            sizeof (NVDSMETAMUX_GROUP) - 1)) {
      EXTRACT_GROUP_ID (NVDSMETAMUX_GROUP);
      GST_DEBUG ("parsing group index = %lu\n", group_index);
      ret = nvdsmetamux_parse_common_group (nvdsmetamux,
          cfg_file_path, cfg_file, *group, group_index);
      if (!ret) {
        g_print ("NVDSMETAMUX_CFG_PARSER: Group '%s' parse failed\n", *group);
        goto done;
      }
    } else {
      g_print ("NVDSMETAMUX_CFG_PARSER: Group '%s' ignored\n", *group);
    }
  }

done:
  return ret;
}
