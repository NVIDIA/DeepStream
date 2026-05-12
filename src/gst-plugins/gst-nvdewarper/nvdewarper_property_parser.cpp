/**
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <string.h>
#include <cuda.h>
#include "nvdewarper_property_parser.h"

GST_DEBUG_CATEGORY (DEWARPER_CFG_PARSER_CAT);

#define PARSE_ERROR(details_fmt,...) \
  G_STMT_START { \
    GST_CAT_ERROR (DEWARPER_CFG_PARSER_CAT, \
        "Failed to parse config file %s: " details_fmt, \
        cfg_file_path, ##__VA_ARGS__); \
    GST_ELEMENT_ERROR (nvdewarper, LIBRARY, SETTINGS, \
        ("Failed to parse config file:%s", cfg_file_path), \
        (details_fmt, ##__VA_ARGS__)); \
    goto done; \
  } G_STMT_END

#define CHECK_ERROR(error) \
  G_STMT_START { \
    if (error) { \
      PARSE_ERROR ("%s", error->message); \
    } \
  } G_STMT_END

#define CHECK_BOOLEAN_VALUE(prop_name,value) \
  G_STMT_START { \
    if ((gint) value < 0 || value > 1) { \
      PARSE_ERROR ("Boolean property '%s' can have values 0 or 1", prop_name); \
    } \
  } G_STMT_END

#define CHECK_INT_VALUE_NON_NEGATIVE(prop_name,value) \
  G_STMT_START { \
    if ((gint) value < 0) { \
      PARSE_ERROR ("Integer property '%s' can have value >=0", prop_name); \
    } \
  } G_STMT_END

#define CHECK_INT_VALUE_LIMIT(prop_name,value,limit) \
  G_STMT_START { \
    if ((gint) value > limit) { \
      PARSE_ERROR ("Integer property '%s' can have max value %d", prop_name, \
          limit); \
    } \
  } G_STMT_END

inline bool CUDA_CHECK_ERR_(int e, int iLine, const char *szFile) {
  if (e != cudaSuccess) {
    std::cout << "CUDA runtime error " << e << " at line " << iLine <<
        " in file " << szFile << endl;
    return false;
  }
  return true;
}

#define cuda_check(call) CUDA_CHECK_ERR_(call, __LINE__, __FILE__)

#define DEFAULT_THRESHOLD 0.2
#define DEFAULT_EPS 0
#define DEFAULT_GROUP_THRESHOLD 0
#define DEFAULT_MIN_BOXES 1

#define MAX_SURFACES 4

gchar *
get_absolute_file_path (gchar * cfg_file_path, gchar * file_path)
{
  gchar abs_cfg_path[PATH_MAX + 1];
  gchar *abs_file_path;
  gchar *delim;

  if (file_path[0] == '/') {
    return file_path;
  }

  if (!realpath (cfg_file_path, abs_cfg_path)) {
    g_free (file_path);
    return NULL;
  }

  delim = g_strrstr (abs_cfg_path, "/");
  *(delim + 1) = '\0';

  abs_file_path = g_strconcat (abs_cfg_path, file_path, NULL);
  g_free (file_path);

  return abs_file_path;
}

static gboolean
nvdewarper_parse_surface_attributes (Gstnvdewarper * nvdewarper,
    GKeyFile * key_file, gchar * group, gchar * cfg_file_path, gint index)
{
  gboolean ret = FALSE;
  gchar **keys = NULL;
  gchar **key = NULL;
  GError *error = NULL;
  NvDewarperParams surfaceParams = {0};
  surfaceParams.control = 0.6f; // set default control = 0.6 to maintain legacy for pushbroom projection

  keys = g_key_file_get_keys (key_file, group, NULL, &error);
  CHECK_ERROR (error);


  for (key = keys; *key; key++) {
    if (!g_strcmp0 (*key, CONFIG_GROUP_DEWARPER_SURFACE_PROJECTION_TYPE)) {
      surfaceParams.projection_type =
          g_key_file_get_integer (key_file, group,
              CONFIG_GROUP_DEWARPER_SURFACE_PROJECTION_TYPE, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_DEWARPER_SURFACE_INDEX)) {
      surfaceParams.surface_index =
          g_key_file_get_integer (key_file, group,
              CONFIG_GROUP_DEWARPER_SURFACE_INDEX, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_DEWARPER_SURFACE_WIDTH)) {
      surfaceParams.dewarpWidth =
          g_key_file_get_integer (key_file, group,
              CONFIG_GROUP_DEWARPER_SURFACE_WIDTH, &error);
      CHECK_INT_VALUE_NON_NEGATIVE(CONFIG_GROUP_DEWARPER_SURFACE_WIDTH,
          surfaceParams.dewarpWidth);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_DEWARPER_SURFACE_HEIGHT)) {
      surfaceParams.dewarpHeight =
          g_key_file_get_integer (key_file, group,
              CONFIG_GROUP_DEWARPER_SURFACE_HEIGHT, &error);
      CHECK_INT_VALUE_NON_NEGATIVE(CONFIG_GROUP_DEWARPER_SURFACE_HEIGHT,
          surfaceParams.dewarpHeight);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_DEWARPER_SURFACE_TOP_ANGLE)) {
      surfaceParams.top_angle =
          g_key_file_get_double (key_file, group,
              CONFIG_GROUP_DEWARPER_SURFACE_TOP_ANGLE, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_DEWARPER_SURFACE_BOTTOM_ANGLE)) {
      surfaceParams.bottom_angle =
          g_key_file_get_double (key_file, group,
              CONFIG_GROUP_DEWARPER_SURFACE_BOTTOM_ANGLE, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_DEWARPER_SURFACE_PITCH)) {
      surfaceParams.pitch =
          g_key_file_get_double (key_file, group,
              CONFIG_GROUP_DEWARPER_SURFACE_PITCH, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_DEWARPER_SURFACE_YAW)) {
      surfaceParams.yaw =
          g_key_file_get_double (key_file, group,
              CONFIG_GROUP_DEWARPER_SURFACE_YAW, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_DEWARPER_SURFACE_ROLL)) {
      surfaceParams.roll =
          g_key_file_get_double (key_file, group,
              CONFIG_GROUP_DEWARPER_SURFACE_ROLL, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_DEWARPER_SURFACE_FOCAL_LENGTH)) {
      gsize length = 0;
      gdouble* focal_length =
          g_key_file_get_double_list (key_file, group,
              CONFIG_GROUP_DEWARPER_SURFACE_FOCAL_LENGTH, &length, &error);
      CHECK_ERROR (error);
      if (length > FOCAL_LENGTH_SIZE){
        g_print("%u focal length parameters provided. Only the first %d will be "
            "used\n", (unsigned)length, FOCAL_LENGTH_SIZE);
        length = FOCAL_LENGTH_SIZE;
      }
      for (size_t i = 0; i < length ; i++)
        surfaceParams.dewarpFocalLength[i] = focal_length[i];
      g_free(focal_length);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_DEWARPER_SURFACE_ADDRESS_MODE)) {
      surfaceParams.addressMode =
          g_key_file_get_integer (key_file, group,
              CONFIG_GROUP_DEWARPER_SURFACE_ADDRESS_MODE, &error);
      CHECK_ERROR (error);

      if (surfaceParams.addressMode < 0 || surfaceParams.addressMode > 1){
        g_print("INVALID cuda-address-mode (%d). Default mode will be used "
            "(%d)\n", surfaceParams.addressMode, 0);
        surfaceParams.addressMode = 0;
      }
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_DEWARPER_SURFACE_FIELD_OF_VIEW)) {
      surfaceParams.srcFov =
          g_key_file_get_double(key_file, group,
              CONFIG_GROUP_DEWARPER_SURFACE_FIELD_OF_VIEW, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_DEWARPER_SURFACE_DISTORTION)) {
      gsize length = 0;
      gdouble* distortion =
          g_key_file_get_double_list (key_file, group,
              CONFIG_GROUP_DEWARPER_SURFACE_DISTORTION, &length, &error);
      CHECK_ERROR (error);

      if (length > DISTORTION_SIZE){
        g_print("%u distortion parameters provided. Only the first %d will be "
            "used\n", (unsigned)length, DISTORTION_SIZE);
        length = DISTORTION_SIZE;
      }

      /* Array is already initialized to zero, so no need to set remaining array
      values if not all the distortion values are provided*/
      for (size_t i = 0; i < length ; i++)
        surfaceParams.distortion[i] = distortion[i];

      g_free(distortion);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_DEWARPER_SURFACE_SRC_X0)) {
      surfaceParams.src_x0 =
          g_key_file_get_double (key_file, group,
              CONFIG_GROUP_DEWARPER_SURFACE_SRC_X0, &error);
      CHECK_ERROR (error);
    }else if (!g_strcmp0 (*key, CONFIG_GROUP_DEWARPER_SURFACE_SRC_Y0)) {
      surfaceParams.src_y0 =
          g_key_file_get_double (key_file, group,
              CONFIG_GROUP_DEWARPER_SURFACE_SRC_Y0, &error);
    }else if (!g_strcmp0 (*key, CONFIG_GROUP_DEWARPER_SURFACE_ROTATION_AXES)) {
      gchar *rot_axes = g_key_file_get_string (key_file, group,
              CONFIG_GROUP_DEWARPER_SURFACE_ROTATION_AXES, &error);
      CHECK_ERROR (error);
      g_strlcpy(surfaceParams.rot_axes, rot_axes, sizeof(surfaceParams.rot_axes));
      g_free(rot_axes);
    }else if (!g_strcmp0 (*key, CONFIG_GROUP_DEWARPER_SURFACE_CONTROL)) {
      surfaceParams.control =
          g_key_file_get_double (key_file, group,
              CONFIG_GROUP_DEWARPER_SURFACE_CONTROL, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_DEWARPER_SURFACE_ROTATION_MATRIX)) {
      gsize length = 0;
      gdouble* matrix =
          g_key_file_get_double_list (key_file, group,
              CONFIG_GROUP_DEWARPER_SURFACE_ROTATION_MATRIX, &length, &error);
      CHECK_ERROR (error);

      if (length != ROTATION_MATRIX_SIZE){
        g_print("%u rotation matrix parameters provided. Expected parameters = %d. Ignoring the matrix!!\n"
            , (unsigned)length,ROTATION_MATRIX_SIZE);
        length = 0;
      }
      else
      {
        surfaceParams.rot_matrix_valid = 1;
        for (size_t i = 0; i < length; i++)
          surfaceParams.rot_matrix[i] = matrix[i];
      }
      g_free(matrix);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_DEWARPER_SURFACE_DST_FOCAL_LENGTH)) {
      gsize length = 0;
      gdouble* focal_length =
          g_key_file_get_double_list (key_file, group,
              CONFIG_GROUP_DEWARPER_SURFACE_DST_FOCAL_LENGTH, &length, &error);
      CHECK_ERROR (error);
      if (length > FOCAL_LENGTH_SIZE){
        g_print("%u focal length parameters provided. Only the first %d will be "
            "used\n", (unsigned)length, FOCAL_LENGTH_SIZE);
        length = FOCAL_LENGTH_SIZE;
      }
      for (size_t i = 0; i < length ; i++)
        surfaceParams.dstFocalLength[i] = focal_length[i];
      g_free(focal_length);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_DEWARPER_SURFACE_DST_PRINCIPAL_POINT)) {
      gsize length = 0;
      gdouble* principal_point =
          g_key_file_get_double_list (key_file, group,
              CONFIG_GROUP_DEWARPER_SURFACE_DST_PRINCIPAL_POINT, &length, &error);
      CHECK_ERROR (error);
      if (length != 2){
        g_print("%u principal point parameters provided. Expecting 2 values. Ignoring the setting!!\n"
            , (unsigned)length);
      }
      else
      {
          surfaceParams.dstPrincipalPoint[0] = principal_point[0];
          surfaceParams.dstPrincipalPoint[1] = principal_point[1];
      }
      g_free(principal_point);
    }
    else {
      g_print ("%s: Unknown key '%s' for group [%s]\n", cfg_file_path, *key,
          group);
    }
  }

  surfaceParams.isValid = 1;
  surfaceParams.id = index;
  nvdewarper->priv->vecDewarpSurface.push_back(surfaceParams);

  ret = TRUE;
done:
  if (error) {
    g_error_free (error);
  }
  if (keys) {
    g_strfreev (keys);
  }
  return ret;
}

gboolean
nvdewarper_parse_props (Gstnvdewarper * nvdewarper, GKeyFile * key_file,
    gchar * group, gchar * cfg_file_path)
{
  gboolean ret = FALSE;
  gchar **keys = NULL;
  gchar **key = NULL;
  GError *error = NULL;

  keys = g_key_file_get_keys (key_file, group, NULL, &error);
  CHECK_ERROR (error);

  for (key = keys; *key; key++) {
    if (!g_strcmp0 (*key, CONFIG_GROUP_DEWARPER_PROPERTY_OUTPUT_WIDTH)) {
      nvdewarper->output_width =
          g_key_file_get_integer (key_file, group,
              CONFIG_GROUP_DEWARPER_PROPERTY_OUTPUT_WIDTH, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key,
        CONFIG_GROUP_DEWARPER_PROPERTY_OUTPUT_HEIGHT)) {
      nvdewarper->output_height =
          g_key_file_get_integer (key_file, group,
              CONFIG_GROUP_DEWARPER_PROPERTY_OUTPUT_HEIGHT, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key,
        CONFIG_GROUP_DEWARPER_PROPERTY_CUDA_MEMORY_TYPE)) {
      nvdewarper->cuda_mem_type =
          static_cast<NvBufSurfaceMemType>(g_key_file_get_integer (key_file, group,
              CONFIG_GROUP_DEWARPER_PROPERTY_CUDA_MEMORY_TYPE, &error));
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_DEWARPER_PROPERTY_DUMP_FRAMES)) {
      nvdewarper->dump_frames =
          g_key_file_get_integer (key_file, group,
              CONFIG_GROUP_DEWARPER_PROPERTY_DUMP_FRAMES, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key,
        CONFIG_GROUP_DEWARPER_PROPERTY_AISLE_CALIB_FILE)) {
      nvdewarper->aisle_calibration_file =
          get_absolute_file_path (cfg_file_path,
              g_key_file_get_string (key_file, group,
              CONFIG_GROUP_DEWARPER_PROPERTY_AISLE_CALIB_FILE, &error));
      nvdewarper->aisle_calibrationfile_set = TRUE;
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key,
        CONFIG_GROUP_DEWARPER_PROPERTY_SPOT_CALIB_FILE)) {
      nvdewarper->spot_calibration_file = get_absolute_file_path (cfg_file_path,
          g_key_file_get_string (key_file, group,
              CONFIG_GROUP_DEWARPER_PROPERTY_SPOT_CALIB_FILE, &error));
      nvdewarper->spot_calibrationfile_set = TRUE;
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key,
        CONFIG_GROUP_DEWARPER_PROPERTY_NUM_BATCH_BUFFERS)) {
      nvdewarper->num_batch_buffers = g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_DEWARPER_PROPERTY_NUM_BATCH_BUFFERS, &error);
      CHECK_INT_VALUE_LIMIT(CONFIG_GROUP_DEWARPER_PROPERTY_NUM_BATCH_BUFFERS,
          nvdewarper->num_batch_buffers, MAX_SURFACES);
      CHECK_ERROR (error);
    } else {
      g_print ("%s: Unknown key '%s' for group [%s]\n", cfg_file_path, *key,
          group);
    }
  }

  ret = TRUE;
done:
  if (error) {
    g_error_free (error);
  }
  if (keys) {
    g_strfreev (keys);
  }
  return ret;
}

gboolean
nvdewarper_parse_config_file (Gstnvdewarper * nvdewarper, gchar * cfg_file_path)
{
  GError *error = NULL;
  gboolean ret = FALSE;
  gchar **groups = NULL;
  gchar **group;
  GKeyFile *cfg_file = g_key_file_new ();

  if (!DEWARPER_CFG_PARSER_CAT) {
    GST_DEBUG_CATEGORY_INIT (DEWARPER_CFG_PARSER_CAT, "NVDEWARPER_CFG_PARSER",
        0, NULL);
  }

  if (!g_key_file_load_from_file (cfg_file, cfg_file_path, G_KEY_FILE_NONE,
          &error)) {
    PARSE_ERROR ("%s", error->message);
  }

  if (!g_key_file_has_group (cfg_file, CONFIG_GROUP_DEWARPER_PROPERTY)) {
    PARSE_ERROR ("Group 'property' not specified");
  }

  if (!nvdewarper_parse_props (nvdewarper, cfg_file,
      CONFIG_GROUP_DEWARPER_PROPERTY, cfg_file_path))
    goto done;

  g_key_file_remove_group (cfg_file, CONFIG_GROUP_DEWARPER_PROPERTY, NULL);

  groups = g_key_file_get_groups (cfg_file, NULL);

  if (!nvdewarper->aisle_calibrationfile_set ||
          !nvdewarper->spot_calibrationfile_set)
  {
    for (group = groups; *group; group++) {
      gboolean parse_err = FALSE;

      GST_CAT_DEBUG (DEWARPER_CFG_PARSER_CAT, "Parsing group: %s", *group);
      if (!strncmp (*group, CONFIG_GROUP_DEWARPER_SURFACE_ATTRS_PREFIX,
          sizeof (CONFIG_GROUP_DEWARPER_SURFACE_ATTRS_PREFIX) - 1)) {

        gchar *key1 = *group +
            sizeof (CONFIG_GROUP_DEWARPER_SURFACE_ATTRS_PREFIX) - 1;
        gchar *endptr = NULL;
        guint64 surface_index = g_ascii_strtoull (key1, &endptr, 10);

        if (surface_index == 0 && endptr == key1) {
          PARSE_ERROR
          ("Invalid group [%s]. surface attributes should be specified using "
              "group name '" CONFIG_GROUP_DEWARPER_SURFACE_ATTRS_PREFIX
              "<surfaceId>'", *group);
        }
        if ((gint) surface_index < 0) {
          PARSE_ERROR ("Invalid group [%s]. class-id should be >= 0", *group);
        }

        parse_err = !nvdewarper_parse_surface_attributes (nvdewarper, cfg_file,
            *group, cfg_file_path, surface_index);
      } else {
        GST_CAT_WARNING (DEWARPER_CFG_PARSER_CAT, "Unknown group '%s'", *group);
      }

      if (parse_err) {
        GST_CAT_ERROR (DEWARPER_CFG_PARSER_CAT, "Failed to parse '%s' group",
            *group);
        goto done;
      }
    }
  }
  else
  {
    // USE CVS INIT Here
    GST_WARNING_OBJECT (nvdewarper,
        "Dewarper Config File: Skipping source group config,"
        " using CSV parsing module\n");
  }
  ret = TRUE;

done:
  if (cfg_file) {
    g_key_file_free (cfg_file);
  }

  if (groups) {
    g_strfreev (groups);
  }

  if (error) {
    g_error_free (error);
  }
  return ret;
}
