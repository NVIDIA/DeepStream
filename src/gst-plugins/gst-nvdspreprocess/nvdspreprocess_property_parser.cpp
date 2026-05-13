/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <iostream>
#include <string>
#include <cstring>
#include <cmath>
#include <algorithm>
#include "nvdspreprocess_property_parser.h"

GST_DEBUG_CATEGORY (NVDSPREPROCESS_CFG_PARSER_CAT);

#define PARSE_ERROR(details_fmt,...) \
  G_STMT_START { \
    GST_CAT_ERROR (NVDSPREPROCESS_CFG_PARSER_CAT, \
        "Failed to parse config file %s: " details_fmt, \
        cfg_file_path, ##__VA_ARGS__); \
    GST_ELEMENT_ERROR (nvdspreprocess, LIBRARY, SETTINGS, \
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
  gchar* temp = g_key_file_get_string(key_file, group, property, &error); \
  field = temp; \
  g_free (temp); \
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
      source_index = 0; \
      tmp = g_strsplit(*for_key, "-", 5); \
      /*g_print("**** %s &&&&&&\n", tmp[g_strv_length(tmp)-1]);*/ \
      source_index = g_ascii_strtoull(tmp[g_strv_length(tmp)-1], &endptr1, 10); \
      g_strfreev(tmp); \
}

#define EXTRACT_GROUP_ID(for_group){\
      gchar *group1 = *group + sizeof (for_group) - 1; \
      gchar *endptr; \
      group_index = g_ascii_strtoull (group1, &endptr, 10); \
}

//sum total of ROIs of all the groups
gint sum_total_rois = 0;

static gboolean
nvdspreprocess_parse_property_group (GstNvDsPreProcess *nvdspreprocess,
    gchar *cfg_file_path, GKeyFile *key_file, gchar *group);

static gboolean
nvdspreprocess_parse_common_group (GstNvDsPreProcess *nvdspreprocess,
    gchar *cfg_file_path, GKeyFile *key_file, gchar *group, guint64 group_id);

static gboolean
nvdspreprocess_parse_user_configs(GstNvDsPreProcess *nvdspreprocess,
    gchar *cfg_file_path, GKeyFile *key_file, gchar *group);

/* Get the absolute path of a file mentioned in the config given a
 * file path absolute/relative to the config file. */
static gboolean
get_absolute_file_path (
    const gchar * cfg_file_path, const gchar * file_path,
    char *abs_path_str)
{
  gchar abs_cfg_path[_PATH_MAX + 1];
  gchar abs_real_file_path[_PATH_MAX + 1];
  gchar *abs_file_path;
  gchar *delim;

  /* Absolute path. No need to resolve further. */
  if (file_path[0] == '/') {
    /* Check if the file exists, return error if not. */
    if (!realpath (file_path, abs_real_file_path)) {
      return FALSE;
    }
    g_strlcpy (abs_path_str, abs_real_file_path, _PATH_MAX);
    return TRUE;
  }

  /* Get the absolute path of the config file. */
  if (!realpath (cfg_file_path, abs_cfg_path)) {
    return FALSE;
  }

  /* Remove the file name from the absolute path to get the directory of the
   * config file. */
  delim = g_strrstr (abs_cfg_path, "/");
  *(delim + 1) = '\0';

  /* Get the absolute file path from the config file's directory path and
   * relative file path. */
  abs_file_path = g_strconcat (abs_cfg_path, file_path, nullptr);

  /* Resolve the path.*/
  if (realpath (abs_file_path, abs_real_file_path) == nullptr) {
    /* Ignore error if file does not exist and use the unresolved path. */
    if (errno == ENOENT)
      g_strlcpy (abs_real_file_path, abs_file_path, _PATH_MAX);
    else {
      g_free (abs_file_path);
      return FALSE;
    }
  }

  g_free (abs_file_path);

  g_strlcpy (abs_path_str, abs_real_file_path, _PATH_MAX);
  return TRUE;
}

static gboolean
nvdspreprocess_parse_property_group (GstNvDsPreProcess *nvdspreprocess,
    gchar *cfg_file_path, GKeyFile *key_file, gchar *group)
{
  g_autoptr(GError)error = nullptr;
  gboolean ret = FALSE;
  g_auto(GStrv)keys=nullptr;
  GStrv key=nullptr;
  gint *network_shape_list = nullptr;
  gsize network_shape_list_len = 0;
  gint *target_unique_ids_list = nullptr;
  gsize target_unique_ids_list_len = 0;

  keys = g_key_file_get_keys (key_file, group, nullptr, &error);
  CHECK_ERROR(error, group);

  for (key = keys; *key; key++){
    if (!g_strcmp0 (*key, NVDSPREPROCESS_PROPERTY_ENABLE)) {
      gboolean val = g_key_file_get_boolean(key_file, group,
          NVDSPREPROCESS_PROPERTY_ENABLE, &error);
      CHECK_ERROR(error, group);
      nvdspreprocess->enable = val;
    }
    else if (!g_strcmp0 (*key, NVDSPREPROCESS_PROPERTY_UNIQUE_ID)) {
      gboolean val = g_key_file_get_integer(key_file, group,
          NVDSPREPROCESS_PROPERTY_UNIQUE_ID, &error);
      CHECK_ERROR(error, group);
      nvdspreprocess->unique_id = val;
    }
    else if (!g_strcmp0 (*key, NVDSPREPROCESS_PROPERTY_GPU_ID)) {
      gboolean val = g_key_file_get_integer(key_file, group,
          NVDSPREPROCESS_PROPERTY_GPU_ID, &error);
      CHECK_ERROR(error, group);
      nvdspreprocess->gpu_id = val;
    }
    else if (!g_strcmp0 (*key, NVDSPREPROCESS_PROPERTY_PROCESS_ON_FRAME)) {
      gboolean val = g_key_file_get_boolean(key_file, group,
          NVDSPREPROCESS_PROPERTY_PROCESS_ON_FRAME, &error);
      CHECK_ERROR(error, group);
      nvdspreprocess->process_on_frame = val;
    }
    else if (!g_strcmp0 (*key, NVDSPREPROCESS_PROPERTY_MAINTAIN_ASPECT_RATIO)) {
      gboolean val = g_key_file_get_boolean(key_file, group,
          NVDSPREPROCESS_PROPERTY_MAINTAIN_ASPECT_RATIO, &error);
      CHECK_ERROR(error, group);
      nvdspreprocess->maintain_aspect_ratio = val;
    }
    else if (!g_strcmp0 (*key, NVDSPREPROCESS_PROPERTY_SYMMETRIC_PADDING)) {
      gboolean val = g_key_file_get_boolean(key_file, group,
          NVDSPREPROCESS_PROPERTY_SYMMETRIC_PADDING, &error);
      CHECK_ERROR(error, group);
      nvdspreprocess->symmetric_padding = val;
    }
    else if (!g_strcmp0 (*key, NVDSPREPROCESS_PROPERTY_PROCESSING_WIDTH)) {
      guint val = g_key_file_get_integer (key_file, group, *key, &error);
      CHECK_ERROR(error, group);
      nvdspreprocess->processing_width = val;
      nvdspreprocess->property_set.processing_width = TRUE;
    }
    else if (!g_strcmp0 (*key, NVDSPREPROCESS_PROPERTY_PROCESSING_HEIGHT)) {
      guint val = g_key_file_get_integer (key_file, group, *key, &error);
      CHECK_ERROR(error, group);
      nvdspreprocess->processing_height = val;
      nvdspreprocess->property_set.processing_height = TRUE;
    }
    else if (!g_strcmp0 (*key, NVDSPREPROCESS_PROPERTY_SCALING_BUF_POOL_SIZE)) {
      guint val = g_key_file_get_integer (key_file, group, *key, &error);
      CHECK_ERROR(error, group);
      nvdspreprocess->scaling_buf_pool_size = val;
    }
    else if (!g_strcmp0 (*key, NVDSPREPROCESS_PROPERTY_TENSOR_BUF_POOL_SIZE)) {
      guint val = g_key_file_get_integer (key_file, group, *key, &error);
      CHECK_ERROR(error, group);
      nvdspreprocess->tensor_buf_pool_size = val;
    }
    else if (!g_strcmp0 (*key, NVDSPREPROCESS_PROPERTY_TARGET_IDS)) {
      target_unique_ids_list = g_key_file_get_integer_list (key_file, group,*key, &target_unique_ids_list_len, &error);
      if (target_unique_ids_list == nullptr) {
        CHECK_ERROR(error, group);
      }
      nvdspreprocess->target_unique_ids.clear();
      for (gsize icnt = 0; icnt < target_unique_ids_list_len; icnt++){
        nvdspreprocess->target_unique_ids.push_back(target_unique_ids_list[icnt]);
        GST_CAT_INFO (NVDSPREPROCESS_CFG_PARSER_CAT, "Parsed '%s=%d' in group '%s'\n",
          *key, target_unique_ids_list[icnt], group);
      }
      g_free(target_unique_ids_list);
      target_unique_ids_list = nullptr;
    }
    else if (!g_strcmp0 (*key, NVDSPREPROCESS_PROPERTY_GIE_ID_FOR_OPERATION)) {
      guint val = g_key_file_get_integer (key_file, group, *key, &error);
      CHECK_ERROR(error, group);
      nvdspreprocess->operate_on_gie_id = val;
    }
    else if (!g_strcmp0 (*key, NVDSPREPROCESS_PROPERTY_NETWORK_INPUT_ORDER)) {
      guint val = g_key_file_get_integer (key_file, group, *key, &error);
      CHECK_ERROR(error, group);
      switch ((NvDsPreProcessNetworkInputOrder) val) {
        case NvDsPreProcessNetworkInputOrder_kNCHW:
        case NvDsPreProcessNetworkInputOrder_kNHWC:
        case NvDsPreProcessNetworkInputOrder_CUSTOM:
          break;
        default:
          g_printerr ("Error. Invalid value for '%s':'%d'\n", *key, val);
          goto done;
      }
      nvdspreprocess->tensor_params.network_input_order = (NvDsPreProcessNetworkInputOrder) val;
      nvdspreprocess->property_set.network_input_order = TRUE;
    }
    else if (!g_strcmp0 (*key, NVDSPREPROCESS_PROPERTY_NETWORK_SHAPE)) {
      network_shape_list = g_key_file_get_integer_list (key_file, group,*key, &network_shape_list_len, &error);
      if (network_shape_list == nullptr) {
        CHECK_ERROR(error, group);
      }
      nvdspreprocess->tensor_params.network_input_shape.clear();
      for (gsize icnt = 0; icnt < network_shape_list_len; icnt++){
        nvdspreprocess->tensor_params.network_input_shape.push_back(network_shape_list[icnt]);
        GST_CAT_INFO (NVDSPREPROCESS_CFG_PARSER_CAT, "Parsed '%s=%d' in group '%s'\n",
          *key, network_shape_list[icnt], group);
      }
      g_free(network_shape_list);
      network_shape_list = nullptr;
      nvdspreprocess->property_set.network_input_shape = TRUE;
    }
    else if (!g_strcmp0 (*key, NVDSPREPROCESS_PROPERTY_NETWORK_COLOR_FORMAT)) {
      guint val = g_key_file_get_integer (key_file, group, *key, &error);
      CHECK_ERROR(error, group);
      switch ((NvDsPreProcessFormat) val) {
        case NvDsPreProcessFormat_RGB:
        case NvDsPreProcessFormat_BGR:
        case NvDsPreProcessFormat_GRAY:
          break;
        default:
          g_printerr ("Error. Invalid value for '%s':'%d'\n", *key, val);
          goto done;
      }
      nvdspreprocess->tensor_params.network_color_format = (NvDsPreProcessFormat) val;
      nvdspreprocess->property_set.network_color_format = TRUE;
    }
    else if (!g_strcmp0 (*key, NVDSPREPROCESS_PROPERTY_SCALING_FILTER)) {
      int val =  g_key_file_get_integer (key_file, group,
        NVDSPREPROCESS_PROPERTY_SCALING_FILTER, &error);
      CHECK_ERROR (error, group);

      switch ((NvBufSurfTransform_Inter) val) {
        case NvBufSurfTransformInter_Nearest:
        case NvBufSurfTransformInter_Bilinear:
        case NvBufSurfTransformInter_Algo1:
        case NvBufSurfTransformInter_Algo2:
        case NvBufSurfTransformInter_Algo3:
        case NvBufSurfTransformInter_Algo4:
        case NvBufSurfTransformInter_Default:
          break;
        default:
          g_printerr ("Error. Invalid value for '%s':'%d'\n",
              NVDSPREPROCESS_PROPERTY_SCALING_FILTER, val);
          goto done;
      }
      nvdspreprocess->scaling_pool_interpolation_filter = (NvBufSurfTransform_Inter) val;
      nvdspreprocess->property_set.scaling_pool_interpolation_filter = TRUE;
    }
    else if (!g_strcmp0 (*key, NVDSPREPROCESS_PROPERTY_TENSOR_DATA_TYPE)) {
      int val =  g_key_file_get_integer (key_file, group, *key, &error);
      CHECK_ERROR (error, group);

      switch (val) {
        case NvDsDataType_FP32:
        case NvDsDataType_UINT8:
        case NvDsDataType_INT8:
        case NvDsDataType_UINT32:
        case NvDsDataType_INT32:
        case NvDsDataType_FP16:
        case NvDsDataType_UINT64:
        case NvDsDataType_INT64:
          break;
        default:
          g_printerr ("Error. Invalid value for '%s':'%d'\n",
              NVDSPREPROCESS_PROPERTY_TENSOR_DATA_TYPE, val);
          goto done;
      }
      nvdspreprocess->tensor_params.data_type = (NvDsDataType) val;
      nvdspreprocess->property_set.tensor_data_type = TRUE;
    }
    else if (!g_strcmp0 (*key, NVDSPREPROCESS_PROPERTY_SCALING_POOL_MEMORY_TYPE)) {
      int val =  g_key_file_get_integer (key_file, group, *key, &error);
      CHECK_ERROR (error, group);

      switch ((NvBufSurfaceMemType)val) {
        case NVBUF_MEM_DEFAULT:
        case NVBUF_MEM_CUDA_PINNED:
        case NVBUF_MEM_CUDA_DEVICE:
        case NVBUF_MEM_CUDA_UNIFIED:
        case NVBUF_MEM_SURFACE_ARRAY:
          break;
        default:
          g_printerr ("Error. Invalid value for '%s':'%d'\n", *key, val);
          goto done;
      }
      nvdspreprocess->scaling_pool_memory_type = (NvBufSurfaceMemType) val;
      nvdspreprocess->property_set.scaling_pool_memory_type = TRUE;
    }
    else if (!g_strcmp0 (*key, NVDSPREPROCESS_PROPERTY_SCALING_POOL_COMPUTE_HW)) {
      int val =  g_key_file_get_integer (key_file, group, *key, &error);
      CHECK_ERROR (error, group);

      switch ((NvBufSurfTransform_Compute)val) {
        case NvBufSurfTransformCompute_Default:
        case NvBufSurfTransformCompute_GPU:
#ifdef __aarch64__
        case NvBufSurfTransformCompute_VIC:
#endif
          break;
        default:
          g_printerr ("Error. Invalid value for '%s':'%d'\n", *key, val);
          goto done;
      }
      nvdspreprocess->scaling_pool_compute_hw = (NvBufSurfTransform_Compute) val;
    }
    else if (!g_strcmp0(*key, NVDSPREPROCESS_PROPERTY_TENSOR_NAME)) {
      GET_STRING_PROPERTY(group, *key, nvdspreprocess->tensor_params.tensor_name);
      GST_CAT_INFO (NVDSPREPROCESS_CFG_PARSER_CAT, "Parsed %s=%s in group '%s'\n",
          *key, nvdspreprocess->tensor_params.tensor_name.c_str(), group);
      nvdspreprocess->property_set.tensor_name = TRUE;
    }
    else if (!g_strcmp0(*key, NVDSPREPROCESS_PROPERTY_CUSTOM_LIB_NAME)) {
      gchar* temp = g_key_file_get_string(key_file, group, *key, &error);
      std::string str = temp;
      g_free (temp);
      nvdspreprocess->custom_lib_path = new gchar[_PATH_MAX];
      if (!get_absolute_file_path (cfg_file_path, str.c_str(), nvdspreprocess->custom_lib_path)) {
        g_printerr ("Error: Could not parse custom lib path\n");
        ret = FALSE;
        delete[] nvdspreprocess->custom_lib_path;
        goto done;
      }
      GST_CAT_INFO (NVDSPREPROCESS_CFG_PARSER_CAT, "Parsed %s=%s in group '%s'\n",
          *key, nvdspreprocess->custom_lib_path, group);
      nvdspreprocess->property_set.custom_lib_path = TRUE;
    }
    else if (!g_strcmp0(*key, NVDSPREPROCESS_PROPERTY_TENSOR_PREPARATION_FUNCTION)) {
      GET_STRING_PROPERTY(group, *key, nvdspreprocess->custom_tensor_function_name);
      GST_CAT_INFO (NVDSPREPROCESS_CFG_PARSER_CAT, "Parsed %s=%s in group '%s'\n",
          *key, nvdspreprocess->custom_tensor_function_name.c_str(), group);
      nvdspreprocess->property_set.custom_tensor_function_name = TRUE;
    }
  }


  if (!(nvdspreprocess->property_set.processing_width &&
      nvdspreprocess->property_set.processing_height &&
      nvdspreprocess->property_set.network_input_order &&
      nvdspreprocess->property_set.network_input_shape &&
      nvdspreprocess->property_set.network_color_format &&
      nvdspreprocess->property_set.tensor_data_type &&
      nvdspreprocess->property_set.tensor_name &&
      nvdspreprocess->property_set.custom_lib_path &&
      nvdspreprocess->property_set.custom_tensor_function_name &&
      nvdspreprocess->property_set.scaling_pool_interpolation_filter &&
      nvdspreprocess->property_set.scaling_pool_memory_type)) {
    printf("ERROR: Some preprocess config properties not set\n");
    return FALSE;
  }

  GST_DEBUG_OBJECT (nvdspreprocess, "Parsed Network shape: %d : %d : %d : %d\n",
          nvdspreprocess->tensor_params.network_input_shape[0], nvdspreprocess->tensor_params.network_input_shape[1],
          nvdspreprocess->tensor_params.network_input_shape[2], nvdspreprocess->tensor_params.network_input_shape[3]);

  GST_DEBUG_OBJECT (nvdspreprocess, "Custom Lib = %s\n Custom Tensor Preparation Function = %s\n",
          nvdspreprocess->custom_lib_path, nvdspreprocess->custom_tensor_function_name.c_str());

  ret = TRUE;

done:
  return ret;
}

static gboolean
nvdspreprocess_parse_common_group (GstNvDsPreProcess *nvdspreprocess,
    gchar *cfg_file_path, GKeyFile *key_file, gchar *group, guint64 group_id)
{
  g_autoptr(GError)error = nullptr;
  gboolean ret = FALSE;
  g_auto(GStrv)keys=nullptr;
  GStrv key=nullptr;
  guint64 source_index=0;
  gint *roi_list = nullptr;
  gint *src_list = nullptr;
  gint *class_list = nullptr;
  std::vector <gint> operate_on_class_ids;
  std::vector <gint> src_ids;
  gsize roi_list_len = 0;
  gsize src_list_len = 0;
  gsize class_list_len = 0;
  gint num_roi_per_stream = 0;
  GstNvDsPreProcessGroup *preprocess_group = nullptr;
  guint num_units = 0;
  guint same_roi_for_all_srcs=0;

  preprocess_group = new GstNvDsPreProcessGroup;

  keys = g_key_file_get_keys (key_file, group, nullptr, &error);
  CHECK_ERROR(error, group);

  /**Default value*/
  preprocess_group->draw_roi=1;
  if (nvdspreprocess->process_on_frame) {
    preprocess_group->roi_color={0,1,0,1};
  }
  else {
    preprocess_group->roi_color={0,1,1,1};
  }

  preprocess_group->min_input_object_width=0;
  preprocess_group->min_input_object_height=0;
  preprocess_group->max_input_object_width=0;
  preprocess_group->max_input_object_height=0;
  preprocess_group->replicated_src_id=0;

  for (key = keys; *key; key++){
    if (!g_strcmp0(*key, NVDSPREPROCESS_GROUP_SRC_IDS)) {
      src_list = g_key_file_get_integer_list (key_file, group,*key, &src_list_len, &error);
      if (src_list == nullptr) {
        CHECK_ERROR(error, group);
      }
      src_ids.clear();
      for (gsize icnt = 0; icnt < src_list_len; icnt++){
        src_ids.push_back(src_list[icnt]);
        GST_CAT_INFO (NVDSPREPROCESS_CFG_PARSER_CAT, "Parsed '%s=%d' in group '%s'\n",
          *key, src_list[icnt], group);
      }
      preprocess_group->src_ids = src_ids;
      nvdspreprocess->property_set.src_ids = TRUE;
      g_free(src_list);
      src_list = nullptr;
    }
    else if (!g_strcmp0(*key, NVDSPREPROCESS_GROUP_CUSTOM_INPUT_PREPROCESS_FUNCTION)) {
      GET_STRING_PROPERTY(group, *key, preprocess_group->custom_transform_function_name);
      GST_CAT_INFO (NVDSPREPROCESS_CFG_PARSER_CAT, "Parsed %s=%s in group '%s'\n",
            NVDSPREPROCESS_GROUP_CUSTOM_INPUT_PREPROCESS_FUNCTION,
            preprocess_group->custom_transform_function_name.c_str(), group);
      GST_DEBUG_OBJECT(nvdspreprocess, "Custom Transformation Function = %s\n",
            preprocess_group->custom_transform_function_name.c_str());
    }
    else if (!g_strcmp0(*key, NVDSPREPROCESS_GROUP_OPERATE_ON_CLASS_IDS)) {
      class_list = g_key_file_get_integer_list (key_file, group,*key, &class_list_len, &error);
      if (class_list == nullptr) {
        CHECK_ERROR(error, group);
      }
      operate_on_class_ids.clear();
      for (gsize icnt = 0; icnt < class_list_len; icnt++){
        operate_on_class_ids.push_back(class_list[icnt]);
        GST_CAT_INFO (NVDSPREPROCESS_CFG_PARSER_CAT, "Parsed '%s=%d' in group '%s'\n",
          *key, class_list[icnt], group);
      }
      preprocess_group->operate_on_class_ids = operate_on_class_ids;
      nvdspreprocess->property_set.operate_on_class_ids = TRUE;
      g_free(class_list);
      class_list = nullptr;
    }
    else if (!g_strcmp0 (*key, NVDSPREPROCESS_GROUP_OBJECT_MIN_WIDTH)) {
      guint val = g_key_file_get_integer(key_file, group, *key, &error);
      if ((gint)val < 0) {
        g_printerr ("Error: Negative value specified for %s(%d)\n",*key,val);
        goto done;
      }
      CHECK_ERROR(error, group);
      preprocess_group->min_input_object_width = val;
      GST_CAT_INFO (NVDSPREPROCESS_CFG_PARSER_CAT, "Parsed %s=%d in group '%s'\n",
            *key, preprocess_group->min_input_object_width, group);
      nvdspreprocess->property_set.min_input_object_width = TRUE;
    }
    else if (!g_strcmp0 (*key, NVDSPREPROCESS_GROUP_OBJECT_MIN_HEIGHT)) {
      guint val = g_key_file_get_integer(key_file, group, *key, &error);
      if ((gint)val < 0) {
        g_printerr ("Error: Negative value specified for %s(%d)\n",*key,val);
        goto done;
      }
      CHECK_ERROR(error, group);
      preprocess_group->min_input_object_height = val;
      GST_CAT_INFO (NVDSPREPROCESS_CFG_PARSER_CAT, "Parsed %s=%d in group '%s'\n",
            *key, preprocess_group->min_input_object_height, group);
      nvdspreprocess->property_set.min_input_object_height = TRUE;
    }
    else if (!g_strcmp0 (*key, NVDSPREPROCESS_GROUP_OBJECT_MAX_WIDTH)) {
      guint val = g_key_file_get_integer(key_file, group, *key, &error);
      if ((gint)val < 0) {
        g_printerr ("Error: Negative value specified for %s(%d)\n",*key,val);
        goto done;
      }
      CHECK_ERROR(error, group);
      preprocess_group->max_input_object_width = val;
      GST_CAT_INFO (NVDSPREPROCESS_CFG_PARSER_CAT, "Parsed %s=%d in group '%s'\n",
            *key, preprocess_group->max_input_object_width, group);
      nvdspreprocess->property_set.max_input_object_width = TRUE;
    }
    else if (!g_strcmp0 (*key, NVDSPREPROCESS_GROUP_OBJECT_MAX_HEIGHT)) {
      guint val = g_key_file_get_integer(key_file, group, *key, &error);
      if ((gint)val < 0) {
        g_printerr ("Error: Negative value specified for %s(%d)\n",*key,val);
        goto done;
      }
      CHECK_ERROR(error, group);
      preprocess_group->max_input_object_height = val;
      GST_CAT_INFO (NVDSPREPROCESS_CFG_PARSER_CAT, "Parsed %s=%d in group '%s'\n",
            *key, preprocess_group->max_input_object_height, group);
      nvdspreprocess->property_set.max_input_object_height = TRUE;
    }
    else if (!g_strcmp0 (*key, NVDSPREPROCESS_GROUP_INTERVAL)) {
      guint val = g_key_file_get_integer(key_file, group, *key, &error);
      if ((gint)val < 0) {
        g_printerr ("Error: Negative value specified for %s(%d)\n",*key,val);
        goto done;
      }
      CHECK_ERROR(error, group);
      preprocess_group->interval = val;
      GST_CAT_INFO (NVDSPREPROCESS_CFG_PARSER_CAT, "Parsed %s=%d in group '%s'\n",
            *key, preprocess_group->interval, group);
      nvdspreprocess->property_set.interval = TRUE;
    }
    else if (!g_strcmp0(*key, NVDSPREPROCESS_GROUP_ROI_COLOR)) {
      gsize roi_color_list_len = 0;
      gdouble *roi_color_list = g_key_file_get_double_list (key_file, group,*key, &roi_color_list_len, &error);
      if (roi_color_list == nullptr) {
        CHECK_ERROR(error, group);
      }
      if (roi_color_list_len != 4) {
        g_printerr("Error: Group %s, Number of Color params should be exactly 4 floats {r, g, b, a} between 0 and 1", group);
        goto done;
      }
      preprocess_group->roi_color.red = roi_color_list[0];
      preprocess_group->roi_color.green = roi_color_list[1];
      preprocess_group->roi_color.blue = roi_color_list[2];
      preprocess_group->roi_color.alpha = roi_color_list[3];
      nvdspreprocess->property_set.roi_color = TRUE;
      g_free(roi_color_list);
      roi_color_list = nullptr;
    }
    else  if (!g_strcmp0 (*key, NVDSPREPROCESS_GROUP_DRAW_ROI)) {
      gboolean val = g_key_file_get_boolean(key_file, group, *key, &error);
      CHECK_ERROR(error, group);
      preprocess_group->draw_roi = val;
      GST_CAT_INFO (NVDSPREPROCESS_CFG_PARSER_CAT, "Parsed %s=%d in group '%s'\n",
            *key, preprocess_group->draw_roi, group);
      nvdspreprocess->property_set.draw_roi = TRUE;
    }
    else  if (!g_strcmp0 (*key, NVDSPREPROCESS_GROUP_PROCESS_ON_ROI)) {
      gboolean val = g_key_file_get_boolean(key_file, group, *key, &error);
      CHECK_ERROR(error, group);
      preprocess_group->process_on_roi = val;
      GST_CAT_INFO (NVDSPREPROCESS_CFG_PARSER_CAT, "Parsed %s=%d in group '%s'\n",
            *key, preprocess_group->process_on_roi, group);
      nvdspreprocess->property_set.process_on_roi = TRUE;
    }
    else  if (!g_strcmp0 (*key, NVDSPREPROCESS_GROUP_PROCESS_ON_ALL_OBJECTS)) {
      gboolean val = g_key_file_get_boolean(key_file, group, *key, &error);
      CHECK_ERROR(error, group);
      preprocess_group->process_on_all_objects = val;
      GST_CAT_INFO (NVDSPREPROCESS_CFG_PARSER_CAT, "Parsed %s=%d in group '%s'\n",
            *key, preprocess_group->process_on_all_objects, group);
      nvdspreprocess->property_set.process_on_all_objects = TRUE;
    }
    else if (!strncmp(*key, NVDSPREPROCESS_GROUP_ROI_PARAMS_SRC,
      sizeof(NVDSPREPROCESS_GROUP_ROI_PARAMS_SRC)-1) && ((nvdspreprocess->process_on_frame && preprocess_group->process_on_roi) || (!nvdspreprocess->process_on_frame && !preprocess_group->process_on_all_objects))) {
        EXTRACT_STREAM_ID(key);
        roi_list = g_key_file_get_integer_list (key_file, group,*key, &roi_list_len, &error);
        if (roi_list == nullptr) {
          CHECK_ERROR(error, group);
        }
        GstNvDsPreProcessFrame preprocess_frame;
        /** check if multiple of 4 */
        if ((roi_list_len & 3) == 0) {
          num_roi_per_stream = (int)roi_list_len/4;
        } else {
          printf ("ERROR: %s : roi list length for source %d is not a multiple of 4\n",
                __func__, (int)source_index);
          goto done;
        }
        sum_total_rois += num_roi_per_stream;
        num_units += num_roi_per_stream;

        GST_DEBUG ("Parsing roi-params source_index = %ld num-roi = %d roilistlen = %ld\n",
            source_index, num_roi_per_stream, roi_list_len);

        for (guint i = 0; i < roi_list_len; i=i+4) {
          NvDsRoiMeta roi_info = {{0}};

          roi_info.roi.left = roi_list[i];
          roi_info.roi.top = roi_list[i+1];
          roi_info.roi.width = roi_list[i+2];
          roi_info.roi.height = roi_list[i+3];
          GST_DEBUG ("parsed ROI left=%f top=%f width=%f height=%f\n",
            roi_info.roi.left, roi_info.roi.top, roi_info.roi.width, roi_info.roi.height);
          preprocess_frame.roi_vector.push_back(roi_info);
        }

        if (same_roi_for_all_srcs) {
          /* same roi of replicated_src is used for all the sources within the group*/
          preprocess_group->framemeta_map.emplace(source_index, preprocess_group->framemeta_map[preprocess_group->replicated_src_id]);
        }
        else {
          preprocess_group->framemeta_map.emplace(source_index, preprocess_frame);
        }

        if (preprocess_group->src_ids[0] == -1) {
          same_roi_for_all_srcs=1;
          preprocess_group->replicated_src_id=source_index;
        }

        nvdspreprocess->src_to_group_map->emplace(source_index,group_id);

        GST_CAT_INFO (NVDSPREPROCESS_CFG_PARSER_CAT, "Parsed '%s' in group '%s'\n",
          NVDSPREPROCESS_GROUP_ROI_PARAMS_SRC, group);
        nvdspreprocess->property_set.roi_params_src = TRUE;
        g_free(roi_list);
        roi_list = nullptr;
    }
  }

  if ((nvdspreprocess->process_on_frame && !preprocess_group->process_on_roi)  || (!nvdspreprocess->process_on_frame && preprocess_group->process_on_all_objects)) {
    for (auto & source_index : preprocess_group->src_ids) {
        GstNvDsPreProcessFrame preprocess_frame;
        NvDsRoiMeta roi_info = {{0}};
        preprocess_frame.roi_vector.push_back(roi_info);
        sum_total_rois ++;
        num_units ++;
        preprocess_group->framemeta_map.emplace(source_index, preprocess_frame);

        if (preprocess_group->src_ids[0] == -1) {
          preprocess_group->replicated_src_id=source_index;
        }
        nvdspreprocess->src_to_group_map->emplace(source_index,group_id);
    }
  }

  preprocess_group->num_units = num_units;

  if(nvdspreprocess->process_on_frame){
      if (preprocess_group->process_on_roi) {
        if (!(nvdspreprocess->property_set.src_ids &&
            nvdspreprocess->property_set.process_on_roi &&
            nvdspreprocess->property_set.roi_params_src)) {
          printf("ERROR: Some preprocess group config properties not set in preprocess config file\n");
          goto done;
        }
      }
      else {
        if (!(nvdspreprocess->property_set.src_ids &&
            nvdspreprocess->property_set.process_on_roi)) {
          printf("ERROR: Some preprocess group config properties not set in preprocess config file\n");
          goto done;
        }
      }
  }
  else {
    if (!preprocess_group->process_on_all_objects) {
        if (!(nvdspreprocess->property_set.src_ids &&
            nvdspreprocess->property_set.process_on_all_objects &&
            nvdspreprocess->property_set.roi_params_src)) {
          printf("ERROR: Some preprocess group config properties not set in sgie preprocess config file\n");
          goto done;
        }
      }
      else {
        if (!(nvdspreprocess->property_set.src_ids &&
            nvdspreprocess->property_set.process_on_all_objects)) {
          printf("ERROR: Some preprocess group config properties not set in sgie preprocess config file\n");
          goto done;
        }
      }
  }

  nvdspreprocess->nvdspreprocess_groups.push_back(preprocess_group);
  ret = TRUE;
  preprocess_group = nullptr;
done:
  if (!ret) {
    delete preprocess_group;
  }
  return ret;
}

static gboolean
nvdspreprocess_parse_user_configs(GstNvDsPreProcess *nvdspreprocess,
    gchar *cfg_file_path, GKeyFile *key_file, gchar *group)
{
  g_autoptr(GError)error = nullptr;
  gboolean ret = FALSE;
  g_auto(GStrv)keys=nullptr;
  GStrv key=nullptr;
  std::unordered_map<std::string, std::string> user_configs;

  keys = g_key_file_get_keys (key_file, group, nullptr, &error);
  CHECK_ERROR(error, group);

  for (key = keys; *key; key++){
    gchar* temp = g_key_file_get_string(key_file, group, *key, &error);
    std::string val = temp;
    g_free (temp);
    GST_DEBUG_OBJECT (nvdspreprocess, "parsed user-config key = %s value = %s\n", *key, val.c_str());
    CHECK_ERROR(error, group);
    user_configs.emplace(std::string(*key), val);
  }
  nvdspreprocess->custom_initparams.user_configs = user_configs;
  ret = TRUE;

done:
  return ret;
}

/* Parse the nvdspreprocess config file. Returns FALSE in case of an error. */
gboolean
nvdspreprocess_parse_config_file (GstNvDsPreProcess * nvdspreprocess, gchar * cfg_file_path)
{
  g_autoptr(GError)error = nullptr;
  gboolean ret = FALSE;
  g_auto(GStrv)groups=nullptr;
  GStrv group;
  g_autoptr(GKeyFile) cfg_file = g_key_file_new ();
  guint64 group_index = 0;

  if (!NVDSPREPROCESS_CFG_PARSER_CAT) {
    GstDebugLevel  level;
    GST_DEBUG_CATEGORY_INIT (NVDSPREPROCESS_CFG_PARSER_CAT, "nvdspreprocess", 0,
        NULL);
    level = gst_debug_category_get_threshold (NVDSPREPROCESS_CFG_PARSER_CAT);
    if (level < GST_LEVEL_ERROR )
      gst_debug_category_set_threshold (NVDSPREPROCESS_CFG_PARSER_CAT, GST_LEVEL_ERROR);
  }

  if (!g_key_file_load_from_file (cfg_file, cfg_file_path, G_KEY_FILE_NONE,
          &error)) {
    PARSE_ERROR ("%s", error->message);
  }

  // Check if 'property' group present
  if (!g_key_file_has_group (cfg_file, NVDSPREPROCESS_PROPERTY)) {
    PARSE_ERROR ("Group 'property' not specified");
  }

  g_key_file_set_list_separator (cfg_file,';');

  groups = g_key_file_get_groups (cfg_file, nullptr);

  for (group = groups; *group; group++) {
    GST_CAT_INFO (NVDSPREPROCESS_CFG_PARSER_CAT, "Group found %s \n", *group);
    if (!strcmp(*group, NVDSPREPROCESS_PROPERTY)){
      ret = nvdspreprocess_parse_property_group(nvdspreprocess,
          cfg_file_path, cfg_file, *group);
      if (!ret){
        g_print("NVDSPREPROCESS_CFG_PARSER: Group '%s' parse failed\n", *group);
        goto done;
      }
    }
    else if (!strncmp(*group, NVDSPREPROCESS_GROUP,
            sizeof(NVDSPREPROCESS_GROUP)-1)){
      EXTRACT_GROUP_ID(NVDSPREPROCESS_GROUP);
      GST_DEBUG("parsing group index = %lu\n", group_index);
      ret = nvdspreprocess_parse_common_group (nvdspreprocess,
                cfg_file_path, cfg_file, *group, group_index);
      if (!ret){
        g_print("NVDSPREPROCESS_CFG_PARSER: Group '%s' parse failed\n", *group);
        goto done;
      }
    }
    else if (!strcmp(*group, NVDSPREPROCESS_USER_CONFIGS)){
      GST_DEBUG ("Parsing User Configs\n");
      ret = nvdspreprocess_parse_user_configs (nvdspreprocess,
                cfg_file_path, cfg_file, *group);
      if (!ret){
        g_print("NVDSPREPROCESS_CFG_PARSER: Group '%s' parse failed\n", *group);
        goto done;
      }
    }
    else {
      g_print("NVDSPREPROCESS_CFG_PARSER: Group '%s' ignored\n", *group);
    }
  }

  GST_DEBUG_OBJECT (nvdspreprocess, "network-input-shape[0] = %d, sum-total-rois=%d\n", nvdspreprocess->tensor_params.network_input_shape [0], sum_total_rois);

  nvdspreprocess->max_batch_size = nvdspreprocess->tensor_params.network_input_shape [0];

done:
  return ret;
}
