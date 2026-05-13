/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "nvdspreprocess_yaml_parser.h"
#include "nvdspreprocess_property_parser.h"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdangling-pointer"
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#include <yaml-cpp/yaml.h>
#pragma GCC diagnostic pop


GST_DEBUG_CATEGORY (NVDSPREPROCESS_YAML_PARSER_CAT);

static gint extract_stream_id (const std::string &prefix, const std::string str)
{
  auto prefix_pos = str.find(prefix);
  if (prefix_pos != std::string::npos) {
    size_t start_pos = prefix_pos + prefix.length();
    auto number = str.substr(start_pos);
    return std::strtol(number.c_str(), nullptr, 10);
  }
  return -1;
}

/*Separate a config file entry with delimiters
 *to be able to parse it.*/
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

/* Get the absolute path of a file mentioned in the config given a
 * file path absolute/relative to the config file. */
static gboolean
get_absolute_file_path (
    const gchar * cfg_file_path, const gchar * file_path,
    char *abs_path_str)
{
  gchar abs_cfg_path[PATH_MAX + 1];
  gchar abs_real_file_path[PATH_MAX + 1];
  gchar *abs_file_path = nullptr;
  gchar *delim = nullptr;

  /* Absolute path. No need to resolve further. */
  if (file_path[0] == '/') {
    /* Check if the file exists, return error if not. */
    if (!realpath (file_path, abs_real_file_path)) {
      /* Ignore error if file does not exist and use the unresolved path. */
      if (errno != ENOENT)
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
gst_preprocess_parse_props_yaml (GstNvDsPreProcess *nvdspreprocess, const gchar * cfg_file_path)
{
  gboolean ret = FALSE;

  YAML::Node configyml = YAML::LoadFile(cfg_file_path);
  if(!(configyml.size() > 0))  {
  	g_printerr ("Can't open config file (%s)\n", cfg_file_path);
    goto done;
  }

  if (nvdspreprocess) {
      for(YAML::const_iterator itr = configyml["property"].begin(); itr != configyml["property"].end(); ++itr) {
        std::string paramKey = itr->first.as<std::string>();
        if (paramKey == NVDSPREPROCESS_PROPERTY_ENABLE) {
          auto val = itr->second.as<gboolean>();
          nvdspreprocess->enable = val;
        } else if (paramKey == NVDSPREPROCESS_PROPERTY_UNIQUE_ID) {
          auto val = itr->second.as<guint>();
          nvdspreprocess->unique_id = val;
        } else if (paramKey == NVDSPREPROCESS_PROPERTY_GPU_ID) {
          auto val = itr->second.as<guint>();
          nvdspreprocess->gpu_id = val;
        } else if (paramKey == NVDSPREPROCESS_PROPERTY_PROCESS_ON_FRAME) {
          auto val = itr->second.as<gboolean>();
          nvdspreprocess->process_on_frame = val;
        } else if (paramKey == NVDSPREPROCESS_PROPERTY_MAINTAIN_ASPECT_RATIO) {
          auto val = itr->second.as<gboolean>();
          nvdspreprocess->maintain_aspect_ratio = val;
        } else if (paramKey == NVDSPREPROCESS_PROPERTY_SYMMETRIC_PADDING) {
          auto val = itr->second.as<gboolean>();
          nvdspreprocess->symmetric_padding = val;
        } else if (paramKey == NVDSPREPROCESS_PROPERTY_PROCESSING_WIDTH) {
          auto val = itr->second.as<guint>();
          nvdspreprocess->processing_width = val;
          nvdspreprocess->property_set.processing_width = TRUE;
        } else if (paramKey == NVDSPREPROCESS_PROPERTY_PROCESSING_HEIGHT) {
          auto val = itr->second.as<guint>();
          nvdspreprocess->processing_height = val;
          nvdspreprocess->property_set.processing_height = TRUE;
        } else if (paramKey == NVDSPREPROCESS_PROPERTY_SCALING_BUF_POOL_SIZE) {
          auto val = itr->second.as<guint>();
          nvdspreprocess->scaling_buf_pool_size = val;
        } else if (paramKey == NVDSPREPROCESS_PROPERTY_TENSOR_BUF_POOL_SIZE) {
          auto val = itr->second.as<guint>();
          nvdspreprocess->tensor_buf_pool_size = val;
        } else if (paramKey == NVDSPREPROCESS_PROPERTY_TARGET_IDS) {
          std::string values = itr->second.as<std::string>();
          std::vector<std::string> vec = split_string(values);
          nvdspreprocess->target_unique_ids.clear();
          for (gsize icnt = 0; icnt < vec.size(); icnt++) {
            nvdspreprocess->target_unique_ids.push_back(std::stoul(vec[icnt]));
          }
        } else if (paramKey == NVDSPREPROCESS_PROPERTY_GIE_ID_FOR_OPERATION) {
          auto val = itr->second.as<guint>();
          nvdspreprocess->operate_on_gie_id = val;
        } else if (paramKey == NVDSPREPROCESS_PROPERTY_NETWORK_INPUT_ORDER) {
          guint val = itr->second.as<guint>();
          switch ((NvDsPreProcessNetworkInputOrder) val) {
            case NvDsPreProcessNetworkInputOrder_kNCHW:
            case NvDsPreProcessNetworkInputOrder_kNHWC:
            case NvDsPreProcessNetworkInputOrder_CUSTOM:
              break;
            default:
              g_printerr ("Error. Invalid value for '%s':'%d'\n", paramKey.c_str(), val);
              goto done;
          }
          nvdspreprocess->tensor_params.network_input_order = (NvDsPreProcessNetworkInputOrder) val;
          nvdspreprocess->property_set.network_input_order = TRUE;
        } else if (paramKey == NVDSPREPROCESS_PROPERTY_NETWORK_SHAPE) {
          std::string values = itr->second.as<std::string>();
          std::vector<std::string> vec = split_string(values);
          nvdspreprocess->tensor_params.network_input_shape.clear();
          for (gsize icnt = 0; icnt < vec.size(); icnt++) {
            nvdspreprocess->tensor_params.network_input_shape.push_back(std::stoi(vec[icnt]));
          }
          nvdspreprocess->property_set.network_input_shape = TRUE;
        } else if (paramKey == NVDSPREPROCESS_PROPERTY_NETWORK_COLOR_FORMAT) {
          auto val = itr->second.as<guint>();
          switch ((NvDsPreProcessFormat) val) {
            case NvDsPreProcessFormat_RGB:
            case NvDsPreProcessFormat_BGR:
            case NvDsPreProcessFormat_GRAY:
              break;
            default:
              g_printerr ("Error. Invalid value for '%s':'%d'\n", paramKey.c_str(), val);
              goto done;
          }
          nvdspreprocess->tensor_params.network_color_format = (NvDsPreProcessFormat) val;
          nvdspreprocess->property_set.network_color_format = TRUE;
        } else if (paramKey == NVDSPREPROCESS_PROPERTY_SCALING_FILTER) {
          auto val = itr->second.as<guint>();
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
              g_printerr ("Error. Invalid value for '%s':'%d'\n", paramKey.c_str(), val);
              goto done;
          }
          nvdspreprocess->scaling_pool_interpolation_filter = (NvBufSurfTransform_Inter) val;
          nvdspreprocess->property_set.scaling_pool_interpolation_filter = TRUE;
        } else if (paramKey == NVDSPREPROCESS_PROPERTY_TENSOR_DATA_TYPE) {
          int val = itr->second.as<int>();
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
              g_printerr ("Error. Invalid value for '%s':'%d'\n", paramKey.c_str(), val);
              goto done;
          }
          nvdspreprocess->tensor_params.data_type = (NvDsDataType) val;
          nvdspreprocess->property_set.tensor_data_type = TRUE;
        } else if (paramKey == NVDSPREPROCESS_PROPERTY_SCALING_POOL_MEMORY_TYPE) {
          int val = itr->second.as<int>();
          switch ((NvBufSurfaceMemType) val) {
            case NVBUF_MEM_DEFAULT:
            case NVBUF_MEM_CUDA_PINNED:
            case NVBUF_MEM_CUDA_DEVICE:
            case NVBUF_MEM_CUDA_UNIFIED:
            case NVBUF_MEM_SURFACE_ARRAY:
              break;
            default:
              g_printerr ("Error. Invalid value for '%s':'%d'\n", paramKey.c_str(), val);
              goto done;
          }
          nvdspreprocess->scaling_pool_memory_type = (NvBufSurfaceMemType) val;
          nvdspreprocess->property_set.scaling_pool_memory_type = TRUE;
        } else if (paramKey == NVDSPREPROCESS_PROPERTY_SCALING_POOL_COMPUTE_HW) {
          int val = itr->second.as<int>();
          switch ((NvBufSurfTransform_Compute) val) {
            case NvBufSurfTransformCompute_Default:
            case NvBufSurfTransformCompute_GPU:
#ifdef __aarch64__
            case NvBufSurfTransformCompute_VIC:
#endif
              break;
            default:
              g_printerr ("Error. Invalid value for '%s':'%d'\n", paramKey.c_str(), val);
              goto done;
          }
          nvdspreprocess->scaling_pool_compute_hw = (NvBufSurfTransform_Compute) val;
        } else if (paramKey == NVDSPREPROCESS_PROPERTY_TENSOR_NAME) {
          auto val = itr->second.as<std::string>();
          nvdspreprocess->tensor_params.tensor_name = val;
          nvdspreprocess->property_set.tensor_name = TRUE;
        } else if (paramKey == NVDSPREPROCESS_PROPERTY_CUSTOM_LIB_NAME) {
          auto val = itr->second.as<std::string>();
          nvdspreprocess->custom_lib_path = new gchar[_PATH_MAX];
          if (!get_absolute_file_path (cfg_file_path, val.c_str(), nvdspreprocess->custom_lib_path)) {
            g_printerr ("Error: Could not parse custom lib path\n");
            goto done;
          }
          nvdspreprocess->property_set.custom_lib_path = TRUE;
        } else if (paramKey == NVDSPREPROCESS_PROPERTY_TENSOR_PREPARATION_FUNCTION) {
          nvdspreprocess->custom_tensor_function_name  = itr->second.as<std::string>();
          nvdspreprocess->property_set.custom_tensor_function_name = TRUE;
        }
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
    g_printerr ("ERROR: Some preprocess config properties not set\n");
    goto done;
  }

  GST_CAT_DEBUG (NVDSPREPROCESS_YAML_PARSER_CAT, "Property parsed successfully:");
  GST_CAT_DEBUG (NVDSPREPROCESS_YAML_PARSER_CAT, "Enable: %d", nvdspreprocess->enable);
  GST_CAT_DEBUG (NVDSPREPROCESS_YAML_PARSER_CAT, "Unique ID: %d", nvdspreprocess->unique_id);
  GST_CAT_DEBUG (NVDSPREPROCESS_YAML_PARSER_CAT, "GPU ID: %d", nvdspreprocess->gpu_id);
  GST_CAT_DEBUG (NVDSPREPROCESS_YAML_PARSER_CAT, "Process on frame: %d", nvdspreprocess->process_on_frame);
  GST_CAT_DEBUG (NVDSPREPROCESS_YAML_PARSER_CAT, "Maintain aspect ratio: %d", nvdspreprocess->maintain_aspect_ratio);
  GST_CAT_DEBUG (NVDSPREPROCESS_YAML_PARSER_CAT, "Symmetric padding: %d", nvdspreprocess->symmetric_padding);
  GST_CAT_DEBUG (NVDSPREPROCESS_YAML_PARSER_CAT, "Processing width: %d", nvdspreprocess->processing_width);
  GST_CAT_DEBUG (NVDSPREPROCESS_YAML_PARSER_CAT, "Processing height: %d", nvdspreprocess->processing_height);
  GST_CAT_DEBUG (NVDSPREPROCESS_YAML_PARSER_CAT, "Scaling buffer pool size: %d", nvdspreprocess->scaling_buf_pool_size);
  GST_CAT_DEBUG (NVDSPREPROCESS_YAML_PARSER_CAT, "Tensor buffer pool size: %d", nvdspreprocess->tensor_buf_pool_size);
  GST_CAT_DEBUG (NVDSPREPROCESS_YAML_PARSER_CAT, "Target IDs:");
  for (auto id : nvdspreprocess->target_unique_ids) {
    GST_CAT_DEBUG (NVDSPREPROCESS_YAML_PARSER_CAT, "%lu ", id);
  }
  GST_CAT_DEBUG (NVDSPREPROCESS_YAML_PARSER_CAT, "GIE ID for operation: %d", nvdspreprocess->operate_on_gie_id);
  GST_CAT_DEBUG (NVDSPREPROCESS_YAML_PARSER_CAT, "Network input order: %d", nvdspreprocess->tensor_params.network_input_order);
  GST_CAT_DEBUG (NVDSPREPROCESS_YAML_PARSER_CAT, "Network input shape:");
  for (auto dim : nvdspreprocess->tensor_params.network_input_shape) {
    GST_CAT_DEBUG (NVDSPREPROCESS_YAML_PARSER_CAT, "%d ", dim);
  }
  GST_CAT_DEBUG (NVDSPREPROCESS_YAML_PARSER_CAT, "Network color format: %d", nvdspreprocess->tensor_params.network_color_format);
  GST_CAT_DEBUG (NVDSPREPROCESS_YAML_PARSER_CAT, "Scaling pool interpolation filter: %d", nvdspreprocess->scaling_pool_interpolation_filter);
  GST_CAT_DEBUG (NVDSPREPROCESS_YAML_PARSER_CAT, "Tensor data type: %d", nvdspreprocess->tensor_params.data_type);
  GST_CAT_DEBUG (NVDSPREPROCESS_YAML_PARSER_CAT, "Scaling pool memory type: %d", nvdspreprocess->scaling_pool_memory_type);
  GST_CAT_DEBUG (NVDSPREPROCESS_YAML_PARSER_CAT, "Scaling pool compute hardware: %d", nvdspreprocess->scaling_pool_compute_hw);
  GST_CAT_DEBUG (NVDSPREPROCESS_YAML_PARSER_CAT, "Tensor name: %s", nvdspreprocess->tensor_params.tensor_name.c_str());
  GST_CAT_DEBUG (NVDSPREPROCESS_YAML_PARSER_CAT, "Custom lib path: %s", nvdspreprocess->custom_lib_path);
  GST_CAT_DEBUG (NVDSPREPROCESS_YAML_PARSER_CAT, "Custom tensor preparation function: %s", nvdspreprocess->custom_tensor_function_name.c_str());

  ret = TRUE;

done:
  return ret;
}

static gboolean
gst_preprocess_parse_user_configs_yaml (GstNvDsPreProcess *nvdspreprocess, const gchar * cfg_file_path)
{
  std::unordered_map<std::string, std::string> user_configs;
  YAML::Node configyml = YAML::LoadFile(cfg_file_path);
  if(!(configyml.size() > 0))  {
  	g_printerr ("Can't open config file (%s)\n", cfg_file_path);
    return FALSE;
  }

  for(YAML::const_iterator itr = configyml["user-configs"].begin(); itr != configyml["user-configs"].end(); ++itr) {
    std::string paramKey = itr->first.as<std::string>();
    auto val = itr->second.as<std::string>();
    GST_CAT_DEBUG (NVDSPREPROCESS_YAML_PARSER_CAT, "User config: %s = %s", paramKey.c_str(), val.c_str());
    user_configs.emplace(std::string(paramKey), std::string(val));
  }
  nvdspreprocess->custom_initparams.user_configs = user_configs;
  return TRUE;
}

static gboolean
gst_preprocess_parse_common_group_yaml (GstNvDsPreProcess *nvdspreprocess, YAML::Node group, guint64 group_index)
{
  gboolean ret = FALSE;
  // Default values
  GstNvDsPreProcessGroup *preprocess_group = new GstNvDsPreProcessGroup;
  preprocess_group->draw_roi = 1;
  if (nvdspreprocess->process_on_frame) {
    preprocess_group->roi_color={0,1,0,1};
  }
  else {
    preprocess_group->roi_color={0,1,1,1};
  }
  preprocess_group->min_input_object_width = 0;
  preprocess_group->min_input_object_height = 0;
  preprocess_group->max_input_object_width = 0;
  preprocess_group->max_input_object_height = 0;
  preprocess_group->replicated_src_id = 0;
  gint num_roi_per_stream = 0;
  gint num_units = 0;
  const std::string roi_params_src = "roi-params-src-";
  guint same_roi_for_all_srcs=0;

  for (YAML::const_iterator itr = group.begin(); itr != group.end(); ++itr) {
    std::string groupKey = itr->first.as<std::string>();
    if (groupKey == NVDSPREPROCESS_GROUP_SRC_IDS) {
      std::string values = itr->second.as<std::string>();
      std::vector<std::string> vec = split_string(values);
      preprocess_group->src_ids.clear();
      for (gsize icnt = 0; icnt < vec.size(); icnt++) {
        preprocess_group->src_ids.push_back(std::stoi(vec[icnt]));
      }
      nvdspreprocess->property_set.src_ids = TRUE;
    } else if (groupKey == NVDSPREPROCESS_GROUP_CUSTOM_INPUT_PREPROCESS_FUNCTION) {
      preprocess_group->custom_transform_function_name = itr->second.as<std::string>();
    } else if (groupKey == NVDSPREPROCESS_GROUP_OPERATE_ON_CLASS_IDS) {
      std::string values = itr->second.as<std::string>();
      std::vector<std::string> vec = split_string(values);
      preprocess_group->operate_on_class_ids.clear();
      for (gsize icnt = 0; icnt < vec.size(); icnt++) {
        preprocess_group->operate_on_class_ids.push_back(std::stoi(vec[icnt]));
      }
      nvdspreprocess->property_set.operate_on_class_ids = TRUE;
    } else if (groupKey == NVDSPREPROCESS_GROUP_OBJECT_MIN_WIDTH) {
      preprocess_group->min_input_object_width = itr->second.as<gint>();
      nvdspreprocess->property_set.min_input_object_width = TRUE;
    } else if (groupKey == NVDSPREPROCESS_GROUP_OBJECT_MIN_HEIGHT) {
      preprocess_group->min_input_object_height = itr->second.as<gint>();
      nvdspreprocess->property_set.min_input_object_height = TRUE;
    } else if (groupKey == NVDSPREPROCESS_GROUP_OBJECT_MAX_WIDTH) {
      preprocess_group->max_input_object_width = itr->second.as<gint>();
      nvdspreprocess->property_set.max_input_object_width = TRUE;
    } else if (groupKey == NVDSPREPROCESS_GROUP_OBJECT_MAX_HEIGHT) {
      preprocess_group->max_input_object_height = itr->second.as<gint>();
      nvdspreprocess->property_set.max_input_object_height = TRUE;
    } else if (groupKey == NVDSPREPROCESS_GROUP_INTERVAL) {
      preprocess_group->interval = itr->second.as<guint>();
      nvdspreprocess->property_set.interval = TRUE;
    } else if (groupKey == NVDSPREPROCESS_GROUP_ROI_COLOR) {
      std::string values = itr->second.as<std::string>();
      std::vector<std::string> vec = split_string(values);
      preprocess_group->roi_color.red = std::stod(vec[0]);
      preprocess_group->roi_color.green = std::stod(vec[1]);
      preprocess_group->roi_color.blue = std::stod(vec[2]);
      preprocess_group->roi_color.alpha = std::stod(vec[3]);
      nvdspreprocess->property_set.roi_color = TRUE;
    } else if (groupKey == NVDSPREPROCESS_GROUP_DRAW_ROI) {
      preprocess_group->draw_roi = itr->second.as<gboolean>();
      nvdspreprocess->property_set.draw_roi = TRUE;
    } else if (groupKey == NVDSPREPROCESS_GROUP_PROCESS_ON_ROI) {
      preprocess_group->process_on_roi = itr->second.as<gboolean>();
      nvdspreprocess->property_set.process_on_roi = TRUE;
    } else if (groupKey == NVDSPREPROCESS_GROUP_PROCESS_ON_ALL_OBJECTS) {
      preprocess_group->process_on_all_objects = itr->second.as<gboolean>();
      nvdspreprocess->property_set.process_on_all_objects = TRUE;
    } else if (groupKey.compare(0, roi_params_src.size(), roi_params_src) == 0) {
      if ((nvdspreprocess->process_on_frame && preprocess_group->process_on_roi) ||
            (!nvdspreprocess->process_on_frame && !preprocess_group->process_on_all_objects)) {
        std::string str = itr->second.as<std::string>();
        std::vector<std::string> vec = split_string(str);
        gint roi_list_len = vec.size();
        gint source_index = extract_stream_id(roi_params_src, groupKey);
        if (source_index < 0) {
          g_printerr ("ERROR: %s : source index not found for group %ld\n", __func__, group_index);
          goto done;
        }
        if ((roi_list_len & 3) == 0) {
          num_roi_per_stream = (int)roi_list_len/4;
        } else {
          g_printerr ("ERROR: %s : roi list length for source %d is not a multiple of 4\n",
              __func__, (int)source_index);
          goto done;
        }
        num_units += num_roi_per_stream;
        GstNvDsPreProcessFrame preprocess_frame;
        GST_CAT_DEBUG (NVDSPREPROCESS_YAML_PARSER_CAT, "Parsing roi-params source_index = %d num-roi = %d roilistlen = %d",
            source_index, num_roi_per_stream, roi_list_len);

        for (gint i = 0; i < roi_list_len; i=i+4) {
          NvDsRoiMeta roi_info = {{0}};

          roi_info.roi.left = std::stoi(vec[i]);
          roi_info.roi.top = std::stoi(vec[i+1]);
          roi_info.roi.width = std::stoi(vec[i+2]);
          roi_info.roi.height = std::stoi(vec[i+3]);
          GST_CAT_DEBUG (NVDSPREPROCESS_YAML_PARSER_CAT, "parsed ROI left=%f top=%f width=%f height=%f",
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

        nvdspreprocess->src_to_group_map->emplace(source_index,group_index);
        nvdspreprocess->property_set.roi_params_src = TRUE;
      } else {
        g_printerr ("ERROR: %s : roi-params-src is not valid for group %ld, ignored\n", __func__, group_index);
      }
    } else {
      g_printerr ("Unknown group key: %s\n", groupKey.c_str());
      goto done;
    }
  }

  GST_CAT_DEBUG (NVDSPREPROCESS_YAML_PARSER_CAT, "Group '%ld' successfully parsed:", group_index);
  GST_CAT_DEBUG (NVDSPREPROCESS_YAML_PARSER_CAT, "Group draw_roi: %d", preprocess_group->draw_roi);
  GST_CAT_DEBUG (NVDSPREPROCESS_YAML_PARSER_CAT, "Group process_on_roi: %d", preprocess_group->process_on_roi);
  GST_CAT_DEBUG (NVDSPREPROCESS_YAML_PARSER_CAT, "Group process_on_all_objects: %d", preprocess_group->process_on_all_objects);
  GST_CAT_DEBUG (NVDSPREPROCESS_YAML_PARSER_CAT, "Group replicated_src_id: %d", preprocess_group->replicated_src_id);
  GST_CAT_DEBUG (NVDSPREPROCESS_YAML_PARSER_CAT, "Group src_ids:");
  for (auto id : preprocess_group->src_ids) {
    GST_CAT_DEBUG (NVDSPREPROCESS_YAML_PARSER_CAT, "%d ", id);
  }
  GST_CAT_DEBUG (NVDSPREPROCESS_YAML_PARSER_CAT, "Group custom_transform_function_name: %s", preprocess_group->custom_transform_function_name.c_str());
  GST_CAT_DEBUG (NVDSPREPROCESS_YAML_PARSER_CAT, "Group operate_on_class_ids:");
  for (auto id : preprocess_group->operate_on_class_ids) {
    GST_CAT_DEBUG (NVDSPREPROCESS_YAML_PARSER_CAT, "%d ", id);
  }

  if ((nvdspreprocess->process_on_frame && !preprocess_group->process_on_roi)  || (!nvdspreprocess->process_on_frame && preprocess_group->process_on_all_objects)) {
    for (auto & source_index : preprocess_group->src_ids) {
        GstNvDsPreProcessFrame preprocess_frame;
        NvDsRoiMeta roi_info = {{0}};
        preprocess_frame.roi_vector.push_back(roi_info);
        num_units ++;
        preprocess_group->framemeta_map.emplace(source_index, preprocess_frame);

        if (preprocess_group->src_ids[0] == -1) {
          preprocess_group->replicated_src_id=source_index;
        }
        nvdspreprocess->src_to_group_map->emplace(source_index, group_index);
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

gboolean gst_nvdspreprocess_parse_config_file_yaml (GstNvDsPreProcess *nvdspreprocess, const gchar * cfg_file_path) {
  gboolean ret = FALSE;

  if (!NVDSPREPROCESS_YAML_PARSER_CAT) {
    GST_DEBUG_CATEGORY_INIT (NVDSPREPROCESS_YAML_PARSER_CAT, "nvdspreprocess", 0,
        NULL);
  }

  YAML::Node configyml = YAML::LoadFile(cfg_file_path);
  if(!(configyml.size() > 0))  {
  	g_printerr ("Can't open config file (%s)\n", cfg_file_path);
    goto done;
  }

  /* 'property' group is mandatory. */
  if(configyml["property"]) {
    if (!gst_preprocess_parse_props_yaml (nvdspreprocess, cfg_file_path)) {
      g_printerr ("Failed to parse group property\n");
      goto done;
    }
  } else {
    g_printerr ("Could not find group property\n");
    goto done;
  }

  for (YAML::const_iterator itr = configyml.begin(); itr != configyml.end(); ++itr) {
    std::string groupKey = itr->first.as<std::string>();
    if (groupKey == "user-configs") {
      if (!gst_preprocess_parse_user_configs_yaml (nvdspreprocess, cfg_file_path)) {
        g_printerr ("Failed to parse group\n");
        goto done;
      }
    } else if (groupKey == "group") {
      auto group_index = 0;
      for (const auto &group : itr->second) {
        if (!gst_preprocess_parse_common_group_yaml (nvdspreprocess, group, group_index)) {
          g_printerr ("Failed to parse group\n");
          goto done;
        }
        group_index++;
      }
      GST_CAT_DEBUG (NVDSPREPROCESS_YAML_PARSER_CAT, " Totally %ld groups added:", nvdspreprocess->nvdspreprocess_groups.size());
      group_index = 0;
      for (auto & group : nvdspreprocess->nvdspreprocess_groups) {
        GST_CAT_DEBUG (NVDSPREPROCESS_YAML_PARSER_CAT, "Total %d units in group '%d':", group->num_units, group_index++);
        for (auto & frame : group->framemeta_map) {
          GST_CAT_DEBUG (NVDSPREPROCESS_YAML_PARSER_CAT, "Source %d has %ld rois", frame.first, frame.second.roi_vector.size());
        }
      }
    } else if (groupKey != "property") {
      g_printerr ("Unknown group: %s\n", groupKey.c_str());
      goto done;
    }
  }
  nvdspreprocess->max_batch_size = nvdspreprocess->tensor_params.network_input_shape [0];
  ret = TRUE;

done:
  if (!ret) {
    g_printerr ("** ERROR: <%s:%d>: failed\n", __func__, __LINE__);
  }
  return ret;

}