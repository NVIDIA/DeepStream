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

#include <gst/gst.h>
#include <assert.h>

#include "gstnvinfer_yaml_parser.h"
#include "gstnvinfer.h"
#include "nvdsinfer_context.h"
#include <yaml-cpp/yaml.h>

#include <string>
#include <iostream>
#include <cstring>

using std::cout;
using std::endl;

extern const int DEFAULT_REINFER_INTERVAL;

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

/* Parse per-class detection parameters. Returns FALSE in case of an error. */
static gboolean
gst_nvinfer_parse_class_attrs (const gchar * cfg_file_path, std::string group_str,
    NvDsInferDetectionParams & detection_params,
    GstNvInferDetectionFilterParams & detection_filter_params,
    GstNvInferColorParams & color_params)
{
  gboolean ret = FALSE;

  YAML::Node configyml = YAML::LoadFile(cfg_file_path);
  const char *group = group_str.c_str();

  for(YAML::const_iterator itr = configyml[group_str].begin(); itr != configyml[group_str].end(); ++itr)
  {
    std::string paramKey = itr->first.as<std::string>();
    if (paramKey ==  "pre-cluster-threshold") {
        detection_params.preClusterThreshold =
          itr->second.as<double>();
      if (detection_params.preClusterThreshold < 0) {
        g_printerr ("Error: Negative pre cluster threshold (%.5f) specified for group %s\n",
            detection_params.preClusterThreshold, group);
        goto done;
      }
    } else if (paramKey == "post-cluster-threshold") {
        detection_params.postClusterThreshold =
          itr->second.as<double>();
      if (detection_params.postClusterThreshold < 0) {
        g_printerr ("Error: Negative post cluster threshold (%.5f) specified for group %s\n",
            detection_params.postClusterThreshold, group);
        goto done;
      }
    } else if (paramKey == "eps") {
      detection_params.eps =
          itr->second.as<double>();
      if (detection_params.eps < 0) {
        g_printerr ("Error: Negative eps (%.5f) specified for group %s\n",
            detection_params.eps, group);
        goto done;
      }
    } else if (paramKey == "group-threshold") {
      detection_params.groupThreshold =
          itr->second.as<int>();
      if (detection_params.groupThreshold < 0) {
        g_printerr
            ("Error: Negative group-threshold (%d) specified for group %s\n",
            detection_params.groupThreshold, group);
        goto done;
      }
    } else if (paramKey == "minBoxes") {
      detection_params.minBoxes =
          itr->second.as<int>();
      if (detection_params.minBoxes < 0) {
        g_printerr
            ("Error: Negative minBoxes (%d) specified for group %s\n",
            detection_params.minBoxes, group);
        goto done;
      }
    } else if (paramKey == "dbscan-min-score") {
      detection_params.minScore =
          itr->second.as<double>();
      if (detection_params.minScore < 0) {
        g_printerr
            ("Error: Negative minScore (%f) specified for group %s\n",
            detection_params.minScore, group);
        goto done;
      }
    } else if (paramKey == "roi-top-offset") {
      detection_filter_params.roiTopOffset =
          itr->second.as<int>();
      if ((gint) detection_filter_params.roiTopOffset < 0) {
        g_printerr
            ("Error: Negative roiTopOffset (%d) specified for group %s\n",
            detection_filter_params.roiTopOffset, group);
        goto done;
      }
    } else if (paramKey == "roi-bottom-offset") {
      detection_filter_params.roiBottomOffset =
          itr->second.as<int>();
      if ((gint) detection_filter_params.roiBottomOffset < 0) {
        g_printerr
            ("Error: Negative roiBottomOffset (%d) specified for group %s\n",
            detection_filter_params.roiBottomOffset, group);
        goto done;
      }
    } else if (paramKey == "detected-min-w") {
      detection_filter_params.detectionMinWidth =
          itr->second.as<int>();
      if ((gint) detection_filter_params.detectionMinWidth < 0) {
        g_printerr
            ("Error: Negative detectionMinWidth (%d) specified for group %s\n",
            detection_filter_params.detectionMinWidth, group);
        goto done;
      }
    } else if (paramKey == "detected-min-h") {
      detection_filter_params.detectionMinHeight =
          itr->second.as<int>();
      if ((gint) detection_filter_params.detectionMinHeight < 0) {
        g_printerr
            ("Error: Negative detectionMinHeight (%d) specified for group %s\n",
            detection_filter_params.detectionMinHeight, group);
        goto done;
      }
    } else if (paramKey == "detected-max-w") {
      detection_filter_params.detectionMaxWidth =
          itr->second.as<int>();
      if ((gint) detection_filter_params.detectionMaxWidth < 0) {
        g_printerr
            ("Error: Negative detectionMaxWidth (%d) specified for group %s\n",
            detection_filter_params.detectionMaxWidth, group);
        goto done;
      }
    } else if (paramKey == "detected-max-h") {
      detection_filter_params.detectionMaxHeight =
          itr->second.as<int>();
      if ((gint) detection_filter_params.detectionMaxHeight < 0) {
        g_printerr
            ("Error: Negative detectionMaxHeight (%d) specified for group %s\n",
            detection_filter_params.detectionMaxHeight, group);
        goto done;
      }
    } else if (paramKey == "border-color") {
      std::string values = itr->second.as<std::string>();
      std::vector<std::string> vec = split_string(values);

      if (vec.size() != 4) {
        g_printerr
            ("Error: Group %s, Number of Color params should be exactly 4 "
            "floats {r, g, b, a} between 0 and 1", group);
        goto done;
      }
      color_params.border_color.red = std::stod(vec[0]);
      color_params.border_color.green = std::stod(vec[1]);
      color_params.border_color.blue = std::stod(vec[2]);
      color_params.border_color.alpha = std::stod(vec[3]);

    } else if (paramKey == "bg-color") {
      std::string values = itr->second.as<std::string>();
      std::vector<std::string> vec = split_string(values);

      if (vec.size() != 4) {
        g_printerr
            ("Error: Group %s, Number of Color params should be exactly 4 "
            "floats {r, g, b, a} between 0 and 1", group);
        goto done;
      }
      color_params.bg_color.red = std::stod(vec[0]);
      color_params.bg_color.green = std::stod(vec[1]);
      color_params.bg_color.blue = std::stod(vec[2]);
      color_params.bg_color.alpha = std::stod(vec[3]);
      color_params.have_bg_color = TRUE;

    } else if (paramKey == "nms-iou-threshold") {
        detection_params.nmsIOUThreshold =  itr->second.as<double>();
        if (detection_params.nmsIOUThreshold < 0 || detection_params.nmsIOUThreshold > 1) {
            g_printerr ("Error: Invalid nms iou threshold (%.2f) specified for group %s."
                         "Enter a value between 0 & 1. \n",
            detection_params.nmsIOUThreshold, group);
        goto done;
      }
    } else if (paramKey == "topk") {
        detection_params.topK = itr->second.as<int>();
        if(detection_params.topK < 0) {
            g_printerr("Error: Invalid topk value %d specified for group %s."
            " topk should be greater than of equal to '0'\n", detection_params.topK, group);
            goto done;
        }
    } else {
      g_printerr ("Unknown key '%s' for group [%s]\n", paramKey.c_str(),
          group);
    }
  }

  ret = TRUE;
done:
  return ret;
}

static gboolean
gst_nvinfer_parse_other_attribute_yaml (GstNvInfer * nvinfer,
    const gchar * cfg_file_path, std::vector<std::string> pair)
{
  gboolean ret = FALSE;

  assert (nvinfer);

  if (pair[0] == "process-mode") {
    if ((*nvinfer->is_prop_set)[PROP_PROCESS_MODE])
      return TRUE;
    guint val = std::stoi(pair[1]);

    switch (val) {
      case 1:
        nvinfer->process_full_frame = TRUE;
        break;
      case 2:
        nvinfer->process_full_frame = FALSE;
        break;
      default:
        g_printerr ("Error: Invalid value for process-mode (%d)\n",
            val);
        goto done;
    }
  } else if (pair[0] == "classifier-async-mode") {
      nvinfer->classifier_async_mode = std::stoi(pair[1]);
  } else if (pair[0] == "classifier-type") {
    char* str2 = (char*) malloc(sizeof(char) * 64);
    std::strncpy (str2, pair[1].c_str(), 63);
    nvinfer->classifier_type = str2;
  } else if (pair[0] == "interval") {
    if ((*nvinfer->is_prop_set)[PROP_INTERVAL])
      return TRUE;
    nvinfer->interval = std::stoi(pair[1]);
    if ((gint) nvinfer->interval < 0) {
      g_printerr ("Error: Negative value (%d) specified for interval\n",
          nvinfer->interval);
      goto done;
    }
  } else if (pair[0] == "output-tensor-meta") {
    nvinfer->output_tensor_meta = std::stoi(pair[1]);
  } else if (pair[0] == "output-instance-mask") {
    nvinfer->output_instance_mask = std::stoi(pair[1]);
  } else if (pair[0] == "secondary-reinfer-interval") {
    nvinfer->secondary_reinfer_interval = std::stoi(pair[1]);
  } else if (pair[0] == "maintain-aspect-ratio") {
    nvinfer->maintain_aspect_ratio = std::stoi(pair[1]);
  } else if (pair[0] == "symmetric-padding") {
    nvinfer->symmetric_padding = std::stoi(pair[1]);
  } else if (pair[0] == "input-object-min-width") {
    nvinfer->min_input_object_width = std::stoi(pair[1]);
    if ((gint) nvinfer->min_input_object_width < 0) {
      g_printerr ("Error: Negative value specified for input-object-min-width (%d)\n",
          nvinfer->min_input_object_width);
      goto done;
    }
  } else if (pair[0] == "input-object-min-height") {
    nvinfer->min_input_object_height = std::stoi(pair[1]);
    if ((gint) nvinfer->min_input_object_height < 0) {
      g_printerr ("Error: Negative value specified for input-object-min-height (%d)\n",
          nvinfer->min_input_object_height);
      goto done;
    }
  } else if (pair[0] == "input-object-max-width") {
    nvinfer->max_input_object_width = std::stoi(pair[1]);
    if ((gint) nvinfer->max_input_object_width < 0) {
      g_printerr ("Error: Negative value specified for input-object-max-width (%d)\n",
          nvinfer->max_input_object_width);
      goto done;
    }
  } else if (pair[0] == "input-object-max-height") {
    nvinfer->max_input_object_height = std::stoi(pair[1]);
    if ((gint) nvinfer->max_input_object_height < 0) {
      g_printerr ("Error: Negative value specified for input-object-max-height (%d)\n",
          nvinfer->max_input_object_height);
      goto done;
    }
  } else if (pair[0] == "operate-on-gie-id") {
    if ((*nvinfer->is_prop_set)[PROP_OPERATE_ON_GIE_ID] ||
        (*nvinfer->is_prop_set)[PROP_OPERATE_ON_CLASS_IDS])
      return TRUE;
    nvinfer->operate_on_gie_id = std::stoi(pair[1]);
  } else if (pair[0] == "operate-on-class-ids") {
    if ((*nvinfer->is_prop_set)[PROP_OPERATE_ON_GIE_ID] ||
        (*nvinfer->is_prop_set)[PROP_OPERATE_ON_CLASS_IDS])
      return TRUE;
    gint max_class_id = -1;
    std::vector<std::string> vec = split_string(pair[1]);

    for(auto& j : vec) {
      if (std::stoi(j) > max_class_id)
        max_class_id = std::stoi(j);
    }
    nvinfer->operate_on_class_ids->assign (max_class_id + 1, FALSE);
    for(auto& j : vec) {
      nvinfer->operate_on_class_ids->at (std::stoi(j)) = TRUE;
    }
  } else if (pair[0] == "filter-out-class-ids") {
    std::vector<std::string> vec = split_string(pair[1]);

    for(auto& j : vec) {
      nvinfer->filter_out_class_ids->insert(std::stoul(j));
    }
  } else if (pair[0] == "scaling-compute-hw") {
    int val = std::stoi(pair[1]);
    switch (val) {
      case NvBufSurfTransformCompute_Default:
      case NvBufSurfTransformCompute_GPU:
#ifdef __aarch64__
      case NvBufSurfTransformCompute_VIC:
#endif
        break;
      default:
        g_printerr ("Error. Invalid value for scaling-compute-hw:'%d'\n",
            val);
        goto done;
    }
    nvinfer->transform_config_params.compute_mode = (NvBufSurfTransform_Compute) val;
  } else if (pair[0] == "scaling-filter") {
    int val = std::stoi(pair[1]);
    switch (val) {
      case NvBufSurfTransformInter_Nearest:
      case NvBufSurfTransformInter_Bilinear:
      case NvBufSurfTransformInter_Algo1:
      case NvBufSurfTransformInter_Algo2:
      case NvBufSurfTransformInter_Algo3:
      case NvBufSurfTransformInter_Algo4:
      case NvBufSurfTransformInter_Default:
        break;
      default:
        g_printerr ("Error. Invalid value for scaling-filter:'%d'\n",
            val);
        goto done;
    }
    nvinfer->transform_params.transform_filter = (NvBufSurfTransform_Inter) val;
  } else if (pair[0] == "crop-objects-to-roi-boundary") {
    nvinfer->crop_objects_to_roi_boundary = std::stoi(pair[1]);
  } else {
      g_printerr ("Unknown or legacy key specified '%s' for group property\n", pair[0].c_str());
    }

  ret = TRUE;

done:
  return ret;
}

/* Parse 'property' group. Returns FALSE in case of an error. If any of the
 * properties are set through the GObject set method this function does not
 * parse those properties i.e. values set through g_object_set override the
 * corresponding properties in the config file. */
static gboolean
gst_nvinfer_parse_props_yaml (GstNvInfer * nvinfer,
    NvDsInferContextInitParams * init_params,
    const gchar * cfg_file_path)
{
  gboolean ret = FALSE;

  assert (init_params != nullptr);

  YAML::Node configyml = YAML::LoadFile(cfg_file_path);
  if(!(configyml.size() > 0))  {
  	cout << "Can't open config file (" << cfg_file_path << ")" << endl;
  }

  if (nvinfer)
    nvinfer->secondary_reinfer_interval = DEFAULT_REINFER_INTERVAL;

  init_params->networkInputFormat = NvDsInferFormat_RGB;

  for(YAML::const_iterator itr = configyml["property"].begin(); itr != configyml["property"].end(); ++itr)
  {
    std::string paramKey = itr->first.as<std::string>();
    if (paramKey == "gie-unique-id") {
      if (nvinfer && (*nvinfer->is_prop_set)[PROP_UNIQUE_ID])
        continue;
      init_params->uniqueID = itr->second.as<unsigned int>();

      if (init_params->uniqueID <= 0) {
        g_printerr ("Error: gie-unique-id (%d) should be > 0\n", nvinfer->unique_id);
        goto done;
      }
      if (nvinfer)
        nvinfer->unique_id = init_params->uniqueID;
    } else if (paramKey == "labelfile-path") {
      std::string temp = itr->second.as<std::string>();

      if (!get_absolute_file_path (cfg_file_path, temp.c_str(),
              init_params->labelsFilePath)) {
        g_printerr ("Error: Could not parse labels file path\n");
        goto done;
      }
    } else if (paramKey == "gpu-id") {
      if (nvinfer && (*nvinfer->is_prop_set)[PROP_GPU_DEVICE_ID])
        continue;
      gint devices;

      init_params->gpuID = itr->second.as<unsigned int>();

      if (cudaGetDeviceCount (&devices) != cudaSuccess) {
        g_printerr ("Error: Could not get cuda device count (%s)\n",
            cudaGetErrorName (cudaGetLastError ()));
        goto done;
      }
      if (init_params->gpuID >= (guint) devices && 0) {
        g_printerr
            ("Error: Invalid gpu device ID (%d). CUDA device count (%d)\n",
            init_params->gpuID, devices);
        goto done;
      }
      if (nvinfer)
        nvinfer->gpu_id = init_params->gpuID;
    } else if (paramKey == "enable-dla") {
      /* Switched to setting the values as set in file rather setting
       * as TRUE if present in file.
       */
      init_params->useDLA = itr->second.as<gboolean>();
    } else if (paramKey == "use-dla-core") {
      init_params->dlaCore = itr->second.as<int>();
    } else if (paramKey == "auto-inc-mem") {
      init_params->autoIncMem = itr->second.as<int>();
    } else if (paramKey == "max-gpu-mem-per") {
      init_params->maxGPUMemPer = itr->second.as<double>();
    } else if (paramKey == "tensor-meta-pool-size") {
      init_params->outputBufferPoolSize = itr->second.as<unsigned int>();;
    } else if (paramKey == "batch-size") {
      if (nvinfer && (*nvinfer->is_prop_set)[PROP_BATCH_SIZE])
        continue;
      init_params->maxBatchSize = itr->second.as<unsigned int>();

      if (init_params->maxBatchSize <= 0
          || init_params->maxBatchSize > NVDSINFER_MAX_BATCH_SIZE) {
        g_printerr ("Error: batch-size(%d) should be in the range [%d,%d]\n",
            nvinfer->max_batch_size, 1, NVDSINFER_MAX_BATCH_SIZE);
        goto done;
      }
      if (nvinfer)
        nvinfer->max_batch_size = init_params->maxBatchSize;
    } else if (paramKey == "force-implicit-batch-dim") {
      /* Switched to setting the values as set in file rather setting
       * as TRUE if present in file.
       */
      init_params->forceImplicitBatchDimension = itr->second.as<gboolean>();
    } else if (paramKey == "workspace-size") {
      init_params->workspaceSize = itr->second.as<unsigned int>();

      if (init_params->workspaceSize <= 0) {
        g_print ("Info: workspace-size is 0, will use default size");
        init_params->workspaceSize = 0;
      }
    } else if (paramKey == "infer-dims") {
      std::string values = itr->second.as<std::string>();
      std::vector<std::string> vec = split_string(values);

      if (vec.size() != 3) {
        printf ("Error. infer-dims array length is %lu. Should be 3 as [c;h;w] order.\n", vec.size());
        goto done;
      }
      init_params->inferInputDims = NvDsInferDimsCHW {
       (unsigned int) std::stoul(vec[0]), (unsigned int) std::stoul(vec[1]),
       (unsigned int) std::stoul(vec[2])};
    } else if (paramKey == "network-mode") {
      guint val = itr->second.as<unsigned int>();

      switch (val) {
        case NvDsInferNetworkMode_FP32:
        case NvDsInferNetworkMode_FP16:
        case NvDsInferNetworkMode_INT8:
	case NvDsInferNetworkMode_BEST:
          break;
        default:
          g_printerr ("Error. Invalid value for network-mode:'%d'\n", val);
          goto done;
          break;
      }
      init_params->networkMode = (NvDsInferNetworkMode) val;
    } else if (paramKey == "model-engine-file") {
      if (nvinfer && (*nvinfer->is_prop_set)[PROP_MODEL_ENGINEFILE])
        continue;
      std::string temp = itr->second.as<std::string>();

      if (!get_absolute_file_path (cfg_file_path, temp.c_str(),
              init_params->modelEngineFilePath)) {
        g_printerr ("Error: Could not parse model engine file path\n");
        goto done;
      }
    } else if (paramKey == "int8-calib-file") {
      std::string temp = itr->second.as<std::string>();

      if (!get_absolute_file_path (cfg_file_path, temp.c_str(),
              init_params->int8CalibrationFilePath)) {
        g_printerr ("Error: Could not parse INT8 calibration file path\n");
        goto done;
      }
    } else if (paramKey == "output-blob-names") {
      std::string str = itr->second.as<std::string>();
      std::vector<std::string> vec = split_string(str);
      gchar **values = nullptr;
      int len = (int) vec.size();
      values = g_new (gchar *, len + 1);

      for (int i = 0; i < len; i++) {
        int size = 64;
        char* str2 = (char*) malloc(sizeof(char) * size);
        std::strncpy (str2, vec[i].c_str(), size-1);
        values[i] = str2;
      }
      values[len] = NULL;

      init_params->outputLayerNames = values;
      init_params->numOutputLayers = len;
    } else if (paramKey == "dump-input-tensor"){
      init_params->dumpIpTensor = itr->second.as<int>();
    } else if (paramKey == "dump-output-tensor"){
      init_params->dumpOpTensor = itr->second.as<int>();
    } else if (paramKey == "overwrite-input-tensor"){
      init_params->overwriteIpTensor = itr->second.as<int>();
    } else if (paramKey == "overwrite-output-tensor"){
      init_params->overwriteOpTensor = itr->second.as<int>();
    } else if (paramKey == "ip-tensor-file"){
      std::string temp = itr->second.as<std::string>();

      if (!get_absolute_file_path (cfg_file_path, temp.c_str(),
              init_params->ipTensorFilePath)) {
        g_printerr ("Error: Could not parse Input Tensor file path\n");
        goto done;
      }
    } else if (paramKey == "op-tensor-files"){
      std::string str = itr->second.as<std::string>();
      std::vector<std::string> vec = split_string(str);
      int length = (int) vec.size();
      init_params->opTensorFilePath = (gchar **) malloc(sizeof(gchar *)*length);

      for (int i = 0; i < length; i++) {
        int size = 64;
        char* str2 = (char*) malloc(sizeof(char) * size);
        if (!get_absolute_file_path (cfg_file_path, vec[i].c_str(),
              str2)) {
          g_printerr ("Error: Could not parse Output Tensor file path\n");
          g_free(str2);
          goto done;
        }
        init_params->opTensorFilePath[i] = str2;
      }
    } else if (paramKey == "output-io-formats") {
      std::string str = itr->second.as<std::string>();
      std::vector<std::string> vec = split_string(str);
      gchar **values = nullptr;
      int len = (int) vec.size();
      values = g_new (gchar *, len + 1);

      for (int i = 0; i < len; i++) {
        int size = 64;
        char* str2 = (char*) malloc(sizeof(char) * size);
        std::strncpy (str2, vec[i].c_str(), size-1);
        values[i] = str2;
      }
      values[len] = NULL;

      init_params->outputIOFormats = values;
      init_params->numOutputIOFormats = len;
    } else if (paramKey == "layer-device-precision") {
      std::string str = itr->second.as<std::string>();
      std::vector<std::string> vec = split_string(str);
      gchar **values = nullptr;
      int len = (int) vec.size();
      values = g_new (gchar *, len + 1);

      for (int i = 0; i < len; i++) {
        int size = vec[i].size() + 1;
        char* str2 = (char*) malloc(sizeof(char) * size);
        std::strncpy (str2, vec[i].c_str(), size);
        values[i] = str2;
      }
      values[len] = NULL;

      init_params->layerDevicePrecisions = values;
      init_params->numLayerDevicePrecisions = len;
    } else if (paramKey == "network-type") {
      guint val = itr->second.as<unsigned int>();

      switch ((NvDsInferNetworkType) val) {
        case NvDsInferNetworkType_Detector:
        case NvDsInferNetworkType_Classifier:
        case NvDsInferNetworkType_Segmentation:
        case NvDsInferNetworkType_InstanceSegmentation:
        case NvDsInferNetworkType_Other:
          init_params->networkType = (NvDsInferNetworkType) val;
          break;
        default:
          g_printerr ("Error. Invalid value for network-type':'%d'\n", val);
          goto done;
          break;
      }
    } else if (paramKey == "model-color-format") {
      guint val = itr->second.as<unsigned int>();
      switch (val) {
        case 0:
          init_params->networkInputFormat = NvDsInferFormat_RGB;
          break;
        case 1:
          init_params->networkInputFormat = NvDsInferFormat_BGR;
          break;
        case 2:
          init_params->networkInputFormat = NvDsInferFormat_GRAY;
          break;
        default:
          g_printerr ("Error. Invalid value for model-color-format:'%d'\n", val);
          goto done;
          break;
      }
    } else if (paramKey == "net-scale-factor") {
      init_params->networkScaleFactor = itr->second.as<double>();
    } else if (paramKey == "offsets") {
      std::string values = itr->second.as<std::string>();
      std::vector<std::string> vec = split_string(values);

      if (vec.size() > _MAX_CHANNELS) {
        g_printerr ("Error. Maximum  length of %d is allowed for offsets\n",
            _MAX_CHANNELS);
        goto done;
      }

      for (unsigned int i = 0; i < vec.size(); i++)
        init_params->offsets[i] = std::stod(vec[i]);
      init_params->numOffsets = vec.size();
    } else if (paramKey == "mean-file") {
      std::string temp = itr->second.as<std::string>();

      if (!get_absolute_file_path (cfg_file_path, temp.c_str(),
              init_params->meanImageFilePath)) {
        g_printerr ("Error: Could not parse mean image file path\n");
        goto done;
      }
    } else if (paramKey == "custom-lib-path") {
      std::string temp = itr->second.as<std::string>();

      if (!get_absolute_file_path (cfg_file_path, temp.c_str(),
              init_params->customLibPath)) {
        g_printerr ("Error: Could not parse custom library path\n");
        goto done;
      }
    } else if (paramKey == "parse-bbox-func-name") {
      std::string temp = itr->second.as<std::string>();
      std::strncpy (init_params->customBBoxParseFuncName, temp.c_str(), 1023);
    } else if (paramKey == "parse-bbox-instance-mask-func-name") {
      std::string temp = itr->second.as<std::string>();
      std::strncpy (init_params->customBBoxInstanceMaskParseFuncName, temp.c_str(), 1023);
    } else if (paramKey == "engine-create-func-name") {
      std::string temp = itr->second.as<std::string>();
      std::strncpy (init_params->customEngineCreateFuncName, temp.c_str(), 1023);
    } else if (paramKey == "parse-classifier-func-name") {
      std::string temp = itr->second.as<std::string>();
      std::strncpy (init_params->customClassifierParseFuncName, temp.c_str(), 1023);
    } else if (paramKey == "parse-segmentation-func-name") {
      std::string temp = itr->second.as<std::string>();
      std::strncpy (init_params->customSegmentationParseFuncName, temp.c_str(), 1023);
    } else if (paramKey == "custom-network-config") {
      std::string temp = itr->second.as<std::string>();
      std::strncpy (init_params->customNetworkConfigFilePath, temp.c_str(), 4095);
    } else if (paramKey == "model-file") {
      std::string temp = itr->second.as<std::string>();

      if (!get_absolute_file_path (cfg_file_path, temp.c_str(),
              init_params->modelFilePath)) {
        g_printerr ("Error: Could not parse model file path\n");
        goto done;
      }
    } else if (paramKey == "proto-file") {
      std::string temp = itr->second.as<std::string>();

      if (!get_absolute_file_path (cfg_file_path, temp.c_str(),
              init_params->protoFilePath)) {
        g_printerr ("Error: Could not parse prototxt file path\n");
        goto done;
      }
    } else if (paramKey == "uff-file") {
      std::string temp = itr->second.as<std::string>();

      if (!get_absolute_file_path (cfg_file_path, temp.c_str(),
              init_params->uffFilePath)) {
        g_printerr ("Error: Could not parse UFF file path\n");
        goto done;
      }
    } else if (paramKey == "network-input-order"){
      gint val = itr->second.as<int>();

      switch (val) {
          case 0:
              init_params->netInputOrder = NvDsInferTensorOrder_kNCHW;
              break;
          case 1:
              init_params->netInputOrder = NvDsInferTensorOrder_kNHWC;
              break;
          default:
              g_printerr ("Error. Invalid value for network-input-order, network input order :%d\n", val);
              goto done;
              break;
      }
    } else if (paramKey == "uff-input-order")  {
      gint val = itr->second.as<int>();

      switch (val) {
        case 0:
          init_params->uffInputOrder = NvDsInferTensorOrder_kNCHW;
          break;
        case 1:
          init_params->uffInputOrder = NvDsInferTensorOrder_kNHWC;
          break;
        case 2:
          init_params->uffInputOrder = NvDsInferTensorOrder_kNC;
          break;
        default:
          g_printerr ("Error. Invalid value for uff-input-order, UFF input order:%d\n", val);
        goto done;
        break;
      }
    } else if (paramKey == "uff-input-blob-name") {
      std::string temp = itr->second.as<std::string>();
      std::strncpy (init_params->uffInputBlobName, temp.c_str(), 1023);
    } else if (paramKey == "tlt-encoded-model") {
      std::string temp = itr->second.as<std::string>();

      if (!get_absolute_file_path (cfg_file_path, temp.c_str(),
              init_params->tltEncodedModelFilePath)) {
        g_printerr ("Error: Could not parse TLT encoded model file path\n");
        goto done;
      }
    } else if (paramKey == "tlt-model-key") {
      std::string temp = itr->second.as<std::string>();
      std::strncpy (init_params->tltModelKey, temp.c_str(), 1023);
    } else if (paramKey == "onnx-file") {
      std::string temp = itr->second.as<std::string>();

      if (!get_absolute_file_path (cfg_file_path, temp.c_str(),
              init_params->onnxFilePath)) {
        g_printerr ("Error: Could not parse ONNX file path\n");
        goto done;
      }
    } else if (paramKey == "num-detected-classes") {
      init_params->numDetectedClasses = itr->second.as<unsigned int>();

      if (init_params->numDetectedClasses < 0) {
        g_printerr ("Error: Negative value specified for num-detected-classes(%d)\n",
            init_params->numDetectedClasses);
        goto done;
      }
    } else if (paramKey == "cluster-mode") {
      gint val = itr->second.as<unsigned int>();
      if(val < 0) {
          g_printerr ("Error: Negative value specified for cluster-mode(%d)\n", val);
          goto done;
      }
      switch (val) {
        case 0:
          init_params->clusterMode = NVDSINFER_CLUSTER_GROUP_RECTANGLES;
          break;
        case 1:
          init_params->clusterMode = NVDSINFER_CLUSTER_DBSCAN;
          break;
        case 2:
          init_params->clusterMode = NVDSINFER_CLUSTER_NMS;
          break;
        case 3:
          init_params->clusterMode = NVDSINFER_CLUSTER_DBSCAN_NMS_HYBRID;
          break;
        case 4:
          init_params->clusterMode = NVDSINFER_CLUSTER_NONE;
          break;
        default:
          g_printerr ("Error. Invalid value for cluster-mode:'%d'\n", val);
          goto done;
          break;
      }
    } else if (paramKey == "classifier-threshold") {
      init_params->classifierThreshold = itr->second.as<double>();

      if (init_params->classifierThreshold < 0) {
        g_printerr ("Error: Negative value specified for classifier-threshold(%.2f)\n",
            init_params->classifierThreshold);
        goto done;
      }
    } else if (paramKey == "segmentation-threshold") {
      init_params->segmentationThreshold = itr->second.as<double>();

      if (init_params->segmentationThreshold < 0) {
        g_printerr ("Error: Negative value specified for segmentation-threshold(%.2f)\n",
            init_params->segmentationThreshold);
        goto done;
      }
    } else if (paramKey == "segmentation-output-order"){
      gint val = itr->second.as<int>();
      switch (val) {
        case 0:
          init_params->segmentationOutputOrder = NvDsInferTensorOrder_kNCHW;
          break;
        case 1:
          init_params->segmentationOutputOrder = NvDsInferTensorOrder_kNHWC;
          break;
        default:
          g_printerr ("Error. Invalid value for segmentation-output-order, Segmentation output order :'%d'\n", val);
          goto done;
          break;
      }
    } else if (paramKey == "input-tensor-from-meta") {
      if ((*nvinfer->is_prop_set)[PROP_INPUT_TENSOR_META])
        continue;
      gboolean val = itr->second.as<gboolean>();
      if (val) {
        nvinfer->input_tensor_from_meta = TRUE;
        init_params->inputFromPreprocessedTensor = TRUE;
      }
    } else if (paramKey == "disable-output-host-copy") {
      init_params->disableOutputHostCopy = itr->second.as<unsigned int>();
    } else if (paramKey == "raw-output-file-write") {
      gboolean val = itr->second.as<gboolean>();
      if (val) {
        nvinfer->write_raw_buffers_to_file = TRUE;
      }
    } else if (paramKey == "strongly-typed") {
      if (nvinfer && (*nvinfer->is_prop_set)[PROP_STRONGLY_TYPED])
        continue;
      init_params->useStronglyTyped = itr->second.as<gboolean>();
    } else if (paramKey == "warmup-engine") {
      init_params->warmupEngine = itr->second.as<gboolean>();
    } else if (paramKey == "stabilize-buffer-count") {
      if (nvinfer)
        nvinfer->stabilize_buffer_count = itr->second.as<unsigned int>();
    } else if (paramKey == "dynamic-properties") {
      if (nvinfer && (*nvinfer->is_prop_set)[PROP_DYNAMIC_PROPERTIES])
        continue;
      std::string temp = itr->second.as<std::string>();
      if (!temp.empty()) {
        // Parse dynamic properties string and set them
        gchar **tokens = g_strsplit(temp.c_str(), ";", -1);
        for (gint i = 0; tokens[i] != NULL; i++) {
          gchar **key_value = g_strsplit(tokens[i], "=", 2);
          if (key_value[0] != NULL && key_value[1] != NULL) {
            // Trim whitespace
            g_strstrip(key_value[0]);
            g_strstrip(key_value[1]);
            NvDsInferContext_SetDynamicProperty(init_params, key_value[0], key_value[1]);
          }
          g_strfreev(key_value);
        }
        g_strfreev(tokens);
      }
    } else if (nvinfer) {
      std::string paramVal = itr->second.as<std::string>();
      std::vector<std::string> values;
      values.push_back(paramKey);
      values.push_back(paramVal);
      if (!gst_nvinfer_parse_other_attribute_yaml (
          nvinfer, cfg_file_path, values)) {
        goto done;
      }
    }
  }

  ret = TRUE;
done:
  return ret;
}

/* Parse the nvinfer yaml config file. Returns FALSE in case of an error. */
gboolean
gst_nvinfer_parse_config_file_yaml (
    GstNvInfer * nvinfer,
    NvDsInferContextInitParams *init_params,
    const gchar * cfg_file_path)
{
  gboolean ret = FALSE;

  YAML::Node configyml = YAML::LoadFile(cfg_file_path);
  if(!(configyml.size() > 0))  {
  	cout << "Can't open config file (" << cfg_file_path << ")" << endl;
  }
  /* 'property' group is mandatory. */
  if(configyml["property"]) {
    if (!gst_nvinfer_parse_props_yaml (nvinfer, init_params, cfg_file_path)) {
      g_printerr ("Failed to parse group property\n");
      goto done;
    }
  }
  else  {
    g_printerr ("Could not find group property\n");
    goto done;
  }

  /* If the nvinfer instance is to be configured as a detector, parse the
   * per-class detection parameters. */
  if (init_params->networkType == NvDsInferNetworkType_Detector ||
           init_params->networkType == NvDsInferNetworkType_InstanceSegmentation) {
    /* Set the default detection parameters. */
    NvDsInferDetectionParams detection_params{{DEFAULT_PRE_CLUSTER_THRESHOLD},
        DEFAULT_POST_CLUSTER_THRESHOLD, DEFAULT_EPS,
        DEFAULT_GROUP_THRESHOLD, DEFAULT_MIN_BOXES,
        DEFAULT_DBSCAN_MIN_SCORE, DEFAULT_NMS_IOU_THRESHOLD, DEFAULT_TOP_K};
    GstNvInferDetectionFilterParams detection_filter_params{0, 0, 0, 0, 0, 0};
    GstNvInferColorParams color_params;
    color_params.have_border_color = TRUE;
    color_params.border_color = (NvOSD_ColorParams) {1, 0, 0, 1};
    color_params.have_bg_color = FALSE;
    color_params.bg_color = {0};

    /* Parse the parameters for "all" classes if the group has been specified. */
    if (configyml["class-attrs-all"]) {
      std::string temp = "class-attrs-all";
      if (!gst_nvinfer_parse_class_attrs (cfg_file_path, temp,
              detection_params, detection_filter_params, color_params)) {
        g_printerr ("Error while parsing group class-attrs-all\n");
        goto done;
      }
    }

    /* Initialize the per-class vector with the same default/parsed values for
     * all classes. */
    init_params->perClassDetectionParams =
        new NvDsInferDetectionParams[init_params->numDetectedClasses];
    for (unsigned int i = 0; i < init_params->numDetectedClasses; i++)
      init_params->perClassDetectionParams[i] = detection_params;
    nvinfer->perClassDetectionFilterParams =
        new std::vector < GstNvInferDetectionFilterParams >
        (init_params->numDetectedClasses, detection_filter_params);
    nvinfer->perClassColorParams =
        new std::vector < GstNvInferColorParams >
        (init_params->numDetectedClasses, color_params);

    /* Parse values for specified classes. */
    for(YAML::const_iterator itr = configyml.begin(); itr != configyml.end(); ++itr)
    {
      std::string paramKey = itr->first.as<std::string>();
      std::string class_str = "class-attrs-";
      if(paramKey != "class-attrs-all") {
        if(paramKey.compare(0,class_str.size(),class_str) == 0) {
          std::string num_str = paramKey.substr(class_str.size());
          guint64 class_index = stoi(num_str);

          /* Check that class_index has been parsed successfully and that it lies
          * within the valid range of class_ids [0, numDetectedClasses - 1]. */
          if ((gint) class_index < 0) {
            g_printerr ("Invalid group [%s]. class-id should be >= 0\n", paramKey.c_str());
            goto done;
          }
          if (class_index >= init_params->numDetectedClasses) {
            g_printerr
                ("Attributes specified for class %lu while element has been "
                "configured with num-detected-classes=%d\n",
                class_index, init_params->numDetectedClasses);
            goto done;
          }

          /* Parse the group. */
          if (!gst_nvinfer_parse_class_attrs (cfg_file_path, paramKey,
                  init_params->perClassDetectionParams[class_index],
                  (*nvinfer->perClassDetectionFilterParams)[class_index],
                  (*nvinfer->perClassColorParams)[class_index])) {
            g_printerr ("Error while parsing group %s\n", paramKey.c_str());
            goto done;
          }
        }
      }
    }
  }
  ret = TRUE;

done:
  if (!ret) {
    g_printerr ("** ERROR: <%s:%d>: failed\n", __func__, __LINE__);
  }
  return ret;
}

/* Parse nvinfer config file for context params. Returns FALSE in case of an error. */
gboolean
gst_nvinfer_parse_context_params_yaml (
    NvDsInferContextInitParams *params,
    const gchar * cfg_file_path)
{
  gboolean ret = FALSE;

  YAML::Node configyml = YAML::LoadFile(cfg_file_path);
  if(!(configyml.size() > 0))  {
  	cout << "Can't open config file (" << cfg_file_path << ")" << endl;
  }
  /* 'property' group is mandatory. */
  if(configyml["property"]) {
    if (!gst_nvinfer_parse_props_yaml (NULL, params, cfg_file_path)) {
      g_printerr ("Failed to parse group property\n");
      goto done;
    }
  }
  else  {
    g_printerr ("Could not find group property\n");
    goto done;
  }
  ret = TRUE;

done:
  if (!ret) {
    g_printerr ("** ERROR: <%s:%d>: failed\n", __func__, __LINE__);
  }
  return ret;
}
