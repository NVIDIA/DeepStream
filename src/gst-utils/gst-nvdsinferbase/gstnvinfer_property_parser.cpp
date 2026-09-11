/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gst/gst.h>
#include <string.h>
#include <assert.h>

#include "gstnvinfer_property_parser.h"
#include "gstnvinferbase.h"

#define CHECK_ERROR(error) \
    if (error) { \
        g_printerr ("Error while parsing config file: %s\n", error->message); \
        goto done; \
    }

#define CHECK_STR_ERROR(error) \
    if (error) { \
        g_printerr ("Error while parsing config file: %s\n", error->message); \
        free (str); \
        goto done; \
    }

extern const int DEFAULT_REINFER_INTERVAL;

/* Get the absolute path of a file mentioned in the config given a
 * file path absolute/relative to the config file. */
static gboolean
get_absolute_file_path (
    const gchar * cfg_file_path, const gchar * file_path,
    char *abs_path_str)
{
  gchar abs_cfg_path[PATH_MAX + 1];
  gchar abs_real_file_path[PATH_MAX + 1];
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

/* Parse 'property' group. Returns FALSE in case of an error. If any of the
 * properties are set through the GObject set method this function does not
 * parse those properties i.e. values set through g_object_set override the
 * corresponding properties in the config file. */
static gboolean
gst_nvinfer_parse_props (GstNvInferBase * nvinfer,
    NvDsInferContextInitParams * init_params,
    GKeyFile * key_file, const gchar * cfg_file_path)
{
  gboolean ret = FALSE;
  gchar **keys = nullptr;
  gchar **key = nullptr;
  GError *error = nullptr;
  GstNvInferBaseClass *klass = GST_NVINFERBASE_GET_CLASS (nvinfer);

  assert (init_params != nullptr);

  /* Handle legacy key names. */
  if (g_key_file_has_key (key_file, CONFIG_GROUP_PROPERTY,
          CONFIG_GROUP_INFER_UFF_INPUT_DIMENSIONS_LEGACY, nullptr)
      && !g_key_file_has_key (key_file, CONFIG_GROUP_PROPERTY,
          CONFIG_GROUP_INFER_UFF_INPUT_DIMENSIONS, nullptr)) {
    /* Do not parse here, set the key-value pair with the new key. */
    gchar *value = g_key_file_get_value (key_file, CONFIG_GROUP_PROPERTY,
        CONFIG_GROUP_INFER_UFF_INPUT_DIMENSIONS_LEGACY, &error);
    CHECK_ERROR (error);
    g_key_file_set_value (key_file, CONFIG_GROUP_PROPERTY,
        CONFIG_GROUP_INFER_UFF_INPUT_DIMENSIONS, value);
    g_free (value);
    g_key_file_remove_key (key_file, CONFIG_GROUP_PROPERTY,
        CONFIG_GROUP_INFER_UFF_INPUT_DIMENSIONS_LEGACY, nullptr);
  }

  if (g_key_file_has_key (key_file, CONFIG_GROUP_PROPERTY,
          CONFIG_GROUP_INFER_IS_CLASSIFIER_LEGACY, nullptr)
      && !g_key_file_has_key (key_file, CONFIG_GROUP_PROPERTY,
          CONFIG_GROUP_INFER_NETWORK_TYPE, nullptr)) {
    /* Do not parse here, set the key-value pair with the new key. */
    gboolean value = g_key_file_get_boolean (key_file, CONFIG_GROUP_PROPERTY,
        CONFIG_GROUP_INFER_IS_CLASSIFIER_LEGACY, &error);
    CHECK_ERROR (error);
    guint new_value = (value) ? NvDsInferNetworkType_Classifier : NvDsInferNetworkType_Detector;
    g_key_file_set_integer (key_file, CONFIG_GROUP_PROPERTY,
        CONFIG_GROUP_INFER_NETWORK_TYPE, new_value);
    g_key_file_remove_key (key_file, CONFIG_GROUP_PROPERTY,
        CONFIG_GROUP_INFER_IS_CLASSIFIER_LEGACY, nullptr);
  }

  keys = g_key_file_get_keys (key_file, CONFIG_GROUP_PROPERTY, nullptr, &error);
  CHECK_ERROR (error);

  init_params->networkInputFormat = klass->get_default_infer_format(nvinfer);

  for (key = keys; *key; key++) {
    if (!g_strcmp0 (*key, CONFIG_GROUP_INFER_UNIQUE_ID)) {
      if (nvinfer && (*nvinfer->is_prop_set)[PROP_UNIQUE_ID])
        continue;
      init_params->uniqueID =
          g_key_file_get_integer (key_file, CONFIG_GROUP_PROPERTY,
          CONFIG_GROUP_INFER_UNIQUE_ID, &error);
      CHECK_ERROR (error);

      if (init_params->uniqueID <= 0) {
        g_printerr ("Error: %s (%d) should be > 0\n",
            CONFIG_GROUP_INFER_UNIQUE_ID, nvinfer->unique_id);
        goto done;
      }
      if (nvinfer)
        nvinfer->unique_id = init_params->uniqueID;
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_INFER_LABEL)) {
      gchar *str = g_key_file_get_string (key_file, CONFIG_GROUP_PROPERTY,
          CONFIG_GROUP_INFER_LABEL, &error);
      CHECK_STR_ERROR (error);

      if (!get_absolute_file_path (cfg_file_path, str,
              init_params->labelsFilePath)) {
        g_printerr ("Error: Could not parse labels file path\n");
        g_free (str);
        goto done;
      }
      g_free (str);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_INFER_GPU_ID)) {
      if (nvinfer && (*nvinfer->is_prop_set)[PROP_GPU_DEVICE_ID])
        continue;
      gint devices;

      init_params->gpuID =
          g_key_file_get_integer (key_file, CONFIG_GROUP_PROPERTY,
          CONFIG_GROUP_INFER_GPU_ID, &error);
      CHECK_ERROR (error);

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
    }  else if (!g_strcmp0 (*key, CONFIG_GROUP_INFER_ENABLE_DLA)) {
      if (g_key_file_get_boolean (key_file, CONFIG_GROUP_PROPERTY,
              CONFIG_GROUP_INFER_ENABLE_DLA, &error))
        init_params->useDLA = TRUE;
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_INFER_USE_DLA_CORE)) {
      init_params->dlaCore =
          g_key_file_get_integer (key_file, CONFIG_GROUP_PROPERTY,
          CONFIG_GROUP_INFER_USE_DLA_CORE, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_INFER_BATCH_SIZE)) {
      if (nvinfer && (*nvinfer->is_prop_set)[PROP_BATCH_SIZE])
        continue;
      init_params->maxBatchSize =
          g_key_file_get_integer (key_file, CONFIG_GROUP_PROPERTY,
          CONFIG_GROUP_INFER_BATCH_SIZE, &error);
      CHECK_ERROR (error);

      if (init_params->maxBatchSize <= 0
          || init_params->maxBatchSize > NVDSINFER_MAX_BATCH_SIZE) {
        g_printerr ("Error: %s(%d) should be in the range [%d,%d]\n",
            CONFIG_GROUP_INFER_BATCH_SIZE, nvinfer->max_batch_size, 1,
            NVDSINFER_MAX_BATCH_SIZE);
        goto done;
      }
      if (nvinfer)
        nvinfer->max_batch_size = init_params->maxBatchSize;
  } else if (!g_strcmp0 (*key, CONFIG_GROUP_INFER_FORCE_IMPLICIT_BATCH_DIM)) {
    if (g_key_file_get_boolean (key_file, CONFIG_GROUP_PROPERTY,
            CONFIG_GROUP_INFER_FORCE_IMPLICIT_BATCH_DIM, &error))
      init_params->forceImplicitBatchDimension = TRUE;
    CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_INFER_WORKSPACE_SIZE)) {
      init_params->workspaceSize =
          g_key_file_get_integer (key_file, CONFIG_GROUP_PROPERTY,
          CONFIG_GROUP_INFER_WORKSPACE_SIZE, &error);
      CHECK_ERROR (error);

      if (init_params->workspaceSize <= 0) {
        g_print ("Info: workspace-size is 0, will use default size");
        init_params->workspaceSize = 0;
      }
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_INFER_INFER_DIMENSIONS)) {
      gsize length;
      gint *int_list = g_key_file_get_integer_list (key_file,
          CONFIG_GROUP_PROPERTY, CONFIG_GROUP_INFER_INFER_DIMENSIONS,
          &length, &error);
      CHECK_ERROR (error);

      if (length != 3) {
        g_printerr ("Error. '%s' array length is %lu. Should be 3 as [c;h;w] order.\n",
            CONFIG_GROUP_INFER_INFER_DIMENSIONS, length);
        goto done;
      }
      init_params->inferInputDims = NvDsInferDimsCHW {
      (unsigned int) int_list[0],
            (unsigned int) int_list[1], (unsigned int) int_list[2]};
      g_free (int_list);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_INFER_NETWORK_MODE)) {
      guint val = g_key_file_get_integer (key_file, CONFIG_GROUP_PROPERTY,
          CONFIG_GROUP_INFER_NETWORK_MODE, &error);
      CHECK_ERROR (error);

      switch (val) {
        case NvDsInferNetworkMode_FP32:
        case NvDsInferNetworkMode_FP16:
        case NvDsInferNetworkMode_INT8:
	case NvDsInferNetworkMode_BEST:
          break;
        default:
          g_printerr ("Error. Invalid value for '%s':'%d'\n",
              CONFIG_GROUP_INFER_NETWORK_MODE, val);
          goto done;
          break;
      }
      init_params->networkMode = (NvDsInferNetworkMode) val;
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_INFER_MODEL_ENGINE)) {
      if (nvinfer && (*nvinfer->is_prop_set)[PROP_MODEL_ENGINEFILE])
        continue;
      gchar *str = g_key_file_get_string (key_file, CONFIG_GROUP_PROPERTY,
          CONFIG_GROUP_INFER_MODEL_ENGINE, &error);
      CHECK_STR_ERROR (error);

      if (!get_absolute_file_path (cfg_file_path, str,
              init_params->modelEngineFilePath)) {
        g_printerr ("Error: Could not parse model engine file path\n");
        g_free (str);
        goto done;
      }
      g_free (str);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_INFER_INT8_CALIBRATION_FILE)) {
      gchar *str = g_key_file_get_string (key_file, CONFIG_GROUP_PROPERTY,
          CONFIG_GROUP_INFER_INT8_CALIBRATION_FILE, &error);
      CHECK_STR_ERROR (error);

      if (!get_absolute_file_path (cfg_file_path, str,
              init_params->int8CalibrationFilePath)) {
        g_printerr ("Error: Could not parse INT8 calibration file path\n");
        g_free (str);
        goto done;
      }
      g_free (str);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_INFER_OUTPUT_BLOB_NAMES)) {
      gsize length;
      init_params->outputLayerNames =
          g_key_file_get_string_list (key_file, CONFIG_GROUP_PROPERTY,
          CONFIG_GROUP_INFER_OUTPUT_BLOB_NAMES, &length, &error);
      init_params->numOutputLayers = length;

      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_INFER_NETWORK_TYPE)) {
      guint val = g_key_file_get_integer (key_file, CONFIG_GROUP_PROPERTY,
          CONFIG_GROUP_INFER_NETWORK_TYPE, &error);
      CHECK_ERROR (error);

      switch ((NvDsInferNetworkType) val) {
        case NvDsInferNetworkType_Detector:
        case NvDsInferNetworkType_Classifier:
        case NvDsInferNetworkType_Segmentation:
        case NvDsInferNetworkType_Other:
          init_params->networkType = (NvDsInferNetworkType) val;
          break;
        default:
          g_printerr ("Error. Invalid value for '%s':'%d'\n",
              CONFIG_GROUP_INFER_NETWORK_TYPE, val);
          goto done;
          break;
      }
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_INFER_MODEL_COLOR_FORMAT)) {
      guint val = g_key_file_get_integer (key_file, CONFIG_GROUP_PROPERTY,
          CONFIG_GROUP_INFER_MODEL_COLOR_FORMAT, &error);
      CHECK_ERROR (error);
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
          g_printerr ("Error. Invalid value for '%s':'%d'\n",
              CONFIG_GROUP_INFER_MODEL_COLOR_FORMAT, val);
          goto done;
          break;
      }
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_INFER_SCALE_FACTOR)) {
      init_params->networkScaleFactor =
          g_key_file_get_double (key_file, CONFIG_GROUP_PROPERTY,
          CONFIG_GROUP_INFER_SCALE_FACTOR, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_INFER_OFFSETS)) {
      gsize length, i;
      gdouble *dbl_list =
          g_key_file_get_double_list (key_file, CONFIG_GROUP_PROPERTY,
          CONFIG_GROUP_INFER_OFFSETS, &length, &error);
      CHECK_ERROR (error);

      if (length > _MAX_CHANNELS) {
        g_printerr ("Error. Maximum  length of %d is allowed for '%s'\n",
            _MAX_CHANNELS, CONFIG_GROUP_INFER_OFFSETS);
        g_free (dbl_list);
        goto done;
      }

      for (i = 0; i < length; i++)
        init_params->offsets[i] = dbl_list[i];
      init_params->numOffsets = length;
      g_free (dbl_list);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_INFER_MEANFILE)) {
      gchar *str = g_key_file_get_string (key_file, CONFIG_GROUP_PROPERTY,
          CONFIG_GROUP_INFER_MEANFILE, &error);
      CHECK_STR_ERROR (error);

      if (!get_absolute_file_path (cfg_file_path, str,
              init_params->meanImageFilePath)) {
        g_printerr ("Error: Could not parse mean image file path\n");
        g_free (str);
        goto done;
      }
      g_free (str);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_INFER_CUSTOM_LIB_PATH)) {
      gchar *str = g_key_file_get_string (key_file, CONFIG_GROUP_PROPERTY,
          CONFIG_GROUP_INFER_CUSTOM_LIB_PATH, &error);
      CHECK_STR_ERROR (error);

      if (!get_absolute_file_path (cfg_file_path, str,
              init_params->customLibPath)) {
        g_printerr ("Error: Could not parse custom library path\n");
        g_free (str);
        goto done;
      }
      g_free (str);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_INFER_CUSTOM_PARSE_BBOX_FUNC)) {
      gchar *str = g_key_file_get_string (key_file, CONFIG_GROUP_PROPERTY,
          CONFIG_GROUP_INFER_CUSTOM_PARSE_BBOX_FUNC, &error);
      CHECK_STR_ERROR (error);
      g_strlcpy (init_params->customBBoxParseFuncName, str,
          _MAX_STR_LENGTH);
      g_free (str);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_INFER_CUSTOM_ENGINE_CREATE_FUNC)) {
      gchar *str = g_key_file_get_string (key_file, CONFIG_GROUP_PROPERTY,
          CONFIG_GROUP_INFER_CUSTOM_ENGINE_CREATE_FUNC, &error);
      CHECK_STR_ERROR (error);
      g_strlcpy (init_params->customEngineCreateFuncName, str,
          _MAX_STR_LENGTH);
      g_free (str);
    } else if (!g_strcmp0 (*key,
            CONFIG_GROUP_INFER_CUSTOM_PARSE_CLASSIFIER_FUNC)) {
      gchar *str = g_key_file_get_string (key_file, CONFIG_GROUP_PROPERTY,
          CONFIG_GROUP_INFER_CUSTOM_PARSE_CLASSIFIER_FUNC, &error);
      CHECK_STR_ERROR (error);
      g_strlcpy (init_params->customClassifierParseFuncName, str,
          _MAX_STR_LENGTH);
      g_free (str);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_INFER_CUSTOM_NETWORK_CONFIG)) {
      gchar *str = g_key_file_get_string (key_file, CONFIG_GROUP_PROPERTY,
          CONFIG_GROUP_INFER_CUSTOM_NETWORK_CONFIG, &error);
      CHECK_STR_ERROR (error);
      g_strlcpy (init_params->customNetworkConfigFilePath, str,
          _MAX_STR_LENGTH);
      g_free (str);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_INFER_MODEL)) {
      gchar *str = g_key_file_get_string (key_file, CONFIG_GROUP_PROPERTY,
          CONFIG_GROUP_INFER_MODEL, &error);
      CHECK_STR_ERROR (error);

      if (!get_absolute_file_path (cfg_file_path, str,
              init_params->modelFilePath)) {
        g_printerr ("Error: Could not parse model file path\n");
        g_free (str);
        goto done;
      }
      g_free (str);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_INFER_PROTO)) {
      gchar *str = g_key_file_get_string (key_file, CONFIG_GROUP_PROPERTY,
          CONFIG_GROUP_INFER_PROTO, &error);
      CHECK_STR_ERROR (error);

      if (!get_absolute_file_path (cfg_file_path, str,
              init_params->protoFilePath)) {
        g_printerr ("Error: Could not parse prototxt file path\n");
        g_free (str);
        goto done;
      }
      g_free (str);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_INFER_UFF)) {
      gchar *str = g_key_file_get_string (key_file, CONFIG_GROUP_PROPERTY,
          CONFIG_GROUP_INFER_UFF, &error);
      CHECK_STR_ERROR (error);

      if (!get_absolute_file_path (cfg_file_path, str,
              init_params->uffFilePath)) {
        g_printerr ("Error: Could not parse UFF file path\n");
        g_free (str);
        goto done;
      }
      g_free (str);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_INFER_UFF_INPUT_DIMENSIONS)) {
      gsize length;
      gint *int_list = g_key_file_get_integer_list (key_file,
          CONFIG_GROUP_PROPERTY, CONFIG_GROUP_INFER_UFF_INPUT_DIMENSIONS,
          &length, &error);
      CHECK_ERROR (error);

      if (length != 4) {
        g_printerr ("Error. '%s' array length is %lu. Should be 4[a;b;c;ORDER].\n",
            CONFIG_GROUP_INFER_UFF_INPUT_DIMENSIONS, length);
        goto done;
      }
      switch (int_list[3]) {
        case 0:
            init_params->uffInputOrder = NvDsInferTensorOrder_kNCHW;
            break;
        case 1:
            init_params->uffInputOrder = NvDsInferTensorOrder_kNHWC;
            break;
        default:
          g_printerr ("Error. Invalid value for '%s', UFF input order :'%d'\n",
              CONFIG_GROUP_INFER_UFF_INPUT_DIMENSIONS, int_list[3]);
          goto done;
          break;
      }
      init_params->uffDimsCHW = NvDsInferDimsCHW {
      (unsigned int) int_list[0],
            (unsigned int) int_list[1], (unsigned int) int_list[2]};
      g_free (int_list);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_INFER_UFF_INPUT_BLOB_NAME)) {
      gchar *str = g_key_file_get_string (key_file, CONFIG_GROUP_PROPERTY,
          CONFIG_GROUP_INFER_UFF_INPUT_BLOB_NAME, &error);
      CHECK_STR_ERROR (error);

      g_strlcpy (init_params->uffInputBlobName, str, _MAX_STR_LENGTH);
      g_free (str);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_INFER_TLT_ENCODED_MODEL)) {
      gchar *str = g_key_file_get_string (key_file, CONFIG_GROUP_PROPERTY,
          CONFIG_GROUP_INFER_TLT_ENCODED_MODEL, &error);
      CHECK_STR_ERROR (error);

      if (!get_absolute_file_path (cfg_file_path, str,
              init_params->tltEncodedModelFilePath)) {
        g_printerr ("Error: Could not parse TLT encoded model file path\n");
        g_free (str);
        goto done;
      }
      g_free (str);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_INFER_TLT_MODEL_KEY)) {
      gchar *str = g_key_file_get_string (key_file, CONFIG_GROUP_PROPERTY,
          CONFIG_GROUP_INFER_TLT_MODEL_KEY, &error);
      CHECK_STR_ERROR (error);

      g_strlcpy (init_params->tltModelKey, str, _MAX_STR_LENGTH);
      g_free (str);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_INFER_ONNX)) {
      gchar *str = g_key_file_get_string (key_file, CONFIG_GROUP_PROPERTY,
          CONFIG_GROUP_INFER_ONNX, &error);
      CHECK_STR_ERROR (error);

      if (!get_absolute_file_path (cfg_file_path, str,
              init_params->onnxFilePath)) {
        g_printerr ("Error: Could not parse ONNX file path\n");
        g_free (str);
        goto done;
      }
      g_free (str);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_INFER_NUM_DETECTED_CLASSES)) {
      init_params->numDetectedClasses =
          g_key_file_get_integer (key_file, CONFIG_GROUP_PROPERTY,
          CONFIG_GROUP_INFER_NUM_DETECTED_CLASSES, &error);
      CHECK_ERROR (error);

      if (init_params->numDetectedClasses < 0) {
        g_printerr ("Error: Negative value specified for %s(%d)\n",
            CONFIG_GROUP_INFER_NUM_DETECTED_CLASSES,
            init_params->numDetectedClasses);
        goto done;
      }
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_INFER_ENABLE_DBSCAN)) {
        g_printerr("Error: 'enable-dbscan' parameter has been deprecated. Use 'cluster-mode' instead.\n");
      if (g_key_file_get_boolean (key_file, CONFIG_GROUP_PROPERTY,
              CONFIG_GROUP_INFER_ENABLE_DBSCAN, &error))
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_INFER_CLUSTER_MODE)) {
        gint val = g_key_file_get_integer (key_file, CONFIG_GROUP_PROPERTY,
          CONFIG_GROUP_INFER_CLUSTER_MODE, &error);
      CHECK_ERROR (error);
      if(val < 0) {
          g_printerr ("Error: Negative value specified for %s(%d)\n",
            CONFIG_GROUP_INFER_CLUSTER_MODE, val);
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
          init_params->clusterMode = NVDSINFER_CLUSTER_NONE;
          break;
        default:
          g_printerr ("Error. Invalid value for '%s':'%d'\n",
              CONFIG_GROUP_INFER_CLUSTER_MODE, val);
          goto done;
          break;
      }
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_INFER_CLASSIFIER_THRESHOLD)) {
      init_params->classifierThreshold =
          g_key_file_get_double (key_file, CONFIG_GROUP_PROPERTY,
          CONFIG_GROUP_INFER_CLASSIFIER_THRESHOLD, &error);
      CHECK_ERROR (error);

      if (init_params->classifierThreshold < 0) {
        g_printerr ("Error: Negative value specified for %s(%.2f)\n",
            CONFIG_GROUP_INFER_CLASSIFIER_THRESHOLD,
            init_params->classifierThreshold);
        goto done;
      }
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_INFER_SEGMENTATION_THRESHOLD)) {
      init_params->segmentationThreshold =
          g_key_file_get_double (key_file, CONFIG_GROUP_PROPERTY,
          CONFIG_GROUP_INFER_SEGMENTATION_THRESHOLD, &error);
      CHECK_ERROR (error);

      if (init_params->segmentationThreshold < 0) {
        g_printerr ("Error: Negative value specified for %s(%.2f)\n",
            CONFIG_GROUP_INFER_SEGMENTATION_THRESHOLD,
            init_params->segmentationThreshold);
        goto done;
      }
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_INFER_OUTPUT_TENSOR_META)) {
      if (g_key_file_get_boolean (key_file, CONFIG_GROUP_PROPERTY,
            CONFIG_GROUP_INFER_OUTPUT_TENSOR_META, &error))
        nvinfer->output_tensor_meta = TRUE;
      CHECK_ERROR (error);
    } else if (klass->parse_other_attributes) {
      if (!klass->parse_other_attributes(nvinfer, key_file, CONFIG_GROUP_PROPERTY, *key)) {
        goto done;
      }
    } else {
      g_printerr ("Unknown key '%s' for group [%s]\n", *key,
          CONFIG_GROUP_PROPERTY);
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

/* Parse the nvinfer config file. Returns FALSE in case of an error. */
gboolean
gst_nvinfer_parse_config_file (
    GstNvInferBase * nvinfer,
    NvDsInferContextInitParams *init_params,
    const gchar * cfg_file_path)
{
  GError *error = nullptr;
  gboolean ret = FALSE;
  gchar **groups = nullptr;
  GKeyFile *cfg_file = g_key_file_new ();
  GstNvInferBaseClass *klass = GST_NVINFERBASE_GET_CLASS (nvinfer);

  if (!g_key_file_load_from_file (cfg_file, cfg_file_path, G_KEY_FILE_NONE,
          &error)) {
    g_printerr ("Failed to load config file: %s\n", error->message);
    goto done;
  }

  /* 'property' group is mandatory. */
  if (!g_key_file_has_group (cfg_file, CONFIG_GROUP_PROPERTY)) {
    g_printerr ("Could not find group %s\n", CONFIG_GROUP_PROPERTY);
    goto done;
  }

  if (!gst_nvinfer_parse_props (nvinfer, init_params, cfg_file, cfg_file_path)) {
    g_printerr ("Failed to parse group %s\n", CONFIG_GROUP_PROPERTY);
    goto done;
  }
  g_key_file_remove_group (cfg_file, CONFIG_GROUP_PROPERTY, nullptr);

  ret = TRUE;

  if (klass->parse_other_groups)
    ret = klass->parse_other_groups (nvinfer, cfg_file, init_params);

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
  if (!ret) {
    g_printerr ("** ERROR: <%s:%d>: failed\n", __func__, __LINE__);
  }
  return ret;
}

/* Parse nvinfer config file for context params. Returns FALSE in case of an error. */
gboolean
gst_nvinfer_parse_context_params (
    NvDsInferContextInitParams *params,
    const gchar * cfg_file_path)
{
  GError *error = nullptr;
  gboolean ret = FALSE;
  GKeyFile *cfg_file = g_key_file_new ();

  if (!g_key_file_load_from_file (cfg_file, cfg_file_path, G_KEY_FILE_NONE,
          &error)) {
    g_printerr ("Failed to load config file: %s\n", error->message);
    goto done;
  }

  /* 'property' group is mandatory. */
  if (!g_key_file_has_group (cfg_file, CONFIG_GROUP_PROPERTY)) {
    g_printerr ("Could not find group %s\n", CONFIG_GROUP_PROPERTY);
    goto done;
  }

  if (!gst_nvinfer_parse_props (NULL, params, cfg_file, cfg_file_path)) {
    g_printerr ("Failed to parse group %s\n", CONFIG_GROUP_PROPERTY);
    goto done;
  }
  ret = TRUE;

done:
  if (cfg_file) {
    g_key_file_free (cfg_file);
  }

  if (error) {
    g_error_free (error);
  }
  if (!ret) {
    g_printerr ("** ERROR: <%s:%d>: failed\n", __func__, __LINE__);
  }
  return ret;
}
