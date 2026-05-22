/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "gstnvinfer.h"

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
    else{
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
gst_nvinfer_parse_class_attrs (GKeyFile * key_file, gchar * group,
    NvDsInferDetectionParams & detection_params,
    GstNvInferDetectionFilterParams & detection_filter_params,
    GstNvInferColorParams & color_params)
{
  gboolean ret = FALSE;
  gchar **keys = nullptr;
  gchar **key = nullptr;
  GError *error = nullptr;

  keys = g_key_file_get_keys (key_file, group, nullptr, &error);
  CHECK_ERROR (error);

  for (key = keys; *key; key++) {
    if (!g_strcmp0 (*key, CONFIG_GROUP_INFER_CLASS_ATTRS_THRESHOLD)) {
      g_printerr("Warn: 'threshold' parameter has been deprecated."
      " Use 'pre-cluster-threshold' instead.\n");
      detection_params.preClusterThreshold =
          g_key_file_get_double (key_file, group,
          CONFIG_GROUP_INFER_CLASS_ATTRS_THRESHOLD, &error);
      CHECK_ERROR (error);
      if (detection_params.preClusterThreshold < 0) {
        g_printerr ("Error: Negative threshold (%.5f) specified for group %s\n",
            detection_params.preClusterThreshold, group);
        goto done;
      }
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_INFER_CLASS_ATTRS_PRE_CLUSTER_THRESHOLD)) {
        detection_params.preClusterThreshold =
          g_key_file_get_double (key_file, group,
          CONFIG_GROUP_INFER_CLASS_ATTRS_PRE_CLUSTER_THRESHOLD, &error);
      CHECK_ERROR (error);
      if (detection_params.preClusterThreshold < 0) {
        g_printerr ("Error: Negative pre cluster threshold (%.5f) specified for group %s\n",
            detection_params.preClusterThreshold, group);
        goto done;
      }
    }  else if (!g_strcmp0 (*key, CONFIG_GROUP_INFER_CLASS_ATTRS_POST_CLUSTER_THRESHOLD)) {
        detection_params.postClusterThreshold =
          g_key_file_get_double (key_file, group,
          CONFIG_GROUP_INFER_CLASS_ATTRS_POST_CLUSTER_THRESHOLD, &error);
      CHECK_ERROR (error);
      if (detection_params.postClusterThreshold < 0) {
        g_printerr ("Error: Negative post cluster threshold (%.5f) specified for group %s\n",
            detection_params.postClusterThreshold, group);
        goto done;
      }
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_INFER_CLASS_ATTRS_EPS)) {
      detection_params.eps =
          g_key_file_get_double (key_file, group,
          CONFIG_GROUP_INFER_CLASS_ATTRS_EPS, &error);
      CHECK_ERROR (error);
      if (detection_params.eps < 0) {
        g_printerr ("Error: Negative eps (%.5f) specified for group %s\n",
            detection_params.eps, group);
        goto done;
      }
    } else if (!g_strcmp0 (*key,
            CONFIG_GROUP_INFER_CLASS_ATTRS_GROUP_THRESHOLD)) {
      detection_params.groupThreshold =
          g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_INFER_CLASS_ATTRS_GROUP_THRESHOLD, &error);
      CHECK_ERROR (error);
      if (detection_params.groupThreshold < 0) {
        g_printerr
            ("Error: Negative group-threshold (%d) specified for group %s\n",
            detection_params.groupThreshold, group);
        goto done;
      }
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_INFER_CLASS_ATTRS_MIN_BOXES)) {
      detection_params.minBoxes =
          g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_INFER_CLASS_ATTRS_MIN_BOXES, &error);
      CHECK_ERROR (error);
      if (detection_params.minBoxes < 0) {
        g_printerr
            ("Error: Negative minBoxes (%d) specified for group %s\n",
            detection_params.minBoxes, group);
        goto done;
      }
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_INFER_CLASS_ATTRS_DBSCAN_MIN_SCORE)) {
      detection_params.minScore =
          g_key_file_get_double (key_file, group,
          CONFIG_GROUP_INFER_CLASS_ATTRS_DBSCAN_MIN_SCORE, &error);
      CHECK_ERROR (error);
      if (detection_params.minScore < 0) {
        g_printerr
            ("Error: Negative minScore (%f) specified for group %s\n",
            detection_params.minScore, group);
        goto done;
      }
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_INFER_CLASS_ATTRS_ROI_TOP_OFFSET)) {
      detection_filter_params.roiTopOffset =
          g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_INFER_CLASS_ATTRS_ROI_TOP_OFFSET, &error);
      CHECK_ERROR (error);
      if ((gint) detection_filter_params.roiTopOffset < 0) {
        g_printerr
            ("Error: Negative roiTopOffset (%d) specified for group %s\n",
            detection_filter_params.roiTopOffset, group);
        goto done;
      }
    } else if (!g_strcmp0 (*key,
            CONFIG_GROUP_INFER_CLASS_ATTRS_ROI_BOTTOM_OFFSET)) {
      detection_filter_params.roiBottomOffset =
          g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_INFER_CLASS_ATTRS_ROI_BOTTOM_OFFSET, &error);
      CHECK_ERROR (error);
      if ((gint) detection_filter_params.roiBottomOffset < 0) {
        g_printerr
            ("Error: Negative roiBottomOffset (%d) specified for group %s\n",
            detection_filter_params.roiBottomOffset, group);
        goto done;
      }
    } else if (!g_strcmp0 (*key,
            CONFIG_GROUP_INFER_CLASS_ATTRS_DETECTED_MIN_WIDTH)) {
      detection_filter_params.detectionMinWidth =
          g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_INFER_CLASS_ATTRS_DETECTED_MIN_WIDTH, &error);
      CHECK_ERROR (error);
      if ((gint) detection_filter_params.detectionMinWidth < 0) {
        g_printerr
            ("Error: Negative detectionMinWidth (%d) specified for group %s\n",
            detection_filter_params.detectionMinWidth, group);
        goto done;
      }
    } else if (!g_strcmp0 (*key,
            CONFIG_GROUP_INFER_CLASS_ATTRS_DETECTED_MIN_HEIGHT)) {
      detection_filter_params.detectionMinHeight =
          g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_INFER_CLASS_ATTRS_DETECTED_MIN_HEIGHT, &error);
      CHECK_ERROR (error);
      if ((gint) detection_filter_params.detectionMinHeight < 0) {
        g_printerr
            ("Error: Negative detectionMinHeight (%d) specified for group %s\n",
            detection_filter_params.detectionMinHeight, group);
        goto done;
      }
    } else if (!g_strcmp0 (*key,
            CONFIG_GROUP_INFER_CLASS_ATTRS_DETECTED_MAX_WIDTH)) {
      detection_filter_params.detectionMaxWidth =
          g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_INFER_CLASS_ATTRS_DETECTED_MAX_WIDTH, &error);
      CHECK_ERROR (error);
      if ((gint) detection_filter_params.detectionMaxWidth < 0) {
        g_printerr
            ("Error: Negative detectionMaxWidth (%d) specified for group %s\n",
            detection_filter_params.detectionMaxWidth, group);
        goto done;
      }
    } else if (!g_strcmp0 (*key,
            CONFIG_GROUP_INFER_CLASS_ATTRS_DETECTED_MAX_HEIGHT)) {
      detection_filter_params.detectionMaxHeight =
          g_key_file_get_integer (key_file, group,
          CONFIG_GROUP_INFER_CLASS_ATTRS_DETECTED_MAX_HEIGHT, &error);
      if ((gint) detection_filter_params.detectionMaxHeight < 0) {
        g_printerr
            ("Error: Negative detectionMaxHeight (%d) specified for group %s\n",
            detection_filter_params.detectionMaxHeight, group);
        goto done;
      }
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_INFER_CLASS_ATTRS_BORDER_COLOR)) {
      gsize length = 0;
      gdouble *list = g_key_file_get_double_list (key_file, group,
          CONFIG_GROUP_INFER_CLASS_ATTRS_BORDER_COLOR, &length, &error);
      CHECK_ERROR (error);
      if (length != 4) {
        g_printerr
            ("Error: Group %s, Number of Color params should be exactly 4 "
            "floats {r, g, b, a} between 0 and 1", group);
        goto done;
      }
      color_params.border_color.red = list[0];
      color_params.border_color.green = list[1];
      color_params.border_color.blue = list[2];
      color_params.border_color.alpha = list[3];

      g_free (list);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_INFER_CLASS_ATTRS_BG_COLOR)) {
      gsize length = 0;
      gdouble *list = g_key_file_get_double_list (key_file, group,
          CONFIG_GROUP_INFER_CLASS_ATTRS_BG_COLOR, &length, &error);
      CHECK_ERROR (error);
      if (length != 4) {
        g_printerr
            ("Error: Group %s, Number of Color params should be exactly 4 "
            "floats {r, g, b, a} between 0 and 1", group);
        goto done;
      }
      color_params.bg_color.red = list[0];
      color_params.bg_color.green = list[1];
      color_params.bg_color.blue = list[2];
      color_params.bg_color.alpha = list[3];
      color_params.have_bg_color = TRUE;

      g_free (list);
    } else if (!g_strcmp0(*key, CONFIG_GROUP_INFER_CLASS_ATTRS_NMS_IOU_THRESHOLD)) {
        detection_params.nmsIOUThreshold =  g_key_file_get_double (key_file, group,
          CONFIG_GROUP_INFER_CLASS_ATTRS_NMS_IOU_THRESHOLD, &error);
                CHECK_ERROR (error);
        if (detection_params.nmsIOUThreshold < 0 || detection_params.nmsIOUThreshold > 1) {
            g_printerr ("Error: Invalid nms iou threshold (%.2f) specified for group %s."
                         "Enter a value between 0 & 1. \n",
            detection_params.nmsIOUThreshold, group);
        goto done;
      }
    } else if (!g_strcmp0(*key, CONFIG_GROUP_INFER_CLASS_ATTRS_TOP_K)) {
        detection_params.topK = g_key_file_get_integer(key_file, group,
        CONFIG_GROUP_INFER_CLASS_ATTRS_TOP_K, &error);
            CHECK_ERROR(error);
            if(detection_params.topK < 0) {
                g_printerr("Error: Invalid topk value %d specified for group %s."
                " topk should be greater than of equal to '0'\n", detection_params.topK, group);
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

static gboolean
gst_nvinfer_parse_other_attribute (GstNvInfer * nvinfer,
    GKeyFile * key_file, const gchar * group_name, const gchar * key,
    const gchar * cfg_file_path)
{
  GError *error = nullptr;
  gboolean ret = FALSE;

  assert (nvinfer);
  if (!g_strcmp0 (key, CONFIG_GROUP_INFER_PROCESS_MODE)) {
    if ((*nvinfer->is_prop_set)[PROP_PROCESS_MODE])
      return TRUE;
    guint val = g_key_file_get_integer (key_file, group_name,
        CONFIG_GROUP_INFER_PROCESS_MODE, &error);
    CHECK_ERROR (error);

    switch (val) {
      case 1:
        nvinfer->process_full_frame = TRUE;
        break;
      case 2:
        nvinfer->process_full_frame = FALSE;
        break;
      default:
        g_printerr ("Error: Invalid value for %s (%d)\n",
            CONFIG_GROUP_INFER_PROCESS_MODE, val);
        goto done;
    }
  } else if (!g_strcmp0 (key, CONFIG_GROUP_INFER_CLASSIFIER_ASYNC_MODE)) {
    if (g_key_file_get_boolean (key_file, group_name,
            CONFIG_GROUP_INFER_CLASSIFIER_ASYNC_MODE, &error))
      nvinfer->classifier_async_mode = TRUE;
    CHECK_ERROR (error);
  } else if (!g_strcmp0 (key, CONFIG_GROUP_INFER_CLASSIFIER_TYPE)) {
    nvinfer->classifier_type = g_key_file_get_string (key_file,
            group_name, CONFIG_GROUP_INFER_CLASSIFIER_TYPE, &error);
    CHECK_ERROR (error);
  } else if (!g_strcmp0 (key, CONFIG_GROUP_INFER_INTERVAL)) {
    if ((*nvinfer->is_prop_set)[PROP_INTERVAL])
      return TRUE;
    nvinfer->interval =
        g_key_file_get_integer (key_file, group_name,
        CONFIG_GROUP_INFER_INTERVAL, &error);
    CHECK_ERROR (error);
    if ((gint) nvinfer->interval < 0) {
      g_printerr ("Error: Negative value (%d) specified for interval\n",
          nvinfer->interval);
      goto done;
    }
  } else if (!g_strcmp0 (key, CONFIG_GROUP_INFER_OUTPUT_TENSOR_META)) {
    if (g_key_file_get_boolean (key_file, group_name,
            CONFIG_GROUP_INFER_OUTPUT_TENSOR_META, &error))
      nvinfer->output_tensor_meta = TRUE;
    CHECK_ERROR (error);
  } else if (!g_strcmp0 (key, CONFIG_GROUP_INFER_OUTPUT_INSTANCE_MASK)) {
    if (g_key_file_get_boolean (key_file, group_name,
            CONFIG_GROUP_INFER_OUTPUT_INSTANCE_MASK, &error))
      nvinfer->output_instance_mask = TRUE;
    CHECK_ERROR (error);
  } else if (!g_strcmp0 (key, CONFIG_GROUP_INFER_SECONDARY_REINFER_INTERVAL)) {
    nvinfer->secondary_reinfer_interval =
        g_key_file_get_integer (key_file, group_name,
        CONFIG_GROUP_INFER_SECONDARY_REINFER_INTERVAL, &error);
    CHECK_ERROR (error);
  } else if (!g_strcmp0 (key, CONFIG_GROUP_INFER_MAINTAIN_ASPECT_RATIO)) {
    if (g_key_file_get_boolean (key_file, group_name,
            CONFIG_GROUP_INFER_MAINTAIN_ASPECT_RATIO, &error))
      nvinfer->maintain_aspect_ratio = TRUE;
    CHECK_ERROR (error);
  } else if (!g_strcmp0 (key, CONFIG_GROUP_INFER_SYMMETRIC_PADDING)) {
    if (g_key_file_get_boolean (key_file, group_name,
            CONFIG_GROUP_INFER_SYMMETRIC_PADDING, &error))
      nvinfer->symmetric_padding = TRUE;
    CHECK_ERROR (error);
  } else if (!g_strcmp0 (key, CONFIG_GROUP_INFER_INPUT_OBJECT_MIN_WIDTH)) {
    nvinfer->min_input_object_width = g_key_file_get_integer (key_file,
        group_name, CONFIG_GROUP_INFER_INPUT_OBJECT_MIN_WIDTH,
        &error);
    CHECK_ERROR (error);
    if ((gint) nvinfer->min_input_object_width < 0) {
      g_printerr ("Error: Negative value specified for %s(%d)\n",
          CONFIG_GROUP_INFER_INPUT_OBJECT_MIN_WIDTH,
          nvinfer->min_input_object_width);
      goto done;
    }
  } else if (!g_strcmp0 (key, CONFIG_GROUP_INFER_INPUT_OBJECT_MIN_HEIGHT)) {
    nvinfer->min_input_object_height = g_key_file_get_integer (key_file,
        group_name, CONFIG_GROUP_INFER_INPUT_OBJECT_MIN_HEIGHT,
        &error);
    CHECK_ERROR (error);
    if ((gint) nvinfer->min_input_object_height < 0) {
      g_printerr ("Error: Negative value specified for %s(%d)\n",
          CONFIG_GROUP_INFER_INPUT_OBJECT_MIN_HEIGHT,
          nvinfer->min_input_object_height);
      goto done;
    }
  } else if (!g_strcmp0 (key, CONFIG_GROUP_INFER_INPUT_OBJECT_MAX_WIDTH)) {
    nvinfer->max_input_object_width = g_key_file_get_integer (key_file,
        group_name, CONFIG_GROUP_INFER_INPUT_OBJECT_MAX_WIDTH,
        &error);
    CHECK_ERROR (error);
    if ((gint) nvinfer->max_input_object_width < 0) {
      g_printerr ("Error: Negative value specified for %s(%d)\n",
          CONFIG_GROUP_INFER_INPUT_OBJECT_MAX_WIDTH,
          nvinfer->max_input_object_width);
      goto done;
    }
  } else if (!g_strcmp0 (key, CONFIG_GROUP_INFER_INPUT_OBJECT_MAX_HEIGHT)) {
    nvinfer->max_input_object_height = g_key_file_get_integer (key_file,
        group_name, CONFIG_GROUP_INFER_INPUT_OBJECT_MAX_HEIGHT,
        &error);
    CHECK_ERROR (error);
    if ((gint) nvinfer->max_input_object_height < 0) {
      g_printerr ("Error: Negative value specified for %s(%d)\n",
          CONFIG_GROUP_INFER_INPUT_OBJECT_MAX_HEIGHT,
          nvinfer->max_input_object_height);
      goto done;
    }
  } else if (!g_strcmp0 (key, CONFIG_GROUP_INFER_GIE_ID_FOR_OPERATION)) {
    if ((*nvinfer->is_prop_set)[PROP_OPERATE_ON_GIE_ID] ||
        (*nvinfer->is_prop_set)[PROP_OPERATE_ON_CLASS_IDS])
      return TRUE;
    nvinfer->operate_on_gie_id = g_key_file_get_integer (key_file,
        group_name, CONFIG_GROUP_INFER_GIE_ID_FOR_OPERATION,
        &error);
    CHECK_ERROR (error);
  } else if (!g_strcmp0 (key, CONFIG_GROUP_INFER_CLASS_IDS_FOR_OPERATION)) {
    if ((*nvinfer->is_prop_set)[PROP_OPERATE_ON_GIE_ID] ||
        (*nvinfer->is_prop_set)[PROP_OPERATE_ON_CLASS_IDS])
      return TRUE;
    gsize length = 0, i = 0;
    gint max_class_id = -1;
    gint *int_list =
        g_key_file_get_integer_list (key_file, group_name,
        CONFIG_GROUP_INFER_CLASS_IDS_FOR_OPERATION, &length, &error);
        CHECK_ERROR(error);

    for (i = 0; i < length; i++) {
      if (int_list[i] > max_class_id)
        max_class_id = int_list[i];
    }
    nvinfer->operate_on_class_ids->assign (max_class_id + 1, FALSE);
    for (i = 0; i < length; i++) {
      nvinfer->operate_on_class_ids->at (int_list[i]) = TRUE;
    }
    g_free (int_list);
  } else if (!g_strcmp0 (key, CONFIG_GROUP_INFER_CLASS_IDS_FOR_FILTERING)) {
    gsize length = 0;

    gint *int_list = g_key_file_get_integer_list(key_file, group_name,
        CONFIG_GROUP_INFER_CLASS_IDS_FOR_FILTERING, &length, &error);
    CHECK_ERROR (error);

    for (guint i = 0; i < length; i++) {
      nvinfer->filter_out_class_ids->insert(int_list[i]);
    }
    g_free(int_list);
  } else if (!g_strcmp0 (key, CONFIG_GROUP_INFER_SCALING_COMPUTE_HW)) {
    int val =  g_key_file_get_integer (key_file, group_name,
    CONFIG_GROUP_INFER_SCALING_COMPUTE_HW, &error);
    CHECK_ERROR (error);

    switch (val) {
      case NvBufSurfTransformCompute_Default:
      case NvBufSurfTransformCompute_GPU:
#ifdef __aarch64__
      case NvBufSurfTransformCompute_VIC:
#endif
        break;
      default:
        g_printerr ("Error. Invalid value for '%s':'%d'\n",
            CONFIG_GROUP_INFER_SCALING_COMPUTE_HW, val);
        goto done;
    }
    nvinfer->transform_config_params.compute_mode = (NvBufSurfTransform_Compute) val;
  } else if (!g_strcmp0 (key, CONFIG_GROUP_INFER_SCALING_FILTER)) {
    int val =  g_key_file_get_integer (key_file, group_name,
    CONFIG_GROUP_INFER_SCALING_FILTER, &error);
    CHECK_ERROR (error);

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
        g_printerr ("Error. Invalid value for '%s':'%d'\n",
            CONFIG_GROUP_INFER_SCALING_FILTER, val);
        goto done;
    }
    nvinfer->transform_params.transform_filter = (NvBufSurfTransform_Inter) val;
  } else if (!g_strcmp0 (key, CONFIG_GROUP_INFER_CROP_OBJECTS_TO_ROI_BOUNDARY)) {
    if (g_key_file_get_boolean (key_file, group_name,
            CONFIG_GROUP_INFER_CROP_OBJECTS_TO_ROI_BOUNDARY, &error))
      nvinfer->crop_objects_to_roi_boundary = TRUE;
    CHECK_ERROR (error);
  } else {
      g_printerr ("Unknown or legacy key specified '%s' for group [%s]\n", key,
          CONFIG_GROUP_PROPERTY);
    }

  ret = TRUE;

done:
  if (error) {
    g_error_free (error);
  }
  return ret;
}

/* Parse 'property' group. Returns FALSE in case of an error. If any of the
 * properties are set through the GObject set method this function does not
 * parse those properties i.e. values set through g_object_set override the
 * corresponding properties in the config file. */
static gboolean
gst_nvinfer_parse_props (GstNvInfer * nvinfer,
    NvDsInferContextInitParams * init_params,
    GKeyFile * key_file, const gchar * cfg_file_path)
{
  gboolean ret = FALSE;
  gchar **keys = nullptr;
  gchar **key = nullptr;
  GError *error = nullptr;

  assert (init_params != nullptr);

  /* Handle legacy key names. */
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

  if (nvinfer)
    nvinfer->secondary_reinfer_interval = DEFAULT_REINFER_INTERVAL;

  init_params->networkInputFormat = NvDsInferFormat_RGB;
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
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_INFER_AUTO_INCREASE_MEMORY)) {
      init_params->autoIncMem =
          g_key_file_get_integer (key_file, CONFIG_GROUP_PROPERTY,
          CONFIG_GROUP_INFER_AUTO_INCREASE_MEMORY, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_INFER_MAX_GPU_MEMORY_PERCENTAGE)) {
      init_params->maxGPUMemPer =
          g_key_file_get_integer (key_file, CONFIG_GROUP_PROPERTY,
          CONFIG_GROUP_INFER_MAX_GPU_MEMORY_PERCENTAGE, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_INFER_TENSOR_META_POOL_SIZE)) {
      init_params->outputBufferPoolSize =
          g_key_file_get_integer (key_file, CONFIG_GROUP_PROPERTY,
          CONFIG_GROUP_INFER_TENSOR_META_POOL_SIZE, &error);
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
      gsize length = 0;
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
      gsize length = 0;
      init_params->outputLayerNames =
          g_key_file_get_string_list (key_file, CONFIG_GROUP_PROPERTY,
          CONFIG_GROUP_INFER_OUTPUT_BLOB_NAMES, &length, &error);
      init_params->numOutputLayers = length;

      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_INFER_DUMP_INPUT_TENSOR)) {
      init_params->dumpIpTensor =
          g_key_file_get_integer (key_file, CONFIG_GROUP_PROPERTY,
          CONFIG_GROUP_INFER_DUMP_INPUT_TENSOR, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_INFER_DUMP_OUTPUT_TENSOR)) {
      init_params->dumpOpTensor =
          g_key_file_get_integer (key_file, CONFIG_GROUP_PROPERTY,
          CONFIG_GROUP_INFER_DUMP_OUTPUT_TENSOR, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_INFER_OVERWRITE_INPUT_TENSOR)) {
      init_params->overwriteIpTensor =
          g_key_file_get_integer (key_file, CONFIG_GROUP_PROPERTY,
          CONFIG_GROUP_INFER_OVERWRITE_INPUT_TENSOR, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_INFER_OVERWRITE_OUTPUT_TENSOR)) {
      init_params->overwriteOpTensor =
          g_key_file_get_integer (key_file, CONFIG_GROUP_PROPERTY,
          CONFIG_GROUP_INFER_OVERWRITE_OUTPUT_TENSOR, &error);
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_INFER_INPUT_TENSOR_FILE)) {
      gchar *str = g_key_file_get_string (key_file, CONFIG_GROUP_PROPERTY,
          CONFIG_GROUP_INFER_INPUT_TENSOR_FILE, &error);
      CHECK_ERROR (error);

      if (!get_absolute_file_path (cfg_file_path, str,
              init_params->ipTensorFilePath)) {
        g_printerr ("Error: Could not parse Input Tensor file path\n");
        g_free (str);
        goto done;
      }
      g_free (str);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_INFER_OUTPUT_TENSOR_FILES)) {
      gsize length = 0;
      char** op_file_list =
          g_key_file_get_string_list (key_file, CONFIG_GROUP_PROPERTY,
          CONFIG_GROUP_INFER_OUTPUT_TENSOR_FILES, &length, &error);
      CHECK_ERROR (error);
      init_params->opTensorFilePath = (gchar **) malloc(sizeof(gchar *)*length);
      for (gsize i = 0; i < length; i++)
      {
        gchar *str = (gchar *) malloc(sizeof(gchar *)*_MAX_STR_LENGTH);
        if (!get_absolute_file_path (cfg_file_path, op_file_list[i],
              str)) {
          g_printerr ("Error: Could not parse output Tensor file path\n");
          g_free(str);
          g_free(op_file_list);
          goto done;
        }
        init_params->opTensorFilePath[i] = str;
      }
      g_free(op_file_list);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_INFER_OUTPUT_IO_FORMATS)) {
        gsize length = 0;
        init_params->outputIOFormats = g_key_file_get_string_list(key_file, CONFIG_GROUP_PROPERTY,
        CONFIG_GROUP_INFER_OUTPUT_IO_FORMATS, &length, &error);
        init_params->numOutputIOFormats = length;
        CHECK_ERROR(error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_INFER_LAYER_DEVICE_PRECISION)) {
        gsize length = 0;
        init_params->layerDevicePrecisions = g_key_file_get_string_list(key_file, CONFIG_GROUP_PROPERTY,
        CONFIG_GROUP_INFER_LAYER_DEVICE_PRECISION, &length, &error);
        init_params->numLayerDevicePrecisions = length;
        CHECK_ERROR(error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_INFER_NETWORK_TYPE)) {
      guint val = g_key_file_get_integer (key_file, CONFIG_GROUP_PROPERTY,
          CONFIG_GROUP_INFER_NETWORK_TYPE, &error);
      CHECK_ERROR (error);

      switch ((NvDsInferNetworkType) val) {
        case NvDsInferNetworkType_Detector:
        case NvDsInferNetworkType_Classifier:
        case NvDsInferNetworkType_Segmentation:
        case NvDsInferNetworkType_InstanceSegmentation:
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
      gsize length = 0, i = 0;
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
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_INFER_CUSTOM_PARSE_BBOX_IM_FUNC)) {
      gchar *str = g_key_file_get_string (key_file, CONFIG_GROUP_PROPERTY,
          CONFIG_GROUP_INFER_CUSTOM_PARSE_BBOX_IM_FUNC, &error);
      CHECK_STR_ERROR (error);
      g_strlcpy (init_params->customBBoxInstanceMaskParseFuncName, str,
          _MAX_STR_LENGTH);
      g_free (str);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_INFER_CUSTOM_PARSE_SEGMENTATION_FUNC)) {
      gchar *str = g_key_file_get_string (key_file, CONFIG_GROUP_PROPERTY,
          CONFIG_GROUP_INFER_CUSTOM_PARSE_SEGMENTATION_FUNC, &error);
      CHECK_STR_ERROR (error);
      g_strlcpy (init_params->customSegmentationParseFuncName, str,
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
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_INFER_UFF_INPUT_DIMENSIONS_LEGACY)) {
    /* Parse here if infer-dims has not been set, else ignore input-dims / uff-input-dims */
    if(!g_key_file_has_key(key_file, CONFIG_GROUP_PROPERTY,
      CONFIG_GROUP_INFER_INFER_DIMENSIONS, nullptr)) {
        g_printerr("Warning: 'input-dims' parameter has been deprecated. Use 'infer-dims' instead.\n");
        gsize length = 0;
        gint *int_list = g_key_file_get_integer_list (key_file,
                         CONFIG_GROUP_PROPERTY, CONFIG_GROUP_INFER_UFF_INPUT_DIMENSIONS_LEGACY,
                         &length, &error);
        CHECK_ERROR (error);

        if (length != 4) {
          g_printerr ("Error. '%s' array length is %lu. Should be 4[a;b;c;ORDER].\n",
                      CONFIG_GROUP_INFER_UFF_INPUT_DIMENSIONS_LEGACY, length);
          goto done;
        }
        switch (int_list[3]) {
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
          g_printerr ("Error. Invalid value for '%s', UFF input order :'%d'\n",
                     CONFIG_GROUP_INFER_UFF_INPUT_DIMENSIONS_LEGACY, int_list[3]);
          goto done;
          break;
        }
        init_params->inferInputDims = NvDsInferDimsCHW {(unsigned int) int_list[0],
                                      (unsigned int) int_list[1], (unsigned int) int_list[2]};
        g_free (int_list);
    }
    else
      g_printerr("Warning: Ignoring 'input-dims' parameter since 'infer-dims' has been set.\n");
  } else if (!g_strcmp0 (*key, CONFIG_GROUP_INFER_UFF_INPUT_DIMENSIONS_LEGACY_V2)) {
    /* Parse here if infer-dims has not been set, else ignore input-dims / uff-input-dims */
    if(!g_key_file_has_key(key_file, CONFIG_GROUP_PROPERTY,
      CONFIG_GROUP_INFER_INFER_DIMENSIONS, nullptr)) {
        g_printerr("Warning: 'input-dims' parameter has been deprecated. Use 'infer-dims' instead.\n");
        gsize length = 0;
        gint *int_list = g_key_file_get_integer_list (key_file,
                         CONFIG_GROUP_PROPERTY, CONFIG_GROUP_INFER_UFF_INPUT_DIMENSIONS_LEGACY_V2,
                         &length, &error);
        CHECK_ERROR (error);

        if (length != 4) {
          g_printerr ("Error. '%s' array length is %lu. Should be 4[a;b;c;ORDER].\n",
                      CONFIG_GROUP_INFER_UFF_INPUT_DIMENSIONS_LEGACY_V2, length);
          goto done;
        }
        switch (int_list[3]) {
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
          g_printerr ("Error. Invalid value for '%s', UFF input order :'%d'\n",
                     CONFIG_GROUP_INFER_UFF_INPUT_DIMENSIONS_LEGACY_V2, int_list[3]);
          goto done;
          break;
        }
        init_params->inferInputDims = NvDsInferDimsCHW {(unsigned int) int_list[0],
                                      (unsigned int) int_list[1], (unsigned int) int_list[2]};
        g_free (int_list);
    }
    else
      g_printerr("Warning: Ignoring 'uff-input-dims' parameter since 'infer-dims' has been set.\n");
  } else if (!g_strcmp0 (*key, CONFIG_GROUP_INFER_NET_INPUT_ORDER)){
    gint val = g_key_file_get_integer (key_file, CONFIG_GROUP_PROPERTY,
    CONFIG_GROUP_INFER_NET_INPUT_ORDER, &error);
    CHECK_ERROR (error);

    switch (val) {
        case 0:
            init_params->netInputOrder = NvDsInferTensorOrder_kNCHW;
            break;
        case 1:
            init_params->netInputOrder = NvDsInferTensorOrder_kNHWC;
            break;
        default:
            g_printerr ("Error. Invalid value for '%s', network input order :'%d'\n",
                CONFIG_GROUP_INFER_NET_INPUT_ORDER, val);
            goto done;
            break;
    }
  } else if (!g_strcmp0 (*key, CONFIG_GROUP_INFER_UFF_INPUT_ORDER)){
        gint val = g_key_file_get_integer (key_file, CONFIG_GROUP_PROPERTY,
          CONFIG_GROUP_INFER_UFF_INPUT_ORDER, &error);
      CHECK_ERROR (error);

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
                g_printerr ("Error. Invalid value for '%s', UFF input order :'%d'\n",
                    CONFIG_GROUP_INFER_UFF_INPUT_ORDER, val);
                goto done;
                break;
        }
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
        g_printerr("Warn: 'enable-dbscan' parameter has been deprecated."
            " Use 'cluster-mode' instead.\n");
        if(g_key_file_has_key (key_file, CONFIG_GROUP_PROPERTY,
          CONFIG_GROUP_INFER_CLUSTER_MODE, nullptr)) {
            g_printerr("Warn: Ignoring 'enable-dbscan' parameter since"
              " 'cluster-mode' has been set.\n");
        } else if (g_key_file_get_boolean (key_file, CONFIG_GROUP_PROPERTY,
              CONFIG_GROUP_INFER_ENABLE_DBSCAN, &error))
              {
                CHECK_ERROR (error);
                init_params->clusterMode = NVDSINFER_CLUSTER_DBSCAN;
              }
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
          init_params->clusterMode = NVDSINFER_CLUSTER_DBSCAN_NMS_HYBRID;
          break;
        case 4:
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
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_INFER_SEGMENTATION_OUTPUT_ORDER)){
      gint val = g_key_file_get_integer (key_file, CONFIG_GROUP_PROPERTY,
          CONFIG_GROUP_INFER_SEGMENTATION_OUTPUT_ORDER, &error);
      CHECK_ERROR (error);
      switch (val) {
        case 0:
          init_params->segmentationOutputOrder = NvDsInferTensorOrder_kNCHW;
          break;
        case 1:
          init_params->segmentationOutputOrder = NvDsInferTensorOrder_kNHWC;
          break;
        default:
          g_printerr ("Error. Invalid value for '%s', Segmentation output order :'%d'\n",
          CONFIG_GROUP_INFER_SEGMENTATION_OUTPUT_ORDER, val);
          goto done;
          break;
      }
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_INFER_INPUT_FROM_META)) {
      if ((*nvinfer->is_prop_set)[PROP_INPUT_TENSOR_META])
        continue;
      if (g_key_file_get_boolean (key_file, CONFIG_GROUP_PROPERTY,
              CONFIG_GROUP_INFER_INPUT_FROM_META, &error)) {
        nvinfer->input_tensor_from_meta = TRUE;
        init_params->inputFromPreprocessedTensor = TRUE;
      }
        CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_INFER_DISABLE_OUTPUT_HOST_COPY)) {
      init_params->disableOutputHostCopy = g_key_file_get_integer (key_file,
          CONFIG_GROUP_PROPERTY, CONFIG_GROUP_INFER_DISABLE_OUTPUT_HOST_COPY,
          &error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_INFER_RAW_OUTPUT_FILE_WRITE)) {
      if (g_key_file_get_boolean (key_file, CONFIG_GROUP_PROPERTY,
              CONFIG_GROUP_INFER_RAW_OUTPUT_FILE_WRITE, &error)) {
        nvinfer->write_raw_buffers_to_file = TRUE;
      }
      CHECK_ERROR (error);
  } else if (!g_strcmp0 (*key, CONFIG_GROUP_INFER_STRONGLY_TYPED)) {
    if (g_key_file_get_boolean (key_file, CONFIG_GROUP_PROPERTY,
            CONFIG_GROUP_INFER_STRONGLY_TYPED, &error))
      init_params->useStronglyTyped = TRUE;
    CHECK_ERROR (error);
  } else if (!g_strcmp0 (*key, CONFIG_GROUP_INFER_WARMUP_ENGINE)) {
    if (g_key_file_get_boolean (key_file, CONFIG_GROUP_PROPERTY,
            CONFIG_GROUP_INFER_WARMUP_ENGINE, &error))
      init_params->warmupEngine = TRUE;
    CHECK_ERROR (error);
  } else if (!g_strcmp0 (*key, CONFIG_GROUP_INFER_STABILIZE_BUFFER_COUNT)) {
    guint val = (guint) g_key_file_get_integer (key_file, CONFIG_GROUP_PROPERTY,
        CONFIG_GROUP_INFER_STABILIZE_BUFFER_COUNT, &error);
    CHECK_ERROR (error);
    if (nvinfer)
      nvinfer->stabilize_buffer_count = val;
  } else if (!g_strcmp0 (*key, CONFIG_GROUP_INFER_DYNAMIC_PROPERTIES)) {
    gchar *str = g_key_file_get_string (key_file, CONFIG_GROUP_PROPERTY,
            CONFIG_GROUP_INFER_DYNAMIC_PROPERTIES, &error);
    CHECK_ERROR (error);
    
    /* Parse dynamic properties string format: key1=value1;key2=value2 */
    if (str && strlen(str) > 0) {
      gchar **pairs = g_strsplit(str, ";", -1);
      if (pairs) {
        for (gchar **pair = pairs; *pair != NULL; pair++) {
          gchar *trimmed = g_strstrip(*pair);
          if (strlen(trimmed) > 0) {
            gchar **kv = g_strsplit(trimmed, "=", 2);
            if (kv && kv[0] && kv[1]) {
              gchar *key_name = g_strstrip(kv[0]);
              gchar *value = g_strstrip(kv[1]);
              if (strlen(key_name) > 0 && strlen(value) > 0) {
                NvDsInferContext_SetDynamicProperty(init_params, key_name, value);
              }
            }
            g_strfreev(kv);
          }
        }
        g_strfreev(pairs);
      }
    }
    g_free (str);
    } else if (nvinfer) {
      if (!gst_nvinfer_parse_other_attribute (
          nvinfer, key_file, CONFIG_GROUP_PROPERTY, *key, cfg_file_path)) {
        goto done;
      }
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
    GstNvInfer * nvinfer,
    NvDsInferContextInitParams *init_params,
    const gchar * cfg_file_path)
{
  GError *error = nullptr;
  gboolean ret = FALSE;
  gchar **groups = nullptr;
  gchar **group = nullptr;
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

  if (!gst_nvinfer_parse_props (nvinfer, init_params, cfg_file, cfg_file_path)) {
    g_printerr ("Failed to parse group %s\n", CONFIG_GROUP_PROPERTY);
    goto done;
  }
  g_key_file_remove_group (cfg_file, CONFIG_GROUP_PROPERTY, nullptr);

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
    if (g_key_file_has_group (cfg_file,
            CONFIG_GROUP_INFER_CLASS_ATTRS_PREFIX "all")) {
      if (!gst_nvinfer_parse_class_attrs (cfg_file,
              (gchar *) CONFIG_GROUP_INFER_CLASS_ATTRS_PREFIX "all",
              detection_params, detection_filter_params, color_params)) {
        g_printerr ("Error while parsing group %s\n",
            CONFIG_GROUP_INFER_CLASS_ATTRS_PREFIX "all");
        goto done;
      }
      g_key_file_remove_group (cfg_file,
          CONFIG_GROUP_INFER_CLASS_ATTRS_PREFIX "all", nullptr);
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
    groups = g_key_file_get_groups (cfg_file, nullptr);
    for (group = groups; *group; group++) {
      if (!strncmp (*group, CONFIG_GROUP_INFER_CLASS_ATTRS_PREFIX,
              sizeof (CONFIG_GROUP_INFER_CLASS_ATTRS_PREFIX) - 1)) {
        gchar *key1 =
            *group + sizeof (CONFIG_GROUP_INFER_CLASS_ATTRS_PREFIX) - 1;
        gchar *endptr = nullptr;
        guint64 class_index = g_ascii_strtoull (key1, &endptr, 10);

        /* Check that class_index has been parsed successfully and that it lies
         * within the valid range of class_ids [0, numDetectedClasses - 1]. */
        if (class_index == 0 && endptr == key1) {
          g_printerr
              ("Invalid group [%s]. Class attributes should be specified using group name '"
              CONFIG_GROUP_INFER_CLASS_ATTRS_PREFIX "<class-id>'\n", *group);
          goto done;
        }
        if ((gint) class_index < 0) {
          g_printerr ("Invalid group [%s]. class-id should be >= 0\n", *group);
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
        if (!gst_nvinfer_parse_class_attrs (cfg_file, *group,
                init_params->perClassDetectionParams[class_index],
                (*nvinfer->perClassDetectionFilterParams)[class_index],
                (*nvinfer->perClassColorParams)[class_index])) {
          g_printerr ("Error while parsing group %s\n", *group);
          goto done;
        }
      }
    }
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
