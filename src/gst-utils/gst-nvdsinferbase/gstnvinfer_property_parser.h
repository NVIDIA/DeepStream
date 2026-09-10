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

#ifndef __GST_NVINFER_PROPERTY_PARSER_H__
#define __GST_NVINFER_PROPERTY_PARSER_H__

#include <glib.h>

#include "nvdsinfer_context.h"
#include "gstnvinferbase.h"

#define CONFIG_GROUP_PROPERTY "property"

#define CONFIG_GROUP_INFER_PARSE_FUNC "parse-func"

/** Gstreamer element configuration. */
#define CONFIG_GROUP_INFER_UNIQUE_ID "gie-unique-id"
#define CONFIG_GROUP_INFER_INTERVAL "interval"
#define CONFIG_GROUP_INFER_LABEL "labelfile-path"
#define CONFIG_GROUP_INFER_GPU_ID "gpu-id"
#define CONFIG_GROUP_INFER_OUTPUT_TENSOR_META "output-tensor-meta"


#define CONFIG_GROUP_INFER_ENABLE_DLA "enable-dla"
#define CONFIG_GROUP_INFER_USE_DLA_CORE "use-dla-core"

/** Runtime engine parameters. */
#define CONFIG_GROUP_INFER_BATCH_SIZE "batch-size"
#define CONFIG_GROUP_INFER_NETWORK_MODE "network-mode"
#define CONFIG_GROUP_INFER_MODEL_ENGINE "model-engine-file"
#define CONFIG_GROUP_INFER_INT8_CALIBRATION_FILE "int8-calib-file"
#define CONFIG_GROUP_INFER_WORKSPACE_SIZE "workspace-size"

/** Generic model parameters. */
#define CONFIG_GROUP_INFER_OUTPUT_BLOB_NAMES "output-blob-names"
#define CONFIG_GROUP_INFER_IS_CLASSIFIER_LEGACY "is-classifier"
#define CONFIG_GROUP_INFER_NETWORK_TYPE "network-type"
#define CONFIG_GROUP_INFER_FORCE_IMPLICIT_BATCH_DIM "force-implicit-batch-dim"
#define CONFIG_GROUP_INFER_INFER_DIMENSIONS "infer-dims"

/** Preprocessing parameters. */
#define CONFIG_GROUP_INFER_MODEL_COLOR_FORMAT "model-color-format"
#define CONFIG_GROUP_INFER_SCALE_FACTOR "net-scale-factor"
#define CONFIG_GROUP_INFER_OFFSETS "offsets"
#define CONFIG_GROUP_INFER_MEANFILE "mean-file"

/** Custom implementation required to support a network. */
#define CONFIG_GROUP_INFER_CUSTOM_LIB_PATH "custom-lib-path"
#define CONFIG_GROUP_INFER_CUSTOM_PARSE_BBOX_FUNC "parse-bbox-func-name"
#define CONFIG_GROUP_INFER_CUSTOM_ENGINE_CREATE_FUNC "engine-create-func-name"
#define CONFIG_GROUP_INFER_CUSTOM_PARSE_CLASSIFIER_FUNC "parse-classifier-func-name"
#define CONFIG_GROUP_INFER_CUSTOM_NETWORK_CONFIG "custom-network-config"

/** Caffe model specific parameters. */
#define CONFIG_GROUP_INFER_MODEL "model-file"
#define CONFIG_GROUP_INFER_PROTO "proto-file"

/** UFF model specific parameters. */
#define CONFIG_GROUP_INFER_UFF "uff-file"
#define CONFIG_GROUP_INFER_UFF_INPUT_DIMENSIONS "uff-input-dims"
#define CONFIG_GROUP_INFER_UFF_INPUT_DIMENSIONS_LEGACY "input-dims"
#define CONFIG_GROUP_INFER_UFF_INPUT_BLOB_NAME "uff-input-blob-name"

/** TLT model parameters. */
#define CONFIG_GROUP_INFER_TLT_ENCODED_MODEL "tlt-encoded-model"
#define CONFIG_GROUP_INFER_TLT_MODEL_KEY "tlt-model-key"

/** ONNX model specific parameters. */
#define CONFIG_GROUP_INFER_ONNX "onnx-file"

/** Detector specific parameters. */
#define CONFIG_GROUP_INFER_NUM_DETECTED_CLASSES "num-detected-classes"
#define CONFIG_GROUP_INFER_ENABLE_DBSCAN "enable-dbscan"
#define CONFIG_GROUP_INFER_CLUSTER_MODE "cluster-mode"

/** Classifier specific parameters. */
#define CONFIG_GROUP_INFER_CLASSIFIER_THRESHOLD "classifier-threshold"

/** Segmentaion specific parameters. */
#define CONFIG_GROUP_INFER_SEGMENTATION_THRESHOLD "segmentation-threshold"

gboolean gst_nvinfer_parse_config_file (GstNvInferBase *nvinfer,
        NvDsInferContextInitParams *init_params, const gchar * cfg_file_path);

gboolean gst_nvinfer_parse_context_params (NvDsInferContextInitParams *params,
        const gchar * cfg_file_path);


#endif /*__GST_NVINFER_PROPERTY_PARSER_H__*/
