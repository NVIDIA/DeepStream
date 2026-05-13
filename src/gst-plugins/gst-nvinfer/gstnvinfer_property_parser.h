/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef __GST_NVINFER_PROPERTY_PARSER_H__
#define __GST_NVINFER_PROPERTY_PARSER_H__

#include <glib.h>

#include "nvdsinfer_context.h"
#include "gstnvinfer.h"

#define DEFAULT_PRE_CLUSTER_THRESHOLD 0.2
#define DEFAULT_POST_CLUSTER_THRESHOLD 0.0
#define DEFAULT_EPS 0.0
#define DEFAULT_GROUP_THRESHOLD 0
#define DEFAULT_MIN_BOXES 0
#define DEFAULT_DBSCAN_MIN_SCORE 0
#define DEFAULT_NMS_IOU_THRESHOLD 0.3
#define DEFAULT_TOP_K -1

#define CONFIG_GROUP_PROPERTY "property"

#define CONFIG_GROUP_INFER_PARSE_FUNC "parse-func"

/** Gstreamer element configuration. */
#define CONFIG_GROUP_INFER_UNIQUE_ID "gie-unique-id"
#define CONFIG_GROUP_INFER_PROCESS_MODE "process-mode"
#define CONFIG_GROUP_INFER_INTERVAL "interval"
#define CONFIG_GROUP_INFER_LABEL "labelfile-path"
#define CONFIG_GROUP_INFER_GPU_ID "gpu-id"
#define CONFIG_GROUP_INFER_SECONDARY_REINFER_INTERVAL "secondary-reinfer-interval"
#define CONFIG_GROUP_INFER_OUTPUT_TENSOR_META "output-tensor-meta"
//gpu_post_processing 0 is not 1 is yes
#define CONFIG_GROUP_INFER_DISABLE_OUTPUT_HOST_COPY "disable-output-host-copy"

#define CONFIG_GROUP_INFER_AUTO_INCREASE_MEMORY "auto-inc-mem"
#define CONFIG_GROUP_INFER_MAX_GPU_MEMORY_PERCENTAGE "max-gpu-mem-per"

#define CONFIG_GROUP_INFER_ENABLE_DLA "enable-dla"
#define CONFIG_GROUP_INFER_USE_DLA_CORE "use-dla-core"

#define CONFIG_GROUP_INFER_DUMP_INPUT_TENSOR "dump-input-tensor"
#define CONFIG_GROUP_INFER_DUMP_OUTPUT_TENSOR "dump-output-tensor"
#define CONFIG_GROUP_INFER_OVERWRITE_INPUT_TENSOR "overwrite-input-tensor"
#define CONFIG_GROUP_INFER_OVERWRITE_OUTPUT_TENSOR "overwrite-output-tensor"
#define CONFIG_GROUP_INFER_INPUT_TENSOR_FILE "ip-tensor-file"
#define CONFIG_GROUP_INFER_OUTPUT_TENSOR_FILES "op-tensor-files"
#define CONFIG_GROUP_INFER_RAW_OUTPUT_FILE_WRITE "raw-output-file-write"

/** Runtime engine parameters. */
#define CONFIG_GROUP_INFER_BATCH_SIZE "batch-size"
#define CONFIG_GROUP_INFER_TENSOR_META_POOL_SIZE "tensor-meta-pool-size"
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
#define CONFIG_GROUP_INFER_OUTPUT_IO_FORMATS "output-io-formats"
#define CONFIG_GROUP_INFER_LAYER_DEVICE_PRECISION "layer-device-precision"

/** Preprocessing parameters. */
#define CONFIG_GROUP_INFER_MODEL_COLOR_FORMAT "model-color-format"
#define CONFIG_GROUP_INFER_SCALE_FACTOR "net-scale-factor"
#define CONFIG_GROUP_INFER_OFFSETS "offsets"
#define CONFIG_GROUP_INFER_MEANFILE "mean-file"
#define CONFIG_GROUP_INFER_MAINTAIN_ASPECT_RATIO "maintain-aspect-ratio"
#define CONFIG_GROUP_INFER_SYMMETRIC_PADDING "symmetric-padding"
#define CONFIG_GROUP_INFER_SCALING_FILTER "scaling-filter"
#define CONFIG_GROUP_INFER_SCALING_COMPUTE_HW "scaling-compute-hw"
#define CONFIG_GROUP_INFER_NET_INPUT_ORDER "network-input-order"
#define CONFIG_GROUP_INFER_INPUT_FROM_META "input-tensor-from-meta"

/** Custom implementation required to support a network. */
#define CONFIG_GROUP_INFER_CUSTOM_LIB_PATH "custom-lib-path"
#define CONFIG_GROUP_INFER_CUSTOM_PARSE_BBOX_FUNC "parse-bbox-func-name"
#define CONFIG_GROUP_INFER_CUSTOM_PARSE_BBOX_IM_FUNC "parse-bbox-instance-mask-func-name"
#define CONFIG_GROUP_INFER_CUSTOM_PARSE_SEGMENTATION_FUNC "parse-segmentation-func-name"
#define CONFIG_GROUP_INFER_CUSTOM_ENGINE_CREATE_FUNC "engine-create-func-name"
#define CONFIG_GROUP_INFER_CUSTOM_PARSE_CLASSIFIER_FUNC "parse-classifier-func-name"
#define CONFIG_GROUP_INFER_CUSTOM_NETWORK_CONFIG "custom-network-config"

/** Caffe model specific parameters. */
#define CONFIG_GROUP_INFER_MODEL "model-file"
#define CONFIG_GROUP_INFER_PROTO "proto-file"

/** UFF model specific parameters. */
#define CONFIG_GROUP_INFER_UFF "uff-file"
#define CONFIG_GROUP_INFER_UFF_INPUT_ORDER "uff-input-order"
#define CONFIG_GROUP_INFER_UFF_INPUT_DIMENSIONS_LEGACY "input-dims"
#define CONFIG_GROUP_INFER_UFF_INPUT_DIMENSIONS_LEGACY_V2 "uff-input-dims"
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
#define CONFIG_GROUP_INFER_CROP_OBJECTS_TO_ROI_BOUNDARY "crop-objects-to-roi-boundary"
#define CONFIG_GROUP_INFER_STRONGLY_TYPED "strongly-typed"
#define CONFIG_GROUP_INFER_DYNAMIC_PROPERTIES "dynamic-properties"
#define CONFIG_GROUP_INFER_WARMUP_ENGINE "warmup-engine"
#define CONFIG_GROUP_INFER_STABILIZE_BUFFER_COUNT "stabilize-buffer-count"

/** Classifier specific parameters. */
#define CONFIG_GROUP_INFER_CLASSIFIER_TYPE "classifier-type"
#define CONFIG_GROUP_INFER_CLASSIFIER_THRESHOLD "classifier-threshold"
#define CONFIG_GROUP_INFER_CLASSIFIER_ASYNC_MODE "classifier-async-mode"

/** Segmentaion specific parameters. */
#define CONFIG_GROUP_INFER_SEGMENTATION_THRESHOLD "segmentation-threshold"
#define CONFIG_GROUP_INFER_SEGMENTATION_OUTPUT_ORDER "segmentation-output-order"

/** Instance Segmentaion specific parameters. */
#define CONFIG_GROUP_INFER_OUTPUT_INSTANCE_MASK "output-instance-mask"

/** Parameters for filtering objects based min/max size threshold when
    operating in secondary mode. */
#define CONFIG_GROUP_INFER_INPUT_OBJECT_MIN_WIDTH "input-object-min-width"
#define CONFIG_GROUP_INFER_INPUT_OBJECT_MIN_HEIGHT "input-object-min-height"
#define CONFIG_GROUP_INFER_INPUT_OBJECT_MAX_WIDTH "input-object-max-width"
#define CONFIG_GROUP_INFER_INPUT_OBJECT_MAX_HEIGHT "input-object-max-height"

/** Parameters for filtering objects based on class-id and unique id of the
    detector when operating in secondary mode. */
#define CONFIG_GROUP_INFER_GIE_ID_FOR_OPERATION "operate-on-gie-id"
#define CONFIG_GROUP_INFER_CLASS_IDS_FOR_OPERATION "operate-on-class-ids"
#define CONFIG_GROUP_INFER_CLASS_IDS_FOR_FILTERING "filter-out-class-ids"

/** Per-class detection/filtering parameters. */
#define CONFIG_GROUP_INFER_CLASS_ATTRS_PREFIX "class-attrs-"
#define CONFIG_GROUP_INFER_CLASS_ATTRS_THRESHOLD "threshold"
#define CONFIG_GROUP_INFER_CLASS_ATTRS_PRE_CLUSTER_THRESHOLD "pre-cluster-threshold"
#define CONFIG_GROUP_INFER_CLASS_ATTRS_POST_CLUSTER_THRESHOLD "post-cluster-threshold"
#define CONFIG_GROUP_INFER_CLASS_ATTRS_EPS "eps"
#define CONFIG_GROUP_INFER_CLASS_ATTRS_GROUP_THRESHOLD "group-threshold"
#define CONFIG_GROUP_INFER_CLASS_ATTRS_MIN_BOXES "minBoxes"
#define CONFIG_GROUP_INFER_CLASS_ATTRS_DBSCAN_MIN_SCORE "dbscan-min-score"
#define CONFIG_GROUP_INFER_CLASS_ATTRS_ROI_TOP_OFFSET "roi-top-offset"
#define CONFIG_GROUP_INFER_CLASS_ATTRS_ROI_BOTTOM_OFFSET "roi-bottom-offset"
#define CONFIG_GROUP_INFER_CLASS_ATTRS_DETECTED_MIN_WIDTH "detected-min-w"
#define CONFIG_GROUP_INFER_CLASS_ATTRS_DETECTED_MIN_HEIGHT "detected-min-h"
#define CONFIG_GROUP_INFER_CLASS_ATTRS_DETECTED_MAX_WIDTH "detected-max-w"
#define CONFIG_GROUP_INFER_CLASS_ATTRS_DETECTED_MAX_HEIGHT "detected-max-h"
#define CONFIG_GROUP_INFER_CLASS_ATTRS_BORDER_COLOR "border-color"
#define CONFIG_GROUP_INFER_CLASS_ATTRS_BG_COLOR "bg-color"
#define CONFIG_GROUP_INFER_CLASS_ATTRS_NMS_IOU_THRESHOLD "nms-iou-threshold"
#define CONFIG_GROUP_INFER_CLASS_ATTRS_TOP_K "topk"

gboolean gst_nvinfer_parse_config_file (GstNvInfer *nvinfer,
        NvDsInferContextInitParams *init_params, const gchar * cfg_file_path);

gboolean gst_nvinfer_parse_context_params (NvDsInferContextInitParams *params,
        const gchar * cfg_file_path);


#endif /*__GST_NVINFER_PROPERTY_PARSER_H__*/
