/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef NVDSPREPROCESS_PROPERTY_FILE_PARSER_H_
#define NVDSPREPROCESS_PROPERTY_FILE_PARSER_H_

#include <gst/gst.h>
#include "gstnvdspreprocess.h"

/**
 * This file describes the Macro defined for config file property parser.
 */

/** max string length */
#define _PATH_MAX 4096

#define NVDSPREPROCESS_PROPERTY "property"
#define NVDSPREPROCESS_PROPERTY_TARGET_IDS "target-unique-ids"
#define NVDSPREPROCESS_PROPERTY_GIE_ID_FOR_OPERATION "operate-on-gie-id"
#define NVDSPREPROCESS_PROPERTY_ENABLE "enable"
#define NVDSPREPROCESS_PROPERTY_UNIQUE_ID "unique-id"
#define NVDSPREPROCESS_PROPERTY_GPU_ID "gpu-id"
#define NVDSPREPROCESS_PROPERTY_PROCESS_ON_FRAME "process-on-frame"
#define NVDSPREPROCESS_PROPERTY_PROCESSING_WIDTH "processing-width"
#define NVDSPREPROCESS_PROPERTY_PROCESSING_HEIGHT "processing-height"
#define NVDSPREPROCESS_PROPERTY_MAINTAIN_ASPECT_RATIO "maintain-aspect-ratio"
#define NVDSPREPROCESS_PROPERTY_SYMMETRIC_PADDING "symmetric-padding"
#define NVDSPREPROCESS_PROPERTY_TENSOR_BUF_POOL_SIZE "tensor-buf-pool-size"

#define NVDSPREPROCESS_PROPERTY_SCALING_BUF_POOL_SIZE "scaling-buf-pool-size"
#define NVDSPREPROCESS_PROPERTY_SCALING_FILTER "scaling-filter"
#define NVDSPREPROCESS_PROPERTY_SCALING_POOL_COMPUTE_HW "scaling-pool-compute-hw"
#define NVDSPREPROCESS_PROPERTY_SCALING_POOL_MEMORY_TYPE "scaling-pool-memory-type"

#define NVDSPREPROCESS_PROPERTY_NETWORK_INPUT_ORDER "network-input-order"
#define NVDSPREPROCESS_PROPERTY_NETWORK_SHAPE "network-input-shape"
#define NVDSPREPROCESS_PROPERTY_NETWORK_COLOR_FORMAT "network-color-format"
#define NVDSPREPROCESS_PROPERTY_TENSOR_DATA_TYPE "tensor-data-type"
#define NVDSPREPROCESS_PROPERTY_TENSOR_NAME "tensor-name"

#define NVDSPREPROCESS_PROPERTY_CUSTOM_LIB_NAME "custom-lib-path"
#define NVDSPREPROCESS_PROPERTY_TENSOR_PREPARATION_FUNCTION "custom-tensor-preparation-function"

#define NVDSPREPROCESS_USER_CONFIGS "user-configs"

#define NVDSPREPROCESS_GROUP "group-"
#define NVDSPREPROCESS_GROUP_SRC_IDS "src-ids"
#define NVDSPREPROCESS_GROUP_PROCESS_ON_ROI "process-on-roi"
#define NVDSPREPROCESS_GROUP_PROCESS_ON_ALL_OBJECTS "process-on-all-objects"
#define NVDSPREPROCESS_GROUP_ROI_PARAMS_SRC "roi-params-src"
#define NVDSPREPROCESS_GROUP_CUSTOM_INPUT_PREPROCESS_FUNCTION "custom-input-transformation-function"
#define NVDSPREPROCESS_GROUP_DRAW_ROI "draw-roi"
#define NVDSPREPROCESS_GROUP_ROI_COLOR "roi-color"
#define NVDSPREPROCESS_GROUP_OPERATE_ON_CLASS_IDS "operate-on-class-ids"
#define NVDSPREPROCESS_GROUP_INTERVAL "interval"
/** Parameters for filtering objects based min/max size threshold
 * when operating in secondary mode.
 */
#define NVDSPREPROCESS_GROUP_OBJECT_MIN_WIDTH "input-object-min-width"
#define NVDSPREPROCESS_GROUP_OBJECT_MIN_HEIGHT "input-object-min-height"
#define NVDSPREPROCESS_GROUP_OBJECT_MAX_WIDTH "input-object-max-width"
#define NVDSPREPROCESS_GROUP_OBJECT_MAX_HEIGHT "input-object-max-height"
/**
 * Get GstNvDsPreProcessMemory structure associated with buffer allocated using
 * GstNvDsPreProcessAllocator.
 *
 * @param nvdspreprocess pointer to GstNvDsPreProcess structure
 *
 * @param cfg_file_path config file path
 *
 * @return boolean denoting if successfully parsed config file
 */
gboolean
nvdspreprocess_parse_config_file (GstNvDsPreProcess *nvdspreprocess, gchar *cfg_file_path);

#endif /* NVDSPREPROCESS_PROPERTY_FILE_PARSER_H_ */
