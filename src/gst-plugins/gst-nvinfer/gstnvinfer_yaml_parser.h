/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef __GST_NVINFER_YAML_PARSER_H__
#define __GST_NVINFER_YAML_PARSER_H__

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


gboolean gst_nvinfer_parse_config_file_yaml (GstNvInfer *nvinfer,
        NvDsInferContextInitParams *init_params, const gchar * cfg_file_path);

gboolean gst_nvinfer_parse_context_params_yaml (NvDsInferContextInitParams *params,
        const gchar * cfg_file_path);


#endif /*__GST_NVINFER_PROPERTY_PARSER_H__*/
