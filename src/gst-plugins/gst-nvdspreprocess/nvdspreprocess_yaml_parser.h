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

#ifndef __GST_NVDS_PREPROCESS_YAML_PARSER_H__
#define __GST_NVDS_PREPROCESS_YAML_PARSER_H__

#include <gst/gst.h>
#include "gstnvdspreprocess.h"

/**
 * Parse config file for GstNvDsPreProcess structure.
 *
 * @param nvdspreprocess pointer to GstNvDsPreProcess structure
 *
 * @param cfg_file_path config file path
 *
 * @return boolean denoting if successfully parsed config file
 */
gboolean gst_nvdspreprocess_parse_config_file_yaml (GstNvDsPreProcess *nvdspreprocess, const gchar * cfg_file_path);

#endif /* __GST_NVDS_PREPROCESS_YAML_PARSER_H__ */