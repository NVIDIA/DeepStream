/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "lidar_feeder.hpp"
#include "custom_factory.hpp"
#include "common_factory.hpp"
#include "plugin.h"

using namespace deepstream;

#define FACTORY_NAME "lidar_feeder"

DS_CUSTOM_FACTORY_DEFINE_PARAMS_BEGIN(param_spec)
DS_CUSTOM_FACTORY_DEFINE_PARAM(
  config-path,
  path,
  "file path for lidar data",
  "the file contains the data for feeding the appsrc",
  ""
)

DS_CUSTOM_FACTORY_DEFINE_PARAMS_END

DS_CUSTOM_PLUGIN_DEFINE(
    lidar_feeder,
    "this is a sample lidar data feeder plugin",
    "0.1",
    "Proprietary")

DS_CUSTOM_FACTORY_DEFINE_WITH_PARAMS(
  FACTORY_NAME,
  "lidar feeder factory",
  "signal",
  "this is a lidar feeder factory",
  "NVIDIA",
  "need-data/enough-data",
  param_spec,
  DataFeeder,
  LiDARDataSource
)
