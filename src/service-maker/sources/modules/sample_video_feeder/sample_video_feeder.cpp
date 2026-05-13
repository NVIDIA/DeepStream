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

#include "sample_video_feeder.hpp"
#include "custom_factory.hpp"
#include "common_factory.hpp"
#include "plugin.h"

using namespace deepstream;

#define FACTORY_NAME "sample_video_feeder"

DS_CUSTOM_FACTORY_DEFINE_PARAMS_BEGIN(param_spec)
DS_CUSTOM_FACTORY_DEFINE_PARAM(
  location,
  path,
  "file path for data",
  "the file contains the data for feeding the appsrc",
  ""
)
DS_CUSTOM_FACTORY_DEFINE_PARAM(
  frame-width,
  integer,
  "width of a video frame",
  "width of a video frame in bytes",
  0
)
DS_CUSTOM_FACTORY_DEFINE_PARAM(
  frame-height,
  integer,
  "height of a video frame",
  "height of a video frame in bytes",
  0
)
DS_CUSTOM_FACTORY_DEFINE_PARAM(
  format,
  string,
  "video format",
  "video format: RGBA/I420/NV12/...",
  "RGBA"
)
DS_CUSTOM_FACTORY_DEFINE_PARAM(
  use-gpu-memory,
  boolean,
  "flag to use GPU memory",
  "flag to use GPU memory",
  false
)
DS_CUSTOM_FACTORY_DEFINE_PARAM(
  use-external-memory,
  boolean,
  "flag to indicate the memory allocation strategy",
  "the plugin will manage the memory outside of the pipeline if set true",
  false
)
DS_CUSTOM_FACTORY_DEFINE_PARAMS_END

DS_CUSTOM_PLUGIN_DEFINE(
    sample_video_feeder,
    "this is a sample data feeder plugin",
    "0.1",
    "Proprietary")

DS_CUSTOM_FACTORY_DEFINE_WITH_PARAMS(
  FACTORY_NAME,
  "sample video data feeder factory",
  "signal",
  "this is a sample video data feeder factory",
  "NVIDIA",
  "need-data/enough-data",
  param_spec,
  DataFeeder,
  FileDataSource
)