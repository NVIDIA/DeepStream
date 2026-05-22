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