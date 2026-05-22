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

#include "plugin.h"
#include "custom_factory.hpp"
#include "common_factory.hpp"
#include "resnet_tensor_parser.hpp"

using namespace deepstream;

#define FACTORY_NAME "resnet_tensor_parser"

DS_CUSTOM_FACTORY_DEFINE_PARAMS_BEGIN(probe_param_spec)
DS_CUSTOM_FACTORY_DEFINE_PARAM(
  network-width,
  integer,
  "network-width",
  "width of the neural network",
  960
)
DS_CUSTOM_FACTORY_DEFINE_PARAM(
  network-height,
  integer,
  "network-height",
  "Height of the neural network",
  544
)
DS_CUSTOM_FACTORY_DEFINE_PARAM(
  stream-width,
  integer,
  "stream-width",
  "width of the streammux output",
  960
)
DS_CUSTOM_FACTORY_DEFINE_PARAM(
  stream-height,
  integer,
  "stream-height",
  "Height of the streammux output",
  544
)
DS_CUSTOM_FACTORY_DEFINE_PARAM(
  num-classes,
  integer,
  "num-classes",
  "number of detected classes",
  4
)
DS_CUSTOM_FACTORY_DEFINE_PARAMS_END

DS_CUSTOM_PLUGIN_DEFINE(
    resnet_tensor_parser,
    "Custom tensor parser for resnet object detector",
    "0.1",
    "Proprietary")

DS_CUSTOM_FACTORY_DEFINE_WITH_PARAMS(
  FACTORY_NAME,
  "resnet tensor parser probe factory",
  "probe",
  "this is a resnet tensor parser probe factory",
  "NVIDIA",
  "",
  probe_param_spec,
  BufferProbe,
  TensorMetaParser
)
