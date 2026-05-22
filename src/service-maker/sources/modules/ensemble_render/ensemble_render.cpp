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

#include "ensemble_render.hpp"
#include "custom_factory.hpp"
#include "common_factory.hpp"
#include "plugin.h"

using namespace deepstream;

#define FACTORY_NAME "ensemble_render"

DS_CUSTOM_FACTORY_DEFINE_PARAMS_BEGIN(param_spec)
DS_CUSTOM_FACTORY_DEFINE_PARAM(
  config-path,
  path,
  "file path for data",
  "the file contains the data for feeding the appsrc",
  ""
)

DS_CUSTOM_FACTORY_DEFINE_PARAMS_END

DS_CUSTOM_PLUGIN_DEFINE(
    ensemble_render,
    "this is a data receiver plugin for rendering",
    "0.1",
    "Proprietary")

DS_CUSTOM_FACTORY_DEFINE_WITH_PARAMS(
  FACTORY_NAME,
  "sample data receiver factory",
  "signal",
  "this is a data receiver factory to render pipeline results",
  "NVIDIA",
  "new-sample",
  param_spec,
  DataReceiver,
  EnsembleRender
)
