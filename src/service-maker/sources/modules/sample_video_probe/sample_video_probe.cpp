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
#include "sample_video_probe.hpp"

using namespace deepstream;

DS_CUSTOM_FACTORY_DEFINE_PARAMS_BEGIN(probe_param_spec)
DS_CUSTOM_FACTORY_DEFINE_PARAM(
  font-size,
  integer,
  "font-size",
  "size of the font to show the counter",
  12
)
DS_CUSTOM_FACTORY_DEFINE_PARAMS_END

#define FACTORY_NAME "sample_video_probe"

DS_CUSTOM_PLUGIN_DEFINE(
    sample_video_probe,
    "this is a sample video buffer probe plugin",
    "0.1",
    "Proprietary")

DS_CUSTOM_FACTORY_DEFINE_WITH_PARAMS(
  FACTORY_NAME,
  "sample video buffer probe factory",
  "probe",
  "this is a sample video buffer probe factory to create a object count marker",
  "NVIDIA",
  "",
  probe_param_spec,
  BufferProbe,
  CountMarker
)