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
#include "measure_fps_probe.hpp"

using namespace deepstream;

DS_CUSTOM_FACTORY_DEFINE_PARAMS_BEGIN(probe_param_spec)
DS_CUSTOM_FACTORY_DEFINE_PARAM(
  interval,
  integer,
  "interval",
  "FPS measurement printing interval in seconds (unsigned integer, must be > 0; default 5)",
  5
)
DS_CUSTOM_FACTORY_DEFINE_PARAMS_END

#define FACTORY_NAME "measure_fps_probe"

DS_CUSTOM_PLUGIN_DEFINE(
    measure_fps_probe,
    "Custom probe to add measure FPS",
    "0.1",
    "Proprietary")

DS_CUSTOM_FACTORY_DEFINE_WITH_PARAMS(
  FACTORY_NAME,
  "fps measurement calculating custom probe factory",
  "probe",
  "this is a fps measurement custom probe factory",
  "NVIDIA",
  "",
  probe_param_spec,
  BufferProbe,
  FPSCounter
)
