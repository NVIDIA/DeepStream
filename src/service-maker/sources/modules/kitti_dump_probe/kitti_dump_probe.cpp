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
#include "kitti_dump_probe.hpp"

using namespace std;
using namespace deepstream;

#define FACTORY_NAME "kitti_dump_probe"

DS_CUSTOM_FACTORY_DEFINE_PARAMS_BEGIN(probe_param_spec)
DS_CUSTOM_FACTORY_DEFINE_PARAM(
  kitti-dir,
  string,
  "kitti-dir",
  "directory of kitti output",
  "/tmp/kitti"
)
DS_CUSTOM_FACTORY_DEFINE_PARAM(
  tracker-kitti-output,
  boolean,
  "tracker-kitti-output",
  "enable tracker kitti output",
  false
)
DS_CUSTOM_FACTORY_DEFINE_PARAMS_END

DS_CUSTOM_PLUGIN_DEFINE(
    kitti_dump_probe,
    "Custom probe for kitti dump",
    "0.1",
    "Proprietary")

DS_CUSTOM_FACTORY_DEFINE_WITH_PARAMS(
  FACTORY_NAME,
  "kitti dump adding custom probe factory",
  "probe",
  "this is a kitti dumping custom probe factory",
  "NVIDIA",
  "",
  probe_param_spec,
  BufferProbe,
  NvDsKittiDump
)