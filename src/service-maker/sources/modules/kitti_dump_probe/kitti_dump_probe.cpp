/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
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