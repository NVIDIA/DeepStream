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

#include "plugin.h"
#include "custom_factory.hpp"
#include "common_factory.hpp"
#include "measure_fps_probe.hpp"

using namespace deepstream;

#define FACTORY_NAME "measure_fps_probe"

DS_CUSTOM_PLUGIN_DEFINE(
    measure_fps_probe,
    "Custom probe to add measure FPS",
    "0.1",
    "Proprietary")

DS_CUSTOM_FACTORY_DEFINE(
  FACTORY_NAME,
  "fps measurement calculating custom probe factory",
  "probe",
  "this is a fps measurement custom probe factory",
  "NVIDIA",
  BufferProbe,
  FPSCounter
)
