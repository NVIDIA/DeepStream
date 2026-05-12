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