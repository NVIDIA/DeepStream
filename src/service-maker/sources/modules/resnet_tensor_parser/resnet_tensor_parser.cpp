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
