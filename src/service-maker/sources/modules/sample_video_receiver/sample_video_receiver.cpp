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

#include "sample_video_receiver.hpp"
#include "custom_factory.hpp"
#include "common_factory.hpp"
#include "plugin.h"

using namespace deepstream;

#define FACTORY_NAME "sample_video_receiver"

DS_CUSTOM_PLUGIN_DEFINE(
    sample_video_receiver,
    "this is a sample data receiver plugin",
    "0.1",
    "Proprietary")

DS_CUSTOM_FACTORY_DEFINE_WITH_SIGNALS(
  FACTORY_NAME,
  "sample video data receiver factory",
  "signal",
  "this is a sample video data receiver factory to create a data receiver to count objects",
  "NVIDIA",
  "new-sample",
  DataReceiver,
  ObjectCounter
)