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
#include "add_message_meta_probe.hpp"

using namespace std;
using namespace deepstream;

DS_CUSTOM_FACTORY_DEFINE_PARAMS_BEGIN(probe_param_spec)
DS_CUSTOM_FACTORY_DEFINE_PARAM(
    frame-interval,
    integer,
    "frame-interval",
    "frame interval for which to generate message meta",
    1)
DS_CUSTOM_FACTORY_DEFINE_PARAM(
    source-config,
    string,
    "source-config",
    "source config file for which to generate message meta",
    "")
DS_CUSTOM_FACTORY_DEFINE_PARAM(
    label-file,
    string,
    "label-file",
    "label file for which to generate message meta",
    "")
DS_CUSTOM_FACTORY_DEFINE_PARAMS_END

#define FACTORY_NAME "add_message_meta_probe"

DS_CUSTOM_PLUGIN_DEFINE(
    add_message_meta_probe,
    "Custom probe to add NVDS_META_EVENT_MSG data to buffer",
    "0.1",
    "Proprietary")

DS_CUSTOM_FACTORY_DEFINE_WITH_PARAMS(
    FACTORY_NAME,
    "message meta adding custom probe factory",
    "probe",
    "this is a message meta adding custom probe factory",
    "NVIDIA",
    "",
    probe_param_spec,
    BufferProbe,
    MsgMetaGenerator)
