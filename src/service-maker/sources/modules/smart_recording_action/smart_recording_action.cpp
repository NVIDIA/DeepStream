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

#include <iostream>

#include "plugin.h"
#include "custom_factory.hpp"
#include "common_factory.hpp"
#include "smart_recording_action.hpp"

using namespace deepstream;

#define FACTORY_NAME "smart_recording_action"

DS_CUSTOM_FACTORY_DEFINE_PARAMS_BEGIN(sr_param_spec)
DS_CUSTOM_FACTORY_DEFINE_PARAM(
    proto-lib,
    string,
    "protocol library",
    "the path to the shared library that implements the device/cloud communication protocol",
    "/opt/nvidia/deepstream/deepstream/lib/libnvds_kafka_proto.so")
DS_CUSTOM_FACTORY_DEFINE_PARAM(
    conn-str,
    string,
    "connection string",
    "string for the connection in the format of 'ip;port'",
    "127.0.0.1;9092")
DS_CUSTOM_FACTORY_DEFINE_PARAM(
    proto-config-file,
    path,
    "protocal config file path",
    "path to the config file of the communication protocal",
    "/opt/nvidia/deepstream/deepstream/sources/libs/kafka_protocol_adaptor/cfg_kafka.txt")
DS_CUSTOM_FACTORY_DEFINE_PARAM(
    msgconv-config-file,
    path,
    "message converter config file path",
    "path to the config file of message converter",
    "")
DS_CUSTOM_FACTORY_DEFINE_PARAM(
    topic-list,
    string,
    "topic list",
    "list of topics to subscribe",
    "")
DS_CUSTOM_FACTORY_DEFINE_PARAMS_END

DS_CUSTOM_PLUGIN_DEFINE(
    smart_recording_action,
    "this is a smart recording action plugin",
    "0.1",
    "Proprietary")

DS_CUSTOM_FACTORY_DEFINE_WITH_PARAMS(
    FACTORY_NAME,
    "smart recording signal action factory",
    "signal",
    "this is a smart recording action factory",
    "NVIDIA",
    "start-sr/stop-sr",
    sr_param_spec,
    SignalEmitter,
    SmartRecordingAction)