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