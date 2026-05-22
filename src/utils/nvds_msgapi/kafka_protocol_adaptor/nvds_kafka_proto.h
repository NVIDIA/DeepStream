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


#ifndef __NVDS_KAFKA_PROTO_H__
#define __NVDS_KAFKA_PROTO_H__

#define NVDS_MSGAPI_VERSION "2.0"
#define NVDS_MSGAPI_PROTOCOL "KAFKA"

#define CONFIG_GROUP_MSG_BROKER_RDKAFKA_CFG "proto-cfg"
#define CONFIG_GROUP_MSG_BROKER_RDKAFKA_PRODUCER_CFG "producer-proto-cfg"
#define CONFIG_GROUP_MSG_BROKER_RDKAFKA_CONSUMER_CFG "consumer-proto-cfg"
#define CONFIG_GROUP_MSG_BROKER_PARTITION_KEY "partition-key"
#define CONFIG_GROUP_MSG_BROKER_CONSUMER_GROUP "consumer-group-id"
#define CONFIG_GROUP_MSG_BROKER_RDKAFKA_SHARE_CONNECTION "share-connection"
#define DEFAULT_KAFKA_CONSUMER_GROUP "test-consumer-group"
#define DEFAULT_PARTITION_NAME "sensor.id"

#endif
