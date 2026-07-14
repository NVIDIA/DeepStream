<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# nvds_msgapi — DeepStream Messaging API

`nvds_msgapi` defines the DeepStream Messaging Interface (DSMI) — a protocol-agnostic API for sending messages from DeepStream pipelines to cloud or on-premises message brokers. The `nvmsgbroker` GStreamer plugin loads a protocol adaptor at runtime via this interface.

---

## Protocol Adaptors

Each subdirectory is a self-contained shared library that implements the DSMI interface for a specific transport:

| Adaptor | Output Library | Transport |
|---|---|---|
| `kafka_protocol_adaptor` | `libnvds_kafka_proto.so` | Apache Kafka |
| `mqtt_protocol_adaptor` | `libnvds_mqtt_proto.so` | MQTT (via Mosquitto) |
| `azure_protocol_adaptor` | `libnvds_azure_proto.so` | Azure IoT Hub (device/module client) |
| `amqp_protocol_adaptor` | `libnvds_amqp_proto.so` | AMQP (via RabbitMQ) |
| `redis_protocol_adaptor` | `libnvds_redis_proto.so` | Redis Pub/Sub |
| `common_src` | — | Shared utility code used across adaptors |

Refer to the `README` in each adaptor subdirectory for prerequisite packages and configuration details.

---

## Usage

The `nvmsgbroker` plugin selects an adaptor at runtime via the `config-file` property which specifies the adaptor `.so` path and broker connection parameters.

See the `deepstream-test4` sample app for an end-to-end example of publishing inference results to a message broker.

For build and installation, see [build/BUILD.md](../../../build/BUILD.md).
