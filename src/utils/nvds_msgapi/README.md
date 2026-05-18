<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: LicenseRef-NvidiaProprietary

NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
property and proprietary rights in and to this material, related
documentation and any modifications thereto. Any use, reproduction,
disclosure or distribution of this material and related documentation
without an express license agreement from NVIDIA CORPORATION or
its affiliates is strictly prohibited.
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
