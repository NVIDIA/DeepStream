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

# DeepStream Utility Libraries

This directory contains source code for DeepStream utility libraries — shared libraries consumed by GStreamer plugins, sample applications, and service-maker modules.

For build and installation instructions, see [build/BUILD.md](../../build/BUILD.md).

---

## Library Index

### Inference

| Library | Output | Description |
|---|---|---|
| `nvdsinfer` | `libnvds_infer.so` | TensorRT inference engine wrapper. Provides the `NvDsInferContext` API for running TRT engines from plugins and apps. See [README](nvdsinfer/README). |
| `nvdsinferserver` | `libnvds_inferserver.so` | Triton Inference Server client wrapper. Provides the `NvDsInferServerContext` API. See [README](nvdsinferserver/README). |
| `nvdsinfer_customparser` | `libnvdsinfer_customparser.so` | Reference implementation for custom bounding-box and classifier parsers. See [README](nvdsinfer_customparser/README). |

### Stream Multiplexing

| Library | Output | Description |
|---|---|---|
| `nvstreammux` | `libnvdsgst_helper.so` | Stream multiplexer helper library — buffer batching and metadata management for nvstreammux. See [README](nvstreammux/README). |

### Analytics

| Library | Output | Description |
|---|---|---|
| `nvds_analytics` | `libnvds_analytics.so` | Line-crossing, ROI, and direction analytics backend used by the nvdsanalytics plugin. See [README](nvds_analytics/README). |

### Cloud Messaging

| Library | Output | Description |
|---|---|---|
| `nvmsgbroker` | `libnvds_msgbroker.so` | Message broker library — transport abstraction over Kafka, MQTT, Azure IoT Hub, AMQP, Redis. See [README](nvmsgbroker/README). |
| `nvmsgconv` | `libnvds_msgconv.so` | Converts DeepStream object metadata to JSON/Protobuf payloads for cloud messaging. See [README](nvmsgconv/README). |
| `nvmsgconv_mega` | `libnvds_msgconv_mega.so` | Extended message converter for MEGA schema payloads. See [README](nvmsgconv_mega/README). |
| `nvmsgconv_mega2d` | `libnvds_msgconv_mega2d.so` | Message converter for MEGA 2D schema payloads. See [README](nvmsgconv_mega2d/README). |
| `nvds_msgapi` | — | DeepStream Messaging API (DSMI) — protocol adaptor interface and reference implementations (Kafka, MQTT, Azure, AMQP, Redis). See [README](nvds_msgapi/README.md). |

### On-Screen Display

| Library | Output | Description |
|---|---|---|
| `nvll_osd` | `libnvll_osd.so` | Low-level OSD rendering library — draws bounding boxes, text, and shapes on GPU buffers. See [README](nvll_osd/README). |

### REST Server

| Library | Output | Description |
|---|---|---|
| `nvds_rest_server` | `libnvds_rest_server.so` | Embedded REST server library for live pipeline control (source add/remove, config reload). See [README](nvds_rest_server/README). |

### 3D Sensing

| Library | Output | Description |
|---|---|---|
| `ds3d` | `libnvds_3d_*.so` | DS3D utility library for 3D sensor pipelines — data loaders (LiDAR, depth camera), data filters, data bridges, and inference custom libs for 3D detection and V2X fusion. See [README](ds3d/README.md). |

---

## Tooling & Model Setup

These entries are not built libraries — they ship setup scripts and model-prep
documentation consumed alongside the DeepStream runtime.

| Entry | Description |
|---|---|
| `nvds_logger` | `setup_nvds_logger.sh` configures `rsyslog` to capture DeepStream `DSLOG` messages to `/var/log/nvds/`. See [README](nvds_logger/README). |
| `nvmultiobjecttracker/model/tracker_ReID` | Model preparation guide (TensorRT) for deep-learning models used by the multi-object tracker: SAM2 segmentation, TAO ReidentificationNet, TAO BodyPose3DNet. See [README](nvmultiobjecttracker/model/tracker_ReID/README). |
