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

# DeepStream GStreamer Plugins

This directory contains source code for all DeepStream GStreamer plugins. Each plugin is a self-contained GStreamer element that integrates with the DeepStream metadata pipeline.

For build and installation instructions, see [build/BUILD.md](../../build/BUILD.md).

---

## Plugin Index

### Inference

| Plugin | Element Name | Description |
|---|---|---|
| `gst-nvinfer` | `nvinfer` | TensorRT-based inference for object detection, classification, and segmentation. Primary and secondary GIE modes. See [README](gst-nvinfer/README). |
| `gst-nvinferserver` | `nvinferserver` | Triton Inference Server-based inference. Supports TensorRT, ONNX, TF, PyTorch backends. See [README](gst-nvinferserver/README). |
| `gst-nvdspreprocess` | `nvdspreprocess` | Custom pre-processing (resize, normalize, crop) on batched frames before inference. See [README](gst-nvdspreprocess/README). |
| `gst-nvdspostprocess` | `nvdspostprocess` | Custom post-processing on inference output tensors. See [README](gst-nvdspostprocess/README). |

### Stream Input / Multiplexing

| Plugin | Element Name | Description |
|---|---|---|
| `gst-nvurisrcbin` | `nvurisrcbin` | URI-based source bin — decodes and batches a single stream. See [README](gst-nvurisrcbin/README). |
| `gst-nvmultiurisrcbin` | `nvmultiurisrcbin` | Multi-URI source bin — manages multiple streams and sends them to nvstreammux. See [README](gst-nvmultiurisrcbin/README). |
| `gst-nvmultistream2` | `nvstreammux` / `nvstreamdemux` | Batch multiple decoded streams into a single buffer; demux back to individual streams. See [README](gst-nvmultistream2/README). |
| `gst-nvdynamicsrcbin` | `nvdynamicsrcbin` | Dynamic source bin — add and remove video sources at runtime without stopping the pipeline. See [README](gst-nvdynamicsrcbin/README). |
| `gst-nvvideotestsrc` | `nvvideotestsrc` | NVMM-backed test video source for pipeline testing without a real input. See [README](gst-nvvideotestsrc/README). |

### Tracking and Analytics

| Plugin | Element Name | Description |
|---|---|---|
| `gst-nvtracker` | `nvtracker` | Multi-object tracker. Supports NvDCF, DeepSORT, IOU, and custom tracker libraries. See [README](gst-nvtracker/README). |
| `gst-nvdsanalytics` | `nvdsanalytics` | Line-crossing counting, ROI filtering, and direction detection analytics. See [README](gst-nvdsanalytics/README). |
| `gst-nvdewarper` | `nvdewarper` | Dewarps fisheye and 360° camera feeds into perspective or panoramic views. See [README](gst-nvdewarper/README). |
| `gst-nvof` *(via nvdsvideotemplate)* | — | NVIDIA Optical Flow SDK integration for motion estimation. |

### On-Screen Display and Metadata

| Plugin | Element Name | Description |
|---|---|---|
| `gst-nvdsosd` | `nvdsosd` | Renders bounding boxes, labels, and custom OSD metadata onto frames. See [README](gst-nvdsosd/README). |
| `gst-nvdsmetautils` | — | Metadata serialization utilities (SEI, video/audio metadata embedding). See [README](gst-nvdsmetautils/README). |
| `gst-nvdsmetamux` | `nvdsmetamux` | Merges metadata from multiple inference branches into a single stream. See [README](gst-nvdsmetamux/README). |

### Cloud Messaging

| Plugin | Element Name | Description |
|---|---|---|
| `gst-nvmsgbroker` | `nvmsgbroker` | Publishes DeepStream metadata to cloud endpoints (Kafka, MQTT, Azure IoT Hub, AMQP). See [README](gst-nvmsgbroker/README). |
| `gst-nvmsgconv` | `nvmsgconv` | Converts DeepStream object metadata to a payload schema for nvmsgbroker. See [README](gst-nvmsgconv/README). |

### Image and Video Utilities

| Plugin | Element Name | Description |
|---|---|---|
| `gst-nvimage` | `nvimgdec` / `nvimgenc` | GPU-accelerated image encode and decode. See [README](gst-nvimage/README). |
| `gst-nvreplay` | `nvreplay` | Seek and replay support for recorded streams. See [README](gst-nvreplay/README). |

### Extensibility Templates

| Plugin | Element Name | Description |
|---|---|---|
| `gst-nvdsvideotemplate` | `nvdsvideotemplate` | Template plugin for custom video processing; ships with example custom libs. See [README](gst-nvdsvideotemplate/README). |
| `gst-dsexample` | `dsexample` | Reference example plugin demonstrating CPU-based custom processing. See [README](gst-dsexample/README). |
| `gst-dsexample-cuda` | `dsexample` (CUDA) | Reference example plugin demonstrating GPU-based custom processing with CUDA. Requires OpenCV 4.x with CUDA. See [README](gst-dsexample-cuda/README). |

### Networking

| Plugin | Element Name | Description |
|---|---|---|
| `gst-nvdsudp` | `nvdsudp` | UDP-based transport for DeepStream metadata. Requires Rivermax SDK. See [README](gst-nvdsudp/README). |
