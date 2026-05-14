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

# DeepStream Sample Applications

This directory contains GStreamer-based DeepStream sample applications. Each app demonstrates a specific pipeline pattern or feature of the DeepStream SDK.

For build and installation instructions, see the repo root [BUILD.md](../../../BUILD.md).

---

## Running Apps

After `build.sh`, binaries are installed to `/opt/nvidia/deepstream/deepstream-9.0/bin/`.

Apps must be run from their **source directory** so relative config and model paths resolve correctly:

```bash
cd src/apps/sample_apps/<app-name>
/opt/nvidia/deepstream/deepstream-9.0/bin/<app-binary> [options]
```

Clear the GStreamer plugin cache after a fresh install:

```bash
rm -rf ~/.cache/gstreamer-1.0/
```

Refer to the `README` in each app's directory for app-specific prerequisites, run commands, and config file descriptions.

---

## Application List

### Core Tutorial Apps

| App | Binary | Description |
|---|---|---|
| `deepstream-test1` | `deepstream-test1-app` | Single H.264 stream decode → infer → OSD → display. Simplest pipeline. |
| `deepstream-test2` | `deepstream-test2-app` | Adds secondary GIE classifiers and tracker to test1. |
| `deepstream-test3` | `deepstream-test3-app` | Multiple input streams using uridecodebin. |
| `deepstream-test4` | `deepstream-test4-app` | Sends inference metadata to cloud via Kafka/MQTT message broker. |
| `deepstream-test5` | `deepstream-test5-app` | Fully configurable multi-stream app with RTSP output. |

### Inference and Metadata

| App | Binary | Description |
|---|---|---|
| `deepstream-infer-tensor-meta-test` | `deepstream-infer-tensor-meta-test` | Access raw tensor output from nvinfer using metadata APIs. |
| `deepstream-preprocess-test` | `deepstream-preprocess-test` | Custom preprocessing with `nvdspreprocess` before inference. |
| `deepstream-gst-metadata-test` | `deepstream-gst-metadata-test` | Attach and read custom GStreamer metadata in a pipeline. |
| `deepstream-user-metadata-test` | `deepstream-user-metadata-test` | Attach custom user metadata using NvDsUserMeta APIs. |
| `deepstream-image-meta-test` | `deepstream-image-meta-test` | Attach and access per-frame image metadata. |
| `deepstream-transfer-learning-app` | `deepstream-transfer-learning-app` | Run custom TensorRT engines trained with Transfer Learning Toolkit. |

### Video Sources and Demuxing

| App | Binary | Description |
|---|---|---|
| `deepstream-appsrc-test` | `deepstream-appsrc-test` | Feed raw buffers into the pipeline using GStreamer appsrc. |
| `deepstream-appsrc-cuda-test` | `deepstream-appsrc-cuda-test` | Feed CUDA-allocated buffers via appsrc. |
| `deepstream-demuxer-static` | `deepstream-demuxer-static` | Demux batched stream to individual sinks (static configuration). |
| `deepstream-demuxer-dynamic` | `deepstream-demuxer-dynamic` | Demux batched stream with runtime source add/remove. |
| `deepstream-dynamicsrcbin-test` | `deepstream-dynamicsrcbin-test` | Dynamic source bin — add and remove sources at runtime. |
| `deepstream-image-decode-test` | `deepstream-image-decode-test` | Decode and process a directory of images. |
| `deepstream-testsr` | `deepstream-testsr-app` | Smart recording — record video segments on trigger. |

### Analytics and Tracking

| App | Binary | Description |
|---|---|---|
| `deepstream-nvdsanalytics-test` | `deepstream-nvdsanalytics-test` | Line-crossing, ROI, and direction analytics using nvdsanalytics. |
| `deepstream-nvof-test` | `deepstream-nvof-test` | NVIDIA Optical Flow SDK integration. |
| `deepstream-dewarper-test` | `deepstream-dewarper-test` | Fisheye / 360° dewarping with nvdewarper. |
| `deepstream-multigpu-nvlink-test` | `deepstream-multigpu-nvlink-test` | Multi-GPU pipeline using NVLink for buffer sharing. **x86 dGPU only** — not supported on Jetson. |

### 3D Sensing

| App | Binary | Description |
|---|---|---|
| `deepstream-3d-action-recognition` | `deepstream-3d-action-recognition-app` | 3D action recognition from video streams. |
| `deepstream-3d-depth-camera` | `deepstream-3d-depth-camera-app` | Depth camera (RGB-D) integration pipeline. |
| `deepstream-3d-lidar-sensor-fusion` | `deepstream-3d-lidar-sensor-fusion-app` | LiDAR point-cloud sensor fusion pipeline. |
| `deepstream-lidar-inference-app` | `deepstream-lidar-inference-app` | LiDAR-based object detection inference. |
| `deepstream-can-orientation-app` | `deepstream-can-orientation-app` | CAN bus + VPI-based vehicle orientation detection. |

### Networking and Server

| App | Binary | Description |
|---|---|---|
| `deepstream-server` | `deepstream-server` | REST server with live pipeline control (source add/remove, config reload). |
| `deepstream-nmos` | `deepstream-nmos-app` | NMOS IS-04/IS-05 media networking and device discovery. |
| `deepstream-ipc-test` | `deepstream-ipc-test` | Share video buffers between pipelines via IPC. **Jetson (aarch64) only** — relies on NVDEC engine sharing not available on x86 or SBSA. |

### OpenCV Integration

| App | Binary | Description |
|---|---|---|
| `deepstream-opencv-test` | `deepstream-opencv-test` | Access frame data in CPU memory for OpenCV processing. |

### Triton Inference Server

These samples do not produce a standalone binary. Each builds an `nvinferserver`
custom-process library (`.so`) and is driven by the deb-installed `deepstream-app`
with the included config file. See each sample's `README` for the prerequisite
Triton container setup and model download steps.

| Sample | Library | Description |
|---|---|---|
| `TritonBackendEnsemble` | `libnvdstriton_custom_impl_ensemble.so` | Triton ensemble model used as SGIE plus a custom Triton C++ backend that reads DeepStream stream IDs. |
| `TritonOnnxYolo` | `libnvdstriton_custom_impl_yolo.so` | DS-Triton custom-lib for ONNX YoloV3 with dynamic-sized output tensors and multi-input postprocessing. |

Run flow:

```bash
cd src/apps/sample_apps/<TritonSample>
deepstream-app -c <included-config>.txt
```
