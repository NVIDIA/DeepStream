<!--
SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# DS3D — DeepStream 3D Sensor Utility Library

DS3D provides the utility framework for building DeepStream pipelines that process 3D sensor data — including LiDAR point clouds, depth cameras, and V2X (vehicle-to-everything) fusion.

Each module below implements one of the DS3D abstract interfaces (`ds3d::dataloader`, `ds3d::datafilter`, `ds3d::datamixer`, `ds3d::datarender`, `ds3d::databridge`) and exchanges data through the `ds3d::datamap` container defined in [`common`](common/README). Apart from `gst`, the modules are loaded at runtime through the `custom_lib_path` / `custom_create_function` keys of a YAML element config rather than being linked directly.

Public headers live in [`includes/ds3d`](../../../includes/ds3d).

---

## Module Index

Each module builds one shared library and has its own `README`.

### Core

| Directory | Output | Description |
|---|---|---|
| [`common`](common/README) | `libnvds_3d_common.so` | Shared runtime for every other DS3D module — the `ds3d::datamap` container, 2D/3D frame wrappers, the abstract data-process interfaces, and the config/YAML helpers. |
| [`glprocess`](glprocess/README) | `libnvds_3d_glprocess.so` | OpenGL ES rendering back-end shared by the data-render modules. Wraps GLFW window management, GLES3 shaders, vertex/texture buffers, CUDA/GL interop, and pango/cairo text rasterization. |
| [`gst`](gst/README) | `libnvdsgst_3d_gst.so` | GStreamer glue layer. Carries a `ds3d::datamap` inside a `GstBuffer` as DS3D meta and wraps DS3D modules into `appsrc`/`appsink` elements. Unlike the others it is linked directly rather than loaded through `custom_lib_path`. |

### Data loaders (`ds3d::dataloader`)

| Directory | Output | Description |
|---|---|---|
| [`dataloader/realsense`](dataloader/realsense/README) | `libnvds_3d_dataloader_realsense.so` | Captures aligned depth and colour frames from an Intel RealSense camera, together with the camera intrinsics. |
| [`dataloader/depthsource`](dataloader/depthsource/README) | `libnvds_3d_depth_datasource.so` | Reads depth and colour image frames from files — the offline counterpart of the RealSense loader. |
| [`dataloader/lidarsource`](dataloader/lidarsource/README) | `libnvds_lidarfileread.so` | Reads binary LiDAR point-cloud frames from files. Reference for implementing a custom sensor or file capture loader. |

### Data filters (`ds3d::datafilter`)

| Directory | Output | Description |
|---|---|---|
| [`datafilter/depth2point`](datafilter/depth2point/README) | `libnvds_3d_depth2point_datafilter.so` | Converts a depth frame into a 3D point cloud using the camera intrinsics and depth scale carried in the same datamap. |
| [`datafilter/lidar_preprocess`](datafilter/lidar_preprocess/README) | `libnvds_3d_lidar_preprocess_datafilter.so` | Point-cloud preprocessing ahead of inference; also the worked example for producing a custom datamap. |
| [`datafilter/lidarinference`](datafilter/lidarinference/README) | `libnvds_tritoninferfilter.so` | Runs Triton inference (through nvdsinferserver) on the point-cloud data and writes the resulting 3D detections back into the datamap. |
| [`datafilter/alignment`](datafilter/alignment/README) | `libnvds_3d_alignment_datafilter.so` | Applies the configured extrinsic transforms so LiDAR and camera sensors share one coordinate system — the calibration step of a fusion pipeline. |
| [`datafilter/sensorfusion_cluster`](datafilter/sensorfusion_cluster/README) | `libnvds_3d_sensor_fusion_cluster.so` | Clusters per-sensor 3D detections into a single fused object list, merging duplicates across sensors. |

### Data mixer (`ds3d::datamixer`)

| Directory | Output | Description |
|---|---|---|
| [`datamixer/multisensor`](datamixer/multisensor/README) | `libnvds_3d_multisensor_mixer.so` | Timestamp-matches frames arriving from multiple cameras and LiDARs and emits a single combined datamap. |

### Data renders (`ds3d::datarender`)

| Directory | Output | Description |
|---|---|---|
| [`datarender/gl3d`](datarender/gl3d/README) | `libnvds_3d_gl_datarender.so` | Displays datamap contents in an OpenGL ES window; delegates the drawing to `libnvds_3d_glprocess.so`. |
| [`datarender/gl3d_ensemble`](datarender/gl3d_ensemble/README) | `libnvds_3d_gles_ensemble_render.so` | Draws several views into one window — for example multi-camera video tiles next to a LiDAR bird's-eye view. |
| [`datarender/lidarsink`](datarender/lidarsink/README) | `libnvds_lidarfilewrite.so` | Writes the point cloud and 3D detection results to files instead of drawing them — the headless counterpart of the GL renders. |

### Data bridge (`ds3d::databridge`)

| Directory | Output | Description |
|---|---|---|
| [`databridge/video`](databridge/video/README) | `libnvds_3d_video_databridge.so` | Connects a standard DeepStream 2D video pipeline to a DS3D pipeline by exposing the incoming batch's `NvBufSurface` and `NvDsBatchMeta` inside a datamap. |

### Inference custom libraries

Loaded by the DS3D inference filters through nvdsinferserver. See [`inference_custom_lib/README`](inference_custom_lib/README).

| Directory | Output | Description |
|---|---|---|
| [`inference_custom_lib/ds3d_lidar_detection_postprocess`](inference_custom_lib/ds3d_lidar_detection_postprocess/README) | `libnvds_3d_infer_postprocess_lidar_detection.so` | Parses Triton output tensors from a LiDAR 3D detection model into `ds3d::Lidar3DBbox` objects. |
| [`inference_custom_lib/ds3d_v2x_infer_custom_preprocess`](inference_custom_lib/ds3d_v2x_infer_custom_preprocess/README) | `libnvds_3d_v2x_infer_custom_preprocess.so` | Feeds LiDAR and image data to the Triton inference server for V2X fusion. |
| [`inference_custom_lib/ds3d_v2x_infer_custom_postprocess`](inference_custom_lib/ds3d_v2x_infer_custom_postprocess/README) | `libnvds_3d_v2x_infer_custom_postprocess.so` | Parses V2X fusion Triton output tensors into `ds3d::Lidar3DBbox` objects. |

### Build support

Neither of these ships a runtime library.

| Directory | Description |
|---|---|
| [`stub/realsense`](stub/realsense/README) | Builds a link-time stub of `librealsense2.so` from `librealsense2.export`, so the RealSense dataloader links without the Intel RealSense SDK installed. At run time the real library is picked up from the system; the stub is never installed. |
| [`3rdparty/realsense2`](3rdparty/realsense2/README) | Vendored `librealsense2` API headers (from [librealsense v2.48.0](https://github.com/IntelRealSense/librealsense/tree/v2.48.0/include)) used to build the RealSense dataloader. |

---

## Directory Layout

```
ds3d/
├── common/                 # shared datamap, frame types, and process interfaces
│   ├── helper/             #   buffer, CUDA, and safe-pointer helpers
│   ├── hpp/                #   C++ wrappers (datamap, yaml_config, profiling, …)
│   └── impl/               #   interface implementations
├── glprocess/              # shared GLES render back-end (window, shaders, CUDA/GL)
├── stub/realsense/         # link-time RealSense stub (not installed)
├── dataloader/
│   ├── realsense/          # Intel RealSense depth + colour capture
│   ├── depthsource/        # depth + colour frames from files
│   └── lidarsource/        # binary LiDAR point-cloud file reader
├── datafilter/
│   ├── depth2point/        # depth frame to 3D point cloud
│   ├── lidar_preprocess/   # point-cloud preprocess ahead of inference
│   ├── lidarinference/     # Triton inference on LiDAR, writes 3D detections
│   ├── alignment/          # LiDAR/camera extrinsic alignment
│   └── sensorfusion_cluster/  # cluster per-sensor detections into one list
├── datamixer/multisensor/  # timestamp-match cameras and LiDARs into one datamap
├── datarender/
│   ├── gl3d/               # OpenGL ES window for a single datamap view
│   ├── gl3d_ensemble/      # multi-view GLES window (tiles + bird's-eye)
│   └── lidarsink/          # write point cloud and detections to files
├── databridge/video/       # expose NvBufSurface / NvDsBatchMeta as a datamap
├── inference_custom_lib/
│   ├── ds3d_lidar_detection_postprocess/  # Triton tensors → Lidar3DBbox
│   ├── ds3d_v2x_infer_custom_preprocess/  # feed LiDAR + image to Triton for V2X
│   └── ds3d_v2x_infer_custom_postprocess/ # V2X Triton tensors → Lidar3DBbox
├── gst/                    # GStreamer glue (datamap-in-GstBuffer, appsrc/appsink)
└── 3rdparty/realsense2/    # vendored librealsense2 API headers
```

---

## Building

The top-level [`Makefile`](Makefile) walks every subdirectory in dependency order — `common`, `glprocess`, and the RealSense stub first, since everything else links against them.

```bash
export CUDA_VER=13.4        # x86. Jetson, SBSA and IGX Thor use CUDA_VER=13.2.
make && sudo -E make install
```

To rebuild a single module, run `make` in its own directory after `common` (and `glprocess`, for the render modules) has been installed. Per-module prerequisites are listed in each subdirectory's `README`. For the full build flow see [build/BUILD.md](../../../build/BUILD.md).

---

## Consumers

**GStreamer plugins** — these link `libnvdsgst_3d_gst.so` and `libnvds_3d_common.so`:

| Plugin | Element |
|---|---|
| [`gst-nvds3dbridge`](../../gst-plugins/gst-nvds3dbridge/README) | `nvds3dbridge` |
| [`gst-nvds3dfilter`](../../gst-plugins/gst-nvds3dfilter/README) | `nvds3dfilter` |
| [`gst-nvds3dmixer`](../../gst-plugins/gst-nvds3dmixer/README) | `nvds3dmixer` |
| [`gst-nvmsgconv`](../../gst-plugins/gst-nvmsgconv/README) | `nvmsgconv` |

**Sample applications**:

| App | Description |
|---|---|
| [`deepstream-3d-lidar-sensor-fusion`](../../apps/sample_apps/deepstream-3d-lidar-sensor-fusion/README) | LiDAR + camera multi-modal sensor fusion pipeline |
| [`deepstream-3d-depth-camera`](../../apps/sample_apps/deepstream-3d-depth-camera/README) | RGB-D depth camera pipeline — depth capture, point-cloud conversion, and GL render |
| [`deepstream-lidar-inference-app`](../../apps/sample_apps/deepstream-lidar-inference-app/README) | End-to-end LiDAR 3D inference and rendering with PointPillars models, configured from a single YAML file |
| [`deepstream-can-orientation-app`](../../apps/sample_apps/deepstream-can-orientation-app/README) | Uses the DS3D common and gst layers (profiling, YAML config, DS3D meta) for a can-orientation inspection pipeline |

---

## Documentation

See the "DeepStream 3D Custom Manual" in the DeepStream SDK documentation: https://docs.nvidia.com/deepstream/9.2/text/DS_3D_Custom_Manual
