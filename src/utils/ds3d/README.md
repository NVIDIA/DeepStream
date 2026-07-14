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

# DS3D — DeepStream 3D Sensor Utility Library

DS3D provides the utility framework for building DeepStream pipelines that process 3D sensor data — including LiDAR point clouds, depth cameras, and V2X (vehicle-to-everything) fusion.

It is consumed by the `gst-nvds3dbridge` and `gst-nvds3dfilter` GStreamer plugins and by the `deepstream-3d-*` family of sample apps.

---

## Directory Layout

```
ds3d/
├── common/               # shared DS3D data types, buffer structures, and API headers
├── dataloader/
│   └── lidarsource/      # LiDAR point-cloud data loader (reads binary LiDAR frames)
├── datafilter/           # data filter libraries (preprocessing for point-cloud pipelines)
├── databridge/           # bridges DS3D buffers to/from standard DeepStream NvBufSurface
├── gst/                  # GStreamer DS3D bridge and filter plugin sources
└── inference_custom_lib/ # custom pre/post-processing libs for 3D inference
    ├── ds3d_lidar_detection_postprocess/    # LiDAR detection output parser
    ├── ds3d_v2x_infer_custom_preprocess/    # V2X fusion pre-processing
    └── ds3d_v2x_infer_custom_postprocess/  # V2X fusion post-processing
```

---

## Related Sample Apps

| App | Description |
|---|---|
| `deepstream-3d-lidar-sensor-fusion` | LiDAR + camera sensor fusion pipeline |
| `deepstream-3d-depth-camera` | RGB-D depth camera pipeline |
| `deepstream-3d-action-recognition` | 3D action recognition from video |
| `deepstream-lidar-inference-app` | Standalone LiDAR object detection |

For build and installation, see [build/BUILD.md](../../../build/BUILD.md).
