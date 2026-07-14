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

# DeepStream Custom Tiler Configuration Sample

## Introduction
This sample demonstrates the usage of "custom-tile-config" of nvmultistreamtiler to customize the tiling positions and sizes of multiple videos within the display window. The rectangle display areas of every video can be configured by the "custom-tile-config" property of nvmultistreamtiler for CustomTileConfig struct.

## Prerequisites
The sample works with DeepStream 9.1 GA or above version. Please follow [DeepStream SDK installation instruction](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Installation.html) or use [DeepStream docker container](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_docker_containers.html) to prepare the DeepStream environment.

## Running the Application
Download the source code and build

```
make
```

Run the sample with four video files and generate mp4 video for the output

```bash
./deepstream-custom-tile-config -i file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4 \
                                   file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4 \
                                   file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4 \
                                   file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4 \
                                   --no-display
```

Run the sample with four video files and dislay on the screen

```bash
./deepstream-custom-tile-config -i file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4 \
                                   file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4 \
                                   file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4 \
                                   file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4
```

## Known Issue
The video blending is not supported now. If there are overlapping parts of the videos, the overlapping parts will flicker. 
