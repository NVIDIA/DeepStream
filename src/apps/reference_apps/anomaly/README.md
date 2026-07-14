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

# ANOMALY DETECTION REFERENCE APP USING DEEPSTREAMSDK 9.1

## Introduction
The project contains anomaly detection application and auxiliary plug-ins to show the
capability of Deepstream SDK.

## Prequisites:
DeepStream SDK installed which is available at  http://developer.nvidia.com/deepstream-sdk
Please follow instructions in the `apps/sample_apps/deepstream-app/README` on how
to install the prequisites for Deepstream SDK apps.

## Getting Started
- Edit the `dsanomaly_pgie_config.txt` or `dsanomaly_pgie_nvinferserver_config.txt` according to the location of the models to be used


## Compilation Steps for dsdirection plugin
The dsdirection plugin can be built and installed separatedly.
```
  $ cd plugins/gst-dsdirection/
  $ sudo make && sudo make install
```

1. Test direction calculation on one video input, on dGPU, run following commands
```
cd /opt/nvidia/deepstream/deepstream/
gst-launch-1.0 filesrc location=samples/streams/sample_1080p_h264.mp4 ! qtdemux ! h264parse ! nvv4l2decoder ! m.sink_0 \
nvstreammux name=m batch-size=1 width=1920 height=1080 ! nvinfer config-file-path=samples/configs/deepstream-app/config_infer_primary.txt  \
! nvof ! tee name=t ! queue ! nvofvisual ! nvmultistreamtiler width=1920 height=1080 !  nveglglessink t. ! queue ! dsdirection ! \
nvmultistreamtiler width=1920 height=1080 ! nvvideoconvert ! nvdsosd ! nveglglessink
```
2. Test direction calculation on one video input, on Jetson, run following commands
```
cd /opt/nvidia/deepstream/deepstream/
gst-launch-1.0 filesrc location=samples/streams/sample_1080p_h264.mp4 ! qtdemux ! h264parse ! nvv4l2decoder ! m.sink_0 \
nvstreammux name=m batch-size=1 width=1280 height=720 ! nvinfer config-file-path=samples/configs/deepstream-app/config_infer_primary.txt  \
! nvof ! tee name=t ! queue ! nvofvisual ! nvmultistreamtiler width=1920 height=1080 ! nv3dsink sync=0 t. ! queue ! dsdirection ! \
nvmultistreamtiler width=1920 height=1080 ! nvvideoconvert ! nvdsosd ! nv3dsink sync=0
```

3. Test direction calculation using optical flow on two video inputs on dGPU, run following commands
```
cd /opt/nvidia/deepstream/deepstream/
gst-launch-1.0 filesrc location=samples/streams/sample_1080p_h264.mp4 ! qtdemux ! h264parse ! nvv4l2decoder ! m.sink_0 \
nvstreammux name=m batch-size=2 width=1920 height=1080 ! nvinfer config-file-path=samples/configs/deepstream-app/config_infer_primary.txt ! \
nvof ! tee name=t ! queue ! nvofvisual ! nvmultistreamtiler width=1920 height=540 !  nveglglessink t. ! queue ! dsdirection ! \
nvmultistreamtiler width=1920 height=540 ! nvvideoconvert ! nvdsosd ! nveglglessink filesrc location=samples/streams/sample_1080p_h264.mp4 ! \
qtdemux ! h264parse ! nvv4l2decoder ! m.sink_1  --gst-debug=3

```
Anomaly detection app pipeline:
![DS Anomaly Detection Pipeline](.dsdirection_pipeline.png)

## Compilation Steps and Execution:
```
 $ cd src/apps/reference_apps/anomaly/
 $ cd apps/deepstream-anomaly-detection-test/
 $ Set CUDA_VER in the MakeFile as per platform.
     CUDA_VER=13.2
 $ sudo make

 $ ./deepstream-anomaly-detection-app <uri1> [uri2] ... [uriN]
   Ex.: ./deepstream-anomaly-detection-app file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4

Use option "-t inferserver" to select nvinferserver as the inference plugin
 $ ./deepstream-anomaly-detection-app -t inferserver <uri1> [uri2] ... [uriN]
   Ex.: ./deepstream-anomaly-detection-app -t inferserver file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4
```
  The result should be like below:
  ![DS Anomaly Detection Screenshot](.opticalflow.png)

## NOTE:
- Minimum supported resolution: DGPU - 160 x 64, Jetson - 256 x 96
- Due to an issue in nvofvisual plugin, when using nvofvisual along with nvof
  plugin, the width of input to nvof should be multiple of 32 on DGPU and multiple
  of 256 on Jetson. This will be fixed in the nvofvisual plugin in the next DeepStream
  release.
