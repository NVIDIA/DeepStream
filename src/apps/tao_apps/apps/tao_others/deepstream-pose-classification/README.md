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

# deepstream-pose-classification

## Introduction
The project contains pose classification application built using  Deepstream SDK.

## Prerequisites:
DeepStream SDK not less than 6.2 installed which is available at  http://developer.nvidia.com/deepstream-sdk

Please follow instructions in the `src/apps/sample_apps/deepstream-app/README` on how to install the prequisites for building Deepstream SDK apps.

## Installation
1. Install libeigen3-dev
```bash
    $ sudo apt install libeigen3-dev
    $ cd /usr/include
    $ sudo ln -sf eigen3/Eigen Eigen
```
2. prepare the models  
  [PeopleNet](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/peoplenet)  
  [BodyPose3DNet](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/bodypose3dnet)  
  [poseclassificationnet](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/poseclassificationnet)  
  [tracker model](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/reidentificationnet)   
## Build the applications
```bash
make
```

## Run the applications
### `deepstream-pose-classification-app`
The command line options of this application are listed below:
```bash
$ ./deepstream-pose-classification-app ../../../configs/app/deepstream_pose_classification_config.yaml
```

## Tips

1. if FPS is getting smaller and smaller when there are many objects in the sources, please increase scaling-buf-pool-size and tensor-buf-pool-size in config_preprocess_bodypose_classification.txt.
