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

# DeepStream Parallel Inference

DeepStream Parallel Inference is one application which run inference in parallel. tee will send batched buffer to different models.  nvdsmetamux will mux batch meta from different models which run in parallel. User can configure the pipeline in application configuration yaml file. User can select source ids which need inference in preprocess configuration file. User can select source ids which need mux in nvdsmetamux configuration file.

## Prerequisites
DeepStream SDK 9.1 GA or above


## Getting Started:
To get started, please follow these steps.
1. Install [DeepStream](https://developer.nvidia.com/deepstream-sdk) on your platform, verify it is working by running deepstream-app.
2. Install triton client video template libraries.
3. Run the program

 ```
  $ cd DeepStream/src/apps/reference_apps/deepstream_parallel_inference_app/tritonclient/sample/apps/deepstream-parallel-infer/
  $ sudo ./deepstream-parallel-infer -c <config.yml>
```
Run parallel inference app with default configure file: $ ./deepstream-parallel-infer -c ../../configs/apps/bodypose_yolo/source4_1080p_dec_parallel_infer.yml

For any issues or questions, please feel free to make a new post on the [DeepStreamSDK forums](https://forums.developer.nvidia.com/c/accelerated-computing/intelligent-video-analytics/deepstream-sdk/).

#### Known Issues

On **DGX Spark** (NVIDIA GB10, compute capability 12.1 / sm_121), the ONNX Runtime (1.23.0) bundled in the `nvcr.io/nvidia/deepstream:9.1-triton-sbsa-dgx-spark` container was not compatible with sm_121.
You can apply the following workaround to the affected **bodypose2d** model:
1.Run inference through the nvinfer plugin
2.If the nvinferserver plugin must be used, configure the model to use the TensorRT backend (platform: "tensorrt_plan" in config.pbtxt) instead of the ONNX Runtime backend
