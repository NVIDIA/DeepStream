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

## Prepare Triton Server For Native Inferencing
As mentioned in the README, the DeepStream applications should work as Triton client with Triton Server running natively for cAPIs. So the [Triton Inference Server libraries](https://github.com/triton-inference-server/client) should be installed in the machine. An easier way is to run the DeepStream application in the [DeepStream Triton container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/deepstream). 

Running DeepStream Triton container, takes the DeepStream 9.1 container as the example:
```
    docker run --gpus all -it  --ipc=host --rm --privileged -v /tmp/.X11-unix:/tmp/.X11-unix  -v $(pwd):/samples   -e DISPLAY=$DISPLAY -w /samples nvcr.io/nvidia/deepstream:9.1-triton-devel
```
Inside the container, prepare model engines for Triton server:
```
    ./build_triton_engine.sh

```

Then the DeepStream sample application can be build and run inside this container.
