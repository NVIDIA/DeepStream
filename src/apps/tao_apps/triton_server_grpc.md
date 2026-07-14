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

# Triton Server
## [Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server) Bring Up

DeepStream applications can work as Triton Inference client. So the corresponding Triton Inference Server should be started before the Triton client start to work.

An immediate way to start a corresponding Triton Server is to use Triton containers provided in [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver). Since every DeepStream version has its corresponding Triton Server version, so the reliable way is to use the [DeepStream Triton container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/deepstream).

* The Triton Server can be started in the same machine which the DeepStream application works in, please make sure the Triton Server is started in a new terminal.

* The Triton Server can be started in another machine as the server which is coonected to the machine for DeepStream applications through ehternet. 

## Prepare Triton Server For gRPC Connection
The following steps take the DeepStream 9.1 GA as an example, if you use other DeepStream versions, the corresponding DeepStream Triton [image](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/deepstream) can be used.

To start Triton Server with DeepStream Triton container, the docker should be run in a new terminal and the following commands should be run in the same path as the deepstream_app_tao_configs codes are downloaded:
* Start the Triton Inference Server with DeepStream Triton docker under src/apps/tao_apps/ directory
```
    //start Triton docker, 10001:8001 is used to map docker container's 8000 port to host's 10000 port, these ports can be changed.
    docker run --gpus all -it  --ipc=host --rm --privileged -v /tmp/.X11-unix:/tmp/.X11-unix  -p 10000:8000 -p 10001:8001 -p 10002:8002  -v $(pwd):/samples   -e DISPLAY=$DISPLAY -w /samples nvcr.io/nvidia/deepstream:9.1-triton-devel
```

For DGX Spark platform, please refer to [DGX Spark Triton docker](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Installation.html#run-dgx-spark-docker) to run Triton docker on DGX Spark. 

Then the model engines should be generated in the server:

```
    ./build_triton_engine.sh
```

If the server is running in the same machine as the DeepStream application, the following command can be used directly. If it is not, please set the gRPC url as the IP address of the server machine in all the configuration files in deepstream_app_tao_configs/triton-grpc:

The gRPC url setting looks like:
```
grpc {
        url: "192.168.0.51:10001"
    }
```

Then the Triton Server service can be started with the following command:
```
    tritonserver --model-repository=/samples/models --strict-model-config=false --grpc-infer-allocation-pool-size=16 --log-verbose=1 --exit-on-error=false

```

Note: The "--allow-client-shm=true" option is needed for the `tritonserver` command to work on DGX Spark platform.

The DeepStream sample application should run in another terminal with the Triton Inference client libraries installed. It is recommend to run the application in the DeepStream Triton container, please refer to [triton_server.md](triton_server.md) for how to start a DeepStream Triton container.
