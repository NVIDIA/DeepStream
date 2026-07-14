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

# DeepStream Source Tree

This directory contains the source for all DeepStream 9.1 components shipped from this repository, along with platform-specific notes, external-prerequisite caveats, and deprecation status.

For build and install instructions, see [build/BUILD.md](../build/BUILD.md).

---

## Components

### Sample Applications

Located in `apps/sample_apps/`. These apps demonstrate core DeepStream pipeline patterns using GStreamer APIs directly.

See [`apps/sample_apps/README.md`](apps/sample_apps/README.md) for the full list with descriptions and run commands. Each app has its own `README` at `apps/sample_apps/<app>/README`.

### Reference Applications

Located in `apps/reference_apps/`. Advanced apps demonstrating specialized use cases.

See [`apps/reference_apps/README.md`](apps/reference_apps/README.md) for details. Each app has its own `README` at `apps/reference_apps/<app>/README`.

### TAO Apps

Located in `apps/tao_apps/`. Sample applications for running [NVIDIA TAO Toolkit](https://developer.nvidia.com/tao-toolkit) models with DeepStream.

See [`apps/tao_apps/README.md`](apps/tao_apps/README.md) for supported models, build, and run instructions.

Supported models include: PeopleNet Transformer, CitySemSegFormer, Mask2Former, Re-Identification, BodyPose3DNet, PoseClassification, OCDNet, OCRNet, LPDNet, LPRNet, Retail Detector, Retail Object Recognition.

### Service Maker

Located in `service-maker/`. A C++ and Python SDK that abstracts GStreamer/GLib APIs for building DeepStream applications declaratively.

See [`service-maker/README.md`](service-maker/README.md) for build instructions, API overview, and app list. See also the [Service Maker introduction](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_service_maker_intro.html).

- **C++ apps** — `service-maker/sources/apps/cpp/`
- **Python apps** — `service-maker/sources/apps/python/` (flow API and pipeline API)
- **Modules** — reusable signal handlers, probes, video sinks/sources, and more

### GStreamer Plugins

Located in `gst-plugins/`. Source for all DeepStream GStreamer plugins.

See [`gst-plugins/README.md`](gst-plugins/README.md) for the full list. Each plugin has its own `README` at `gst-plugins/<plugin>/README` with properties and usage. See also the [DeepStream Plugins introduction](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_Intro.html).

### GStreamer Utilities

Located in `gst-utils/`. Each utility has its own `README` at `gst-utils/<utility>/README`:

- [`gstnvcustomhelper`](gst-utils/gstnvcustomhelper/README)
- [`gstnvdscustomhelper`](gst-utils/gstnvdscustomhelper/README)
- [`gst-nvdssr`](gst-utils/gst-nvdssr/README)

### Utility Libraries

Located in `utils/`.

See [`utils/README.md`](utils/README.md) for the full list. Each library has its own `README` at `utils/<library>/README`.

---

## Platform-Specific Components

| Component | Platform | Notes |
|---|---|---|
| `deepstream-ipc-test` | Jetson (aarch64) only | Relies on NVDEC engine sharing not available on x86 or SBSA |
| `deepstream-multigpu-nvlink-test` | x86 dGPU only | Requires NVLink; not supported on Jetson |
| `deepstream-ucx-test` | x86 only | No UCX support shipped for aarch64 or SBSA; skipped by `build/build.sh` on those platforms |

## Components with External Prerequisites

The following components build successfully only when the listed external SDK is installed on the system. See each component's README for installation links and setup details.

| Component | External Prerequisite | Details |
|---|---|---|
| `gst-dsexample-cuda` | OpenCV 4.x built with CUDA support | [gst-plugins/gst-dsexample-cuda/README](gst-plugins/gst-dsexample-cuda/README) |
| `gst-nvdsudp` | Rivermax SDK (`rivermax_api.h`) | [gst-plugins/gst-nvdsudp/README](gst-plugins/gst-nvdsudp/README) |
| `libnvdsgst_inferserver` | Triton Inference Server client | [gst-plugins/gst-nvinferserver/README](gst-plugins/gst-nvinferserver/README) |

---

## Triton Inference Server Setup

Components that use `nvinferserver` (Gst-nvinferserver / Triton) require the Triton Inference Server and its backends to be available before use. Setup differs by platform.

### x86 (dGPU)

On x86 dGPU platforms, run these components inside DeepStream's Triton Inference Server container, which ships with Triton and the supported backend libraries pre-installed.

1. Pull the container (`[version]` is the DeepStream version, e.g. `9.1`):

```bash
docker pull nvcr.io/nvidia/deepstream:[version]-triton
```

2. Allow applications to connect to the host X display and start the container with the correct GPU id:

```bash
xhost +
docker run --gpus '"device=0"' -it --rm -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e DISPLAY=$DISPLAY --net=host nvcr.io/nvidia/deepstream:[version]-triton
```

### Jetson (bare-metal)

The DeepStream Triton container already includes Triton Server and backends. To run Triton bare-metal (directly on a Jetson device), the Triton Server must be set up first.

1. Run the backend setup script (must be run as root) and export the Triton paths:

```bash
cd <DS_ROOT>/scripts/
sudo ./triton_backend_setup.sh
export PATH="${PATH:+${PATH}:}/opt/tritonserver/bin"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}/opt/tritonserver/lib"
```

> **Notes:**
>
> 1. By default the script downloads Triton Server version 2.60.0. For setting up any other version, change the package path accordingly. Without running this script and exporting the paths above, components using `nvinferserver` fail at runtime on Jetson.
> 2. The script extracts the Triton server executable to the `/opt/tritonserver/bin` folder. To run both the Triton server and DeepStream applications on the same Jetson device, use the following steps to start the Triton server:
>
> ```bash
> cd /opt/tritonserver/bin
> ./tritonserver --model-repository=/opt/nvidia/deepstream/deepstream/samples/triton_model_repo
> # Or for TAO Toolkit models:
> ./tritonserver --model-repository=/opt/nvidia/deepstream/deepstream/samples/triton_tao_model_repo
> ```
>
> **Note:** `<DS_ROOT>` is the path where the repo is cloned.

### Preparing classification video samples

Some Triton classification samples require a classification test video. Prepare it as follows:

1. Install `ffmpeg` (a prerequisite for the next step):

```bash
sudo apt-get update && sudo apt-get install ffmpeg
```

2. Generate the classification video stream into `samples/streams/classification_test_video.mp4`:

```bash
cd <DS_ROOT>/scripts/
./prepare_classification_test_video.sh
```

> **Note:** You can also copy your own classification video files to the same location instead of running this step.

### Preparing model repositories

The following steps download/generate the sample models used by the Triton configs (run on both x86 inside the container and Jetson):

1. Prepare the TensorRT and ONNX sample models into `samples/triton_model_repo`:

```bash
cd <DS_ROOT>/scripts/
./prepare_ds_triton_model_repo.sh
```

2. Prepare NVIDIA TAO Toolkit models (e.g. PeopleNet Transformer) into `samples/triton_tao_model_repo`:

```bash
cd <DS_ROOT>/scripts/
./prepare_ds_triton_tao_model_repo.sh
```

> **Note:** Run the scripts with `sudo -E` or as root if there are file-permission issues.

For gRPC-based Triton usage, the same setup applies; Triton Server runs in a separate DeepStream Triton container (x86) or on the L4T host (Jetson), and the client communicates over gRPC.

---

## Deprecated Components

The following components are **legacy and no longer maintained**. They are still shipped with the DeepStream 9.1 public release but have been excluded from this repository. Deprecation was announced as part of the DS 9.1 public release.

### Sample Applications

- `deepstream-audio`
- `deepstream-avsync`
- `deepstream-mrcnn-test`
- `deepstream-segmentation-test`
- `deepstream_asr_app`
- `deepstream_asr_tts_app`

### GStreamer Plugins

- `gst-nvdsA2Vtemplate`
- `gst-nvdsaudiotemplate`
- `gst-nvdsspeech`
- `gst-nvdstexttospeech`

For component details, source, and configuration, refer to the [DeepStream 9.1 public release documentation](https://docs.nvidia.com/metropolis/deepstream/dev-guide/).
