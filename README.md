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

# NVIDIA DeepStream SDK 9.0 — Source Repository

[NVIDIA DeepStream SDK](https://developer.nvidia.com/deepstream-sdk) is a streaming analytics toolkit for AI-based video and image understanding. It provides a GStreamer-based framework for building multi-stream, multi-model inference pipelines that run on NVIDIA GPUs (dGPU and Jetson).

# Overview

This repository contains the complete source code for DeepStream 9.0: GStreamer plugins, utility libraries, sample applications, reference applications, TAO-integrated apps, and the Service Maker C++/Python SDK.

## Repository Structure

> **Note:** Currently applicable only for DeepStream 9.0.

```
DeepStream/
├── .agents/                             # placeholder — reserved for AI agent definitions
├── .claude/                             # placeholder — Claude Code configuration
├── .github/                             # placeholder — GitHub workflows / metadata
├── BUILD.md                             # build instructions
├── build.sh                             # top-level build driver
├── SECURITY.md                          # vulnerability disclosure
├── example_prompts/                     # example prompts for AI coding agents
├── skills/                              # placeholder — AI agent skills
├── scripts/
│   └── install_opensource_deps.sh       # builds open-source deps (OpenTelemetry, civetweb, …)
├── includes/                            # shared public headers
│   ├── ds3d/
│   ├── nvdsinferserver/
│   ├── opentelemetry/
│   └── prometheus/
├── tools/                               # standalone tools (see Components section)
│   ├── sam2-onnx-tensorrt/              # SAM2 ONNX-to-TensorRT conversion
│   └── yolo_deepstream/                 # YOLO + TensorRT integration
└── src/
    ├── apps/
    │   ├── common/                      # shared app helpers (config parsers, etc.)
    │   ├── sample_apps/                 # GStreamer-based sample applications
    │   ├── reference_apps/              # advanced reference applications
    │   └── tao_apps/                    # TAO-model integration apps
    ├── gst-plugins/                     # GStreamer plugin sources (per-plugin subdirs)
    ├── gst-utils/                       # GStreamer utility library sources
    │   ├── gstnvcustomhelper/
    │   ├── gstnvdscustomhelper/
    │   └── gst-nvdssr/
    ├── service-maker/                   # Service Maker C++ / Python SDK and apps
    │   ├── cmake/                       # CMake configuration
    │   ├── includes/                    # Service Maker public headers
    │   └── sources/
    │       ├── apps/
    │       │   ├── cpp/                 # C++ sample apps
    │       │   └── python/              # Python sample apps (flow + pipeline API)
    │       └── modules/                 # reusable signal handlers, probes, sinks
    └── utils/                           # utility library sources (per-library subdirs)
```

## Components

### Sample Applications

Located in `src/apps/sample_apps/`. These apps demonstrate core DeepStream pipeline patterns using GStreamer APIs directly.

See [`src/apps/sample_apps/README.md`](src/apps/sample_apps/README.md) for the full list with descriptions and run commands. Each app has its own `README` at `src/apps/sample_apps/<app>/README`.

### Reference Applications

Located in `src/apps/reference_apps/`. Advanced apps demonstrating specialized use cases.

See [`src/apps/reference_apps/README.md`](src/apps/reference_apps/README.md) for details. Each app has its own `README` at `src/apps/reference_apps/<app>/README`.

### TAO Apps

Located in `src/apps/tao_apps/`. Sample applications for running [NVIDIA TAO Toolkit](https://developer.nvidia.com/tao-toolkit) models with DeepStream.

See [`src/apps/tao_apps/README.md`](src/apps/tao_apps/README.md) for supported models, build, and run instructions.

Supported models include: PeopleNet Transformer, CitySemSegFormer, Mask2Former, Re-Identification, BodyPose3DNet, PoseClassification, OCDNet, OCRNet, LPDNet, LPRNet, Retail Detector, Retail Object Recognition.

### Service Maker

Located in `src/service-maker/`. A C++ and Python SDK that abstracts GStreamer/GLib APIs for building DeepStream applications declaratively.

See [`src/service-maker/README.md`](src/service-maker/README.md) for build instructions, API overview, and app list. See also the [Service Maker introduction](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_service_maker_intro.html).

- **C++ apps** — `src/service-maker/sources/apps/cpp/`
- **Python apps** — `src/service-maker/sources/apps/python/` (flow API and pipeline API)
- **Modules** — reusable signal handlers, probes, video sinks/sources, and more

### GStreamer Plugins

Located in `src/gst-plugins/`. Source for all DeepStream GStreamer plugins.

See [`src/gst-plugins/README.md`](src/gst-plugins/README.md) for the full list. Each plugin has its own `README` at `src/gst-plugins/<plugin>/README` with properties and usage. See also the [DeepStream Plugins introduction](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_Intro.html).

### GStreamer Utilities

Located in `src/gst-utils/`. Each utility has its own `README` at `src/gst-utils/<utility>/README`:

- [`gstnvcustomhelper`](src/gst-utils/gstnvcustomhelper/README)
- [`gstnvdscustomhelper`](src/gst-utils/gstnvdscustomhelper/README)
- [`gst-nvdssr`](src/gst-utils/gst-nvdssr/README)

### Utility Libraries

Located in `src/utils/`.

See [`src/utils/README.md`](src/utils/README.md) for the full list. Each library has its own `README` at `src/utils/<library>/README`.

### Tools

Located in `tools/`. Each tool has its own `README` at `tools/<tool>/README.md`:

- [`sam2-onnx-tensorrt`](tools/sam2-onnx-tensorrt/README.md)
- [`yolo_deepstream`](tools/yolo_deepstream/README.md)

## Deprecated Components

The following components are **legacy and no longer maintained**. They are still shipped with the DeepStream 9.0 public release but have been excluded from this repository. Deprecation was announced as part of the DS 9.0 public release.

### Sample Applications

- `deepstream-audio`
- `deepstream-avsync`
- `deepstream-mrcnn-test`
- `deepstream-segmentation-test`
- `deepstream-ucx-test`
- `deepstream_asr_app`
- `deepstream_asr_tts_app`

### GStreamer Plugins

- `gst-nvdsA2Vtemplate`
- `gst-nvdsaudiotemplate`
- `gst-nvdsspeech`
- `gst-nvdstexttospeech`

For component details, source, and configuration, refer to the [DeepStream 9.0 public release documentation](https://docs.nvidia.com/metropolis/deepstream/dev-guide).

# Getting Started

Before using this repository, ensure the following are already set up on your system:

1. **NVIDIA compute stack** — driver, CUDA, cuDNN, and TensorRT at the versions listed in the [Requirements](#requirements) section. Refer to the [DeepStream SDK Installation Guide](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Installation.html) for installation steps.

2. **DeepStream 9.0** — installed via the DS 9.0 public released Debian package and its `install.sh` script.

> **SBSA / DGX Spark:** No DS deb package exists for this platform. The NVIDIA SBSA Docker container serves as the environment — it includes the compute stack and DeepStream. See [BUILD.md](BUILD.md) for container setup steps.

Once the above are in place, clone this repository and follow [BUILD.md](BUILD.md) to build and run:

```bash
sudo apt-get install git-lfs
git clone https://github.com/NVIDIA/DeepStream && cd DeepStream
git lfs install && git lfs pull
sudo bash build.sh
```

See **[BUILD.md](BUILD.md)** for complete instructions covering:

- System package dependencies (x86, aarch64, SBSA / DGX Spark)
- Git LFS setup for prebuilt artifact tarballs
- `build.sh` usage and environment variables (`CUDA_VER`, `NVDS_VERSION`)
- Build output locations under `/opt/nvidia/deepstream/deepstream-9.0/`
- Service Maker local CMake build

# Requirements

## Supported Platforms

| Platform | Architecture | Notes |
|---|---|---|
| x86 dGPU | x86_64 | Ubuntu 24.04, CUDA 13.1, TensorRT 10.14.x, driver 590+ |
| Jetson | aarch64 | JetPack 7.1 GA (CUDA 13.0, TensorRT 10.13.x) |
| SBSA / DGX Spark | aarch64 | No DS tar/deb package; build and run inside the NVIDIA SBSA Docker container |

## Platform-Specific Components

| Component | Platform | Notes |
|---|---|---|
| `deepstream-ipc-test` | Jetson (aarch64) only | Relies on NVDEC engine sharing not available on x86 or SBSA |
| `deepstream-multigpu-nvlink-test` | x86 dGPU only | Requires NVLink; not supported on Jetson |

## Components with External Prerequisites

The following components build successfully only when the listed external SDK is installed on the system. See each component's README for installation links and setup details.

| Component | External Prerequisite | Details |
|---|---|---|
| `gst-dsexample-cuda` | OpenCV 4.x built with CUDA support | [src/gst-plugins/gst-dsexample-cuda/README](src/gst-plugins/gst-dsexample-cuda/README) |
| `gst-nvdsudp` | Rivermax SDK (`rivermax_api.h`) | [src/gst-plugins/gst-nvdsudp/README](src/gst-plugins/gst-nvdsudp/README) |
| `libnvdsgst_inferserver` | Triton Inference Server client | [src/gst-plugins/gst-nvinferserver/README](src/gst-plugins/gst-nvinferserver/README) |

# Usage

## Running Sample Apps

After building with `build.sh`, binaries are installed to `/opt/nvidia/deepstream/deepstream-9.0/bin/`.

All apps must be run from their source directory so relative config paths resolve correctly. Refer to the `README` in each app's directory for app-specific run instructions and config file options.

### Run `deepstream-app` (the reference application)

Navigate to the `configs/deepstream-app` directory on the development kit:

```bash
cd /opt/nvidia/deepstream/deepstream-9.0/samples/configs/deepstream-app
```

Run the reference application:

```bash
deepstream-app -c <path_to_config_file>
```

For example:

```bash
deepstream-app -c source30_1080p_dec_infer-resnet_tiled_display.txt
```

`<path_to_config_file>` is the path to one of the reference application's configuration files in `configs/deepstream-app/`. See *Package Contents in `configs/deepstream-app/`* for the list of available files. See also the [DeepStream Quickstart — Run deepstream-app](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Quickstart.html#run-deepstream-app-the-reference-application).

After first install, clear the GStreamer plugin cache:

```bash
rm -rf ~/.cache/gstreamer-1.0/
```

## Running with Triton Inference Server (Docker)

NVIDIA publishes a DeepStream Docker image bundled with Triton Inference Server. This is the recommended path when you need Triton-backed inference (`gst-nvinferserver` / `libnvdsgst_inferserver`), and is the only supported runtime path on SBSA / DGX Spark, where no native DeepStream deb package is available.

### Prerequisites

- **Docker** — install `docker-ce` per [Docker's official install guide](https://docs.docker.com/engine/install).
- **NVIDIA Container Toolkit** — required for GPU access; see the [install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).
- **NVIDIA Driver** — 590+ for dGPU.
- **NGC login** — pulling the image requires NGC authentication.

```bash
# Get an API key from https://ngc.nvidia.com, then:
docker login nvcr.io
# Username: $oauthtoken
# Password: <YOUR_NGC_API_KEY>
```

### Pull the Triton Image

| Platform | Image |
|---|---|
| x86 dGPU / Jetson | `nvcr.io/nvidia/deepstream:9.0-triton-multiarch` |
| SBSA / DGX Spark (ARM dGPU) | `nvcr.io/nvidia/deepstream:9.0-triton-arm-sbsa` |

```bash
docker pull nvcr.io/nvidia/deepstream:9.0-triton-multiarch
```

The image bundles the DeepStream runtime, Triton Inference Server, and a development environment (compilers + headers) so the repository sources can also be built from inside the container.

### Launch the Container

With display (for `nveglglessink` / `nv3dsink`):

```bash
export DISPLAY=:0
xhost +

docker run -it --rm \
    --gpus all \
    --network=host \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix/:/tmp/.X11-unix \
    nvcr.io/nvidia/deepstream:9.0-triton-multiarch
```

Headless (use `fakesink` / `filesink` in your pipeline):

```bash
docker run -it --rm --gpus all \
    nvcr.io/nvidia/deepstream:9.0-triton-multiarch
```

### Triton Model Repository

The container ships sample Triton model repositories at:

```
/opt/nvidia/deepstream/deepstream/samples/triton_model_repo
/opt/nvidia/deepstream/deepstream/samples/triton_tao_model_repo
```

The `*_inferserver_config.txt` files shipped under `samples/configs/deepstream-app-triton/` and the `tritonclient` reference apps point at these paths via `infer_config.backend.triton.model_repo.root`.

### Run a Triton-Backed Sample

Inside the container:

```bash
cd /opt/nvidia/deepstream/deepstream-9.0/samples/configs/deepstream-app-triton
deepstream-app -c source30_1080p_dec_infer-resnet_tiled_display.txt
```

For gRPC-backed Triton, use `samples/configs/deepstream-app-triton-grpc/` instead. After the first launch, clear the GStreamer plugin cache once:

```bash
rm -rf ~/.cache/gstreamer-1.0/
```

# Performance (Optional)
Summary of benchmarks; link to detailed results and hardware used.

## Releases & Roadmap 
- Releases/Changelog: <link>
- (Optional) Next milestones or link to `ROADMAP.md`.
  
# Contribution Guidelines
- Start here: `CONTRIBUTING.md`
- Code of Conduct: `CODE_OF_CONDUCT.md`
- Development quickstart (build/test):
```bash
<clone> && <deps> && <build/test>
```
## Governance & Maintainers
- Governance: `GOVERNANCE.md`
- Maintainers: <team/handles>
- Labeling/triage policy: <link>

## Security
- Vulnerability disclosure: `SECURITY.md`
- Do not file public issues for security reports.

## Support
- Level: <Experimental | Maintained | Stable>
- How to get help: Issues/Discussions/<channel link>
- Response expectations (if any).

# Community
Provide the channel for community communications.

# References
Provide a list of related references

# License

Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

LICENSE details yet to be finalized — tracked in OSRB Bug 6155166.
