# DeepStream

[NVIDIA DeepStream SDK](https://developer.nvidia.com/deepstream-sdk) is a streaming analytics toolkit for AI-based video and image understanding, providing a GStreamer-based framework to build multi-stream, multi-model inference pipelines on NVIDIA GPUs (dGPU and Jetson).

DeepStream pipelines combine hardware-accelerated decoding/encoding, TensorRT inference, object tracking, and message-broker integrations to deliver real-time video analytics across dGPU and Jetson platforms.

# Overview

This repository contains the complete source code for DeepStream 9.0.

**Components** ([`src/`](src/README.md)):
- [`src/gst-plugins/`](src/README.md#gstreamer-plugins) — DeepStream GStreamer plugin sources
- [`src/utils/`](src/README.md#utility-libraries) — utility library sources
- [`src/apps/sample_apps/`](src/README.md#sample-applications) — GStreamer-based sample applications
- [`src/apps/reference_apps/`](src/README.md#reference-applications) — advanced reference applications
- [`src/apps/tao_apps/`](src/README.md#tao-apps) — TAO-model integration apps
- [`src/service-maker/`](src/README.md#service-maker) — Service Maker C++/Python SDK and apps

**Tools** (`tools/`):
- [`inference_builder`](https://github.com/NVIDIA-AI-IOT/inference_builder) — visual inference pipeline builder
- [`auto-magic-calib`](https://github.com/NVIDIA-AI-IOT/auto-magic-calib) — camera auto-calibration tool
- [`yolo_deepstream`](tools/yolo_deepstream/README.md) — YOLO + TensorRT integration
- [`sam2-onnx-tensorrt`](tools/sam2-onnx-tensorrt/README.md) — SAM2 ONNX-to-TensorRT conversion

**AI agent skills** ([`skills/`](skills/README.md), for Claude Code & compatible coding agents):
- `deepstream-dev` — general DeepStream development
- `deepstream-import-vision-model` — autonomous vision-model onboarding

# Requirements

Before building, ensure the following prerequisites are installed:

- **NVIDIA compute stack** — driver, CUDA, cuDNN, and TensorRT at the versions listed below. See the [DeepStream SDK Installation Guide](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Installation.html).
- **DeepStream 9.0** — installed via the DS 9.0 public Debian package and its `install.sh`.

> **SBSA / DGX Spark:** no DS deb package exists for this platform — use the NVIDIA SBSA Docker container, which bundles the compute stack and DeepStream. See [build/BUILD.md](build/BUILD.md).

# Getting Started

```bash
# Install Git LFS (required for sample video streams used by some apps)
sudo apt-get install git-lfs

# Clone the repo with submodules
git clone --recurse-submodules https://github.com/NVIDIA/DeepStream.git && cd DeepStream

# Pull LFS-tracked files
git lfs install && git lfs pull

# Build. The script prompts for sudo only when installing to system paths.
bash build/build.sh
```

See **[build/BUILD.md](build/BUILD.md)** for full build instructions, including system package dependencies (x86, aarch64, SBSA / DGX Spark), `build/build.sh` usage and environment variables (`CUDA_VER`, `NVDS_VERSION`), and build output locations under `/opt/nvidia/deepstream/deepstream-9.0/`.

**Supported Platforms**

| Platform | Architecture | Notes |
|---|---|---|
| x86 dGPU | x86_64 | Ubuntu 24.04, CUDA 13.1, TensorRT 10.14.x, driver 590+ |
| Jetson | aarch64 | JetPack 7.1 GA (CUDA 13.0, TensorRT 10.13.x) |
| SBSA / DGX Spark | aarch64 | No DS tar/deb package; build and run inside the NVIDIA SBSA Docker container |

# Usage

After `bash build/build.sh`, binaries are installed to `/opt/nvidia/deepstream/deepstream-9.0/bin/`. Run the reference `deepstream-app` with one of the sample configs:

```bash
cd /opt/nvidia/deepstream/deepstream-9.0/samples/configs/deepstream-app
deepstream-app -c source30_1080p_dec_infer-resnet_tiled_display.txt

# After the first install, clear the GStreamer plugin cache if needed:
rm -rf ~/.cache/gstreamer-1.0/
```

Each app must be run from its source directory so relative config paths resolve correctly. Refer to the `README` inside each app directory for app-specific run instructions and config options.

## Running with Triton Inference Server (Docker)

> **Optional.** Skip this if the bare-metal build above already works for you. Use the Triton container when you need Triton-backed inference, or when you're on SBSA / DGX Spark (where no native DS deb package exists).

NVIDIA publishes a DeepStream Docker image bundled with Triton Inference Server.

```bash
# One-time NGC login (get an API key from https://ngc.nvidia.com)
docker login nvcr.io           # username: $oauthtoken,  password: <NGC API key>

# Pull the image (use 9.0-triton-arm-sbsa instead on SBSA / DGX Spark)
docker pull nvcr.io/nvidia/deepstream:9.0-triton-multiarch

# Launch with display (use 'fakesink' in your pipeline for headless)
export DISPLAY=:0 && xhost +
docker run -it --rm --gpus all --network=host \
    -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix \
    nvcr.io/nvidia/deepstream:9.0-triton-multiarch

# Inside the container, run a Triton-backed sample
cd /opt/nvidia/deepstream/deepstream-9.0/samples/configs/deepstream-app-triton
deepstream-app -c source30_1080p_dec_infer-resnet_tiled_display.txt
```

**Prerequisites:** Docker (`docker-ce`), the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html), NVIDIA driver 590+, and an NGC API key. Triton sample model repos ship under `/opt/nvidia/deepstream/deepstream/samples/triton_model_repo/`. For gRPC-backed Triton, use `samples/configs/deepstream-app-triton-grpc/` instead.

- More examples/tutorials: [DeepStream Quickstart Guide](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Quickstart.html#run-deepstream-app-the-reference-application)
- API reference: [DeepStream SDK Developer Guide](https://docs.nvidia.com/metropolis/deepstream/dev-guide/)

# Documentation

| Page | Description |
|------|-------------|
| [Overview & Architecture](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Overview.html) | What DeepStream is, key features, and the GStreamer-based pipeline architecture. |
| [Release Notes](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Release_notes.html) | What's new in this release, supported platforms, and known issues. |
| [Installation](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Installation.html) | Step-by-step installation of the NVIDIA compute stack and DeepStream SDK on x86 and Jetson. |
| [Docker Containers](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_docker_containers.html) | Available DeepStream NGC container images (incl. Triton variants) and how to pull and run them. |
| [DeepStream Samples](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_C_Sample_Apps.html) | Reference walkthroughs of the bundled C/C++ sample applications and what each one demonstrates. |
| [Reference Applications](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_ref_app_github.html) | Advanced GitHub-hosted reference apps demonstrating specialized end-to-end pipelines. |
| [DeepStream Plugins](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_Intro.html) | Reference for every DeepStream GStreamer plugin — properties, pad caps, and usage. |
| [Service Maker](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_service_maker_intro.html) | Build DeepStream pipelines declaratively with the C++ / Python Service Maker SDK. |
| [Inference Builder](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Inference_Builder.html) | Compose and configure DeepStream inference pipelines visually with the Inference Builder tool. |
| [DeepStream Coding Agent](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_AI_Agent.html) | Use the bundled `skills/` with Claude Code and other AI coding assistants to generate DeepStream pipelines. |

# Repository Structure

```
DeepStream/
├── skills/                                  # product-local AI agent skills
│   ├── deepstream-dev/                      # general DS development skill
│   └── deepstream-import-vision-model/      # autonomous vision-model onboarding skill
├── .claude/                                 # Claude Code project-local config (settings, commands)
├── .github/                                 # GitHub workflows, issue templates, CODEOWNERS
├── build/
│   ├── build.sh                             # top-level build driver
│   └── BUILD.md                             # build instructions
├── example_prompts/                         # example prompts for AI coding agents
├── includes/                                # shared public headers (ds3d, nvdsinferserver, …)
├── scripts/
│   ├── install_opensource_deps.sh           # builds open-source deps (OpenTelemetry, civetweb, …)
│   └── print_env.sh                         # env diagnostic helper for bug reports
├── src/
│   ├── apps/                                # sample, reference, and TAO sample applications
│   ├── gst-plugins/                         # GStreamer plugin sources (per-plugin subdirs)
│   ├── gst-utils/                           # GStreamer utility library sources
│   ├── service-maker/                       # Service Maker C++/Python SDK and apps
│   └── utils/                               # utility library sources (per-library subdirs)
└── tools/                                   # standalone tools
    ├── auto-magic-calib/                    # camera auto-calibration tool
    ├── inference_builder/                   # visual inference pipeline builder
    ├── sam2-onnx-tensorrt/                  # SAM2 ONNX-to-TensorRT conversion
    └── yolo_deepstream/                     # YOLO + TensorRT integration
```

# Performance
Summary of benchmarks; for detailed performance numbers and hardware used, refer to the [DeepStream Performance Guide](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Performance.html).

## Releases & Roadmap
- Releases/Changelog: <link>
- **`deepstream_libraries`** is not currently part of this GitHub OSS repository. For details and usage, refer to the [DeepStream Libraries documentation](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Libraries.html).
  
# Contribution Guidelines
This project is currently not accepting contributions.

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

- [DeepStream NGC container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/deepstream) — prebuilt runtime image (also bundles Triton).
- [NVIDIA TAO Toolkit](https://developer.nvidia.com/tao-toolkit) — model training/optimization, source of the TAO sample apps.
- [GStreamer documentation](https://gstreamer.freedesktop.org/documentation/) — underlying framework.

# License
This project is licensed under [CC-BY-4.0 AND Apache-2.0](./LICENSE).
- SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
