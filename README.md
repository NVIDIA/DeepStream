# DeepStream

[NVIDIA DeepStream SDK](https://developer.nvidia.com/deepstream-sdk) is a streaming analytics toolkit for AI-based video and image understanding, providing a GStreamer-based framework to build multi-stream, multi-model inference pipelines on NVIDIA GPUs (dGPU and Jetson).

DeepStream pipelines combine hardware-accelerated decoding/encoding, TensorRT inference, object tracking, and message-broker integrations to deliver real-time video analytics across dGPU and Jetson platforms.

DeepStream is published as a unified GitHub repository; sources are to be used under the project [LICENSE](LICENSE) (documentation: CC-BY-4.0; source code: Apache-2.0). GitHub Release assets specific license details can be referred at [Release Asset License](#release-asset-license) under [DeepStream Release Assets](#deepstream-release-assets). The project is actively maintained, but currently not accepting code contributions — refer to the [Contribution Guidelines](#contribution-guidelines) for more details.

# Overview

This repository contains the complete source code for DeepStream 9.1.

**Components** ([`src/`](src/README.md)):
- [`src/gst-plugins/`](src/README.md#gstreamer-plugins) — DeepStream GStreamer plugin sources
- [`src/utils/`](src/README.md#utility-libraries) — utility library sources
- [`src/apps/sample_apps/`](src/README.md#sample-applications) — GStreamer-based sample applications
- [`src/apps/reference_apps/`](src/README.md#reference-applications) — advanced reference applications
- [`src/apps/tao_apps/`](src/README.md#tao-apps) — TAO-model integration apps
- [`src/service-maker/`](src/README.md#service-maker) — Service Maker C++/Python SDK and apps

**Tools** ([`tools/`](tools/)):
- [`inference_builder`](https://github.com/NVIDIA-AI-IOT/inference_builder/tree/3f0c09f2e3da076cbbb75e17bdebd565b03d1a18) — visual inference pipeline builder
- [`auto-magic-calib`](https://github.com/NVIDIA-AI-IOT/auto-magic-calib/tree/bc5b1d9b3154d4242f147c35754670bc7981d37a) — camera auto-calibration tool
- [`yolo_deepstream`](tools/yolo_deepstream/README.md) — YOLO + TensorRT integration
- [`sam2-onnx-tensorrt`](tools/sam2-onnx-tensorrt/README.md) — SAM2 ONNX-to-TensorRT conversion
- [`deepstream-host-provisioning`](tools/deepstream-host-provisioning/README.md) — Ansible playbooks to provision Ubuntu hosts (x86_64 & ARM SBSA) with the full NVIDIA compute stack (driver, CUDA, cuDNN, TensorRT, Docker) and DeepStream dependencies
- [`deepstream-otelcol`](tools/deepstream-otelcol/README.md) — OpenTelemetry Collector setup for streaming metrics

**AI agent skills** ([`skills/`](skills/README.md), for Claude Code & compatible coding agents):
- [`deepstream-dev`](skills/deepstream-dev/SKILL.md) — general DeepStream development
- [`deepstream-generate-pipeline`](skills/deepstream-generate-pipeline/SKILL.md) — interactive `gst-launch` pipeline builder
- [`deepstream-profile-pipeline`](skills/deepstream-profile-pipeline/SKILL.md) — Nsight Systems profiling & config derivation
- [`deepstream-sop`](skills/deepstream-sop/SKILL.md) — SOP step-sequence compliance microservice
- [`deepstream-import-vision-model`](skills/deepstream-import-vision-model/SKILL.md) — autonomous vision-model onboarding
- [`deepstream-run-mv3dt`](skills/deepstream-run-mv3dt/SKILL.md) — run the MV3DT reference app on sample or custom datasets
- [`amc-setup-calibration-stack`](skills/amc-setup-calibration-stack/SKILL.md) — launch the AutoMagicCalib microservice and UI
- [`amc-run-sample-calibration`](skills/amc-run-sample-calibration/SKILL.md) — verify AMC with the bundled sample dataset
- [`amc-run-video-calibration`](skills/amc-run-video-calibration/SKILL.md) — calibrate user-provided MP4 camera videos
- [`amc-run-rtsp-calibration`](skills/amc-run-rtsp-calibration/SKILL.md) — calibrate live RTSP camera streams through VIOS capture

# Requirements

Before building, ensure the following prerequisites are installed:

- **NVIDIA compute stack** — driver, CUDA, cuDNN, and TensorRT at the versions listed below. See the [DeepStream SDK Installation Guide](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Installation.html).
- **DeepStream 9.1 proprietary runtime** — the prebuilt libraries and sample data are published as GitHub Release assets. `bash build/build.sh` automatically downloads the packages into `artifacts/` before the build process starts.

  > **Note:** After a successful build, `artifacts/` is removed automatically. Pass `--keep-assets` to retain them.

> **SBSA / DGX Spark:** DeepStream bare-metal installation is not supported on SBSA. Build and validation must be done inside the NVIDIA SBSA Docker container, which ships with DeepStream pre-installed — `build/build.sh` is **not required** inside the container. If you modify an open-source component, run `make && make install` in that component's directory to deploy the updated binaries/libs. See [build/BUILD.md](build/BUILD.md).

# Getting Started

```bash
# Install Git LFS (required for LFS-tracked sample media in reference apps)
sudo apt-get install git-lfs && git lfs install

# Clone the repo with submodules
git clone --recurse-submodules https://github.com/NVIDIA/DeepStream.git && cd DeepStream
```

See **[build/BUILD.md](build/BUILD.md)** for full build instructions, including system package dependencies (x86, aarch64, SBSA / DGX Spark), `build/build.sh` usage and environment variables (`CUDA_VER`, `NVDS_VERSION`, `INSTALL_METHOD`), and build output locations under `/opt/nvidia/deepstream/deepstream-9.1/`.

**Supported Platforms**

| Platform | Architecture | Notes |
|---|---|---|
| x86 dGPU | x86_64 | Ubuntu 24.04, CUDA 13.2, TensorRT 10.16.x, driver 595+ |
| Jetson | aarch64 | JetPack 7.2 GA (CUDA 13.2, TensorRT 10.16.x) |
| SBSA / DGX Spark | aarch64 | Bare-metal DS installation not supported; build and run inside the NVIDIA SBSA Docker container. Only open-source components are built — no artifacts installed. |

```bash
# Build. The script prompts for sudo only when installing to system paths.
bash build/build.sh
```

# Usage

After `bash build/build.sh`, binaries are installed to `/opt/nvidia/deepstream/deepstream-9.1/bin/`. Run the reference `deepstream-app` with one of the sample configs:

```bash
cd /opt/nvidia/deepstream/deepstream-9.1/samples/configs/deepstream-app
deepstream-app -c source30_1080p_dec_infer-resnet_tiled_display.txt

# After the first install, clear the GStreamer plugin cache if needed:
rm -rf ~/.cache/gstreamer-1.0/
```

- More such examples: [DeepStream Quickstart Guide](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Quickstart.html#run-deepstream-app-the-reference-application)
- Detailed API reference: [DeepStream SDK Developer Guide](https://docs.nvidia.com/metropolis/deepstream/dev-guide/)

Each app must be run from its source directory so relative config paths resolve correctly. Refer to the `README` inside each app directory for app-specific run instructions and config options.

## Running with Triton Inference Server (Docker)

> **Optional.** Skip this if the bare-metal build above already works for you. Use the Triton container when you need Triton-backed inference. On SBSA / DGX Spark, use the SBSA Docker container as described above — bare-metal installation is not supported.

NVIDIA publishes a DeepStream Docker image bundled with Triton Inference Server.

```bash
# One-time NGC login (get an API key from https://ngc.nvidia.com)
docker login nvcr.io           # username: $oauthtoken,  password: <NGC API key>

# Pull the image (use 9.1-triton-arm-sbsa instead on SBSA / DGX Spark)
docker pull nvcr.io/nvidia/deepstream:9.1-triton-multiarch

# Launch with display (use 'fakesink' in your pipeline for headless)
export DISPLAY=:0 && xhost +
docker run -it --rm --gpus all --network=host \
    -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix \
    nvcr.io/nvidia/deepstream:9.1-triton-multiarch

# Inside the container, run a Triton-backed sample
cd /opt/nvidia/deepstream/deepstream-9.1/samples/configs/deepstream-app-triton
deepstream-app -c source30_1080p_dec_infer-resnet_tiled_display.txt
```

**Prerequisites:** Docker (`docker-ce`), the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html), NVIDIA driver 595+, and an NGC API key. Triton sample model repos ship under `/opt/nvidia/deepstream/deepstream/samples/triton_model_repo/`. For gRPC-backed Triton, use `samples/configs/deepstream-app-triton-grpc/` instead.

# Documentation

| Page | Description |
|------|-------------|
| [Overview & Architecture](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Overview.html) | What DeepStream is, key features, and the GStreamer-based pipeline architecture. |
| [Release Notes](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Release_notes.html) | What's new in this release, supported platforms, and known issues. |
| [Installation](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Installation.html) | Step-by-step installation of the NVIDIA compute stack and DeepStream SDK on x86 and Jetson. |
| [Docker Containers](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_docker_containers.html) | Available DeepStream NGC container images (incl. Triton variants) and how to pull and run them. |
| [DeepStream Samples](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_C_Sample_Apps.html) | Reference walkthroughs of the bundled C/C++ sample applications and what each one demonstrates. |
| [Reference Applications](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_ref_app_github.html) | Advanced GitHub-hosted reference apps demonstrating specialized end-to-end pipelines. |
| [TAO Apps](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_TAO_integration.html) | DeepStream sample apps integrating TAO-trained models (detection, classification, segmentation, pose, LPR). |
| [DeepStream Plugins](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_Intro.html) | Reference for every DeepStream GStreamer plugin — properties, pad caps, and usage. |
| [Service Maker](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_service_maker_intro.html) | Build DeepStream pipelines declaratively with the C++ / Python Service Maker SDK. |
| [Inference Builder](https://github.com/NVIDIA-AI-IOT/inference_builder/tree/3f0c09f2e3da076cbbb75e17bdebd565b03d1a18) | Compose and configure DeepStream inference pipelines visually with the Inference Builder tool. |
| [Auto Magic Calib](https://github.com/NVIDIA-AI-IOT/auto-magic-calib/tree/bc5b1d9b3154d4242f147c35754670bc7981d37a) | Automatic camera calibration tool for multi-camera DeepStream deployments. |
| [DeepStream Libraries](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Libraries.html) | DeepStream utility and helper libraries bundled as the `deepstream_libraries` submodule. |
| [DeepStream Coding Agent](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_AI_Agent.html) | Use the bundled `skills/` with Claude Code and other AI coding assistants to generate DeepStream pipelines. |

# Repository Structure

```
DeepStream/
├── .claude/                                 # Claude Code project-local config (settings, commands)
├── .github/                                 # GitHub workflows, issue templates, CODEOWNERS
├── build/
│   ├── BUILD.md                             # build instructions
│   └── build.sh                             # top-level build driver
├── example_prompts/                         # example prompts for AI coding agents
├── includes/                                # shared public headers (ds3d, nvdsinferserver, …)
├── scripts/
│   ├── install_artifacts.sh                 # installs proprietary libs + sample data to /opt
│   ├── install_opensource_deps.sh           # builds open-source deps (OpenTelemetry, civetweb, …)
│   ├── install.sh                           # system integration (update-alternatives, pyservicemaker)
│   ├── uninstall.sh                         # removes a DeepStream installation from /opt
│   ├── triton_backend_setup.sh              # sets up Triton inference server backends
│   ├── prepare_ds_triton_model_repo.sh      # prepares the Triton model repository
│   ├── prepare_ds_triton_tao_model_repo.sh  # prepares the Triton model repository for TAO models
│   ├── prepare_classification_test_video.sh # generates a classification test video
│   ├── rtpmanager/update_rtpmanager.sh      # rtpjitterbuffer/rtsp EOS-hang fix script + patch files
│   └── print_env.sh                         # env diagnostic helper for bug reports
├── skills/
│   ├── amc-run-sample-calibration/          # AMC sample-dataset calibration skill
│   ├── amc-run-video-calibration/           # AMC custom-video calibration skill
│   ├── amc-run-rtsp-calibration/            # AMC RTSP-stream calibration skill
│   ├── amc-setup-calibration-stack/         # AMC microservice + UI launch skill
│   ├── deepstream-dev/                      # general DS development skill
│   ├── deepstream-generate-pipeline/        # interactive gst-launch pipeline builder
│   ├── deepstream-profile-pipeline/         # Nsight Systems profiling & config derivation
│   ├── deepstream-sop/                      # SOP step-sequence compliance microservice
│   ├── deepstream-import-vision-model/      # autonomous vision-model onboarding skill
│   └── deepstream-run-mv3dt/                # MV3DT reference-app operations skill
├── src/
│   ├── apps/                                # sample, reference, and TAO sample applications
│   ├── gst-plugins/                         # GStreamer plugin sources (per-plugin subdirs)
│   ├── gst-utils/                           # GStreamer utility library sources
│   ├── service-maker/                       # Service Maker C++/Python SDK and apps
│   └── utils/                               # utility library sources (per-library subdirs)
├── deepstream_libraries/                    # DeepStream utility and helper libraries (submodule)
├── deepstream_dockers/                      # DeepStream dockers
└── tools/
    ├── auto-magic-calib/                    # camera auto-calibration tool
    ├── deepstream-otelcol/                  # OpenTelemetry Collector setup for streaming metrics
    ├── inference_builder/                   # visual inference pipeline builder
    ├── deepstream-host-provisioning/        # Ansible playbooks for host provisioning (x86_64 & ARM SBSA)
    ├── sam2-onnx-tensorrt/                  # SAM2 ONNX-to-TensorRT conversion
    └── yolo_deepstream/                     # YOLO + TensorRT integration
```

> **NOTE:** DeepStream-specific repositories previously hosted under the NVIDIA-AI-IOT GitHub organization are now consolidated into this repository.

The table below maps the existing NVIDIA-AI-IOT GitHub repository sources to their location in this repository.


| NVIDIA-AI-IOT GitHub repo                                                               | Location in this repository                            |
| --------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| [deepstream_tao_apps](https://github.com/NVIDIA-AI-IOT/deepstream_tao_apps)             | [src/apps/tao_apps/](src/apps/tao_apps/)               |
| [deepstream_reference_apps](https://github.com/NVIDIA-AI-IOT/deepstream_reference_apps) | [src/apps/reference_apps/](src/apps/reference_apps/)   |
| [deepstream_tools](https://github.com/NVIDIA-AI-IOT/deepstream_tools)                   | [tools/](tools/)                                       |
| [DeepStream_Coding_Agent](https://github.com/NVIDIA-AI-IOT/DeepStream_Coding_Agent)     | [skills/](skills/README.md)                            |
| [inference_builder](https://github.com/NVIDIA-AI-IOT/inference_builder)                 | [tools/inference_builder/](tools/inference_builder/)   |
| [auto-magic-calib](https://github.com/NVIDIA-AI-IOT/auto-magic-calib)                   | [tools/auto-magic-calib/](tools/auto-magic-calib/)     |
| [deepstream_dockers](https://github.com/NVIDIA-AI-IOT/deepstream_dockers)               | [deepstream_dockers/](deepstream_dockers/)             |
| [deepstream_libraries](https://github.com/NVIDIA-AI-IOT/deepstream_libraries)           | [deepstream_libraries/](deepstream_libraries/)         |



# Performance
Summary of benchmarks; for detailed performance numbers and hardware used, refer to the [DeepStream Performance Guide](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Performance.html).

# Releases & Roadmap

- **Releases / Changelog:** [CHANGELOG.md](./CHANGELOG.md)

## DeepStream Release Assets

The following `.deb` and `.tar` artifacts are published as GitHub Release assets. `build/build.sh` downloads them automatically into `artifacts/` before installation:

### Debian packages (`.deb`)

| Asset | Architecture | Description |
|---|---|---|
| `deepstream-9.1_9.1.0-1_amd64.deb` | x86_64 | DeepStream 9.1 SDK Debian package for x86 dGPU. |
| `deepstream-9.1_9.1.0-1_arm64.deb` | aarch64 | DeepStream 9.1 SDK Debian package for Jetson. |
| `deepstream-binaries-x86_9.1.0_amd64.deb` | x86_64 | Prebuilt proprietary runtime libraries for x86. |
| `deepstream-binaries-aarch64_9.1.0_arm64.deb` | aarch64 | Prebuilt proprietary runtime libraries for Jetson. |
| `deepstream-sample-data_9.1.0.deb` | all | Sample models, configs, and video streams. |

### Tarball archives (`.tar.gz` / `.tbz2`)

| Asset | Architecture | Description |
|---|---|---|
| `deepstream_sdk_v9.1.0_x86_64.tbz2` | x86_64 | DeepStream 9.1 SDK tarball for x86 dGPU. |
| `deepstream_sdk_v9.1.0_jetson.tbz2` | aarch64 | DeepStream 9.1 SDK tarball for Jetson. |
| `deepstream-binaries-x86_9.1.0.tar.gz` | x86_64 | Prebuilt proprietary runtime libraries for x86. |
| `deepstream-binaries-aarch64_9.1.0.tar.gz` | aarch64 | Prebuilt proprietary runtime libraries for Jetson. |
| `deepstream-sample-data_9.1.0.tar.gz` | all | Sample models, configs, and video streams. |

### Python wheels (`.whl`)

| Asset | Architecture | Description |
|---|---|---|
| `deepstream_libraries-1.4-cp312-cp312-linux_x86_64.whl` | x86_64 | DeepStream Libraries Python wheel (CPython 3.12) for x86. |

### Download installer(s)

Release assets can be downloaded via `curl` or `wget` from the command line.

**curl**

```bash
curl -LO 'https://github.com/NVIDIA/DeepStream/releases/download/v9.1.0/deepstream-9.1_9.1.0-1_amd64.deb'
```

**wget**

```bash
wget --content-disposition 'https://github.com/NVIDIA/DeepStream/releases/download/v9.1.0/deepstream-9.1_9.1.0-1_amd64.deb'
```

> **NOTE:** Replace `deepstream-9.1_9.1.0-1_amd64.deb` with the name of the asset you want to download.

## Docker Images (NGC)

DeepStream Docker images are released through NGC. Browse and pull the images from the [DeepStream NGC container catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/-/containers/deepstream/-?_lr=1).

> **NOTE:** For `docker login` and `docker pull` command details, along with the full list of available image tags, refer to the [DeepStream NGC container catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/-/containers/deepstream/-?_lr=1).

# Contribution Guidelines
This project is currently not accepting contributions. The product roadmap is managed internally by NVIDIA.

## Security
- Vulnerability disclosure: File an issue at [Issues](https://github.com/NVIDIA/DeepStream/issues).
- Do not file public issues for security reports.

## Support
- Level: Maintained
- How to get help: [Issues](https://github.com/NVIDIA/DeepStream/issues) / [DeepStream SDK Developer Forum](https://forums.developer.nvidia.com/c/accelerated-computing/intelligent-video-analytics/deepstream-sdk/15)
- Filing a bug? Use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md) and include: platform (x86 / Jetson / SBSA), DeepStream version, install method (Debian / Docker / source), and the output of `scripts/print_env.sh`.

# Community

Join the DeepStream community to ask questions, share feedback, and report issues.

- [GitHub Issues](https://github.com/NVIDIA/DeepStream/issues)
- [GitHub Discussions](https://github.com/NVIDIA/DeepStream/discussions)
- [DeepStream SDK Developer Forum](https://forums.developer.nvidia.com/c/accelerated-computing/intelligent-video-analytics/deepstream-sdk/15)

# References

- [DeepStream NGC container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/deepstream) — prebuilt runtime image (also bundles Triton).
- [NVIDIA TAO Toolkit](https://developer.nvidia.com/tao-toolkit) — model training/optimization, source of the TAO sample apps.
- [GStreamer documentation](https://gstreamer.freedesktop.org/documentation/) — underlying framework.

# License
This project is licensed under [CC-BY-4.0 AND Apache-2.0](./LICENSE).
- SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0

### Release Asset License

All GitHub Release assets (`.deb` packages and `.tar.gz` tarballs under the `deepstream-binaries-*` and `deepstream-sample-data_*` names) are governed by the [Software License Agreement For NVIDIA Software Development Kits](https://developer.download.nvidia.com/assets/Deepstream/DeepStream_6.0/NVIDIA_DeepStream_SDK_EULA_6.0.pdf), refer the "LicenseAgreement" included within all GitHub Release Assets packages.

## Third‑Party License/Notice

Refer [THIRD_PARTY_LICENSES](./THIRD_PARTY_LICENSES.txt) for all 3rd Party OSS licenses and Notices

