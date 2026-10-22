# DeepStream

[NVIDIA DeepStream SDK](https://developer.nvidia.com/deepstream-sdk) is a streaming analytics toolkit for AI-based video and image understanding, providing a GStreamer-based framework to build multi-stream, multi-model inference pipelines on NVIDIA GPUs (dGPU and Jetson).

DeepStream pipelines combine hardware-accelerated decoding/encoding, TensorRT inference, object tracking, and message-broker integrations to deliver real-time video analytics across dGPU and Jetson platforms.

DeepStream is published as a unified GitHub repository; sources are to be used under the project [LICENSE](LICENSE) (documentation: CC-BY-4.0; source code: Apache-2.0). GitHub Release assets specific license details can be referred at [Release Asset License](#release-asset-license) under [DeepStream Release Assets](#deepstream-release-assets). The project is actively maintained, but currently not accepting code contributions — refer to the [Contribution Guidelines](#contribution-guidelines) for more details.

# Overview

This repository contains the complete source code for DeepStream 9.2.

**Components** ([`src/`](src/README.md)):
- [`src/gst-plugins/`](src/README.md#gstreamer-plugins) — DeepStream GStreamer plugin sources
- [`src/gst-utils/`](src/README.md#gstreamer-utilities) — GStreamer utility library sources
- [`src/utils/`](src/README.md#utility-libraries) — utility library sources
- [`src/apps/sample_apps/`](src/README.md#sample-applications) — GStreamer-based sample applications
- [`src/apps/reference_apps/`](src/README.md#reference-applications) — advanced reference applications
- [`src/apps/tao_apps/`](src/README.md#tao-apps) — TAO-model integration apps
- [`src/service-maker/`](src/README.md#service-maker) — Service Maker C++/Python SDK and apps
- [`src/3rdparty/`](src/3rdparty/) — x86-only vendored sources (`gst-nvunixfd`, `gst-v4l2`, `v4l2`) built by `build/build.sh`

**Tools** ([`tools/`](tools/)):
- [`inference_builder`](tools/inference_builder/README.md) — visual inference pipeline builder
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
- [`deepstream-setup-on-wsl`](skills/deepstream-setup-on-wsl/SKILL.md) — set up and verify DeepStream in Docker inside WSL2
- [`inference-builder`](skills/inference-builder/SKILL.md) — generate deployable GPU inference pipelines from declarative configs
- [`deepstream-eval-and-finetune`](skills/deepstream-eval-and-finetune/SKILL.md) — evaluate and improve object detectors for DeepStream
- [`amc-setup-calibration-stack`](skills/amc-setup-calibration-stack/SKILL.md) — launch the AutoMagicCalib microservice and UI
- [`amc-run-sample-calibration`](skills/amc-run-sample-calibration/SKILL.md) — verify AMC with the bundled sample dataset
- [`amc-run-video-calibration`](skills/amc-run-video-calibration/SKILL.md) — calibrate user-provided MP4 camera videos
- [`amc-run-rtsp-calibration`](skills/amc-run-rtsp-calibration/SKILL.md) — calibrate live RTSP camera streams through VIOS capture
- [`rtvi-cv-customize-model`](skills/rtvi-cv-customize-model/SKILL.md) — swap the CV detection model in RTVI-CV
- [`rtvi-vlm-customize-model`](skills/rtvi-vlm-customize-model/SKILL.md) — swap the VLM in the VSS Alerts Blueprint
- [`rtvi-cv-scaffold-vss-service`](skills/rtvi-cv-scaffold-vss-service/SKILL.md) — scaffold a custom RTVI CV microservice that publishes to VSS via Kafka

The Brev launchable workflow is available in [`deploy/brev/`](deploy/brev/README.md).

# Requirements

Before building, ensure the following prerequisites are installed:

- **NVIDIA compute stack** — driver, CUDA, cuDNN, and TensorRT at the versions listed below. See the [DeepStream SDK Installation Guide](https://docs.nvidia.com/deepstream/9.2/text/DS_Installation).
- **DeepStream 9.2 proprietary runtime** — the prebuilt libraries and sample data are published as GitHub Release assets. `bash build/build.sh` automatically downloads the packages into `artifacts/` before the build process starts.

  > **Note:** After a successful build, `artifacts/` is removed automatically. Pass `--keep-assets` to retain them.

> **SBSA / DGX Spark:** DeepStream bare-metal installation is not supported on SBSA. Build and validation must be done inside the NVIDIA SBSA Docker container, which ships with DeepStream pre-installed — `build/build.sh` is **not required** inside the container. If you modify an open-source component, run `make && make install` in that component's directory to deploy the updated binaries/libs. See [build/BUILD.md](build/BUILD.md).

# Getting Started

```bash
# Install Git LFS (required for LFS-tracked sample media in reference apps)
sudo apt-get install git-lfs && git lfs install

# Clone the repo with submodules
git clone --recurse-submodules https://github.com/NVIDIA/DeepStream.git && cd DeepStream
```

See **[build/BUILD.md](build/BUILD.md)** for full build instructions, including system package dependencies (x86, aarch64, SBSA / DGX Spark), `build/build.sh` usage and environment variables (`CUDA_VER`, `NVDS_VERSION`, `INSTALL_METHOD`), and build output locations under `/opt/nvidia/deepstream/deepstream-9.2/`.

See [`src/README.md`](src/README.md) for component-specific READMEs, prerequisites, and platform limitations (on SBSA / DGX Spark only the source-available components are built), and [`CHANGELOG.md`](CHANGELOG.md) for the complete 9.2.0 release notes.

**Supported Platforms**

| Platform | Architecture | Notes |
|---|---|---|
| x86 dGPU | x86_64 | Ubuntu 24.04, CUDA 13.2, TensorRT 10.16.x, driver 595+ |
| Jetson | aarch64 | JetPack 7.2.1 (CUDA 13.2, TensorRT 10.16.x) |
| IGX Thor | aarch64 | IGX-SW 2.0 GA (CUDA 13.0, TensorRT 10.13.x, GPU driver 580.00) |
| SBSA / DGX Spark | aarch64 | Bare-metal DS installation not supported; build and run inside the NVIDIA SBSA Docker container. Only open-source components are built — no artifacts installed. |

```bash
# Build. The script prompts for sudo only when installing to system paths.
bash build/build.sh
```

# Usage

After `bash build/build.sh`, binaries are installed to `/opt/nvidia/deepstream/deepstream-9.2/bin/`. Run the reference `deepstream-app` with one of the sample configs:

```bash
cd /opt/nvidia/deepstream/deepstream-9.2/samples/configs/deepstream-app
deepstream-app -c source30_1080p_dec_infer-resnet_tiled_display.txt

# After the first install, clear the GStreamer plugin cache if needed:
rm -rf ~/.cache/gstreamer-1.0/
```

- More such examples: [DeepStream Quickstart Guide](https://docs.nvidia.com/deepstream/9.2/text/DS_Quickstart#run-deepstream-app-the-reference-application)
- Detailed API reference: [DeepStream SDK Developer Guide](https://docs.nvidia.com/deepstream/9.2)

Each app must be run from its source directory so relative config paths resolve correctly. Refer to the `README` inside each app directory for app-specific run instructions and config options.

## Running with Triton Inference Server (Docker)

> **Optional.** Skip this if the bare-metal build above already works for you. Use the Triton container when you need Triton-backed inference. On SBSA / DGX Spark, use the SBSA Docker container as described above — bare-metal installation is not supported.

NVIDIA publishes a DeepStream Docker image bundled with Triton Inference Server.

```bash
# Pull the image (use 9.2-triton-sbsa-dgx-spark instead on SBSA / DGX Spark)
docker pull ghcr.io/nvidia/deepstream:9.2-triton-multiarch

# Launch with display (use 'fakesink' in your pipeline for headless)
export DISPLAY=:0 && xhost +
docker run -it --rm --gpus all --network=host \
    -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix \
    ghcr.io/nvidia/deepstream:9.2-triton-multiarch

# Inside the container, run a Triton-backed sample
cd /opt/nvidia/deepstream/deepstream-9.2/samples/configs/deepstream-app-triton
deepstream-app -c source30_1080p_dec_infer-resnet_tiled_display.txt
```

**Prerequisites:** Docker (`docker-ce`), the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html), and NVIDIA driver 595+. Triton sample model repos ship under `/opt/nvidia/deepstream/deepstream/samples/triton_model_repo/`. For gRPC-backed Triton, use `samples/configs/deepstream-app-triton-grpc/` instead.

## Running with Triton Inference Server on Jetson (bare metal)

Prepare Triton dependencies and sample models on Jetson.

> **Building `gst-nvinferserver` standalone?** `bash build/build.sh` sets up the Triton client SDK under `/opt/tritonclient` automatically. If you instead build the plugin standalone (bypassing `build/build.sh`) — including rebuilding it from the `sources/` tree shipped inside a [`build/package.sh`](build/BUILD.md)-produced `.deb`/tarball, which does not bundle the Triton client SDK — set up the Triton client SDK yourself first. See the "Other dependencies" section in [`src/gst-plugins/gst-nvinferserver/README`](src/gst-plugins/gst-nvinferserver/README).

```bash
# Run from the cloned repository root after the DeepStream build has completed.

# Installs the Jetson-compatible Triton server libraries and backends.
sudo ./scripts/triton_backend_setup.sh

# Clear the plugin cache after the Triton libraries are installed.
rm -rf ~/.cache/gstreamer-1.0/

# Generate TensorRT engines and prepare the Triton model repository.
./scripts/prepare_ds_triton_model_repo.sh

# Run the local nvinferserver sample.
cd /opt/nvidia/deepstream/deepstream-9.2/samples/configs/deepstream-app-triton
deepstream-app -c source30_1080p_dec_infer-resnet_tiled_display.txt
```

> **Jetson only.** `triton_backend_setup.sh` downloads and installs the
> Jetson-compatible Triton components under `/opt`. For remote Triton inference
> over gRPC, use `samples/configs/deepstream-app-triton-grpc/` and start
> `tritonserver` separately.

# Documentation

| Page | Description |
|------|-------------|
| [Overview & Architecture](https://docs.nvidia.com/deepstream/9.2/text/DS_Overview) | What DeepStream is, key features, and the GStreamer-based pipeline architecture. |
| [Release Notes](https://docs.nvidia.com/deepstream/9.2/text/DS_Release_notes) | What's new in this release, supported platforms, and known issues. |
| [Installation](https://docs.nvidia.com/deepstream/9.2/text/DS_Installation) | Step-by-step installation of the NVIDIA compute stack and DeepStream SDK on x86 and Jetson. |
| [Docker Containers](https://docs.nvidia.com/deepstream/9.2/text/DS_docker_containers) | Available DeepStream container images (incl. Triton variants) and how to pull and run them. |
| [DeepStream Samples](https://docs.nvidia.com/deepstream/9.2/text/DS_C_Sample_Apps) | Reference walkthroughs of the bundled C/C++ sample applications and what each one demonstrates. |
| [Reference Applications](https://docs.nvidia.com/deepstream/9.2/text/DS_ref_app_github) | Advanced GitHub-hosted reference apps demonstrating specialized end-to-end pipelines. |
| [TAO Apps](https://docs.nvidia.com/deepstream/9.2/text/DS_TAO_integration) | DeepStream sample apps integrating TAO-trained models (detection, classification, segmentation, pose, LPR). |
| [DeepStream Plugins](https://docs.nvidia.com/deepstream/9.2/text/DS_plugin_Intro) | Reference for every DeepStream GStreamer plugin — properties, pad caps, and usage. |
| [Service Maker](https://docs.nvidia.com/deepstream/9.2/text/DS_service_maker_intro) | Build DeepStream pipelines declaratively with the C++ / Python Service Maker SDK. |
| [Inference Builder](tools/inference_builder/README.md) | Compose and configure DeepStream inference pipelines visually with the Inference Builder tool. |
| [Auto Magic Calib](https://github.com/NVIDIA-AI-IOT/auto-magic-calib/tree/bc5b1d9b3154d4242f147c35754670bc7981d37a) | Automatic camera calibration tool for multi-camera DeepStream deployments. |
| [DeepStream Libraries](https://docs.nvidia.com/deepstream/9.2/text/DS_Libraries) | DeepStream utility and helper libraries bundled in this repository at `deepstream_libraries/`. |
| [DeepStream Coding Agent](https://docs.nvidia.com/deepstream/9.2/text/DS_AI_Agent) | Use the bundled `skills/` with Claude Code and other AI coding assistants to generate DeepStream pipelines. |

# Repository Structure

```
DeepStream/
├── .claude/                                 # Claude Code project-local config (settings, commands)
├── .github/                                 # GitHub workflows, issue templates, CODEOWNERS
├── build/
│   ├── BUILD.md                             # build instructions
│   └── build.sh                             # top-level build driver
├── example_prompts/                         # example prompts for AI coding agents
├── deploy/brev/                              # Brev launchable workflow for DeepStream Code Agent
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
│   ├── rtvi-cv-customize-model/              # RTVI-CV model customization skill
│   ├── rtvi-vlm-customize-model/             # RTVI VLM model customization skill
│   └── rtvi-cv-scaffold-vss-service/         # RTVI CV VSS-service scaffolding skill
├── src/
│   ├── 3rdparty/                            # x86-only vendored gst-nvunixfd, gst-v4l2, v4l2
│   ├── apps/                                # sample, reference, and TAO sample applications
│   ├── gst-plugins/                         # GStreamer plugin sources (per-plugin subdirs)
│   ├── gst-utils/                           # GStreamer utility library sources
│   ├── service-maker/                       # Service Maker C++/Python SDK and apps
│   └── utils/                               # utility library sources (per-library subdirs)
├── deepstream_libraries/                    # DeepStream utility and helper libraries
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



## Git Submodules

| Submodule | Path | License |
| --------- | ---- | ------- |
| [NVIDIA-Optical-Character-Detection-and-Recognition-Solution (nvocdr)](https://github.com/NVIDIA-AI-IOT/NVIDIA-Optical-Character-Detection-and-Recognition-Solution) | [src/apps/tao_apps/apps/tao_others/deepstream-nvocdr-app/nvocdr/](src/apps/tao_apps/apps/tao_others/deepstream-nvocdr-app/nvocdr/) | [NVIDIA OCDR Solution Software License](src/apps/tao_apps/apps/tao_others/deepstream-nvocdr-app/nvocdr/LICENSE) |

# Performance
Summary of benchmarks; for detailed performance numbers and hardware used, refer to the [DeepStream Performance Guide](https://docs.nvidia.com/deepstream/9.2/text/DS_Performance).

# Releases & Roadmap

- **Releases / Changelog:** [CHANGELOG.md](./CHANGELOG.md)

## DeepStream Release Assets

The following `.deb` and `.tar` artifacts are published as GitHub Release assets. `build/build.sh` downloads them automatically into `artifacts/` before installation:

### Debian packages (`.deb`)

| Asset | Architecture | Description |
|---|---|---|
| `deepstream-9.2_9.2.0-1_amd64.deb` | x86_64 | DeepStream 9.2 SDK Debian package for x86 dGPU. |
| `deepstream-9.2_9.2.0-1_arm64.deb` | aarch64 | DeepStream 9.2 SDK Debian package for Jetson. |
| `deepstream-binaries-x86_9.2.0_amd64.deb` | x86_64 | Prebuilt proprietary runtime libraries for x86. |
| `deepstream-binaries-aarch64_9.2.0_arm64.deb` | aarch64 | Prebuilt proprietary runtime libraries for Jetson. |
| `deepstream-sample-data_9.2.0.deb` | all | Sample models, configs, and video streams. |

### Tarball archives (`.tar.gz` / `.tbz2`)

| Asset | Architecture | Description |
|---|---|---|
| `deepstream_sdk_v9.2.0_x86_64.tbz2` | x86_64 | DeepStream 9.2 SDK tarball for x86 dGPU. |
| `deepstream_sdk_v9.2.0_jetson.tbz2` | aarch64 | DeepStream 9.2 SDK tarball for Jetson. |
| `deepstream_sdk_v9.2.0_sbsa.tbz2` | aarch64 (SBSA) | DeepStream 9.2 SDK tarball for SBSA. Tarball only (no deb). |
| `deepstream-binaries-x86_9.2.0.tar.gz` | x86_64 | Prebuilt proprietary runtime libraries for x86. |
| `deepstream-binaries-aarch64_9.2.0.tar.gz` | aarch64 | Prebuilt proprietary runtime libraries for Jetson. |
| `deepstream-sample-data_9.2.0.tar.gz` | all | Sample models, configs, and video streams. |

### Python wheels (`.whl`)

| Asset | Architecture | Description |
|---|---|---|
| `deepstream_libraries-1.4-cp312-cp312-linux_x86_64.whl` | x86_64 | DeepStream Libraries Python wheel (CPython 3.12) for x86. |

### Download installer(s)

Release assets can be downloaded via `curl` or `wget` from the command line.

**curl**

```bash
curl -LO 'https://github.com/NVIDIA/DeepStream/releases/download/v9.2.0/deepstream-9.2_9.2.0-1_amd64.deb'
```

**wget**

```bash
wget --content-disposition 'https://github.com/NVIDIA/DeepStream/releases/download/v9.2.0/deepstream-9.2_9.2.0-1_amd64.deb'
```

> **NOTE:** Replace `deepstream-9.2_9.2.0-1_amd64.deb` with the name of the asset you want to download.

## Docker Images

DeepStream Docker images are available from two registries, depending on the release:

- DeepStream 9.2.0 and future releases: [GHCR](https://github.com/orgs/NVIDIA/packages/container/package/deepstream) (`ghcr.io/nvidia/deepstream`)
- DeepStream 9.1.0 and previous releases: [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/deepstream)

For example, to pull the DeepStream 9.2 Triton image from GHCR:

```bash
docker pull ghcr.io/nvidia/deepstream:9.2-triton-multiarch
```

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

- DeepStream container — prebuilt runtime image (also bundles Triton).
  - DeepStream 9.2.0 and future releases: [GHCR](https://github.com/orgs/NVIDIA/packages/container/package/deepstream)
  - DeepStream 9.1.0 and previous releases: [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/deepstream)
- [NVIDIA TAO Toolkit](https://developer.nvidia.com/tao-toolkit) — model training/optimization, source of the TAO sample apps.
- [GStreamer documentation](https://gstreamer.freedesktop.org/documentation/) — underlying framework.

# License
This project is licensed under [CC-BY-4.0 AND Apache-2.0](./LICENSE).
- SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0

The CC-BY-4.0 AND Apache-2.0 project license is not applicable to [`src/3rdparty/`](src/3rdparty/). Those vendored sources keep their own licenses as listed below.

### Release Asset License

All GitHub Release assets (`.deb` packages and `.tar.gz` tarballs under the `deepstream-binaries-*` and `deepstream-sample-data_*` names) are governed by the [Software License Agreement For NVIDIA Software Development Kits](https://developer.download.nvidia.com/assets/Deepstream/DeepStream_6.0/NVIDIA_DeepStream_SDK_EULA_6.0.pdf), refer the "LicenseAgreement" included within all GitHub Release Assets packages.

## Third‑Party License/Notice

Refer [THIRD_PARTY_LICENSES](./THIRD_PARTY_LICENSES.txt) for all 3rd Party OSS licenses and Notices

Vendored sources under [`src/3rdparty/`](src/3rdparty/) (x86 only) keep their upstream licenses:

| Path | License |
| ---- | ------- |
| [src/3rdparty/gst-nvunixfd/](src/3rdparty/gst-nvunixfd/) | LGPL-2.1-only ([LICENSE](src/3rdparty/gst-nvunixfd/LICENSE)); NVIDIA-authored files (`Makefile`, `gstnvipcbufferpool.*`) are Apache-2.0 |
| [src/3rdparty/gst-v4l2/](src/3rdparty/gst-v4l2/) | LGPL-2.0 ([LICENSE.gst-nvvideo4linux2](src/3rdparty/gst-v4l2/LICENSE.gst-nvvideo4linux2)); NVIDIA-authored files (`Makefile`, `README.txt`, `nvhelper_dyn.*`, `wsl_utils.*`) are Apache-2.0; `nalutils.*` and `gstv4l2h26xparser.*` are derived from GStreamer `gst-plugins-bad` codecparsers (LGPL-2.0-or-later) |
| [src/3rdparty/v4l2/](src/3rdparty/v4l2/) | LGPL-2.1-or-later (`libv4l2` / `libv4lconvert` from v4l-utils); `libv4lconvert` bundles TinyJPEG (BSD-3-Clause) and an IDCT from the Independent JPEG Group (IJG), so this software is based in part on the work of the Independent JPEG Group (notices in `THIRD_PARTY_LICENSES.txt`); NVIDIA-authored `README` files are Apache-2.0 |

A few component subdirectories retain their own vendored/upstream `LICENSE` file (in addition to being covered by `THIRD_PARTY_LICENSES.txt` and/or the root [`LICENSE`](./LICENSE)); these are kept in place for build/CI tooling that expects a local license file at that path:

| Path | License |
| ---- | ------- |
| [src/3rdparty/gst-nvunixfd/LICENSE](src/3rdparty/gst-nvunixfd/LICENSE) | LGPL-2.1-only (vendored GStreamer unixfd; also listed in `THIRD_PARTY_LICENSES.txt`) |
| [src/3rdparty/gst-v4l2/LICENSE.gst-nvvideo4linux2](src/3rdparty/gst-v4l2/LICENSE.gst-nvvideo4linux2) | LGPL-2.0 (vendored GStreamer V4L2; also listed in `THIRD_PARTY_LICENSES.txt`) |
| [tools/inference_builder/dependencies/triton-inference-server/common/LICENSE](tools/inference_builder/dependencies/triton-inference-server/common/LICENSE) | BSD-3-Clause (vendored from [triton-inference-server/common](https://github.com/triton-inference-server/common); also listed in `THIRD_PARTY_LICENSES.txt`) |
| [skills/deepstream-sop/references/license.txt](skills/deepstream-sop/references/license.txt) | Apache-2.0 (copy of the root project license, kept local for the SOP microservice's own packaging needs) |
| [tools/auto-magic-calib/LICENSE](tools/auto-magic-calib/LICENSE) | Apache-2.0 (copy of the root project license) |
| [tools/auto-magic-calib/assets/LICENSE](tools/auto-magic-calib/assets/LICENSE) | CC-BY-4.0 (copy of the root project license, scoped to the `assets/` directory) |
| [src/utils/ds3d/3rdparty/realsense2/LICENSE](src/utils/ds3d/3rdparty/realsense2/LICENSE) | Apache-2.0, Copyright 2017 Intel Corporation (vendored Intel RealSense SDK `librealsense2` headers used by the DS3D RealSense dataloader; also listed in `THIRD_PARTY_LICENSES.txt`) |
