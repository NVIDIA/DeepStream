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

# DeepStream — Build & Run Guide

## Prerequisites

### NVIDIA Compute Stack

Before building from source, the NVIDIA compute stack must be installed on the host. Required versions:

| Platform | OS | Driver | CUDA | cuDNN | TensorRT |
|---|---|---|---|---|---|
| x86 dGPU | Ubuntu 24.04 | 595.58.03+ | 13.2 | 9.20.0.48 | 10.16.x |
| Jetson (aarch64) | JetPack 7.2.1 | — (bundled) | 13.2 | 9.20.0.48 | 10.16.x |
| IGX Thor (aarch64) | IGX-SW 2.0 GA | — (bundled) | 13.0 | 9.12.0.46 | 10.13.x |
| SBSA / DGX Spark | Ubuntu 24.04 | 595.58.03+ | 13.2 | 9.20.0.48 | 10.16.x |

For installation instructions, refer to the official [DeepStream SDK Installation Guide](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Installation.html).

> **Docker users:** NVIDIA NGC provides pre-built DeepStream 9.1 containers with the full compute stack pre-installed. If using a container, skip this section and proceed directly to the [Build and Install](#build-and-install) section below (after cloning the repo and running `git lfs install`).

---

### Install cmake and wget

```bash
sudo apt-get install cmake wget
```

### x86 (dGPU)

The DeepStream proprietary runtime libraries and sample data are downloaded from
the [DeepStream GitHub release](https://github.com/NVIDIA/DeepStream/releases) into
`artifacts/` and installed automatically by the `artifacts` stage of `build/build.sh`.

1. Install system dependencies:
   ```bash
   sudo apt update && sudo apt install \
     libssl3 \
     libssl-dev \
     libcurl4-openssl-dev \
     libgles2-mesa-dev \
     libgstreamer1.0-0 \
     gstreamer1.0-tools \
     gstreamer1.0-plugins-good \
     gstreamer1.0-plugins-bad \
     gstreamer1.0-plugins-ugly \
     gstreamer1.0-libav \
     libgstreamer-plugins-base1.0-dev \
     libgstrtspserver-1.0-0 \
     libjansson4 \
     libjsoncpp-dev \
     libyaml-cpp-dev \
     libmosquitto1 \
     python3-dev
   ```

   See also the [DeepStream Installation Guide — Install Prerequisite Packages](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Installation.html#id2).

### aarch64 (Jetson)

The DeepStream proprietary runtime libraries and sample data are downloaded from
the [DeepStream GitHub release](https://github.com/NVIDIA/DeepStream/releases) into
`artifacts/` and installed automatically by the `artifacts` stage of `build/build.sh`.

1. Install system dependencies:
   ```bash
   sudo apt update && sudo apt install \
     libssl3 \
     libssl-dev \
     libcurl4-openssl-dev \
     libgstreamer1.0-0 \
     gstreamer1.0-tools \
     gstreamer1.0-plugins-good \
     gstreamer1.0-plugins-bad \
     gstreamer1.0-plugins-ugly \
     gstreamer1.0-libav \
     libgstreamer-plugins-base1.0-dev \
     libgstrtspserver-1.0-0 \
     libjansson4 \
     libjsoncpp-dev \
     libyaml-cpp-dev \
     libmosquitto1 \
     python3-dev
   ```

   See also the [DeepStream Installation Guide — Install Prerequisite Packages](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Installation.html#install-prerequisite-packages).

### SBSA / DGX Spark (Server Base System Architecture)

SBSA and DGX Spark share the same code path. **DeepStream bare-metal installation
is not supported on SBSA.** There is no standalone DS tar/deb package for this
platform. Build and validation must be done **inside the NVIDIA SBSA Docker
container**, which ships with DeepStream pre-installed.

When `bash build/build.sh` is run inside the SBSA container, the `artifacts`
stage is skipped automatically (no SBSA artifacts are shipped in this repository).
Only the open-source components are built and installed on top of the
Docker-bundled DeepStream installation.

> **Note:** `build/build.sh` is **not required** inside the Docker container.
> DeepStream is already fully installed in the OOB image. If you modify an
> open-source component, run `make && make install` in that component's directory
> to compile and deploy the updated binaries/libs directly.

1. Start the SBSA Docker container (with GPU access):
   ```bash
   sudo docker run -it --rm --runtime=nvidia --network=host \
       -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video,graphics \
       --gpus all \
       --privileged \
       -e DISPLAY=:0 \
       -v /tmp/.X11-unix:/tmp/.X11-unix \
       -v /etc/X11:/etc/X11 \
       <sbsa-docker-image> bash
   ```

   > **Note:** `--privileged` grants near-root host access to all host devices and low-level driver operations. On shared or production systems, prefer `--device` / `--cap-add` instead — use `--privileged` only if that level of access is acceptable.

2. Install build prerequisites inside the container:
   ```bash
   apt-get update && apt-get install -y git git-lfs cmake make build-essential
   ```

3. Clone the repository (with submodules):
   ```bash
   git clone --recurse-submodules https://github.com/NVIDIA/DeepStream.git
   cd DeepStream
   ```

#### Components that must be built and run on the host (baremetal)

The SBSA Docker flow above does **not** cover every component. The following components are not validated inside the container and must instead be built and run from a separate clone of the repository on the baremetal SBSA / DGX Spark host:

- `tools/inference_builder` — see [`tools/inference_builder/README.md`](../tools/inference_builder/README.md)
- `src/apps/reference_apps/deepstream-tracker-3d-multi-view` — see [`src/apps/reference_apps/deepstream-tracker-3d-multi-view/README.md`](../src/apps/reference_apps/deepstream-tracker-3d-multi-view/README.md)

Clone the repository a second time on the host (outside any container) and follow each component's own README for its build and run steps.

---

### RTSP streams — rtpjitterbuffer fix

With RTSP streams the application can get stuck on reaching EOS. This is caused by an issue in the `rtpjitterbuffer` component. To fix it, on baremetal run the [`scripts/rtpmanager/update_rtpmanager.sh`](../scripts/rtpmanager/update_rtpmanager.sh) script:

```bash
sudo bash scripts/rtpmanager/update_rtpmanager.sh
```

Run this script once, **after** installing the system dependencies.

---

## Build and Install

> **Release assets are downloaded automatically (x86 and Jetson only).**
> The prebuilt runtime assets are hosted on the
> [DeepStream GitHub release](https://github.com/NVIDIA/DeepStream/releases). The
> `artifacts` stage of `build/build.sh` downloads (via `wget`) only the assets
> for the detected host platform and the selected install method into
> `artifacts/` (the directory is created if missing), then installs them. By
> default these downloaded assets are **deleted once the build succeeds**; pass
> `--keep-assets` to retain them.

> **Note for Jetson Orin users:** External storage integration is required to support the build and execution of DeepStream on the Jetson Orin platform. The downloaded artifacts may exhaust internal storage and cause `No space left on device` errors without it.

`build/build.sh` auto-detects the host platform and runs the following stages in order:

1. **artifacts** — downloads the prebuilt proprietary libs and sample data from the [DeepStream GitHub release](https://github.com/NVIDIA/DeepStream/releases) into `artifacts/`, then installs them to `/opt/nvidia/deepstream/deepstream-<NVDS_VERSION>/` via [`scripts/install_artifacts.sh`](../scripts/install_artifacts.sh). Only the assets for the selected install method are downloaded. By default the downloaded assets are removed after a successful build (use `--keep-assets` to retain them).
   - Default method: `deb` — downloads and installs `deepstream-binaries-<platform>_<arch>.deb` and `deepstream-sample-data_*.deb` via dpkg.
   - `--install-method=tar` — downloads and extracts the equivalent tarballs (`deepstream-binaries-<platform>_*.tar.gz`, `deepstream-sample-data_*.tar`).
2. **deps** — builds and installs open-source dependencies (OpenTelemetry, civetweb, …) via [`scripts/install_opensource_deps.sh`](../scripts/install_opensource_deps.sh).
3. **source stages** — builds all source components in order: `gst-utils`, `utils`, `gst-plugins`, `sample_apps`, `yolo` (YOLO custom inference lib in `tools/yolo_deepstream/`), `tao_apps`, `reference_apps`, `service-maker`, and installs them to `/opt/nvidia/deepstream/deepstream-<NVDS_VERSION>/`. The `service-maker` stage builds the runtime core (`libnvds_service_maker.so` and `libnvds_service_maker_utils.a`) and the `ds-launch` engine first, then the sample apps and loadable modules that link against them. Last it builds the `pyservicemaker` wheel from `sources/python/` and copies it over the prebuilt wheel that the `artifacts` stage placed under `service-maker/python/`, so the bindings match the core they were built against.
4. **install.sh** — registers binaries via `update-alternatives` and pip-installs the `pyservicemaker` wheel from `service-maker/python/`. Skipped for scoped `--only=` builds; run a full build to finalize system integration.

Additional behavior:
- On SBSA / DGX Spark: passes `AARCH64_IS_SBSA=1` to all Makefiles (both `aarch64` and `sbsa` report `uname -m` as `aarch64`; SBSA is identified by the absence of `/etc/nv_tegra_release`); the artifacts stage is skipped automatically
- Prompts for `sudo` only when a step must install packages or write to system paths
- Tracks completed stages in `build/.stage-state` (build stages), `build/.stage-state.deps` (dependency sub-stages), and `build/.stage-state.artifacts` (artifact sub-stages) so builds can be resumed after interruption

From the repo root:

```bash
bash build/build.sh
```

> **Note:** `build/build.sh` does **not** build `gst-nvdsudp` or `gst-dsexample-cuda`. Compile and install those plugins separately by following their READMEs:
>
> - [`src/gst-plugins/gst-nvdsudp/README`](../src/gst-plugins/gst-nvdsudp/README)
> - [`src/gst-plugins/gst-dsexample-cuda/README`](../src/gst-plugins/gst-dsexample-cuda/README)

> **Note:** Some sample apps are platform-specific and are skipped automatically depending on the detected `PLATFORM`:
>
> | Sample app | Built on | Skipped on |
> |---|---|---|
> | `deepstream-ucx-test` | `x86` only | `aarch64`, `sbsa` |
> | `deepstream-multigpu-nvlink-test` | `x86` only | `aarch64`, `sbsa` |
> | `deepstream-ipc-test` | `aarch64` (Jetson) only | `x86`, `sbsa` |
> | `deepstream-appsrc-cuda-test` | `x86`, `sbsa` only | `aarch64` |

Each run writes a full transcript to `build/build.log`. Output still appears on the terminal.

Show all options:

```bash
bash build/build.sh --help
```

### CLI reference

| Flag | Purpose |
|---|---|
| `--install-method=deb\|tar` | How to install prebuilt artifacts: `deb` (default, uses dpkg) or `tar` (extracts tarballs). Only the assets for the chosen method are downloaded |
| `--keep-assets` | Keep the downloaded release assets in `artifacts/` after a successful build (default: delete them once the build succeeds) |
| `--skip-artifacts` | Skip the artifact install stage (use when artifacts are already installed) |
| `--skip-deps` | Skip open-source dependency install (use when dependencies are already installed) |
| `--only=STAGE[,STAGE]` | Build only named stage(s): `gst-utils`, `utils`, `gst-plugins`, `sample_apps`, `tao_apps`, `reference_apps`, `service-maker` |
| `--resume` | Resume from the last successful stage |
| `--verbose` | Show sub-make stderr (no `2>/dev/null` suppression) |
| `-j N` | Parallel make/cmake jobs (default: `nproc`) |
| `--package` | After a full successful build, package the install tree into `build/` via [`build/package.sh`](package.sh) (emits both `.deb` and `.tar.gz`) |
| `--package-format=deb\|tar\|both` | Package format to emit (default: `both`); implies `--package`. Skipped for `--only=` scoped builds |
| `--deepstream-libraries-wheel[=DIR]` | Build only the DeepStream Libraries wheel. Output defaults to `artifacts/`; a relative `DIR` is resolved from the repository root. |
| `--deepstream-libraries-wheel-version=VERSION` | Build the DeepStream Libraries wheel at `VERSION` (default: `1.4`). |

Environment variables may also be passed on the command line: `CUDA_VER=13.2`, `NVDS_VERSION=9.1`, `NVDS_ARTIFACT_VERSION=9.1.1`, `CMAKE_BIN=/usr/bin/cmake`. To switch to tarball install: `INSTALL_METHOD=tar`.

### Examples

```bash
# Full build (default — installs artifacts via deb, then builds from source)
bash build/build.sh

# Use tarball install instead of deb
bash build/build.sh --install-method=tar

# Keep the downloaded release assets in artifacts/ after the build succeeds
bash build/build.sh --keep-assets

# Resume after interruption or partial failure
bash build/build.sh --resume

# Rebuild GStreamer plugins when artifacts and dependencies are already installed
bash build/build.sh --only=gst-plugins --skip-deps

# Resume a scoped build
bash build/build.sh --only=gst-plugins --resume --skip-deps

# Debug a failing sub-make with full stderr
bash build/build.sh --only=utils --skip-deps --verbose

# Rebuild service-maker only, 8-way parallel
bash build/build.sh --only=service-maker --skip-deps -j8

# Skip artifact install explicitly (artifacts already installed, do not reinstall)
bash build/build.sh --skip-artifacts --skip-deps

# Full build, then package into build/ (both .deb and .tar.gz)
bash build/build.sh --package

# Full build, then emit only a Debian package
bash build/build.sh --package-format=deb

# Build the DeepStream Libraries wheel in its default output directory
bash build/build.sh --deepstream-libraries-wheel

# Build the DeepStream Libraries wheel into artifacts/
bash build/build.sh --deepstream-libraries-wheel=artifacts

# Build version 1.4 of the DeepStream Libraries wheel
bash build/build.sh --deepstream-libraries-wheel-version=1.4
```

Default CUDA version is `13.2` (`13.0` for IGX Thor systems). Set `CUDA_VER` accordingly:

```bash
CUDA_VER=13.2 bash build/build.sh
```

To override the install-path version (default: `9.1`) or the GitHub Release
tag / asset-name version (default: `9.1.1`):

```bash
NVDS_VERSION=9.1 bash build/build.sh
NVDS_ARTIFACT_VERSION=9.1.1 bash build/build.sh
```

If a user-local `cmake` wrapper shadows the system CMake, either remove it from `PATH` or point the build script at a known-good binary:

```bash
CMAKE_BIN=/usr/bin/cmake bash build/build.sh
```

After first run, clear the GStreamer plugin cache so it picks up newly installed plugins:

```bash
rm -rf ~/.cache/gstreamer-1.0/
```

**Build outputs:**
- GStreamer plugins       → `/opt/nvidia/deepstream/deepstream-<NVDS_VERSION>/lib/gst-plugins/`
- Utility libs            → `/opt/nvidia/deepstream/deepstream-<NVDS_VERSION>/lib/`
- Sample app binaries     → `/opt/nvidia/deepstream/deepstream-<NVDS_VERSION>/bin/`
- service-maker app bins  → `/opt/nvidia/deepstream/deepstream-<NVDS_VERSION>/bin/service-maker-<app>`
- service-maker modules   → `/opt/nvidia/deepstream/deepstream-<NVDS_VERSION>/service-maker/modules/`

> **Note:** `/opt/nvidia/deepstream/deepstream-<NVDS_VERSION>/` is installed as root.
> Helper scripts under `scripts/` (for example `prepare_classification_test_video.sh`,
> `prepare_ds_triton_model_repo.sh`, `prepare_ds_triton_tao_model_repo.sh`) and other
> sample/build steps write into that tree. Either run those commands with `sudo`, or
> make the install tree writable for your user:
>
> ```bash
> sudo chmod -R 777 /opt/nvidia/deepstream/deepstream-<NVDS_VERSION>
> ```

---

## Uninstall

To remove a DeepStream installation from `/opt`:

```bash
# Remove DeepStream 9.1 (default)
sudo bash scripts/uninstall.sh

# Remove a specific version (e.g. 9.0)
sudo bash scripts/uninstall.sh 9.0
# or equivalently:
sudo PREV_DS_VER=9.0 bash scripts/uninstall.sh
```

`uninstall.sh` removes binaries, libs, samples, and service-maker files from
`/opt/nvidia/deepstream/deepstream-<version>/`, deregisters `update-alternatives`
entries, uninstalls the `pyservicemaker` Python package, deregisters Debian
packages that were installed via the `deb` method (dpkg -r), and clears the
GStreamer plugin cache. It does **not** affect other installed DeepStream
versions.

> **Note:** Do **not** run `uninstall.sh` inside the SBSA / DGX Spark Docker
> container. DeepStream is already pre-installed on this platform (the artifacts
> stage is skipped during the build — see [SBSA / DGX Spark](#sbsa--dgx-spark-server-base-system-architecture)),
> and running the uninstall script will remove the bundled DeepStream installation.

---

## service-maker — Local Compilation

Individual service-maker apps can be compiled locally without running `build/build.sh`.
From within any app directory (e.g. `src/service-maker/sources/apps/cpp/deepstream_test1_app/`):

```bash
mkdir build && cd build && cmake .. && make
```

The binary is placed in the local `build/` directory and nothing is installed to `/opt`.
See each app's `README` for usage instructions.

---

## Packaging (Debian package / tarball)

Once DeepStream has been built and installed to
`/opt/nvidia/deepstream/deepstream-<NVDS_VERSION>/`, you can bundle that install
tree into a redistributable **Debian package (`.deb`)** and/or **tarball
(`.tar.gz`)** using [`build/package.sh`](package.sh). The package installs to
`/opt/nvidia/deepstream/deepstream-<NVDS_VERSION>/` on the target machine and
ships the runtime (`bin/`, `lib/`, `samples/`, `service-maker/`) alongside the
repo sources (`includes/`, `sources/`, helper scripts).

> **Important — package only after a successful build.** `package.sh` copies
> from the already-built install tree at
> `/opt/nvidia/deepstream/deepstream-<NVDS_VERSION>/` (produced by
> `build/build.sh`). It does **not** build anything itself. Always run
> `build/build.sh` to completion **successfully** first; if the build failed or
> was partial, the resulting package will be incomplete or the script will error
> out on the missing install tree.

### Option 1 — package as part of the build (recommended)

Add `--package` (or `--package-format=`) to `build/build.sh`. Packaging runs
automatically at the very end, but **only** after a full build succeeds (it is
skipped for `--only=` scoped builds and never runs if any stage failed):

```bash
# Full build, then emit both .deb and .tar.gz into build/
bash build/build.sh --package

# Full build, then emit only the Debian package
bash build/build.sh --package-format=deb

# Full build, then emit only the tarball
bash build/build.sh --package-format=tar
```

### Option 2 — run `package.sh` separately

If DeepStream is already built and installed (a previous successful
`build/build.sh` run), you can invoke the packager directly:

```bash
bash build/package.sh
```

> **Prerequisite:** ensure `build/build.sh` has already completed successfully
> and that `/opt/nvidia/deepstream/deepstream-<NVDS_VERSION>/` exists with the
> full runtime. `package.sh` aborts with an error if the built install tree (or
> any of `bin/`, `lib/`, `samples/`, `service-maker/`, `LicenseAgreement.pdf`)
> is missing.

Show all packaging options:

```bash
bash build/package.sh --help
```

### `package.sh` CLI reference

| Flag | Purpose |
|---|---|
| `--format=deb\|tar\|both` | What to build (default: `both`) |
| `--arch=amd64\|arm64` | Target Debian architecture (default: auto-detected from host) |
| `--output-dir=PATH` | Where to write the `.deb` / `.tar.gz` (default: `build/`) |
| `--stage-dir=PATH` | Scratch staging directory (default: `build/.pkg-stage`) |
| `--keep-stage` | Do not delete the staging directory on exit (useful for inspection) |

Environment overrides: `NVDS_VERSION` (default `9.1.1`, must be `MAJOR.MINOR.PATCH`) and `PKG_VERSION`
(default: same as `NVDS_VERSION`).

### Output

Packages are written to the output directory (default `build/`) and named:

```text
deepstream-<NVDS_VERSION>_<PKG_VERSION>-<revision>_<arch>.deb
deepstream-<NVDS_VERSION>_<PKG_VERSION>-<revision>_<arch>.tar.gz
```

The generated `.deb` can then be installed on a target machine with:

```bash
sudo apt install ./deepstream-<NVDS_VERSION>_<PKG_VERSION>-1_<arch>.deb
```
