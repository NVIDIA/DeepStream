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

## TensorRT 11 upgrade branch

Source builds and packages in this branch default to **DeepStream 9.2.1**; the
install directory remains `deepstream-9.2`. Published release-asset pins remain
independent of this source version. A version bump does not rebuild or relabel
existing runtime binaries; supply a matching rebuilt runtime for an upgrade drop.

The x86 and generic-SBSA upgrade candidate follows Triton26.10's core versions:
CUDA **13.4.3**, TensorRT **11.4.0.106** and cuDNN **9.27.0.42**. Jetson retains
JetPack **7.2.2+**, CUDA **13.2.2**, TensorRT **11.4.1.31** and cuDNN
**9.20.0.46**. ARM uses VPI **4.1.6**; x86 retains VPI **4.1.3**. These are
declared build targets, not completed build/runtime qualification. IGX's separate
legacy stack is not upgraded here. See the [official26.10 contents](https://docs.nvidia.com/deeplearning/frameworks/cuda-dl-release-notes/rel-26-10.html).

`CUDA_VER` selects the toolkit **family directory** (`13.4` x86/SBSA, `13.2`
Jetson), not the full release. Install the full matching toolkit separately;
setting `CUDA_VER=13.4` does not prove that CUDA13.4.3 is installed. Explicit
environment or command-line `CUDA_VER` overrides are retained. TensorRT selection
is passed as `TRT_VER`; `CUDNN_VERSION` records the declared baseline only, not a
package installation or verification result. `pyservicemaker` remains **0.0.1**.
Explicit `VPI_DIR`, `TORCH_PATH`, `TRT_INCLUDE_DIR`, `TRT_LIB_DIR`,
`EIGEN_INCLUDE_DIR`, `JSON_INCLUDE_DIR` and `DEEPSTREAM_VARIANT` environment
overrides are forwarded as individual Make arguments for build and install,
including when `sudo` filters the invoking environment. ARM builds otherwise
pass the detected `tegra`, `sbsa` or unchanged legacy `igx` variant explicitly.

Use matching runtime binaries for the selected upgrade stack. Do not combine
these sources with opaque binaries from an older TensorRT stack merely because
the SDK version is also 9.2. A source build still needs all development packages
and component-specific prerequisites documented below.

For an engineering drop retaining **all** supplied SDK runtime binaries, first
extract the trusted complete SDK package into a private directory (this does not
install it):

```bash
mkdir -p /absolute/path/sdk-input
dpkg-deb -x /absolute/path/deepstream-9.2_9.2.1-1_amd64.deb /absolute/path/sdk-input
bash build/build.sh --package \
  --package-from=/absolute/path/sdk-input/opt/nvidia/deepstream/deepstream-9.2 \
  --package-platform=x86
```

This route skips downloads, dependency installation, SDK compilation and system
integration. It packages the supplied runtime plus the selected public sources;
it does not establish that those binaries were built from those sources. Verify
the package's build/source identity separately. Do not add proprietary binaries,
vendor snapshots, internal manifests or build caches to Git.

`--package-platform=jetson` and `--package-platform=sbsa` select ARM packaging;
they do not perform a cross-build. Runtime inference ELF architecture and
TensorRT ABI are checked before staging. This is not a complete ABI/runtime audit.
During local development only, add `--include-untracked-sources` to include new,
non-ignored files within the source allowlist. Normal packaging remains tracked-only.
Custom staging directories must be empty or carry this packager's matching
ownership sidecar. Repository sources, build scripts, runtime inputs and broad
system directories are never valid disposable staging locations.

The package includes conversion scripts under `samples/modelopt`. Conversion and
target engine building happen explicitly after installation; see
[`tools/modelopt/README.md`](../tools/modelopt/README.md). Original ONNX filenames
remain canonical; preserved originals use the `fp32_` prefix. No ModelOpt venv,
CUDA toolkit download cache or generated engine is added by these source rules.

## Prerequisites

### NVIDIA Compute Stack

Before building from source, provision the selected stack on the target or in
its supported container. Current upgrade targets are:

| Platform | OS | Driver | CUDA | cuDNN | TensorRT |
|---|---|---|---|---|---|
| x86 dGPU | Ubuntu 24.04 | 615.71.09+; qualify target driver | 13.4.3 | 9.27.0.42 | 11.4.0.106 |
| Jetson (aarch64) | JetPack7.2.2+ matching BSP | — (bundled) | 13.2.2 | 9.20.0.46 | 11.4.1.31 |
| IGX Thor (aarch64) | IGX-SW 2.0 GA | — (bundled) | 13.0 | 9.12.0.46 | 10.13.x |
| SBSA / DGX Spark | Ubuntu24.04-based matching DS container | 615.71.09+; qualify target driver | 13.4.3 | 9.27.0.42 | 11.4.0.106 |

For installation instructions, refer to the official [DeepStream SDK Installation Guide](https://docs.nvidia.com/deepstream/9.2/text/DS_Installation).

> **Docker users:** use a reviewed image matching the selected upgrade stack and
> SDK binaries. An older DeepStream9.2 image is not automatically compatible
> merely because the install directory still ends in `9.2`. Stock Triton26.10 is
> an x86/SBSA candidate, not a Jetson JP7.2.2/TRT11.4.1 provider. Containers still
> require a supported host driver/BSP.

### Matching Triton client SDK and VPI inputs

The dependency script no longer downloads the old x8626.03/Jetson26.04 client
SDKs for this upgrade. For x86/SBSA, preprovision the matching26.10 C++ SDK at
`/opt/tritonclient`, or supply `TRITON_CLIENT_SDK_ROOT=/path/to/sdk/install` to
`scripts/install_opensource_deps.sh`. The official26.10 SDK is supplied through
the `26.10-py3-sdk` container: its `/workspace/install/include` and
`/workspace/install/lib` are the inputs. No container is pulled by these scripts.
See the [Triton2.74.0 release](https://github.com/triton-inference-server/server/releases/tag/v2.74.0).
Jetson needs an explicitly qualified JP7.2.2/TRT11.4.1-compatible SDK instead;
do not substitute the stock SBSA SDK simply because both are ARM64.

Provider checks require C++ headers, static client dependencies (including
`libcares.a`), and `libgrpcclient.so` or `libgrpcclient.a` for the SDK's actual
`-lgrpcclient` link. A differently named `libgrpcclient_static.a` alone is not
sufficient. These checks do **not** verify every transitive static library,
release provenance or runtime compatibility. Preserve the SDK
image digest/release identity separately. A custom provider is copied into
`/opt/tritonclient` only when the dependency installer is explicitly run.

With `--skip-deps`, an explicit `TRITON_CLIENT_SDK_ROOT` is used directly for
linking and runtime registration instead. `TRITON_GRPC_CLIENT` is an explicit
consumer-root override and takes precedence. The build records the effective
provider in resume state and passes it through privileged build/install calls;
changing providers requires a fresh build state. A shared client remains an
external runtime dependency: creating an SDK tar does not copy `/opt/tritonclient`
into it. Provision the matching provider on the runtime system or in the image.

### Installation integration and existing installations

Installation preflights required targets and alternative-link ownership before
making changes. Missing optional sample applications are skipped; a conflicting
regular executable or unmanaged symlink produces an actionable error instead
of silently retaining an older app. Existing manual alternatives selections are
preserved with a notice. The known legacy UCX group is reused when it already
owns the link; uninstall removes only the selected SDK's registration.

For a version mismatch, inspect the canonical executable directly:

```bash
/opt/nvidia/deepstream/deepstream-9.2/bin/deepstream-app --version-all
```

Review package ownership before migrating an old `/usr/bin/deepstream-app`;
the installer does not force-overwrite it or change global CUDA selection.
The reported CUDA runtime is queried from the selected library, not inferred
from `CUDA_VER`. A matching compiler does not prove matching runtime selection.
ARM installation retains packaged SBSA media libraries and links missing media
only to existing platform providers; Jetson-only IPC registration is gated by
the Jetson release marker. These checks do not qualify a BSP or Triton provider.

An explicit `TORCH_PATH` is also preserved through dependency preparation: the
Sparse4D helper checks its CUDA-library layout, ABI1 evidence and native ELF
architecture before any APT operation, and does not replace it or install another
Torch wheel. Empty, missing, CPU-only, ABI0 or unknown providers fail explicitly.
These checks do not establish vendor provenance or qualify the runtime stack.
When `TORCH_PATH` is unset, the existing default PyTorch installation is retained.

ARM must already have genuine matching **VPI4.1.6** development headers at
`/opt/nvidia/vpi4` (or `VPI_DIR`). Missing/mismatched headers fail rather than
adding an x86 repository or guessing a vendor download. Provision the target
vendor VPI runtime independently. IGX keeps its separate legacy VPI stack.
The x86 installer keeps the **4.1.3** package family; if that version is absent
from the configured repository, supply the reviewed vendor package explicitly.

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

   See also the [DeepStream Installation Guide — Install Prerequisite Packages](https://docs.nvidia.com/deepstream/9.2/text/DS_Installation#install-prerequisite-packages).

2. Install the VPI development headers, which `src/utils/nvmultiobjecttracker`
   needs to build. They come from NVIDIA's public VPI repository (Ubuntu 24.04
   shown; use `jammy` for 22.04). `apt` also installs the `libnvvpi4` runtime
   package as a dependency.
   ```bash
   sudo apt install gnupg software-properties-common
   sudo apt-key adv --fetch-key https://repo.download.nvidia.com/jetson/jetson-ota-public.asc
   sudo add-apt-repository 'deb https://repo.download.nvidia.com/jetson/x86_64/noble r39.2 main'
   sudo apt update
   # Select the reviewed4.1.3 package revision from this repository.
   apt-cache madison vpi4-dev
   sudo apt install 'vpi4-dev=<reviewed-4.1.3-package-revision>'
   ```
   The headers install to `/opt/nvidia/vpi4/include`. Override the location with
   `VPI_DIR=<path>` when running `make` in `src/utils/nvmultiobjecttracker`.

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

   See also the [DeepStream Installation Guide — Install Prerequisite Packages](https://docs.nvidia.com/deepstream/9.2/text/DS_Installation#install-prerequisite-packages).

### SBSA / DGX Spark (Server Base System Architecture)

SBSA and DGX Spark share the same code path. **DeepStream bare-metal installation
is not supported on SBSA.** There is no standalone DS tar/deb package for this
platform. Build and validation must be done **inside the NVIDIA SBSA Docker
container**, which ships with the matching DeepStream runtime pre-installed.
The build/dependency drivers reject SBSA bare-metal installation. This does not
prevent the offline `--package-from` assembly of an explicitly supplied runtime.

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

> **Note:** `build/build.sh` does **not** build `gst-nvdsudp`, `gst-dsexample-cuda`, or `gst-nvdstextembedder`. Compile and install `gst-nvdsudp` and `gst-dsexample-cuda` separately by following their READMEs. `gst-nvdstextembedder` is a Python plugin and is not compiled.
>
> - [`src/gst-plugins/gst-nvdsudp/README`](../src/gst-plugins/gst-nvdsudp/README)
> - [`src/gst-plugins/gst-dsexample-cuda/README`](../src/gst-plugins/gst-dsexample-cuda/README)
> - [`src/gst-plugins/gst-nvdstextembedder/README.md`](../src/gst-plugins/gst-nvdstextembedder/README.md)

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

Environment variables may also be passed on the command line: `CUDA_VER=13.4`, `NVDS_VERSION=9.2.1`, `CMAKE_BIN=/usr/bin/cmake`. To switch to tarball install: `INSTALL_METHOD=tar`.

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

Default toolkit families are `13.4` for x86/SBSA, `13.2` for Jetson and unchanged
`13.0` for IGX Thor. Set `CUDA_VER` explicitly when selecting another supported
toolkit family; environment and CLI overrides are not overwritten:

```bash
CUDA_VER=13.4 bash build/build.sh
```

Offline `--package-from --package-platform=jetson|sbsa|x86|igx` uses the selected
package target's defaults, not the packaging host's defaults. It still bypasses
dependency installation, compilation, stage state and system integration.
The legacy `igx` target retains CUDA 13.0 and requires an explicit full TensorRT10
`TRT_VER` for offline packaging. Native IGX packaging can instead read the selected
`NvInferVersion.h` (`TRT_INCLUDE_DIR`, default `/usr/include/aarch64-linux-gnu`;
preprocessed with `CC`, default `cc`). Its inference runtime must be ARM64/TRT10;
the other three targets still require TRT11. The ELF check does not prove the
declared full TensorRT patch/build provenance. IGX tarballs use the `_igx` suffix,
not `_jetson`; the existing ARM reduced OCR/Sparse4D payload default is unchanged.
The packaged IGX Triton helper also requires an explicit matching server/client
archive pair qualified separately for the legacy IGX stack; it guesses no URL
and does not establish compatibility merely from matching release numbers.

Supply materialized runtime directories (`bin`, `lib`, `samples`, `service-maker`
and optional `sources/sparse4d`). Top-level directory aliases and nested directory
aliases outside `samples` are rejected before an existing stage is changed.
Within `samples`, relative directory aliases such as the Triton model-repository
links are supported when their targets stay inside `samples`. Absolute aliases,
paths escaping and re-entering `samples`, and cyclic directory graphs are
rejected. All sample links must resolve; dangling links and resolution loops
fail closed. Accepted directory aliases are normalized only in staging so that
overlays and pruning cannot follow them into the input installation. The input
SDK is not rewritten. A symlink alias for the `--package-from` root itself is
resolved normally; ordinary library/file links are otherwise preserved. If a
source overlay replaces a file symlink, it replaces the staged entry without
writing through to the original target.

The existing `build/.stage-state` now records platform, CUDA family, declared
TensorRT/cuDNN versions, SDK version and explicitly supplied provider paths.
It records selections, not hashes of provider contents. `--resume` rejects old state without
this identity or state from another stack instead of silently skipping rebuilds.
Clean old component objects explicitly before a stack change, then run without
`--resume`; recording new stage identity does not itself rebuild stale objects.

To override the DeepStream version (default: `9.2.1`, must be `MAJOR.MINOR.PATCH`;
the install tree uses the truncated `MAJOR.MINOR` form). The GitHub Release
proprietary-libs / sample-data assets are always fetched from the `v9.2.0`
release (hardcoded, not derived from `NVDS_VERSION`):

```bash
NVDS_VERSION=9.2.1 bash build/build.sh
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
# Remove DeepStream 9.2 (default version 9.2.1 → install tree deepstream-9.2/)
sudo bash scripts/uninstall.sh

# Remove a specific version (MAJOR.MINOR.PATCH)
sudo bash scripts/uninstall.sh 9.2.1
# or equivalently:
sudo PREV_DS_VER=9.2.1 bash scripts/uninstall.sh
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
# Full build, then emit both .deb and .tbz2 into build/
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
| `--runtime-root=PATH` | Complete extracted SDK directory instead of the installed `/opt` tree |
| `--platform=x86\|jetson\|sbsa` | Select platform-specific dependencies and tarball name |
| `--include-untracked-sources` | Local validation only: include non-ignored new source files |

Environment overrides: `NVDS_VERSION` (default `9.2.1`, must be `MAJOR.MINOR.PATCH`) and `PKG_VERSION`
(default: same as `NVDS_VERSION`).

### Output

Packages are written to the output directory (default `build/`) and named:

```text
deepstream-<NVDS_VERSION>_<PKG_VERSION>-<revision>_<arch>.deb
deepstream_sdk_v<full-version>_<x86_64|jetson|sbsa>.tbz2
```

The generated `.deb` can then be installed on a target machine with:

```bash
sudo apt install ./deepstream-<NVDS_VERSION>_<PKG_VERSION>-1_<arch>.deb
```

## Run the PR validation helper

`build/validate_pr.py` runs the existing checks and saves each command, exit code,
duration and log, plus `summary.json` and `junit.xml`. It records Git HEAD,
working-tree changes and submodules. Local uncommitted changes are labelled;
they are not validation of the pushed PR. It never clones/switches branches,
cleans files, installs prerequisites automatically, pushes or merges.

Start with CPU-only checks from the repository root:

```bash
python3.12 build/validate_pr.py
```

These execute shell syntax checks and the existing `build/test_*.py` and
`tools/modelopt/test_*.py` unit tests. Use an existing Python environment with
`numpy` and `onnx` via `--test-python /path/to/venv/bin/python` if needed. This
keeps CPU test dependencies separate from `--python`, the SDK runtime Python.
Packaging
fixtures also need `cc`, `git`, `readelf`, `rsync`, `dpkg-deb`, `tar` and `bzip2`.
Missing dependencies or skipped tests are not a complete pass. No GPU, dependency
installation or SDK installation is attempted by the default mode.

Preview the x86 sequence without executing it:

```bash
python3.12 build/validate_pr.py --stage full --headless --dry-run
```

On a **designated x86 test environment** with the matching complete SDK runtime,
compute stack and development inputs already provisioned:

```bash
# Non-root builds require cached sudo credentials before the detached commands.
sudo -v
python3.12 build/validate_pr.py --stage full --headless --allow-system-changes
```

The opt-in is significant: source/dependency builds can run sudo, APT and pip,
install into `/opt`, and update the test environment. Model setup can download
its isolated tool environment and rewrite sample models/configurations. The
runner extracts the freshly built SDK tarball into a private, user-owned
`runtime-sdk` beneath its report directory. Model conversion, locks, engines,
sample configuration updates and the isolated tool cache use that workspace;
the models/runtime stages do not change `/opt` ownership, the installed models,
system Python or the canonical SDK symlink. Do not run the whole runner as root.
The runner itself does not repair missing providers or change drivers/BSP.

`full` runs CPU checks, the real source build/package with `--skip-artifacts`,
explicit platform pins and no `--resume`, workspace ModelOpt setup, then two
source4 pipeline runs. It rejects old source objects rather than deleting them.
Default public artifact downloads remain pinned to 9.2.0, so the matching
9.2.1 runtime must be provisioned beforehand. This does **not** use
`--package-from` as a substitute for compiling the PR.

Individual stages are available with `--stage build`, `models`, or `runtime` and
the same explicit opt-in. To retry after a successful build without compiling
again:

```bash
python3.12 build/validate_pr.py --stage models --allow-system-changes
# Use the runtime-sdk path printed above and recorded in runtime-workspace.json:
python3.12 build/validate_pr.py --stage runtime --headless \
  --runtime-workspace /absolute/path/to/previous-report/runtime-sdk \
  --allow-system-changes
```

`models` defaults to this platform's tarball in `build/`; `--runtime-package`
can explicitly select another matching tarball for this standalone stage. `full`
always uses its own freshly built artifact. Extraction does not execute install
scripts. Unsafe archive paths, external links and mismatched packaged ModelOpt
helpers fail closed. The manifest records the package hash, helper identities,
relocated workspace files and model-preparation status. Runtime requires the
same successfully prepared workspace; it does not silently recreate an SDK and
lose its engines. The private workspace/cache is retained for inspection and
retry, not automatically deleted. A partial model setup is not a completed pass.

Runtime checks assert SDK 9.2.1, inspect selected plugin and linked DS-library
paths, import Service Maker 0.0.1 from the packaged wheel privately (no pip
install), verify its Python/loaded DS-library providers, and require successful application completion,
EOS and positive FPS from all four sources. `--headless` uses a unique temporary
configuration beside the workspace source4 config to preserve relative paths;
workspace-only SDK path relocation selects its own tracker library. Only sink0
is changed to a clocked fakesink by the headless option. The source installation
is not modified and
the owned copy is removed afterward. Two runs are not automatically cold/warm
benchmarks; engine-cache snapshots are recorded without deleting engines.
`--service-maker-tests` also runs owned copies of the existing pytest inference
tests, relocating only their SDK paths and selecting the packaged Python module
(requires pytest and additional GPU time). Import alone does not qualify Service Maker.

The aggregate `full` suite is x86-only because some existing CPU fixtures are
x86-specific. Select the individual native build/models/runtime stages on
Jetson or SBSA, using their actual matching providers. A platform selector is
not a cross-build or qualification of another platform. The helper checks
Debian compute-package identities; a tar-only compute installation needs
separate verification rather than bypassing this check.

For optional **x86 Docker assembly**, first produce the matching SDK tarball and
provision the three reviewed GStreamer files under `CONTENT/x86/gst` as described
in `deepstream_dockers/README.md`:

```bash
python3.12 build/validate_pr.py --stage docker \
  --gst-input-root /absolute/path/to/reviewed/CONTENT \
  --allow-system-changes
```

This stages inputs and invokes the existing setup/build scripts. Different
existing context files are not overwritten. It can pull images and install
packages inside the Docker build; it does not run or publish the final image.
Use `--docker-kind samples` for the samples recipe. ARM Docker providers are
not automated by this helper.

Results default to a unique private directory under
`~/deepstream-validation-results/`; `--output-dir` selects another parent outside
the repository. Logs are retained on failure or interruption. Long commands have
timeouts; termination is limited to their process group, and privileged children
may need operator inspection. No unrelated process or Docker image is cleaned up.
For an exact submitted revision, use `--expected-commit FULL_SHA --require-clean`.
Review logs before attaching them to a public PR.

Exit codes: **0** requested checks passed (or explicitly labelled dry run),
**1** test failure, **2** unavailable/incomplete validation, **130** interrupted.
A `PASS` covers only requested checks. Fresh installation in a second matching
environment, final-image inference, output accuracy, performance acceptance,
stress and other platforms remain separate gates before release qualification.
