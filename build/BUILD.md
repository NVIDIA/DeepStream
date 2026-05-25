# DeepStream Mono-Repo — Build & Run Guide

## Prerequisites

### NVIDIA Compute Stack

Before building from source, the NVIDIA compute stack must be installed on the host. Required versions:

| Platform | OS | Driver | CUDA | cuDNN | TensorRT |
|---|---|---|---|---|---|
| x86 dGPU | Ubuntu 24.04 | 590.48.01+ | 13.1 | 9.18.0 | 10.14.x |
| Jetson (aarch64) | JetPack 7.1 GA | — (bundled) | 13.0 | 9.12.0 | 10.13.x |
| SBSA / DGX Spark | Ubuntu 24.04 | 590.48.01+ | 13.0 | 9.18.0 | 10.14.x |

For installation instructions, refer to the official [DeepStream SDK Installation Guide](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Installation.html).

> **Docker users:** NVIDIA NGC provides pre-built DeepStream 9.0 containers with the full compute stack pre-installed. If using a container, skip this section and proceed to Git LFS setup below.

---

### Install cmake

```bash
sudo apt-get install cmake
```

### x86 (dGPU)

1. Download the DS 9.0 public released Debian package, install it, and add the
   DeepStream lib directory to `LD_LIBRARY_PATH` so the runtime can locate the
   shared libraries:
   ```bash
   sudo apt update
   sudo apt remove libglapi-amber
   sudo apt install ./deepstream-9.0_9.0.0-1_amd64.deb
   sudo /opt/nvidia/deepstream/deepstream-9.0/install.sh
   export LD_LIBRARY_PATH=/opt/nvidia/deepstream/deepstream-9.0/lib:$LD_LIBRARY_PATH
   ```
   Append the `export` line to `~/.bashrc` to persist it across shells.

2. Install system dependencies:
   ```bash
   sudo apt install \
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
     libmosquitto1
   ```

   See also the [DeepStream Installation Guide — Install Prerequisite Packages](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Installation.html#id2).

### aarch64 (Jetson)

1. Download the DS 9.0 public released Debian package, install it, and add the
   DeepStream lib directory to `LD_LIBRARY_PATH` so the runtime can locate the
   shared libraries:
   ```bash
   sudo apt update
   sudo apt remove libglapi-amber
   sudo apt install ./deepstream-9.0_9.0.0-1_arm64.deb
   sudo /opt/nvidia/deepstream/deepstream-9.0/install.sh
   export LD_LIBRARY_PATH=/opt/nvidia/deepstream/deepstream-9.0/lib:$LD_LIBRARY_PATH
   ```
   Append the `export` line to `~/.bashrc` to persist it across shells.

2. Install system dependencies:
   ```bash
   sudo apt install \
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
     libmosquitto1
   ```

   See also the [DeepStream Installation Guide — Install Prerequisite Packages](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Installation.html#install-prerequisite-packages).

### SBSA / DGX Spark (Server Base System Architecture)

SBSA and DGX Spark share the same code path. There is no DS tar/deb package for this platform. The validated environment is the SBSA Docker image.
Build and validation must be done **inside the Docker container**.

1. Start the SBSA Docker container (with GPU access):
   ```bash
   docker run -it --gpus all <sbsa-docker-image> bash
   ```

2. Install build prerequisites inside the container:
   ```bash
   apt-get update && apt-get install -y git cmake make build-essential
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

## Build and Install

`build/build.sh` auto-detects the host platform and:
- On SBSA / DGX Spark: passes `AARCH64_IS_SBSA=1` to all Makefiles (both `aarch64` and `sbsa` report `uname -m` as `aarch64`; SBSA is identified by the absence of `/etc/nv_tegra_release`)
- Builds and installs open-source dependencies (opentelemetry, civetweb, prometheus-cpp, azure-iot-sdk)
- Builds all source components and installs them directly to `/opt/nvidia/deepstream/deepstream-<NVDS_VERSION>/`
- Prompts for `sudo` only when a step must install packages or write to system paths

From the repo root:

```bash
bash build/build.sh
```

Default CUDA version is architecture-dependent (`13.1` for x86_64, `13.0` for aarch64/sbsa/DGX Spark). To override:

```bash
CUDA_VER=13.1 bash build/build.sh   # x86
CUDA_VER=13.0 bash build/build.sh   # aarch64
```

To override the target DeepStream version (default: 9.0):

```bash
NVDS_VERSION=9.0 bash build/build.sh
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

---

## service-maker — Local Compilation

Individual service-maker apps can be compiled locally without running `build/build.sh`.
From within any app directory (e.g. `src/service-maker/sources/apps/cpp/deepstream_test1_app/`):

```bash
mkdir build && cd build && cmake .. && make
```

The binary is placed in the local `build/` directory and nothing is installed to `/opt`.
See each app's `README` for usage instructions.
