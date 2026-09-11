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

# DeepStream Service Maker

Service Maker is a C++ and Python SDK that provides a high-level abstraction over GStreamer/GLib APIs for building DeepStream applications. Instead of writing boilerplate GStreamer pipeline code, you compose pipelines declaratively using a fluent C++ API or Python flow/pipeline APIs.

---

## Layout

```
service-maker/
├── cmake/                        # CMake find-modules and toolchain files
├── includes/                     # Public C++ headers
└── sources/
    ├── core/                     # Service Maker runtime library (libnvds_service_maker.so)
    ├── engine/                   # ds-launch — runs a pipeline from a YAML graph file
    ├── python/                   # pyservicemaker: pybind11 bindings + Python package
    ├── apps/
    │   ├── cpp/                  # C++ sample applications
    │   └── python/
    │       ├── flow_api/         # Python flow API samples
    │       └── pipeline_api/     # Python pipeline API samples
    └── modules/                  # Reusable pipeline modules (probes, sinks, sources, …)
```

---

## Prerequisites

- DeepStream 9.1 installed (via `build/build.sh` or the Debian package + `install.sh`)
- CMake 3.16+
- GCC 9+

For full system dependency setup, see [build/BUILD.md](../../build/BUILD.md).

---

## Build

### Full Build (via build/build.sh)

Service Maker is built automatically as part of the top-level `build/build.sh`. The
`service-maker` stage builds, in order:

1. `sources/core/src/gst/utils` → `libnvds_service_maker_utils.a`, the static helper
   archive that loadable modules link against
2. `sources/core` → `libnvds_service_maker.so`, the runtime library
3. `sources/engine` → `ds-launch`
4. `sources/apps/cpp/*` and `sources/modules/*`
5. `sources/python` → the `pyservicemaker` wheel

The core library must be installed before the apps and modules, which resolve it
through `find_package(nvds_service_maker)`.

Libraries are installed to:

```
/opt/nvidia/deepstream/deepstream-9.1/lib/
```

Binaries are installed to:

```
/opt/nvidia/deepstream/deepstream-9.1/bin/service-maker-<app>
/opt/nvidia/deepstream/deepstream-9.1/bin/ds-launch
```

Modules are installed to:

```
/opt/nvidia/deepstream/deepstream-9.1/service-maker/modules/
```

The `pyservicemaker` wheel is installed to:

```
/opt/nvidia/deepstream/deepstream-9.1/service-maker/python/
```

overwriting the prebuilt wheel placed there by the `artifacts` stage, so the
bindings always match the core they were built against. `scripts/install.sh`
then `pip install`s it.

### Building the Python bindings alone

`sources/python` needs the pybind11 and dlpack headers that
`scripts/install_opensource_deps.sh` installs to `/opt/pybind11` and
`/opt/dlpack`, and it links the core library, so build and install that first.

```bash
bash sources/python/build.sh --output-dir /tmp/whl
python3 -m pip install /tmp/whl/pyservicemaker-*.whl --force-reinstall --break-system-packages
```

`build.sh` also accepts `--minor-ver <N>` to target a different Python 3 minor
version and `--arch aarch64` to cross-compile. Override `PYBIND11_ROOT` or
`DLPACK_ROOT` to build against headers installed elsewhere.

### Building the core library or engine alone

`sources/core` and `sources/engine` use Makefiles rather than CMake:

```bash
make -C sources/core/src/gst/utils CUDA_VER=13.2
make -C sources/core CUDA_VER=13.2
make -C sources/engine CUDA_VER=13.2
sudo make -C sources/core install     # and likewise for the others
```

### Local CMake Build (single app)

Individual apps can be built and run locally without `build/build.sh`:

```bash
cd sources/apps/cpp/<app-name>
mkdir build && cd build
cmake ..
make
```

The binary is placed in the local `build/` directory and nothing is installed to `/opt`.

---

## C++ API Overview

Service Maker provides a `Pipeline` class with a chainable API:

```cpp
#include "pipeline.hpp"

using namespace deepstream;

int main(int argc, char** argv) {
    try {
        Pipeline pipeline("deepstream-test");
        pipeline
            .add("nvurisrcbin",   "src",   "uri", argv[1])
            .add("nvstreammux",   "mux",   "batch-size", 1, "width", 1280, "height", 720)
            .add("nvinferbin",    "infer", "config-file-path", "/path/to/config.txt")
            .add("nvdsosdbin",    "osd")
            .add("nveglglessink", "sink");
        pipeline
            .link({"src", "mux"}, {"", "sink_%u"})
            .link("mux", "infer", "osd", "sink");
        pipeline.start();
        pipeline.wait();
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
    return 0;
}
```

Pipelines can also be loaded from a YAML graph file:

```cpp
Pipeline pipeline("MyGraph", "MyGraph.yaml");
pipeline.start();
pipeline.wait();
```

### CMake Template

```cmake
cmake_minimum_required(VERSION 3.16)
project(my_ds_app LANGUAGES CXX)

find_library(nvds_service_maker NAMES nvds_service_maker
    HINTS /opt/nvidia/deepstream/deepstream/service-maker)
include_directories(/opt/nvidia/deepstream/deepstream/sources/includes)

add_executable(my-app my_app.cpp)
target_link_libraries(my-app PRIVATE nvds_service_maker)
```

---

## Python API Overview

Two Python APIs are available:

### Flow API

Compose a pipeline by chaining element additions and links:

```python
from deepstream import Pipeline

pipeline = Pipeline("deepstream-test")
pipeline \
    .add("nvurisrcbin", "src", uri=sys.argv[1]) \
    .add("nvstreammux", "mux", batch_size=1, width=1280, height=720) \
    .add("nvinferbin", "infer", config_file_path="/path/to/config.txt") \
    .add("nvdsosdbin", "osd") \
    .add("nveglglessink", "sink")
pipeline.link("src", "mux").link("mux", "infer", "osd", "sink")
pipeline.start()
pipeline.wait()
```

See `sources/apps/python/flow_api/` for full examples.

### Pipeline API

Build pipelines using a higher-level declarative pipeline description. See `sources/apps/python/pipeline_api/` for examples.

---

## C++ Sample Applications

| App | Description |
|---|---|
| `deepstream_test1_app` | Single-stream decode → infer → OSD |
| `deepstream_test2_app` | Adds secondary classifiers and tracker |
| `deepstream_test3_app` | Multi-stream input |
| `deepstream_test4_app` | Cloud messaging via message broker |
| `deepstream_test5_app` | Fully configurable multi-stream pipeline |
| `deepstream_appsrc_test_app` | Custom appsrc buffer feeding |
| `deepstream_sr_test_app` | Smart recording |
| `deepstream_3d_action_recognition_app` | 3D action recognition |

Refer to the `README` in each app's directory for run instructions.

---

## Reusable Modules

Located in `sources/modules/`. These are loadable shared-library modules that plug into Service Maker pipelines:

| Module | Description |
|---|---|
| `sample_video_feeder` | Feeds video from a URI into the pipeline |
| `sample_video_probe` | Attaches a frame-level probe callback |
| `sample_video_receiver` | Receives and renders video output |
| `resnet_tensor_parser` | Parses ResNet tensor output into object metadata |
| `measure_fps_probe` | Measures and logs pipeline FPS |
| `measure_latency_probe` | Measures per-frame end-to-end latency |
| `kitti_dump_probe` | Dumps detection results in KITTI format |
| `add_message_meta_probe` | Attaches cloud message metadata to frames |
| `smart_recording_action` | Triggers smart recording on detection events |
| `smart_recording_signal` | Sends recording control signals |
| `sample_signal_handler` | Generic pipeline signal handling |
| `ensemble_render` | Renders multi-model ensemble results |
| `lidar_feeder` | Feeds LiDAR point-cloud data into the pipeline |

---

## Further Reading

- [DeepStream Service Maker Developer Guide](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_service_maker_python.html)
- [DeepStream SDK Documentation](https://docs.nvidia.com/metropolis/deepstream/dev-guide/)
