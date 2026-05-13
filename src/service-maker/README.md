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

# DeepStream Service Maker

Service Maker is a C++ and Python SDK that provides a high-level abstraction over GStreamer/GLib APIs for building DeepStream applications. Instead of writing boilerplate GStreamer pipeline code, you compose pipelines declaratively using a fluent C++ API or Python flow/pipeline APIs.

---

## Layout

```
service-maker/
├── cmake/                        # CMake find-modules and toolchain files
├── includes/                     # Public C++ headers
├── python/                       # Python bindings
└── sources/
    ├── apps/
    │   ├── cpp/                  # C++ sample applications
    │   └── python/
    │       ├── flow_api/         # Python flow API samples
    │       └── pipeline_api/     # Python pipeline API samples
    └── modules/                  # Reusable pipeline modules (probes, sinks, sources, …)
```

---

## Prerequisites

- DeepStream 9.0 installed (via `build.sh` or the Debian package + `install.sh`)
- CMake 3.16+
- GCC 9+

For full system dependency setup, see the repo root [BUILD.md](../../BUILD.md).

---

## Build

### Full Build (via build.sh)

Service Maker apps are built automatically as part of the top-level `build.sh`. Binaries are installed to:

```
/opt/nvidia/deepstream/deepstream-9.0/bin/service-maker-<app>
```

Modules are installed to:

```
/opt/nvidia/deepstream/deepstream-9.0/service-maker/modules/
```

### Local CMake Build (single app)

Individual apps can be built and run locally without `build.sh`:

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
