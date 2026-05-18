# DeepStream Source Tree

This directory contains the source for all DeepStream 9.0 components shipped from this repository, along with platform-specific notes, external-prerequisite caveats, and deprecation status.

For build and install instructions, see [BUILD.md](../build/BUILD.md).

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

## Components with External Prerequisites

The following components build successfully only when the listed external SDK is installed on the system. See each component's README for installation links and setup details.

| Component | External Prerequisite | Details |
|---|---|---|
| `gst-dsexample-cuda` | OpenCV 4.x built with CUDA support | [gst-plugins/gst-dsexample-cuda/README](gst-plugins/gst-dsexample-cuda/README) |
| `gst-nvdsudp` | Rivermax SDK (`rivermax_api.h`) | [gst-plugins/gst-nvdsudp/README](gst-plugins/gst-nvdsudp/README) |
| `libnvdsgst_inferserver` | Triton Inference Server client | [gst-plugins/gst-nvinferserver/README](gst-plugins/gst-nvinferserver/README) |

---

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
