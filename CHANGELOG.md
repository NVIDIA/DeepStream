# DeepStream 9.2.0 (19 October 2026)

## Source availability

- Added source for the following DeepStream components, previously shipped
  only as prebuilt binaries:

  | Component | Description |
  |---|---|
  | `gst-nvdsvisionencoder-c` (`nvdsvisionencoder`) | C plugin that generates ReID embeddings for tracked objects (TensorRT; Triton client optional) and attaches `NVDS_TRACKER_OBJ_REID_META`. |
  | `gst-nvdstextembedder` (`nvdstextembedder`) | Python plugin that generates DFN5B-CLIP text embeddings for use with `nvdsvisionencoder`. Not compiled by `build/build.sh`. |
  | `ds-sparse4d` (via `gst-nvdsvideotemplate`) | Sparse4D multi-camera 3D detection and tracking custom lib (`libnvdsgst_sparse4d.so`). Requires PyTorch 2.9.0+cu130. |
  | `nvmultiobjecttracker` | Low-level multi-object tracker library (`libnvds_nvmultiobjecttracker.so`) loaded by `gst-nvtracker`. Built by `build/build.sh` in the `utils` stage. VPI headers are already present in the DS 9.2 build image; bare-metal x86 builds outside the image require the `vpi4-dev` package (see [build/BUILD.md](build/BUILD.md)). |

  For details, see [src/gst-plugins/README.md](src/gst-plugins/README.md).

- Added in-tree source for the V4L2 and unixfd GStreamer plugins
  (`gst-v4l2`, `gst-nvunixfd`) under `src/3rdparty`, with build instructions in
  their READMEs. These plugins are no longer shipped in the binaries packages.

## New Features

- Contributions are now accepted through pull requests to the `develop` branch.
  Read [CONTRIBUTING.md](CONTRIBUTING.md) and the
  [pull request guidelines](PULL_REQUEST_GUIDELINES.md) for the required full
  build, source4 sanity test, and DCO sign-off.

- Added a Python probe authoring framework for Service Maker: reusable Python
  probe modules (`add_message_meta_probe`, `measure_fps_probe`,
  `kitti_dump_probe`) plus flow-API and pipeline-API
  `deepstream_python_probe_app` sample apps. See
  [src/service-maker/sources/modules/probe/python](src/service-maker/sources/modules/probe/python).

- Added wheel packaging for DeepStream Libraries: run
  `bash build/build.sh --deepstream-libraries-wheel=artifacts` to build
  `deepstream_libraries-<ver>-cp312-cp312-linux_x86_64.whl` locally. See
  [deepstream_libraries/build/README.md](deepstream_libraries/build/README.md).

- The build and install scripts (`build/build.sh`, `build/package.sh`,
  `scripts/install.sh`, `scripts/install_artifacts.sh`,
  `scripts/install_opensource_deps.sh`, `scripts/uninstall.sh`) now require
  `NVDS_VERSION` to be a full `MAJOR.MINOR.PATCH` value (e.g. `9.2.0`)
  instead of `MAJOR.MINOR`.

- Added agent skills under [skills/](skills/README.md) for common DeepStream
  workflows, including building the compute stack, importing vision models,
  generating pipeline configs, and setting up DeepStream on WSL.

- Added a public SBSA docker build recipe under `deepstream_dockers/`, and
  published the full DeepStream 9.2.0 SBSA SDK tarball (`deepstream_sdk_v9.2.0_sbsa.tbz2`) as a
  release asset.

## Platform Support

- Added IGX Thor (aarch64) support: IGX-SW 2.0 GA, CUDA 13.0, TensorRT
  10.13.x, GPU driver 580.00. Set `CUDA_VER=13.0` when building for IGX Thor;
  other platforms keep the default of `13.2`.

____

# DeepStream 9.1.1

## Source availability

- Added source for the following DeepStream components, previously shipped
  only as prebuilt binaries:

  | gst-plugins | apps | utils |
  |---|---|---|
  | `gst-nvblender` | `back-to-back-detectors` | `nvcudaimage` |
  | `gst-nvbufsurfacewrite` | `deepstream-blend` | `nvdewarper` |
  | `gst-nvdsinferbase` | `deepstream-kitti-overlay` | `nvds_stats` |
  | | `perf-demo` | `nvdsinfer_logger` |

  For details, see [src/gst-plugins/README.md](src/gst-plugins/README.md).

____

# DeepStream 9.1 (14 July 2026)

## New Features

- Refer DeepStream-9.1 SDK supported new features at https://docs.nvidia.com/deepstream/9.2/text/DS_Release_notes#whats-new
- VLM support (Cosmos Reason 2) via the vLLM Plugin — a GStreamer plugin that integrates Vision-Language Models using vLLM for real-time video understanding. See the [DeepStream vLLM Plugin README](https://github.com/NVIDIA/DeepStream/tree/main/src/apps/reference_apps#deepstream-vllm-plugin-readme) for setup and usage.

____

# DeepStream 9.0 (24 March 2026)

## New Features

- Refer DeepStream-9.0 SDK supported new features at https://docs.nvidia.com/deepstream/9.2/text/DS_Release_notes#whats-new

____
