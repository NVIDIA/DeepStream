## Introduction

The examples in this folder demonstrates how to use Inference Builder to create inference pipelines with various backend, such as Deepstream, TensorRT, Triton, TensorRT-LLM, etc.

With the provided Dockerfile, you can package the generated pipeline into a container image and run it as a standalone app or a microservice. Build steps vary by backend. Check the corresponding README.md in each example folder for exact instructions. For examples that run as microservices, we've provided an all-in-one [docker-compose.yml](./docker-compose.yml) to manage them together. You can customize the container behavior by changing the configurations there accordingly.

## Models

Some of the models for examples are from the NVIDIA GPU Cloud (NGC) repository, and certain models from NGC require active subscription. 

## List of Examples

### DeepStream Backend Examples

| Name | Description | Models | Backend | Output |
|------|-------------|---------|---------|---------|
| [ds_app](./ds_app/) | examples of building standalone deepstream application | TAO Computer Vision models | Deepstream | command line interface application |
| [tao](./tao/) | examples of building inference microservices using deepstream pipeline and fastapi | TAO Computer Vision models | Deepstream | microservice |

### Triton Backend Examples

| Name | Description | Models | Backend | Output |
|------|-------------|---------|---------|---------|
| [changenet](./changenet/) | example of building inference microservices with triton server | Visual ChangeNet | Triton/TensorRT | microservice |

### TensorRT Backend Examples

| Name | Description | Models | Backend | Output |
|------|-------------|---------|---------|---------|
| [nvclip](./nvclip/) | example of building inference microservices with TensorRT backend | NVCLIP | TensorRT | microservice |

### TensorRT-LLM Backend Examples

| Name | Description | Models | Backend | Output |
|------|-------------|---------|---------|---------|
| [qwen](./qwen/) | example of building inference microservices with TensorRT-LLM backend for vlm models | Qwen 2.5 VL models | TensorRT-LLM, Pytorch | microservice |

### Multiple Model Examples

| Name | Description | Models | Backend | Output |
|------|-------------|---------|---------|---------|
| [cradio](./cradio/) | two-stage pipeline: PeopleNet Transformer (detection) then C-RADIOv3-H (per-detection embeddings) | PeopleNet Transformer, C-RADIOv3-H | DeepStream/nvinfer, TensorRT (polygraphy) | command line interface application |

### VLLM Backend Examples

| Name | Description | Models | Backend | Output |
|------|-------------|---------|---------|---------|
| [vllm](./vllm/) | example of building inference microservices with vLLM backend and DeepStream MediaExtractor | Cosmos-Reason2-2B, Qwen3-VL-2B-Instruct | vLLM, DeepStream | microservice |

## Dockerfiles

| Sample | Dockerfile | Base Image | Platform | Key Steps |
|--------|------------|------------|----------|-----------|
| changenet | `changenet/Dockerfile` | `gitlab-master.nvidia.com:5005/deepstreamsdk/release_image/deepstream:9.1.1-26.08.2-triton-20260817-ma-dev99` | x86_64 | pip install (torch, transformers, fastapi), CVCUDA 0.16.0 .deb install, Triton server entrypoint |
| cradio | `cradio/Dockerfile` | `gitlab-master.nvidia.com:5005/deepstreamsdk/release_image/deepstream:9.1.1-26.08.2-triton-20260817-ma-dev99` | x86_64 | TensorRT OSS build (v10.16), DeepStream TAO post-processors build, pip install (torch, transformers, polygraphy, tensorrt) |
| ds_app | `ds_app/Dockerfile` | `gitlab-master.nvidia.com:5005/deepstreamsdk/release_image/deepstream:9.1.1-26.08.2-triton-20260817-ma-dev99` | x86_64 | TensorRT OSS build disabled by default, DeepStream TAO post-processors, pip install (torch, transformers, opencv) |
| ds_app (Jetson) | `ds_app/Dockerfile.tegra` | Multi-stage: `pytorch:25.08-py3` + `gitlab-master.nvidia.com:5005/deepstreamsdk/release_image/deepstream:9.1.1-26.08.2-triton-20260817-ma-dev99` | aarch64 (Jetson) | Multi-stage pytorch copy, TensorRT OSS build disabled by default, DeepStream TAO post-processors |
| ds_app (DGX Spark) | `ds_app/Dockerfile.dgxspark` | Multi-stage: `pytorch:25.08-py3` + `gitlab-master.nvidia.com:5005/deepstreamsdk/release_image/deepstream:arm-sbsa-9.1.1-26.08.2-triton-20260818-dev250` | aarch64 (DGX SBSA) | Multi-stage pytorch copy, TensorRT OSS build disabled by default, DeepStream TAO post-processors |
| vllm | `vllm/Dockerfile` | Multi-stage: `gitlab-master.nvidia.com:5005/deepstreamsdk/release_image/deepstream:9.1.1-26.08.2-triton-20260817-ma-dev99` + `vllm:26.02-py3` | x86_64 | Multi-stage DeepStream libs copy, pip install (qwen-vl-utils, aiohttp, omegaconf), custom shell entrypoint |
| nvclip | `nvclip/Dockerfile` | `gitlab-master.nvidia.com:5005/deepstreamsdk/release_image/deepstream:9.1.1-26.08.2-triton-20260817-ma-dev99` | x86_64 | pip install (torch, transformers, tensorrt, polygraphy, fastapi), user/group setup, custom shell entrypoint |
| nvclip optimizer | `nvclip/optimizer/Dockerfile` | `nvcr.io/nvidia/tritonserver:26.03-py3` | x86_64 | pip install (open_clip_torch, onnx, polygraphy), ONNX export/TensorRT optimization pipeline |
| tao | `tao/Dockerfile` | `gitlab-master.nvidia.com:5005/deepstreamsdk/release_image/deepstream:9.1.1-26.08.2-triton-20260817-ma-dev99` | x86_64 | TensorRT OSS build disabled by default, DeepStream TAO post-processors, TAO symlink preparation script |
| tao validation | `tao/validation/Dockerfile.validation` | `python:3.10-slim` | x86_64 | Minimal validation image, pip install from requirements.txt, validation script entrypoint |
| qwen | `qwen/Dockerfile` | Multi-stage: `gitlab-master.nvidia.com:5005/deepstreamsdk/release_image/deepstream:9.1.1-26.08.2-triton-20260817-ma-dev99` + `tensorrt-llm/release:1.1.0` | x86_64 | Multi-stage DeepStream libs copy, pip install (qwen-vl-utils, aiohttp, omegaconf), TensorRT-LLM + DeepStream integration |

### Common Patterns

- **Base image**: Choose based on the primary inference stack — DeepStream/Triton pipelines use `gitlab-master.nvidia.com:5005/deepstreamsdk/release_image/deepstream:9.1.1-26.08.2-triton-20260817-ma-dev99` for x86_64 and Jetson, or `gitlab-master.nvidia.com:5005/deepstreamsdk/release_image/deepstream:arm-sbsa-9.1.1-26.08.2-triton-20260818-dev250` for SBSA/DGX Spark; vLLM pipelines use `nvcr.io/nvidia/vllm:26.02-py3` as the final stage; TensorRT-LLM pipelines use `nvcr.io/nvidia/tensorrt-llm/release:1.1.0` as the final stage. ARM64 targets (Jetson, DGX Spark) further add `nvcr.io/nvidia/pytorch` as a builder stage to supply pre-built ARM wheels.
- **TensorRT OSS**: Some DeepStream pipelines compile TensorRT OSS in-container for custom plugin support. The ds_app and tao Dockerfiles keep the build commented out because the DeepStream 9.1 base images already include TensorRT plugin/parser libraries and `trtexec`. Enable the TensorRT OSS build if specific operators required by a model are missing.
- **TAO post-processors**: CV inference samples (ds_app, tao, changenet, cradio) build `libnvds_infercustomparser_tao.so` inside the container after DeepStream installation.
- **Multi-stage for vLLM/TRT-LLM**: Because DeepStream is not bundled in the vLLM/TensorRT-LLM images, these Dockerfiles use a multi-stage build to copy the required GStreamer libs and DeepStream binaries from the DeepStream image into the final stage for hardware accelerated decoder support.
- **Entrypoint**: Inference Builder generates a shell entrypoint that runs `__main__.py`; Triton-backed pipelines also launch `tritonserver` as a subprocess.
- **ds_app high-batch decoding (agent note)**: In `ds_app/Dockerfile`, the `ENTRYPOINT` must keep `ulimit -Sn $(ulimit -Hn)` immediately before `exec python3 … __main__.py`. That bumps the soft `RLIMIT_NOFILE` to the hard limit so DeepStream decoding and nvstreammux can sustain large batches without hitting “too many open files” or similar failures. When editing this Dockerfile, copying its pattern to another image, or suggesting run commands, preserve this wrapper; a plain `python3 __main__.py` start without raising the limit can regress high-batch workloads.
- **BuildKit caching**: All pip installs use `--mount=type=cache,target=/root/.cache/pip` — build with `DOCKER_BUILDKIT=1`.
