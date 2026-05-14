## Introduction

The examples in this folder demonstrates how to use Inference Builder to create inference pipelines with various backend, such as Deepstream, TensorRT, Triton, TensorRT-LLM, etc.

With the provided Dockerfile, you can package the generated pipeline into a container image and run it as a standalone app or a microservice. Build steps vary by backend. Check the corresponding README.md in each example folder for exact instructions. For examples that run as microservices, we've provided an all-in-one [docker-compose.yml](./docker-compose.yml) to manage them together. You can customize the container behavior by changing the configurations there accordingly.

## Models

Some of the models for examples are from the NVIDIA GPU Cloud (NGC) repository, and certain models from NGC require active subscription. Please download and install the NGC CLI from the [NGC page](https://org.ngc.nvidia.com/setup/installers/cli) and follow the [NGC CLI Guide](https://docs.ngc.nvidia.com/cli/index.html) to set up the tool.

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

