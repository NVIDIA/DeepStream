## Introduction

This example demonstrates how to build an inference pipeline and for the Qwen family of VLM models using the Inference Builder tool. Following models have been tested:
- Qwen/Qwen2.5-VL-3B-Instruct
- Qwen/Qwen2.5-VL-7B-Instruct
- Qwen/Qwen2.5-VL-32B-Instruct

Three configurations are provided, allowing you to choose based on your specific software and hardware environment:

1. pytorch_qwen.yaml: leveraging the transformer APIs and fits all the models
2. trtllm_qwen.yaml: leveraging TensorRT LLM APIs for better performance
3. trtllm_nvdec_qwen.yaml: leveraging h/w decoder and TensorRT LLM for the best performance

We provide a sample Dockerfile for the example, which you can use to build a Docker image and test the microservice on any x86 system.

**⚠️ System Requirements:** This model requires significant GPU resources and is optimized for high-performance systems. We recommend testing on H100 or B200 GPUs for optimal performance. Docker build may fail on less powerful hardware.

## Prerequisites

**Note:** Make sure you are in the root directory (`path/to/inference_builder`) to execute the commands in this README. All relative paths and commands assume you are running from the inference_builder root directory. Also ensure that your virtual environment is activated before running any commands.

Before downloading the model file, you need to set up your model repository:

```bash
mkdir -p ~/.cache/model-repo && chmod 777 ~/.cache/model-repo
export MODEL_REPO=~/.cache/model-repo
```

The model checkpoints can be downloaded from huggingface. First, install git-lfs for large model files:

```bash
sudo apt install git-lfs
git lfs install
```

Then download the model:

```bash
git clone https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct  $MODEL_REPO/Qwen2.5-VL
```

Assume you've followed the [top level instructions](../../../README.md#getting-started) to set up the environment and be sure you're in the inference_builder folder, then activate your virtual environment:

```bash
source .venv/bin/activate
```

## Build and Test the Qwen2.5-VL-7B-Instruct Inference Microservice

### Generate the Inference Pipeline

For Pytorch backend with transformers API, use the command below:

```bash
python builder/main.py builder/samples/qwen/pytorch_qwen.yaml --api-spec builder/samples/qwen/openapi.yaml \
-o builder/samples/qwen/ -c builder/samples/qwen/processors.py -t
```

For TensorRT-LLM backend with S/W decoder, use the command below:

```bash
python builder/main.py builder/samples/qwen/trtllm_qwen.yaml --api-spec builder/samples/qwen/openapi.yaml \
-o builder/samples/qwen/ -c builder/samples/qwen/processors.py -t
```

If you want to acheive the best performance with Deepstream and TensoRT-LLM, please use following command:

```bash
# Enable hardware decoder
python builder/main.py builder/samples/qwen/trtllm_nvdec_qwen.yaml --api-spec builder/samples/qwen/openapi.yaml \
-o builder/samples/qwen/ -c builder/samples/qwen/processors.py -t
```

### Build and Start the Inference Microservice:

```bash
cd builder/samples && docker compose up ms-qwen --build
```

The build may take a while—images are pulled from NGC and speed depends on your network. ⌛

### Test the Inference Microservice with a client

Wait for the server to start, then open a new terminal in your inference_builder folder. The sample includes a test OpenAI client. For image input, run the client with the path to an image file.


```bash
source .venv/bin/activate && cd builder/samples/qwen
python client.py --images <your_image.jpg> # replace placeholder <your_image.jpg> with an actual file
```

For video input, you need to first upload a test video file.

**⚠️ Important:** **replace the placeholder <your_video.mp4> in below command with an actual file path in your system**.

```bash
export VIDEO_FILE=<your_video.mp4>
curl -X "POST" \
  "http://localhost:8800/v1/files" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@$VIDEO_FILE;type=video/mp4"
```

You'll get a response 200 with a json body which includes the id and path of the uploaded asset like below:

{
  "data": {
    "id": "577a9f11-2b24-4db8-82c8-2601e0c2b6e4",
    "path": "/tmp/assets/577a9f11-2b24-4db8-82c8-2601e0c2b6e4/its_1920_30s.mp4",
    "size": 3472221,
    "duration": 30000000000,
    "contentType": "video/mp4"
  }
}

Run the client.py with the video path from your response if the inference pipeline is built from trtllm_qwen.yaml or pytorch_qwen.yaml without H/W decoder:

```bash
# Please use the path from your file upload response
python client.py --videos /tmp/assets/577a9f11-2b24-4db8-82c8-2601e0c2b6e4/its_1920_30s.mp4
```

OR run the client.py with the returned asset id if the inference pipeline is built from trtllm_nvdec_qwen.yaml with H/W decoder enabled:

```bash
# Please replace 577a9f11-2b24-4db8-82c8-2601e0c2b6e4 with the id returned from your file upload response
python client.py --videos 577a9f11-2b24-4db8-82c8-2601e0c2b6e4?frames=8
```