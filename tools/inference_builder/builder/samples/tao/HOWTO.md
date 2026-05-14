# Introduction

This example demonstrates how to build Metropolis Computer Vision Inference Microservices with Inference Builder and use them to perform inference on images and videos.

We provide the Dockerfile which you can use to build a Docker image and deploy the microservice on any x86 system with an NVIDIA Ampere, Hopper, and Blackwell GPU. We also provide a sample OpenAPI specification that you can use as a reference when customizing your own API definitions.

You can use the models listed below with your microservices, or use your own checkpoints generated from TAO Fine-Tune Microservices. In addition, you can modify the Dockerfile to add or remove dependencies that are specifically required by your model.

# Prerequisites

### Models

Some of the models used by the examples are from the NVIDIA GPU Cloud (NGC) repository, and certain models from NGC require active subscription. Please download and install the NGC CLI from the [NGC page](https://org.ngc.nvidia.com/setup/installers/cli) and follow the [NGC CLI Guide](https://docs.ngc.nvidia.com/cli/index.html) to set up the tool.

| Name of the Model | Runtime Configuration Directory* | Build Configuration** | Type of the Model |
| :-- | :-- | :-- | :-- |
| [PCB Classification](https://catalog.ngc.nvidia.com/orgs/nvaie/models/pcbclassification) | builder/samples/ds_app/classification/pcbclassification/ | ds_tao.yaml | classification |
| [Visual Changenet Classification](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/visual_changenet_classification) | builder/samples/ds_app/classification/changenet-classify/ | ds_changenet.yaml | classification |
| [CitySemSegFormer](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/citysemsegformer) | builder/samples/ds_app/segmentation/citysemsegformer/ | ds_tao.yaml | segmentation |
| [Grounding DINO](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/grounding_dino) | builder/samples/ds_app/gdino/gdino/ | ds_gdino.yaml | detection |
| [Mask Grounding DINO](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/mask_grounding_dino) | builder/samples/ds_app/gdino/mask_gdino/ | ds_gdino.yaml | segmentation |
| [Resnet50 RT-DETR](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/trafficcamnet_transformer_lite) | builder/samples/ds_app/detection/rtdetr/ | ds_tao.yaml | detection |

*:  Runtime configuration files are used by Deepstream during runtime and need to be put to the model directory.

**: The build configuration file is used for building the inference pipeline

Apart from the models listed in the above table, the Inference Builder also support building CV microservices for fine-tuned models exported from [TAO Deploy](https://docs.nvidia.com/tao/tao-toolkit/text/tao_deploy/tao_deploy_overview.html).

### Sample Videos

If you don't have any test video in hand, you can copy it from deepstream-sample container image using following commands:

```bash
docker pull nvcr.io/nvidia/deepstream:9.0-triton-multiarch
docker create --name temp nvcr.io/nvidia/deepstream:9.0-triton-multiarch
docker cp temp:/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4 ~/Videos
docker rm temp
```

You'll have a sample traffic street video under `~/Videos`, which you can use for testing in the following steps.

# Build and Test CV Inference Microservices

## General Instructions

There're three CV Inference Microservices in the example and all of them are built the same way; the only differences are their configurations and customized processors.

Using the same steps shown in this example, you can also build the CV inference microservice with fine-tuned models exported from [TAO Deploy](https://docs.nvidia.com/tao/tao-toolkit/text/tao_deploy/tao_deploy_overview.html).

Before you begin, create a model repository. Place all model files and runtime configuration files in a single, self-contained folder. The pipeline selects the model based on the TAO_MODEL_NAME environment variable (set in docker-compose.yaml), which must match the model folder name.

For Finetuned models from TAO-Deploy, the output can be directly dropped into the corresponding folder in the model reposiotry.

```bash
mkdir -p ~/.cache/model-repo && chmod 777 ~/.cache/model-repo
export MODEL_REPO=~/.cache/model-repo
```

Assume you've followed the [top level instructions](../../../README.md#getting-started) to set up the environment and be sure you're in the inference_builder folder, then activate your virtual environment:

```bash
source .venv/bin/activate
```

## Step-by-step Guide

Make sure you are in the root directory (`path/to/inference_builder`) before you start with any of the microservices. All relative paths and commands assume you are running from the inference_builder root directory. Also ensure that your virtual environment is activated before running any commands.

### Generic CV Microservice for detection, classification and segmentation models.

This microservice supports common CV models including image classification, object detection and segmentation.

#### 1. Generate the inference pipeline using inference builder

```bash
python builder/main.py builder/samples/tao/ds_tao.yaml \
-a builder/samples/tao/openapi.yaml -o builder/samples/tao -t
```

#### 2. Download the model files from NGC and apply the configurations.

Here, we use object detection model `trafficcamnet_transformer_lite` as an example.

```bash
ngc registry model download-version "nvidia/tao/trafficcamnet_transformer_lite:deployable_v1.0" --dest $MODEL_REPO
chmod 777 $MODEL_REPO/trafficcamnet_transformer_lite_vdeployable_v1.0
export TAO_MODEL_NAME=trafficcamnet_transformer_lite_vdeployable_v1.0
cp builder/samples/ds_app/detection/rtdetr/* $MODEL_REPO/$TAO_MODEL_NAME/
chmod 666 $MODEL_REPO/$TAO_MODEL_NAME/*
```

After completing above steps, run `ls $MODEL_REPO/$TAO_MODEL_NAME`, and the below files are expected in the model folder:

```
labels.txt  nvdsinfer_config.yaml  resnet50_trafficamnet_rtdetr.onnx
```

#### 3. Build and run the container image

Building the Docker image from the provided Dockerfile may take 10–20 minutes, depending on your network bandwidth and the hardware in use.

```bash
cd builder/samples
sed -i "s/TAO_MODEL_NAME: .*/TAO_MODEL_NAME: $TAO_MODEL_NAME/" docker-compose.yml
docker compose up tao-cv --build
```

#### 4. Test the microservice

The microservice exposes a REST API that accepts inference requests with images and videos over HTTP, and it works with any frontend that is compatible with the OpenAPI spec. Once the server is ready, an OpenAPI compatible interactive documentation endpoint is available on the server for detailed API usage: http://localhost:8800/docs. Considering the compatibility, we test the raw API using curl commands.

Examples to show the basic inference use cases are listed as below:

##### Run inference on a single image:

If you don't have any test image in hand, you can use [this one](./validation/citysemsegformer/sample_720p.jpg).

**Note:** replace the placeholder `<absolute_path_to_your_file.jpg>` in below command with an actual file with absolute path in your system.


```bash
PAYLOAD=$(echo -n "data:image/jpeg;base64,"$(base64 -w 0 "<absolute_path_to_your_file.jpg>"))
cat > payload.json <<EOF
{
  "input": [ "$PAYLOAD" ]
}
EOF

curl -X POST \
  'http://localhost:8800/v1/inference' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d @payload.json
```

##### Run inference on a single video:

First you need to upload a video file to the server (if you don't have a sample video in hand, follow the [prerequisites](#sample-videos) to get one).

**Note:** replace the placeholder `<your_video.mp4>` in below command with an actual file path in your system.

```bash
export VIDEO_FILE=<your_video.mp4> # replace the placeholder <your_video.mp4> with an actual file
curl -X "POST" \
  "http://localhost:8800/v1/files" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@$VIDEO_FILE;type=video/mp4"
```

The expected response would be like:

{
  "data": {
    "id": "53c2d620-976e-49a4-90a3-3db20b95d225",
    "path": "/tmp/assets/53c2d620-976e-49a4-90a3-3db20b95d225/output.mp4",
    "size": 82223,
    "duration": 2000000000,
    "contentType": "video/mp4"
  }
}

Now you can invoke the inference API based on the data object in the above response following the command below.

**Note:** Please replace the `id` and `path` below with the values from your return response and run the curl command.

```base
curl -X 'POST' \
  'http://localhost:8800/v1/inference' \
  -H 'accept: application/x-ndjson' \
  -H 'Content-Type: application/json' \
  -d '{
  "input": [ {
    "id": "53c2d620-976e-49a4-90a3-3db20b95d225",
    "path": "/tmp/assets/53c2d620-976e-49a4-90a3-3db20b95d225/output.mp4",
    "size": 82223,
    "duration": 2000000000,
    "contentType": "video/mp4"
  } ]
}' -N
```

The inference results are returned in the JSON payload of the HTTP response, including the detected bounding boxes, associated probabilities, labels, and other metadata. For image input, the payload contains a single data object, whereas for video input, it contains multiple data objects—one for each frame. Given the model is trained for traffic scenes, it detects "car", "roadsign", "bicycle", "person" and "background".

### Generic CV Microservice for detection, classification and segmentation on live videos.

This microservice supports common CV models including image classification, object detection and segmentation on live streams.

#### 1. Generate the inference pipeline using inference builder

```bash
python builder/main.py builder/samples/tao/ds_tao_live.yaml \
-a builder/samples/tao/openapi.yaml -o builder/samples/tao -t
```

#### 2. Download the model files from NGC and apply the configurations.

Here, we use object detection model `trafficcamnet_transformer_lite` as an example.

```bash
ngc registry model download-version "nvidia/tao/trafficcamnet_transformer_lite:deployable_v1.0" --dest $MODEL_REPO
chmod 777 $MODEL_REPO/trafficcamnet_transformer_lite_vdeployable_v1.0
export TAO_MODEL_NAME=trafficcamnet_transformer_lite_vdeployable_v1.0
cp builder/samples/ds_app/detection/rtdetr/* $MODEL_REPO/$TAO_MODEL_NAME/
cp builder/samples/tao/config/live_source.yaml $MODEL_REPO/$TAO_MODEL_NAME/
chmod 666 $MODEL_REPO/$TAO_MODEL_NAME/*
```

After completing above steps, run `ls $MODEL_REPO/$TAO_MODEL_NAME`, and the below files are expected in the model folder:

```
labels.txt  live_source.yaml  nvdsinfer_config.yaml  resnet50_trafficamnet_rtdetr.onnx
```

#### 3. Build and run the container image

Building the Docker image from the provided Dockerfile may take 10–20 minutes, depending on your network bandwidth and the hardware in use.

```bash
cd builder/samples
sed -i "s/TAO_MODEL_NAME: .*/TAO_MODEL_NAME: $TAO_MODEL_NAME/" docker-compose.yml
docker compose up tao-cv --build
```

#### 4. Test the microservice

The microservice exposes a REST API that accepts inference requests with images and videos over HTTP, and it works with any frontend that is compatible with the OpenAPI spec. Once the server is ready, an OpenAPI compatible interactive documentation endpoint is available on the server for detailed API usage: http://localhost:8800/docs.

Before adding live streams, you need first send an inference request and waiting for the responses:

```bash
curl -X POST \
  'http://localhost:8800/v1/inference' \
  -H 'accept: application/x-ndjson' \
  -H 'Content-Type: application/json' -d '{ "input": []}'
```

Define the RTSP feed you want to perform inference on:

```bash
# replace the RTSP URL with real source
export LIVE_SOURCE="rtsp://10.111.53.230:8554/video-stream"
```

Then you can add an RTSP stream through the Deepstream add/delete REST APIs (mapped to 8803 on the host):

```bash
curl -XPOST 'http://localhost:8803/api/v1/stream/add' -d "{
  \"key\": \"sensor\",
  \"value\": {
     \"camera_id\": \"my_camera_0\",
     \"camera_name\": \"traffic\",
     \"camera_url\": \"$LIVE_SOURCE\",
     \"change\": \"camera_add\",
     \"metadata\": {
         \"resolution\": \"1920 x1080\",
         \"codec\": \"h264\",
         \"framerate\": 30
     }
 },
 \"headers\": {
     \"source\": \"vst\",
     \"created_at\": \"2021-06-01T14:34:13.417Z\"
 }
}"
```

Similarly, you can remove the RTSP stream through the same APIs:

```bash
curl -XPOST 'http://localhost:8803/api/v1/stream/remove' -d "{
    \"key\": \"sensor\",
    \"value\": {
        \"camera_id\": \"my_camera_0\",
        \"camera_name\": \"traffic\",
        \"camera_url\": \"$LIVE_SOURCE\",
        \"change\": \"camera_remove\",
        \"metadata\": {
            \"resolution\": \"1920 x1080\",
            \"codec\": \"h264\",
            \"framerate\": 30
        }
    },
    \"headers\": {
        \"source\": \"vst\",
        \"created_at\": \"2021-06-01T14:34:13.417Z\"
    }
}"
```

### Microservice for Grounding Dino and Mask Grounding Dino models.

This microservice supports Grounding Dino model and Mask Grounding Dino model.

#### 1. Generate the inference pipeline using inference builder


```bash
python builder/main.py builder/samples/tao/ds_gdino.yaml \
-a builder/samples/tao/openapi.yaml -o builder/samples/tao -c builder/samples/tao/processors.py -t
```

#### 2. Download the model files from NGC and apply the configurations.

Here, we use Grounding Dino model as an example.

```bash
ngc registry model download-version "nvidia/tao/grounding_dino:grounding_dino_swin_tiny_commercial_deployable_v1.0" --dest $MODEL_REPO
chmod 777 $MODEL_REPO/grounding_dino_vgrounding_dino_swin_tiny_commercial_deployable_v1.0
export TAO_MODEL_NAME=grounding_dino_vgrounding_dino_swin_tiny_commercial_deployable_v1.0
cp builder/samples/ds_app/gdino/gdino/* $MODEL_REPO/$TAO_MODEL_NAME/
chmod 666 $MODEL_REPO/$TAO_MODEL_NAME/*
```

After completing above steps, run `ls $MODEL_REPO/$TAO_MODEL_NAME`, and the below files are expected in the model folder:

```
experiment.yaml  grounding_dino_swin_tiny_commercial_deployable.onnx  nvdsinfer_config.yaml  nvdspreprocess_config.yaml
```

#### 3. Build and run the container image

Building the Docker image from the provided Dockerfile may take 10–20 minutes, depending on your network bandwidth and the hardware in use.

```bash
cd builder/samples
sed -i "s/TAO_MODEL_NAME: .*/TAO_MODEL_NAME: $TAO_MODEL_NAME/" docker-compose.yml
docker compose up tao-cv --build
```

#### 4. Test  the microservice

The microservice exposes a REST API that accepts inference requests with images and videos over HTTP, and it works with any frontend that is compatible with the OpenAPI spec. Once the server is ready, an OpenAPI compatible interactive documentation endpoint is available on the server for detailed API usage: http://localhost:8800/docs. Considering the compatibility, we test the raw API using curl commands.

Examples to show the basic inference use cases are listed as below:

##### Run inference on a single image:

If you don't have any test image in hand, you can use the one under [builder/samples/tao//validation/citysemsegformer/sample_720p.jpg](./validation/citysemsegformer/sample_720p.jpg).

**Note:** replace the placeholder `<absolute_path_to_your_file.jpg>` in below command with an actual file with absolute path in your system.

```bash
PAYLOAD=$(echo -n "data:image/jpeg;base64,"$(base64 -w 0 "<absolute_path_to_your_file.jpg>"))
cat > payload.json <<EOF
{
  "input": [ "$PAYLOAD" ],
  "text": [ ["car", "people"] ]
}
EOF

curl -X POST \
  'http://localhost:8800/v1/inference' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d @payload.json
```

##### Run inference on a single video:

First you need to upload a video file to the server (if you don't have a sample video in hand, follow the [prerequisites](#sample-videos) to get one).

**Note:** replace the placeholder `<your_video.mp4>` in below command with an actual file path in your system.

```bash
export VIDEO_FILE=<your_video.mp4> # replace the placeholder <your_video.mp4> with an actual file
curl -X "POST" \
  "http://localhost:8800/v1/files" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@$VIDEO_FILE;type=video/mp4"
```

The expected response would be like:

{
  "data": {
    "id": "53c2d620-976e-49a4-90a3-3db20b95d225",
    "path": "/tmp/assets/53c2d620-976e-49a4-90a3-3db20b95d225/output.mp4",
    "size": 82223,
    "duration": 2000000000,
    "contentType": "video/mp4"
  }
}

Now you can invoke the inference API based on the data object in the above response following the command below.

**Note:** Please replace the `id` and `path` below with the values from your return response and run the curl command.


```bash
curl -X 'POST' \
  'http://localhost:8800/v1/inference' \
  -H 'accept: application/x-ndjson' \
  -H 'Content-Type: application/json' \
  -d '{
  "input": [ {
    "id": "53c2d620-976e-49a4-90a3-3db20b95d225",
    "path": "/tmp/assets/53c2d620-976e-49a4-90a3-3db20b95d225/output.mp4",
    "size": 82223,
    "duration": 2000000000,
    "contentType": "video/mp4"
  } ],
  "text": [
    ["car", "people"]
  ]
}' -N
```

The inference results are returned in the JSON payload of the HTTP response, including the detected bounding boxes, associated probabilities, labels, and other metadata. For image input, the payload contains a single data object, whereas for video input, it contains multiple data objects—one for each frame.

### Microservice for visual changenet models with two image inputs.

This microservice supports Visual Changenet Classification model.

#### 1. Generate the inference pipeline using inference builder.

```bash
python builder/main.py builder/samples/tao/ds_changenet.yaml \
-a builder/samples/tao/openapi.yaml -o builder/samples/tao -t
```

#### 2. Download the model files from NGC and apply the configurations.

```bash
ngc registry model download-version "nvidia/tao/visual_changenet_classification:visual_changenet_nvpcb_deployable_v1.0" --dest $MODEL_REPO
chmod 777 $MODEL_REPO/visual_changenet_classification_vvisual_changenet_nvpcb_deployable_v1.0
export TAO_MODEL_NAME=visual_changenet_classification_vvisual_changenet_nvpcb_deployable_v1.0
cp builder/samples/ds_app/classification/changenet-classify/* $MODEL_REPO/$TAO_MODEL_NAME/
chmod 666 $MODEL_REPO/$TAO_MODEL_NAME/*
```

After completing above steps, run `ls $MODEL_REPO/$TAO_MODEL_NAME`, and the below files are expected in the model folder:

```
changenet-classify.onnx labels.txt nvdsinfer_config.yaml nvdspreprocess_config_0.yaml nvdspreprocess_config_1.yaml
```

#### 3. Build and run the container image

Building the Docker image from the provided Dockerfile may take 10–20 minutes, depending on your network bandwidth and the hardware in use.

```bash
cd builder/samples
sed -i "s/TAO_MODEL_NAME: .*/TAO_MODEL_NAME: $TAO_MODEL_NAME/" docker-compose.yml
docker compose up tao-cv --build
```

#### 4. Test the microservice.

Changenet Classification model detects if a part is missing by comparing the test image against a golden image.

Open the a new terminal and go the inference_builder folder, run the commands from your console:

```bash
GOLDEN_PAYLOAD=$(echo -n "data:image/png;base64,"$(base64 -w 0 "builder/samples/tao/pass_0.png"))
TEST_PAYLOAD=$(echo -n "data:image/png;base64,"$(base64 -w 0 "builder/samples/tao/pass_1.png"))
cat > payload.json <<EOF
{
  "input": [ "$GOLDEN_PAYLOAD", "$TEST_PAYLOAD" ]
}
EOF
curl -X POST \
  'http://localhost:8800/v1/inference' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d @payload.json
```

The inference results are returned in the JSON payload of the HTTP response, including the detected bounding boxes, associated probabilities, labels, and other metadata. For the above sample input, a label of "notdefect" is expected.
