## Introduction

This sample demonstrates how to build a deepstream application with Inference Builder using grounding dino  models:
1. gdino: grounding_dino_swin_tiny_commercial_deployable_v1.0 from https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/grounding_dino
2. mask_gdino: mask_grounding_dino_swin_tiny_commercial_deployable_v1.0 from https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/mask_grounding_dino

**Note:** For both gdino and mask_gdino models, the steps are the same as described below. Users just need to replace the model name from "gdino" to "mask_gdino" in the commands and directory names.

## Prerequisites

**Note:** Make sure you are in the root directory (`path/to/inference_builder`) to execute the commands in this README. All relative paths and commands assume you are running from the inference_builder root directory. Also ensure that your virtual environment is activated before running any commands.

Model files are loaded from '/workspace/models/{MODEL_NAME}' within the container, thus the volume must be correctly mapped from the host.
You need to export MODEL_REPO environment variable to the path where you want to store the model files.

```bash
mkdir ~/.cache/model-repo/
sudo chmod -R 777 ~/.cache/model-repo/
export MODEL_REPO=~/.cache/model-repo
```

You need first download the model files from the NGC catalog and put them in the $MODEL_REPO/{model-name}/ directory, then copy the other required configurations to the same directory:

**Note:** If NGC commands fail, make sure you have access to the models you are trying to download. Some models require an active subscription. Ensure NGC is set up properly, or alternatively try using the NGC web UI to directly download the model from the links provided [here](../README.md#models-used-in-the-samples)

### for gdino sample

Please use `gdino` as the directory:

```bash
ngc registry model download-version "nvidia/tao/grounding_dino:grounding_dino_swin_tiny_commercial_deployable_v1.0"
# Move the folder to the model-repo directory, and the sample uses ~/.cache/model-repo by default
mv grounding_dino_vgrounding_dino_swin_tiny_commercial_deployable_v1.0 $MODEL_REPO/gdino
chmod 777 $MODEL_REPO/gdino
cp -r builder/samples/ds_app/gdino/gdino/* $MODEL_REPO/gdino/
```

### for mask_gdino sample

Please use `mask_gdino` as the directory:

```bash
ngc registry model download-version "nvidia/tao/mask_grounding_dino:mask_grounding_dino_swin_tiny_commercial_deployable_v1.0"
# Move the folder to the model-repo directory, and the sample uses ~/.cache/model-repo by default
mv mask_grounding_dino_vmask_grounding_dino_swin_tiny_commercial_deployable_v1.0 $MODEL_REPO/mask_gdino
chmod 777 $MODEL_REPO/mask_gdino
cp -r builder/samples/ds_app/gdino/mask_gdino/* $MODEL_REPO/mask_gdino/
```

## Generate the deepstream application package and build it into a container image:

Assume you've followed the [top level instructions](../../../README.md#getting-started) to set up the environment and be sure you're in the inference_builder folder, then activate your virtual environment:

```bash
source .venv/bin/activate
```

### For x86 Architecture

```bash
python builder/main.py builder/samples/ds_app/gdino/ds_gdino.yaml \
    -o builder/samples/ds_app \
    --server-type serverless \
    -c builder/samples/tao/processors.py \
    -t \
&& docker build -t deepstream-app builder/samples/ds_app
```

### For Tegra Architecture

```bash
python builder/main.py builder/samples/ds_app/gdino/ds_gdino.yaml \
    -o builder/samples/ds_app \
    --server-type serverless \
    -c builder/samples/tao/processors.py \
    -t \
&& docker build \
    -t deepstream-app \
    -f builder/samples/ds_app/Dockerfile.tegra \
    builder/samples/ds_app
```

#### For Dgx Spark

```bash
python builder/main.py builder/samples/ds_app/gdino/ds_gdino.yaml \
    -o builder/samples/ds_app \
    --server-type serverless \
    -c builder/samples/tao/processors.py \
    -t \
&& docker build \
    -t deepstream-app \
    -f builder/samples/ds_app/Dockerfile.dgxspark \
    builder/samples/ds_app
```

## Run the deepstream app with different inputs:

**Note:** The TensorRT engine is generated during the first time run and it takes several minutes.

**Note:** You can optionally set the `$SAMPLE_INPUT` environment variable to point to your input media directory if you want to perform inference on media files stored on your host machine.

**Note:** To save the inference results, append the `-s result.json` option to your `docker run` command.

```bash
# Update this with your actual samples directory path
export SAMPLE_INPUT=/path/to/your/samples/directory
```

**Note:** When you set `enable_display: true` under the `render_config` section of
[ds_gdino.yaml](ds_gdino.yaml), you need to have a display on your host and run both commands in this order to give the container access to it. For more information about render configuration options, see the [render configuration section](../README.md#render-configuration).


First, set the display environment variable:
```bash
export DISPLAY=:0  # or :1 depending on your system
```

Then, allow X server connections from any host:
```bash
xhost +
```

If the configuration is successful, you will see this message in the log: `access control disabled, clients can connect from any host`.

### Run with video input

```bash
# media-url: the path or URL to the input media.
# mime: the media type (e.g., "video/mp4" or "image/jpeg").
# text: the text prompt for object detection (e.g., "car,person").
docker run --rm --network=host --gpus all --privileged --runtime=nvidia \
    -v $MODEL_REPO:/workspace/models \
    -v /tmp/.X11-unix/:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video,graphics \
    deepstream-app \
    --media-url /opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4 \
    --mime video/mp4 \
    --text "car,person"
```

### Run with RTSP input

**Note:** Replace `rtsp://<url_path>` (which is just a placeholder) with your actual RTSP stream URL. The application supports various RTSP stream formats including H.264, H.265, and MJPEG.

```bash
# media-url: the path or URL to the input media.
# mime: the media type (e.g., "video/mp4" or "image/jpeg").
# text: the text prompt for object detection (e.g., "car,person").

# Note: Replace rtsp://<url_path> with your actual RTSP stream URL

docker run --rm --network=host --gpus all --privileged --runtime=nvidia \
    -v $MODEL_REPO:/workspace/models \
    -v /tmp/.X11-unix/:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video,graphics \
    deepstream-app \
    --media-url rtsp://<url_path> \
    --mime video/mp4 \
    --text "car,person"
```

### Run with image input

```bash
# media-url: the path or URL to the input media.
# mime: the media type (e.g., "video/mp4" or "image/jpeg").
# text: the text prompt for object detection (e.g., "car,person").

# Note: /sample_input/test.jpg is just a placeholder for any image present in $SAMPLE_INPUT directory

docker run --rm --network=host --gpus all --privileged --runtime=nvidia \
    -v $SAMPLE_INPUT:/sample_input \
    -v $MODEL_REPO:/workspace/models \
    -v /tmp/.X11-unix/:/tmp/.X11-unix \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video,graphics \
    -e DISPLAY=$DISPLAY \
    deepstream-app \
    --media-url /sample_input/test.jpg \
    --mime image/jpeg \
    --text "car,person"
```