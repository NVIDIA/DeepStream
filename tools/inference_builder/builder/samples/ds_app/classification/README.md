## Introduction

This sample demonstrates how to build a deepstream application with Inference Builder using classification models:
1. changenet-classify: visual_changenet_nvpcb_deployable_v1.0 from https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/visual_changenet_classification
2. pcbclassification: deployable_v1.1 from https://catalog.ngc.nvidia.com/orgs/nvaie/models/pcbclassification

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

### For pcbclassification sample

Please use `pcbclassification` as the directory:

```bash
ngc registry model download-version "nvaie/pcbclassification:deployable_v1.1"
# Move the folder to the model-repo directory, and the sample uses ~/.cache/model-repo by default
mv pcbclassification_vdeployable_v1.1 $MODEL_REPO/pcbclassification
chmod 777 $MODEL_REPO/pcbclassification
cp -r builder/samples/ds_app/classification/pcbclassification/* $MODEL_REPO/pcbclassification/
```

### For changenet-classify sample

Please use `changenet-classify` as the directory:

```bash
ngc registry model download-version "nvidia/tao/visual_changenet_classification:visual_changenet_nvpcb_deployable_v1.0"
mv visual_changenet_classification_vvisual_changenet_nvpcb_deployable_v1.0 $MODEL_REPO/changenet-classify/
# Move the folder to the model-repo directory, and the sample uses ~/.cache/model-repo by default
chmod 777 $MODEL_REPO/changenet-classify
cp -r builder/samples/ds_app/classification/changenet-classify/* $MODEL_REPO/changenet-classify/
```

## Generate the DeepStream Application Package and Build Container Image

Assume you've followed the [top level instructions](../../../README.md#getting-started) to set up the environment and be sure you're in the inference_builder folder, then activate your virtual environment:

```bash
source .venv/bin/activate
```

### For pcbclassification sample

Please use ds_pcb.yaml as the configuration:

#### For x86 Architecture

```bash
python builder/main.py builder/samples/ds_app/classification/ds_pcb.yaml \
    -o builder/samples/ds_app \
    --server-type serverless \
    -t \
&& docker build -t deepstream-app builder/samples/ds_app
```

#### For Tegra Architecture

```bash
python builder/main.py builder/samples/ds_app/classification/ds_pcb.yaml \
    -o builder/samples/ds_app \
    --server-type serverless \
    -t \
&& docker build -t deepstream-app \
    -f builder/samples/ds_app/Dockerfile.tegra \
    builder/samples/ds_app
```

#### For Dgx Spark

```bash
python builder/main.py builder/samples/ds_app/classification/ds_pcb.yaml \
    -o builder/samples/ds_app \
    --server-type serverless \
    -t \
&& docker build \
    -t deepstream-app \
    -f builder/samples/ds_app/Dockerfile.dgxspark \
    builder/samples/ds_app
```

### For changenet-classify sample

Please use ds_changenet.yaml as the configuration:

#### For x86 Architecture

```bash
python builder/main.py builder/samples/ds_app/classification/ds_changenet.yaml \
    -o builder/samples/ds_app \
    --server-type serverless \
    -t \
&& docker build -t deepstream-app builder/samples/ds_app
```

#### For Tegra Architecture

```bash
python builder/main.py builder/samples/ds_app/classification/ds_changenet.yaml \
    -o builder/samples/ds_app \
    --server-type serverless \
    -t \
&& docker build \
    -t deepstream-app \
    -f builder/samples/ds_app/Dockerfile.tegra \
    builder/samples/ds_app
```

#### For Dgx Spark

```bash
python builder/main.py builder/samples/ds_app/classification/ds_changenet.yaml \
    -o builder/samples/ds_app \
    --server-type serverless \
    -t \
&& docker build \
    -t deepstream-app \
    -f builder/samples/ds_app/Dockerfile.dgxspark \
    builder/samples/ds_app
```

## Run the deepstream app with image inputs:

**Note:** The TensorRT engine is generated during the first time run and it takes several minutes.

**Note:** You can optionally set the `$SAMPLE_INPUT` environment variable to point to your input media directory if you want to perform inference on media files stored on your host machine.

**Note:** To save the inference results, append the `-s result.json` option to your `docker run` command.

```bash
# Update this with your actual samples directory path
export SAMPLE_INPUT=/path/to/your/samples/directory
```

### For pcbclassification sample

```bash
# media-url: the path or URL to the input media.
# mime: the media type (e.g., "video/mp4" or "image/jpeg").
export SAMPLE_INPUT=$(realpath builder/samples/ds_app/classification/sample-inputs/)
docker run --rm --network=host --gpus all --privileged --runtime=nvidia \
    -v $SAMPLE_INPUT:/sample_input \
    -v $MODEL_REPO:/workspace/models \
    -v /tmp/.X11-unix/:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video,graphics \
    deepstream-app \
    --media-url /sample_input/IMG_0002_C71.png \
    --mime image/png
```

### For changenet-classify sample

**Note:** For changenet classification model, it requires two images as input. Sample test images (`pass_0.png` and `pass_1.png`) are embedded in the Docker image at `/workspace/test_data/` for testing purposes.

**Using embedded test images (recommended for testing):**

```bash
docker run --rm --network=host --gpus all --privileged --runtime=nvidia \
    -v $MODEL_REPO:/workspace/models \
    -v /tmp/.X11-unix/:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video,graphics \
    deepstream-app \
    --media-url /workspace/test_data/pass_0.png /workspace/test_data/pass_1.png \
    --mime image/png image/png
```

**Using your own images from host:**

```bash
# media-url: the path or URL to the input media.
# mime: the media type (e.g., "video/mp4" or "image/jpeg").
export SAMPLE_INPUT=/path/to/your/images
docker run --rm --network=host --gpus all --privileged --runtime=nvidia \
    -v $SAMPLE_INPUT:/sample_input \
    -v $MODEL_REPO:/workspace/models \
    -v /tmp/.X11-unix/:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video,graphics \
    deepstream-app \
    --media-url /sample_input/<your_image1.png> /sample_input/<your_image2.png> \
    --mime image/png image/png
```

For classification samples, the output is a list of labels defined in labels.txt. e.g, for pcbclassification model, the output label will be "missing" if there is a part missing from the input image.

**Note:** You may see `pybind11::handle::dec_ref()` related errors after the inference is completed. This is a known issue and will be fixed in the next release.