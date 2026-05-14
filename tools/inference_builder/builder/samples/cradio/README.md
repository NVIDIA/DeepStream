# PeopleNet Transformer + C-RADIOv3-H Pipeline

A two-stage serverless inference pipeline that detects people/objects in images using PeopleNet Transformer (DeepStream/nvinfer), then generates per-detection embeddings using C-RADIOv3-H (TensorRT/polygraphy). The pipeline runs as a CLI application (serverless, no HTTP server).

We provide a sample Dockerfile for the example, which you can use to build a Docker image and run inference on any x86 system with an NVIDIA Ampere, Hopper, or Blackwell GPU.

## Pipeline Structure

```
Input (image path + MIME)
        |
        v
 +--------------+
 |   PeopleNet   |  DeepStream/nvinfer backend
 |  Transformer  |  - Decodes image (DS_IMAGE)
 |               |  - Runs detection (DS_METADATA)
 |               |  - C++ parser: NvDsInferParseCustomDDETRTAO
 +--------------+
    |          |
    |          +---> detections (direct to output)
    |                {shape, bboxes, probs, labels, ...}
    v
 BboxCropPostprocessor
    - Crops each bbox from decoded frame
    - Resizes to 256x256 (bicubic)
    - CLIP mean/std normalization
    - Returns N [3, 256, 256] crops (batched by model operator)
    |
    v
 +--------------+
 | C-RADIOv3-H  |  polygraphy/TensorRT backend
 |              |  - Batch up to 16 crops
 +--------------+
    |
    v
 Output:
   - detections: {shape, bboxes, probs, labels, seg_maps, objects, timestamp}
   - summary: [N, 3840] per-detection embeddings
   - spatial_features: [N, 256, 1280] per-detection spatial tokens
```

### Data Flow

```
route: input[media_url, mime] --> peoplenet
route: peoplenet[pixel_values] --> cradio_v3_h
route: peoplenet[detections] --> output
route: cradio_v3_h[summary, spatial_features] --> output
```

### Custom Processors

- **BboxCropPostprocessor** (postprocessor on peoplenet): Crops detected bounding boxes from the decoded frame, resizes/normalizes for C-RADIO input, and returns a list of `[3, 256, 256]` tensors. The model operator batches them into `[N, 3, 256, 256]` for the TRT engine.

# Prerequisites

**Note:** Make sure you are in the root directory (`path/to/inference_builder`) to execute the commands in this README. All relative paths and commands assume you are running from the inference_builder root directory. Also ensure that your virtual environment is activated before running any commands.

Refer to the [top-level README](../../../README.md#getting-started) for base setup (virtual environment, protobuf, etc.).

Before downloading the model files, set up your model repository:

```bash
mkdir -p ~/.cache/model-repo && chmod 777 ~/.cache/model-repo
export MODEL_REPO=~/.cache/model-repo
```

# Download Models

## PeopleNet Transformer (NGC)

The PeopleNet Transformer model is available from NGC: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/peoplenet_transformer

Download using the NGC CLI and move the files into the expected model directory:

```bash
ngc registry model download-version \
  nvidia/tao/peoplenet_transformer:deployable_v1.1 \
  --dest /tmp
mv /tmp/peoplenet_transformer_vdeployable_v1.1 $MODEL_REPO/peoplenet
```

Run `ls $MODEL_REPO/peoplenet/` and verify you have `resnet50_peoplenet_transformer_op17.onnx`.

Copy the nvinfer config and labels file to the model directory:

```bash
cp builder/samples/ds_app/detection/peoplenet_transformer/nvdsinfer_config.yaml \
  builder/samples/ds_app/detection/peoplenet_transformer/labels.txt \
  $MODEL_REPO/peoplenet/
```

**Note:** The ONNX model uses custom TensorRT plugins. The Dockerfile builds TRT OSS from source to provide these plugins. The TensorRT engine is built automatically by DeepStream on first run.

**Note:** If NGC commands fail, make sure you have access to the models you are trying to download. Some models require an active subscription. Ensure NGC is set up properly, or alternatively try using the NGC web UI to directly download the model.

## C-RADIOv3-H (HuggingFace)

Make sure you have the HuggingFace CLI installed and are logged in:

```bash
pip install huggingface_hub
hf auth login
```

Or set the `HF_TOKEN` environment variable:

```bash
export HF_TOKEN=<your-hf-token>
```

Then download the model:

```bash
git lfs install
git clone https://huggingface.co/nvidia/C-RADIOv3-H $MODEL_REPO/cradio_v3_h
```

**Note:** ONNX export and TRT engine build for this model happen after the Docker image is built, since they require the GPU and container environment. See [Export C-RADIOv3-H ONNX & Build TRT Engine](#export-c-radiov3-h-onnx--build-trt-engine) below.

# Build the Inference Pipeline

```bash
source .venv/bin/activate
python builder/main.py builder/samples/cradio/ds_cradio.yaml \
  -c builder/samples/cradio/processors.py \
  -o builder/samples/cradio \
  --server-type serverless -t
```

The `-t` flag creates `ds-cradio.tgz` in the output directory (from the `name` in `ds_cradio.yaml`). The Dockerfile expects this tarball.

# Build the Docker Image

```bash
docker build -f builder/samples/cradio/Dockerfile -t cradio:latest builder/samples/cradio
```

# Export C-RADIOv3-H ONNX & Build TRT Engine

## Export to ONNX

Run the export script inside the container (requires GPU):

```bash
docker run --rm --gpus all \
  -v $MODEL_REPO:/models \
  -v $MODEL_REPO/cradio_v3_h:/root/.cache/huggingface/modules/transformers_modules/cradio_v3_h \
  --entrypoint python3 \
  cradio:latest /workspace/export_onnx.py \
    --output-dir /models/cradio_v3_h \
    --resolution 256 \
    --model-path /models/cradio_v3_h
```

## Build TensorRT Engine

Build the TensorRT engine with polygraphy:

```bash
docker run --rm --gpus all \
  -v $MODEL_REPO:/models \
  --entrypoint polygraphy \
  cradio:latest convert /models/cradio_v3_h/model.onnx \
    --fp16 \
    --trt-min-shapes pixel_values:[1,3,256,256] \
    --trt-opt-shapes pixel_values:[8,3,256,256] \
    --trt-max-shapes pixel_values:[16,3,256,256] \
    -o /models/cradio_v3_h/model.plan
```

Run `ls $MODEL_REPO/cradio_v3_h/model.plan` to verify the engine was created.

## Automated Script

Alternatively, `prepare_engine.sh` automates both steps (skips if outputs already exist):

```bash
bash builder/samples/cradio/prepare_engine.sh cradio:latest $MODEL_REPO
```

This is the same script used by the CI test via `prerequisite_script` in `test_config.json`.

# Run Inference

Run inference on a single image (output to stdout):

```bash
docker run --rm --gpus all \
  -v $MODEL_REPO:/models \
  -e MODEL_REPO=/models \
  cradio:latest \
  --media-url <path-to-image> --mime image/jpeg -s stdout
```

To save the output to a file:

```bash
docker run --rm --gpus all \
  -v $MODEL_REPO:/models \
  -e MODEL_REPO=/models \
  cradio:latest \
  --media-url <path-to-image> --mime image/jpeg -s /models/output.json
```

**Important:** Replace `<path-to-image>` with an actual image path accessible inside the container (e.g., mount a volume or use a path already in the image such as `/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.jpg`).

## Expected Output Format

```json
{
  "detections": {
    "shape": [544, 960],
    "bboxes": [[x1, y1, x2, y2], ...],
    "probs": [0.977, 0.953, ...],
    "labels": [["Person"], ["Person"], ...],
    "seg_maps": [],
    "objects": [],
    "timestamp": 1234567890
  },
  "summary": [[...3840 floats...], ...],
  "spatial_features": [[[...1280 floats...] x 256], ...]
}
```

- `detections`: PeopleNet bounding boxes, labels, and probabilities
- `summary`: `[N, 3840]` per-detection embeddings (N = number of detections, max 16)
- `spatial_features`: `[N, 256, 1280]` per-detection spatial tokens
- When no objects are detected, N=1 with a dummy zero-crop embedding

# Model Repository Layout

```
$MODEL_REPO/
  peoplenet/
    resnet50_peoplenet_transformer_op17.onnx
    nvdsinfer_config.yaml
    labels.txt
  cradio_v3_h/
    model.plan              # TensorRT engine (built from ONNX)
    model.onnx              # Exported ONNX model
    ...                     # HuggingFace model files (used for ONNX export)
```

# Project Files

| File | Description |
|------|-------------|
| `ds_cradio.yaml` | Pipeline configuration (models, routes, I/O types) |
| `processors.py` | Custom postprocessor (BboxCropPostprocessor) |
| `Dockerfile` | Docker build (DeepStream base + TRT OSS + TAO parser + pipeline) |
| `export_onnx.py` | C-RADIOv3-H ONNX export script |
| `prepare_engine.sh` | Automated ONNX export + TRT engine build (used by CI) |
| `nvdsinfer_config.yaml` | PeopleNet nvinfer runtime config (copy from `ds_app/detection/peoplenet_transformer/` to `$MODEL_REPO/peoplenet/`) |
| `test_config.json` | Test configuration for CI |
