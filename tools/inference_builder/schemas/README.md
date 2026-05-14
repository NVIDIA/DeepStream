# Inference Builder JSON Schemas

This directory contains JSON Schema definitions for NVIDIA Inference Builder YAML configurations. These schemas provide validation, autocompletion, and documentation for creating inference pipeline configurations.

## Overview

The Inference Builder allows you to create inference pipelines with various backends such as DeepStream, TensorRT, Triton, TensorRT-LLM, vLLM, etc. These schemas help validate your configuration files and provide IDE support.

## Schema Organization

```
schemas/
├── config.schema.json              # Main configuration schema
├── backends/                       # Backend-specific schemas
│   ├── deepstream.schema.json     # DeepStream backend
│   ├── triton.schema.json         # Triton backend
│   ├── vllm.schema.json           # vLLM backend
│   ├── tensorrtllm.schema.json    # TensorRT-LLM backend
│   ├── polygraphy.schema.json     # Polygraphy backend
│   ├── dummy.schema.json          # Dummy backend
│   └── parameters/                # Backend-specific parameters
│       ├── deepstream-parameters.schema.json
│       ├── triton-parameters.schema.json
│       ├── vllm-parameters.schema.json
│       ├── tensorrtllm-parameters.schema.json
│       ├── polygraphy-parameters.schema.json
│       └── dummy-parameters.schema.json
├── common/                         # Common definitions
│   ├── base-model.schema.json     # Base model schema (shared structure)
│   ├── definitions.schema.json    # Shared type definitions
│   ├── preprocessors.schema.json  # Preprocessor schemas
│   └── postprocessors.schema.json # Postprocessor schemas
└── README.md                       # This file
```

### Schema Composition

The schemas use JSON Schema's composition features for better maintainability:

- **Base Model Schema** (`common/base-model.schema.json`): Defines the common structure shared by all backends (name, backend, input, output, max_batch_size, preprocessors, postprocessors)
- **Backend Schemas** (`backends/*.schema.json`): Extend the base model using `allOf` and specify which backend types are valid
- **Parameter Schemas** (`backends/parameters/*.schema.json`): Define backend-specific parameters with validation rules

This approach:
- ✅ Eliminates duplication across backend schemas
- ✅ Ensures consistency in common fields
- ✅ Makes it easy to add new backends
- ✅ Provides clear validation for backend-specific options

## Main Configuration Schema

The main schema (`config.schema.json`) defines the structure of inference builder configuration files:

### Required Fields

- `name`: String identifier for the microservice
- `model_repo`: Path to the directory containing model files
- `models`: Array of model specifications (at least one required)

### Optional Fields

- `input`: Array of input tensor specifications (required when pipeline includes multiple models or when inputs use custom types)
- `output`: Array of output tensor specifications (required when pipeline includes multiple models or when outputs use custom types)
- `server`: Server configuration including responders (not required for serverless)
- `routes`: Route map for tensor flow between models (required for multi-model pipelines)
- `postprocessors`: Top-level postprocessors (for consolidating outputs from multiple models)

### Example Configuration

```yaml
name: "my_service"
model_repo: "/workspace/model-repo"

input:
  - name: "text"
    data_type: TYPE_STRING
    dims: [-1]
  - name: "images"
    data_type: TYPE_CUSTOM_IMAGE_BASE64
    dims: [-1]
    optional: true

output:
  - name: "output"
    data_type: TYPE_FP32
    dims: [-1, -1, -1]

models:
  - name: "my_model"
    backend: "triton/tensorrt"
    max_batch_size: 1
    input:
      - name: "input0"
        data_type: TYPE_FP32
        dims: [3, 768, 768]
    output:
      - name: "output0"
        data_type: TYPE_FP32
        dims: [10, 768, 768]
```

## Backend-Specific Schemas

### DeepStream Backend

**Files**:
- Schema: `backends/deepstream.schema.json`
- Parameters: `backends/parameters/deepstream-parameters.schema.json`

Used for DeepStream-based inference pipelines.

**Backend Type**: `deepstream/nvinfer`

**Parameters** (defined in `deepstream-parameters.schema.json`):
- `infer_config_path`: Array of paths to DeepStream nvinfer runtime configuration files (required). The configuration file typically accompanies the ONNX model from NGC as a YAML or TXT file - always prefer using this provided configuration when available. If no configuration file is provided with the model, one must be created based on the model architecture, inference implementation details, and post-processing approaches.
- `preprocess_config_path`: Array of paths to nvdspreprocess configuration files (optional)
- `batch_timeout`: Timeout in microseconds for batching multiple inputs
- `inference_timeout`: Inference timeout in seconds
- `tracker_config`: Configuration for object tracking
- `msgbroker_config`: Configuration for message broker integration
- `render_config`: Configuration for rendering and display
- `perf_config`: Performance monitoring configuration
- `kitti_output_path`: Paths for KITTI format output dumps

**Example**:
```yaml
models:
  - name: tao
    backend: deepstream/nvinfer
    max_batch_size: 1
    parameters:
      infer_config_path:
        - nvdsinfer_config.yaml
      preprocess_config_path:
        - nvdspreprocess_config.yaml
      batch_timeout: -1
      inference_timeout: 5
```

### Triton Backend

**Files**:
- Schema: `backends/triton.schema.json`
- Parameters: `backends/parameters/triton-parameters.schema.json`

Used for Triton Inference Server with various frameworks.

**Backend Types**:
- `triton/python`
- `triton/tensorrt`
- `triton/onnx`
- `triton/pytorch`

**Parameters**:
Triton backend parameters are intentionally open and backend-specific. All parameters are converted to `string_value` in the generated `config.pbtxt` file as defined in Triton's ModelConfig protobuf schema ([model_config.proto](https://github.com/triton-inference-server/common/blob/main/protobuf/model_config.proto)).

Common parameters include:
- `FORCE_CPU_ONLY_INPUT_TENSORS`: Controls whether input tensors are kept on GPU or copied to CPU (Triton Python backend). Set to `"no"` to keep tensors on GPU for better performance.
- Backend-specific parameters (e.g., TensorRT engine paths, ONNX model paths, etc.)

For complete parameter documentation, refer to the [Triton Model Configuration documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html).

**Example**:
```yaml
models:
  - name: visual_changenet
    backend: "triton/tensorrt"
    max_batch_size: 1
    parameters:
      FORCE_CPU_ONLY_INPUT_TENSORS: "no"
```

### vLLM Backend

**File**: `backends/vllm.schema.json`

Used for vLLM-based large language model inference.

**Backend Type**: `vllm`

**Key Parameters**:
- `max_num_tokens`: Maximum number of tokens
- `async_mode`: Enable asynchronous mode
- `gpu_memory_utilization`: GPU memory utilization (0.0 to 1.0)
- `tensor_parallel_size`: Number of GPUs for tensor parallelism

**Example**:
```yaml
models:
  - name: "Cosmos"
    backend: "vllm"
    parameters:
      max_num_tokens: 19200
      async_mode: true
```

### TensorRT-LLM Backend

**File**: `backends/tensorrtllm.schema.json`

Used for TensorRT-LLM inference with optional PyTorch fallback.

**Backend Types**:
- `tensorrtllm`
- `tensorrtllm/pytorch`

**Key Parameters**:
- `max_num_tokens`: Maximum number of tokens
- `kv_cache_config`: KV cache configuration
  - `enable_block_reuse`: Enable KV cache block reuse
- `decoding_config`: Decoding configuration

**Example**:
```yaml
models:
  - name: "Qwen2.5-VL"
    backend: "tensorrtllm/pytorch"
    parameters:
      max_num_tokens: 19200
      kv_cache_config:
        enable_block_reuse: false
```

### Polygraphy Backend

**File**: `backends/polygraphy.schema.json`

Used for TensorRT inference via Polygraphy.

**Backend Type**: `polygraphy`

**Key Parameters**:
- `tensorrt_engine`: Path to TensorRT engine file (required)
- `precision`: Inference precision (fp32, fp16, int8)

**Example**:
```yaml
models:
  - name: nvclip_text
    backend: "polygraphy"
    max_batch_size: 64
    parameters:
      tensorrt_engine: "model.plan"
```

## Data Types

The following data types are supported:

### Numeric Types
- `TYPE_BOOL`, `TYPE_UINT8`, `TYPE_UINT16`, `TYPE_UINT32`, `TYPE_UINT64`
- `TYPE_INT8`, `TYPE_INT16`, `TYPE_INT32`, `TYPE_INT64`
- `TYPE_FP16`, `TYPE_FP32`, `TYPE_FP64`

### Basic Types
- `TYPE_STRING`: String data

### Custom Types
- `TYPE_CUSTOM_IMAGE_BASE64`: Base64-encoded JPEG/PNG image data as a data URI string. Each element must follow the format `data:image/{format};base64,{payload}` where `{format}` is `jpeg`, `jpg`, or `png`, and `{payload}` is the base64-encoded image bytes. This type is mapped to `TYPE_STRING` in Triton and arrives as a list of strings. When used as a top-level input, it creates an `ImageInputDataFlow` that parses the data URI prefix to determine the image format, decodes the base64 payload into raw bytes, and decodes the image using a hardware-accelerated DeepStream pipeline (`nvjpegdec`/`pngdec`) to produce RGB `uint8` image tensors in HWC layout `[H, W, 3]` on GPU at the original image resolution. These decoded tensors are then passed to downstream models. When used for an output, image tensors are encoded into a base64 string.
- `TYPE_CUSTOM_IMAGE_ASSETS`: Image asset identifiers (file paths, URLs, or asset IDs) referring to JPEG/PNG images. When used as a top-level input, it creates an `ImageInputDataFlow` that loads and decodes the referenced images into image tensors before they are passed to downstream models.
- `TYPE_CUSTOM_LONG_VIDEO_ASSETS`: Video asset identifiers for **long videos and live streams requiring temporal chunking**. Particularly suitable for: (1) **live streams** such as RTSP feeds that need continuous processing in chunks, (2) long video files that should be split into temporal segments, (3) scenarios where frames need to be delivered in multiple batches over time. Supports file paths, URLs, RTSP streams, or asset IDs, with query parameters such as `?frames=N&interval=INTERVAL_NSEC&chunks=M&scale=WIDTHxHEIGHT`. When used as a top-level input, it creates a `VideoInputDataFlow` that asynchronously extracts frames in chunks, with each chunk containing N frames sampled at the specified interval. Results are delivered via a queue as they become available, enabling streaming processing. Processing always starts from the beginning of the video or live stream. Use `TYPE_CUSTOM_VIDEO_CHUNK_ASSETS` instead for short videos that fit in a single batch without temporal chunking, or when you need to specify a start position. Supported query parameters:
  - `frames`: (Required) Number of frames to sample per chunk.
  - `interval`: (Required) Time interval between frames in nanoseconds (e.g., `100*1e6` for 100ms between frames).
  - `chunks`: (Optional) Maximum number of chunks to yield before stopping. Useful for RTSP streams that don't have a natural end-of-stream.
  - `scale`: (Optional) Target resolution for frame scaling in format `WIDTHxHEIGHT` (e.g., `scale=640x360`). When specified, all extracted frames are resized to the target resolution. This is useful for reducing memory usage and inference time when processing high-resolution video streams. All assets in a batch must use the same scale value.
- `TYPE_CUSTOM_VIDEO_CHUNK_ASSETS`: Video asset identifiers for **short videos requiring simple frame sampling without temporal chunking**. Use this type when:
  1. Processing short video clips that fit in memory as a single batch
  2. Extracting a fixed number of frames from a specific video segment
  3. Simple uniform frame sampling from a video duration is needed

  Supports file paths, URLs, or asset IDs, with query parameters such as `?frames=N&start=START_NSEC&duration=DURATION_NSEC`. When used as a top-level input, it creates a `VideoFrameSamplingDataFlow` that asynchronously extracts exactly N frames evenly distributed over the specified duration (interval is automatically calculated as duration/frames), decodes them, and produces a single frame batch delivered via queue. The result is one batch containing all requested frames. For long videos or live streams requiring temporal chunking and streaming delivery of multiple batches over time, use `TYPE_CUSTOM_LONG_VIDEO_ASSETS` instead. Supported query parameters:
  - `frames`: (Required) Number of frames to sample.
  - `start`: (Optional) Start timestamp in nanoseconds (default 0).
  - `duration`: (Optional) Duration to sample from in nanoseconds (defaults to entire video).
- `TYPE_CUSTOM_BINARY_BASE64`: Base64-encoded binary data. For inputs, the strings are automatically decoded into uint8 tensors before being passed downstream. For outputs, binary tensors are base64-encoded into strings.
- `TYPE_CUSTOM_LIST`: Converts the input values into a list, which implies explicit batching of the input data. Use this for non-DS inputs that need list conversion (e.g., video URLs for VLM models).
- `TYPE_CUSTOM_DS_MEDIA_URL`: DeepStream media URL input. A string URL (file path or remote URL) pointing to an image or video for DeepStream pipeline processing. Unlike `TYPE_CUSTOM_LIST`, this type follows the implicit batch path and is recognized by the DeepStream backend as a media source. Must be paired with a `TYPE_CUSTOM_DS_MIME` input.
- `TYPE_CUSTOM_VLM_INPUT`: Vision-Language Model input in dictionary form, passed through to downstream models as-is.
- `TYPE_CUSTOM_OBJECT`: Generic custom object type for complex structured data.

### DeepStream-Specific Types

#### TYPE_CUSTOM_DS_IMAGE
DeepStream custom type for image buffers, usable as both input and output.

**As Input:** A 1-D uint8 tensor (or list of tensors) containing encoded JPEG/PNG image bytes that can be passed directly into DeepStream backends without additional decoding in the Python runtime. **This input must be defined in the model configuration to enable the image decoder in the DeepStream pipeline.**

**As Output:** Extracts the decoded image frames from the DeepStream pipeline and returns them as RGB uint8 image tensors in HWC (Height, Width, Channels) layout. Each tensor has shape `[H, W, 3]` where Channels is 3 (R, G, B), representing the decoded frame at its processed resolution.

**Example Output Usage:**
```yaml
output:
  - name: "decoded_frames"
    data_type: TYPE_CUSTOM_DS_IMAGE
    dims: [-1, -1, 3]
```

#### TYPE_CUSTOM_DS_METADATA
DeepStream custom type for metadata output, containing per-frame detection, classification, and segmentation results. The output is returned as a **dictionary** with the following structure (defined in `definitions/deepstreamMetadata`):

**Output Structure:**
```python
{
  'shape': [height, width],              # Image dimensions in pixels
  'bboxes': [[left, top, right, bottom], ...],  # Bounding boxes for detected objects
  'probs': [0.95, 0.87, ...],           # Confidence scores (0.0-1.0)
  'labels': [["car"], ["person"], ...],  # Classification labels (multi-label support)
  'seg_maps': [array([[...]]), ...],    # Segmentation maps (2D arrays)
  'objects': [1, 2, 3, ...],            # Tracking IDs (when tracker enabled)
  'timestamp': 1234567890               # Buffer PTS in nanoseconds
}
```

**Field Details:**
- **`shape`**: Array of 2 integers `[height, width]` representing frame dimensions
- **`bboxes`**: Array of bounding boxes, each with 4 integers `[left, top, right, bottom]` in pixel coordinates
- **`probs`**: Array of confidence scores (floats between 0.0 and 1.0)
- **`labels`**: Array of arrays of label strings, enabling multi-label classification (e.g., object type and attributes)
- **`seg_maps`**: Array of 2D segmentation maps for semantic/instance segmentation (numpy arrays or lists)
- **`objects`**: Array of tracking IDs (integers) assigned by DeepStream tracker (populated only when tracker is enabled)
- **`timestamp`**: Buffer presentation timestamp in nanoseconds for temporal synchronization

**Important**: This type requires a low-level C++ postprocessor library (e.g., `libnvds_infercustomparser_tao.so`) to be available and copied into the container image. The library can be obtained from:
1. DeepStream SDK built-in parsers
2. TAO model parsers from https://github.com/NVIDIA-AI-IOT/deepstream_tao_apps.git
3. Custom user-implemented parsers

Without such a library, use regular tensor output types (`TYPE_FP32`, etc.) with Python postprocessors instead.

**Example Usage in Response Templates:**
```yaml
responses:
  InferenceResponse: >
    {
      "data": [
        {% for item in response.output %} {
          "shape": {{item['shape']}},
          "bboxes": {{item['bboxes']}},
          "probs": {{item['probs']}},
          "labels": {{item['labels']|tojson}},
          "masks": {{item['seg_maps']}},
          "timestamp": {{item['timestamp']}}
        } {% if not loop.last %}, {% endif %} {% endfor %}
      ]
    }
```

#### TYPE_CUSTOM_DS_MIME
DeepStream custom type for MIME type string (for example, `image/jpeg` or `image/png`) describing the associated DeepStream image or media buffers. Required alongside `TYPE_CUSTOM_DS_IMAGE` or `TYPE_CUSTOM_DS_MEDIA_URL` inputs so the DeepStream pipeline can select the appropriate decoder.

#### TYPE_CUSTOM_DS_SOURCE_CONFIG
DeepStream custom type for path to a YAML source configuration file. When used as a model input, consider using it when sources such as live streams are dynamically added and removed.

## Preprocessors and Postprocessors

Processors transform data at different stages of the inference pipeline. Preprocessors operate before model inference, while postprocessors operate after.

**Base Schema**: `common/base-processor.schema.json`
**Preprocessor Schema**: `common/preprocessors.schema.json`
**Postprocessor Schema**: `common/postprocessors.schema.json`

### Data Flow and Processor Interface

Processors follow a standard interface pattern where data flows through dictionaries:

**Input Flow**:
1. Upstream data (from model outputs, top-level inputs, or previous processors) is provided as a dictionary
2. The processor's `input` field specifies keys to extract from this dictionary, in order
3. Extracted values are passed as positional arguments to the processor's `__call__` method

**Output Flow**:
1. The processor's `__call__` method returns a tuple
2. The tuple length must match the length of the `output` field
3. Returned values are mapped to output names to form a dictionary for downstream consumption

**Example Flow**:
```python
# Upstream data dictionary
upstream = {
    'image': <image_tensor>,
    'text': <text_string>,
    'metadata': <metadata_dict>
}

# Processor configuration
processor = {
    'input': ['text', 'image'],    # Extract in this order
    'output': ['tokens', 'pixels']  # Map results to these names
}

# Processor invocation
result = processor.__call__(upstream['text'], upstream['image'])
# result = (<token_tensor>, <pixel_tensor>)

# Downstream receives
downstream = {
    'tokens': <token_tensor>,
    'pixels': <pixel_tensor>
}
```

### Preprocessors

**File**: `common/preprocessors.schema.json`

Preprocessors transform input data before model inference. They typically handle tasks like:
- Image normalization and resizing
- Text tokenization
- Data format conversions
- Feature extraction

**Required Fields**:
- `kind`: Processor type (`"custom"` or `"builtin"`)
- `name`: Unique identifier for the processor; should match the `name` attribute defined within the processor class
- `input`: Array of input tensor names (extracted from upstream data)
- `output`: Array of output tensor names (mapped to returned tuple)
- `config`: Optional processor-specific configuration

**Example**:
```yaml
preprocessors:
  - kind: "custom"
    name: "image-normalizer"
    input: ["raw_image"]          # Extract 'raw_image' from upstream
    output: ["normalized_image"]   # Return maps to 'normalized_image'
    config:
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
```

**Corresponding Python Implementation**:
```python
class ImageNormalizer:
    name = "image-normalizer"

    def __init__(self, config):
        self.mean = config['mean']
        self.std = config['std']

    def __call__(self, raw_image):
        # Process the input
        normalized = (raw_image - self.mean) / self.std
        # Return tuple matching output length
        return (normalized,)
```

### Postprocessors

**File**: `common/postprocessors.schema.json`

Postprocessors transform model outputs after inference. They typically handle tasks like:
- Output decoding
- Masking and filtering
- Embedding normalization
- Result formatting

**Required Fields**: Same as preprocessors

**Example**:
```yaml
postprocessors:
  - kind: "custom"
    name: "detection-decoder"
    input: ["boxes", "scores", "classes"]  # Extract from model outputs
    output: ["detections"]                  # Return as single output
    config:
      confidence_threshold: 0.5
      max_detections: 100
```

**Corresponding Python Implementation**:
```python
class DetectionDecoder:
    name = "detection-decoder"

    def __init__(self, config):
        self.threshold = config['confidence_threshold']
        self.max_detections = config['max_detections']

    def __call__(self, boxes, scores, classes):
        # Process three inputs
        mask = scores > self.threshold
        filtered_boxes = boxes[mask][:self.max_detections]
        filtered_scores = scores[mask][:self.max_detections]
        filtered_classes = classes[mask][:self.max_detections]

        detections = {
            'boxes': filtered_boxes,
            'scores': filtered_scores,
            'classes': filtered_classes
        }
        # Return tuple with one element
        return (detections,)
```

### Multi-Input, Multi-Output Example

Processors can handle multiple inputs and outputs:

```yaml
preprocessors:
  - kind: "custom"
    name: "dual-stream-processor"
    input: ["reference_image", "test_image", "threshold"]
    output: ["ref_features", "test_features"]
    config:
      feature_dim: 256
```

```python
class DualStreamProcessor:
    name = "dual-stream-processor"

    def __init__(self, config):
        self.feature_dim = config['feature_dim']

    def __call__(self, reference_image, test_image, threshold):
        # Process inputs (receives 3 arguments in order)
        ref_features = self.extract_features(reference_image)
        test_features = self.extract_features(test_image)

        # Apply threshold
        ref_features = ref_features * (ref_features > threshold)
        test_features = test_features * (test_features > threshold)

        # Return tuple matching output length (2 elements)
        return (ref_features, test_features)
```

## Server Configuration

The server section defines how the inference service handles requests and responses.

### Responders

Responders map API operations to request/response transformations using Jinja2 templates.

**Available Responder Keys** (must match templates in `templates/responder/`):

| Responder | Description | Input | Output |
|-----------|-------------|-------|--------|
| `infer` | Primary inference endpoint. Handles model inference requests. Supports streaming responses with `Accept: application/x-ndjson` header. | Top-level pipeline inputs (as defined in config `input` section) | Top-level pipeline outputs (as defined in config `output` section) |
| `add_file` | File upload responder. Adds files to the asset store for later processing. | Multipart file upload (file object with `filename` and `content_type`) | Asset dict: `id`, `file_name`, `content_type`, `created_at` |
| `del_file` | File deletion responder. Removes files from the asset store. | `asset_id` (string, from path parameter) | `status` (boolean, deletion success) |
| `list_files` | File listing responder. Returns a list of files in the asset store. | None | `assets` (array of asset objects with `id`, `file_name`, `content_type`, `created_at`) |
| `add_live_stream` | Live stream addition responder. Registers a new RTSP or video stream for processing. | `url` (string, required), `description` (string, optional), `username` (string, optional), `password` (string, optional) | Asset dict: `id`, `url`, `description`, `created_at` |
| `del_live_stream` | Live stream removal responder. Stops and removes a registered live stream. | `asset_id` (string, from path parameter) | `status` (boolean, deletion success) |
| `list_live_streams` | Live stream listing responder. Returns all registered live streams. | None | `assets` (array of stream asset objects with `id`, `url`, `description`, `created_at`) |
| `healthy_ready` | Health check responder. Reports service health and readiness status. No request/response template mapping needed. | None | HTTP 200 with "Ready" or HTTP 503 with "Service Unavailable" |

**Request/Response Template Mapping**:

- **`requests`**: Jinja2 templates to map HTTP request body to responder inputs. Keys are **OpenAPI request schema/class names** (e.g., `InferenceRequest`, `AddLiveStreamRequest`), values are Jinja2 templates that transform the incoming `request` object into a JSON structure matching the responder's expected inputs.

- **`responses`**: Jinja2 templates to map responder outputs to HTTP response body. Keys are **OpenAPI response schema/class names** (e.g., `InferenceResponse`, `AddFileResponse`), values are Jinja2 templates that transform the `response` object from the responder into the desired HTTP response JSON structure.

**Common Operations** (values for the `operation` field):
- `inference`: General inference operation
- `create_chat_completion_v1_chat_completions_post`: Chat completion for LLMs
- `create_embedding`: Embedding generation
- `add_media_file`, `delete_media_file`, `list_media_files`: Media file management
- `add_live_stream`, `delete_live_stream`, `list_live_streams`: Live stream management
- `health_ready_v1_health_ready_get`: Health check

See [RESPONDERS.md](RESPONDERS.md) for complete documentation.

**Example**:
```yaml
server:
  responders:
    infer:
      operation: inference
      requests:
        InferenceRequest: >
          {
            "images": [{{ request.input[0]|tojson }}]
          }
      responses:
        InferenceResponse: >
          {
            "data": {{ response.output|tojson }},
            "model": "my-model"
          }
```

## Routes

Routes define the flow of tensors between models in multi-model pipelines.

**Format**: `<source>:<tensor_list> : <destination>:<tensor_list>`

- Source/destination can be model names or `:` for top-level
- Tensor list is optional: `["tensor1", "tensor2"]`

**Example**:
```yaml
routes:
  ':["reference_image", "test_image"]': 'visual_changenet'
  'visual_changenet:["output_final"]': ':["output"]'
```

## Using the Schemas

### In VS Code

Add to your workspace settings (`.vscode/settings.json`):

```json
{
  "yaml.schemas": {
    "./schemas/config.schema.json": "*.yaml"
  }
}
```

Or use the GitHub URL for the latest schema:

```json
{
  "yaml.schemas": {
    "https://raw.githubusercontent.com/NVIDIA-AI-IOT/inference_builder/main/schemas/config.schema.json": "*.yaml"
  }
}
```

### With YAML Language Server

Add to your `.yaml-language-server.json`:

```json
{
  "yaml.schemas": {
    "file:///path/to/schemas/config.schema.json": "/*.yaml"
  }
}
```

### Validation with Python

```python
import json
import yaml
from jsonschema import validate

# Load schema
with open('schemas/config.schema.json') as f:
    schema = json.load(f)

# Load and validate config
with open('config.yaml') as f:
    config = yaml.safe_load(f)

validate(instance=config, schema=schema)
```

## Examples

See the `builder/samples/` directory for complete examples of configurations for different backends:

- `builder/samples/dummy/dummy.yaml` - Dummy backend example
- `builder/samples/vllm/vllm_cosmos.yaml` - vLLM backend example
- `builder/samples/qwen/trtllm_qwen.yaml` - TensorRT-LLM backend example
- `builder/samples/tao/ds_tao.yaml` - DeepStream backend example
- `builder/samples/changenet/trt_changenet.yaml` - Triton/TensorRT backend example
- `builder/samples/nvclip/tensorrt_nvclip.yaml` - Polygraphy backend example

## References

- [Inference Builder Documentation](../README.md)
- [JSON Schema Specification](https://json-schema.org/)
- [YAML Language Server](https://github.com/redhat-developer/yaml-language-server)

## Contributing

When adding new backends or features:

1. Update the appropriate schema files
2. Add examples to this README
3. Test with sample configurations
4. Update the `common/definitions.schema.json` if adding new types

## License

SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

