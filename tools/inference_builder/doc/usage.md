# Inference Builder Usage

## Command Line Arguments

For generating the inference code with the corresponding server implementation, you can run the following command:

```bash
python builder/main.py -h
usage: Inference Builder [-h] [--server-type [{triton,fastapi,serverless}]] [-o [OUTPUT_DIR]] [-a [API_SPEC]] [-c [CUSTOM_MODULE ...]] [-x] [-t] [-v VALIDATION_DIR] [--no-docker] [--test-cases-abs-path] config

positional arguments:
  config                Path the the configuration

options:
  -h, --help            show this help message and exit
  --server-type [{triton,fastapi,serverless}]
                        Choose the server type
  -o [OUTPUT_DIR], --output-dir [OUTPUT_DIR]
                        Output directory
  -a [API_SPEC], --api-spec [API_SPEC]
                        File for OpenAPI specification
  -c [CUSTOM_MODULE ...], --custom-module [CUSTOM_MODULE ...]
                        Custom python modules
  -x, --exclude-lib     Do not include common lib to the generated code.
  -t, --tar-output      Zip the output to a single file
  -v, --version         show program's version number and exit
  --validation-dir VALIDATION_DIR
                        valid validation directory path to build validator
  --no-docker           Use local OpenAPI Generator instead of Docker for OpenAPI client generation
  --test-cases-abs-path
                        Use absolute paths in generated test_cases.yaml
```
## Configuration File

Before using the tool, you must prepare a YAML configuration file to define the inference flow. If server integration is required, you also need to provide an OpenAPI specification that defines the server and update the configuration with server templates based on the OpenAPI specification.

For comprehensive documentation on configuration schemas, including validation, IDE support, and detailed backend specifications, see [schemas/README.md](../schemas/README.md).

The configuration file is a YAML file that defines the inference flow and server implementation. It contains the following sections:

- **name**: The name of the inference pipeline, which will be also used as the name of the folder for saving the generated inference code or the name of the tarball if '-t' is specified.
- **model_repo**: Specifies the path to the model repository from which the inference pipeline searches and loads the required model files. Each model has a separate folder for their files.
- **models**: List of model definitions. A inference flow can incorporate multiple models, each might be implemented with different backends to achieve optimal performance.
- **input** (optional): Defines the top-level inputs of the inference flow. This field is required only when the pipeline includes multiple models or when at least one input is a custom type—such as in cases that require standard preprocessing like video decoding before the input data is passed to the model.
- **output** (optional): Defines the top-level outputs of the inference flow. This field is required only when the pipeline includes multiple models or when at least one output is custom type—such as in cases that require standard preprocessing like video encoding before the output data is passed out.
- **server** (optional): Defines the endpoint templates for the server implementation. This field is not required if the server type is set to "serverless".
- **routes** (optional): Defines the routing rules for the inference flow when multiple models are involved.
- **postprocessors** (optional): Defines the top-level post-processors for the inference flow. This field is required only when the pipeline includes multiple models and the output of these models need to be consolidated.

A configuration file can be as simple as the following example:

```yaml

name: "detector"
model_repo: "/workspace/models"

models:
- name: rtdetr
  backend: deepstream/nvinfer
  max_batch_size: 4
  input:
  - name: media_url
    data_type: TYPE_CUSTOM_DS_MEDIA_URL
    dims: [ -1 ]
  - name: mime
    data_type: TYPE_CUSTOM_DS_MIME
    dims: [ -1 ]
  output:
  - name: output
    data_type: TYPE_CUSTOM_DS_METADATA
    dims: [ -1 ]
  parameters:
    infer_config_path:
      - nvdsinfer_config.yaml
```

With the above configuration file, you can generate inference code for the RT-DETR object detection model using the DeepStream backend, which takes image or video url as input and produces bounding boxes as output. (Step by step guide can be found in the [detection README](./builder/samples/ds_app/detection/README.md))

Breakdown of the configuration:

- The inference package is named `detector`, and the model search path is set to `/workspace/models`. The pipeline uses one model and will look for the model file (in this case, an ONNX file and a yaml config for Deepstream) under the `rtdetr` directory under `/workspace/models`, as specified by the model name and model_repo.
- Inference is performed using nvinfer from the NVIDIA DeepStream SDK. A corresponding configuration file named `nvdsinfer_config.yaml` must be present under `/workspace/models/rtdetr`.
- The pipeline supports two inputs:
  - media_url: the path or URL to the input media.
  - mime: the media type (e.g., "video/mp4" or "image/jpeg").
- The pipeline supports batch processing with up to 4 media items at a time as indicated by max_batch_size, and the number can be adjusted to suit the capabilities of the hardware platform and the requirements of the model.
- The output is a custom DeepStream metadata dictionary (`TYPE_CUSTOM_DS_METADATA`), which carries per-frame information such as detected bounding boxes, confidence scores, labels, segmentation maps, tracking IDs, and timestamps. The structure is flexible and returned as a dictionary with fields like `shape`, `bboxes`, `probs`, `labels`, `seg_maps`, `objects`, and `timestamp`. For complete structure details, see the [DeepStream-Specific Types section in schemas/README.md](../schemas/README.md#type_custom_ds_metadata).

### Input and Output Definition

Input and output definitions are required at the model level, and in some cases also at the top level of the pipeline. Each definition specifies the name, data type, dimensions, and optional flags for tensors flowing through the pipeline.

For complete documentation on input/output specifications, supported data types (including custom types like TYPE_CUSTOM_IMAGE_BASE64, TYPE_CUSTOM_LONG_VIDEO_ASSETS, etc.), and field descriptions, see the [Data Types section in schemas/README.md](../schemas/README.md#data-types).

### Model Definition

The model definition specifies the inference backend, input/output tensors, batch size, and optional preprocessors/postprocessors. The definition is derived from the Triton model configuration and extended to support various backends including DeepStream, TensorRT, TensorRT-LLM, vLLM, and others.

For complete documentation on model specifications, including:
- Supported backend types and their hierarchical structure (e.g., `deepstream/nvinfer`, `triton/tensorrt`, `tensorrtllm/pytorch`, `vllm`)
- Backend-specific parameters and configuration options
- Input/output tensor definitions
- Preprocessor and postprocessor specifications

See the [Backend-Specific Schemas section in schemas/README.md](../schemas/README.md#backend-specific-schemas).


### Custom Preprocessors and Postprocessors

Custom preprocessors and postprocessors integrate user-defined Python code into the inference flow, offering a new paradigm for programming with neural network models.

Both custom preprocessors and postprocessors are defined using the following specification:
- **kind**: The kind of the processor: "custom" or "auto". Only "custom" processors are supported in the current release.
- **name**: The name of the processor. It must match the "name" defined in the user-implemented processor class.
- **input**: Specifies the names of the processor's inputs in order. This defines how tensors from the inference flow are passed to the processor.
- **output**: Specifies the names of the processor's outputs in order. This defines how the inference flow extracts the tensors from the processor.
- **config**: Defines the processor's configuration as a dictionary. The contents are implementation-specific.

For a list of common preprocessors (image normalizers, tokenizers, VLM loaders) and postprocessors (masking, embedding processors) with their configurations, see the [Preprocessors and Postprocessors section in schemas/README.md](../schemas/README.md#preprocessors-and-postprocessors).

#### Custom Preprocessor/Postprocessor Implementation Requirements

When implementing a custom preprocessor or postprocessor, the class must adhere to the following specification:
- A class variable named `"name"` must be defined to uniquely identify the processor.
- An `__init__` method must be implemented to accept a single argument: a dictionary containing the processor's configuration (`config`).
- A `__call__` method must be defined to accept multiple positional arguments. The number and order of these arguments must match the input definition of the processor in the configuration file.
- The `__call__` method must return a tuple of tensors, in the order specified by the output definition of the processor in the configuration file.
- Input and output data are expected to be either NumPy arrays or PyTorch tensors, unless a custom data type is explicitly specified.

### Server Definition

The server definition is required at the top level unless the server type is set to "serverless". It specifies the server implementation along with the corresponding request and response templates. Both templates must be written in Jinja2, and are used to:
- Extract the necessary inputs from the server request.
- Format the inference outputs into the desired server response.

The "responders" section allows users to map server implementations to operations defined in the OpenAPI specification. Each responder corresponds to a specific endpoint or operation.
For detailed information about responders, operations, and examples, see the [Server Configuration section in schemas/README.md](../schemas/README.md#server-configuration).

### Routing

The routes section is optional and typically used for multi-model inference flows. It defines custom routing rules that control how data flows between different models in the pipeline.

The routes definition is a map where each entry specifies a connection between a source and destination:
- Format: `<source_model>:<tensor_list> : <destination_model>:<tensor_list>`
- If the model name is omitted (e.g., `:`), the tensors are at the top level of the inference flow.
- Tensor lists are optional and specified as JSON arrays (e.g., `["tensor1", "tensor2"]`).

For detailed examples and explanation of routing logic, see the [Routes section in schemas/README.md](../schemas/README.md#routes).

## Runtime Environment Variables

The following environment variables can be used to control runtime behavior of the generated inference pipelines.

### Logging & Debugging

| Variable | Default | Description |
|----------|---------|-------------|
| `DEBUG` | `0` | Enable debug mode (sets log level to DEBUG when non-zero) |
| `LOG_LEVEL` | `INFO` | Log verbosity level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |

### Inference Performance

| Variable | Default | Description |
|----------|---------|-------------|
| `N_CODEC_INSTANCES` | `1` | Number of video codec instances for parallel decoding. Increase for higher throughput when processing multiple video streams. |
| `MAX_BATCH_SIZE` | `16` | Maximum batch size for live stream frame collection. Controls memory usage vs throughput trade-off. |

### Server Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `HTTP_PORT` | `8000` | HTTP port for FastAPI server when using `fastapi` server type. |

### Testing & Validation

| Variable | Default | Description |
|----------|---------|-------------|
| `TEST_HOST` | `http://127.0.0.1:8800` | Target host URL for the test runner to connect to. |
| `TEST_TOLERANCE` | `1e-5` | Numerical tolerance for floating-point comparisons in test assertions. |
| `ERROR_EXPORT_PATH` | `/tmp/inference_errors.json` | File path to export collected errors for test validation and debugging. |

### Dynamic Configuration Override

The generated configuration module (`config.py`) supports dynamic environment variable substitution using the pattern `$ENV_VAR|default_value`. This allows any configuration value to be overridden at runtime.

**Format:** `$VARIABLE_NAME|default_value`

**Example configuration:**
```yaml
models:
- name: my_model
  parameters:
    batch_size: $MODEL_BATCH_SIZE|8
    precision: $MODEL_PRECISION|fp16
```

In this example:
- `batch_size` will use the value of `MODEL_BATCH_SIZE` environment variable if set, otherwise defaults to `8`
- `precision` will use `MODEL_PRECISION` if set, otherwise defaults to `fp16`

Type conversion is automatic based on the default value:
- `"true"`/`"false"` → boolean
- Numeric strings → int or float
- Everything else → string