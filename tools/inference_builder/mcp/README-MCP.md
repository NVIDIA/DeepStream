# MCP Server Internal Documentation

This document provides detailed technical documentation for the Inference Builder MCP server. For getting started and basic usage, see the [main README](../README.md#mcp-integration).

## Tool Reference

### `generate_inference_pipeline`

Generates an inference pipeline from a YAML configuration file.

| Parameter | Required | Description |
|-----------|----------|-------------|
| `config_file` | Yes | Path to the YAML configuration file |
| `server_type` | No | Type of server to generate (`triton`, `fastapi`, `nim`, `serverless`) |
| `output_dir` | No | Output directory for generated code |
| `api_spec` | No | Path to OpenAPI specification file |
| `custom_modules` | No | List of custom Python module files |
| `exclude_lib` | No | Exclude common library from generated code |
| `tar_output` | No | Create a tar.gz archive of the output |
| `validation_dir` | No | Validation directory path to build validator |
| `no_docker` | No | Use local OpenAPI Generator instead of Docker |
| `test_cases_abs_path` | No | Use absolute paths in generated test_cases.yaml |

### `build_docker_image`

Builds a Docker image from a generated inference pipeline.

| Parameter | Required | Description |
|-----------|----------|-------------|
| `image_name` | Yes | Name for the Docker image |
| `dockerfile` | Yes | Path to Dockerfile (must be placed in the output directory) |

### `prepare_model_repository`

Prepares a model repository by downloading models from NGC or Hugging Face and copying configuration files.

| Parameter | Required | Description |
|-----------|----------|-------------|
| `model_configs` | Yes | List of model configurations to prepare |
| `config_dir` | No | Base directory for resolving relative config paths |

### `docker_run_image`

Runs a Docker image with optional model repository mounting and environment configuration.

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `image_name` | Yes | - | Name of the Docker image to run |
| `model_repo_host` | No | - | Host path to model repository |
| `model_repo_container` | No | `/models` | Container path for model repository |
| `server_type` | No | `serverless` | Server type hint |
| `env` | No | - | Environment variables to set |
| `cmd` | No | - | Command-line arguments. For serverless flows, supply input values here using `--<name> <value>` flags. **Use hyphens, not underscores** in argument names (e.g., `media_url` → `--media-url`). Each flag and value is a separate list item: `["--media-url", "/path/to/video.mp4"]` |
| `gpus` | No | `all` | GPU devices to use |
| `timeout` | No | `300` | Timeout in seconds |

### `generate_nvinfer_config`

Generates a DeepStream nvinfer runtime configuration file (nvdsinfer_config.yaml).

**IMPORTANT**: Before generating a new config file, first call `prepare_model_repository` to download the model and check for existing nvinfer configuration files. Models from NGC typically include a configuration file (YAML or TXT) with all required inference parameters - always prefer using this provided configuration when available.

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `output_path` | Yes | - | Path where the generated config file should be saved |
| `onnx_file` | Yes | - | Name of the ONNX model file |
| `network_type` | Yes | - | Network type (0=detection, 1=classification, 2=segmentation, 3=instance_segmentation, 100=custom) |
| `input_dims` | Yes | - | Input dimensions in format `channel;height;width` (e.g., `3;224;224`) |
| `label_file` | Yes | - | Name of the label file |
| `precision_mode` | No | `2` | Precision mode (0=FP32, 1=INT8, 2=FP16) |
| `custom_lib_path` | No | - | Path to a C++ shared library (.so) for custom output parsing. Required for all models except classic ResNet when network_type is 0-3. For TAO models, build from https://github.com/NVIDIA-AI-IOT/deepstream_tao_apps/tree/master/post_processor |
| `custom_parse_func` | No | - | Symbol name of the custom parsing function. (0) Detection: built-in assumes ResNet; (1) Classification: built-in treats as softmax; (2,3) Segmentation: required; (100) Custom: not required |
| `num_classes` | No | - | Number of detected classes |
| `gie_unique_id` | No | `1` | Unique ID for this GIE |
| `net_scale_factor` | No | `0.00392156862745098` | Scale factor for normalization |
| `offsets` | No | - | Mean subtraction values in format `R;G;B` |
| `classifier_threshold` | No | `0.0` | Confidence threshold for classification |
| `input_tensor_from_meta` | No | `0` | Whether to read input from metadata (0 or 1) |
| `output_tensor_meta` | No | `0` | Whether to output raw tensor format (0 or 1) |

**Important**: The `net_scale_factor` and `offsets` parameters must match your model's training preprocessing. See:
- [QUICK_NORMALIZATION_REFERENCE.md](QUICK_NORMALIZATION_REFERENCE.md) - Quick reference guide
- [STD_NORMALIZATION_CALCULATOR.md](STD_NORMALIZATION_CALCULATOR.md) - For models with std normalization

## Resource Navigation

### Documentation Resources

| URI | Description |
|-----|-------------|
| `docs://README.md` | Main project README with overview, getting started, and installation |
| `docs://mcp/README-MCP.md` | This document - MCP server internal documentation |

### Schema Resources

| URI | Description |
|-----|-------------|
| `schema://config.schema.json` | Main JSON Schema for configuration files |
| `schema://readme` | Comprehensive schema documentation |
| `schema://index.json` | Schema navigation index mapping backend types to parameter schemas |
| `schema://backends/{backend}.schema.json` | Backend-specific schemas (deepstream, triton, vllm, etc.) |
| `schema://backends/parameters/{backend}-parameters.schema.json` | Detailed parameter schemas |

**Schema Navigation Flow:**
1. Read `schema://config.schema.json` for the main configuration structure
2. When a model specifies a `backend` type (e.g., `vllm`, `triton/python`), read `schema://index.json` to find the corresponding parameter schema path
3. Read the backend-specific parameter schema to get valid parameter options

### Sample Resources

Sample resources are dynamically discovered and categorized:

| URI Pattern | Description |
|-------------|-------------|
| `samples://config/{path}` | Pipeline/application configuration YAML files |
| `samples://runtime_config/{path}` | DeepStream nvinfer runtime configuration files |
| `samples://runtime_preprocess/{path}` | DeepStream nvdspreprocess runtime configuration files |
| `samples://openapi/{path}` | OpenAPI server specification YAML files |
| `samples://dockerfile/{path}` | Sample Dockerfiles |
| `samples://processor/{path}` | Preprocessor/postprocessor Python modules |

## Usage Examples

### Generate a DeepStream Pipeline

```
Use the generate_inference_pipeline tool to create a DeepStream object detection pipeline using the ds_detect.yaml configuration with serverless output.
```

### Explore Available Samples

```
List the available MCP resources to see sample configurations, Dockerfiles, and processors.
```

### View Sample Configurations

```
Read the samples://config/ds_app/detection/ds_detect.yaml resource to see the DeepStream detection configuration.
```

### Build Docker Image

```
Build a Docker image from the generated deepstream-app pipeline with the name 'my-detector' using the build_docker_image tool.
```

## Workflow Integration

### Typical Development Workflow

1. **Explore Samples**: Browse `samples://config/*` resources to see available configurations
2. **Examine Configurations**: Read sample resources to understand configuration patterns
3. **Create Your Config**: Modify a sample configuration or create your own
4. **Generate Pipeline**: Use `generate_inference_pipeline` to create your code
5. **Build Container**: Use `build_docker_image` to create a deployable container

### Integration with Version Control

The generated code can be:
- Committed to version control
- Used as a starting point for custom modifications
- Shared with team members
- Deployed to production environments

## Troubleshooting

### Common Issues

**MCP Server Not Found**
- Ensure the MCP server is running
- Check that the path in your MCP client configuration is correct
- Verify Python environment has required dependencies

**Configuration Validation Errors**
- Check that required fields are present in your YAML
- Ensure model definitions are complete
- Verify backend specifications are correct

**Pipeline Generation Failures**
- Check that the configuration file path is correct
- Ensure all required dependencies are installed
- Verify that the output directory is writable

### Debug Mode

To run the MCP server with debug output:

```bash
python -u mcp_server.py
```

## Extending the MCP Server

To add new tools or resources:

1. **Add new tools** to the `InferenceBuilderMCPServer` class in `mcp_server.py`
2. **Update tool schemas** in the `list_tools` method
3. **Implement tool logic** in the `call_tool` method
4. **Add tests** for new functionality in `test_mcp_server.py`

### Server Types

The `server_type` parameter supports:

| Type | Description |
|------|-------------|
| `fastapi` | RESTful API server with automatic OpenAPI generation |
| `serverless` | Standalone command-line application |

### Custom Processors

Custom preprocessors and postprocessors can be integrated:

1. Create custom processor classes following the base interface
2. Reference them in your configuration file
3. Use the `custom_modules` parameter when generating pipelines
