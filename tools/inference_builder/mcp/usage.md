# MCP Server — User Guide

Human-readable reference for the Inference Builder MCP server. For agent-facing documentation (workflow guidance, troubleshooting), see [README-MCP.md](README-MCP.md).

---

## `generate_inference_pipeline`

Generates a deployable inference application from a YAML configuration. Produces a `{name}.tgz` archive in the session workspace.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `config` | string | Yes | — | Pipeline YAML as inline text. |
| `server_type` | string | No | `serverless` | `serverless` for standalone batch apps; `fastapi` for HTTP microservices. |
| `api_spec` | string | No | — | OpenAPI spec as inline YAML/JSON. Required for `fastapi`, `triton`, and `nim` server types. |
| `custom_modules` | string[] | No | — | Inline Python processor modules. Each must define a class with a `name` attribute and `__call__` method. |

**Returns (success):** JSON with `status: "success"` and `url: "http://{mcp_server}/{name}.tgz"` (HTTP mode) or the local filesystem path (stdio mode).

**Returns (failure):** JSON with `status: "error"` and `url: "http://{mcp_server}/logs/{name}-generate.log"` pointing to the captured error output.

---

## `get_system_info`

Returns hardware and software information from the deployment machine where the MCP server is running — which may be a remote machine, not necessarily the same as the agent or user's local environment. No parameters.

Call this before building a Docker image to select the correct base image and platform flags.

**Returns:** JSON object with:
- `arch` — e.g. `x86_64`, `aarch64`
- `os` — `id`, `version`, `pretty_name`
- `cuda_version`
- `gpus` — list of `name`, `driver_version`, `memory_mib`, `compute_capability`
- `docker` — version string

---

## `build_docker_image`

Builds a Docker image from a generated inference pipeline. The Dockerfile should consume the `.tgz` produced by `generate_inference_pipeline` (e.g. `ADD <pipeline>.tgz /app`).

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `image_name` | string | Yes | Name for the Docker image. |
| `dockerfile` | string | Yes | Dockerfile as inline text. Written to the session workspace alongside the pipeline `.tgz`, which serves as the build context. |

> **Tip:** Read `doc/platform-guide.md` first to pick the right base image for your target hardware (x86_64 datacenter, Jetson/Tegra, or arm-sbsa servers like GB10/GB300/DGX Spark).

---

## `docker_run_image`

Runs a Docker image with optional model repository mounting. Useful for testing and troubleshooting after `prepare_model_repository` and `build_docker_image`.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `image_name` | string | Yes | — | Docker image to run. |
| `model_repo_container` | string | No | — | Container path where the model repository is mounted. Must exactly match `model_repo` in the pipeline config (baked in at build time). Omit if no model repo is needed. |
| `server_type` | string | No | `serverless` | Type hint to choose networking defaults. Non-serverless servers run on the host network. |
| `env` | object | No | — | Environment variables (`KEY: VALUE` pairs). |
| `cmd` | string[] | No | — | Command-line arguments after the image name. For serverless flows, pass inputs as `--<name> <value>` flags using **hyphens** (not underscores): e.g. `["--media-url", "/path/to/video.mp4"]`. |
| `timeout` | integer | No | `300` | Seconds to wait before timing out. For persistent servers, this is how long to wait before collecting logs. |

**Returns:** JSON with `container_name`, `image_name`, `gpu_device` (index of the GPU selected), `status` (`"running"`, `"exited"`, or `"timeout"`), `exit_code` when exited, and `url: "http://{mcp_server}/logs/{container}.log"` (HTTP mode) when a log file was captured.

---

## `docker_stop_container`

Stops and removes a running Docker container by name.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `container_name` | string | Yes | — | Use the `container_name` from `docker_run_image`'s response. |
| `remove_image` | boolean | No | `false` | Also remove the Docker image after stopping the container. |

**Returns:** JSON with `container_name` and `status` (`"removed"` on success). When `remove_image` is true, also includes `image_name` and `image_status`.

---

## `docker_fetch_log`

Fetches logs from a running or stopped Docker container. All lines include Docker timestamps.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `container_name` | string | Yes | — | Container to fetch logs from. |
| `tail` | integer | No | all | Number of lines to return from the end. |
| `since` | string | No | — | Return only lines after this point. Accepts a Docker timestamp (`2024-01-15T10:30:00Z`) or relative duration (`5s`, `2m`, `1h`). Pass `last_timestamp` from a previous call for incremental polling. |
| `follow_seconds` | integer | No | — | Stream the live log for this many seconds before returning (max 60). |

**Returns:** Raw log text plus a JSON summary with `container_name`, `line_count`, and `last_timestamp` (for incremental polling).

---

## `prepare_model_repository`

Downloads models from NGC or Hugging Face and copies runtime config files into a model repository layout. Models are stored under the server's shared model root (`MCP_MODEL_ROOT`).

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model_configs` | object[] | Yes | List of model configuration objects (see below). |

Each item in `model_configs`:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Model identifier used in the pipeline config. Files land in `{model_repo}/{name}/`. |
| `source` | string | No | `"NGC"` or `"HF"` (default `"NGC"`). |
| `path` | string | No | NGC registry path (e.g. `nvidia/tao/grounding_dino`) or HF repo id (e.g. `Qwen/Qwen2.5-VL-3B-Instruct`). |
| `version` | string | No | NGC version tag (e.g. `grounding_dino_swin_tiny_commercial_deployable_v1.0`). Required for NGC. |
| `configs` | object | No | `{filename: inline_content}` dict of runtime config files to write into the model directory. |
| `post_script` | string | No | Inline shell script executed after download, from the model directory. |

> **Tip:** If the NGC model path or version is unknown, run `ngc registry model info <org>/<team>/<model_name>` to find correct values before calling this tool.

**Returns:** JSON with a `models` list, each entry having `name`, `host_path`, `files`, and `status`.

---

## `generate_nvinfer_config`

Generates a DeepStream nvinfer runtime configuration file (`nvdsinfer_config.yaml`) for placement in the model repository.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `onnx_file` | string | Yes | — | ONNX model filename (must end in `.onnx`). Use `read_model_file` to inspect the directory if unknown. |
| `network_type` | integer | Yes | — | `0`=detection, `1`=classification, `2`=segmentation, `3`=instance_segmentation, `100`=custom (raw tensor output). |
| `input_dims` | string | Yes | — | Input shape as `channel;height;width` (e.g. `3;224;224`). |
| `label_file` | string | Yes | — | Label filename (e.g. `labels.txt`), one class per line. |
| `precision_mode` | integer | No | `2` | `0`=FP32, `1`=INT8, `2`=FP16. |
| `custom_lib_path` | string | No | — | Path to a `.so` custom output-parsing library. Required for all models except classic ResNet when `network_type` is 0–3. For TAO models, build from the [deepstream_tao_apps post_processor](https://github.com/NVIDIA-AI-IOT/deepstream_tao_apps/tree/master/post_processor). |
| `custom_parse_func` | string | No | — | Exported symbol name of the custom parse function. Not required for `network_type=100`. |
| `num_classes` | integer | No | — | Number of classes; derivable from the label file. |
| `gie_unique_id` | integer | No | `1` | Unique GIE ID; use distinct values for multiple inference engines. |
| `net_scale_factor` | number | No | `1/255` | Input normalization scale factor. **Must match training.** |
| `offsets` | string | No | — | Per-channel mean subtraction as `R;G;B` (e.g. `123.675;116.28;103.53`). **Must match training.** |
| `classifier_threshold` | number | No | `0.0` | Confidence threshold (classification only). |
| `input_tensor_from_meta` | integer | No | `0` | Set to `1` when using nvdspreprocess for custom preprocessing. |
| `output_tensor_meta` | integer | No | `0` | Set to `1` to output raw tensors for downstream custom processing. |

> **Important:** `net_scale_factor` and `offsets` must exactly match your model's training preprocessing. Incorrect values silently produce poor accuracy.
> - See [QUICK_NORMALIZATION_REFERENCE.md](QUICK_NORMALIZATION_REFERENCE.md) for common patterns.
> - See [STD_NORMALIZATION_CALCULATOR.md](STD_NORMALIZATION_CALCULATOR.md) for models with per-channel std normalization.

> **Note:** Review at least one example under `builder/samples/` (look for `nvdsinfer_config.yaml` files) to understand the expected structure. Also check whether the NGC model already includes an nvinfer config — if so, use its values as the basis.

---

## `read_model_file`

Reads a text file from the shared model root (`MCP_MODEL_ROOT`). Use this to inspect files downloaded by `prepare_model_repository`.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `path` | string | Yes | Relative path within the model root (e.g. `trafficcamnet/nvdsinfer_config.yaml`). Absolute paths and `../` traversal are rejected. |

**Returns:** JSON with `path` and `content` (file text), or `error` on failure.

---

## HTTP Mode — Session Metadata

In HTTP mode every tool response appends a trailing content block:

```json
{"session_id": "<id>"}
```

Use the `session_id` to construct workspace URLs (see below). In stdio mode this block is omitted.

---

## HTTP Mode — Session Workspace Browser

Each HTTP session has an isolated workspace directory. Files produced by tool calls (pipeline archives, container logs) are accessible via:

```
GET http://<host>:<port>/{session_id}/{path}
```

- **File** — served with auto-detected MIME type (`text/plain` for `.log`/`.yaml`/`.json`, `application/octet-stream` for `.tgz`, etc.)
- **Directory** — returns a JSON listing: `{"path": "...", "entries": [{"name": "...", "type": "file|directory", "size": ...}, ...]}`
- **Root listing** — `GET /{session_id}/` lists all files in the session workspace

Container logs are always written under `logs/`:

```
GET http://<host>:<port>/{session_id}/logs/{container_name}.log
```

The `url` field returned by `generate_inference_pipeline` and `docker_run_image` already contains the full path template with `{mcp_server}` as a placeholder — replace it with the actual host and port.

---

## Server Setup

### Local / Debug Mode

Run the server directly from the Inference Builder component directory to see full debug output:

```bash
python -u mcp/mcp_server.py
```

### Remote Server Setup

For deployments where the MCP server runs on a separate host, use `mcp/server_manager.sh` to manage the server process:

```bash
# Start (picks up MCP_API_KEY and MCP_WORKSPACE_ROOT from the environment)
./mcp/server_manager.sh start --port 8888

# Start with explicit options
./mcp/server_manager.sh start --port 8888 --api-key MY_SECRET \
    --workspace-root /var/tmp/ib-workspaces \
    --model-root /var/tmp/ib-models

# Check health
./mcp/server_manager.sh status

# Watch live logs
./mcp/server_manager.sh logs

# Stop
./mcp/server_manager.sh stop
```

Configure the remote client by pointing it at the `/mcp` endpoint:

```json
{
  "mcpServers": {
    "deepstream-inference-builder": {
      "url": "http://<host>:<port>/mcp",
      "headers": { "Authorization": "Bearer <token>" }
    }
  }
}
```

Drop the `headers` block when running without `--api-key`.
