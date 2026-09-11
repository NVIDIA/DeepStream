# Available Responders

This document lists the available responder templates that can be used in the `server.responders` section of the configuration.

## Overview

Responders map API endpoints to their implementation. Each responder corresponds to a template file in `templates/responder/` and handles a specific operation.

## Available Responder Keys

The following keys are valid for use in `server.responders`:

| Key | Template File | Operation | Description |
|-----|--------------|-----------|-------------|
| `infer` | `infer.jinja.py` | `inference` or chat completion | Main inference endpoint for model predictions |
| `add_file` | `add_file.jinja.py` | `add_media_file` | Upload and register a media file as an asset |
| `del_file` | `del_file.jinja.py` | `delete_media_file` | Delete a registered media file |
| `list_files` | `list_files.jinja.py` | `list_media_files` | List all registered media files |
| `add_live_stream` | `add_live_stream.jinja.py` | `add_live_stream` | Add a live stream source |
| `del_live_stream` | `del_live_stream.jinja.py` | `delete_live_stream` | Delete a live stream source |
| `list_live_streams` | `list_live_streams.jinja.py` | `list_live_streams` | List all registered live streams |
| `healthy_ready` | `healthy_ready.jinja.py` | `health_ready_v1_health_ready_get` | Health check endpoint |

## Schema Validation

The schema restricts responder keys to the above list using `propertyNames.enum`. Invalid responder keys will cause validation to fail:

```yaml
server:
  responders:
    infer:           # ✅ Valid
      operation: inference

    custom_op:       # ❌ Error: invalid responder key
      operation: custom
```

**Error message**: `Property name must be one of: infer, add_file, del_file, list_files, add_live_stream, del_live_stream, list_live_streams, healthy_ready`

## Usage Examples

### Basic Inference

```yaml
server:
  responders:
    infer:
      operation: inference
      requests:
        InferenceRequest: >
          {
            "images": {{ request.input|tojson }}
          }
      responses:
        InferenceResponse: >
          {
            "data": {{ response.output|tojson }},
            "model": "my-model"
          }
```

### LLM Chat Completion

```yaml
server:
  responders:
    infer:
      operation: create_chat_completion_v1_chat_completions_post
      requests:
        NIMLLMChatCompletionRequest: >
          {
            "messages": {{ request.messages|tojson }},
            "max_tokens": {{ request.max_tokens }}
          }
      responses:
        NIMLLMChatCompletionResponse: >
          {
            "id": "{{ request._id }}",
            "choices": [...]
          }
```

### File Management

```yaml
server:
  responders:
    add_file:
      operation: add_media_file
      responses:
        AddFileResponse: >
          {
            "data": {
              "id": {{response.id|tojson}},
              "path": {{response.path|tojson}},
              "contentType": {{response.mime_type|tojson}}
            }
          }

    list_files:
      operation: list_media_files
      responses:
        ListFilesResponse: >
          {
            "data": [
              {% for item in response.assets %}
              {
                "id": {{item.id|tojson}},
                "path": {{item.path|tojson}}
              }
              {% if not loop.last %}, {% endif %}
              {% endfor %}
            ]
          }
```

### Health Check

```yaml
server:
  responders:
    healthy_ready:
      operation: health_ready_v1_health_ready_get
```

## Responder Configuration

Each responder has the following structure:

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `operation` | ✅ Yes | string | Name of the operation (must match OpenAPI spec) |
| `requests` | ❌ No | object | Request transformation templates (Jinja2) |
| `responses` | ❌ No | object | Response transformation templates (Jinja2) |

### Request Templates

Transform incoming API requests to the internal inference format:

```yaml
requests:
  RequestTypeName: >
    {
      "field1": {{ request.some_field|tojson }},
      "field2": "{{ request.another_field }}"
    }
```

- Keys can be any string (typically match OpenAPI schema names)
- Values are Jinja2 templates
- Access request data via `request.*` variables

### Response Templates

Transform internal inference results to API responses:

```yaml
responses:
  ResponseTypeName: >
    {
      "result": {{ response.output|tojson }},
      "metadata": {
        "model": "my-model"
      }
    }
```

- Keys can be any string (typically match OpenAPI schema names)
- Values are Jinja2 templates
- Access response data via `response.*` variables

## Common Operations

### Inference Operations

- `inference` - General inference endpoint
- `create_chat_completion_v1_chat_completions_post` - LLM chat completion

### Media File Operations

- `add_media_file` - Upload media file
- `delete_media_file` - Delete media file
- `list_media_files` - List media files

### Live Stream Operations

- `add_live_stream` - Add live stream
- `delete_live_stream` - Delete live stream
- `list_live_streams` - List live streams

### System Operations

- `health_ready_v1_health_ready_get` - Health check

## Template Implementation

Responder templates are located in `templates/responder/`. Each template:

1. **Receives** the processed request data
2. **Calls** the inference pipeline
3. **Returns** the formatted response

Templates use Jinja2 syntax and have access to:
- `request` - Incoming request data
- `response` - Inference pipeline output
- Various helper filters (e.g., `tojson`, `tojinja`)

## Adding Custom Responders

To add a new responder:

1. Create template file: `templates/responder/my_responder.jinja.py`
2. Implement the responder logic
3. Add `"my_responder"` to the enum in:
   - `schemas/config.schema.json` (propertyNames.enum)
   - `schemas/common/definitions.schema.json` (responderNames)
4. Update this documentation

## Related Documentation

- [Usage Guide](../../doc/usage.md) - Server configuration details
- [Architecture](../ARCHITECTURE.md) - How responders fit into the system
- [OpenAPI Specification](../../builder/samples/*/openapi.yaml) - API definitions

## References

- Templates: `templates/responder/*.jinja.py`
- Schema: `schemas/config.schema.json#/definitions/serverConfig`
- Common Definitions: `schemas/common/definitions.schema.json#/definitions/responderNames`

