# Schema Architecture

This document describes the unified schema architecture for the Inference Builder.

## Design Principles

1. **DRY (Don't Repeat Yourself)**: Common structures are defined once in base schemas
2. **Composition over Inheritance**: Use JSON Schema's `allOf` to compose schemas
3. **Separation of Concerns**: Backend logic is separate from parameter validation
4. **Extensibility**: Easy to add new backends without duplicating code

## Schema Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│                    config.schema.json                       │
│              (Top-level configuration)                      │
│  ┌───────────────────────────────────────────────────────┐ │
│  │ - name, model_repo                                    │ │
│  │ - input[], output[]                                   │ │
│  │ - server (responders)                                 │ │
│  │ - models[]  ←──────────────────────────────────┐     │ │
│  │ - routes                                       │     │ │
│  └────────────────────────────────────────────────│─────┘ │
└─────────────────────────────────────────────────────────────┘
                                                    │
                                                    ▼
                    ┌────────────────────────────────────────┐
                    │   common/base-model.schema.json       │
                    │   (Base Model Structure)               │
                    │  ┌──────────────────────────────────┐  │
                    │  │ - name                           │  │
                    │  │ - backend                        │  │
                    │  │ - max_batch_size                 │  │
                    │  │ - input[]                        │  │
                    │  │ - output[]                       │  │
                    │  │ - parameters (generic)           │  │
                    │  │ - preprocessors[]                │  │
                    │  │ - postprocessors[]               │  │
                    │  └──────────────────────────────────┘  │
                    └────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │         allOf Composition     │
                    │                               │
        ┌───────────▼──────────┐      ┌────────────▼───────────┐
        │  Backend Schemas     │      │   Parameter Schemas    │
        │  (backends/*.json)   │      │  (parameters/*.json)   │
        │                      │      │                        │
        │ ┌──────────────────┐ │      │ ┌────────────────────┐ │
        │ │ deepstream.json  ├─┼──────┼→│ deepstream-params  │ │
        │ └──────────────────┘ │      │ └────────────────────┘ │
        │                      │      │                        │
        │ ┌──────────────────┐ │      │ ┌────────────────────┐ │
        │ │ triton.json      ├─┼──────┼→│ triton-params      │ │
        │ └──────────────────┘ │      │ └────────────────────┘ │
        │                      │      │                        │
        │ ┌──────────────────┐ │      │ ┌────────────────────┐ │
        │ │ vllm.json        ├─┼──────┼→│ vllm-params        │ │
        │ └──────────────────┘ │      │ └────────────────────┘ │
        │                      │      │                        │
        │ ┌──────────────────┐ │      │ ┌────────────────────┐ │
        │ │ tensorrtllm.json ├─┼──────┼→│ tensorrtllm-params │ │
        │ └──────────────────┘ │      │ └────────────────────┘ │
        │                      │      │                        │
        │ ┌──────────────────┐ │      │ ┌────────────────────┐ │
        │ │ polygraphy.json  ├─┼──────┼→│ polygraphy-params  │ │
        │ └──────────────────┘ │      │ └────────────────────┘ │
        │                      │      │                        │
        │ ┌──────────────────┐ │      │ ┌────────────────────┐ │
        │ │ dummy.json       ├─┼──────┼→│ dummy-params       │ │
        │ └──────────────────┘ │      │ └────────────────────┘ │
        └──────────────────────┘      └────────────────────────┘
```

## Schema Composition Example

### DeepStream Backend

The DeepStream backend schema is composed as follows:

```json
{
  "allOf": [
    {
      "$ref": "../common/base-model.schema.json"  // Inherit common structure
    },
    {
      "properties": {
        "backend": {
          "enum": ["deepstream/nvinfer"]          // Restrict backend type
        },
        "parameters": {
          "$ref": "parameters/deepstream-parameters.schema.json"  // Specific params
        }
      }
    }
  ]
}
```

This composition means:
- ✅ The model **must have** all fields from `base-model.schema.json` (name, backend, input, output, etc.)
- ✅ The `backend` field **must be** `"deepstream/nvinfer"`
- ✅ The `parameters` field **must validate** against `deepstream-parameters.schema.json`

## Benefits of This Architecture

### 1. Maintainability
- **Single source of truth**: Common fields defined once in `base-model.schema.json`
- **Easy updates**: Change base schema → all backends updated
- **Clear ownership**: Parameters are isolated in their own files

### 2. Consistency
- **Guaranteed structure**: All backends share the same base structure
- **Type safety**: Common types defined in `definitions.schema.json`
- **Validation rules**: Consistent validation across all backends

### 3. Extensibility
- **Add new backend**: Create 2 files (backend schema + parameters)
- **No duplication**: Reuse base model schema via `allOf`
- **Clear pattern**: Follow existing backend schemas

### 4. Developer Experience
- **Better IDE support**: $ref resolution provides autocomplete
- **Clear documentation**: Each parameter schema documents its options
- **Validation messages**: Precise error messages point to exact issues

## Adding a New Backend

To add a new backend (e.g., `onnxruntime`):

1. **Create parameter schema**: `backends/parameters/onnxruntime-parameters.schema.json`
   ```json
   {
     "$schema": "http://json-schema.org/draft-07/schema#",
     "$id": "https://raw.githubusercontent.com/NVIDIA-AI-IOT/inference_builder/main/schemas/backends/parameters/onnxruntime-parameters.schema.json",
     "title": "ONNX Runtime Backend Parameters",
     "type": "object",
     "properties": {
       "execution_providers": {
         "type": "array",
         "items": {"type": "string"}
       },
       "graph_optimization_level": {
         "type": "string",
         "enum": ["basic", "extended", "all"]
       }
     }
   }
   ```

2. **Create backend schema**: `backends/onnxruntime.schema.json`
   ```json
   {
     "$schema": "http://json-schema.org/draft-07/schema#",
     "$id": "https://raw.githubusercontent.com/NVIDIA-AI-IOT/inference_builder/main/schemas/backends/onnxruntime.schema.json",
     "title": "ONNX Runtime Backend Configuration",
     "allOf": [
       {"$ref": "../common/base-model.schema.json"},
       {
         "properties": {
           "backend": {
             "const": "onnxruntime"
           },
           "parameters": {
             "$ref": "parameters/onnxruntime-parameters.schema.json"
           }
         }
       }
     ]
   }
   ```

3. **Update definitions**: Add `"onnxruntime"` to `common/definitions.schema.json#/definitions/backendTypes`

4. **Update index**: Add entry to `index.json`

That's it! No need to duplicate input/output/preprocessor definitions.

## Schema References

### Internal References (within a schema)
```json
"$ref": "#/definitions/tensorSpec"
```

### External References (to other schemas)
```json
"$ref": "../common/base-model.schema.json"
"$ref": "parameters/deepstream-parameters.schema.json"
"$ref": "definitions.schema.json#/definitions/dataTypes"
```

### GitHub URL References (for public access)
```json
"$id": "https://raw.githubusercontent.com/NVIDIA-AI-IOT/inference_builder/main/schemas/config.schema.json"
```

## Validation Flow

```
User YAML Config
       ↓
config.schema.json (validates top-level)
       ↓
models[] → base-model.schema.json (validates common fields)
       ↓
backend type check → specific backend schema
       ↓
parameters validation → backend-specific parameter schema
       ↓
✓ Valid Configuration
```

## Testing Schemas

### Using the validation script:
```bash
python schemas/validate_config.py builder/samples/vllm/vllm_cosmos.yaml
```

### Using Python directly:
```python
import json
import yaml
from jsonschema import validate

with open('schemas/config.schema.json') as f:
    schema = json.load(f)

with open('builder/samples/vllm/vllm_cosmos.yaml') as f:
    config = yaml.safe_load(f)

validate(instance=config, schema=schema)
```

## Future Enhancements

1. **Conditional validation**: Use `if/then/else` to validate parameters based on backend type
2. **oneOf for backend selection**: Automatically select parameter schema based on backend
3. **Custom formats**: Define custom string formats for paths, URIs, etc.
4. **Schema versioning**: Support multiple schema versions for backward compatibility

