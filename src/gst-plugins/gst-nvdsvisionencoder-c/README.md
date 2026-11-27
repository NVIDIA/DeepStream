<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# NvDsVisionEncoder C Plugin

Pure C/C++ GStreamer plugin for generating ReID embeddings in DeepStream pipelines.

## Overview

This plugin generates ReID (Re-Identification) embeddings for tracked objects using Triton Inference Server and properly attaches them as `NVDS_TRACKER_OBJ_REID_META` to DeepStream metadata.

### Advantages over Python Plugin

1. **Correct Metadata Attachment**: Works directly with C pointers - no wrapper/binding issues
2. **Better Performance**: Native C/C++ code without Python overhead
3. **Reliable Memory Management**: Proper GStreamer/DeepStream memory lifecycle
4. **Downstream Compatibility**: Metadata is correctly visible to all downstream components (test5-app, nvmsgconv, etc.)

## Architecture

```
┌──────────────────┐
│  GStreamer       │
│  Pipeline        │
└────────┬─────────┘
         │ NvBufSurface (NVMM)
         ▼
┌──────────────────────────────────┐
│  nvdsvisionencoder (C Plugin)    │
│  ┌────────────────────────────┐  │
│  │ 1. Extract frame           │  │
│  │ 2. Crop objects            │  │
│  │ 3. Call Triton inference   │  │
│  │ 4. Attach NvDsObjReid meta │  │
│  └────────────────────────────┘  │
└────────┬─────────────────────────┘
         │ NvBufSurface + Metadata
         ▼
┌──────────────────┐
│  Downstream      │
│  (test5, msgconv)│
└──────────────────┘
```

## Building

### Prerequisites

- DeepStream SDK installed (`/opt/nvidia/deepstream/deepstream`)
- CUDA Toolkit, TensorRT and GStreamer development packages
- `CUDA_VER` set to the installed CUDA toolkit version. The Makefile stops with
  `CUDA_VER is not set` otherwise.

### Build Steps

Run from this directory (`src/gst-plugins/gst-nvdsvisionencoder-c` in the repository):

```bash
export CUDA_VER=13.4        # x86. Use 13.2 on Jetson, SBSA and IGX Thor.

# Build the plugin
make

# Install to DeepStream plugin directory (-E keeps CUDA_VER under sudo)
sudo -E make install
```

This creates `libnvdsgst_visionencoder.so` and installs it to
`/opt/nvidia/deepstream/deepstream-<version>/lib/gst-plugins/`. Verify with:

```bash
gst-inspect-1.0 nvdsvisionencoder
```

## Model Download

The plugin needs an embedding model; none is shipped with it. Download the
deployable RADIO-CLIP model from NGC:

<https://catalog.ngc.nvidia.com/orgs/nvidia/tao/models/radio-clip/deployable_v1.0/file-browser>

The download contains `radio-clip_v1.0.onnx`, its external weights file
`radio-clip_v1.0_weights.bin` (~5 GB) and `experiment.yaml`. Keep the `.onnx`
and the `_weights.bin` in the same directory.

SigLIP2 is also supported
(<https://catalog.ngc.nvidia.com/orgs/nvidia/tao/models/siglip_v2>). Do not
rename its `siglip_v2_v2.0.onnx`: the plugin selects SigLIP preprocessing from
"siglip" in the `onnx-model` path, and uses RADIO-CLIP preprocessing otherwise.

## Usage

### GStreamer Pipeline

```bash
gst-launch-1.0 \
    uridecodebin uri=file:///path/to/video.mp4 ! \
    nvstreammux ! \
    nvinfer config-file-path=pgie_config.txt ! \
    nvtracker ! \
    nvdsvisionencoder \
        triton-url=localhost:8001 \
        triton-model=YOUR_TRITON_MODEL \
        batch-size=16 \
        enable=true ! \
    nvmultistreamtiler ! \
    nvdsosd ! \
    nveglglessink
```

`YOUR_TRITON_MODEL` is the name of your own Triton model. It must take an
FP32 input named `pixel_values` (NCHW) and return an output named
`image_embeddings`.

### TensorRT backend example (RADIO-CLIP ONNX from NGC)

```bash
gst-launch-1.0 \
    uridecodebin uri=file:///path/to/video.mp4 ! mux.sink_0 \
    nvstreammux name=mux batch-size=1 width=1920 height=1080 ! \
    nvinfer config-file-path=pgie_config.txt ! \
    nvtracker ll-lib-file=/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so ! \
    nvdsvisionencoder backend=tensorrt \
        onnx-model=/path/to/radio-clip_v1.0.onnx batch-size=16 ! \
    fakesink
```

The TensorRT engine is built from the ONNX on first start, which can take
several minutes. To reuse a prebuilt engine, set `tensorrt-engine=/path/to/model.plan`
instead.

### Properties

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| enable | boolean | true | Enable/disable vision encoding |
| backend | string | triton | Inference backend: `triton` or `tensorrt` |
| tensorrt-engine | string | NULL | Path to a TensorRT engine (`.plan`); can be changed while PLAYING to reload |
| onnx-model | string | NULL | Path to an ONNX model; used to build the engine if none is found (cold start only) |
| triton-url | string | localhost:8001 | Triton server URL |
| triton-model | string | radio_clip_trt | Triton model name |
| batch-size | uint | 16 | Maximum batch size for inference |
| min-crop-size | uint | 32 | Minimum object size (width/height) |
| verbose | boolean | false | Enable verbose logging |
| gpu-id | uint | 0 | GPU device ID |

### DeepStream Config File

```ini
[visionencoder]
enable=1
backend=triton
triton-url=localhost:8001
triton-model=YOUR_TRITON_MODEL
batch-size=16
min-crop-size=32
verbose=1
gpu-id=0
```

## Implementation Status

### ✅ Completed

- [x] GStreamer plugin skeleton
- [x] Property handling
- [x] Triton client C++ wrapper
- [x] Metadata attachment (NvDsObjReid)
- [x] Memory management (copy/release callbacks)
- [x] Build system
- [x] Frame extraction, object cropping and preprocessing (`preprocess.cu`)
- [x] TensorRT backend (`trt_encoder.cpp`)

### 🚧 TODO

- [ ] Performance profiling

## Metadata Format

The plugin attaches `NVDS_TRACKER_OBJ_REID_META` with the following structure:

```c
typedef struct _NvDsObjReid {
    unsigned int featureSize;  // Embedding dimension (e.g., 1024)
    float *ptr_host;           // Embedding data on CPU
    float *ptr_dev;            // Embedding data on GPU (NULL)
} NvDsObjReid;
```

Downstream components can read this metadata:

```c
// In deepstream-test5 or message converter
for (NvDsUserMetaList *l = obj_meta->obj_user_meta_list; l != NULL; l = l->next) {
    NvDsUserMeta *user_meta = (NvDsUserMeta *)l->data;
    if (user_meta->base_meta.meta_type == NVDS_TRACKER_OBJ_REID_META) {
        NvDsObjReid *reid = (NvDsObjReid *)user_meta->user_meta_data;
        // Use reid->ptr_host[0...reid->featureSize-1]
    }
}
```

## Testing

Install the plugin first (see Build Steps), then run deepstream-test5 with a
config that includes a `[visionencoder]` section:

```bash
deepstream-test5-app -c configs/test_config.txt
```

For a quick standalone check, run one of the pipelines in [Usage](#usage) with
the model downloaded from NGC.

Check that:
1. Plugin loads successfully
2. Triton connection established
3. Metadata is attached (check DEBUG-TEST5 logs showing `reid_meta_count > 0`)
4. JSON output contains `"reid"` and `"embedding"` fields

## Performance

Expected performance with optimized implementation:
- **Frame extraction**: ~5-10ms (NvBufSurface → CPU/GPU)
- **Cropping**: ~1-2ms per object
- **Triton inference**: ~20-30ms for batch of 16 crops
- **Metadata attachment**: <1ms
- **Total**: ~30-50ms for 16 objects = **20-30 FPS**

## Troubleshooting

### Plugin not found
```bash
export GST_PLUGIN_PATH=/path/to/plugin/directory
gst-inspect-1.0 nvdsvisionencoder
```

### Triton connection failed
```bash
# Check Triton server is running
curl localhost:8000/v2/health/ready

# Check model is loaded
curl localhost:8000/v2/models/YOUR_TRITON_MODEL/ready
```

### No metadata in output
- Enable verbose mode: `verbose=1`
- Check logs for attachment success
- Verify downstream component reads `NVDS_TRACKER_OBJ_REID_META`

## References

- [DeepStream SDK Documentation](https://docs.nvidia.com/deepstream/9.2)
- [GStreamer Plugin Writer's Guide](https://gstreamer.freedesktop.org/documentation/plugin-development/)
- [Triton Client Libraries](https://github.com/triton-inference-server/client)
