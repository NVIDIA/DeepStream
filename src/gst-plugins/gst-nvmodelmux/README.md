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

# nvmodelmux

`nvmodelmux` is a DeepStream GStreamer bin that routes dynamic input streams to per-stream primary and optional shadow inference models behind one batched input and one batched output. It supports runtime model and stream-routing control through the DeepStream REST/control-message interfaces.

## Inference position

`nvmodelmux` runs at either inference position, selected by the `process-mode` property following nvinfer's own values: `1` = primary (full frame), `2` = secondary (on the objects a primary detected). A pipeline may hold several stages; a control request names a MODEL, never a stage, and the stage whose catalog declares that name answers it. Model names must therefore be unique across every `nvmodelmux` in the pipeline, which is validated at start-up.

At a SECONDARY stage:

- Exactly one model infers, serving every stream that reaches it. `model/load` makes another version resident but IDLE; an all-streams `model/routing` promotes it, and the displaced version stays warm so the reverse call is instant.
- There is no per-stream routing and no A/B shadow lane. Both are refused with a reason naming the constraint rather than a generic error.
- Backends: nvinfer and nvdsvisionencoder (object embeddings). A backend used here must annotate the batch IN PLACE; one that replaces the buffer has its output silently discarded, so the stage refuses to run rather than produce nothing (see the batch-metadata check in `gstnvmodelmux_sgie.c`).
- The pgie-only settings (`model-sharding`, `per-model-batch-size`, `unified-batch`, `buffer-copy-mode`, `copy-buffer-pool-size`, `compact-idle-models-across-pools`, `shadow-model`, `[muxer]`) describe machinery this position does not have. They are ignored, and any that are set are named once at start-up.

**Runtime changes do not survive a full restart.** A model promoted through the API is a live change only: `gst_modelmux_build()` runs on NULL->READY and re-reads the configuration, so a promotion to v2 silently reverts to whatever the config file declares. There is no error -- just different results after the restart. A rollout that is meant to be durable must be written to the config file as well as applied through the API.

## Prerequisites

- CUDA runtime matching the DeepStream build
- GStreamer development packages
- `json-glib`

On Ubuntu:

```bash
sudo apt-get install libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev libjson-glib-dev
```

## Build and install

The repo's build discovers the plug-in automatically:

```bash
CUDA_VER=13.2 bash build/build.sh --only=gst-plugins
```

To build only this plug-in, first build the REST server and custom-helper dependencies, then run:

```bash
CUDA_VER=13.2 make -C src/utils/nvds_rest_server
CUDA_VER=13.2 make -C src/gst-utils/gstnvdscustomhelper
CUDA_VER=13.2 make -C src/gst-plugins/gst-nvmodelmux
```

The plug-in target is `libnvdsgst_modelmux.so`. To inspect a locally built plug-in:

```bash
export GST_PLUGIN_PATH=$PWD/src/gst-plugins/gst-nvmodelmux
gst-inspect-1.0 nvmodelmux
```

## Configuration and diagnostics

Set the element's `config-file-path` property to a model-mux configuration. See [`configs/config_modelmux.txt`](configs/config_modelmux.txt) and the accompanying inference configuration examples. For a sample SECONDARY stage (referenced from a `[secondary-gie]` group with `plugin-type=4`), see [`configs/config_modelmux_sgie_vehicletypes.txt`](configs/config_modelmux_sgie_vehicletypes.txt); the backend it serves is configured the usual way, in [`configs/config_infer_secondary_vehicletypes.txt`](configs/config_infer_secondary_vehicletypes.txt).

### DeepStream test5 A/B sample

The packaged `deepstream-test5` A/B comparison sample uses `nvmodelmux` as
its primary inference backend and runs Trafficcamnet as the primary model and
Car as the shadow model across four sample streams. After a full build, run it
from its configuration directory so relative paths resolve correctly:

```bash
cd src/apps/sample_apps/deepstream-test5/configs/ab_testing
deepstream-test5-app -c test5_config_file_src_modelmux.txt -t
```

See [`../../apps/sample_apps/deepstream-test5/README`](../../apps/sample_apps/deepstream-test5/README)
for the configuration details and runtime model-status query.

For targeted diagnostics, use `GST_DEBUG=nvmodelmux:5`; `MM_DEBUG=1`, `MM_OVERLAY=1`, and `MM_FRAME_TRACE=1` provide additional runtime detail.
