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

# nvdsframeselector

`nvdsframeselector` is a GStreamer transform element that reduces a video stream to representative frames. It supports equidistant selection, SAD-based range selection, and optical-flow-based selection for streams with sporadic motion.

The repo's build discovers this plug-in automatically. Build the complete plug-in stage with:

```bash
CUDA_VER=13.2 bash build/build.sh --only=gst-plugins
```

Or build this plug-in directly:

```bash
CUDA_VER=13.2 make -C src/gst-plugins/gst-nvdsframeselector
```

## Key properties

| Property | Description |
|---|---|
| `enable` | Enables frame selection. |
| `cache-size` | Number of received frames retained before a selection decision. |
| `selection-count` | Number of frames selected from each cache window. |
| `frame-selection-algorithm` | `0`: BASIC/equidistant; `1`: RANGE_BASED/SAD; `2`: optical flow (default). |
| `enable-motion-detection` | Enables optical-flow motion detection. |
| `optical-flow-interval` | Analyze every Nth received frame; `0` automatically targets an approximately 0.5-second optical-flow interval. |
| `source-fps` | Native stream FPS; use when the input is decimated before this element. |
| `equidistant-output` | Emit evenly spaced frames while retaining the count selected by the configured algorithm. |
| `bypass-mode` | Pass all frames through without selection. |

Use `gst-inspect-1.0 nvdsframeselector` to view the complete property list and defaults. The `feed_fselect` utility in this directory demonstrates a pipeline that feeds the selector and exposes the commonly used options.

## Optical Flow headers

The required NVIDIA Optical Flow SDK interface headers are carried under `includes/`. Their MIT license is preserved in the headers and recorded in `THIRD_PARTY_LICENSES.txt`.
