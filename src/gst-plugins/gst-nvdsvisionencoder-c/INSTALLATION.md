################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

# Installation Guide

## Overview

The `nvdsvisionencoder` plugin now follows the standard DeepStream installation structure.

## Installation Directories

### Standard DeepStream Structure

The plugin installs to the standard DeepStream GST plugins directory:

```
/opt/nvidia/deepstream/deepstream-X.Y/lib/gst-plugins/
```

Where `X.Y` is the DeepStream version (e.g., `9.2`).

### Library Naming

The plugin follows DeepStream naming conventions:
- **Library name:** `libnvdsgst_visionencoder.so`
- **Prefix:** `libnvdsgst_` (standard for DeepStream GST plugins)

## Build and Install

### Build

```bash
# from the repository root
cd src/gst-plugins/gst-nvdsvisionencoder-c
export CUDA_VER=13.4        # x86. Use 13.2 on Jetson, SBSA and IGX Thor. The Makefile fails if unset.
make clean
make -j$(nproc)
```

### Install (requires sudo)

```bash
sudo -E make install
```

This will:
1. Create the directory `/opt/nvidia/deepstream/deepstream-X.Y/lib/gst-plugins/` if it doesn't exist
2. Copy `libnvdsgst_visionencoder.so` to that directory with permissions `755`
3. Display the installation location

**Expected output:**
```
Installing to DeepStream plugin directory...
Installed to: /opt/nvidia/deepstream/deepstream-9.2/lib/gst-plugins/libnvdsgst_visionencoder.so

Plugin installed successfully!
Location: /opt/nvidia/deepstream/deepstream-9.2/lib/gst-plugins/libnvdsgst_visionencoder.so
```

### Install to Docker Container

For DeepStream running in Docker:

```bash
# Build on host
make clean && make -j$(nproc)

# Copy to container
docker cp libnvdsgst_visionencoder.so <container_name>:/opt/nvidia/deepstream/deepstream/lib/gst-plugins/

# Verify
docker exec <container_name> ls -lh /opt/nvidia/deepstream/deepstream/lib/gst-plugins/libnvdsgst_visionencoder.so
```

## Verification

After installation, verify the plugin is detected by GStreamer:

```bash
# Check if plugin is loaded
gst-inspect-1.0 nvdsvisionencoder

# List all DeepStream plugins
ls -lh /opt/nvidia/deepstream/deepstream/lib/gst-plugins/ | grep nvdsgst
```

## Configuration Variables

The Makefile uses the following variables that can be overridden:

```makefile
NVDS_VERSION       := 9.2
GST_INSTALL_DIR    ?= /opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/gst-plugins/
LIB_INSTALL_DIR    ?= /opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/
```

To install to a different location:

```bash
sudo -E make install GST_INSTALL_DIR=/path/to/lib/gst-plugins/
```

## Comparison with Other Plugins

This plugin now follows the same installation pattern as other DeepStream plugins:

| Plugin | Location | Name |
|--------|----------|------|
| nvdsosd | `/opt/nvidia/deepstream/deepstream-X.Y/lib/gst-plugins/` | `libnvdsgst_osd.so` |
| nvtracker | `/opt/nvidia/deepstream/deepstream-X.Y/lib/gst-plugins/` | `libnvdsgst_tracker.so` |
| **visionencoder** | `/opt/nvidia/deepstream/deepstream-X.Y/lib/gst-plugins/` | `libnvdsgst_visionencoder.so` |

## Troubleshooting

### Permission Denied

If you get "Permission denied" during install:
```bash
# Use sudo
sudo -E make install

# Or change ownership
sudo chown -R $USER:$USER /opt/nvidia/deepstream/
```

### Wrong Directory

If the plugin installs to the wrong directory, check:
1. `NVDS_VERSION_MAJOR` and `NVDS_VERSION_MINOR` variables
2. Override them: `make install NVDS_VERSION_MAJOR=X NVDS_VERSION_MINOR=Y`

### Plugin Not Found

If `gst-inspect-1.0` doesn't find the plugin:
1. Check `GST_PLUGIN_PATH` environment variable
2. Verify file permissions: `ls -l /opt/nvidia/deepstream/deepstream/lib/gst-plugins/libnvdsgst_visionencoder.so`
3. Ensure it's executable: `sudo chmod 755 /opt/nvidia/deepstream/deepstream/lib/gst-plugins/libnvdsgst_visionencoder.so`
