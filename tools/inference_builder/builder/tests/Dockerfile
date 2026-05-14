# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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



FROM "nvcr.io/nvidia/deepstream:9.0-triton-multiarch" AS inference_base

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    python-multipart==0.0.9 \
    aiofiles==23.2.1 \
    aiohttp==3.11.0 \
    annotated-types==0.7.0 \
    python-magic==0.4.27 \
    torch==2.4.0 \
    numpy==1.26.4 \
    omegaconf==2.3.0 \
    fastapi==0.115.12 \
    uvicorn==0.29.0 \
    requests==2.32.5

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

ARG TEST_APP_NAME

WORKDIR /workspace
ADD ${TEST_APP_NAME}/test-app.tgz /workspace

## Set Environment variables
ENV GST_PLUGIN_PATH=/opt/nvidia/deepstream/deepstream/lib/gst-plugins
ENV LD_LIBRARY_PATH=/opt/nvidia/deepstream/deepstream/lib:$LD_LIBRARY_PATH
ENV NVSTREAMMUX_ADAPTIVE_BATCHING=yes
ENV TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 9.0 10.0"
# Enable latency measurement
#ENV NVDS_ENABLE_LATENCY_MEASUREMENT=1
#ENV NVDS_ENABLE_COMPONENT_LATENCY_MEASUREMENT=1

ENTRYPOINT ["python3", "/workspace/__main__.py"]
#ENTRYPOINT ["/bin/bash"]
CMD []
