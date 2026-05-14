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


FROM nvcr.io/nvidia/tritonserver:26.01-py3

WORKDIR /workspace/

# Environment variables
ENV PIP_NO_CACHE_DIR=off
ENV CUDA_MODULE_LOADING=LAZY
ENV PYTHONUNBUFFERED=1

RUN apt update
RUN apt-get install -y git
RUN pip install open_clip_torch==3.2.0 onnx==1.20.1 onnxscript==0.1.0 transformers==4.48.1 tokenizers>=0.21
RUN pip install --no-cache-dir --extra-index-url https://pypi.ngc.nvidia.com polygraphy

# Copy files into the container
COPY ./configs ./configs
COPY ./CLIP.png .
COPY ./export_to_onnx.py .
COPY ./optimize.sh .

ENV MAX_BATCH_SIZE=64
ENV INPUT_HEIGHT=224
ENV INPUT_WIDTH=224
ENV TEXT_LENGTH=77

# Make the script executable
RUN chmod +x /workspace/optimize.sh


# Set entrypoint to execute the installation script and then run optimize.sh
ENTRYPOINT ["/bin/bash", "-c", "/workspace/optimize.sh"]