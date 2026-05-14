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


# Sample Dockerfile for a DeepStream TAO CV pipeline (e.g., changenet,
# GDINO) using a Triton-enabled DeepStream base image. This is a
# reference for this specific TAO sample, not a universal template for
# all DeepStream deployments.

# pull the latest DeepStream Triton multiarch image
FROM "nvcr.io/nvidia/deepstream:9.0-triton-multiarch" AS cv_tao_base

# Install system dependencies used to build TensorRT OSS plugins and
# DeepStream TAO post-processors (yaml-cpp, cmake, etc.).
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends libyaml-cpp-dev pkg-config cmake

# Install Python dependencies required by THIS TAO sample (request
# client, FastAPI server, Torch, etc.). Versions are pinned for
# reproducibility and can be adjusted for other pipelines.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    requests==2.32.3 \
    python-multipart==0.0.9 \
    aiofiles==23.2.1 \
    aiohttp==3.11.0 \
    annotated-types==0.7.0 \
    python-magic==0.4.27 \
    torch==2.4.0 \
    numpy==1.26.4 \
    omegaconf==2.3.0 \
    transformers==4.36.2 \
    fastapi==0.115.12 \
    uvicorn==0.29.0

ENV TRANSFORMERS_CACHE=/tmp

# Build TensorRT OSS to provide customized plugins (e.g., ONNX parser)
# and the trtexec tool. The version and GPU architectures are controlled
# by the ARGs below; update them if you target different CUDA/TRT
# versions or GPU architectures.
ARG TRT_VERSION_MAJOR=10
ARG TRT_VERSION_MINOR=14
ARG TRT_VERSION_PATCH=1
ARG TRT_VERSION_BUILD=48

ARG TRT_VERSION_MAJOR_MINOR=$TRT_VERSION_MAJOR.$TRT_VERSION_MINOR
ARG TRT_VERSION_MAJOR_MINOR_PATCH=$TRT_VERSION_MAJOR.$TRT_VERSION_MINOR.$TRT_VERSION_PATCH
ARG TRT_VERSION_FULL=$TRT_VERSION_MAJOR_MINOR_PATCH.$TRT_VERSION_BUILD

ARG CUDA_VERSION_MAJOR=13
ARG CUDA_VERSION_MINOR=1
ARG CUDA_VERSION_PATCH=80
ARG CUDA_VERSION_BUILD=36836380_0
ARG CUDA_VERSION_MAJOR_MINOR=$CUDA_VERSION_MAJOR.$CUDA_VERSION_MINOR
ARG CUDA_VERSION_FULL=$CUDA_VERSION_MAJOR_MINOR.$CUDA_VERSION_PATCH.$CUDA_VERSION_BUILD
ARG CUDNN_VERSION=9.17.0.29

ENV TRT_VERSION=$TRT_VERSION_FULL+cuda$CUDA_VERSION_FULL

WORKDIR /tmp
ENV TRT_TAG="release/$TRT_VERSION_MAJOR_MINOR"
RUN mkdir trt_oss_src && \
   cd trt_oss_src && \
   echo "$PWD Building TRT OSS..." && \
   git clone -b $TRT_TAG https://github.com/nvidia/TensorRT TensorRT && \
   cd TensorRT && \
   git submodule update --init --recursive && \
   mkdir -p build && cd build && \
   cmake .. \
    -DGPU_ARCHS="80;86;90;100;120" \
    -DTRT_LIB_DIR=/usr/lib/x86_64-linux-gnu \
    -DTRT_OUT_DIR=/tmp/out \
    -DCUDA_VERSION=$CUDA_VERSION_MAJOR_MINOR \
    -DCUDNN_VERSION=$CUDNN_VERSION \
    && for i in {1..50}; do make -j 12 && break; done
RUN   cp /tmp/out/libnvinfer_plugin.so.$TRT_VERSION_MAJOR_MINOR_PATCH /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.$TRT_VERSION_MAJOR_MINOR_PATCH && \
   cp /tmp/out/libnvinfer_plugin_static.a /usr/lib/x86_64-linux-gnu/libnvinfer_plugin_static.a && \
   cp /tmp/out/libnvonnxparser.so.$TRT_VERSION_MAJOR_MINOR_PATCH /usr/lib/x86_64-linux-gnu/libnvonnxparser.so.$TRT_VERSION_MAJOR_MINOR_PATCH && \
   cp /tmp/out/trtexec /usr/local/bin/ && \
   cd ../../../ && \
   rm -rf trt_oss_src

# Build DeepStream TAO app post-processors. This must come AFTER the
# DeepStream SDK is installed in the base image. The resulting
# libnvds_infercustomparser_tao.so is copied into the DeepStream lib
# directory for use by TAO pipelines. Adjust the repo/tag or CUDA
# version if you use different DS/TAO releases.
ENV DS_TAO_APPS_REPO=https://github.com/NVIDIA-AI-IOT/deepstream_tao_apps.git
ENV DS_TAO_APPS_TAG="master"
ARG CACHE_BUSTER=unique_value_to_bypass_docker_cache
RUN mkdir ds_tao_apps_src && \
   cd ds_tao_apps_src && \
   echo "$PWD Pulling DS TAO apps repo..." && \
   git clone -b $DS_TAO_APPS_TAG $DS_TAO_APPS_REPO ds_tao_apps && \
   cd ds_tao_apps/post_processor && \
   echo "$PWD Building DS TAO apps post processing..." && \
   CUDA_MODULE_LOADING=LAZY CUDA_VER=$CUDA_VERSION_MAJOR_MINOR make && \
   [ -f libnvds_infercustomparser_tao.so ] && echo "$PWD post processing build successful" || echo "$PWD post processing build falied" && \
   cp libnvds_infercustomparser_tao.so /opt/nvidia/deepstream/deepstream/lib && \
   cd ../../../ && \
   rm -rf ds_tao_apps_src



# Patch JPEG Decoder to avoid using the nvcodec plugin in this sample
# and clear any cached GStreamer state. This workaround is specific to
# this TAO pipeline and may not be required for other deployments.
RUN if [ -e "/usr/lib/x86_64-linux-gnu/gstreamer-1.0/libgstnvcodec.so" ]; then \
        mv /usr/lib/x86_64-linux-gnu/gstreamer-1.0/libgstnvcodec.so /usr/lib/x86_64-linux-gnu/gstreamer-1.0/libgstnvcodec.so_bkp; \
    else \
        echo "File 'libgstnvcodec.so' does not exist"; \
    fi
RUN rm -rf /root/.cache/gstreamer-1.0/
RUN mkdir /tmp/assets && chmod -R 777 /tmp/assets

WORKDIR /workspace

# Copy the inference flow package generated by Inference Builder.
# This tarball (here: tao.tgz) is typically produced when running the
# builder with tarball output enabled, and its name should match the
# generated pipeline artifact you want to deploy.
ADD tao.tgz /workspace

RUN groupadd --gid 1000 --non-unique nvs && \
    useradd --create-home --shell /usr/sbin/nologin --uid 1000 --non-unique --gid 1000 nvs && \
    chown -R 1000.1000 /workspace && chmod o+w /workspace

USER nvs:1000

## Set Environment variables required for DeepStream/Triton runtime.
ENV LD_LIBRARY_PATH /opt/tritonserver/lib:$LD_LIBRARY_PATH
ENV NVSTREAMMUX_ADAPTIVE_BATCHING=yes

# Create the entrypoint script that prepares the TAO model symlink in
# the model repository cache and then runs the Inference Builder-
# generated __main__.py for this TAO pipeline.
RUN touch /workspace/start_server.sh && \
    chmod a+rx /workspace/start_server.sh && \
    cat > /workspace/start_server.sh <<-EOF
	#!/usr/bin/env bash
    set -eu

	# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
	# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
    # Check if the symlink exists
    SYMLINK="/workspace/.cache/model-repo/tao"
    if [ -L "\$SYMLINK" ]; then
        # If it exists, delete the symlink
        rm "\$SYMLINK"
        echo "Symlink deleted: \$SYMLINK"
    fi
    ln -s /workspace/.cache/model-repo/\${TAO_MODEL_NAME} "\$SYMLINK"
    python3 __main__.py
#    while true; do sleep 5; done
EOF

ENTRYPOINT ["/workspace/start_server.sh"]
CMD []

