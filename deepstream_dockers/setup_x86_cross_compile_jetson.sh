#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# limitations under the License


# special docker setup for x86 cross-compile for Jetson
sudo docker --version


sudo apt-get install qemu binfmt-support qemu-user-static
docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
# Verify everything works.
sudo systemctl restart docker
# docker run --rm -t arm64v8/ubuntu uname -m
# Use Docker buildx
# Create builder for building on x86
sudo docker buildx create --name myjetbuilder
# Create a new builder instance with support for multiple architectures
sudo docker buildx use myjetbuilder
# Inspect available platforms and enable ARM64
sudo docker buildx inspect --bootstrap
# Make sure your Dockerfile is compatible with the ARM64 architecture.
# If you need to install architecture-specific packages, use a multi-stage build and specify the appropriate base image for each stage
# Build the multi-arch image:
# Build the image with the `--platform` flag to specify the target architecture:
# docker buildx build --platform linux/arm64 -t myimage:arm64 -f Dockerfile99 .

