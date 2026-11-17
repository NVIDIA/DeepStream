# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#################################################################

# Example script to build the SBSA docker (SBSA triton devel).
# Run ./setup_sbsa_build.sh first.
#
# Prefer a native ARM64 host. On x86 run ./setup_x86_cross_compile_sbsa.sh
# first and expect a long build: opentelemetry-cpp and gRPC compile from
# source under emulation.

cd sbsa_dockerfiles

sudo docker build --platform linux/arm64 --network host --progress=plain -t deepstream:9.2.0-triton-arm-sbsa-local -f Dockerfile_SBSA_Triton ..

cd ..
