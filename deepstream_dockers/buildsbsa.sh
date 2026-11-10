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

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
for variable in VPI_LIB_URL VPI_DEV_URL VPI_LIB_SHA256 VPI_DEV_SHA256; do
    if [[ -z "${!variable:-}" ]]; then
        echo "Missing ${variable}: supply genuine SBSA VPI4.1.6 inputs before Docker build." >&2
        exit 1
    fi
done
for digest in "$VPI_LIB_SHA256" "$VPI_DEV_SHA256"; do
    [[ "$digest" =~ ^[0-9a-f]{64}$ ]] || { echo 'VPI checksums must be lowercase SHA256.' >&2; exit 1; }
done
sdk_tarball=${DS_SDK_TARBALL:-deepstream_sdk_v${NVDS_VERSION:-9.2.1}_sbsa.tbz2}
[[ -f "sbsa_dockerfiles/$sdk_tarball" ]] || { echo "Stage the matching SDK tarball: sbsa_dockerfiles/$sdk_tarball" >&2; exit 1; }
sudo docker build --platform linux/arm64 --network host --progress=plain \
    --build-arg "DS_SDK_TARBALL=$sdk_tarball" \
    --build-arg "VPI_LIB_URL=$VPI_LIB_URL" --build-arg "VPI_DEV_URL=$VPI_DEV_URL" \
    --build-arg "VPI_LIB_SHA256=$VPI_LIB_SHA256" --build-arg "VPI_DEV_SHA256=$VPI_DEV_SHA256" \
    -t "deepstream:${NVDS_VERSION:-9.2.1}-triton-arm-sbsa-local" \
    -f sbsa_dockerfiles/Dockerfile_SBSA_Triton "$@" .
