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

# Usage: bash buildjet.sh [triton|samples] [additional docker build arguments].
# Base images are resolved only by the requested Docker build, never setup.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
image_type=${1:-triton}
if [[ $# -gt 0 ]]; then shift; fi
arguments=()
case "$image_type" in
    triton)
        : "${JETSON_TRITON_IMAGE:?Supply a matching JetPack7.2.2/TRT11.4.1 server image}"
        : "${JETSON_TRITON_SDK_IMAGE:?Supply its matching Jetson Triton client SDK image}"
        arguments+=(--build-arg "JETSON_TRITON_IMAGE=$JETSON_TRITON_IMAGE"
                    --build-arg "JETSON_TRITON_SDK_IMAGE=$JETSON_TRITON_SDK_IMAGE")
        recipe=Dockerfile_Jetson_Devel ;;
    samples)
        : "${JETSON_RUNTIME_IMAGE:?Supply a JetPack7.2.2/CUDA13.2.2/TRT11.4.1 runtime image}"
        arguments+=(--build-arg "JETSON_RUNTIME_IMAGE=$JETSON_RUNTIME_IMAGE")
        recipe=Dockerfile_Jetson_Run ;;
    *) echo 'Expected triton or samples.' >&2; exit 1 ;;
esac
sdk_tarball=${DS_SDK_TARBALL:-deepstream_sdk_v${NVDS_VERSION:-9.2.1}_jetson.tbz2}
[[ -f "jetson_dockerfiles/$sdk_tarball" ]] || { echo "Stage the matching SDK tarball: jetson_dockerfiles/$sdk_tarball" >&2; exit 1; }
sudo docker build --platform linux/arm64 --network host --progress=plain \
    --build-arg "DS_SDK_TARBALL=$sdk_tarball" \
    "${arguments[@]}" -t "deepstream-l4t:${NVDS_VERSION:-9.2.1}-${image_type}-local" \
    -f "jetson_dockerfiles/${recipe}" "$@" .
