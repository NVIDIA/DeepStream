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

# Usage: bash buildx86.sh [triton|samples] [additional docker build arguments].
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
image_type=${1:-triton}
if [[ $# -gt 0 ]]; then shift; fi
case "$image_type" in
    triton) recipe=Dockerfile_triton_x86 ;;
    samples) recipe=Dockerfile_samples_x86 ;;
    *) echo 'Expected triton or samples.' >&2; exit 1 ;;
esac
sdk_tarball=${DS_SDK_TARBALL:-deepstream_sdk_v${NVDS_VERSION:-9.2.1}_x86_64.tbz2}
[[ -f "x86_dockerfiles/$sdk_tarball" ]] || { echo "Stage the matching SDK tarball: x86_dockerfiles/$sdk_tarball" >&2; exit 1; }
sudo docker build --platform linux/amd64 --network host --progress=plain \
    --build-arg "DS_SDK_TARBALL=$sdk_tarball" \
    -t "deepstream:${NVDS_VERSION:-9.2.1}-${image_type}-local" \
    -f "x86_dockerfiles/${recipe}" "$@" .
