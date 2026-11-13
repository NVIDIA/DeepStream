#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Script for gRPC C++ Installation
# Installs gRPC C++ v1.54.3
# Based on steps provided at: https://grpc.io/docs/languages/cpp/quickstart/
# Two changes with respect to steps in the above link:
# 1. Add -DBUILD_SHARED_LIBS=ON to build shared libraries
# 2. Use 'make -j4' instead of 'make -j' to avoid system becoming unresponsive

set -euo pipefail

if [[ "${1:-}" == "-h" ]] || [[ "${1:-}" == "--help" ]]; then
    echo "Usage: bash gRPC_installation.sh <installation_dir> "
    echo "Installs to \$HOME/.local if installation dir is not provided."
    exit 0;
fi

CMD_PREFIX=()
if [[ $(id -u) -ne 0 ]]; then
    CMD_PREFIX=(sudo)
fi

export MY_INSTALL_DIR="${1:-${HOME}/.local}"
echo "Using $MY_INSTALL_DIR as the gRPC installation directory"

mkdir -p -- "$MY_INSTALL_DIR"
export PATH="$MY_INSTALL_DIR/bin:$PATH"

# Keep downloads/builds private to this invocation. Never reuse or remove a
# caller's /tmp/grpc checkout, and remove our own scratch tree on failure too.
DOWNLOAD_DIR=$(mktemp -d "${TMPDIR:-/tmp}/deepstream-grpc.XXXXXXXX")
readonly DOWNLOAD_DIR
cleanup() {
    local result=$?
    trap - EXIT
    if ! rm -rf -- "$DOWNLOAD_DIR"; then
        echo "Could not remove private gRPC build directory: $DOWNLOAD_DIR" >&2
        if [[ $result -eq 0 ]]; then result=1; fi
    fi
    exit "$result"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

"${CMD_PREFIX[@]}" env DEBIAN_FRONTEND=noninteractive apt-get update
"${CMD_PREFIX[@]}" env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git wget cmake curl
if [[ $(uname -m) == "aarch64" ]]; then
    cmake_platform=Linux-aarch64
else
    cmake_platform=Linux-x86_64
fi
wget -q -O "$DOWNLOAD_DIR/cmake-linux.sh" \
    "https://github.com/Kitware/CMake/releases/download/v3.19.6/cmake-3.19.6-${cmake_platform}.sh"
sh "$DOWNLOAD_DIR/cmake-linux.sh" -- --skip-license "--prefix=$MY_INSTALL_DIR"
"${CMD_PREFIX[@]}" env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends build-essential autoconf libtool pkg-config
git clone --recurse-submodules -b v1.54.3 --depth 1 --shallow-submodules \
    https://github.com/grpc/grpc "$DOWNLOAD_DIR/grpc"

cmake -S "$DOWNLOAD_DIR/grpc" -B "$DOWNLOAD_DIR/grpc/cmake/build" \
      -DgRPC_INSTALL=ON \
      -DgRPC_BUILD_TESTS=OFF \
      -DBUILD_SHARED_LIBS=ON \
      "-DCMAKE_INSTALL_PREFIX=$MY_INSTALL_DIR"

make -C "$DOWNLOAD_DIR/grpc/cmake/build" -j 4
make -C "$DOWNLOAD_DIR/grpc/cmake/build" install

# Update the user's shell only after every install command succeeded. %q keeps
# spaces and shell metacharacters in an explicitly supplied prefix literal.
{
    printf 'export PATH=%q:"${PATH:-}"\n' "$MY_INSTALL_DIR/bin"
    printf 'export LD_LIBRARY_PATH=%q:"${LD_LIBRARY_PATH:-}"\n' "$MY_INSTALL_DIR/lib"
    printf 'export PKG_CONFIG_PATH=%q:"${PKG_CONFIG_PATH:-}"\n' "$MY_INSTALL_DIR/lib/pkgconfig"
} >> "$HOME/.profile"
