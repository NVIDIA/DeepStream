#!/bin/bash
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
# limitations under the License.

set -e

TRITON_DOWNLOADS=/tmp/triton_server_downloads
TRITON_PKG_PATH=${TRITON_PKG_PATH:=https://github.com/triton-inference-server/server/releases/download/v2.60.0/tritonserver2.60.0-agx.tar}
TRITON_BACKEND_DIR=/opt/tritonserver/backends
DEEPSTREAM_LIB_DIR=/opt/nvidia/deepstream/deepstream/lib
TRITON_SERVER_BIN=/opt/tritonserver/bin

echo "Creating ${TRITON_DOWNLOADS} directory ..."
mkdir -p $TRITON_DOWNLOADS

echo "Installing Triton prerequisites ..."
if [[ $EUID -ne 0 ]]; then
    echo "Must be run as root or sudo"
    exit 1
fi

apt-get update && \
    apt-get install -y --no-install-recommends libb64-dev libre2-dev libopenblas-dev

wget -O ${TRITON_DOWNLOADS}/boost.tar.gz https://archives.boost.io/release/1.80.0/source/boost_1_80_0.tar.gz

pushd ${TRITON_DOWNLOADS}
tar xvf boost.tar.gz
cd boost_*/
./bootstrap.sh
./b2 install
popd

echo "Downloading ${TRITON_PKG_PATH} to ${TRITON_DOWNLOADS} ... "
wget -O $TRITON_DOWNLOADS/jetpack.tgz $TRITON_PKG_PATH

echo "Creating ${TRITON_BACKEND_DIR} directory ... "
mkdir -p ${TRITON_BACKEND_DIR}

echo "Creating ${TRITON_SERVER_BIN} directory ... "
mkdir -p ${TRITON_SERVER_BIN}

echo "Extracting the Triton library and backend binaries ..."

tar -xzf $TRITON_DOWNLOADS/jetpack.tgz -C $DEEPSTREAM_LIB_DIR --strip-components=2 tritonserver/lib/libtritonserver.so
tar -xzf $TRITON_DOWNLOADS/jetpack.tgz -C $TRITON_BACKEND_DIR --wildcards --strip-components=2 tritonserver/backends/*
tar -xzf $TRITON_DOWNLOADS/jetpack.tgz -C $TRITON_SERVER_BIN --strip-components=2 tritonserver/bin/tritonserver
chmod -R +r $TRITON_BACKEND_DIR
chmod +rx $TRITON_SERVER_BIN

# Install libssl1.1 (OpenSSL 1.1) for the Triton gRPC client (libgrpcclient.so),
# since Ubuntu 22.04/24.04 ship only OpenSSL 3. It coexists with libssl.so.3.
if ldconfig -p | grep -q libssl.so.1.1; then
    echo "libssl1.1 already installed; skipping ..."
else
    echo "Installing libssl1.1 for the Triton gRPC client ..."
    LIBSSL_DEB=$TRITON_DOWNLOADS/libssl1.1.deb
    LIBSSL_SHA256="dded4572af8b0a9e0310909f211a519cc6409fda31ea81132a77e268b0ec0f2f"
    wget -O $LIBSSL_DEB https://ports.ubuntu.com/pool/main/o/openssl/libssl1.1_1.1.1f-1ubuntu2.24_arm64.deb
    echo "$LIBSSL_SHA256  $LIBSSL_DEB" | sha256sum --check --strict
    dpkg -i $LIBSSL_DEB
fi

wget -O /var/tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/cuda-keyring_1.1-1_all.deb
apt install --yes /var/tmp/cuda-keyring.deb
rm /var/tmp/cuda-keyring.deb
apt update
apt install --yes --no-install-recommends datacenter-gpu-manager-4-core=1:4.4.0-1 datacenter-gpu-manager-4-dev=1:4.4.0-1

ldconfig
echo "cleaning up ${TRITON_DOWNLOADS} directory ..."
rm -rf $TRITON_DOWNLOADS
