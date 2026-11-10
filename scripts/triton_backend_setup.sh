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

# Neither legacy JetPack26.04 archives nor stock SBSA26.10 inputs qualify the
# JetPack7.2.2/TRT11.4.1 target. Require a reviewed explicit server/client pair
# for either ARM path before mkdir, APT, download or any host mutation.
case "${DEEPSTREAM_VARIANT:-${PLATFORM:-}}" in
    sbsa) TRITON_SETUP_SBSA=1 ;;
    jetson|tegra|aarch64) TRITON_SETUP_SBSA=0 ;;
    "")
        TRITON_SETUP_SBSA=0
        if [[ "$(uname -m)" == aarch64 && ! -f /etc/nv_tegra_release ]]; then
            TRITON_SETUP_SBSA=1
        fi
        ;;
    *) echo 'Select DEEPSTREAM_VARIANT=jetson|tegra|sbsa for ARM backend setup' >&2; exit 1 ;;
esac
if [[ -z "${TRITON_PKG_PATH:-}" || -z "${TRITON_CLIENT_SDK_PKG_PATH:-}" ]]; then
    if [[ "$TRITON_SETUP_SBSA" -eq 1 ]]; then
        echo 'SBSA uses the matching Triton26.10 ARM64 container/runtime, not the JetPack26.04 archive default.' >&2
    else
        echo 'Jetson requires explicit JetPack7.2.2/TRT11.4.1-qualified server/client archives; the legacy26.04 pair is not a fallback.' >&2
    fi
    echo 'An archive install requires explicit matching TRITON_PKG_PATH and TRITON_CLIENT_SDK_PKG_PATH; no image is pulled by this script.' >&2
    exit 1
fi
if [[ "$TRITON_SETUP_SBSA" -eq 1 ]]; then
    echo 'Explicit SBSA Triton archive overrides selected; target compatibility remains unverified.' >&2
else
    echo 'Explicit Jetson Triton archive overrides selected; target compatibility remains unverified.' >&2
fi

TRITON_DOWNLOADS=/tmp/triton_server_downloads
TRITON_BACKEND_DIR=/opt/tritonserver/backends
DEEPSTREAM_LIB_DIR=${DEEPSTREAM_LIB_DIR:=/opt/nvidia/deepstream/deepstream/lib}
TRITON_SERVER_BIN=/opt/tritonserver/bin

if [[ ! "${TRITON_PKG_PATH}" =~ /v([0-9]+[.][0-9]+[.][0-9]+)/ ]]; then
    echo "Could not determine Triton server version from TRITON_PKG_PATH"
    exit 1
fi
TRITON_SERVER_VERSION=${BASH_REMATCH[1]}

if [[ ! "${TRITON_CLIENT_SDK_PKG_PATH}" =~ /v([0-9]+[.][0-9]+[.][0-9]+)/ ]]; then
    echo "Could not determine Triton client SDK version from TRITON_CLIENT_SDK_PKG_PATH"
    exit 1
fi
TRITON_CLIENT_SDK_VERSION=${BASH_REMATCH[1]}

if [[ "${TRITON_SERVER_VERSION}" != "${TRITON_CLIENT_SDK_VERSION}" ]]; then
    echo "Triton server (${TRITON_SERVER_VERSION}) and client SDK (${TRITON_CLIENT_SDK_VERSION}) versions must match"
    exit 1
fi

echo "Creating ${TRITON_DOWNLOADS} directory ..."
mkdir -p $TRITON_DOWNLOADS

echo "Installing Triton prerequisites ..."
if [[ $EUID -ne 0 ]]; then
    echo "Must be run as root or sudo"
    exit 1
fi

apt-get update && \
    apt-get install -y --no-install-recommends libb64-dev libre2-dev libopenblas-dev unzip build-essential ca-certificates perl

OPENSSL11_VERSION=1.1.1w
OPENSSL11_INSTALL_DIR=/opt/openssl-${OPENSSL11_VERSION}
OPENSSL11_SOURCE_URL=https://github.com/openssl/openssl/archive/refs/tags/OpenSSL_1_1_1w.tar.gz
OPENSSL11_TARBALL=${TRITON_DOWNLOADS}/openssl-${OPENSSL11_VERSION}.tar.gz
OPENSSL11_SOURCE_DIR=${TRITON_DOWNLOADS}/openssl-OpenSSL_1_1_1w
OPENSSL11_LDCONF_FILE=/etc/ld.so.conf.d/openssl-${OPENSSL11_VERSION}.conf

if [[ ! -x "${OPENSSL11_INSTALL_DIR}/bin/openssl" ]] || \
    ! "${OPENSSL11_INSTALL_DIR}/bin/openssl" version | grep -q "OpenSSL ${OPENSSL11_VERSION}" || \
    ! ldconfig -p | grep -q "libssl.so.1.1" || \
    ! ldconfig -p | grep -q "libcrypto.so.1.1"; then
    echo "OpenSSL 1.1 compatibility libraries not found. Building OpenSSL 1.1.1w for Triton ..."
    wget -O "${OPENSSL11_TARBALL}" "${OPENSSL11_SOURCE_URL}"
    tar xzf "${OPENSSL11_TARBALL}" -C "${TRITON_DOWNLOADS}"

    pushd "${OPENSSL11_SOURCE_DIR}"
    ./config shared --prefix="${OPENSSL11_INSTALL_DIR}" --openssldir="${OPENSSL11_INSTALL_DIR}/ssl" --libdir=lib
    make -j"$(nproc)"
    make install_sw
    popd

    echo "${OPENSSL11_INSTALL_DIR}/lib" > "${OPENSSL11_LDCONF_FILE}"
    ldconfig

    if ! ldconfig -p | grep -q "libssl.so.1.1" || ! ldconfig -p | grep -q "libcrypto.so.1.1"; then
        echo "OpenSSL ${OPENSSL11_VERSION} installation did not expose libssl.so.1.1 and libcrypto.so.1.1"
        exit 1
    fi

    if ! "${OPENSSL11_INSTALL_DIR}/bin/openssl" version | grep -q "OpenSSL ${OPENSSL11_VERSION}"; then
        echo "OpenSSL ${OPENSSL11_VERSION} installation version check failed"
        exit 1
    fi
fi

wget -O ${TRITON_DOWNLOADS}/boost.tar.gz https://archives.boost.io/release/1.80.0/source/boost_1_80_0.tar.gz

pushd ${TRITON_DOWNLOADS}
tar xvf boost.tar.gz
cd boost_*/
./bootstrap.sh
./b2 install
popd

echo "Downloading ${TRITON_PKG_PATH} to ${TRITON_DOWNLOADS} ... "
wget -O $TRITON_DOWNLOADS/jetpack.zip $TRITON_PKG_PATH

echo "Creating ${TRITON_BACKEND_DIR} directory ... "
mkdir -p ${TRITON_BACKEND_DIR}

echo "Creating ${TRITON_SERVER_BIN} directory ... "
mkdir -p ${TRITON_SERVER_BIN}

echo "Extracting the Triton library and backend binaries ..."

TRITON_ZIP_EXTRACT_DIR=${TRITON_DOWNLOADS}/triton_zip_extract
mkdir -p "${TRITON_ZIP_EXTRACT_DIR}"
unzip -q "${TRITON_DOWNLOADS}/jetpack.zip" -d "${TRITON_ZIP_EXTRACT_DIR}"

TRITON_ROOT_DIR=$(find "${TRITON_ZIP_EXTRACT_DIR}" -type d -path "*/tritonserver" | head -n 1)
if [[ -z "${TRITON_ROOT_DIR}" ]]; then
    echo "Could not locate tritonserver directory in downloaded zip package"
    exit 1
fi

TRITON_LIBTRITONSERVER_PATH=$(find "${TRITON_ROOT_DIR}" -type f -name libtritonserver.so | head -n 1)
if [[ -z "${TRITON_LIBTRITONSERVER_PATH}" ]]; then
    echo "Could not locate libtritonserver.so in downloaded zip package"
    exit 1
fi

if [[ ! -d "${DEEPSTREAM_LIB_DIR}" ]]; then
    echo "DeepStream library directory does not exist: ${DEEPSTREAM_LIB_DIR}"
    exit 1
fi

cp "${TRITON_LIBTRITONSERVER_PATH}" "${DEEPSTREAM_LIB_DIR}/"

# nvinferserver requires Triton's gRPC client at runtime. The client SDK under
# /opt/tritonclient is a build dependency and QVS removes it before testing, so
# install the runtime library in the DeepStream library directory instead.
TRITON_GRPC_CLIENT_PATH=/opt/tritonclient/lib/libgrpcclient.so
if [[ ! -r "${TRITON_GRPC_CLIENT_PATH}" ]]; then
    echo "Downloading Triton client SDK to install libgrpcclient.so ..."
    wget -O "${TRITON_DOWNLOADS}/tritonclient.zip" "${TRITON_CLIENT_SDK_PKG_PATH}"

    TRITON_CLIENT_SDK_EXTRACT_DIR=${TRITON_DOWNLOADS}/tritonclient_sdk_extract
    mkdir -p "${TRITON_CLIENT_SDK_EXTRACT_DIR}"
    unzip -q "${TRITON_DOWNLOADS}/tritonclient.zip" -d "${TRITON_CLIENT_SDK_EXTRACT_DIR}"

    TRITON_GRPC_CLIENT_PATH=$(find "${TRITON_CLIENT_SDK_EXTRACT_DIR}" -type f -name libgrpcclient.so | head -n 1)
fi

if [[ -z "${TRITON_GRPC_CLIENT_PATH}" ]] || [[ ! -r "${TRITON_GRPC_CLIENT_PATH}" ]]; then
    echo "Could not locate Triton libgrpcclient.so"
    exit 1
fi

cp -a "${TRITON_GRPC_CLIENT_PATH}" "${DEEPSTREAM_LIB_DIR}/"
cp -a "${TRITON_ROOT_DIR}/backends/." "${TRITON_BACKEND_DIR}/"
cp "${TRITON_ROOT_DIR}/bin/tritonserver" "${TRITON_SERVER_BIN}/"
chmod -R +r $TRITON_BACKEND_DIR
chmod +rx $TRITON_SERVER_BIN

wget -O /var/tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/cuda-keyring_1.1-1_all.deb
apt install --yes /var/tmp/cuda-keyring.deb
rm /var/tmp/cuda-keyring.deb
apt update
apt install --yes --no-install-recommends datacenter-gpu-manager-4-core=1:4.4.0-1 datacenter-gpu-manager-4-dev=1:4.4.0-1

ldconfig

if ! gst-inspect-1.0 nvinferserver >/dev/null; then
    echo "nvinferserver is not loadable after Triton backend setup"
    exit 1
fi

echo "cleaning up ${TRITON_DOWNLOADS} directory ..."
rm -rf $TRITON_DOWNLOADS
