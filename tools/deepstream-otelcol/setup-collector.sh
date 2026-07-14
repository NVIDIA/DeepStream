#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

set -euo pipefail

VERSION="0.91.0"
INSTALL_DIR="$HOME/.local/bin"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Detect architecture
OS="$(uname -s)"
if [[ "${OS}" != "Linux" ]]; then
    echo "Unsupported OS: ${OS} (Linux required)"
    exit 1
fi

MACHINE_ARCH=$(uname -m)
case "${MACHINE_ARCH}" in
    x86_64)
        ARCH="linux_amd64"
        ;;
    aarch64|arm64)
        ARCH="linux_arm64"
        ;;
    *)
        echo "Unsupported architecture: ${MACHINE_ARCH}"
        exit 1
        ;;
esac

BASE_URL="https://github.com/open-telemetry/opentelemetry-collector-releases/releases/download/v${VERSION}"
TARBALL="otelcol_${VERSION}_${ARCH}.tar.gz"
DOWNLOAD_URL="${BASE_URL}/${TARBALL}"
CHECKSUMS="opentelemetry-collector-releases_checksums.txt"
CHECKSUMS_URL="${BASE_URL}/${CHECKSUMS}"

# Use a temporary directory so extraction does not pollute the script directory.
# The release tarball also ships LICENSE and README.md at its root, which would
# otherwise overwrite the files in this folder.
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

echo "Detected architecture: ${MACHINE_ARCH} -> ${ARCH}"
echo "Downloading OpenTelemetry Collector ${VERSION}..."
wget -O "${TMP_DIR}/${TARBALL}" "${DOWNLOAD_URL}"
wget -O "${TMP_DIR}/${CHECKSUMS}" "${CHECKSUMS_URL}"

EXPECTED_SHA="$(awk -v f="${TARBALL}" '$2 == f {print $1}' "${TMP_DIR}/${CHECKSUMS}")"
if [[ -z "${EXPECTED_SHA}" ]]; then
  echo "Failed to find checksum entry for ${TARBALL}"
  exit 1
fi
echo "${EXPECTED_SHA}  ${TMP_DIR}/${TARBALL}" | sha256sum -c -

echo "Extracting..."
tar -xzf "${TMP_DIR}/${TARBALL}" -C "${TMP_DIR}"

echo "Moving binary to ${INSTALL_DIR}..."
mkdir -p "${INSTALL_DIR}"
mv "${TMP_DIR}/otelcol" "${INSTALL_DIR}/"

echo "OpenTelemetry Collector installed successfully!"
echo "Binary location: ${INSTALL_DIR}/otelcol"
echo ""
echo "To run the collector with the config:"
echo "${INSTALL_DIR}/otelcol --config=${SCRIPT_DIR}/otel-collector-config.yaml"

