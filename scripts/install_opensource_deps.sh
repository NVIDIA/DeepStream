#!/usr/bin/env bash
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

# Fetch, build, and install open-source DeepStream dependencies from GitHub.
# Installs shared libraries directly to the DeepStream opt path.
#
# Dependencies built:
#   opentelemetry-cpp  v1.23.0            -> libopentelemetry_*.so  (HTTP + gRPC exporters)
#   civetweb           v1.16              -> libcivetweb.so
#   prometheus-cpp     v1.2.4             -> libprometheus-cpp-core.so
#   azure-iot-sdk-c    commit dbeb521e     -> libiothub_client.so (v1.11.0)
#
# Additional prerequisites installed:
#   half               v2.1.0             -> /opt/half
#   triton client SDK  x86 v2.67.0 nv26.03 / aarch64 v2.68.0 nv26.04 (zip) -> /opt/tritonclient
#   protobuf compiler  v3.21.12           -> /opt/proto
#   apt packages: libpango, libcairo, libjson-glib, librabbitmq, librdkafka, etc.
#   mosquitto (from PPA)
#
# Usage:
#   sudo PLATFORM=x86 bash scripts/install_opensource_deps.sh [NVDS_VERSION=9.1]
#   sudo PLATFORM=aarch64 bash scripts/install_opensource_deps.sh [NVDS_VERSION=9.1]

set -e

export DEBIAN_FRONTEND=noninteractive

NVDS_VERSION=${NVDS_VERSION:-9.1}
INSTALL_DIR=/opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/lib
BUILD_ROOT=$(mktemp -d /tmp/ds-deps-build.XXXXXX)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
STAGE_STATE_FILE="${DEPS_STAGE_STATE_FILE:-$SCRIPT_DIR/../build/.stage-state.deps}"

# Detect platform if not passed via environment
if [[ -z "${PLATFORM:-}" ]]; then
  ARCH=$(uname -m)
  case "$ARCH" in
    x86_64)  PLATFORM=x86 ;;
    aarch64)
      if [[ -f /etc/nv_tegra_release ]]; then
        PLATFORM=aarch64
      else
        PLATFORM=sbsa
      fi
      ;;
    *)  echo "Unsupported architecture: $ARCH"; exit 1 ;;
  esac
fi

stage_is_done() {
  local stage=$1
  [[ -f "$STAGE_STATE_FILE" ]] && grep -qx "DONE ${stage}" "$STAGE_STATE_FILE"
}

commit_stage_state_tmp() {
  cat "${STAGE_STATE_FILE}.tmp" >"$STAGE_STATE_FILE"
  rm -f "${STAGE_STATE_FILE}.tmp"
}

mark_stage_done() {
  local stage=$1
  if [[ -f "$STAGE_STATE_FILE" ]]; then
    grep -vx "DONE ${stage}" "$STAGE_STATE_FILE" > "${STAGE_STATE_FILE}.tmp" 2>/dev/null || true
    commit_stage_state_tmp
  else
    mkdir -p "$(dirname "$STAGE_STATE_FILE")"
    : >"$STAGE_STATE_FILE"
  fi
  echo "DONE ${stage}" >>"$STAGE_STATE_FILE"
}

echo "==> Installing open-source DeepStream dependencies"
echo "    Install dir : $INSTALL_DIR"
echo "    Build root  : $BUILD_ROOT"
echo "    Stage state : $STAGE_STATE_FILE"

mkdir -p "$INSTALL_DIR"

# ---------------------------------------------------------------------------
# 1. opentelemetry-cpp v1.23.0
# ---------------------------------------------------------------------------
echo ""
if stage_is_done deps-opentelemetry; then
  echo "--- opentelemetry-cpp v1.23.0 (skipped, already DONE) ---"
else
  echo "--- opentelemetry-cpp v1.23.0 ---"
  OTEL_DIR=$BUILD_ROOT/opentelemetry-cpp
  git clone --depth 1 --branch v1.23.0 \
      https://github.com/open-telemetry/opentelemetry-cpp.git "$OTEL_DIR"
  cmake -S "$OTEL_DIR" -B "$OTEL_DIR/build" \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_SHARED_LIBS=ON \
      -DWITH_OTLP_HTTP=ON \
      -DWITH_OTLP_GRPC=ON \
      -DWITH_PROMETHEUS=OFF \
      -DBUILD_TESTING=OFF \
      -DWITH_EXAMPLES=OFF \
      -DWITH_BENCHMARK=OFF \
      -DCMAKE_INSTALL_PREFIX="$BUILD_ROOT/otel-install" \
      -Wno-dev
  cmake --build "$OTEL_DIR/build" -j$(nproc)
  cmake --install "$OTEL_DIR/build"

  # Copy only the required .so files
  for lib in \
      libopentelemetry_common.so \
      libopentelemetry_metrics.so \
      libopentelemetry_resources.so \
      libopentelemetry_exporter_ostream_metrics.so \
      libopentelemetry_exporter_otlp_http_client.so \
      libopentelemetry_exporter_otlp_http_metric.so \
      libopentelemetry_exporter_otlp_grpc.so \
      libopentelemetry_exporter_otlp_grpc_client.so \
      libopentelemetry_exporter_otlp_grpc_metric.so \
      libopentelemetry_exporter_otlp_grpc_log.so \
      libopentelemetry_otlp_recordable.so \
      libopentelemetry_proto.so \
      libopentelemetry_http_client_curl.so \
      libopentelemetry_logs.so; do
      find "$BUILD_ROOT/otel-install/lib" -name "${lib}*" -exec cp -Pv {} "$INSTALL_DIR/" \;
  done
  mark_stage_done deps-opentelemetry
  echo "    Stage deps-opentelemetry: DONE"
fi

# ---------------------------------------------------------------------------
# 2. civetweb v1.16
# ---------------------------------------------------------------------------
echo ""
if stage_is_done deps-civetweb; then
  echo "--- civetweb v1.16 (skipped, already DONE) ---"
else
  CIVETWEB_DIR=$BUILD_ROOT/civetweb
  git clone --depth 1 --branch v1.16 \
      https://github.com/civetweb/civetweb.git "$CIVETWEB_DIR"

  # Create linker version script
  # This is necessary to avoid conflicts with the version script in the DS 9.1 SDK
  cat > "$CIVETWEB_DIR/civetweb.ver" << 'VEOF'
CIVETWEB_1.16 {
    global:
        *;
    local:
        duk_*;
};
VEOF

  make -C "$CIVETWEB_DIR" slib WITH_ALL=1 WITH_CPP=1 LDFLAGS="-Wl,--version-script,$CIVETWEB_DIR/civetweb.ver"
  cp -Pv "$CIVETWEB_DIR/libcivetweb.so"* "$INSTALL_DIR/"
  mark_stage_done deps-civetweb
  echo "    Stage deps-civetweb: DONE"
fi

# ---------------------------------------------------------------------------
# 3. prometheus-cpp v1.2.4
# ---------------------------------------------------------------------------
echo ""
if stage_is_done deps-prometheus; then
  echo "--- prometheus-cpp v1.2.4 (skipped, already DONE) ---"
else
  echo "--- prometheus-cpp v1.2.4 ---"
  PROM_DIR=$BUILD_ROOT/prometheus-cpp
  git clone --depth 1 --branch v1.2.4 \
      https://github.com/jupp0r/prometheus-cpp.git "$PROM_DIR"
  git -C "$PROM_DIR" submodule update --init --recursive
  cmake -S "$PROM_DIR" -B "$BUILD_ROOT/prom-build" \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_SHARED_LIBS=ON \
      -DENABLE_TESTING=OFF \
      -DENABLE_PUSH=OFF \
      -DENABLE_COMPRESSION=OFF \
      -DCMAKE_INSTALL_PREFIX="$BUILD_ROOT/prom-install" \
      -Wno-dev
  cmake --build "$BUILD_ROOT/prom-build" -j$(nproc)
  cmake --install "$BUILD_ROOT/prom-build"

  find "$BUILD_ROOT/prom-install/lib" -name "libprometheus-cpp-core.so*" -exec cp -Pv {} "$INSTALL_DIR/" \;
  mark_stage_done deps-prometheus
  echo "    Stage deps-prometheus: DONE"
fi

# ---------------------------------------------------------------------------
# 4. azure-iot-sdk-c v1.11.0
#    (follows README in src/utils/nvds_msgapi/azure_protocol_adaptor/module_client)
# ---------------------------------------------------------------------------
echo ""
if stage_is_done deps-azure-iot; then
  echo "--- azure-iot-sdk-c v1.11.0 (skipped, already DONE) ---"
else
  echo "--- azure-iot-sdk-c v1.11.0 ---"
  apt-get -y install uuid-dev
  AZURE_DIR=$BUILD_ROOT/azure-iot-sdk-c
  git clone https://github.com/Azure/azure-iot-sdk-c.git "$AZURE_DIR"
  git -C "$AZURE_DIR" checkout tags/1.11.0
  git -C "$AZURE_DIR" submodule update --init

  # Enable build_as_dynamic and use_edge_modules in CMakeLists.txt
  sed -i 's/\(option(build_as_dynamic.*\)OFF)/\1ON)/' "$AZURE_DIR/CMakeLists.txt"
  sed -i 's/\(option(use_edge_modules.*\)OFF)/\1ON)/' "$AZURE_DIR/CMakeLists.txt"

  mkdir -p "$AZURE_DIR/cmake"
  cmake -S "$AZURE_DIR" -B "$AZURE_DIR/cmake" \
      -DCMAKE_BUILD_TYPE=Release \
      -Dskip_samples=ON \
      -Wno-dev
  cmake --build "$AZURE_DIR/cmake" -j$(nproc)
  cmake --install "$AZURE_DIR/cmake"

  find /usr/local/lib -name "libiothub_client.so*" -exec cp -Pv {} "$INSTALL_DIR/" \;
  mark_stage_done deps-azure-iot
  echo "    Stage deps-azure-iot: DONE"
fi

# ---------------------------------------------------------------------------
# 5. Half v2.1.0
# ---------------------------------------------------------------------------
echo ""
if stage_is_done deps-half; then
  echo "--- Half v2.1.0 (skipped, already DONE) ---"
else
  echo "--- Half v2.1.0 ---"
  mkdir -p /opt/half
  wget -q -P /tmp https://sourceforge.net/projects/half/files/half/2.1.0/half-2.1.0.zip
  unzip -o /tmp/half-2.1.0.zip -d /opt/half/
  rm -f /tmp/half-2.1.0.zip
  mark_stage_done deps-half
  echo "    Stage deps-half: DONE"
fi

# ---------------------------------------------------------------------------
# 6. Triton client SDK (x86 v2.67.0 nv26.03 / aarch64 v2.68.0 nv26.04)
# ---------------------------------------------------------------------------
echo ""
if stage_is_done deps-tritonclient; then
  echo "--- Triton client SDK (skipped, already DONE) ---"
else
  echo "--- Triton client SDK (PLATFORM=$PLATFORM) ---"
  mkdir -p /opt/tritonclient/
  # Clear any prior target dirs so the mv below renames (lib64 -> lib) instead of
  # moving into an existing dir. Keeps this stage idempotent across re-runs.
  rm -rf /opt/tritonclient/lib /opt/tritonclient/include
  if [[ "$PLATFORM" = "x86" ]]; then
    # x86_64: download the prebuilt Triton SDK and lay headers/libs into
    # /opt/tritonclient (lib64 -> lib so consumers see a single lib dir).
    TRITON_SDK_ZIP=tritonserver-sdk-2.67.0+nv26.03-cu132-cp312-manylinux_2_28-x86_64.zip
    wget -q -P /tmp https://github.com/triton-inference-server/server/releases/download/v2.67.0/${TRITON_SDK_ZIP}
    rm -rf /tmp/tritonclient-sdk
    unzip -q -o "/tmp/${TRITON_SDK_ZIP}" -d /tmp/tritonclient-sdk
    mv /tmp/tritonclient-sdk/tritonserver_sdk/install/lib64   /opt/tritonclient/lib
    mv /tmp/tritonclient-sdk/tritonserver_sdk/install/include /opt/tritonclient/include
    rm -rf /tmp/tritonclient-sdk "/tmp/${TRITON_SDK_ZIP}"
  else
    # aarch64 / SBSA: download the prebuilt aarch64 Triton SDK zip (same flow as
    # x86). lib64 -> lib so consumers see a single lib dir. The public asset name
    # uses an underscore (tritonserver_sdk) plus a release build id.
    TRITON_SDK_ZIP=tritonserver_sdk-2.68.0+nv26.04-49346681-cu132-cp312-manylinux_2_28-aarch64.zip
    wget -q -P /tmp https://github.com/triton-inference-server/server/releases/download/v2.68.0/${TRITON_SDK_ZIP}
    rm -rf /tmp/tritonclient-sdk
    unzip -q -o "/tmp/${TRITON_SDK_ZIP}" -d /tmp/tritonclient-sdk
    mv /tmp/tritonclient-sdk/tritonserver_sdk/install/lib64   /opt/tritonclient/lib
    mv /tmp/tritonclient-sdk/tritonserver_sdk/install/include /opt/tritonclient/include
    rm -rf /tmp/tritonclient-sdk "/tmp/${TRITON_SDK_ZIP}"
  fi
  mark_stage_done deps-tritonclient
  echo "    Stage deps-tritonclient: DONE"
fi

# ---------------------------------------------------------------------------
# 7. Protobuf compiler v3.21.12
# ---------------------------------------------------------------------------
echo ""
if stage_is_done deps-protoc; then
  echo "--- Protobuf compiler v3.21.12 (skipped, already DONE) ---"
else
  echo "--- Protobuf compiler v3.21.12 (PLATFORM=$PLATFORM) ---"
  mkdir -p /opt/proto
  PB_REL="https://github.com/protocolbuffers/protobuf/releases"
  if [[ "$PLATFORM" = "x86" ]]; then
    curl -Lo /tmp/protoc-21.12.zip "$PB_REL/download/v21.12/protoc-21.12-linux-x86_64.zip"
  else
    curl -Lo /tmp/protoc-21.12.zip "$PB_REL/download/v21.12/protoc-21.12-linux-aarch_64.zip"
  fi
  unzip -o /tmp/protoc-21.12.zip -d /opt/proto
  chmod -R +rx /opt/proto/
  rm -f /tmp/protoc-21.12.zip
  mark_stage_done deps-protoc
  echo "    Stage deps-protoc: DONE"
fi

# ---------------------------------------------------------------------------
# 8. APT library dependencies
# ---------------------------------------------------------------------------
echo ""
if stage_is_done deps-apt-libs; then
  echo "--- APT library dependencies (skipped, already DONE) ---"
else
  echo "--- APT library dependencies ---"
  apt-get update
  apt-get install -y \
    libpango1.0-dev libcairo2-dev libjpeg-dev \
    libglib2.0-dev libjson-glib-dev uuid-dev libyaml-cpp-dev \
    librabbitmq-dev librdkafka-dev \
    libjansson4 libjansson-dev \
    libssl-dev libcjson-dev libhiredis-dev protobuf-compiler
  mark_stage_done deps-apt-libs
  echo "    Stage deps-apt-libs: DONE"
fi

# ---------------------------------------------------------------------------
# 9. Mosquitto
# ---------------------------------------------------------------------------
echo ""
if stage_is_done deps-mosquitto; then
  echo "--- Mosquitto (skipped, already DONE) ---"
else
  echo "--- Mosquitto ---"
  apt-get update
  apt install -y software-properties-common
  apt-add-repository -y ppa:mosquitto-dev/mosquitto-ppa
  apt-get update
  apt-get install -y libmosquitto-dev mosquitto
  mark_stage_done deps-mosquitto
  echo "    Stage deps-mosquitto: DONE"
fi

# ---------------------------------------------------------------------------
# 10. Sample-app build prerequisites
# ---------------------------------------------------------------------------
echo ""
if stage_is_done deps-sample-app-prereqs; then
  echo "--- Sample-app build prerequisites (skipped, already DONE) ---"
else
  echo "--- Sample-app build prerequisites ---"
  apt-get update
  apt-get install -y \
    libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev \
    libgstrtspserver-1.0-dev libx11-dev \
    gstreamer1.0-libav libavahi-compat-libdnssd-dev \
    libjson-glib-dev libjsoncpp-dev libyaml-cpp-dev \
    libgbm1 libglapi-mesa libgles2-mesa-dev \
    rapidjson-dev
  mark_stage_done deps-sample-app-prereqs
  echo "    Stage deps-sample-app-prereqs: DONE"
fi

# ---------------------------------------------------------------------------
# Cleanup and ldconfig
# ---------------------------------------------------------------------------
echo ""
echo "==> Cleaning build directory"
rm -rf "$BUILD_ROOT"

echo "==> Running ldconfig"
ldconfig

echo ""
echo "==> Done. Installed to $INSTALL_DIR:"
ls "$INSTALL_DIR"/libopentelemetry_* "$INSTALL_DIR"/libcivetweb* "$INSTALL_DIR"/libprometheus* "$INSTALL_DIR"/libiothub_client* 2>/dev/null
