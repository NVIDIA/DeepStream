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
#   triton client SDK  reviewed matching vendor SDK -> /opt/tritonclient
#                      x86, Jetson, and SBSA: published v2.74.0 nv26.10 SDK zip
#                      for that architecture when the default tree is absent
#   protobuf compiler  explicit SDK-matched provider, or the published compiler
#                      for protobuf 3.21.12 and 33.5 -> /opt/proto
#   pybind11           v2.12.0 (headers)  -> /opt/pybind11
#   dlpack             v0.8    (headers)  -> /opt/dlpack
#   apt packages: python3-build, python3-venv, python3-setuptools, python3-wheel,
#                 python3-pip (needed by sources/python/build.sh and Sparse4D torch)
#   apt packages: libpango, libcairo, libjson-glib, librabbitmq, librdkafka,
#                 libboost-dev, libeigen3-dev, nlohmann-json3-dev, libharfbuzz-dev,
#                 libprotobuf-dev, libglm-dev (prefer 0.9.9.8), libglfw3-dev
#                 (prefer 3.3.10), openucx/libucx-dev (prefer 1.13.1), etc.
#   mosquitto (from Ubuntu packages)
#   libnvvpi4 + vpi4-dev 4.1.4 (x86, required by nvmultiobjecttracker)
#   Sparse4D: ds-sparse4d/configs/install_dependencies.sh (PyTorch 2.9.0+cu130,
#             nlohmann-json, Eigen, OpenCV)
#
# Usage:
#   sudo PLATFORM=x86 bash scripts/install_opensource_deps.sh [NVDS_VERSION=9.2.1]
#   sudo PLATFORM=aarch64 bash scripts/install_opensource_deps.sh [NVDS_VERSION=9.2.1]
#   sudo PLATFORM=sbsa bash scripts/install_opensource_deps.sh [NVDS_VERSION=9.2.1]
#     (SBSA/DGX Spark Docker: stages 1–3 and 5–7 skipped; azure-iot still built)
#
# Environment:
#   NVDS_VERSION              MAJOR.MINOR.PATCH (default: 9.2.1).
#                             Install path still uses MAJOR.MINOR for now.
#   RESUME=1                  Skip stages already marked DONE (used by build.sh --resume).
#                             Default 0: always rebuild/install; stage state is reset first.
#   DEPS_STAGE_STATE_FILE     Stage resume file (default: build/.stage-state.deps)
#   DS_DEPS_WORK_ROOT         Existing writable parent for dependency build
#                             directories (default: /tmp).
#   TRITON_CLIENT_SDK_ROOT   Preprovisioned matching C++ SDK (default: /opt/tritonclient).
#                            A missing default tree is filled from the published
#                            Triton 2.74.0 nv26.10 SDK archive for x86, Jetson,
#                            or SBSA. An explicit incomplete root is rejected.
#   PROTOC_ROOT              Optional protoc distribution matching the selected
#                            Triton SDK protobuf headers; copied to /opt/proto.
#                            Must contain bin/protoc and include/. No image pull.
#   VPI_DIR                  Explicit preprovisioned vendor VPI installation:
#                            x86 4.1.4, ARM 4.1.6 (default: /opt/nvidia/vpi4).
#                            An explicit x86 provider is validated, not installed
#                            or replaced through APT.
#   TORCH_PATH               Optional preprovisioned CUDA ABI1 LibTorch provider;
#                            preserved by the Sparse4D helper, never replaced by pip.

set -e

export DEBIAN_FRONTEND=noninteractive

NVDS_VERSION=${NVDS_VERSION:-9.2.1}
# NVDS_VERSION must be MAJOR.MINOR.PATCH (default: 9.2.1). MAJOR.MINOR alone is
# rejected. Install tree still uses MAJOR.MINOR for now (e.g. deepstream-9.2/).
if [[ ! "$NVDS_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "error: invalid NVDS_VERSION='$NVDS_VERSION' (expected MAJOR.MINOR.PATCH, e.g. 9.2.1)" >&2
  exit 1
fi
NVDS_VERSION="${NVDS_VERSION%.*}"
INSTALL_DIR=/opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/lib
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
STAGE_STATE_FILE="${DEPS_STAGE_STATE_FILE:-$SCRIPT_DIR/../build/.stage-state.deps}"
RESUME=${RESUME:-0}

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
case "$PLATFORM" in
  x86|aarch64|sbsa) ;;
  *) echo "error: unsupported PLATFORM='$PLATFORM'" >&2; exit 1 ;;
esac
if [[ "$PLATFORM" == sbsa && ! -f /.dockerenv && ! -f /run/.containerenv ]]; then
  echo "error: SBSA dependency installation requires the matching DeepStream SBSA container, not a bare-metal host." >&2
  exit 1
fi
DS_DEPS_WORK_ROOT=${DS_DEPS_WORK_ROOT-/tmp}
if [[ ! -d "$DS_DEPS_WORK_ROOT" || ! -w "$DS_DEPS_WORK_ROOT" || ! -x "$DS_DEPS_WORK_ROOT" ]]; then
  echo "error: DS_DEPS_WORK_ROOT must be an existing writable, searchable directory: $DS_DEPS_WORK_ROOT" >&2
  exit 1
fi
BUILD_ROOT=$(mktemp -d "$DS_DEPS_WORK_ROOT/ds-deps-build.XXXXXX")

stage_is_done() {
  local stage=$1
  # Manual / non-resume runs always execute; skip only when RESUME=1 (build.sh --resume).
  [[ "$RESUME" -eq 1 ]] || return 1
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
echo "    Resume      : $RESUME"

# Standalone / non-resume runs always rebuild; reset stage markers first.
if [[ "$RESUME" -eq 0 ]]; then
  mkdir -p "$(dirname "$STAGE_STATE_FILE")"
  : >"$STAGE_STATE_FILE"
fi

mkdir -p "$INSTALL_DIR"

# ---------------------------------------------------------------------------
# 1. opentelemetry-cpp v1.23.0
# ---------------------------------------------------------------------------
echo ""
if [[ "$PLATFORM" = "sbsa" ]]; then
  echo "--- opentelemetry-cpp v1.23.0 (skipped, preinstalled on SBSA) ---"
  mark_stage_done deps-opentelemetry
elif stage_is_done deps-opentelemetry; then
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
if [[ "$PLATFORM" = "sbsa" ]]; then
  echo "--- civetweb v1.16 (skipped, preinstalled on SBSA) ---"
  mark_stage_done deps-civetweb
elif stage_is_done deps-civetweb; then
  echo "--- civetweb v1.16 (skipped, already DONE) ---"
else
  CIVETWEB_DIR=$BUILD_ROOT/civetweb
  git clone --depth 1 --branch v1.16 \
      https://github.com/civetweb/civetweb.git "$CIVETWEB_DIR"

  # Create linker version script
  # This is necessary to avoid conflicts with the version script in the DS 9.2 SDK
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
if [[ "$PLATFORM" = "sbsa" ]]; then
  echo "--- prometheus-cpp v1.2.4 (skipped, preinstalled on SBSA) ---"
  mark_stage_done deps-prometheus
elif stage_is_done deps-prometheus; then
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
if [[ "$PLATFORM" = "sbsa" ]]; then
  echo "--- Half v2.1.0 (skipped, preinstalled on SBSA) ---"
  mark_stage_done deps-half
elif stage_is_done deps-half; then
  echo "--- Half v2.1.0 (skipped, already DONE) ---"
else
  echo "--- Half v2.1.0 ---"
  mkdir -p /opt/half
  # Drop any stale download first: wget -P never overwrites an existing file (it
  # saves half-2.1.0.zip.1 next to it), so unzip would keep reading an old or
  # bad copy (e.g. an HTML error page from an earlier failed attempt).
  rm -f /tmp/half-2.1.0.zip
  wget -q -O /tmp/half-2.1.0.zip https://sourceforge.net/projects/half/files/half/2.1.0/half-2.1.0.zip
  if ! unzip -o /tmp/half-2.1.0.zip -d /opt/half/; then
    echo "error: could not unpack half-2.1.0.zip (the download may have returned an error page); stage deps-half not marked DONE" >&2
    rm -f /tmp/half-2.1.0.zip
    exit 1
  fi
  rm -f /tmp/half-2.1.0.zip
  mark_stage_done deps-half
  echo "    Stage deps-half: DONE"
fi

# ---------------------------------------------------------------------------
# 6. Triton client SDK: preserve the selected server/platform provider
# ---------------------------------------------------------------------------
echo ""
TRITON_CLIENT_SDK_ROOT=${TRITON_CLIENT_SDK_ROOT-/opt/tritonclient}
install_published_triton_sdk() {
  local zip_name zip_url zip_sha tmp_dir include_dir lib_dir actual_sha
  case "$PLATFORM" in
    x86)
      zip_name=tritonserver_sdk-2.74.0+nv26.10-cu132-cp312-manylinux_2_34-x86_64.zip
      zip_sha=4e7c451dac46a1bbae243609c225f6547d74f386107bbe767065593d7682f2fd
      ;;
    aarch64|sbsa)
      zip_name=tritonserver_sdk-2.74.0+nv26.10-cu132-cp312-manylinux_2_34-aarch64.zip
      zip_sha=27094abcc4e323400c1c58f60f5dae33b7bd8b9c29b25a65bc0ad34474dd44dd
      ;;
    *)
      echo "error: no published Triton 2.74.0 client SDK is selected for PLATFORM=$PLATFORM." >&2
      return 1
      ;;
  esac
  zip_url=https://github.com/triton-inference-server/server/releases/download/v2.74.0/${zip_name}
  tmp_dir=$(mktemp -d)
  echo "--- Triton client SDK: downloading published $PLATFORM 2.74.0 / 26.10 archive ---"
  if ! curl -fL --retry 2 -o "$tmp_dir/$zip_name" "$zip_url"; then
    rm -rf "$tmp_dir"
    echo "error: could not download the published Triton 2.74.0 $PLATFORM client SDK." >&2
    return 1
  fi
  actual_sha=$(sha256sum "$tmp_dir/$zip_name" | awk '{print $1}')
  if [[ "$actual_sha" != "$zip_sha" ]]; then
    rm -rf "$tmp_dir"
    echo "error: published Triton SDK checksum mismatch (got $actual_sha)." >&2
    return 1
  fi
  if ! unzip -q -o "$tmp_dir/$zip_name" -d "$tmp_dir/sdk"; then
    rm -rf "$tmp_dir"
    echo "error: could not unpack the published Triton SDK archive." >&2
    return 1
  fi
  include_dir=$tmp_dir/sdk/tritonserver_sdk/install/include
  lib_dir=$tmp_dir/sdk/tritonserver_sdk/install/lib64
  if [[ ! -r "$include_dir/grpc_client.h" || ! -r "$include_dir/common.h" ||
        ! -r "$lib_dir/libgrpcclient.so" || ! -r "$lib_dir/libprotobuf.a" ||
        ! -r "$lib_dir/libcares.a" ]]; then
    rm -rf "$tmp_dir"
    echo "error: published Triton SDK archive is missing include/ or lib64/ inputs." >&2
    return 1
  fi
  mkdir -p /opt/tritonclient
  rm -rf /opt/tritonclient/include /opt/tritonclient/lib
  mv "$include_dir" /opt/tritonclient/include
  mv "$lib_dir" /opt/tritonclient/lib
  rm -rf "$tmp_dir"
  TRITON_CLIENT_SDK_ROOT=/opt/tritonclient
}
triton_headers_ready=1
for header in grpc_client.h common.h; do
  if [[ ! -r "$TRITON_CLIENT_SDK_ROOT/include/$header" ]]; then
    triton_headers_ready=0
    break
  fi
done
if [[ "$triton_headers_ready" -eq 0 ]]; then
  if [[ "$TRITON_CLIENT_SDK_ROOT" == /opt/tritonclient ]]; then
    install_published_triton_sdk || exit 1
  else
    echo "error: provide the matching vendor Triton C++ SDK at TRITON_CLIENT_SDK_ROOT (missing include/grpc_client.h or include/common.h)." >&2
    echo "The default /opt/tritonclient tree is downloaded from the published Triton 2.74.0 nv26.10 SDK for x86, Jetson, and SBSA. An explicit TRITON_CLIENT_SDK_ROOT must already be complete." >&2
    exit 1
  fi
fi
triton_client_library_dir_is_complete() {
  local directory=$1 library
  for library in libprotobuf.a libcares.a; do
    [[ -r "$directory/$library" ]] || return 1
  done
  # The public ARM64 SDK bundles gRPC into libgrpcclient.so and does not ship
  # libgrpc++.a. The source Makefiles link -lgrpcclient, not a separate gRPC
  # archive. Static-only providers still need the complete static dependency set.
  [[ ! -r "$directory/libgrpcclient.so" ]] || return 0
  for library in libgrpcclient.a libgrpcclient_static.a libgrpc++.a; do
    [[ -r "$directory/$library" ]] || return 1
  done
}
# Some SDK layouts contain both directories. Prefer lib/ only when complete;
# never combine dependencies from two incomplete provider directories.
TRITON_CLIENT_SDK_LIB_DIR=
for triton_client_candidate in "$TRITON_CLIENT_SDK_ROOT/lib" "$TRITON_CLIENT_SDK_ROOT/lib64"; do
  if triton_client_library_dir_is_complete "$triton_client_candidate"; then
    TRITON_CLIENT_SDK_LIB_DIR=$triton_client_candidate
    break
  fi
done
if [[ -z "$TRITON_CLIENT_SDK_LIB_DIR" ]]; then
  echo "error: provide one complete matching Triton client SDK in $TRITON_CLIENT_SDK_ROOT/lib or lib64." >&2
  echo 'Required: libprotobuf.a, libcares.a, and libgrpcclient.so; static-only providers also need libgrpcclient.a, libgrpcclient_static.a, and libgrpc++.a.' >&2
  exit 1
fi
echo "--- Triton client SDK: caller/preinstalled provider $TRITON_CLIENT_SDK_ROOT ---"
echo "    File checks do not prove provider release identity; use a reviewed matching SDK."
install_triton_client_tree() {
  local source=$1 destination=$2 source_path destination_path
  [[ "$source" -ef "$destination" ]] && return 0
  source_path=$(cd "$source" && pwd -P)
  destination_path=$(realpath -m "$destination")
  case "$destination_path/" in
    "$source_path/"*)
      echo "error: Triton SDK destination must not be nested inside its source: $destination" >&2
      return 1
      ;;
  esac
  mkdir -p "$destination"
  # Replace colliding entries, not the targets of links left by another SDK.
  cp -a --remove-destination "$source/." "$destination/"
}
if stage_is_done deps-tritonclient &&
    [[ "$TRITON_CLIENT_SDK_ROOT/include" -ef /opt/tritonclient/include ]] &&
    [[ "$TRITON_CLIENT_SDK_LIB_DIR" -ef /opt/tritonclient/lib ]]; then
  echo "--- Triton client SDK (skipped, already DONE) ---"
else
  # Even a provider already rooted at /opt/tritonclient may use lib64/. The
  # public Makefiles consume lib/, so normalize it rather than skipping copy.
  install_triton_client_tree "$TRITON_CLIENT_SDK_ROOT/include" /opt/tritonclient/include
  # -lgrpcclient prefers .so over .a. A selected static-only provider must not
  # silently link an older shared client left in the merge destination. Remove
  # only that entry (never a symlink target), and leave self-copy unchanged.
  if [[ ! "$TRITON_CLIENT_SDK_LIB_DIR" -ef /opt/tritonclient/lib &&
        ! -r "$TRITON_CLIENT_SDK_LIB_DIR/libgrpcclient.so" &&
        -r "$TRITON_CLIENT_SDK_LIB_DIR/libgrpcclient.a" ]]; then
    rm -f -- /opt/tritonclient/lib/libgrpcclient.so
  fi
  install_triton_client_tree "$TRITON_CLIENT_SDK_LIB_DIR" /opt/tritonclient/lib
  mark_stage_done deps-tritonclient
  echo "    Stage deps-tritonclient: DONE"
fi

# ---------------------------------------------------------------------------
# 7. Protobuf compiler: explicit SDK-matched provider or legacy v3.21.12
# ---------------------------------------------------------------------------
echo ""
required_protoc_version() {
  local sdk_root=$1 version_code
  local cpp_major cpp_minor cpp_patch
  version_code=$(awk '$1 == "#define" && $2 == "GOOGLE_PROTOBUF_VERSION" && $3 ~ /^[0-9]+$/ { print $3; exit }' \
    "$sdk_root/include/google/protobuf/stubs/common.h" 2>/dev/null)
  if [[ ! "$version_code" =~ ^[0-9]+$ ]]; then
    echo 'error: cannot identify the selected Triton SDK protobuf header version.' >&2
    return 1
  fi
  cpp_major=$((version_code / 1000000))
  cpp_minor=$((version_code / 1000 % 1000))
  cpp_patch=$((version_code % 1000))
  # Modern language-specific runtime major versions differ from protoc's
  # release numbering (for example C++ 6.33.5 uses protoc 33.5).
  if (( cpp_major >= 4 )); then
    printf '%s\n' "$cpp_minor.$cpp_patch"
  else
    printf '%s\n' "$cpp_major.$cpp_minor.$cpp_patch"
  fi
}

validate_protoc_provider() {
  local root=$1 sdk_root=$2 compiler_version expected_version
  if [[ -z "$root" || ! -x "$root/bin/protoc" ||
        ! -r "$root/include/google/protobuf/descriptor.proto" ]]; then
    echo 'error: PROTOC_ROOT must contain bin/protoc and the matching include/google/protobuf/*.proto files.' >&2
    return 1
  fi
  expected_version=$(required_protoc_version "$sdk_root") || return 1
  compiler_version=$("$root/bin/protoc" --version) || return 1
  if [[ "$compiler_version" != "libprotoc $expected_version" ]]; then
    echo "error: selected Triton SDK requires protoc $expected_version; provider reports $compiler_version." >&2
    return 1
  fi
  printf '%s\n' "$expected_version"
}

PROTOC_REQUIRED_VERSION=$(required_protoc_version "$TRITON_CLIENT_SDK_ROOT") || exit 1
if [[ -v PROTOC_ROOT ]]; then
  # Recheck explicit providers on resume rather than trusting an older marker.
  PROTOC_SELECTED_VERSION=$(validate_protoc_provider "$PROTOC_ROOT" "$TRITON_CLIENT_SDK_ROOT") || exit 1
  echo "--- Protobuf compiler $PROTOC_SELECTED_VERSION (explicit provider: $PROTOC_ROOT) ---"
  mkdir -p /opt/proto/bin
  if [[ ! "$PROTOC_ROOT/bin/protoc" -ef /opt/proto/bin/protoc ]]; then
    cp --remove-destination -- "$PROTOC_ROOT/bin/protoc" /opt/proto/bin/protoc
    chmod 755 /opt/proto/bin/protoc
  fi
  install_triton_client_tree "$PROTOC_ROOT/include" /opt/proto/include
  mark_stage_done deps-protoc
  echo "    Stage deps-protoc: DONE"
elif PROTOC_SELECTED_VERSION=$(validate_protoc_provider /opt/proto "$TRITON_CLIENT_SDK_ROOT" 2>/dev/null); then
  echo "--- Protobuf compiler $PROTOC_SELECTED_VERSION (using matching /opt/proto provider) ---"
  mark_stage_done deps-protoc
  echo "    Stage deps-protoc: DONE"
elif [[ "$PROTOC_REQUIRED_VERSION" == 33.5 ]]; then
  echo "--- Protobuf compiler v33.5 (PLATFORM=$PLATFORM) ---"
  case "$PLATFORM" in
    x86)
      protoc_name=protoc-33.5-linux-x86_64.zip
      protoc_sha=24e58fb231d50306ee28491f33a170301e99540f7e29ca461e0e80fd1239f8d1
      ;;
    aarch64|sbsa)
      protoc_name=protoc-33.5-linux-aarch_64.zip
      protoc_sha=2b0fcf9b2c32cbadccc0eb7a88b841fffecd4a06fc80acdba2b5be45e815c38a
      ;;
    *)
      echo "error: no protoc 33.5 archive is selected for PLATFORM=$PLATFORM." >&2
      exit 1
      ;;
  esac
  curl -fL --retry 2 -o "/tmp/$protoc_name" \
    "https://github.com/protocolbuffers/protobuf/releases/download/v33.5/$protoc_name"
  if [[ "$(sha256sum "/tmp/$protoc_name" | awk '{print $1}')" != "$protoc_sha" ]]; then
    rm -f "/tmp/$protoc_name"
    echo "error: protoc 33.5 archive checksum mismatch." >&2
    exit 1
  fi
  mkdir -p /opt/proto
  unzip -o "/tmp/$protoc_name" -d /opt/proto
  chmod -R +rx /opt/proto/
  rm -f "/tmp/$protoc_name"
  PROTOC_SELECTED_VERSION=$(validate_protoc_provider /opt/proto "$TRITON_CLIENT_SDK_ROOT") || exit 1
  mark_stage_done deps-protoc
  echo "    Stage deps-protoc: DONE"
elif [[ "$PROTOC_REQUIRED_VERSION" != 3.21.12 || "$PLATFORM" = "sbsa" ]]; then
  echo "error: selected Triton SDK requires protoc $PROTOC_REQUIRED_VERSION, but /opt/proto is missing or mismatched." >&2
  echo 'Set PROTOC_ROOT to a matching compiler distribution, or provision a matching compiler and include tree at /opt/proto.' >&2
  exit 1
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
  # OpenUCX: prefer v1.13.1 when that version is present in apt; otherwise
  # install the distro candidate (DeepStream Gst-NvDsUcx requires UCX >= 1.13).
  UCX_APT_PKGS=(libucx-dev libucx0)
  if apt-cache madison libucx0 2>/dev/null | awk '{print $3}' | grep -qE '^1\.13\.1'; then
    echo "    OpenUCX: installing preferred apt version 1.13.1"
    UCX_APT_PKGS=(libucx-dev=1.13.1* libucx0=1.13.1*)
  else
    UCX_CANDIDATE=$(apt-cache policy libucx0 2>/dev/null | awk '/Candidate:/ {print $2; exit}')
    echo "    OpenUCX: preferred 1.13.1 not in apt; installing candidate ${UCX_CANDIDATE:-unknown}"
  fi

  # GLM / GLFW: used by DS3D glprocess. Prefer GLM 0.9.9.8 (same as the old
  # vendored tree) and GLFW 3.3.10 (Ubuntu 24.04); otherwise the candidate.
  GLM_APT_PKG=libglm-dev
  if apt-cache madison libglm-dev 2>/dev/null | awk '{print $3}' | grep -qE '^0\.9\.9\.8'; then
    echo "    GLM: installing preferred apt version 0.9.9.8"
    GLM_APT_PKG=libglm-dev=0.9.9.8*
  else
    GLM_CANDIDATE=$(apt-cache policy libglm-dev 2>/dev/null | awk '/Candidate:/ {print $2; exit}')
    echo "    GLM: preferred 0.9.9.8 not in apt; installing candidate ${GLM_CANDIDATE:-unknown}"
  fi
  GLFW_APT_PKG=libglfw3-dev
  if apt-cache madison libglfw3-dev 2>/dev/null | awk '{print $3}' | grep -qE '^3\.3\.10'; then
    echo "    GLFW: installing preferred apt version 3.3.10"
    GLFW_APT_PKG=libglfw3-dev=3.3.10*
  else
    GLFW_CANDIDATE=$(apt-cache policy libglfw3-dev 2>/dev/null | awk '/Candidate:/ {print $2; exit}')
    echo "    GLFW: preferred 3.3.10 not in apt; installing candidate ${GLFW_CANDIDATE:-unknown}"
  fi

  apt-get install -y \
    libpango1.0-dev libcairo2-dev libjpeg-dev \
    libglib2.0-dev libjson-glib-dev uuid-dev libyaml-cpp-dev \
    librabbitmq-dev librdkafka-dev \
    libjansson4 libjansson-dev \
    libssl-dev libcjson-dev libhiredis-dev protobuf-compiler libprotobuf-dev \
    libboost-dev libeigen3-dev nlohmann-json3-dev libharfbuzz-dev \
    "${GLM_APT_PKG}" "${GLFW_APT_PKG}" \
    "${UCX_APT_PKGS[@]}"

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
  # Use Ubuntu's packages for the local build instead of the Mosquitto PPA.
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
    libgbm1 libglapi-mesa libgles-dev libglvnd-dev libegl-dev \
    rapidjson-dev
  mark_stage_done deps-sample-app-prereqs
  echo "    Stage deps-sample-app-prereqs: DONE"
fi

# ---------------------------------------------------------------------------
# 11. pybind11 v2.12.0 (header-only; Service Maker Python bindings)
# ---------------------------------------------------------------------------
echo ""
if stage_is_done deps-pybind11; then
  echo "--- pybind11 v2.12.0 (skipped, already DONE) ---"
else
  echo "--- pybind11 v2.12.0 ---"
  # Clear any prior target dir so the extraction below is idempotent across
  # re-runs. --strip-components=1 drops the pybind11-2.12.0/ prefix from the
  # release tarball so headers land at /opt/pybind11/include.
  rm -rf /opt/pybind11
  mkdir -p /opt/pybind11
  # Drop any stale tarball first: with fs.protected_regular set, root cannot
  # open a /tmp file owned by another user, so curl -o would fail outright.
  rm -f /tmp/pybind11-2.12.0.tar.gz
  curl -Lo /tmp/pybind11-2.12.0.tar.gz \
    https://github.com/pybind/pybind11/archive/refs/tags/v2.12.0.tar.gz
  tar -xzf /tmp/pybind11-2.12.0.tar.gz -C /opt/pybind11 --strip-components=1
  rm -f /tmp/pybind11-2.12.0.tar.gz
  mark_stage_done deps-pybind11
  echo "    Stage deps-pybind11: DONE"
fi

# ---------------------------------------------------------------------------
# 12. dlpack v0.8 (header-only; tensor interop for the Python bindings)
# ---------------------------------------------------------------------------
echo ""
if stage_is_done deps-dlpack; then
  echo "--- dlpack v0.8 (skipped, already DONE) ---"
else
  echo "--- dlpack v0.8 ---"
  # Same idempotency and prefix-stripping rationale as the pybind11 stage above.
  rm -rf /opt/dlpack
  mkdir -p /opt/dlpack
  # Same stale-tarball guard as the pybind11 stage above.
  rm -f /tmp/dlpack-0.8.tar.gz
  curl -Lo /tmp/dlpack-0.8.tar.gz \
    https://github.com/dmlc/dlpack/archive/refs/tags/v0.8.tar.gz
  tar -xzf /tmp/dlpack-0.8.tar.gz -C /opt/dlpack --strip-components=1
  rm -f /tmp/dlpack-0.8.tar.gz
  mark_stage_done deps-dlpack
  echo "    Stage deps-dlpack: DONE"
fi

# ---------------------------------------------------------------------------
# 13. Python wheel build prerequisites (pyservicemaker)
# ---------------------------------------------------------------------------
echo ""
if stage_is_done deps-python-build; then
  echo "--- Python wheel build prerequisites (skipped, already DONE) ---"
else
  echo "--- Python wheel build prerequisites ---"
  # python3-build provides 'python3 -m build', which sources/python/build.sh
  # uses to produce the wheel. It builds in an isolated environment by default,
  # which needs python3-venv; setuptools and wheel are the build backend.
  # python3-dev supplies Python.h and libpython3.x.so, which the pybind11
  # extension in sources/python/src includes and links.
  apt-get install -y python3-build python3-venv python3-setuptools python3-wheel \
    python3-dev python3-pip
  mark_stage_done deps-python-build
  echo "    Stage deps-python-build: DONE"
fi

# ---------------------------------------------------------------------------
# 14. Sparse4D (ds-sparse4d) — selected LibTorch + remaining apt deps
# ---------------------------------------------------------------------------
echo ""
# Explicit providers are rechecked even on resume, rather than trusting a marker
# for a provider that may have disappeared or changed in place.
if [[ ! -v TORCH_PATH ]] && stage_is_done deps-sparse4d; then
  echo "--- Sparse4D dependencies (skipped, already DONE) ---"
else
  echo "--- Sparse4D dependencies ---"
  SPARSE4D_DEPS="$SCRIPT_DIR/../src/gst-plugins/gst-nvdsvideotemplate/ds-sparse4d/configs/install_dependencies.sh"
  if [[ ! -f "$SPARSE4D_DEPS" ]]; then
    echo "error: missing $SPARSE4D_DEPS" >&2
    exit 1
  fi
  bash "$SPARSE4D_DEPS"
  mark_stage_done deps-sparse4d
  echo "    Stage deps-sparse4d: DONE"
fi

# ---------------------------------------------------------------------------
# VPI4 development headers (ARM must retain the target vendor4.1.6 provider)
# ---------------------------------------------------------------------------

ds_vpi_installed_package() {
  local package=$1 state version error_flag query_status
  if state=$(dpkg-query -W -f='${db:Status-Status}\t${db:Status-Eflag}\t${Version}\n' "$package" 2>/dev/null); then
    query_status=0
  else
    query_status=$?
    if [[ "$query_status" -ne 1 || -n "$state" ]]; then
      echo "error: could not reliably query dpkg state for $package (status $query_status)." >&2
      return 1
    fi
    printf '\n'
    return 0
  fi
  IFS=$'\t' read -r state error_flag version <<<"$state"
  if [[ "$error_flag" != ok ]]; then
    echo "error: $package has dpkg error flag '$error_flag'; repair it before building." >&2
    return 1
  fi
  case "$state" in
    installed)
      [[ -n "$version" ]] || { echo "error: missing installed version for $package." >&2; return 1; }
      printf '%s\n' "$version" ;;
    ''|not-installed|config-files) printf '\n' ;;
    *) echo "error: $package has partial dpkg state '$state'; repair it before building." >&2; return 1 ;;
  esac
}

ds_vpi_is_x86_package_version() {
  [[ "$1" =~ ^([0-9]+:)?4\.1\.4([+-]|$) ]]
}

ds_vpi_install_x86() {
  local provider=$1 destination=$2 explicit_provider=$3
  local runtime_version dev_version selected= candidate package installed holds
  local runtime_versions dev_versions
  # shellcheck source=lib_vpi_runtime.sh
  source "$SCRIPT_DIR/lib_vpi_runtime.sh"
  if [[ "$explicit_provider" -eq 0 ]]; then
    runtime_version=$(ds_vpi_installed_package libnvvpi4) || return 1
    dev_version=$(ds_vpi_installed_package vpi4-dev) || return 1
    if [[ -n "$runtime_version" && "$runtime_version" == "$dev_version" ]] &&
       ds_vpi_is_x86_package_version "$runtime_version"; then
      echo "--- VPI4.1.4 (healthy installed runtime/development pair; no APT changes) ---"
    else
      apt-get install -y gnupg software-properties-common || return 1
      apt-key adv --fetch-key https://repo.download.nvidia.com/jetson/jetson-ota-public.asc || return 1
      if [[ "$(lsb_release -cs)" == noble ]]; then
        add-apt-repository -y 'deb https://repo.download.nvidia.com/jetson/x86_64/noble r39.2 main' || return 1
      else
        add-apt-repository -y 'deb https://repo.download.nvidia.com/jetson/x86_64/jammy r39.2 main' || return 1
      fi
      apt-get update || return 1
      runtime_versions=$(apt-cache madison libnvvpi4 | awk '{print $3}') || return 1
      dev_versions=$(apt-cache madison vpi4-dev | awk '{print $3}') || return 1
      while IFS= read -r candidate; do
        if ds_vpi_is_x86_package_version "$candidate" && grep -Fxq -- "$candidate" <<<"$dev_versions"; then
          selected=$candidate
          break
        fi
      done <<<"$runtime_versions"
      if [[ -z "$selected" ]]; then
        echo "error: the x86 repository has no matching libnvvpi4/vpi4-dev VPI4.1.4 package pair." >&2
        return 1
      fi
      holds=$(apt-mark showhold) || return 1
      for package in libnvvpi4 vpi4-dev; do
        installed=$runtime_version
        [[ "$package" != vpi4-dev ]] || installed=$dev_version
        if [[ -n "$installed" ]] && dpkg --compare-versions "$installed" gt "$selected"; then
          echo "error: refusing to downgrade $package from $installed to $selected; review the VPI provider separately." >&2
          return 1
        fi
        if grep -Fxq -- "$package" <<<"$holds" && [[ "$installed" != "$selected" ]]; then
          echo "error: $package is held at $installed; the build will not override the hold for $selected." >&2
          return 1
        fi
      done
      apt-get install -y --no-remove "libnvvpi4=$selected" "vpi4-dev=$selected" || return 1
      runtime_version=$(ds_vpi_installed_package libnvvpi4) || return 1
      dev_version=$(ds_vpi_installed_package vpi4-dev) || return 1
      if [[ "$runtime_version" != "$selected" || "$dev_version" != "$selected" ]]; then
        echo "error: VPI installation did not produce the matching $selected runtime/development pair." >&2
        return 1
      fi
    fi
  else
    echo "--- VPI4.1.4 (validating explicit preprovisioned x86 provider $provider; no APT changes) ---"
  fi
  ds_vpi_validate_x86_headers "$provider" || return 1
  ds_vpi_stage_x86_runtime "$provider/lib/x86_64-linux-gnu" "$destination" || return 1
  mark_stage_done deps-vpi4
  echo "    Stage deps-vpi4: DONE (x86 VPI4.1.4 headers and SDK runtime verified)"
}

echo ""
if [[ "$PLATFORM" == aarch64 && -r /proc/device-tree/model ]] &&
    [[ "$(tr -d '\0' < /proc/device-tree/model)" == *IGX* ]]; then
  echo "--- VPI (IGX keeps its existing platform stack; this upgrade does not install a replacement) ---"
  mark_stage_done deps-vpi4
elif [[ "$PLATFORM" == aarch64 || "$PLATFORM" == sbsa ]]; then
  VPI_DIR=${VPI_DIR-/opt/nvidia/vpi4}
  VPI_HEADER="$VPI_DIR/include/vpi/VersionDef.h"
  if [[ ! -r "$VPI_HEADER" ]]; then
    echo "error: ARM requires authentic VPI4.1.6 development headers at $VPI_DIR; install the matching vendor package first." >&2
    exit 1
  fi
  VPI_VERSION=$(printf '%s\n' 'NV_VPI_VERSION_MAJOR NV_VPI_VERSION_MINOR NV_VPI_VERSION_PATCH' |
    cc -E -P -x c -include "$VPI_HEADER" -) || exit 1
  VPI_VERSION=$(echo "$VPI_VERSION" | xargs | tr ' ' '.')
  if [[ "$VPI_VERSION" != 4.1.6 ]]; then
    echo "error: ARM requires VPI4.1.6; selected headers declare $VPI_VERSION." >&2
    exit 1
  fi
  echo "--- VPI4.1.6 (using existing ARM vendor headers; no x86 repository added) ---"
  mark_stage_done deps-vpi4
else
  # A previous DONE marker must not mask a 4.1.3 bundle restored by artifacts.
  VPI_EXPLICIT_PROVIDER=0
  [[ ! -v VPI_DIR ]] || VPI_EXPLICIT_PROVIDER=1
  ds_vpi_install_x86 "${VPI_DIR-/opt/nvidia/vpi4}" "$INSTALL_DIR" "$VPI_EXPLICIT_PROVIDER"
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
if [[ "$PLATFORM" = "sbsa" ]]; then
  echo "==> Done (SBSA: stages 1–3 and 5–7 skipped; azure-iot and remaining deps installed)."
  ls "$INSTALL_DIR"/libiothub_client* 2>/dev/null
else
  echo "==> Done. Installed to $INSTALL_DIR:"
  ls "$INSTALL_DIR"/libopentelemetry_* "$INSTALL_DIR"/libcivetweb* "$INSTALL_DIR"/libprometheus* "$INSTALL_DIR"/libiothub_client* 2>/dev/null
fi
