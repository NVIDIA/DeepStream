#!/usr/bin/env bash
# Fetch, build, and install open-source DeepStream dependencies from GitHub.
# Installs shared libraries directly to the DeepStream opt path.
#
# Dependencies built:
#   opentelemetry-cpp  v1.23.0            -> libopentelemetry_*.so  (HTTP + gRPC exporters)
#   civetweb           v1.16              -> libcivetweb.so
#   prometheus-cpp     v1.2.4             -> libprometheus-cpp-core.so
#   azure-iot-sdk-c    commit dbeb521e     -> libiothub_client.so (v1.11.0)
#
# Usage:
#   sudo bash scripts/install_opensource_deps.sh [NVDS_VERSION=9.0]

set -e

NVDS_VERSION=${NVDS_VERSION:-9.0}
INSTALL_DIR=/opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/lib
BUILD_ROOT=$(mktemp -d /tmp/ds-deps-build.XXXXXX)

echo "==> Installing open-source DeepStream dependencies"
echo "    Install dir : $INSTALL_DIR"
echo "    Build root  : $BUILD_ROOT"

mkdir -p "$INSTALL_DIR"

# ---------------------------------------------------------------------------
# 1. opentelemetry-cpp v1.23.0
# ---------------------------------------------------------------------------
echo ""
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

# ---------------------------------------------------------------------------
# 2. civetweb v1.16
# ---------------------------------------------------------------------------
echo ""
echo "--- civetweb v1.16 ---"
CIVETWEB_DIR=$BUILD_ROOT/civetweb
git clone --depth 1 --branch v1.16 \
    https://github.com/civetweb/civetweb.git "$CIVETWEB_DIR"
cmake -S "$CIVETWEB_DIR" -B "$BUILD_ROOT/civetweb-build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DCIVETWEB_ENABLE_CXX=ON \
    -DCIVETWEB_BUILD_TESTING=OFF \
    -DCIVETWEB_ENABLE_SERVER_EXECUTABLE=OFF \
    -DCMAKE_INSTALL_PREFIX="$BUILD_ROOT/civetweb-install" \
    -Wno-dev
cmake --build "$BUILD_ROOT/civetweb-build" -j$(nproc)
cmake --install "$BUILD_ROOT/civetweb-build"

find "$BUILD_ROOT/civetweb-install/lib" -name "libcivetweb.so*" -o -name "libcivetweb-cpp.so*" | xargs -I{} cp -Pv {} "$INSTALL_DIR/"

# ---------------------------------------------------------------------------
# 3. prometheus-cpp v1.2.4
# ---------------------------------------------------------------------------
echo ""
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

# ---------------------------------------------------------------------------
# 4. azure-iot-sdk-c LTS_01_2023_Ref01 (v1.11.0)
# ---------------------------------------------------------------------------
echo ""
echo "--- azure-iot-sdk-c v1.11.0 (commit dbeb521e) ---"
AZURE_DIR=$BUILD_ROOT/azure-iot-sdk-c
# LTS_01_2023_Ref01 tag = v1.10.0; v1.11.0 is the next commit (dbeb521e)
git clone --recursive https://github.com/Azure/azure-iot-sdk-c.git "$AZURE_DIR"
git -C "$AZURE_DIR" checkout dbeb521e
git -C "$AZURE_DIR" submodule update --init --recursive
cmake -S "$AZURE_DIR" -B "$BUILD_ROOT/azure-build" \
    -DCMAKE_BUILD_TYPE=Release \
    -Dbuild_as_dynamic=ON \
    -Duse_edge_modules=ON \
    -Dskip_samples=ON \
    -Duse_amqp=ON \
    -Duse_mqtt=ON \
    -Duse_http=ON \
    -DCMAKE_INSTALL_PREFIX="$BUILD_ROOT/azure-install" \
    -Wno-dev
cmake --build "$BUILD_ROOT/azure-build" -j$(nproc)
cmake --install "$BUILD_ROOT/azure-build"

find "$BUILD_ROOT/azure-install/lib" -name "libiothub_client.so*" -exec cp -Pv {} "$INSTALL_DIR/" \;

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
