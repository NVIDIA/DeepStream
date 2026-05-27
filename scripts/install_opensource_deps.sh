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
# Additional prerequisites installed:
#   half               v2.1.0             -> /opt/half
#   triton gRPC client v2.65.0            -> /opt/tritonclient
#   protobuf compiler  v3.21.12           -> /opt/proto
#   apt packages: libpango, libcairo, libjson-glib, librabbitmq, librdkafka, etc.
#   mosquitto (from PPA)
#
# Usage:
#   sudo PLATFORM=x86 bash scripts/install_opensource_deps.sh [NVDS_VERSION=9.0]
#   sudo PLATFORM=aarch64 bash scripts/install_opensource_deps.sh [NVDS_VERSION=9.0]

set -e

NVDS_VERSION=${NVDS_VERSION:-9.0}
INSTALL_DIR=/opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/lib
BUILD_ROOT=$(mktemp -d /tmp/ds-deps-build.XXXXXX)

# Detect platform if not passed via environment
if [ -z "${PLATFORM:-}" ]; then
  ARCH=$(uname -m)
  case "$ARCH" in
    x86_64)  PLATFORM=x86 ;;
    aarch64)
      if [ -f /etc/nv_tegra_release ]; then
        PLATFORM=aarch64
      else
        PLATFORM=sbsa
      fi
      ;;
    *)  echo "Unsupported architecture: $ARCH"; exit 1 ;;
  esac
fi

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
# 4. azure-iot-sdk-c v1.11.0
#    (follows README in src/utils/nvds_msgapi/azure_protocol_adaptor/module_client)
# ---------------------------------------------------------------------------
echo ""
echo "--- azure-iot-sdk-c v1.11.0 ---"
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

# ---------------------------------------------------------------------------
# 5. Half v2.1.0
# ---------------------------------------------------------------------------
echo ""
echo "--- Half v2.1.0 ---"
mkdir -p /opt/half
wget -q -P /tmp https://sourceforge.net/projects/half/files/half/2.1.0/half-2.1.0.zip
unzip -o /tmp/half-2.1.0.zip -d /opt/half/
rm -f /tmp/half-2.1.0.zip

# ---------------------------------------------------------------------------
# 6. Triton gRPC Client v2.65.0
# ---------------------------------------------------------------------------
echo ""
echo "--- Triton gRPC Client v2.65.0 (PLATFORM=$PLATFORM) ---"
mkdir -p /opt/tritonclient/
if [ "$PLATFORM" = "x86" ]; then
  wget -q -P /tmp https://github.com/triton-inference-server/server/releases/download/v2.65.0/v2.65.0_ubuntu2404.clients.tar.gz
  tar xzf /tmp/v2.65.0_ubuntu2404.clients.tar.gz -C /opt/tritonclient/ lib include
  rm -f /tmp/v2.65.0_ubuntu2404.clients.tar.gz
else
  # Jetson (aarch64) — use the igpu tar; switch to tritonserver2.60.0-agx.tar for AGX systems
  wget -q -P /tmp https://github.com/triton-inference-server/server/releases/download/v2.60.0/tritonserver2.60.0-agx.tar
  tar xf /tmp/tritonserver2.60.0-agx.tar -C /opt/tritonclient/ --strip-components=2 tritonserver/clients/lib/ tritonserver/clients/include/
  rm -f /tmp/tritonserver2.60.0-agx.tar
fi

# ---------------------------------------------------------------------------
# 7. Protobuf compiler v3.21.12
# ---------------------------------------------------------------------------
echo ""
echo "--- Protobuf compiler v3.21.12 (PLATFORM=$PLATFORM) ---"
mkdir -p /opt/proto
PB_REL="https://github.com/protocolbuffers/protobuf/releases"
if [ "$PLATFORM" = "x86" ]; then
  curl -Lo /tmp/protoc-21.12.zip "$PB_REL/download/v21.12/protoc-21.12-linux-x86_64.zip"
else
  curl -Lo /tmp/protoc-21.12.zip "$PB_REL/download/v21.12/protoc-21.12-linux-aarch_64.zip"
fi
unzip -o /tmp/protoc-21.12.zip -d /opt/proto
chmod -R +rx /opt/proto/
rm -f /tmp/protoc-21.12.zip

# ---------------------------------------------------------------------------
# 8. APT library dependencies
# ---------------------------------------------------------------------------
echo ""
echo "--- APT library dependencies ---"
apt-get update
apt-get install -y \
  libpango1.0-dev libcairo2-dev libjpeg-dev \
  libglib2.0-dev libjson-glib-dev uuid-dev libyaml-cpp-dev \
  librabbitmq-dev librdkafka-dev \
  libjansson4 libjansson-dev \
  libssl-dev libcjson-dev libhiredis-dev

# ---------------------------------------------------------------------------
# 9. Mosquitto
# ---------------------------------------------------------------------------
echo ""
echo "--- Mosquitto ---"
apt-add-repository -y ppa:mosquitto-dev/mosquitto-ppa
apt-get update
apt-get install -y libmosquitto-dev mosquitto

# ---------------------------------------------------------------------------
# 10. Sample-app build prerequisites
# ---------------------------------------------------------------------------
echo ""
echo "--- Sample-app build prerequisites ---"
apt-get update
apt-get install -y \
  libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev \
  libgstrtspserver-1.0-dev libx11-dev \
  gstreamer1.0-libav libavahi-compat-libdnssd-dev \
  libjson-glib-dev libjsoncpp-dev libyaml-cpp-dev \
  libgbm1 libglapi-mesa libgles2-mesa-dev \
  rapidjson-dev

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
