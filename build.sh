#!/usr/bin/env bash
# Build and install all DeepStream mono-repo components.
# Usage: bash build.sh [CUDA_VER=<ver>]
set -e

ARCH=$(uname -m)
case "$ARCH" in
  x86_64)  PLATFORM=x86;     CUDA_VER_DEFAULT=13.1 ;;
  aarch64)
    # Distinguish Jetson (has /etc/nv_tegra_release) from SBSA server hardware
    if [ -f /etc/nv_tegra_release ]; then
      PLATFORM=aarch64
    else
      PLATFORM=sbsa
    fi
    CUDA_VER_DEFAULT=13.0
    ;;
  *)        echo "Unsupported architecture: $ARCH"; exit 1 ;;
esac

CUDA_VER=${CUDA_VER:-$CUDA_VER_DEFAULT}
NVDS_VERSION=${NVDS_VERSION:-9.0}
SM_APP_BUILD=/tmp/ds-sm-apps
SM_MOD_BUILD=/tmp/ds-sm-modules
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Extra make flags for SBSA — passes AARCH64_IS_SBSA=1 to all Makefiles
PLATFORM_MAKE_FLAGS=
[ "$PLATFORM" = "sbsa" ] && PLATFORM_MAKE_FLAGS="AARCH64_IS_SBSA=1"

# ---------------------------------------------------------------------------
# Prerequisites: install open-source dependencies
# (Proprietary prebuilt libs are provided by the DS 9.0 deb install at
#  /opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/lib/.)
# ---------------------------------------------------------------------------
echo "==> Installing open-source dependencies"
sudo NVDS_VERSION=$NVDS_VERSION bash "$SCRIPT_DIR/scripts/install_opensource_deps.sh"

echo "==> Building PLATFORM=$PLATFORM CUDA_VER=$CUDA_VER (outputs directly to /opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/)"

MK="CUDA_VER=$CUDA_VER NVDS_VERSION=$NVDS_VERSION $PLATFORM_MAKE_FLAGS"

# nvds_rest_server must be built before gstnvdscustomhelper (it links against it)
make -C src/utils/nvds_rest_server $MK

# gst-utils (order matters — gstnvcustomhelper must come first)
make -C src/gst-utils/gstnvcustomhelper $MK
make -C src/gst-utils/gst-nvdssr $MK
make -C src/gst-utils/gstnvdscustomhelper $MK

# Utils (no top-level Makefile — build each individually)
for dir in src/utils/*/; do
  make -C "$dir" $MK 2>/dev/null || true
done

# nvds_msgapi message broker adaptors (Makefiles live one level deeper than
# the rest of src/utils, and azure_protocol_adaptor splits further into
# device_client/ and module_client/).
for dir in src/utils/nvds_msgapi/*/; do
  if [ -f "$dir/Makefile" ]; then
    make -C "$dir" $MK 2>/dev/null || true
  else
    for sub in "$dir"*/; do
      [ -f "$sub/Makefile" ] && make -C "$sub" $MK 2>/dev/null || true
    done
  fi
done

# GStreamer plugins (no top-level Makefile — build each individually)
for dir in src/gst-plugins/*/; do
  make -C "$dir" $MK 2>/dev/null || true
done

# Sample apps (no top-level Makefile — build each individually)
# Build to a staging dir first so a failed link step does not delete the
# deb-installed binary from the DS bin dir (make removes its output on error).
SA_BIN_STAGE=/tmp/ds-sample-apps-bin
DS_BIN=/opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/bin
mkdir -p "$SA_BIN_STAGE"
for dir in src/apps/sample_apps/*/; do
  make -C "$dir" $MK BIN_DIR="$SA_BIN_STAGE" 2>/dev/null || true
  for bin in "$SA_BIN_STAGE"/deepstream-*; do
    [ -f "$bin" ] && [ -x "$bin" ] && cp -v "$bin" "$DS_BIN/" || true
  done
  rm -f "$SA_BIN_STAGE"/deepstream-*
done
rm -rf "$SA_BIN_STAGE"

# Prepare Eigen lib
sudo apt update || true
sudo apt install -y libeigen3-dev
sudo ln -sf /usr/include/eigen3/Eigen /usr/include/Eigen

#Customized Yolo postprocessing lib
make -C tools/yolo_deepstream/deepstream_yolo/nvdsinfer_custom_impl_Yolo/ $MK 2>/dev/null || true

# tao-apps — build and install via top-level Makefile
git submodule update --init --recursive || true
make -C src/apps/tao_apps $MK 2>/dev/null || true

# Reference apps (no top-level Makefile — build each individually)
RA_BIN_STAGE=/tmp/ds-reference-apps-bin
mkdir -p "$RA_BIN_STAGE"
for dir in src/apps/reference_apps/*/; do
  make -C "$dir" $MK BIN_DIR="$RA_BIN_STAGE" 2>/dev/null || true
  for bin in "$RA_BIN_STAGE"/deepstream-*; do
    [ -f "$bin" ] && [ -x "$bin" ] && cp -v "$bin" "$DS_BIN/" || true
  done
  rm -f "$RA_BIN_STAGE"/deepstream-*
done
rm -rf "$RA_BIN_STAGE"

# service-maker apps (cmake-based) — build and install
SM_BIN_INSTALL=/opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/bin
for dir in src/service-maker/sources/apps/cpp/*/; do
  app=$(basename "$dir")
  cmake -S "$dir" -B "$SM_APP_BUILD/$app" -DCMAKE_BUILD_TYPE=Release
  cmake --build "$SM_APP_BUILD/$app" -j$(nproc)
  for bin in "$SM_APP_BUILD/$app"/deepstream-*; do
    [ -f "$bin" ] && [ -x "$bin" ] || continue
    name=$(basename "$bin" | sed 's/^deepstream-/service-maker-/')
    cp -v "$bin" "$SM_BIN_INSTALL/$name"
  done
done
rm -rf "$SM_APP_BUILD"

# service-maker modules (cmake-based) — build and install
SM_MODULES_INSTALL=/opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/service-maker/modules
mkdir -p "$SM_MODULES_INSTALL"
for dir in src/service-maker/sources/modules/*/; do
  mod=$(basename "$dir")
  cmake -S "$dir" -B "$SM_MOD_BUILD/$mod" -DCMAKE_BUILD_TYPE=Release
  cmake --build "$SM_MOD_BUILD/$mod" -j$(nproc)
  cp -v "$SM_MOD_BUILD/$mod/lib${mod}.so" "$SM_MODULES_INSTALL/"
done
rm -rf "$SM_MOD_BUILD"

echo "==> Done"
