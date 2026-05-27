#!/usr/bin/env bash
# Build and install all DeepStream components from this repository.
# Usage (run from repo root): bash build/build.sh [CUDA_VER=<ver>]
set -e

run_as_root() {
  if [ "$(id -u)" -eq 0 ]; then
    "$@"
  elif command -v sudo >/dev/null 2>&1; then
    if ! sudo -n true 2>/dev/null; then
      if [ -r /dev/tty ] && [ -w /dev/tty ]; then
        printf '==> Root privileges required for: %s\n' "$*" >/dev/tty
        sudo -v </dev/tty >/dev/tty 2>&1
      else
        echo "==> Root privileges required for: $*" >&2
        sudo -v
      fi
    fi
    sudo "$@"
  else
    echo "This step needs root privileges, but sudo was not found. Re-run as root or install sudo." >&2
    exit 1
  fi
}

find_cmake() {
  local candidate resolved
  local candidates=()

  [ -n "${CMAKE_BIN:-}" ] && candidates+=("$CMAKE_BIN")
  candidates+=("cmake" "/usr/bin/cmake")

  for candidate in "${candidates[@]}"; do
    if resolved=$(command -v "$candidate" 2>/dev/null); then
      if "$resolved" --version >/dev/null 2>&1; then
        printf '%s\n' "$resolved"
        return 0
      fi
    fi
  done

  echo "Unable to find a working cmake. Install cmake or set CMAKE_BIN=/path/to/cmake." >&2
  exit 1
}

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
CMAKE_BIN=$(find_cmake)
SM_APP_BUILD=/tmp/ds-sm-apps
SM_MOD_BUILD=/tmp/ds-sm-modules
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# build.sh lives in build/; cd to the repo root so the relative
# make -C tools/... and src/... paths below resolve regardless of where
# the script was invoked from.
cd "$SCRIPT_DIR/.."

# Extra make flags for SBSA — passes AARCH64_IS_SBSA=1 to all Makefiles
PLATFORM_MAKE_FLAGS=
[ "$PLATFORM" = "sbsa" ] && PLATFORM_MAKE_FLAGS="AARCH64_IS_SBSA=1"

# ---------------------------------------------------------------------------
# Prerequisites: install open-source dependencies
# (Proprietary prebuilt libs are provided by the DS 9.0 deb install at
#  /opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/lib/.)
# ---------------------------------------------------------------------------
echo "==> Installing open-source dependencies"
run_as_root env NVDS_VERSION="$NVDS_VERSION" PLATFORM="$PLATFORM" bash "$SCRIPT_DIR/../scripts/install_opensource_deps.sh"

echo "==> Building PLATFORM=$PLATFORM CUDA_VER=$CUDA_VER (outputs directly to /opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/)"
echo "==> Using cmake: $CMAKE_BIN"

MK="CUDA_VER=$CUDA_VER NVDS_VERSION=$NVDS_VERSION $PLATFORM_MAKE_FLAGS"

FAILED_BUILDS=()

# nvds_rest_server must be built before gstnvdscustomhelper (it links against it)
run_as_root make -C src/utils/nvds_rest_server $MK

# gst-utils (order matters — gstnvcustomhelper must come first)
run_as_root make -C src/gst-utils/gstnvcustomhelper $MK
run_as_root make -C src/gst-utils/gst-nvdssr $MK
run_as_root make -C src/gst-utils/gstnvdscustomhelper $MK

# Utils (no top-level Makefile — build each individually)
for dir in src/utils/*/; do
  [ -f "$dir/Makefile" ] || continue
  run_as_root make -C "$dir" $MK 2>/dev/null || FAILED_BUILDS+=("$dir")
done

# nvds_msgapi message broker adaptors (Makefiles live one level deeper than
# the rest of src/utils, and azure_protocol_adaptor splits further into
# device_client/ and module_client/).
for dir in src/utils/nvds_msgapi/*/; do
  if [ -f "$dir/Makefile" ]; then
    run_as_root make -C "$dir" $MK 2>/dev/null || FAILED_BUILDS+=("$dir")
  else
    for sub in "$dir"*/; do
      if [ -f "$sub/Makefile" ]; then
        run_as_root make -C "$sub" $MK 2>/dev/null || FAILED_BUILDS+=("$sub")
      fi
    done
  fi
done

# ds3d utilities (Makefiles are two levels deep: category/component/Makefile)
for dir in src/utils/ds3d/*/; do
  if [ -f "$dir/Makefile" ]; then
    run_as_root make -C "$dir" $MK 2>/dev/null || FAILED_BUILDS+=("$dir")
  else
    for sub in "$dir"*/; do
      if [ -f "$sub/Makefile" ]; then
        run_as_root make -C "$sub" $MK 2>/dev/null || FAILED_BUILDS+=("$sub")
      fi
    done
  fi
done

# GStreamer plugins (no top-level Makefile — build each individually)
for dir in src/gst-plugins/*/; do
  run_as_root make -C "$dir" $MK 2>/dev/null || FAILED_BUILDS+=("$dir")
done

# Sample apps (no top-level Makefile — build each individually)
# Build to a staging dir first so a failed link step does not delete the
# deb-installed binary from the DS bin dir (make removes its output on error).
SA_BIN_STAGE=/tmp/ds-sample-apps-bin
DS_BIN=/opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/bin
mkdir -p "$SA_BIN_STAGE"
for dir in src/apps/sample_apps/*/; do
  # deepstream-ucx-test is x86-only (no aarch64 UCX support shipped)
  if [ "$PLATFORM" != "x86" ] && [ "$(basename "$dir")" = "deepstream-ucx-test" ]; then
    echo "Skipping deepstream-ucx-test on $PLATFORM (x86 only)"
    continue
  fi
  if make -C "$dir" $MK BIN_DIR="$SA_BIN_STAGE" 2>/dev/null; then
    for bin in "$SA_BIN_STAGE"/deepstream-*; do
      [ -f "$bin" ] && [ -x "$bin" ] && run_as_root cp -v "$bin" "$DS_BIN/" || true
    done
  else
    FAILED_BUILDS+=("$dir")
  fi
  rm -f "$SA_BIN_STAGE"/deepstream-*
done
rm -rf "$SA_BIN_STAGE"

# Prepare Eigen lib
run_as_root apt update || true
run_as_root apt install -y libeigen3-dev
run_as_root ln -sf /usr/include/eigen3/Eigen /usr/include/Eigen

#Customized Yolo postprocessing lib
make -C tools/yolo_deepstream/deepstream_yolo/nvdsinfer_custom_impl_Yolo/ $MK 2>/dev/null || FAILED_BUILDS+=("tools/yolo_deepstream/deepstream_yolo/nvdsinfer_custom_impl_Yolo/")

# tao-apps — build and install via top-level Makefile
git submodule update --init --recursive || true
run_as_root make -C src/apps/tao_apps $MK 2>/dev/null || FAILED_BUILDS+=("src/apps/tao_apps")

# Reference apps (no top-level Makefile — build each individually)
RA_BIN_STAGE=/tmp/ds-reference-apps-bin
mkdir -p "$RA_BIN_STAGE"
for dir in src/apps/reference_apps/*/; do
  if make -C "$dir" $MK BIN_DIR="$RA_BIN_STAGE" 2>/dev/null; then
    for bin in "$RA_BIN_STAGE"/deepstream-*; do
      [ -f "$bin" ] && [ -x "$bin" ] && run_as_root cp -v "$bin" "$DS_BIN/" || true
    done
  else
    FAILED_BUILDS+=("$dir")
  fi
  rm -f "$RA_BIN_STAGE"/deepstream-*
done
rm -rf "$RA_BIN_STAGE"

# service-maker apps (cmake-based) — build and install
SM_BIN_INSTALL=/opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/bin
for dir in src/service-maker/sources/apps/cpp/*/; do
  app=$(basename "$dir")
  "$CMAKE_BIN" -S "$dir" -B "$SM_APP_BUILD/$app" -DCMAKE_BUILD_TYPE=Release
  "$CMAKE_BIN" --build "$SM_APP_BUILD/$app" -j$(nproc)
  for bin in "$SM_APP_BUILD/$app"/deepstream-*; do
    [ -f "$bin" ] && [ -x "$bin" ] || continue
    name=$(basename "$bin" | sed 's/^deepstream-/service-maker-/')
    run_as_root cp -v "$bin" "$SM_BIN_INSTALL/$name"
  done
done
rm -rf "$SM_APP_BUILD"

# service-maker modules (cmake-based) — build and install
SM_MODULES_INSTALL=/opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/service-maker/modules
run_as_root mkdir -p "$SM_MODULES_INSTALL"
for dir in src/service-maker/sources/modules/*/; do
  mod=$(basename "$dir")
  "$CMAKE_BIN" -S "$dir" -B "$SM_MOD_BUILD/$mod" -DCMAKE_BUILD_TYPE=Release
  "$CMAKE_BIN" --build "$SM_MOD_BUILD/$mod" -j$(nproc)
  run_as_root cp -v "$SM_MOD_BUILD/$mod/lib${mod}.so" "$SM_MODULES_INSTALL/"
done
rm -rf "$SM_MOD_BUILD"

if [ ${#FAILED_BUILDS[@]} -gt 0 ]; then
  echo ""
  echo "==> Build completed with ${#FAILED_BUILDS[@]} failure(s):"
  for fb in "${FAILED_BUILDS[@]}"; do
    echo "    FAILED: $fb"
  done
  echo ""
else
  echo "==> Done (all builds succeeded)"
fi
