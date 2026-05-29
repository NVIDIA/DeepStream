#!/usr/bin/env bash
# Build and install all DeepStream components from this repository.
# Usage (run from repo root): bash build/build.sh [OPTIONS] [CUDA_VER=<ver>] [NVDS_VERSION=<ver>]
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

usage() {
  cat <<'EOF'
Usage: bash build/build.sh [OPTIONS] [CUDA_VER=<ver>] [NVDS_VERSION=<ver>] [CMAKE_BIN=<path>]

Build and install DeepStream open-source components from this repository.

Options:
  -h, --help              Show this help and exit
  --skip-deps             Skip scripts/install_opensource_deps.sh
  --only=STAGE[,STAGE]    Build only the named stage(s):
                            gst-utils, utils, gst-plugins, sample_apps,
                            tao_apps, reference_apps, service-maker
  --verbose               Show stderr from sub-makes (no 2>/dev/null suppression)
  -j N                    Parallel jobs for make and cmake (default: nproc)

Environment variables (alternative to CLI):
  CUDA_VER                CUDA toolkit version (default: 13.1 x86, 13.0 aarch64/sbsa)
  NVDS_VERSION            DeepStream install version (default: 9.0)
  CMAKE_BIN               Path to cmake binary

Examples:
  bash build/build.sh
  bash build/build.sh --only=gst-plugins --skip-deps
  bash build/build.sh --only=service-maker -j8
  bash build/build.sh --only=gst-plugins --verbose
  CUDA_VER=13.1 bash build/build.sh --skip-deps
EOF
}

ARCH=$(uname -m)
case "$ARCH" in
  x86_64)  PLATFORM=x86;     CUDA_VER_DEFAULT=13.1 ;;
  aarch64)
    if [ -f /etc/nv_tegra_release ]; then
      PLATFORM=aarch64
    else
      PLATFORM=sbsa
    fi
    CUDA_VER_DEFAULT=13.0
    ;;
  *)        echo "Unsupported architecture: $ARCH"; exit 1 ;;
esac

SKIP_DEPS=0
VERBOSE=0
ONLY_STAGES=()
JOBS=$(nproc 2>/dev/null || echo 4)
CUDA_VER=${CUDA_VER:-$CUDA_VER_DEFAULT}
NVDS_VERSION=${NVDS_VERSION:-9.0}

while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --skip-deps)       SKIP_DEPS=1 ;;
    --verbose)         VERBOSE=1 ;;
    --only=*)
      IFS=',' read -ra _only_parts <<< "${1#--only=}"
      ONLY_STAGES+=("${_only_parts[@]}")
      ;;
    --only)
      shift
      if [ $# -eq 0 ]; then
        echo "error: --only requires a stage name" >&2
        exit 1
      fi
      IFS=',' read -ra _only_parts <<< "$1"
      ONLY_STAGES+=("${_only_parts[@]}")
      ;;
    -j)
      shift
      if [ $# -eq 0 ]; then
        echo "error: -j requires a job count" >&2
        exit 1
      fi
      JOBS="$1"
      ;;
    -j[0-9]*)
      JOBS="${1#-j}"
      ;;
    CUDA_VER=*)
      CUDA_VER="${1#CUDA_VER=}"
      ;;
    NVDS_VERSION=*)
      NVDS_VERSION="${1#NVDS_VERSION=}"
      ;;
    CMAKE_BIN=*)
      CMAKE_BIN="${1#CMAKE_BIN=}"
      ;;
    *)
      echo "error: unknown argument: $1 (try --help)" >&2
      exit 1
      ;;
  esac
  shift
done

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# build.sh lives in build/; cd to the repo root so relative paths resolve.
cd "$SCRIPT_DIR/.."

VALID_ONLY_STAGES=(
  gst-utils utils gst-plugins sample_apps
  tao_apps reference_apps service-maker
)

want_stage() {
  local stage=$1
  if [ ${#ONLY_STAGES[@]} -eq 0 ]; then
    return 0
  fi
  local s
  for s in "${ONLY_STAGES[@]}"; do
    if [ "$s" = "$stage" ]; then
      return 0
    fi
  done
  return 1
}

for s in "${ONLY_STAGES[@]}"; do
  valid=0
  for v in "${VALID_ONLY_STAGES[@]}"; do
    if [ "$s" = "$v" ]; then
      valid=1
      break
    fi
  done
  if [ "$valid" -eq 0 ]; then
    echo "error: unknown --only stage '$s' (valid: ${VALID_ONLY_STAGES[*]})" >&2
    exit 1
  fi
done

begin_stage() {
  local stage=$1
  if ! want_stage "$stage"; then
    return 1
  fi
  echo "==> Stage: $stage"
  return 0
}

run_submake() {
  if [ "$VERBOSE" -eq 1 ]; then
    run_as_root make -j"$JOBS" "$@"
  else
    run_submake_quiet "$@"
  fi
}

run_submake_quiet() {
  run_as_root make -j"$JOBS" "$@" 2>/dev/null
}

run_user_make() {
  if [ "$VERBOSE" -eq 1 ]; then
    make -j"$JOBS" "$@"
  else
    make -j"$JOBS" "$@" 2>/dev/null
  fi
}

# Extra make flags for SBSA
PLATFORM_MAKE_FLAGS=
[ "$PLATFORM" = "sbsa" ] && PLATFORM_MAKE_FLAGS="AARCH64_IS_SBSA=1"

CMAKE_BIN=$(find_cmake)
SM_APP_BUILD=/tmp/ds-sm-apps
SM_MOD_BUILD=/tmp/ds-sm-modules

BUILD_LOG="$SCRIPT_DIR/build.log"
: >"$BUILD_LOG"
exec > >(tee "$BUILD_LOG") 2>&1

echo "==> Build log: $BUILD_LOG (overwritten each run)"
echo "==> Building PLATFORM=$PLATFORM CUDA_VER=$CUDA_VER NVDS_VERSION=$NVDS_VERSION (outputs to /opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/)"
echo "==> Using cmake: $CMAKE_BIN (jobs: $JOBS)"
if [ ${#ONLY_STAGES[@]} -gt 0 ]; then
  echo "==> Scoped build: ${ONLY_STAGES[*]}"
fi

MK="CUDA_VER=$CUDA_VER NVDS_VERSION=$NVDS_VERSION $PLATFORM_MAKE_FLAGS"
FAILED_BUILDS=()

# ---------------------------------------------------------------------------
# Stage: deps
# ---------------------------------------------------------------------------
if begin_stage deps; then
  if [ "$SKIP_DEPS" -eq 1 ]; then
    echo "    (--skip-deps: skipping install_opensource_deps.sh)"
  else
    echo "==> Installing open-source dependencies"
    run_as_root env NVDS_VERSION="$NVDS_VERSION" PLATFORM="$PLATFORM" bash "$SCRIPT_DIR/../scripts/install_opensource_deps.sh"
  fi
fi

# ---------------------------------------------------------------------------
# Stage: gst-utils
# ---------------------------------------------------------------------------
if begin_stage gst-utils; then
  run_as_root make -j"$JOBS" -C src/utils/nvds_rest_server $MK
  run_as_root make -j"$JOBS" -C src/gst-utils/gstnvcustomhelper $MK
  run_as_root make -j"$JOBS" -C src/gst-utils/gst-nvdssr $MK
  run_as_root make -j"$JOBS" -C src/gst-utils/gstnvdscustomhelper $MK
fi

# ---------------------------------------------------------------------------
# Stage: utils
# ---------------------------------------------------------------------------
if begin_stage utils; then
  for dir in src/utils/*/; do
    [ -f "$dir/Makefile" ] || continue
    run_submake -C "$dir" $MK || FAILED_BUILDS+=("$dir")
  done

  for dir in src/utils/nvds_msgapi/*/; do
    if [ -f "$dir/Makefile" ]; then
      run_submake -C "$dir" $MK || FAILED_BUILDS+=("$dir")
    else
      for sub in "$dir"*/; do
        if [ -f "$sub/Makefile" ]; then
          run_submake -C "$sub" $MK || FAILED_BUILDS+=("$sub")
        fi
      done
    fi
  done

  for dir in src/utils/ds3d/*/; do
    if [ -f "$dir/Makefile" ]; then
      run_submake -C "$dir" $MK || FAILED_BUILDS+=("$dir")
    else
      for sub in "$dir"*/; do
        if [ -f "$sub/Makefile" ]; then
          run_submake -C "$sub" $MK || FAILED_BUILDS+=("$sub")
        fi
      done
    fi
  done
fi

# ---------------------------------------------------------------------------
# Stage: gst-plugins
# ---------------------------------------------------------------------------
if begin_stage gst-plugins; then
  for dir in src/gst-plugins/*/; do
    run_submake -C "$dir" $MK || FAILED_BUILDS+=("$dir")
  done
fi

# ---------------------------------------------------------------------------
# Stage: sample_apps
# ---------------------------------------------------------------------------
if begin_stage sample_apps; then
  SA_BIN_STAGE=/tmp/ds-sample-apps-bin
  DS_BIN=/opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/bin
  mkdir -p "$SA_BIN_STAGE"
  for dir in src/apps/sample_apps/*/; do
    app=$(basename "$dir")
    if [ "$PLATFORM" != "x86" ] && [[ "$app" == "deepstream-ucx-test" || "$app" == "deepstream-multigpu-nvlink-test" ]]; then
      echo "Skipping $app on $PLATFORM (x86 only)"
      continue
    fi
    if [ "$PLATFORM" = "x86" ] && [ "$app" = "deepstream-ipc-test" ]; then
      echo "Skipping $app on $PLATFORM (Jetson only)"
      continue
    fi
    if run_user_make -C "$dir" $MK BIN_DIR="$SA_BIN_STAGE"; then
      for bin in "$SA_BIN_STAGE"/deepstream-*; do
        [ -f "$bin" ] && [ -x "$bin" ] && run_as_root cp -v "$bin" "$DS_BIN/" || true
      done
    else
      FAILED_BUILDS+=("$dir")
    fi
    rm -f "$SA_BIN_STAGE"/deepstream-*
  done
  rm -rf "$SA_BIN_STAGE"
fi

# ---------------------------------------------------------------------------
# Stage: yolo (apt / eigen / Yolo custom lib)
# ---------------------------------------------------------------------------
if begin_stage yolo; then
  last_update=$(stat -c %Y /var/cache/apt/pkgcache.bin 2>/dev/null || echo 0)
  now=$(date +%s)
  if [ $((now - last_update)) -gt 86400 ]; then
    run_as_root apt update || true
  else
    echo "==> apt cache is fresh, skipping apt update"
  fi
  if ! dpkg -s libeigen3-dev >/dev/null 2>&1; then
    run_as_root apt install -y libeigen3-dev
  else
    echo "==> libeigen3-dev already installed"
  fi
  run_as_root ln -sf /usr/include/eigen3/Eigen /usr/include/Eigen
  run_user_make -C tools/yolo_deepstream/deepstream_yolo/nvdsinfer_custom_impl_Yolo/ $MK \
    || FAILED_BUILDS+=("tools/yolo_deepstream/deepstream_yolo/nvdsinfer_custom_impl_Yolo/")
fi

# ---------------------------------------------------------------------------
# Stage: tao_apps
# ---------------------------------------------------------------------------
if begin_stage tao_apps; then
  if git submodule status 2>/dev/null | grep -q '^-'; then
    git submodule update --init --recursive || true
  else
    echo "==> git submodules already initialized"
  fi
  run_submake -C src/apps/tao_apps $MK || FAILED_BUILDS+=("src/apps/tao_apps")
fi

# ---------------------------------------------------------------------------
# Stage: reference_apps
# ---------------------------------------------------------------------------
if begin_stage reference_apps; then
  RA_BIN_STAGE=/tmp/ds-reference-apps-bin
  DS_BIN=/opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/bin
  mkdir -p "$RA_BIN_STAGE"
  for dir in src/apps/reference_apps/*/; do
    if [ ! -f "$dir/Makefile" ]; then
      echo "Skipping $(basename "$dir") (no Makefile; see its README for build steps)"
      continue
    fi
     if run_submake -C "$dir" $MK BIN_DIR="$RA_BIN_STAGE"; then
      for bin in "$RA_BIN_STAGE"/deepstream-*; do
        [ -f "$bin" ] && [ -x "$bin" ] && run_as_root cp -v "$bin" "$DS_BIN/" || true
      done
    else
      FAILED_BUILDS+=("$dir")
    fi
    rm -f "$RA_BIN_STAGE"/deepstream-*
  done
  rm -rf "$RA_BIN_STAGE"
fi

# ---------------------------------------------------------------------------
# Stage: service-maker
# ---------------------------------------------------------------------------
if begin_stage service-maker; then
  SM_BIN_INSTALL=/opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/bin
  for dir in src/service-maker/sources/apps/cpp/*/; do
    app=$(basename "$dir")
    "$CMAKE_BIN" -S "$dir" -B "$SM_APP_BUILD/$app" -DCMAKE_BUILD_TYPE=Release
    "$CMAKE_BIN" --build "$SM_APP_BUILD/$app" -j"$JOBS"
    for bin in "$SM_APP_BUILD/$app"/deepstream-*; do
      [ -f "$bin" ] && [ -x "$bin" ] || continue
      name=$(basename "$bin" | sed 's/^deepstream-/service-maker-/')
      run_as_root cp -v "$bin" "$SM_BIN_INSTALL/$name"
    done
  done

  SM_MODULES_INSTALL=/opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/service-maker/modules
  run_as_root mkdir -p "$SM_MODULES_INSTALL"
  for dir in src/service-maker/sources/modules/*/; do
    mod=$(basename "$dir")
    "$CMAKE_BIN" -S "$dir" -B "$SM_MOD_BUILD/$mod" -DCMAKE_BUILD_TYPE=Release
    "$CMAKE_BIN" --build "$SM_MOD_BUILD/$mod" -j"$JOBS"
    run_as_root cp -v "$SM_MOD_BUILD/$mod/lib${mod}.so" "$SM_MODULES_INSTALL/"
  done
  rm -rf "$SM_APP_BUILD" "$SM_MOD_BUILD"
fi

echo ""
echo "==> Build log: $BUILD_LOG"

if [ ${#FAILED_BUILDS[@]} -gt 0 ]; then
  echo "==> Build completed with ${#FAILED_BUILDS[@]} failure(s):"
  for fb in "${FAILED_BUILDS[@]}"; do
    echo "    FAILED: $fb"
  done
  exit 1
fi

echo "==> Done (all builds succeeded)"
exit 0
