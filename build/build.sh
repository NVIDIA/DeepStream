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

# Build and install all DeepStream components from this repository.
# Usage (run from repo root): bash build/build.sh [OPTIONS] [CUDA_VER=<ver>] [NVDS_VERSION=<ver>]
set -e

# Build stages are written to build/.stage-state as "DONE <name>".
# Dependency sub-stages (deps-*) use build/.stage-state.deps (install_opensource_deps.sh).
# Artifact sub-stages use build/.stage-state.artifacts (install_artifacts.sh).
readonly ARTIFACTS_STAGE='artifacts'
readonly DEPS_STAGE='deps'
readonly VALID_ONLY_STAGES=(
  gst-utils utils gst-plugins sample_apps
  tao_apps reference_apps service-maker
)

run_as_root() {
  if [[ "$(id -u)" -eq 0 ]]; then
    "$@"
  elif command -v sudo >/dev/null 2>&1; then
    if ! sudo -n true 2>/dev/null; then
      if [[ -r /dev/tty ]] && [[ -w /dev/tty ]]; then
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

  [[ -n "${CMAKE_BIN:-}" ]] && candidates+=("$CMAKE_BIN")
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
  --install-method     Artifact install method: deb (default) | tar
  --keep-assets           Keep the downloaded release assets in ./artifacts after a
                          successful build (default: delete them once the build succeeds)
  --skip-deps             Skip dependency install; do not read or write deps state
  --skip-artifacts        Skip artifact install; do not read or write artifacts state
  --only=STAGE[,STAGE]    Build only the named component stage(s); see below
  --resume                Skip stages already marked complete in build/.stage-state
  --verbose               Show stderr from sub-makes (no 2>/dev/null suppression)
  -j N                    Parallel jobs for make and cmake (default: nproc)

Component stages (--only=):
  gst-utils, utils, gst-plugins, sample_apps, tao_apps, reference_apps, service-maker

Full builds also run: artifacts (proprietary libs and samples via
scripts/install_artifacts.sh), deps (open-source dependencies), yolo (YOLO
custom lib), and install.sh (system integration).
These are not selectable via --only.

Release assets:
  On x86 and aarch64 (Jetson), the artifacts stage downloads the prebuilt
  release assets from the DeepStream GitHub release into ./artifacts (created
  if missing) with wget, then installs them. Only the assets for the selected
  install method are pulled: deb (default) pulls the .deb packages, tar pulls
  the tarballs. By default the downloaded assets are deleted once the whole
  build succeeds; pass --keep-assets to retain them. On SBSA no assets are
  downloaded (DeepStream is pre-installed in the container).

Stage state:
  build/.stage-state            — build stages (artifacts, deps, gst-utils, ...)
  build/.stage-state.deps       — dependency sub-stages (deps-opentelemetry, ...)
  build/.stage-state.artifacts  — artifact sub-stages (artifacts-proprietary-*, ...)
  Each completed stage is recorded as "DONE <name>". Without --resume, the
  relevant file is reset before that stage runs. With --resume, completed
  stages are skipped. Sub-stages are managed by scripts/install_artifacts.sh
  and scripts/install_opensource_deps.sh in their own state files.

Artifact install method (--install-method=):
  deb (default)  Download and install prebuilt Debian packages
                 (deepstream-binaries-*, deepstream-sample-data_*) with dpkg.
  tar            Download and extract the deepstream-binaries-* and
                 deepstream-sample-data_* tarballs into the opt install tree.

Environment variables (alternative to CLI):
  CUDA_VER                CUDA toolkit version (default: 13.2)
  NVDS_VERSION            DeepStream install version (default: 9.1)
  INSTALL_METHOD          Artifact install method: deb (default) | tar
  CMAKE_BIN               Path to cmake binary

Examples:
  bash build/build.sh
  bash build/build.sh --install-method=tar
  bash build/build.sh --keep-assets
  bash build/build.sh --resume
  bash build/build.sh --only=gst-plugins --skip-deps
  bash build/build.sh --only=service-maker -j8
  bash build/build.sh --only=gst-plugins --verbose
  CUDA_VER=13.2 bash build/build.sh --skip-deps
EOF
}

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
  mkdir -p "$(dirname "$STAGE_STATE_FILE")"
  if [[ -f "$STAGE_STATE_FILE" ]]; then
    grep -vx "DONE ${stage}" "$STAGE_STATE_FILE" > "${STAGE_STATE_FILE}.tmp" 2>/dev/null || true
    commit_stage_state_tmp
  else
    : >"$STAGE_STATE_FILE"
  fi
  echo "DONE ${stage}" >>"$STAGE_STATE_FILE"
}

clear_stage_done() {
  local stage=$1
  [[ -f "$STAGE_STATE_FILE" ]] || return 0
  if grep -qx "DONE ${stage}" "$STAGE_STATE_FILE"; then
    grep -vx "DONE ${stage}" "$STAGE_STATE_FILE" > "${STAGE_STATE_FILE}.tmp" 2>/dev/null || true
    commit_stage_state_tmp
  fi
}

reset_stage_state() {
  if [[ ${#ONLY_STAGES[@]} -gt 0 ]]; then
    local stage
    for stage in "${ONLY_STAGES[@]}"; do
      clear_stage_done "$stage"
    done
    return
  fi

  : >"$STAGE_STATE_FILE"
}

ARCH=$(uname -m)
case "$ARCH" in
  x86_64)  PLATFORM=x86;     CUDA_VER_DEFAULT=13.2 ;;
  aarch64)
    if [[ -f /etc/nv_tegra_release ]]; then
      PLATFORM=aarch64
    else
      PLATFORM=sbsa
    fi
    CUDA_VER_DEFAULT=13.2
    ;;
  *)        echo "Unsupported architecture: $ARCH"; exit 1 ;;
esac

SKIP_DEPS=0
# SBSA bare-metal DeepStream installation is not supported; artifacts are shipped
# inside the Docker container only. Skip the artifacts stage automatically on SBSA.
SKIP_ARTIFACTS=0
[[ "$PLATFORM" = "sbsa" ]] && SKIP_ARTIFACTS=1
VERBOSE=0
RESUME=0
# Downloaded release assets are removed after a successful build unless --keep-assets.
KEEP_ASSETS=0
ONLY_STAGES=()
JOBS=$(nproc 2>/dev/null || echo 4)
CUDA_VER=${CUDA_VER:-$CUDA_VER_DEFAULT}
NVDS_VERSION=${NVDS_VERSION:-9.1}
INSTALL_METHOD=${INSTALL_METHOD:-deb}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --skip-deps)       SKIP_DEPS=1 ;;
    --skip-artifacts)  SKIP_ARTIFACTS=1 ;;
    --keep-assets)     KEEP_ASSETS=1 ;;
    --resume)          RESUME=1 ;;
    --verbose)         VERBOSE=1 ;;
    --install-method=*)
      INSTALL_METHOD="${1#--install-method=}"
      ;;
    --install-method)
      shift
      if [[ $# -eq 0 ]]; then
        echo "error: --install-method requires a value (deb | tar)" >&2
        exit 1
      fi
      INSTALL_METHOD="$1"
      ;;
    --only=*)
      IFS=',' read -ra _only_parts <<< "${1#--only=}"
      ONLY_STAGES+=("${_only_parts[@]}")
      ;;
    --only)
      shift
      if [[ $# -eq 0 ]]; then
        echo "error: --only requires a stage name" >&2
        exit 1
      fi
      IFS=',' read -ra _only_parts <<< "$1"
      ONLY_STAGES+=("${_only_parts[@]}")
      ;;
    -j)
      shift
      if [[ $# -eq 0 ]]; then
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
STAGE_STATE_FILE="$SCRIPT_DIR/.stage-state"
DEPS_STAGE_STATE_FILE="$SCRIPT_DIR/.stage-state.deps"
ARTIFACTS_STAGE_STATE_FILE="$SCRIPT_DIR/.stage-state.artifacts"

# build.sh lives in build/; cd to the repo root so relative paths resolve.
cd "$SCRIPT_DIR/.."

want_stage() {
  local stage=$1
  if [[ ${#ONLY_STAGES[@]} -eq 0 ]]; then
    return 0
  fi
  local s
  for s in "${ONLY_STAGES[@]}"; do
    if [[ "$s" = "$stage" ]]; then
      return 0
    fi
  done
  return 1
}

case "$INSTALL_METHOD" in
  deb|tar) ;;
  *) echo "error: invalid --install-method '$INSTALL_METHOD' (valid: deb | tar)" >&2; exit 1 ;;
esac

for s in "${ONLY_STAGES[@]}"; do
  valid=0
  for v in "${VALID_ONLY_STAGES[@]}"; do
    if [[ "$s" = "$v" ]]; then
      valid=1
      break
    fi
  done
  if [[ "$valid" -eq 0 ]]; then
    echo "error: unknown --only stage '$s' (valid: ${VALID_ONLY_STAGES[*]})" >&2
    exit 1
  fi
done

if [[ "$RESUME" -eq 0 ]]; then
  reset_stage_state
fi

begin_stage() {
  local stage=$1
  if ! want_stage "$stage"; then
    return 1
  fi
  if [[ "$RESUME" -eq 1 ]] && stage_is_done "$stage"; then
    echo "==> Stage: $stage (skipped, already complete)"
    return 1
  fi
  echo "==> Stage: $stage"
  return 0
}

finish_stage() {
  local stage=$1
  mark_stage_done "$stage"
  echo "    $stage: complete"
}

stage_had_failures() {
  local before=$1
  [[ "${#FAILED_BUILDS[@]}" -gt "$before" ]]
}

run_submake() {
  if [[ "$VERBOSE" -eq 1 ]]; then
    run_as_root make -j"$JOBS" "$@"
  else
    run_submake_quiet "$@"
  fi
}

run_submake_quiet() {
  run_as_root make -j"$JOBS" "$@" 2>/dev/null
}

run_user_make() {
  if [[ "$VERBOSE" -eq 1 ]]; then
    make -j"$JOBS" "$@"
  else
    make -j"$JOBS" "$@" 2>/dev/null
  fi
}

# Extra make flags for SBSA
PLATFORM_MAKE_FLAGS=
[[ "$PLATFORM" = "sbsa" ]] && PLATFORM_MAKE_FLAGS="AARCH64_IS_SBSA=1"

CMAKE_BIN=$(find_cmake)
SM_APP_BUILD=/tmp/ds-sm-apps-$(id -un)
SM_MOD_BUILD=/tmp/ds-sm-modules-$(id -un)

BUILD_LOG="$SCRIPT_DIR/build.log"
: >"$BUILD_LOG"
exec > >(tee "$BUILD_LOG") 2>&1

echo "==> Build log: $BUILD_LOG (overwritten each run)"
echo "==> Building PLATFORM=$PLATFORM CUDA_VER=$CUDA_VER NVDS_VERSION=$NVDS_VERSION (outputs to /opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/)"
echo "==> Using cmake: $CMAKE_BIN (jobs: $JOBS)"
if [[ ${#ONLY_STAGES[@]} -gt 0 ]]; then
  echo "==> Scoped build: ${ONLY_STAGES[*]}"
fi
if [[ "$SKIP_DEPS" -eq 1 ]]; then
  echo "==> Skipping deps stage (--skip-deps; $DEPS_STAGE_STATE_FILE is unchanged)"
fi
if [[ "$SKIP_ARTIFACTS" -eq 1 ]]; then
  if [[ "$PLATFORM" = "sbsa" ]]; then
    echo "==> Skipping artifacts stage (SBSA: bare-metal DS installation not supported; use Docker)"
  else
    echo "==> Skipping artifacts stage (--skip-artifacts; $ARTIFACTS_STAGE_STATE_FILE is unchanged)"
  fi
else
  echo "==> Artifact install method: $INSTALL_METHOD (override with --install-method=deb|tar)"
fi
if [[ "$RESUME" -eq 1 ]]; then
  echo "==> Resume enabled: skipping stages already complete in $STAGE_STATE_FILE"
fi

MK="CUDA_VER=$CUDA_VER NVDS_VERSION=$NVDS_VERSION $PLATFORM_MAKE_FLAGS"
FAILED_BUILDS=()

DS_ROOT="/opt/nvidia/deepstream/deepstream-${NVDS_VERSION}"
ARTIFACTS_DIR="$(pwd)/artifacts"

# Release assets are fetched from the DeepStream GitHub release into ARTIFACTS_DIR.
GITHUB_RELEASE_BASE="https://github.com/NVIDIA/DeepStream/releases/download/v${NVDS_VERSION}.0"
# Track what we downloaded so it can be cleaned up after a successful build.
DOWNLOADED_ASSETS=()
ARTIFACTS_DIR_CREATED=0

finalize_install() {
  echo "==> Running install.sh (NVDS_VERSION=$NVDS_VERSION)"
  run_as_root env NVDS_VERSION="$NVDS_VERSION" bash "$SCRIPT_DIR/../scripts/install.sh"
}

# Download a single release asset into ARTIFACTS_DIR (skip if already present).
download_asset() {
  local name=$1
  local url="$GITHUB_RELEASE_BASE/$name"
  local dest="$ARTIFACTS_DIR/$name"

  if [[ -s "$dest" ]]; then
    echo "    already present, skipping download: $name"
    DOWNLOADED_ASSETS+=("$dest")
    return 0
  fi
  if ! command -v wget >/dev/null 2>&1; then
    echo "error: wget is required to download release assets but was not found." >&2
    exit 1
  fi
  echo "    downloading $url"
  if ! wget -q --show-progress -O "$dest" "$url"; then
    rm -f "$dest"
    echo "error: failed to download $url" >&2
    exit 1
  fi
  DOWNLOADED_ASSETS+=("$dest")
}

# Fetch the release assets needed for the selected install method and platform.
# deb (default) pulls only the .deb packages; tar pulls only the tarballs.
# Sample data is common to x86 and aarch64; the binaries asset is platform-specific.
download_release_assets() {
  local sample_asset binaries_asset
  if [[ "$INSTALL_METHOD" == "deb" ]]; then
    sample_asset="deepstream-sample-data_${NVDS_VERSION}.0.deb"
    case "$PLATFORM" in
      x86)     binaries_asset="deepstream-binaries-x86_${NVDS_VERSION}.0_amd64.deb" ;;
      aarch64) binaries_asset="deepstream-binaries-aarch64_${NVDS_VERSION}.0_arm64.deb" ;;
      *) echo "error: unsupported platform for asset download: $PLATFORM" >&2; exit 1 ;;
    esac
  else
    sample_asset="deepstream-sample-data_${NVDS_VERSION}.0.tar.gz"
    case "$PLATFORM" in
      x86)     binaries_asset="deepstream-binaries-x86_${NVDS_VERSION}.0.tar.gz" ;;
      aarch64) binaries_asset="deepstream-binaries-aarch64_${NVDS_VERSION}.0.tar.gz" ;;
      *) echo "error: unsupported platform for asset download: $PLATFORM" >&2; exit 1 ;;
    esac
  fi

  echo "==> Downloading release assets (method=$INSTALL_METHOD) into $ARTIFACTS_DIR"
  if [[ ! -d "$ARTIFACTS_DIR" ]]; then
    mkdir -p "$ARTIFACTS_DIR"
  fi
  ARTIFACTS_DIR_CREATED=1
  download_asset "$sample_asset"
  download_asset "$binaries_asset"
}

# Remove the assets downloaded this run once the whole build has succeeded,
# unless the user asked to keep them via --keep-assets.
cleanup_downloaded_assets() {
  if [[ "$KEEP_ASSETS" -eq 1 ]]; then
    if [[ ${#DOWNLOADED_ASSETS[@]} -gt 0 ]]; then
      echo "==> Keeping downloaded release assets in $ARTIFACTS_DIR (--keep-assets)"
    fi
    return 0
  fi
  [[ ${#DOWNLOADED_ASSETS[@]} -eq 0 ]] && return 0

  echo "==> Removing downloaded release assets (pass --keep-assets to retain them)"
  local f
  for f in "${DOWNLOADED_ASSETS[@]}"; do
    rm -f "$f" && echo "    removed $(basename "$f")"
  done
  # If we created the artifacts directory this run and it is now empty, remove it.
  if [[ "$ARTIFACTS_DIR_CREATED" -eq 1 ]] && [[ -d "$ARTIFACTS_DIR" ]] && \
     [[ -z "$(ls -A "$ARTIFACTS_DIR" 2>/dev/null)" ]]; then
    rmdir "$ARTIFACTS_DIR" && echo "    removed empty $ARTIFACTS_DIR"
  fi
}

# ---------------------------------------------------------------------------
# Stage: artifacts (proprietary libs + sample payloads)
# ---------------------------------------------------------------------------
if [[ "$SKIP_ARTIFACTS" -eq 0 ]] && begin_stage "$ARTIFACTS_STAGE"; then
  if [[ "$RESUME" -eq 0 ]]; then
    mkdir -p "$(dirname "$ARTIFACTS_STAGE_STATE_FILE")"
    : >"$ARTIFACTS_STAGE_STATE_FILE"
  fi
  download_release_assets
  run_as_root env NVDS_VERSION="$NVDS_VERSION" PLATFORM="$PLATFORM" \
    INSTALL_METHOD="$INSTALL_METHOD" \
    ARTIFACTS_DIR="$ARTIFACTS_DIR" \
    ARTIFACTS_STAGE_STATE_FILE="$ARTIFACTS_STAGE_STATE_FILE" \
    bash "$SCRIPT_DIR/../scripts/install_artifacts.sh"
  finish_stage "$ARTIFACTS_STAGE"
fi

# ---------------------------------------------------------------------------
# Stage: deps
# ---------------------------------------------------------------------------
if [[ "$SKIP_DEPS" -eq 0 ]] && begin_stage "$DEPS_STAGE"; then
  if [[ "$RESUME" -eq 0 ]]; then
    mkdir -p "$(dirname "$DEPS_STAGE_STATE_FILE")"
    : >"$DEPS_STAGE_STATE_FILE"
  fi
  run_as_root env NVDS_VERSION="$NVDS_VERSION" PLATFORM="$PLATFORM" \
    DEPS_STAGE_STATE_FILE="$DEPS_STAGE_STATE_FILE" \
    bash "$SCRIPT_DIR/../scripts/install_opensource_deps.sh"
  finish_stage "$DEPS_STAGE"
fi

# ---------------------------------------------------------------------------
# Stage: gst-utils
# ---------------------------------------------------------------------------
if begin_stage gst-utils; then
  _fb_before=${#FAILED_BUILDS[@]}
  for dir in \
    src/gst-utils/gstnvcustomhelper \
    src/gst-utils/gst-nvdssr \
    src/gst-utils/gstnvdscustomhelper\
    src/utils/nvds_rest_server; do
    if run_submake -C "$dir" $MK; then
      run_submake -C "$dir" $MK install || FAILED_BUILDS+=("$dir (install)")
    else
      FAILED_BUILDS+=("$dir")
    fi
  done
  if ! stage_had_failures "$_fb_before"; then
    finish_stage gst-utils
  fi
fi

# ---------------------------------------------------------------------------
# Stage: utils
# ---------------------------------------------------------------------------
if begin_stage utils; then
  _fb_before=${#FAILED_BUILDS[@]}
  for dir in src/utils/*/; do
    [[ -f "$dir/Makefile" ]] || continue
    # nvstreammux has no install target; skip install step for it
    if run_submake -C "$dir" $MK; then
      [[ "$(basename "$dir")" != "nvstreammux" ]] && \
        { run_submake -C "$dir" $MK install || FAILED_BUILDS+=("$dir (install)"); }
    else
      FAILED_BUILDS+=("$dir")
    fi
  done

  for dir in src/utils/nvds_msgapi/*/; do
    if [[ -f "$dir/Makefile" ]]; then
      if run_submake -C "$dir" $MK; then
        run_submake -C "$dir" $MK install || FAILED_BUILDS+=("$dir (install)")
      else
        FAILED_BUILDS+=("$dir")
      fi
    else
      for sub in "$dir"*/; do
        if [[ -f "$sub/Makefile" ]]; then
          if run_submake -C "$sub" $MK; then
            run_submake -C "$sub" $MK install || FAILED_BUILDS+=("$sub (install)")
          else
            FAILED_BUILDS+=("$sub")
          fi
        fi
      done
    fi
  done

  for dir in src/utils/ds3d/*/; do
    if [[ -f "$dir/Makefile" ]]; then
      if run_submake -C "$dir" $MK; then
        run_submake -C "$dir" $MK install || FAILED_BUILDS+=("$dir (install)")
      else
        FAILED_BUILDS+=("$dir")
      fi
    else
      for sub in "$dir"*/; do
        if [[ -f "$sub/Makefile" ]]; then
          if run_submake -C "$sub" $MK; then
            run_submake -C "$sub" $MK install || FAILED_BUILDS+=("$sub (install)")
          else
            FAILED_BUILDS+=("$sub")
          fi
        fi
      done
    fi
  done
  if ! stage_had_failures "$_fb_before"; then
    finish_stage utils
  fi
fi

# ---------------------------------------------------------------------------
# Stage: gst-plugins
# ---------------------------------------------------------------------------
if begin_stage gst-plugins; then
  _fb_before=${#FAILED_BUILDS[@]}
  run_as_root mkdir -p "/opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/lib/gst-plugins/"
  for dir in src/gst-plugins/*/; do
    plugin=$(basename "$dir")
    if [[ "$plugin" == "gst-nvdsudp" || "$plugin" == "gst-dsexample-cuda" ]]; then
      echo "Skipping $plugin (see its README for build steps)"
      continue
    fi
    if run_submake -C "$dir" $MK; then
      run_submake -C "$dir" $MK install || FAILED_BUILDS+=("$dir (install)")
    else
      FAILED_BUILDS+=("$dir")
    fi
  done
  # Several plugins ship custom implementation/serialization libs in
  # subdirectories that their parent Makefile does not descend into
  # (the parent $(DEP) recursion rule is a no-op because $(DEP) is empty).
  # Build and install each one explicitly.
  for sublib_dir in \
    src/gst-plugins/gst-nvdspreprocess/nvdspreprocess_lib \
    src/gst-plugins/gst-nvdsmetautils/sei_serialization \
    src/gst-plugins/gst-nvdsmetautils/audio_metadata_serialization \
    src/gst-plugins/gst-nvdsmetautils/video_metadata_serialization \
    src/gst-plugins/gst-nvdspostprocess/postprocesslib_impl \
    src/gst-plugins/gst-nvdsvideotemplate/customlib_impl; do
    [[ -f "$sublib_dir/Makefile" ]] || continue
    if run_submake -C "$sublib_dir" $MK; then
      run_submake -C "$sublib_dir" $MK install || FAILED_BUILDS+=("$sublib_dir (install)")
    else
      FAILED_BUILDS+=("$sublib_dir")
    fi
  done
  if ! stage_had_failures "$_fb_before"; then
    finish_stage gst-plugins
  fi
fi

# ---------------------------------------------------------------------------
# Stage: sample_apps
# ---------------------------------------------------------------------------
if begin_stage sample_apps; then
  _fb_before=${#FAILED_BUILDS[@]}
  run_as_root mkdir -p "/opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/bin/"
  for dir in src/apps/sample_apps/*/; do
    app=$(basename "$dir")
    if [[ "$PLATFORM" != "x86" ]] && [[ "$app" == "deepstream-ucx-test" || "$app" == "deepstream-multigpu-nvlink-test" ]]; then
      echo "Skipping $app on $PLATFORM (x86 only)"
      continue
    fi
    if [[ "$PLATFORM" != "aarch64" ]] && [[ "$app" = "deepstream-ipc-test" ]]; then
      echo "Skipping $app on $PLATFORM (Jetson/aarch64 only)"
      continue
    fi
    if [[ "$PLATFORM" != "x86" ]] && [[ "$PLATFORM" != "sbsa" ]] && [[ "$app" = "deepstream-appsrc-cuda-test" ]]; then
      echo "Skipping $app on $PLATFORM (x86 and sbsa only)"
      continue
    fi
    if grep -q "^install:" "$dir/Makefile" 2>/dev/null; then
      if run_submake -C "$dir" $MK; then
        run_submake -C "$dir" $MK install || FAILED_BUILDS+=("$dir (install)")
      else
        FAILED_BUILDS+=("$dir")
      fi
    else
      # Triton* dirs build helper libs only; no install target
      run_user_make -C "$dir" $MK || FAILED_BUILDS+=("$dir")
    fi
  done
  if ! stage_had_failures "$_fb_before"; then
    finish_stage sample_apps
  fi
fi

# ---------------------------------------------------------------------------
# Stage: yolo (apt / eigen / Yolo custom lib)
# ---------------------------------------------------------------------------
if begin_stage yolo; then
  _fb_before=${#FAILED_BUILDS[@]}
  last_update=$(stat -c %Y /var/cache/apt/pkgcache.bin 2>/dev/null || echo 0)
  now=$(date +%s)
  if [[ $((now - last_update)) -gt 86400 ]]; then
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
  if ! stage_had_failures "$_fb_before"; then
    finish_stage yolo
  fi
fi

# ---------------------------------------------------------------------------
# Stage: tao_apps
# ---------------------------------------------------------------------------
if begin_stage tao_apps; then
  _fb_before=${#FAILED_BUILDS[@]}
  if git submodule status 2>/dev/null | grep -q '^-'; then
    git submodule update --init --recursive || true
  else
    echo "==> git submodules already initialized"
  fi
  run_submake -C src/apps/tao_apps $MK || FAILED_BUILDS+=("src/apps/tao_apps")
  if ! stage_had_failures "$_fb_before"; then
    finish_stage tao_apps
  fi
fi

# ---------------------------------------------------------------------------
# Stage: reference_apps
# ---------------------------------------------------------------------------
if begin_stage reference_apps; then
  _fb_before=${#FAILED_BUILDS[@]}
  RA_BIN_STAGE=/tmp/ds-reference-apps-bin
  DS_BIN=/opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/bin
  mkdir -p "$RA_BIN_STAGE"
  # Directories that ship only configs/assets (no Makefile to build).
  REF_APPS_SKIP=(
    "deepstream-masktracker"
    "deepstream-tracker-3d-multi-view"
    "deepstream-tracker-3d"
    "deepstream-vllm-plugin"
    "pyservicemaker_sample_apps"
  )
  for dir in src/apps/reference_apps/*/; do
    app=$(basename "$dir")
    skip=0
    for s in "${REF_APPS_SKIP[@]}"; do
      if [[ "$app" = "$s" ]]; then skip=1; break; fi
    done
    if [[ "$skip" -eq 1 ]]; then
      echo "Skipping $app (config-only, no build required)"
      continue
    fi
    if [[ ! -f "$dir/Makefile" ]]; then
      echo "Skipping $app (no Makefile; see its README for build steps)"
      continue
    fi
    if grep -q "^install:" "$dir/Makefile" 2>/dev/null; then
      if run_submake -C "$dir" $MK BIN_DIR="$RA_BIN_STAGE"; then
        for bin in "$RA_BIN_STAGE"/deepstream-*; do
          [[ -f "$bin" ]] && [[ -x "$bin" ]] && run_as_root cp -v "$bin" "$DS_BIN/" || true
        done
      else
        FAILED_BUILDS+=("$dir")
      fi
    else
      run_user_make -C "$dir" $MK BIN_DIR="$RA_BIN_STAGE" || FAILED_BUILDS+=("$dir")
    fi
    rm -f "$RA_BIN_STAGE"/deepstream-*
  done
  rm -rf "$RA_BIN_STAGE"
  if ! stage_had_failures "$_fb_before"; then
    finish_stage reference_apps
  fi
fi

# ---------------------------------------------------------------------------
# Stage: service-maker
# ---------------------------------------------------------------------------
if begin_stage service-maker; then
  _fb_before=${#FAILED_BUILDS[@]}
  SM_BIN_INSTALL=/opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/bin
  for dir in src/service-maker/sources/apps/cpp/*/; do
    app=$(basename "$dir")
    # Drop a stale build dir whose cache was generated from a different source
    # path (e.g. the repo moved/renamed); otherwise cmake refuses to reconfigure.
    cache="$SM_APP_BUILD/$app/CMakeCache.txt"
    if [[ -f "$cache" ]] && ! grep -qxF "CMAKE_HOME_DIRECTORY:INTERNAL=$(cd "$dir" && pwd)" "$cache"; then
      rm -rf "$SM_APP_BUILD/$app"
    fi
    if ! "$CMAKE_BIN" -S "$dir" -B "$SM_APP_BUILD/$app" -DCMAKE_BUILD_TYPE=Release; then
      FAILED_BUILDS+=("$dir (cmake configure)")
      continue
    fi
    if ! "$CMAKE_BIN" --build "$SM_APP_BUILD/$app" -j"$JOBS"; then
      FAILED_BUILDS+=("$dir (cmake build)")
      continue
    fi
    for bin in "$SM_APP_BUILD/$app"/deepstream-*; do
      [[ -f "$bin" ]] && [[ -x "$bin" ]] || continue
      name=$(basename "$bin" | sed 's/^deepstream-/service-maker-/')
      run_as_root cp -v "$bin" "$SM_BIN_INSTALL/$name" \
        || FAILED_BUILDS+=("$dir (install $name)")
    done
  done

  SM_MODULES_INSTALL=/opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/service-maker/modules
  run_as_root mkdir -p "$SM_MODULES_INSTALL"
  for dir in src/service-maker/sources/modules/*/; do
    mod=$(basename "$dir")
    # Drop a stale build dir whose cache was generated from a different source
    # path (e.g. the repo moved/renamed); otherwise cmake refuses to reconfigure.
    cache="$SM_MOD_BUILD/$mod/CMakeCache.txt"
    if [[ -f "$cache" ]] && ! grep -qxF "CMAKE_HOME_DIRECTORY:INTERNAL=$(cd "$dir" && pwd)" "$cache"; then
      rm -rf "$SM_MOD_BUILD/$mod"
    fi
    if ! "$CMAKE_BIN" -S "$dir" -B "$SM_MOD_BUILD/$mod" -DCMAKE_BUILD_TYPE=Release; then
      FAILED_BUILDS+=("$dir (cmake configure)")
      continue
    fi
    if ! "$CMAKE_BIN" --build "$SM_MOD_BUILD/$mod" -j"$JOBS"; then
      FAILED_BUILDS+=("$dir (cmake build)")
      continue
    fi
    run_as_root cp -v "$SM_MOD_BUILD/$mod/lib${mod}.so" "$SM_MODULES_INSTALL/" \
      || FAILED_BUILDS+=("$dir (install lib${mod}.so)")
  done
  if ! stage_had_failures "$_fb_before"; then
    finish_stage service-maker
  fi
fi

echo ""
echo "==> Build log: $BUILD_LOG"

if [[ ${#FAILED_BUILDS[@]} -gt 0 ]]; then
  echo "==> Build completed with ${#FAILED_BUILDS[@]} failure(s):"
  for fb in "${FAILED_BUILDS[@]}"; do
    echo "    FAILED: $fb"
  done
  exit 1
fi

if [[ ${#ONLY_STAGES[@]} -gt 0 ]]; then
  echo "==> Scoped build (--only): skipping install.sh (run a full build to finalize system integration)"
  echo "==> Done (scoped build succeeded: ${ONLY_STAGES[*]})"
  exit 0
fi

finalize_install

cleanup_downloaded_assets

echo "==> Done (all builds succeeded; installed to $DS_ROOT)"
exit 0
