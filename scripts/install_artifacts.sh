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

# Install DeepStream artifacts (proprietary libs + sample payloads).
# Proprietary libs and sample data are selected by platform (x86 / aarch64 / sbsa).
#
# Two install methods are supported, selected via INSTALL_METHOD (default: deb):
#   deb  Install the prebuilt Debian packages with dpkg (default):
#          deepstream-binaries-<platform>_*.deb    -> <INSTALL_ROOT>/lib, samples, ...
#          deepstream-sample-data_*.deb            -> <INSTALL_ROOT>/samples/
#   tar  Extract the tarballs into the opt install tree:
#          deepstream-binaries-<platform>_<ver>.tar.gz -> <INSTALL_ROOT>/
#          deepstream-sample-data_<ver>.tar.gz     -> <INSTALL_ROOT>/samples/
#
# Sample configs ship inside both the sample-data deb and tarball (under
# samples/configs), so they no longer need a symlink to <repo>/configs.
#
# Usage:
#   sudo bash scripts/install_artifacts.sh
#   sudo INSTALL_METHOD=tar bash scripts/install_artifacts.sh
#   sudo NVDS_VERSION=9.1 bash scripts/install_artifacts.sh
#   sudo PLATFORM=x86 bash scripts/install_artifacts.sh
#
# Environment:
#   INSTALL_METHOD            deb (default) | tar
#   NVDS_VERSION              DeepStream version (default: 9.1)
#   PLATFORM                  x86 | aarch64 | sbsa (auto-detected if unset)
#   ARTIFACTS_DIR             Local artifacts directory (default: <repo>/artifacts)
#   ARTIFACTS_STAGE_STATE_FILE Stage resume file (default: build/.stage-state.artifacts)
#   PROPRIETARY_LIBS_URL      Override proprietary tarball URL (tar method download fallback)
#   ARTIFACTS_LOCAL_ONLY=1    Require local artifacts; do not download proprietary libs

set -e

INSTALL_METHOD=${INSTALL_METHOD:-deb}
NVDS_VERSION=${NVDS_VERSION:-9.1}
INSTALL_ROOT=/opt/nvidia/deepstream/deepstream-${NVDS_VERSION}
DEEPSTREAM_BASE=/opt/nvidia/deepstream
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
ARTIFACTS_DIR="${ARTIFACTS_DIR:-$REPO_ROOT/artifacts}"
STAGE_STATE_FILE="${ARTIFACTS_STAGE_STATE_FILE:-$REPO_ROOT/build/.stage-state.artifacts}"
ARTIFACTS_LOCAL_ONLY=${ARTIFACTS_LOCAL_ONLY:-0}

case "$INSTALL_METHOD" in
  deb|tar) ;;
  *) echo "error: invalid INSTALL_METHOD='$INSTALL_METHOD' (expected: deb | tar)" >&2; exit 1 ;;
esac

if [[ "${PLATFORM:-}" = "sbsa" ]] || { [[ -z "${PLATFORM:-}" ]] && [[ "$(uname -m)" = "aarch64" ]] && [[ ! -f /etc/nv_tegra_release ]]; }; then
  echo "error: DeepStream bare-metal installation is not supported on SBSA / DGX Spark." >&2
  echo "       Run build/build.sh inside the NVIDIA SBSA Docker container instead." >&2
  exit 1
fi

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
    *) echo "Unsupported architecture: $ARCH" >&2; exit 1 ;;
  esac
fi

# Tarball names mirror the .deb names (minus the arch/_all suffix).
PROPRIETARY_TAR_NAME="deepstream-binaries-${PLATFORM}_${NVDS_VERSION}.0.tar.gz"
GITHUB_RELEASE_URL="https://github.com/NVIDIA/DeepStream/releases/download/v${NVDS_VERSION}.0/${PROPRIETARY_TAR_NAME}"

# Debian package name globs (version/arch are baked in at build time).
PROPRIETARY_DEB_GLOB="deepstream-binaries-${PLATFORM}_*.deb"
SAMPLE_TAR_NAME="deepstream-sample-data_${NVDS_VERSION}.0.tar.gz"
SAMPLE_DEB_GLOB="deepstream-sample-data_*.deb"

# Exact deb/tar names and GitHub Release download URLs for each artifact.
case "$PLATFORM" in
  x86)     DEB_ARCH=amd64 ;;
  aarch64) DEB_ARCH=arm64 ;;
  *)       DEB_ARCH=unknown ;;
esac
PROPRIETARY_DEB_NAME="deepstream-binaries-${PLATFORM}_${NVDS_VERSION}.0_${DEB_ARCH}.deb"
SAMPLE_DEB_NAME="deepstream-sample-data_${NVDS_VERSION}.0.deb"
GITHUB_RELEASE_BASE="https://github.com/NVIDIA/DeepStream/releases/download/v${NVDS_VERSION}.0"
GITHUB_RELEASE_DEB_URL="${GITHUB_RELEASE_BASE}/${PROPRIETARY_DEB_NAME}"
GITHUB_RELEASE_SAMPLE_DEB_URL="${GITHUB_RELEASE_BASE}/${SAMPLE_DEB_NAME}"
GITHUB_RELEASE_SAMPLE_TAR_URL="${GITHUB_RELEASE_BASE}/${SAMPLE_TAR_NAME}"

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

resolve_proprietary_deb() {
  local deb_path
  if deb_path=$(resolve_artifact_deb "$PROPRIETARY_DEB_GLOB"); then
    printf '%s\n' "$deb_path"
    return 0
  fi

  if [[ "$ARTIFACTS_LOCAL_ONLY" -eq 1 ]]; then
    echo "error: missing local proprietary deb: $ARTIFACTS_DIR/$PROPRIETARY_DEB_GLOB" >&2
    echo "       (ARTIFACTS_LOCAL_ONLY=1; download disabled)" >&2
    exit 1
  fi

  local deb_url="${PROPRIETARY_LIBS_DEB_URL:-$GITHUB_RELEASE_DEB_URL}"
  local tmp_deb
  tmp_deb=$(mktemp /tmp/ds-proprietary-XXXXXX.deb)
  echo "    downloading $deb_url"
  wget -q --show-progress -O "$tmp_deb" "$deb_url"
  printf '%s\n' "$tmp_deb"
}

resolve_sample() {
  if [[ "$INSTALL_METHOD" == "tar" ]]; then
    local local_tar="$ARTIFACTS_DIR/$SAMPLE_TAR_NAME"
    if [[ -f "$local_tar" ]]; then
      printf '%s\n' "$local_tar"; return 0
    fi
    if [[ "$ARTIFACTS_LOCAL_ONLY" -eq 1 ]]; then
      echo "error: missing local sample archive: $local_tar" >&2
      echo "       (ARTIFACTS_LOCAL_ONLY=1; download disabled)" >&2
      exit 1
    fi
    local tmp_tar
    tmp_tar=$(mktemp /tmp/ds-sample-XXXXXX.tar.gz)
    echo "    downloading $GITHUB_RELEASE_SAMPLE_TAR_URL"
    wget -q --show-progress -O "$tmp_tar" "$GITHUB_RELEASE_SAMPLE_TAR_URL"
    printf '%s\n' "$tmp_tar"
  else
    local deb_path
    if deb_path=$(resolve_artifact_deb "$SAMPLE_DEB_GLOB"); then
      printf '%s\n' "$deb_path"; return 0
    fi
    if [[ "$ARTIFACTS_LOCAL_ONLY" -eq 1 ]]; then
      echo "error: missing local sample deb: $ARTIFACTS_DIR/$SAMPLE_DEB_GLOB" >&2
      echo "       (ARTIFACTS_LOCAL_ONLY=1; download disabled)" >&2
      exit 1
    fi
    local tmp_deb
    tmp_deb=$(mktemp /tmp/ds-sample-XXXXXX.deb)
    echo "    downloading $GITHUB_RELEASE_SAMPLE_DEB_URL"
    wget -q --show-progress -O "$tmp_deb" "$GITHUB_RELEASE_SAMPLE_DEB_URL"
    printf '%s\n' "$tmp_deb"
  fi
}

extract_tar_to() {
  local tar_path=$1
  local dest_dir=$2
  echo "    extracting $(basename "$tar_path") -> $dest_dir/"
  mkdir -p "$dest_dir"
  tar -xzf "$tar_path" -C "$dest_dir"
}

install_deb() {
  local deb_path=$1
  echo "    installing $(basename "$deb_path") via dpkg"
  dpkg -i "$deb_path"
}

# Print the first artifact in ARTIFACTS_DIR matching a glob, or fail.
resolve_artifact_deb() {
  local glob=$1
  local matches=()
  shopt -s nullglob
  matches=( "$ARTIFACTS_DIR"/$glob )
  shopt -u nullglob
  if [[ ${#matches[@]} -eq 0 ]]; then
    return 1
  fi
  printf '%s\n' "${matches[0]}"
}

resolve_proprietary_tar() {
  local local_tar="$ARTIFACTS_DIR/$PROPRIETARY_TAR_NAME"

  if [[ -f "$local_tar" ]]; then
    printf '%s\n' "$local_tar"
    return 0
  fi

  if [[ "$ARTIFACTS_LOCAL_ONLY" -eq 1 ]]; then
    echo "error: missing local proprietary archive: $local_tar" >&2
    echo "       (ARTIFACTS_LOCAL_ONLY=1; download disabled)" >&2
    exit 1
  fi

  local tar_url="${PROPRIETARY_LIBS_URL:-$GITHUB_RELEASE_URL}"
  local tmp_tar
  tmp_tar=$(mktemp /tmp/ds-proprietary-XXXXXX.tar.gz)
  echo "    downloading $tar_url"
  wget -q --show-progress -O "$tmp_tar" "$tar_url"
  printf '%s\n' "$tmp_tar"
}

install_setup_stage() {
  local stage=artifacts-setup

  if stage_is_done "$stage"; then
    echo "--- Install root setup (skipped, already DONE) ---"
    return 0
  fi

  echo ""
  echo "--- Install root setup ---"
  mkdir -p "$INSTALL_ROOT"/lib/gst-plugins "$INSTALL_ROOT"/samples "$INSTALL_ROOT"/bin
  mkdir -p "$INSTALL_ROOT"/sources
  mkdir -p "$DEEPSTREAM_BASE"
  ln -sfn "deepstream-${NVDS_VERSION}" "$DEEPSTREAM_BASE/deepstream"
  ln -sfn "$REPO_ROOT/includes" "$INSTALL_ROOT/sources/includes"
  mkdir -p "$INSTALL_ROOT/sources/apps"
  ln -sfn "$REPO_ROOT/src/apps/sample_apps" "$INSTALL_ROOT/sources/apps/sample_apps"
  mkdir -p "$INSTALL_ROOT/sources/libs"
  ln -sfn "$REPO_ROOT/src/utils/nvds_msgapi/kafka_protocol_adaptor" "$INSTALL_ROOT/sources/libs/kafka_protocol_adaptor"
  ln -sfn "$REPO_ROOT/src/utils/nvds_msgapi/common_src" "$INSTALL_ROOT/sources/libs/nvds_msgapi_common_src"
  ln -sfn "$REPO_ROOT/src/utils/nvds_msgapi/amqp_protocol_adaptor" "$INSTALL_ROOT/sources/libs/amqp_protocol_adaptor"
  mkdir -p "$INSTALL_ROOT/sources/libs/azure_protocol_adaptor"
  ln -sfn "$REPO_ROOT/src/utils/nvds_msgapi/azure_protocol_adaptor/device_client" "$INSTALL_ROOT/sources/libs/azure_protocol_adaptor/device_client"
  ln -sfn "$REPO_ROOT/src/utils/nvds_msgapi/azure_protocol_adaptor/module_client" "$INSTALL_ROOT/sources/libs/azure_protocol_adaptor/module_client"
  ln -sfn "$REPO_ROOT/src/utils/nvds_msgapi/mqtt_protocol_adaptor" "$INSTALL_ROOT/sources/libs/mqtt_protocol_adaptor"
  ln -sfn "$REPO_ROOT/src/utils/nvds_msgapi/redis_protocol_adaptor" "$INSTALL_ROOT/sources/libs/redis_protocol_adaptor"
  if [[ "$PLATFORM" = "aarch64" ]]; then
    local base_lib_dir="/usr/lib/aarch64-linux-gnu"
    local nvidia_lib_dir
    if [[ -d "$base_lib_dir/nvidia" ]]; then
      nvidia_lib_dir="$base_lib_dir/nvidia"
    else
      nvidia_lib_dir="$base_lib_dir/tegra"
    fi
    ln -sf "$nvidia_lib_dir/libnvbufsurface.so"       "$INSTALL_ROOT/lib/libnvbufsurface.so"
    ln -sf "$nvidia_lib_dir/libnvbufsurftransform.so" "$INSTALL_ROOT/lib/libnvbufsurftransform.so"
    ln -sf "$nvidia_lib_dir/libnvdsbufferpool.so"     "$INSTALL_ROOT/lib/libnvdsbufferpool.so"
    ln -sf "$nvidia_lib_dir/libgstnvcustomhelper.so"  "$INSTALL_ROOT/lib/libgstnvcustomhelper.so"
  fi
  mark_stage_done "$stage"
  echo "    Stage $stage: DONE"
}

install_proprietary_stage() {
  local stage="artifacts-proprietary-${PLATFORM}"
  local tar_path tmp_to_remove=0 deb_path

  if stage_is_done "$stage"; then
    echo "--- Proprietary libs ($PLATFORM) (skipped, already DONE) ---"
    return 0
  fi

  echo ""
  echo "--- Proprietary libs ($PLATFORM, method=$INSTALL_METHOD) ---"

  if [[ "$INSTALL_METHOD" == "deb" ]]; then
    local tmp_deb_to_remove=0
    deb_path=$(resolve_proprietary_deb)
    if [[ "$deb_path" == /tmp/ds-proprietary-*.deb ]]; then
      tmp_deb_to_remove=1
    fi
    install_deb "$deb_path"
    if [[ "$tmp_deb_to_remove" -eq 1 ]]; then rm -f "$deb_path"; fi
  else
    tar_path=$(resolve_proprietary_tar)
    if [[ "$tar_path" == /tmp/ds-proprietary-* ]]; then
      tmp_to_remove=1
    fi
    extract_tar_to "$tar_path" "$INSTALL_ROOT"
    if [[ "$tmp_to_remove" -eq 1 ]]; then
      rm -f "$tar_path"
    fi
  fi

  mark_stage_done "$stage"
  echo "    Stage $stage: DONE"
}

install_sample_stage() {
  local stage="artifacts-samples"

  if stage_is_done "$stage"; then
    echo "--- sample-data (skipped, already DONE) ---"
    return 0
  fi

  echo ""
  echo "--- sample-data (method=$INSTALL_METHOD) ---"

  local sample_path tmp_sample_to_remove=0
  sample_path=$(resolve_sample)
  if [[ "$sample_path" == /tmp/ds-sample-* ]]; then
    tmp_sample_to_remove=1
  fi
  if [[ "$INSTALL_METHOD" == "deb" ]]; then
    install_deb "$sample_path"
  else
    extract_tar_to "$sample_path" "$INSTALL_ROOT/samples"
  fi
  if [[ "$tmp_sample_to_remove" -eq 1 ]]; then rm -f "$sample_path"; fi

  mark_stage_done "$stage"
  echo "    Stage $stage: DONE"
}

install_samples_post_stage() {
  local stage=artifacts-samples-post

  if stage_is_done "$stage"; then
    echo "--- Sample post-install (skipped, already DONE) ---"
    return 0
  fi

  echo ""
  echo "--- Sample post-install ---"
  if [[ ! -e "$INSTALL_ROOT/samples/trtis_model_repo" ]]; then
    ln -sfn triton_model_repo "$INSTALL_ROOT/samples/trtis_model_repo"
  fi
  mark_stage_done "$stage"
  echo "    Stage $stage: DONE"
}

echo "==> Installing DeepStream artifacts"
echo "    Method      : $INSTALL_METHOD"
echo "    Platform    : $PLATFORM"
echo "    Install root: $INSTALL_ROOT"
echo "    Artifacts   : $ARTIFACTS_DIR"
echo "    Stage state : $STAGE_STATE_FILE"

install_setup_stage
install_proprietary_stage

install_sample_stage

install_samples_post_stage

echo ""
echo "==> Running ldconfig"
ldconfig

echo ""
echo "==> Done. Artifacts installed to $INSTALL_ROOT"
