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

# ---------------------------------------------------------------------------
# Stage the DeepStream install tree and emit a Debian package and/or a tarball.
#
# The package installs to:
#   /opt/nvidia/deepstream/deepstream-<NVDS_VERSION>   (referred to as DS_ROOT)
#
# It is assembled from two places:
#   1. The already-built install tree at $DS_INSTALL_ROOT (bin/, lib/, samples/,
#      service-maker/, LicenseAgreement.pdf) produced by build/build.sh.
#   2. The repository sources (includes/, src/, scripts/) copied into DS_ROOT so
#      the package ships source alongside the runtime.
#
# Usage (run from anywhere):
#   bash build/package.sh [OPTIONS]
#
# Options:
#   -h, --help              Show this help and exit
#   --format=FMT            What to build: deb | tar | both  (default: both)
#   --arch=ARCH            Target debian arch: amd64 | arm64 (default: host)
#   --output-dir=PATH       Where to write the .deb / .tbz2 (default: build/)
#   --stage-dir=PATH        Scratch staging directory (default: build/.pkg-stage)
#   --keep-stage            Do not delete the staging dir on exit
#
# Environment overrides (CLI flags take precedence):
#   NVDS_VERSION   MAJOR.MINOR.PATCH (default: 9.1.1)
#   PKG_VERSION    Package version recorded in control  (default: NVDS_VERSION)
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

NVDS_VERSION=${NVDS_VERSION:-9.1.1}

# NVDS_VERSION must be MAJOR.MINOR.PATCH (default: 9.1.1). MAJOR.MINOR alone is
# rejected. The install tree and deb package name still stay at MAJOR.MINOR for
# now; PKG_VERSION and output filenames use the full version.
if [[ ! "$NVDS_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "error: invalid NVDS_VERSION='$NVDS_VERSION' (expected MAJOR.MINOR.PATCH, e.g. 9.1.1)" >&2
  exit 1
fi
NVDS_FULL_VERSION="$NVDS_VERSION"
NVDS_VERSION="${NVDS_FULL_VERSION%.*}"

PKG_VERSION=${PKG_VERSION:-$NVDS_FULL_VERSION}
PKG_REVISION=1
# Current build timestamp (UTC), e.g. "Mon Jun 29 08:00:00 UTC 2026".
BUILD_DATE="$(date -u '+%a %b %e %H:%M:%S UTC %Y')"

FORMAT=both
DEB_ARCH=""
OUTPUT_DIR="$SCRIPT_DIR"
STAGE_DIR="$SCRIPT_DIR/.pkg-stage"
KEEP_STAGE=0

usage() {
  cat <<'EOF'
Stage the DeepStream install tree and emit a Debian package and/or a tarball.

The package installs to:
  /opt/nvidia/deepstream/deepstream-<NVDS_VERSION>   (referred to as DS_ROOT)

Assembled from the built install tree ($DS_INSTALL_ROOT: bin/, lib/, samples/,
service-maker/, LicenseAgreement.pdf) plus repo sources (includes/, src/, scripts/).

Usage:
  bash build/package.sh [OPTIONS]

Options:
  -h, --help          Show this help and exit
  --format=FMT        What to build: deb | tar | both  (default: both)
  --arch=ARCH         Target debian arch: amd64 | arm64 (default: host)
  --output-dir=PATH   Where to write the .deb / .tbz2 (default: build/)
  --stage-dir=PATH    Scratch staging directory (default: build/.pkg-stage)
  --keep-stage        Do not delete the staging dir on exit

Environment overrides (CLI flags take precedence):
  NVDS_VERSION   MAJOR.MINOR.PATCH (default: 9.1.1)
  PKG_VERSION    Package version recorded in control   (default: NVDS_VERSION)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)        usage; exit 0 ;;
    --format=*)       FORMAT="${1#*=}" ;;
    --arch=*)         DEB_ARCH="${1#*=}" ;;
    --output-dir=*)   OUTPUT_DIR="${1#*=}" ;;
    --stage-dir=*)    STAGE_DIR="${1#*=}" ;;
    --keep-stage)     KEEP_STAGE=1 ;;
    *) echo "error: unknown argument: $1 (try --help)" >&2; exit 1 ;;
  esac
  shift
done

case "$FORMAT" in deb|tar|both) ;; *)
  echo "error: invalid --format '$FORMAT' (valid: deb | tar | both)" >&2; exit 1 ;;
esac

# Validate an explicitly-passed --arch; empty means auto-detect below.
if [[ -n "$DEB_ARCH" ]]; then
  case "$DEB_ARCH" in amd64|arm64) ;; *)
    echo "error: invalid --arch '$DEB_ARCH' (valid: amd64 | arm64)" >&2; exit 1 ;;
  esac
fi

# Resolve target architecture (debian naming) and aarch64 flag.
HOST_ARCH=$(uname -m)
if [[ -z "$DEB_ARCH" ]]; then
  case "$HOST_ARCH" in
    x86_64)  DEB_ARCH=amd64 ;;
    aarch64) DEB_ARCH=arm64 ;;
    *) echo "error: unsupported host arch '$HOST_ARCH'; pass --arch=" >&2; exit 1 ;;
  esac
fi
IS_AARCH64=0
[[ "$DEB_ARCH" = "arm64" ]] && IS_AARCH64=1

DS_INSTALL_ROOT=/opt/nvidia/deepstream/deepstream-${NVDS_VERSION}
DS_REL="opt/nvidia/deepstream/deepstream-${NVDS_VERSION}"
DS_ROOT="$STAGE_DIR/$DS_REL"

log() { printf '==> %s\n' "$*"; }

require() {
  local p=$1 what=$2
  [[ -e "$p" ]] || { echo "error: missing $what: $p" >&2; exit 1; }
}

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
require "$DS_INSTALL_ROOT" "built DeepStream install tree (run build/build.sh first)"
for sub in bin lib samples service-maker LicenseAgreement.pdf; do
  require "$DS_INSTALL_ROOT/$sub" "$DS_INSTALL_ROOT/$sub"
done

log "Packaging DeepStream $NVDS_VERSION (pkg $PKG_VERSION-$PKG_REVISION, arch $DEB_ARCH, aarch64=$IS_AARCH64)"
log "Source install tree : $DS_INSTALL_ROOT"
log "Staging directory   : $STAGE_DIR"
log "Output directory    : $OUTPUT_DIR"

# ---------------------------------------------------------------------------
# Stage the install tree
# ---------------------------------------------------------------------------
rm -rf "$STAGE_DIR"
mkdir -p "$DS_ROOT" "$OUTPUT_DIR"

copy_into() {  # copy_into <src> <dest-dir>   (preserves attrs, copies the entry itself)
  local src=$1 dest=$2
  require "$src" "$src"
  mkdir -p "$dest"
  cp -a "$src" "$dest/"
}

copy_contents_into() {  # copy_contents_into <src-dir> <dest-dir>  (copies dir contents)
  local src=$1 dest=$2
  require "$src" "$src"
  mkdir -p "$dest"
  cp -a "$src"/. "$dest/"
}

# Literal (non-regex) in-place replacement on a STAGED file only. Handles
# multi-line search strings (whole-file slurp). Fails loudly if the file or the
# search string is missing (so upstream drift is caught), and is idempotent
# (skips when the replacement is already present).
patch_staged_file() {  # patch_staged_file <staged-rel-path> <old-literal> <new-literal>
  local rel=$1 old=$2 new=$3
  local f="$DS_ROOT/$rel"
  [[ -f "$f" ]] || { echo "error: patch target missing in stage: $rel" >&2; exit 1; }
  if ! OLD="$old" perl -0777 -ne 'exit(index($_, $ENV{OLD}) >= 0 ? 0 : 1)' "$f"; then
    if NEW="$new" perl -0777 -ne 'exit(index($_, $ENV{NEW}) >= 0 ? 0 : 1)' "$f"; then
      log "  $rel: already patched, skipping"
      return 0
    fi
    echo "error: search string not found in staged $rel" >&2
    exit 1
  fi
  OLD="$old" NEW="$new" perl -0777 -pi -e 's/\Q$ENV{OLD}\E/$ENV{NEW}/g' "$f"
  log "  patched $rel"
}

# Ship only git-tracked files from the repo. Build a mirror containing just the
# tracked files under the paths we package, then point REPO_ROOT at it so every
# repo-source copy below pulls tracked files only -- untracked build artifacts
# (compiled binaries, .o files, .build dirs, local scratch) never reach the
# deb/tar. The /opt runtime tree (bin/lib/samples/...) is not from git and is
# unaffected. Mirror lives outside STAGE_DIR so it is never itself packaged.
SRC_MIRROR=""
cleanup_mirror() { [[ -n "${SRC_MIRROR:-}" ]] && rm -rf "$SRC_MIRROR"; }
trap cleanup_mirror EXIT
if git -C "$REPO_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  SRC_MIRROR=$(mktemp -d "${TMPDIR:-/tmp}/ds-srcmirror.XXXXXX")
  log "Mirroring git-tracked sources only (excluding untracked files)"
  git -C "$REPO_ROOT" ls-files -z -- \
      includes \
      src/apps/sample_apps src/apps/common \
      src/gst-plugins src/utils src/gst-utils src/service-maker \
      scripts \
      tools/deepstream-otelcol \
    | rsync -a --files-from=- --from0 "$REPO_ROOT/" "$SRC_MIRROR/"
  REPO_ROOT="$SRC_MIRROR"
else
  log "WARNING: $REPO_ROOT is not a git work tree; copying repo sources as-is (untracked included)"
fi

# 1. Runtime + assets copied as-is from the built install tree.
log "Staging runtime (bin, lib, samples, service-maker, LicenseAgreement.pdf)"
for item in bin lib samples service-maker LicenseAgreement.pdf; do
  copy_into "$DS_INSTALL_ROOT/$item" "$DS_ROOT"
done
# Keep only the top-level license; the duplicate bundled with samples is excluded.
rm -f "$DS_ROOT/samples/LicenseAgreement.pdf"

# The installed /opt samples carry the UNION of Jetson + x86 deepstream-app
# configs; ship only the ones for the target platform. (Names referenced from
# the per-platform sample trees; most configs are common to both.)
DSAPP_CFG_DIR="$DS_ROOT/samples/configs/deepstream-app"
JETSON_ONLY_CONFIGS=(
  config_inferserver_primary.txt
  source1_csi_dec_infer_resnet.txt
  source2_csi_usb_dec_infer_resnet.txt
  source6_csi_dec_infer_resnet.txt
)
X86_ONLY_CONFIGS=(
  config_infer_primary_endv.txt
  source4_1080p_dec_infer-resnet_tracker_sgie_tiled_display_gpu1.txt
)
if [[ -d "$DSAPP_CFG_DIR" ]]; then
  if [[ "$DEB_ARCH" = "amd64" ]]; then
    prune_configs=("${JETSON_ONLY_CONFIGS[@]}")
  else
    prune_configs=("${X86_ONLY_CONFIGS[@]}")
  fi
  for cfg in "${prune_configs[@]}"; do
    if [[ -f "$DSAPP_CFG_DIR/$cfg" ]]; then
      rm -f "$DSAPP_CFG_DIR/$cfg"
      log "Excluded $DEB_ARCH-foreign deepstream-app config: $cfg"
    fi
  done
fi

# 2. Repo service-maker sources merged into DS_ROOT/service-maker.
log "Merging repo src/service-maker into service-maker/"
copy_contents_into "$REPO_ROOT/src/service-maker" "$DS_ROOT/service-maker"

# OpenTelemetry collector helper: ship at DS_ROOT/otelcol (renamed from
# tools/deepstream-otelcol).
log "Staging tools/deepstream-otelcol -> otelcol/"
copy_contents_into "$REPO_ROOT/tools/deepstream-otelcol" "$DS_ROOT/otelcol"

# 3. sources/ tree (also creates sources/apps).
log "Staging sources/ tree"
mkdir -p "$DS_ROOT/sources/apps"
copy_into "$REPO_ROOT/includes"               "$DS_ROOT"                  # -> DS_ROOT/includes (apps/plugins -I paths resolve here)
copy_into "$REPO_ROOT/src/apps/sample_apps"   "$DS_ROOT/sources/apps"     # -> sources/apps/sample_apps
copy_into "$REPO_ROOT/src/apps/common"        "$DS_ROOT/sources/apps"     # -> sources/apps/common

# These two ship at sources/ top level, not under sources/apps/sample_apps.
for t in TritonBackendEnsemble TritonOnnxYolo; do
  if [[ -d "$DS_ROOT/sources/apps/sample_apps/$t" ]]; then
    rm -rf "$DS_ROOT/sources/$t"
    mv "$DS_ROOT/sources/apps/sample_apps/$t" "$DS_ROOT/sources/$t"
    log "Relocated $t -> sources/$t"
  fi
done

# Drop platform-specific sample apps that build/build.sh does not build on this
# target (mirrors its sample_apps skip logic; sbsa ignored for now):
#   x86 only      : deepstream-ucx-test, deepstream-multigpu-nvlink-test
#   x86 (+sbsa)   : deepstream-appsrc-cuda-test
#   Jetson only   : deepstream-ipc-test
if [[ "$DEB_ARCH" = "amd64" ]]; then
  SAMPLE_APPS_SKIP=(deepstream-ipc-test)
else
  SAMPLE_APPS_SKIP=(
    deepstream-ucx-test
    deepstream-multigpu-nvlink-test
    deepstream-appsrc-cuda-test
  )
fi
for app in "${SAMPLE_APPS_SKIP[@]}"; do
  if [[ -d "$DS_ROOT/sources/apps/sample_apps/$app" ]]; then
    rm -rf "$DS_ROOT/sources/apps/sample_apps/$app"
    log "Excluding sample app (not for $DEB_ARCH): $app"
  fi
done

copy_into "$REPO_ROOT/src/gst-plugins"        "$DS_ROOT/sources"          # -> sources/gst-plugins
copy_contents_into "$REPO_ROOT/src/utils"     "$DS_ROOT/sources/libs"     # -> sources/libs/*
copy_contents_into "$REPO_ROOT/src/gst-utils" "$DS_ROOT/sources/libs"     # -> sources/libs/*

# Drop plugin / gst-utils sources that build/build.sh gates off this target, so
# the shipped sources match what the package's runtime was actually built from.
# gst-nvdsudp and gst-dsexample-cuda are deliberately kept on both: build.sh
# skips them, but they are meant to be built by hand per their READMEs.
if [[ "$DEB_ARCH" = "amd64" ]]; then
  # gst-nvipcmeta is Jetson/aarch64 only.
  if [[ -d "$DS_ROOT/sources/libs/gst-nvipcmeta" ]]; then
    rm -rf "$DS_ROOT/sources/libs/gst-nvipcmeta"
    log "Excluding gst-utils source (not for $DEB_ARCH): gst-nvipcmeta"
  fi
else
  for plugin in gst-nvblender gst-nvbufferpool gst-nvbufsurfacewrite gst-nvdsucx gst-nvdsxfer; do
    if [[ -d "$DS_ROOT/sources/gst-plugins/$plugin" ]]; then
      rm -rf "$DS_ROOT/sources/gst-plugins/$plugin"
      log "Excluding plugin source (not for $DEB_ARCH): $plugin"
    fi
  done
fi

# Restructure nvds_msgapi: protocol adaptors and common_src move to libs/ top level.
# common_src is renamed to nvds_msgapi_common_src; protocol adaptors stay at libs/.
log "Restructuring nvds_msgapi protocol adaptors"
if [[ -d "$DS_ROOT/sources/libs/nvds_msgapi" ]]; then
  cd "$DS_ROOT/sources/libs"
  for proto in amqp_protocol_adaptor azure_protocol_adaptor kafka_protocol_adaptor mqtt_protocol_adaptor redis_protocol_adaptor; do
    if [[ -d "nvds_msgapi/$proto" ]]; then
      mv "nvds_msgapi/$proto" .
      log "  moved nvds_msgapi/$proto -> libs/$proto"
    fi
  done
  if [[ -d "nvds_msgapi/common_src" ]]; then
    mv "nvds_msgapi/common_src" "nvds_msgapi_common_src"
    log "  renamed nvds_msgapi/common_src -> libs/nvds_msgapi_common_src"
  fi
  # Remove the now-empty nvds_msgapi directory (including README.md).
  rm -rf "nvds_msgapi"
  log "  removed nvds_msgapi directory"
  cd - >/dev/null
fi

# tracker_ReID (README) ships at sources/ top level, relocated out of the
# nvmultiobjecttracker copy. The rest of nvmultiobjecttracker source is not
# shipped under sources/libs (only its built .so ships in lib/).
if [[ -d "$DS_ROOT/sources/libs/nvmultiobjecttracker/model/tracker_ReID" ]]; then
  rm -rf "$DS_ROOT/sources/tracker_ReID"
  mv "$DS_ROOT/sources/libs/nvmultiobjecttracker/model/tracker_ReID" "$DS_ROOT/sources/tracker_ReID"
  log "Relocated tracker_ReID -> sources/tracker_ReID"
fi
rm -rf "$DS_ROOT/sources/libs/nvmultiobjecttracker"

# nvds_logger ships under sources/tools, not sources/libs.
if [[ -d "$DS_ROOT/sources/libs/nvds_logger" ]]; then
  mkdir -p "$DS_ROOT/sources/tools"
  rm -rf "$DS_ROOT/sources/tools/nvds_logger"
  mv "$DS_ROOT/sources/libs/nvds_logger" "$DS_ROOT/sources/tools/nvds_logger"
  log "Relocated nvds_logger -> sources/tools/nvds_logger"
fi

# 4. Scripts shipped at DS_ROOT root / samples.
log "Staging install/uninstall + helper scripts"
copy_into "$REPO_ROOT/scripts/uninstall.sh" "$DS_ROOT"
copy_into "$REPO_ROOT/scripts/install.sh"   "$DS_ROOT"
# rtpmanager: contents copied directly into DS_ROOT (no enclosing folder).
copy_contents_into "$REPO_ROOT/scripts/rtpmanager" "$DS_ROOT"

# prepare_* helpers go into samples/.
for s in prepare_classification_test_video.sh \
         prepare_ds_triton_tao_model_repo.sh \
         prepare_ds_triton_model_repo.sh; do
  copy_into "$REPO_ROOT/scripts/$s" "$DS_ROOT/samples"
done
# triton_backend_setup.sh only for aarch64 packaging.
if [[ "$IS_AARCH64" -eq 1 ]]; then
  log "aarch64 packaging: including triton_backend_setup.sh in samples/"
  copy_into "$REPO_ROOT/scripts/triton_backend_setup.sh" "$DS_ROOT/samples"
fi

# 4b. Patch staged Makefiles so paths resolve against the installed sources/
# layout. Both src/utils/ and src/gst-utils/ are shipped under sources/libs/,
# so '../../utils' and '../../gst-utils' both become '../../libs'. Repo working
# tree is untouched; edits apply to deb and tar.
log "Patching staged Makefile paths (utils -> libs)"
for p in gst-nvinfer gst-nvinferserver gst-nvmsgconv gst-nvmultistream2 \
         gst-nvmultiurisrcbin gst-nvds3dbridge gst-nvds3dfilter gst-nvds3dmixer \
         gst-nvdsbins gst-nvdslogger gst-nvmodelmux gst-nvmultistream gst-nvsegvisual \
         gst-nvtiler; do
  patch_staged_file "sources/gst-plugins/$p/Makefile" "../../utils" "../../libs"
done

# gst-nvdsucx is x86-only, so it is absent from arm64 packages (pruned above).
if [[ -f "$DS_ROOT/sources/gst-plugins/gst-nvdsucx/Makefile" ]]; then
  patch_staged_file "sources/gst-plugins/gst-nvdsucx/Makefile" "../../utils" "../../libs"
fi

# These plug-ins additionally include gst-utils headers, which land in
# sources/libs/ alongside everything else.
for p in gst-nvdsbins gst-nvinfereval gst-nvmodelmux gst-nvmultiurisrcbin; do
  patch_staged_file "sources/gst-plugins/$p/Makefile" "../../gst-utils" "../../libs"
done

# gst-nvdsinferbase ships under sources/libs/ itself but still refers to
# sibling utils/ headers.
patch_staged_file "sources/libs/gst-nvdsinferbase/Makefile" "../../utils" "../../libs"

# nvds_utils.cpp consumers: the restructuring above lifted common_src out to
# libs/nvds_msgapi_common_src and deleted nvds_msgapi/, so point them all at the
# new location. gst-nvmultiurisrcbin is matched with a libs/ prefix because the
# utils -> libs loop above has already rewritten it.
for a in deepstream-app deepstream-test5 deepstream-transfer-learning-app; do
  patch_staged_file "sources/apps/sample_apps/$a/Makefile" \
    "../../../../src/utils/nvds_msgapi/common_src/nvds_utils.cpp" \
    "../../../libs/nvds_msgapi_common_src/nvds_utils.cpp"
  patch_staged_file "sources/apps/sample_apps/$a/Makefile" \
    "../../../gst-utils" \
    "../../../libs"
done
patch_staged_file "sources/gst-plugins/gst-nvmultiurisrcbin/Makefile" \
  "../../libs/nvds_msgapi/common_src/nvds_utils.cpp" \
  "../../libs/nvds_msgapi_common_src/nvds_utils.cpp"

# deepstream-test5: nvds_msgapi include dir under installed sources/libs.
patch_staged_file "sources/apps/sample_apps/deepstream-test5/Makefile" \
  "-I../../../src/utils/nvds_msgapi/inc" \
  "-I../../../libs/nvds_msgapi/inc"

# deepstream-server: link against libcurl.
patch_staged_file "sources/apps/sample_apps/deepstream-server/Makefile" \
  "-lnvds_rest_server -lm" \
  "-lnvds_rest_server -lm -lcurl"

# deepstream-infer-tensor-meta-test: custom bbox parser source under sources/libs.
patch_staged_file "sources/apps/sample_apps/deepstream-infer-tensor-meta-test/Makefile" \
  "../../../utils/nvdsinfer_customparser" \
  "../../../libs/nvdsinfer_customparser"

# nvds_msgapi protocol adaptors: DS_INC paths adjusted for new libs/ top-level location,
# and common_src renamed to nvds_msgapi_common_src.
log "Patching nvds_msgapi protocol adaptor Makefiles"
for proto in amqp_protocol_adaptor kafka_protocol_adaptor mqtt_protocol_adaptor redis_protocol_adaptor; do
  if [[ -f "$DS_ROOT/sources/libs/$proto/Makefile" ]]; then
    patch_staged_file "sources/libs/$proto/Makefile" \
      "DS_INC:= ../../../../includes" \
      "DS_INC:= ../../../includes"
    patch_staged_file "sources/libs/$proto/Makefile" \
      "../common_src/nvds_utils.cpp" \
      "../nvds_msgapi_common_src/nvds_utils.cpp"
  fi
  if [[ -f "$DS_ROOT/sources/libs/$proto/Makefile.test" ]]; then
    if [[ "$proto" = "amqp_protocol_adaptor" || "$proto" = "mqtt_protocol_adaptor" ]]; then
      test_ds_inc="DS_INC:= ../../../../includes/"
    else
      test_ds_inc="DS_INC:= ../../../../includes"
    fi
    patch_staged_file "sources/libs/$proto/Makefile.test" \
      "$test_ds_inc" \
      "DS_INC:= ../../../includes"
  fi
done

# azure_protocol_adaptor has device_client and module_client subdirectories.
for client in device_client module_client; do
  if [[ -f "$DS_ROOT/sources/libs/azure_protocol_adaptor/$client/Makefile" ]]; then
    patch_staged_file "sources/libs/azure_protocol_adaptor/$client/Makefile" \
      "DS_INC:= ../../../../../includes" \
      "DS_INC:= ../../../../includes"
    if [[ "$client" = "device_client" ]]; then
      patch_staged_file "sources/libs/azure_protocol_adaptor/$client/Makefile" \
        "../../common_src/nvds_utils.cpp" \
        "../../nvds_msgapi_common_src/nvds_utils.cpp"
    fi
  fi
  if [[ -f "$DS_ROOT/sources/libs/azure_protocol_adaptor/$client/Makefile.test" ]]; then
    patch_staged_file "sources/libs/azure_protocol_adaptor/$client/Makefile.test" \
      "DS_INC:= ../../../../../includes" \
      "DS_INC:= ../../../../includes"
  fi
done

# includes now live at DS_ROOT top-level (deepstream/includes), not under
# sources/; drop the 'sources/' segment in these Makefiles' -I paths. Matches
# both '$(DS_HOME)/sources/includes' and '.../deepstream/sources/includes' while
# leaving '/sources/libs' untouched.
for mk in \
  sources/libs/ds3d/dataloader/lidarsource/Makefile \
  sources/TritonBackendEnsemble/nvdsinferserver_custom_impl_ensemble/Makefile \
  sources/TritonOnnxYolo/nvdsinfer_custom_impl_yolo/Makefile \
  sources/TritonOnnxYolo/nvdsinferserver_custom_impl_yolo/Makefile; do
  patch_staged_file "$mk" "/sources/includes" "/includes"
done

# The repo CMake config lives at src/service-maker/cmake, where ../../../includes
# resolves to the repo include root. In the package it lives at
# service-maker/cmake, so the staged copy needs one fewer parent traversal to
# resolve to DS_ROOT/includes. Keep the repo config unchanged for source builds.
patch_staged_file "service-maker/cmake/nvds_service_makerConfig.cmake" \
  '"${CMAKE_CURRENT_LIST_DIR}/../../../includes"' \
  '"${CMAKE_CURRENT_LIST_DIR}/../../includes"'

# service-maker core/engine: the installed tree has no sibling apps/common, so
# point APP_COMMON_DIR at its installed location using the versioned path, and
# add an INC_DIR knob for the installed include root. We move NVDS_VERSION
# definition before APP_COMMON_DIR so the latter can reference it. These insert
# lines above their anchor, which is only safe because STAGE_DIR is recreated
# from scratch on every run.
log "Patching staged service-maker Makefiles"

# For core Makefile: remove existing NVDS_VERSION, then add it back with APP_COMMON_DIR
patch_staged_file "service-maker/sources/core/Makefile" \
  "NVDS_VERSION:=${NVDS_VERSION}" \
  ''

patch_staged_file "service-maker/sources/core/Makefile" \
  'APP_COMMON_DIR:= ../../../apps/common' \
  "NVDS_VERSION:=${NVDS_VERSION}
APP_COMMON_DIR:=/opt/nvidia/deepstream/deepstream-\$(NVDS_VERSION)/sources/apps/common"

for mk in service-maker/sources/core/Makefile service-maker/sources/engine/Makefile; do
  patch_staged_file "$mk" \
    "NVDS_VERSION:=${NVDS_VERSION}" \
    "INC_DIR?=/opt/nvidia/deepstream/deepstream-\$(NVDS_VERSION)/includes
NVDS_VERSION:=${NVDS_VERSION}"
  patch_staged_file "$mk" \
    '-I/usr/local/cuda-$(CUDA_VER)/include' \
    '-I$(INC_DIR) -I/usr/local/cuda-$(CUDA_VER)/include'
done

# uninstall.sh: this bundle ships as a single Debian package named
# deepstream-${NVDS_VERSION}, not the split deepstream-binaries-*/sample-data
# packages. Deregister the right package, and skip dpkg -r when invoked from the
# package's own prerm (dpkg sets DPKG_MAINTSCRIPT_PACKAGE and handles removal
# itself) so it does not recurse.
log "Patching staged uninstall.sh (deregister deepstream-<ver> package)"
patch_staged_file "uninstall.sh" \
'remove_if_deb "deepstream-sample-data"
if [ -n "${PLATFORM}" ]; then
  remove_if_deb "deepstream-binaries-${PLATFORM}"
fi' \
'# Single self-contained package: deregister deepstream-${PREV_DS_VER}. When run
# from this package own prerm, dpkg already deregisters it (it sets
# DPKG_MAINTSCRIPT_PACKAGE), so skip dpkg -r here to avoid recursion.
if [ -z "${DPKG_MAINTSCRIPT_PACKAGE}" ]; then
  remove_if_deb "deepstream-${PREV_DS_VER}"
fi'

# 5. version file.
log "Writing version file (Version: $PKG_VERSION, DATE: $BUILD_DATE)"
cat > "$DS_ROOT/version" <<EOF
Version: $PKG_VERSION
DATE: $BUILD_DATE
EOF

# Versionless convenience symlink: /opt/nvidia/deepstream/deepstream -> deepstream-<ver>.
# Relative target so it resolves once installed/extracted under /opt/nvidia/deepstream/.
log "Creating symlink deepstream -> deepstream-${NVDS_VERSION}"
ln -sfn "deepstream-${NVDS_VERSION}" "$STAGE_DIR/opt/nvidia/deepstream/deepstream"

# Strip any VCS metadata that came along with the copied sources so it never
# ends up in the deb/tar (.git history, .gitignore, .gitkeep, submodule files,
# CI config dirs, etc.).
log "Pruning git/VCS metadata from staged tree"
git_pruned=$(find "$STAGE_DIR" -depth \( \
    -name '.git' -o -name '.github' -o -name '.gitlab' \
    -o -name '.gitignore' -o -name '.gitattributes' \
    -o -name '.gitmodules' -o -name '.gitkeep' \
    -o -name '.gitlab-ci.yml' \) -print -exec rm -rf {} + | wc -l)
log "  removed $git_pruned git/VCS entr$([[ "$git_pruned" -eq 1 ]] && echo y || echo ies)"

# Rules.mk files are unused build scaffolding (nothing includes them); keep them
# out of the shipped sources.
log "Pruning unused Rules.mk from staged tree"
rules_pruned=$(find "$STAGE_DIR" -depth -name 'Rules.mk' -print -exec rm -f {} + | wc -l)
log "  removed $rules_pruned Rules.mk file(s)"

# Repository README files are not part of the SDK payload, except for the
# OpenTelemetry collector documentation shipped with otelcol.
log "Pruning README.md files from staged tree (except otelcol/README.md)"
readmes_pruned=$(find "$DS_ROOT" -type f -name 'README.md' \
  ! -path "$DS_ROOT/otelcol/README.md" -print -delete | wc -l)
log "  removed $readmes_pruned README.md file(s)"

# ---------------------------------------------------------------------------
# Emit tarball
# ---------------------------------------------------------------------------
if [[ "$DEB_ARCH" = "arm64" ]]; then
  TAR_PLATFORM="jetson"
else
  TAR_PLATFORM="x86_64"
fi
TAR_NAME="deepstream_sdk_v${NVDS_FULL_VERSION}_${TAR_PLATFORM}.tbz2"
build_tar() {
  log "Building tarball: $OUTPUT_DIR/$TAR_NAME"
  # Include both the versioned tree and the sibling "deepstream" symlink.
  tar -C "$STAGE_DIR" -cjf "$OUTPUT_DIR/$TAR_NAME" \
    "$DS_REL" "opt/nvidia/deepstream/deepstream"
  log "  wrote $(du -h "$OUTPUT_DIR/$TAR_NAME" | cut -f1) -> $OUTPUT_DIR/$TAR_NAME"
}

# ---------------------------------------------------------------------------
# Emit Debian package
# ---------------------------------------------------------------------------
DEB_NAME="deepstream-${NVDS_VERSION}_${PKG_VERSION}-${PKG_REVISION}_${DEB_ARCH}.deb"
build_deb() {
  command -v dpkg-deb >/dev/null 2>&1 || {
    echo "error: dpkg-deb not found; cannot build .deb (use --format=tar)" >&2; exit 1; }

  local debian_dir="$STAGE_DIR/DEBIAN"
  mkdir -p "$debian_dir"
  local installed_size
  installed_size=$(du -s -k "$DS_ROOT" | cut -f1)

  # Architecture-specific runtime dependencies / conflicts.
  #
  # This self-contained package installs into the same DS_ROOT and ships the
  # very files (LicenseAgreement.pdf, libs, samples, ...) that the split
  # artifact packages own once build/build.sh installs them via dpkg
  # (deepstream-binaries-<platform>, deepstream-sample-data). dpkg refuses to
  # let one installed package overwrite another's files, so declare Conflicts
  # on those packages: apt then removes them before unpacking this one (works
  # with `apt install ./pkg.deb`; a plain `dpkg -i` still needs them removed
  # first). Harmless no-op when the artifacts were installed via tar (no
  # registered dpkg owner).
  local depends conflicts
  if [[ "$DEB_ARCH" = "arm64" ]]; then
    depends="cuda-cudart-13-0 | cuda-cudart-13-2, cuda-cudart-dev-13-0 | cuda-cudart-dev-13-2, libnpp-13-0 | libnpp-13-2, libnpp-dev-13-0 | libnpp-dev-13-2, libcufft-13-0 | libcufft-13-2, libcairo2 (>= 1.16.0), libglib2.0-0 (>= 2.72.4), libgstreamer1.0-0 (>= 1.20.3), libgstreamer1.0-dev (>= 1.20.3), libgstreamer-plugins-base1.0-0 (>= 1.20.1), libgstreamer-plugins-base1.0-dev (>= 1.20.1), libnvinfer10 (>= 10.0.0), libnvinfer-dev (>= 10.0.0), libnvonnxparsers10 (>= 10.0.0), libnvonnxparsers-dev (>= 10.0.0), libnvinfer-plugin10 (>= 10.0.0), libnvinfer-plugin-dev (>= 10.0.0), libpangocairo-1.0-0 (>= 1.40.14), libx11-6, libnvvpi4, libyaml-cpp-dev (>=0.6.2), python3-pip (>=24.0+dfsg-1ubuntu1.1)"
    conflicts="deepstream-binaries-aarch64, deepstream-sample-data"
  else
    depends="cuda-cudart-13-0 | cuda-cudart-13-2, cuda-cudart-dev-13-0 | cuda-cudart-dev-13-2, libnpp-13-0 | libnpp-13-2, libnpp-dev-13-0 | libnpp-dev-13-2, libcairo2 (>= 1.16.0), libglib2.0-0 (>= 2.72.4), libgstreamer1.0-0 (>= 1.20.3), libgstreamer1.0-dev (>= 1.20.3), libgstreamer-plugins-base1.0-0 (>= 1.20.1), libgstreamer-plugins-base1.0-dev (>= 1.20.1), libnvinfer10 (>= 10.0.0), libnvinfer-dev (>= 10.0.0), libnvonnxparsers10 (>= 10.0.0), libnvonnxparsers-dev (>= 10.0.0), libnvinfer-plugin10 (>= 10.0.0), libnvinfer-plugin-dev (>= 10.0.0), libpangocairo-1.0-0 (>= 1.40.14), libx11-6, libyaml-cpp-dev (>=0.6.2), libgbm1 (>= 23.0.4-0ubuntu1~24.04.1), libglapi-mesa (>= 23.0.4-0ubuntu1~24.04.1), libgles2-mesa-dev (>= 23.0.4-0ubuntu1~24.04.1), python3-pip (>=24.0+dfsg-1ubuntu1.1)"
    conflicts="libglapi-amber, deepstream-binaries-x86, deepstream-sample-data"
  fi

  {
    echo "Package: deepstream-${NVDS_VERSION}"
    echo "Version: ${PKG_VERSION}-${PKG_REVISION}"
    echo "Maintainer: DeepStreamSDK-Support <DeepStreamSDK-Support@nvidia.com>"
    echo "Architecture: ${DEB_ARCH}"
    echo "Description: DeepStreamSDK runtime libraries, development files and samples"
    echo "Depends: ${depends}"
    [[ -n "$conflicts" ]] && echo "Conflicts: ${conflicts}"
    echo "Source: deepstream"
    echo "Section: Utils"
    echo "Priority: Standard"
    echo "Installed-Size: ${installed_size}"
  } > "$debian_dir/control"

  cat > "$debian_dir/preinst" <<EOF
#!/bin/bash
set -e
# Remove old symlinks from install_artifacts.sh before installing new package structure.
INSTALL_ROOT=/opt/nvidia/deepstream/deepstream-${NVDS_VERSION}
if [ -L "\$INSTALL_ROOT/sources/apps/sample_apps" ]; then
  rm -f "\$INSTALL_ROOT/sources/apps/sample_apps"
fi
if [ -L "\$INSTALL_ROOT/sources/libs/kafka_protocol_adaptor" ]; then
  rm -f "\$INSTALL_ROOT/sources/libs/kafka_protocol_adaptor"
fi
if [ -L "\$INSTALL_ROOT/sources/libs/nvds_msgapi_common_src" ]; then
  rm -f "\$INSTALL_ROOT/sources/libs/nvds_msgapi_common_src"
fi
if [ -L "\$INSTALL_ROOT/sources/libs/amqp_protocol_adaptor" ]; then
  rm -f "\$INSTALL_ROOT/sources/libs/amqp_protocol_adaptor"
fi
if [ -L "\$INSTALL_ROOT/sources/libs/azure_protocol_adaptor/device_client" ]; then
  rm -f "\$INSTALL_ROOT/sources/libs/azure_protocol_adaptor/device_client"
fi
if [ -L "\$INSTALL_ROOT/sources/libs/azure_protocol_adaptor/module_client" ]; then
  rm -f "\$INSTALL_ROOT/sources/libs/azure_protocol_adaptor/module_client"
fi
if [ -L "\$INSTALL_ROOT/sources/libs/mqtt_protocol_adaptor" ]; then
  rm -f "\$INSTALL_ROOT/sources/libs/mqtt_protocol_adaptor"
fi
if [ -L "\$INSTALL_ROOT/sources/libs/redis_protocol_adaptor" ]; then
  rm -f "\$INSTALL_ROOT/sources/libs/redis_protocol_adaptor"
fi
EOF

  cat > "$debian_dir/postinst" <<EOF
#!/bin/bash
set -e
# Guard against a missing install.sh so reconfigure/cleanup never hard-fails
# (e.g. when the payload was removed out-of-band).
if [ -x /opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/install.sh ]; then
  NVDS_VERSION=${NVDS_FULL_VERSION} /opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/install.sh
fi
echo "---------------------------------------------------------------------------------------"
echo "NOTE: sources and samples folders will be found in /opt/nvidia/deepstream/deepstream-${NVDS_VERSION}"
echo "---------------------------------------------------------------------------------------"
EOF

  cat > "$debian_dir/prerm" <<EOF
#!/bin/bash
set -e
# prerm runs on remove and upgrade (old-prerm upgrade <new-version>). Only run
# destructive uninstall cleanup on actual removal; upgrades are handled by the
# new package's postinst (install.sh). Guard against a missing uninstall.sh so
# removal never hard-fails if the payload was already deleted out-of-band.
if [ "\$1" = "remove" ]; then
  if [ -x /opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/uninstall.sh ]; then
    /opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/uninstall.sh ${NVDS_FULL_VERSION}
  fi
  echo "---------------------------------------------------------------------------------------"
  echo "DeepStream ${NVDS_VERSION} was removed from the system"
  echo "---------------------------------------------------------------------------------------"
fi
EOF
  chmod 0755 "$debian_dir/preinst" "$debian_dir/postinst" "$debian_dir/prerm"

  log "Building Debian package: $OUTPUT_DIR/$DEB_NAME"
  # fakeroot (when available) gives the payload root:root ownership like the
  # shipped packages; otherwise fall back to plain dpkg-deb.
  if command -v fakeroot >/dev/null 2>&1; then
    fakeroot dpkg-deb --build --root-owner-group "$STAGE_DIR" "$OUTPUT_DIR/$DEB_NAME" >/dev/null
  else
    dpkg-deb --build --root-owner-group "$STAGE_DIR" "$OUTPUT_DIR/$DEB_NAME" >/dev/null
  fi
  rm -rf "$debian_dir"
  log "  wrote $(du -h "$OUTPUT_DIR/$DEB_NAME" | cut -f1) -> $OUTPUT_DIR/$DEB_NAME"
}

case "$FORMAT" in
  deb)  build_deb ;;
  tar)  build_tar ;;
  both) build_tar; build_deb ;;
  *)    echo "ERROR: Unknown format '$FORMAT'. Expected one of: deb, tar, both." >&2; exit 1 ;;
esac

if [[ "$KEEP_STAGE" -eq 0 ]]; then
  rm -rf "$STAGE_DIR"
else
  log "Keeping staging directory: $STAGE_DIR"
fi

log "Done."
