#!/bin/bash
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

# Convert the official WILDTRACK and SCOUT downloads from EPFL into datasets/wildtrack and datasets/scout.
# Both datasets come with their own terms, which the user must accept before downloading.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MV3DT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$MV3DT_DIR"

WILDTRACK_ZIP_URL="https://documents.epfl.ch/groups/c/cv/cvlab-unit/www/data/Wildtrack/Wildtrack_dataset_full.zip"
# Google Drive ids of the seven camera videos listed on the official WILDTRACK page.
WILDTRACK_VIDEO_IDS=(
    1sGUnExmJM2_tFuBd9LNlexf0LN2m0_c-
    1OnaRN2qYhZ2n4rSaNZQJzQPd1Cl2fluk
    1I7ARLVVfZqdZTbYG_TPEfDt1Bc8N35Lb
    1sXn70X-bV_YGPv43r4-iMtK_Js09eUVB
    1ExTyezMWqmLefi2kJTQwYxZivJuag7G5
    1NM4kNjdyiC6JioOT90s9vFmMUkeSnV0z
    1pZ3pWBuaLgPWGfOcY-tAtsBAUyQZW_YX
)

DATASETS_DIR="${MV3DT_DATASETS_DIR:-$MV3DT_DIR/datasets}"
DETECTIONS_DIR="$MV3DT_DIR/assets/detections"
DATASET="all"
WILDTRACK_SOURCE="$DATASETS_DIR/raw/wildtrack"
SCOUT_SOURCE="$DATASETS_DIR/raw/scout"
DOWNLOAD=0
FORCE=0
JOBS=""
DETECTIONS_ONLY=0

usage() {
    cat <<EOF
Usage: bash scripts/setup_datasets.sh [OPTIONS]

Convert the official EPFL downloads into datasets/wildtrack and datasets/scout.

Options:
  --dataset NAME         wildtrack, scout or all (default: all)
  --download             Fetch the WILDTRACK zip and camera videos into datasets/raw/wildtrack.
                         SCOUT cannot be downloaded by script, instructions are printed when it is missing.
  --jobs N               Cameras converted in parallel
  --detections-only      Only copy the recorded detections into an existing generated dataset
  --force                Overwrite an existing generated dataset
  -h, --help             Show this help

Official sources and terms:
  WILDTRACK  https://www.epfl.ch/labs/cvlab/data/data-wildtrack/
  SCOUT      https://scout.epfl.ch/download

Both datasets are provided by EPFL CVLab under their own terms. Read and follow them, obtain
permission for your intended use, and do not redistribute the downloaded or generated files.
EOF
}

log_info() { echo "[INFO] $*"; }
log_error() { echo "[ERROR] $*" >&2; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) DATASET="${2:?--dataset needs a value}"; shift 2 ;;
        --download) DOWNLOAD=1; shift ;;
        --jobs) JOBS="${2:?--jobs needs a value}"; shift 2 ;;
        --detections-only) DETECTIONS_ONLY=1; shift ;;
        --force) FORCE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) log_error "Unknown option: $1"; usage >&2; exit 2 ;;
    esac
done
case "$DATASET" in wildtrack|scout|all) ;; *) log_error "--dataset must be wildtrack, scout or all"; exit 2 ;; esac

PYTHON="python3"
if [[ -x "$MV3DT_DIR/mv3dt_venv/bin/python" ]]; then
    PYTHON="$MV3DT_DIR/mv3dt_venv/bin/python"
fi
if [[ "$DOWNLOAD" = 1 && "$DATASET" = scout ]]; then
    echo "[WARN] --download only applies to WILDTRACK and is ignored for --dataset scout" >&2
fi

require_ffmpeg() {
    command -v ffmpeg >/dev/null 2>&1 || { log_error "ffmpeg is required. Install it with: sudo apt-get install ffmpeg"; exit 1; }
}

EXTRA_ARGS=()
[[ "$FORCE" = 1 ]] && EXTRA_ARGS+=(--force)
[[ -n "$JOBS" ]] && EXTRA_ARGS+=(--jobs "$JOBS")

# Called under ||, which disables errexit, so every step returns 1 on failure.
download_wildtrack() {
    mkdir -p "$WILDTRACK_SOURCE" || return 1
    local folder
    if [[ ! -d "$WILDTRACK_SOURCE/annotations_positions" || ! -d "$WILDTRACK_SOURCE/calibrations" || ! -d "$WILDTRACK_SOURCE/Image_subsets" ]]; then
        local zip="$WILDTRACK_SOURCE/Wildtrack_dataset_full.zip"
        local unzipped="$WILDTRACK_SOURCE/_unzipped"
        log_info "Downloading Wildtrack_dataset_full.zip (about 6.8 GB) from EPFL. The transfer resumes if interrupted."
        curl -L --fail -C - -o "$zip" "$WILDTRACK_ZIP_URL" || { log_error "Downloading $WILDTRACK_ZIP_URL failed. Rerun to resume."; return 1; }
        log_info "Extracting the official files"
        rm -rf "$unzipped"
        unzip -q "$zip" 'Wildtrack_dataset/*' -d "$unzipped" || { log_error "Extracting $zip failed. Delete it and rerun."; return 1; }
        for folder in annotations_positions calibrations Image_subsets; do
            mv "$unzipped/Wildtrack_dataset/$folder" "$WILDTRACK_SOURCE/$folder" || { log_error "Cannot move $folder out of $unzipped"; return 1; }
        done
        if [[ -f "$unzipped/Wildtrack_dataset/rectangles.pom" ]]; then
            mv "$unzipped/Wildtrack_dataset/rectangles.pom" "$WILDTRACK_SOURCE/" || { log_error "Cannot move rectangles.pom out of $unzipped"; return 1; }
        fi
        rm -rf "$unzipped"
    fi
    local have_gdown=0 index id video
    "$PYTHON" -c 'import gdown' 2>/dev/null && have_gdown=1
    for index in "${!WILDTRACK_VIDEO_IDS[@]}"; do
        id="${WILDTRACK_VIDEO_IDS[$index]}"
        video="$WILDTRACK_SOURCE/cam$((index + 1)).mp4"
        [[ -s "$video" ]] && continue
        if [[ "$have_gdown" = 0 ]]; then
            log_error "Camera $((index + 1)) video is missing. Install gdown with: $PYTHON -m pip install gdown, then rerun with --download."
            log_error "Or download it manually from https://drive.google.com/open?id=$id and save it as $video"
            return 1
        fi
        log_info "Downloading camera $((index + 1)) video (about 0.5 GB) from Google Drive"
        "$PYTHON" -m gdown "https://drive.google.com/uc?id=$id" -O "$video" || { rm -f "$video"; log_error "Downloading the camera $((index + 1)) video failed"; return 1; }
    done
}

print_scout_instructions() {
    cat >&2 <<EOF
[ERROR] The official SCOUT files were not found in: $SCOUT_SOURCE

SCOUT cannot be downloaded by this script. To get it:
  1. Open https://scout.epfl.ch/download, read the dataset terms and follow the download instructions.
  2. The sample uses sequence_1, cameras 0 to 7, and the second half of the sequence. If the site
     offers downloads per camera or per sequence, download only that subset, otherwise the full dataset
     is about 350 GB.
  3. Place the files so that these folders exist under $SCOUT_SOURCE:
       annotations/  calibrations/  timestamps/  images/sequence_1/cam_0 ... cam_7/
  4. Cite the dataset paper (arXiv 2507.08494) in any work that uses it.
EOF
}

# Copy the recorded detections from assets/detections next to the generated dataset for replay.
# Existing files are never overwritten.
install_detections() {
    local name="$1"
    [[ -d "$DETECTIONS_DIR/$name" ]] || return 0
    mkdir -p "$DATASETS_DIR/$name/detections"
    cp -r --update=none "$DETECTIONS_DIR/$name/." "$DATASETS_DIR/$name/detections/"
    log_info "Recorded detections copied to $DATASETS_DIR/$name/detections"
}

status=0

if [[ "$DETECTIONS_ONLY" = 1 ]]; then
    for name in wildtrack scout; do
        if [[ "$DATASET" = all || "$DATASET" = "$name" ]]; then
            [[ -d "$DATASETS_DIR/$name/videos" ]] || { log_error "datasets/$name has not been generated yet"; status=1; continue; }
            install_detections "$name"
        fi
    done
    exit "$status"
fi

if [[ "$DATASET" = wildtrack || "$DATASET" = all ]]; then
    if [[ "$DOWNLOAD" = 1 ]]; then
        download_wildtrack || status=1
    fi
    if [[ "$status" = 0 ]]; then
        require_ffmpeg
        log_info "Generating datasets/wildtrack from $WILDTRACK_SOURCE"
        if "$PYTHON" utils/prepare_wildtrack.py --source "$WILDTRACK_SOURCE" --output "$DATASETS_DIR/wildtrack" "${EXTRA_ARGS[@]}"; then
            install_detections wildtrack
        else
            status=1
        fi
    fi
fi

if [[ "$DATASET" = scout || "$DATASET" = all ]]; then
    if [[ ! -d "$SCOUT_SOURCE/annotations" || ! -d "$SCOUT_SOURCE/images" ]]; then
        print_scout_instructions
        status=1
    else
        require_ffmpeg
        log_info "Generating datasets/scout from $SCOUT_SOURCE"
        if "$PYTHON" utils/prepare_scout.py --source "$SCOUT_SOURCE" --output "$DATASETS_DIR/scout" "${EXTRA_ARGS[@]}"; then
            install_detections scout
        else
            status=1
        fi
    fi
fi

if [[ "$status" != 0 ]]; then
    log_error "Dataset setup did not complete. Fix the errors above and rerun."
fi
exit "$status"
