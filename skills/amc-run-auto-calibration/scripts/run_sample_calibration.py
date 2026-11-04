#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run AMC calibration against the bundled SDG sample dataset.

Usage: python3 run_sample_calibration.py
Arguments: environment inputs DEEPSTREAM_REPO and AMC URL/port; optional
    PROJECT_NAME, RUN_VGGT. See sample-dataset.md for archive discovery paths.
Output: project ID, terminal states, evaluation metrics, and artifact location.
Exit codes: 0 on success; 1 on validation, API, or unhandled execution failure.
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
import zipfile
from pathlib import Path

from amc_common import (
    DEFAULT_TIMEOUT,
    check_ready,
    create_project,
    default_base_url,
    env_bool,
    infer_amc_root,
    infer_deepstream_repo,
    maybe_run_vggt,
    poll_project_completion,
    require_existing_file,
    require_requests,
    run_postprocess,
    stage_linear_media,
    upload_binary_file,
    upload_json_file,
    upload_videos,
    verify_project,
)

SAMPLE_NAME = "sdg_08_2_sample_data_010926"
SAMPLE_ZIP = f"{SAMPLE_NAME}.zip"
# SHA-256 from the tracked archive's Git LFS oid in this release.
SAMPLE_SHA256 = "0dceb0cc8324f5775b0c2007efe7a3e7c36fda10c5964b88e20712b002d98bdb"


def verify_sample_archive(path: Path, expected_sha256: str = SAMPLE_SHA256) -> None:
    with path.open("rb") as handle:
        checksum = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(chunk)
        digest = checksum.hexdigest()
    if digest != expected_sha256:
        raise RuntimeError("Sample archive checksum mismatch (or unmaterialized Git LFS pointer). Restore the release archive with git lfs pull; do not calibrate this unverified copy.")


def safe_extract(zip_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    resolved_destination = destination.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if resolved_destination not in (target, *target.parents):
                raise RuntimeError(f"Refusing to extract path outside destination: {member.filename}")
        archive.extractall(destination)


def find_sample_archive(deepstream_repo: Path, amc_root: Path) -> Path:
    candidates = [
        amc_root / "assets" / SAMPLE_ZIP,
        deepstream_repo / "data" / "amc" / SAMPLE_ZIP,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise SystemExit(
        f"Missing {SAMPLE_ZIP} under the tracked AMC directory. Restore or update the "
        "DeepStream checkout. If the archive is still missing, fetch it from "
        "https://github.com/NVIDIA-AI-IOT/auto-magic-calib into $DEEPSTREAM_REPO/data/amc/."
    )


def find_one(root: Path, filename: str) -> Path:
    hits = sorted(root.rglob(filename))
    if not hits:
        raise SystemExit(f"Could not find {filename} under {root}")
    return hits[0]


def find_videos(sample_dir: Path) -> list[Path]:
    video_dirs = [path for path in sample_dir.rglob("videos") if path.is_dir()]
    if video_dirs:
        videos = sorted(video_dirs[0].glob("cam_*.mp4"))
    else:
        videos = sorted(sample_dir.rglob("cam_*.mp4"))
    if len(videos) < 2:
        raise SystemExit(f"Need at least two cam_*.mp4 files under {sample_dir}; found {len(videos)}")
    if len(videos) > 16:
        raise SystemExit(
            f"Found {len(videos)} cam_*.mp4 files under {sample_dir}; SAMPLE_DIR is probably too broad."
        )
    return videos


def main() -> None:
    requests = require_requests()
    is_interactive = sys.stdin.isatty()
    deepstream_repo = infer_deepstream_repo()
    amc_root = infer_amc_root(deepstream_repo)
    base_url = default_base_url(amc_root)
    run_vggt = env_bool("RUN_VGGT", True)

    sample_archive = require_existing_file(find_sample_archive(deepstream_repo, amc_root), "sample archive")
    verify_sample_archive(sample_archive)
    sample_dir = Path(os.environ.get("SAMPLE_DIR", deepstream_repo / "data" / "amc" / "sample" / SAMPLE_NAME))
    if not sample_dir.exists() or not any(sample_dir.iterdir()):
        print(f"Extracting {sample_archive} -> {sample_dir}")
        safe_extract(sample_archive, sample_dir)

    videos = find_videos(sample_dir)
    alignment = find_one(sample_dir, "alignment_data.json")
    layout = find_one(sample_dir, "layout.png")
    gt_zip = find_one(sample_dir, "GT.zip")

    print(f"Base URL:   {base_url}")
    print(f"Sample dir: {sample_dir}")
    print(f"Videos:     {[video.name for video in videos]}")

    session = requests.Session()
    check_ready(session, base_url)
    print("[0] AMC microservice is ready")

    project_name = os.environ.get("PROJECT_NAME", f"sample_test_{int(time.time())}")
    project_id = create_project(session, base_url, project_name)
    print(f"[1] Created project {project_name} -> {project_id}")

    upload_videos(session, base_url, project_id, videos)
    print(f"[2] Uploaded {len(videos)} videos")

    stage_linear_media(session, base_url, project_id, is_interactive=is_interactive)

    # The bundled file was authored against the original uploaded videos.
    coord_space = "original"
    upload_json_file(session, f"{base_url}/upload_alignment/{project_id}?coord_space={coord_space}", alignment, "alignment_file")
    print("[3] Uploaded alignment JSON")

    upload_binary_file(session, f"{base_url}/upload_layout/{project_id}", layout, "layout_file", "image/png")
    print("[4] Uploaded layout PNG")

    upload_binary_file(session, f"{base_url}/upload_gt_file/{project_id}", gt_zip, "gt_file", "application/zip")
    print("[5] Uploaded GT zip")

    verify_project(session, base_url, project_id)
    print("[6] verify_project -> READY")

    # VGGT is independent of AMC and runs first by default. Its post-process
    # pass is performed by maybe_run_vggt before AMC resets that state.
    vggt_completed = maybe_run_vggt(session, base_url, project_id, run_vggt, is_interactive)

    response = session.post(f"{base_url}/calibrate/{project_id}", json={"detector_type": "resnet"}, timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()
    print("[7] Calibration started (detector=resnet)")

    print("[8] Polling calibration (expect 10-30 min)")
    poll_project_completion(session, base_url, project_id, timeout_seconds=3600)
    run_postprocess(session, base_url, project_id)

    response = session.get(f"{base_url}/result/{project_id}/evaluation_statistics", timeout=DEFAULT_TIMEOUT)
    if response.status_code == 200:
        print("\n[9] Evaluation statistics:")
        for key, value in (response.json().get("statistics") or response.json()).items():
            print(f"    {key}: {value}")
    else:
        print(f"\n[9] evaluation_statistics returned HTTP {response.status_code}: {response.text[:200]}")

    if vggt_completed:
        print(
            "\nCompare the VGGT and AMC metrics plus Results-page overlays, then "
            "select the more accurate calibration for export."
        )

    output_dir = amc_root / "projects" / f"project_{project_id}" / "output"
    print(f"\nProject ID: {project_id}")
    print(f"Output dir: {output_dir}")
    print("Final camera parameters: output/multi_view_results/BA_output/results_ba/refined/camInfo_XX.yaml")


if __name__ == "__main__":
    main()
