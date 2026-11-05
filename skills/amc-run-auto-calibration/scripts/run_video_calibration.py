#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run AMC calibration for user-provided pre-recorded MP4 files.

Usage: python3 run_video_calibration.py
Arguments: environment inputs VIDEO_DIR, DETECTOR_TYPE, MEDIA_MODE, plus service
    URL and optional settings/alignment/layout paths documented in video-files.md.
Output: project ID, terminal states, metrics, and server-side artifact location.
Exit codes: 0 on success; 1 on validation, API, or unhandled execution failure.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from amc_common import (
    DEFAULT_TIMEOUT,
    check_ready,
    create_project,
    default_base_url,
    detector_from_config,
    env_bool,
    env_float_list,
    env_path,
    infer_amc_root,
    infer_deepstream_repo,
    maybe_run_vggt,
    poll_project_completion,
    require_detector,
    require_existing_dir,
    require_alignment_coord_space,
    require_requests,
    resolve_local,
    run_postprocess,
    stage_linear_media,
    upload_binary_file,
    upload_json_file,
    upload_videos,
    verify_project,
)


def verify_manual_alignment(projects_dir: Path, project_id: str) -> None:
    manual_dir = projects_dir / f"project_{project_id}" / "manual_adjustment"
    alignment = manual_dir / "alignment_data.json"
    layout = manual_dir / "layout.png"
    if not alignment.exists() or not layout.exists():
        raise RuntimeError(f"Alignment files missing under {manual_dir}. Re-check UI Step 5 and click Save.")
    print(f"    Alignment files verified at {manual_dir}")


def discover_video_files(video_dir: Path) -> list[Path]:
    """Return readable MP4 inputs in deterministic camera-index order."""
    video_files = sorted(
        path for path in video_dir.glob("*.mp4") if path.is_file() and os.access(path, os.R_OK)
    )
    if not video_files:
        raise SystemExit(f"No readable *.mp4 files found under {video_dir}")
    return video_files


def main() -> None:
    requests = require_requests()
    deepstream_repo = infer_deepstream_repo()
    amc_root = infer_amc_root(deepstream_repo)
    base_url = default_base_url(amc_root)
    is_interactive = sys.stdin.isatty()
    assume_yes = env_bool("ASSUME_YES", False)

    video_dir_value = os.environ.get("VIDEO_DIR")
    if not video_dir_value:
        raise SystemExit("Set VIDEO_DIR to a directory containing MP4 camera files.")
    video_dir = require_existing_dir(Path(video_dir_value).expanduser(), "VIDEO_DIR")
    video_files = discover_video_files(video_dir)

    first_level_dirs = sorted((path for path in video_dir.iterdir() if path.is_dir()), key=lambda path: path.name)
    scan_dirs = [video_dir, *first_level_dirs, video_dir.parent]

    config_file = resolve_local(env_path("CONFIG_FILE"), ["settings.json", "config.json", "calibration_config.json"], scan_dirs, "settings")
    alignment_json = resolve_local(env_path("ALIGNMENT_JSON"), ["alignment_data.json"], scan_dirs, "alignment")
    layout_png = resolve_local(env_path("LAYOUT_PNG"), ["layout.png"], scan_dirs, "layout")
    gt_zip = resolve_local(env_path("GT_ZIP"), ["GT.zip", "gt.zip"], scan_dirs, "ground truth")

    focal_lengths = env_float_list("FOCAL_LENGTHS")
    detector_type = os.environ.get("DETECTOR_TYPE") or detector_from_config(config_file)
    run_vggt = env_bool("RUN_VGGT", True)
    projects_dir = Path(os.environ.get("PROJECTS_DIR", amc_root / "projects"))
    project_name = os.environ.get("PROJECT_NAME", "video_calibration_run")

    session = requests.Session()
    check_ready(session, base_url)
    print("[0] AMC microservice is ready")

    project_id = create_project(session, base_url, project_name)
    print(f"[1] Created project: {project_id}")

    upload_videos(session, base_url, project_id, video_files)
    print(f"[2] Uploaded {len(video_files)} videos")

    stage_linear_media(session, base_url, project_id, is_interactive=is_interactive)

    if config_file:
        response = session.post(
            f"{base_url}/config/{project_id}",
            data=config_file.read_bytes(),
            headers={"Content-Type": "application/json"},
            timeout=DEFAULT_TIMEOUT,
        )
        response.raise_for_status()
        print(f"[3] Applied calibration settings: {config_file}")
        detector_type = os.environ.get("DETECTOR_TYPE") or detector_from_config(config_file) or detector_type

    if alignment_json:
        coord_space = require_alignment_coord_space(is_interactive)
        upload_json_file(session, f"{base_url}/upload_alignment/{project_id}?coord_space={coord_space}", alignment_json, "alignment_file")
        print(f"[3] Uploaded alignment: {alignment_json}")
    if layout_png:
        upload_binary_file(session, f"{base_url}/upload_layout/{project_id}", layout_png, "layout_file", "image/png")
        print(f"[3] Uploaded layout: {layout_png}")
    if gt_zip:
        upload_binary_file(session, f"{base_url}/upload_gt_file/{project_id}", gt_zip, "gt_file", "application/zip")
        print(f"[3] Uploaded GT zip: {gt_zip}")
    if focal_lengths:
        response = session.post(f"{base_url}/upload_focal_length/{project_id}", data={"focal_length": focal_lengths}, timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
        print(f"[3] Uploaded focal lengths: {focal_lengths}")

    ui_tasks = []
    if not config_file:
        ui_tasks.append("Step 3 (Parameters): tune settings or accept defaults, then Save.")
    if not alignment_json or not layout_png:
        ui_tasks.append("Step 5 (Manual Alignment): upload layout, mark correspondence points, then Save.")
    if ui_tasks:
        print(f"\n[4] UI action required for project {project_id}:")
        for task in ui_tasks:
            print(f"    - {task}")
        if not is_interactive:
            raise RuntimeError(
                "UI action is required before continuing. Run interactively, or provide "
                "CONFIG_FILE, ALIGNMENT_JSON, and LAYOUT_PNG for unattended execution."
            )
        input("    Press Enter when the UI steps are complete...")
        if not alignment_json or not layout_png:
            verify_manual_alignment(projects_dir, project_id)

    verify_project(session, base_url, project_id)
    print("[5] verify_project -> READY")

    vggt_completed = maybe_run_vggt(session, base_url, project_id, run_vggt, is_interactive)

    detector_type = require_detector(detector_type, is_interactive)
    print("\n[6] Calibration plan:")
    print(f"    Detector:             {detector_type}")
    print(f"    Calibration settings: {config_file if config_file else 'UI Step 3 settings/defaults'}")
    print(f"    Alignment JSON:       {alignment_json if alignment_json else 'UI Step 5/manual_adjustment'}")
    print(f"    Layout PNG:           {layout_png if layout_png else 'UI Step 5/manual_adjustment'}")
    print(f"    Ground truth zip:     {gt_zip if gt_zip else 'not provided'}")
    print(f"    Focal lengths:        {focal_lengths if focal_lengths else 'not provided'}")
    if is_interactive:
        answer = input("    Start calibration? [y/N]: ").strip().lower()
        if answer not in {"y", "yes"}:
            raise SystemExit("Stopped before calibration.")
    elif not assume_yes:
        raise SystemExit("Set ASSUME_YES=true to start calibration in non-interactive mode.")

    response = session.post(f"{base_url}/calibrate/{project_id}", json={"detector_type": detector_type}, timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()
    print(f"[6] Calibration started (detector={detector_type})")

    print("[7] Polling calibration (10-60 min typical)")
    poll_project_completion(session, base_url, project_id, timeout_seconds=5400)
    run_postprocess(session, base_url, project_id)

    response = session.get(f"{base_url}/result/{project_id}/evaluation_statistics", timeout=DEFAULT_TIMEOUT)
    if response.status_code == 200:
        print("\n[8] Evaluation statistics:")
        for key, value in (response.json().get("statistics") or response.json()).items():
            print(f"    {key}: {value}")
    else:
        print("\n[8] No GT provided or evaluation statistics unavailable")

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
