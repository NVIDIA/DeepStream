#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run AMC calibration from live RTSP streams captured through VIOS.

Usage: python3 run_rtsp_calibration.py
Arguments: environment inputs RTSP_URLS or STREAMS_JSON, VIOS_BASE_URL,
    DETECTOR_TYPE, MEDIA_MODE, DURATION_SECONDS; optional VIOS_TOKEN and
    SSL_VERIFY (true by default). See rtsp.md for asset and service inputs.
Output: redacted stream summary, project/session IDs, states, and artifact path.
Exit codes: 0 on success; 1 on validation, API, or unhandled execution failure.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from amc_common import (
    DEFAULT_TIMEOUT,
    check_ready,
    create_project,
    csv_env,
    default_base_url,
    detector_from_config,
    env_bool,
    env_float_list,
    env_path,
    infer_amc_root,
    infer_deepstream_repo,
    maybe_run_vggt,
    poll_project_completion,
    redact_rtsp_url,
    redact_value,
    require_detector,
    require_alignment_coord_space,
    require_requests,
    resolve_local,
    run_postprocess,
    stage_linear_media,
    upload_binary_file,
    upload_json_file,
    verify_project,
)


def capture_ssl_verify() -> bool:
    """Verify VIOS certificates unless the caller explicitly opts out."""
    verify = env_bool("SSL_VERIFY", True)
    if not verify:
        if os.environ.get("AMC_ALLOW_INSECURE") != "1":
            raise SystemExit("SSL_VERIFY=false requires AMC_ALLOW_INSECURE=1 after explicit isolated-test-network approval; prefer installing the VIOS CA on the AMC backend.")
        print("WARNING: SSL_VERIFY=false disables VIOS certificate verification; use only on a trusted isolated test network.", file=sys.stderr)
    return verify


def parse_streams() -> list[dict[str, str | None]]:
    streams_json = os.environ.get("STREAMS_JSON")
    if streams_json:
        streams = json.loads(streams_json)
        if not isinstance(streams, list):
            raise SystemExit("STREAMS_JSON must be a JSON list of stream objects.")
    else:
        urls = csv_env("RTSP_URLS")
        if not urls:
            raise SystemExit("Set RTSP_URLS or STREAMS_JSON before running.")
        names = csv_env("CAMERA_NAMES") or [f"cam_{index:02d}" for index in range(len(urls))]
        sensor_ids = csv_env("SENSOR_IDS")
        if len(names) != len(urls):
            raise SystemExit("CAMERA_NAMES must have the same count as RTSP_URLS.")
        if sensor_ids and len(sensor_ids) != len(urls):
            raise SystemExit("SENSOR_IDS must have the same count as RTSP_URLS when set.")
        streams = [
            {"rtsp_url": url, "camera_name": names[index], "sensor_id": sensor_ids[index] if sensor_ids else None}
            for index, url in enumerate(urls)
        ]

    normalized = []
    for index, stream in enumerate(streams):
        if not isinstance(stream, dict):
            raise SystemExit(f"Stream entry {index} must be an object.")
        rtsp_url = stream.get("rtsp_url")
        camera_name = stream.get("camera_name") or f"cam_{index:02d}"
        if not rtsp_url or not str(rtsp_url).startswith("rtsp://"):
            raise SystemExit(f"Stream entry {index} must include an rtsp:// URL.")
        normalized.append({"rtsp_url": str(rtsp_url), "camera_name": str(camera_name), "sensor_id": stream.get("sensor_id")})
    return normalized


def check_vios_ready(requests: Any, base_url: str) -> None:
    response = requests.get(f"{base_url.rstrip('/')}/vst/api/v1/sensor/list", timeout=10)
    response.raise_for_status()


def default_ui_url(base_url: str) -> str:
    explicit = os.environ.get("AMC_UI_URL") or os.environ.get("UI_URL")
    if explicit:
        return explicit.rstrip("/")
    parts = urlsplit(base_url.rstrip("/"))
    host = parts.hostname or "<HOST_IP>"
    if ":" in host:
        host = f"[{host}]"
    scheme = parts.scheme or "http"
    ui_port = os.environ.get("AUTO_MAGIC_CALIB_UI_PORT", "").strip() or "5000"
    if not ui_port.isascii() or not ui_port.isdigit() or not 1 <= int(ui_port) <= 65535:
        raise SystemExit("AUTO_MAGIC_CALIB_UI_PORT must be an integer from 1 to 65535.")
    return f"{scheme}://{host}:{int(ui_port)}"


def json_response(response: Any, label: str) -> dict[str, Any]:
    response.raise_for_status()
    payload = response.json()
    code = payload.get("code")
    if code not in (None, 0, "0"):
        raise RuntimeError(f"{label} returned code={code}: {redact_value(payload)}")
    return payload


def validate_capture_duration(duration_seconds: int) -> int:
    if not 60 <= duration_seconds <= 1200:
        raise SystemExit("DURATION_SECONDS must be between 60 and 1200 seconds.")
    return duration_seconds


def session_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    session = payload.get("session") or payload
    if "session_id" not in session:
        raise RuntimeError(f"RTSP capture response did not include session_id: {redact_value(payload)}")
    return session


def verify_manual_alignment(projects_dir: Path, project_id: str) -> None:
    manual_dir = projects_dir / f"project_{project_id}" / "manual_adjustment"
    alignment = manual_dir / "alignment_data.json"
    layout = manual_dir / "layout.png"
    if not alignment.exists() or not layout.exists():
        raise RuntimeError(f"Alignment files missing under {manual_dir}. Re-check UI Step 5 and click Save.")
    print(f"    Alignment files verified at {manual_dir}")


def main() -> None:
    requests = require_requests()
    deepstream_repo = infer_deepstream_repo()
    amc_root = infer_amc_root(deepstream_repo)
    base_url = default_base_url(amc_root)
    is_interactive = sys.stdin.isatty()
    assume_yes = env_bool("ASSUME_YES", False)

    streams = parse_streams()
    duration_seconds = validate_capture_duration(int(os.environ.get("DURATION_SECONDS", "180")))

    config_file = env_path("CONFIG_FILE")
    calib_asset_dir = env_path("CALIB_ASSET_DIR")
    alignment_json = env_path("ALIGNMENT_JSON")
    layout_png = env_path("LAYOUT_PNG")
    gt_zip = env_path("GT_ZIP")
    focal_lengths = env_float_list("FOCAL_LENGTHS")
    detector_type = os.environ.get("DETECTOR_TYPE")
    run_vggt = env_bool("RUN_VGGT", True)
    vios_token = os.environ.get("VIOS_TOKEN") or None
    ssl_verify = capture_ssl_verify()
    vios_base_url = os.environ.get("VIOS_BASE_URL", "").rstrip("/")
    require_vios_health = env_bool("REQUIRE_VIOS_HEALTH", True)
    projects_dir = Path(os.environ.get("PROJECTS_DIR", amc_root / "projects"))
    project_name = os.environ.get("PROJECT_NAME", "rtsp_calibration_run")
    amc_ui_url = default_ui_url(base_url)

    if calib_asset_dir and not calib_asset_dir.is_dir():
        raise SystemExit(f"CALIB_ASSET_DIR must be a directory: {calib_asset_dir}")
    for label, path in (
        ("CONFIG_FILE", config_file),
        ("ALIGNMENT_JSON", alignment_json),
        ("LAYOUT_PNG", layout_png),
        ("GT_ZIP", gt_zip),
    ):
        if path and not path.exists():
            raise SystemExit(f"{label} path not found: {path}")

    config_file = resolve_local(
        config_file,
        ["settings.json", "config.json", "calibration_config.json"],
        [calib_asset_dir] if calib_asset_dir else [],
        "settings",
    )
    scan_dir = calib_asset_dir or (config_file.parent if config_file else None)
    scan_dirs = [scan_dir] if scan_dir else []
    alignment_json = resolve_local(alignment_json, ["alignment_data.json"], scan_dirs, "alignment")
    layout_png = resolve_local(layout_png, ["layout.png"], scan_dirs, "layout")
    gt_zip = resolve_local(gt_zip, ["GT.zip", "gt.zip"], scan_dirs, "ground truth")
    detector_type = os.environ.get("DETECTOR_TYPE") or detector_from_config(config_file) or detector_type

    if not any([config_file, alignment_json, layout_png]):
        print(
            "[0] No local calibration asset source was provided for these RTSP streams.\n"
            "    Provide CALIB_ASSET_DIR or explicit CONFIG_FILE/ALIGNMENT_JSON/LAYOUT_PNG,\n"
            f"    or use AMC UI upload/tuning at {amc_ui_url}.\n"
            "    Do not reuse sample dataset assets unless they are explicitly intended for this RTSP scene."
        )
        if not is_interactive:
            raise SystemExit("Set CALIB_ASSET_DIR or explicit asset paths, or run interactively for UI fallback.")
        answer = input("    Continue and complete settings/alignment in the AMC UI after capture? [y/N]: ").strip().lower()
        if answer not in {"y", "yes"}:
            raise SystemExit("Stopped before capture.")

    session = requests.Session()
    print("[0] Checking AMC microservice")
    check_ready(session, base_url)

    print("[0] Checking VIOS")
    if vios_base_url:
        check_vios_ready(requests, vios_base_url)
        print(f"    VIOS reachable at {vios_base_url}")
    else:
        try:
            check_vios_ready(requests, "http://localhost:30888")
            vios_base_url = "http://localhost:30888"
            print("    VIOS reachable at http://localhost:30888")
        except Exception as exc:
            if require_vios_health:
                raise SystemExit(
                    "VIOS health check failed. Set VIOS_BASE_URL to a reachable VIOS endpoint, "
                    "or set REQUIRE_VIOS_HEALTH=false only after verifying AMC has VIOS_BASE_URL configured."
                ) from exc
            print("    VIOS health check skipped; ensure AMC has VIOS_BASE_URL configured.")

    print("\n[1] RTSP capture plan:")
    print(f"    Project:              {project_name}")
    print(f"    Stream count:         {len(streams)}")
    print(f"    Duration seconds:     {duration_seconds}")
    print(f"    AMC UI:               {amc_ui_url}")
    print(f"    Calibration assets:   {calib_asset_dir if calib_asset_dir else 'explicit files or UI'}")
    print(f"    Calibration settings: {config_file if config_file else 'UI Step 3 settings/defaults'}")
    print(f"    Alignment JSON:       {alignment_json if alignment_json else 'UI Step 5/manual_adjustment'}")
    print(f"    Layout PNG:           {layout_png if layout_png else 'UI Step 5/manual_adjustment'}")
    print(f"    Detector:             {detector_type if detector_type else 'ask before calibration'}")
    for index, stream in enumerate(streams):
        sensor_state = "provided" if stream.get("sensor_id") else "auto"
        print(f"    Stream {index}:           {stream['camera_name']} {sensor_state} {redact_rtsp_url(str(stream['rtsp_url']))}")
    if is_interactive:
        answer = input("    Start RTSP capture? [y/N]: ").strip().lower()
        if answer not in {"y", "yes"}:
            raise SystemExit("Stopped before capture.")
    elif not assume_yes:
        raise SystemExit("Set ASSUME_YES=true to start RTSP capture in non-interactive mode.")

    project_id = create_project(session, base_url, project_name)
    print(f"\n[2] Created project: {project_id}")

    capture_body = {
        "streams": streams,
        "duration_seconds": duration_seconds,
        "vios_token": vios_token,
        "ssl_verify": ssl_verify,
    }
    response = session.post(f"{base_url}/rtsp/capture/{project_id}", json=capture_body, timeout=30)
    payload = json_response(response, "rtsp capture")
    session_info = session_from_payload(payload)
    session_id = session_info["session_id"]
    print(f"[3] Capture session: {session_id}")

    print(f"[4] Polling capture status (timeout {duration_seconds + 600}s)")
    start = time.time()
    last_state = ""
    while time.time() - start < duration_seconds + 600:
        response = session.get(f"{base_url}/rtsp/capture/{project_id}/{session_id}", timeout=30)
        payload = json_response(response, "rtsp capture status")
        capture_session = payload.get("session") or payload
        state = (capture_session.get("status") or capture_session.get("state") or "").upper()
        elapsed = int(time.time() - start)
        if state != last_state:
            print(f"    [{elapsed:>4}s] {state}", flush=True)
            last_state = state
        if state == "INGESTED":
            print("[4] Recorded clips automatically ingested")
            break
        if state == "COMPLETED":
            # v3.3.0-3 reports INGESTED as the only successful terminal state.
            # Continue polling so a transitional legacy COMPLETED value is not
            # reported as a failed capture during a deployment upgrade.
            print("    capture reported legacy COMPLETED; waiting for INGESTED", flush=True)
        if state in {"ERROR", "CANCELLED"}:
            raise RuntimeError(f"Capture {state}: {redact_value(payload)}")
        time.sleep(10)
    else:
        raise RuntimeError(f"RTSP observation timed out in {last_state or 'UNKNOWN'} for project {project_id}, session {session_id}. The session may still be active; inspect AMC/VIOS logs and resume polling these IDs rather than starting another capture.")

    stage_linear_media(session, base_url, project_id, is_interactive=is_interactive, ui_url=amc_ui_url)

    if config_file:
        response = session.post(
            f"{base_url}/config/{project_id}",
            data=config_file.read_bytes(),
            headers={"Content-Type": "application/json"},
            timeout=DEFAULT_TIMEOUT,
        )
        response.raise_for_status()
        print(f"[5] Applied calibration settings: {config_file}")
        detector_type = os.environ.get("DETECTOR_TYPE") or detector_from_config(config_file) or detector_type
    if alignment_json:
        coord_space = require_alignment_coord_space(is_interactive)
        upload_json_file(session, f"{base_url}/upload_alignment/{project_id}?coord_space={coord_space}", alignment_json, "alignment_file")
        print(f"[5] Uploaded alignment: {alignment_json}")
    if layout_png:
        upload_binary_file(session, f"{base_url}/upload_layout/{project_id}", layout_png, "layout_file", "image/png")
        print(f"[5] Uploaded layout: {layout_png}")
    if gt_zip:
        upload_binary_file(session, f"{base_url}/upload_gt_file/{project_id}", gt_zip, "gt_file", "application/zip")
        print(f"[5] Uploaded GT zip: {gt_zip}")
    if focal_lengths:
        response = session.post(f"{base_url}/upload_focal_length/{project_id}", data={"focal_length": focal_lengths}, timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
        print(f"[5] Uploaded focal lengths: {focal_lengths}")

    ui_tasks = []
    if not config_file:
        ui_tasks.append("Step 3 (Parameters): tune settings or accept defaults, then Save.")
    if not alignment_json or not layout_png:
        ui_tasks.append("Step 5 (Manual Alignment): upload layout, mark correspondence points, then Save. Do not re-upload videos.")
    if ui_tasks:
        print(f"\n[5] UI action required for project {project_id}:")
        print(f"    Open AMC UI: {amc_ui_url}")
        for task in ui_tasks:
            print(f"    - {task}")
        if not is_interactive:
            raise RuntimeError("UI action is required before continuing. Provide assets or run interactively.")
        input("    Press Enter when done...")
        if not alignment_json or not layout_png:
            verify_manual_alignment(projects_dir, project_id)

    verify_project(session, base_url, project_id)
    print("[6] verify_project -> READY")
    vggt_completed = maybe_run_vggt(session, base_url, project_id, run_vggt, is_interactive)
    detector_type = require_detector(detector_type, is_interactive)

    print("\n[7] Calibration plan:")
    print(f"    Detector:             {detector_type}")
    print(f"    Stream count:         {len(streams)}")
    print(f"    Duration seconds:     {duration_seconds}")
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
    print(f"[7] Calibration started (detector={detector_type})")

    print("[8] Polling calibration (10-60 min typical)")
    poll_project_completion(session, base_url, project_id, timeout_seconds=5400)
    run_postprocess(session, base_url, project_id)

    response = session.get(f"{base_url}/result/{project_id}/evaluation_statistics", timeout=DEFAULT_TIMEOUT)
    if response.status_code == 200:
        print("\n[9] Evaluation statistics:")
        for key, value in (response.json().get("statistics") or response.json()).items():
            print(f"    {key}: {value}")
    else:
        print("\n[9] No GT provided or evaluation statistics unavailable")

    if vggt_completed:
        print(
            "\nCompare the VGGT and AMC metrics plus Results-page overlays, then "
            "select the more accurate calibration for export."
        )

    output_dir = amc_root / "projects" / f"project_{project_id}" / "output"
    print(f"\nProject ID: {project_id}")
    print(f"Capture session ID: {session_id}")
    print(f"Output dir: {output_dir}")


if __name__ == "__main__":
    main()
