# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the AutoMagicCalib skill scripts.

Usage: import amc_common from a calibration entry point; not a standalone CLI.
Arguments: explicit helper arguments and named environment inputs in callers.
Output: validated paths/settings, API responses, and redacted progress messages.
Exit codes: none for this library; failures raise exceptions or SystemExit(1)
    for the calling entry point to report.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

DEFAULT_TIMEOUT = (10, 120)
VALID_DETECTORS = {"resnet", "transformer"}


def require_requests() -> Any:
    try:
        import requests
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "The AMC helper scripts require the 'requests' Python package. "
            "Install it in an activated environment, for example: python3 -m pip install requests"
        ) from exc
    return requests


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else None


def timeout_from_env(name: str, default: int) -> int:
    """Read a positive stage timeout without permitting an unbounded poll."""
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError as exc:
        raise SystemExit(f"{name} must be a positive integer in seconds.") from exc
    if value <= 0:
        raise SystemExit(f"{name} must be a positive integer in seconds.")
    return value


def env_float_list(name: str) -> list[float] | None:
    value = os.environ.get(name)
    if not value:
        return None
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def csv_env(name: str) -> list[str]:
    value = os.environ.get(name, "")
    return [item.strip() for item in value.split(",") if item.strip()]


def infer_deepstream_repo() -> Path:
    explicit = os.environ.get("DEEPSTREAM_REPO")
    if explicit:
        return Path(explicit).expanduser().resolve()
    return Path(__file__).resolve().parents[3]


def infer_amc_root(deepstream_repo: Path | None = None) -> Path:
    explicit = os.environ.get("AMC_ROOT")
    if explicit:
        return Path(explicit).expanduser().resolve()
    repo = deepstream_repo or infer_deepstream_repo()
    return repo / "tools" / "auto-magic-calib"


def normalize_base_url(value: str) -> str:
    base = value.rstrip("/")
    return base if base.endswith("/v1") else f"{base}/v1"


def runtime_config_value(amc_root: Path, key: str) -> str | None:
    config_path = Path(
        os.environ.get("AMC_RUNTIME_CONFIG", amc_root.parents[1] / "data" / "amc" / "amc-runtime.conf")
    ).expanduser()
    try:
        for line in config_path.read_text(encoding="utf-8").splitlines():
            if line.startswith(f"{key}="):
                return line.split("=", 1)[1].strip()
    except FileNotFoundError:
        return None
    return None


def default_base_url(amc_root: Path) -> str:
    explicit = os.environ.get("BASE_URL")
    if explicit:
        return normalize_base_url(explicit)
    port = os.environ.get("MS_PORT") or os.environ.get("AUTO_MAGIC_CALIB_MS_PORT") or runtime_config_value(
        amc_root, "AUTO_MAGIC_CALIB_MS_PORT"
    )
    if not port:
        raise RuntimeError(
            "Set BASE_URL or AUTO_MAGIC_CALIB_MS_PORT, or run the AMC setup workflow to create "
            "the non-secret runtime configuration."
        )
    return normalize_base_url(f"http://localhost:{port}/v1")


def require_existing_file(path: Path, label: str) -> Path:
    if not path.exists() or not path.is_file():
        raise SystemExit(f"{label} not found: {path}")
    return path


def require_existing_dir(path: Path, label: str) -> Path:
    if not path.exists() or not path.is_dir():
        raise SystemExit(f"{label} not found or not a directory: {path}")
    return path


def resolve_local(override: Path | None, candidate_names: list[str], scan_dirs: list[Path], label: str) -> Path | None:
    """Use an explicit asset path or resolve one unambiguous local candidate."""
    if override:
        if override.exists():
            return override
        raise SystemExit(f"{label} path was provided but does not exist: {override}")

    hits: list[Path] = []
    for directory in scan_dirs:
        for name in candidate_names:
            candidate = directory / name
            if candidate.exists() and candidate not in hits:
                hits.append(candidate)
    if len(hits) == 1:
        print(f"    auto-detected {label}: {hits[0]}")
        return hits[0]
    if len(hits) > 1:
        print(f"    multiple {label} candidates found: {hits}; set an explicit path or use UI fallback")
    else:
        print(f"    no {label} file auto-detected; set an explicit path or use UI fallback")
    return None


def redact_rtsp_url(url: str) -> str:
    try:
        parts = urlsplit(url)
        if parts.scheme.lower() not in {"rtsp", "rtsps"}:
            return url
        host = parts.hostname or ""
        if ":" in host:
            host = f"[{host}]"
        port = f":{parts.port}" if parts.port else ""
        userinfo = "<redacted>@" if "@" in parts.netloc else ""
        query = "<redacted>" if parts.query else ""
        fragment = "<redacted>" if parts.fragment else ""
        return urlunsplit((parts.scheme, f"{userinfo}{host}{port}", parts.path, query, fragment))
    except Exception:
        return "<redacted-rtsp-url>"


def redact_value(value: Any) -> Any:
    if isinstance(value, str):
        text = re.sub(r"rtsps?://[^\s\"'<>]+", lambda match: redact_rtsp_url(match.group()), value, flags=re.I)
        for name in ("VIOS_TOKEN", "HF_TOKEN", "NGC_API_KEY", "NGC_CLI_API_KEY"):
            secret = os.environ.get(name)
            if secret:
                text = text.replace(secret, "<redacted>")
        return text
    if isinstance(value, list):
        return [redact_value(item) for item in value]
    if isinstance(value, dict):
        return {
            key: "<redacted>" if str(key).lower() in {"password", "token", "vios_token", "api_key", "authorization"} else redact_value(item)
            for key, item in value.items()
        }
    return value


def detector_from_config(path: Path | None) -> str | None:
    if not path or not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    detector = data.get("detector") or data.get("detector_type")
    return detector if detector in VALID_DETECTORS else None


def require_detector(current: str | None, is_interactive: bool) -> str:
    if current in VALID_DETECTORS:
        return current
    if not is_interactive:
        raise RuntimeError(
            "Detector type is required before calibration. Set DETECTOR_TYPE=resnet or "
            "DETECTOR_TYPE=transformer, or provide CONFIG_FILE with detector/detector_type."
        )
    while True:
        answer = input("    Detector type for calibration [resnet/transformer]: ").strip().lower()
        if answer in VALID_DETECTORS:
            return answer
        print("    Enter 'resnet' or 'transformer'.")


def require_alignment_coord_space(is_interactive: bool) -> str:
    coord_space = os.environ.get("ALIGNMENT_COORD_SPACE", "").strip().lower()
    if coord_space in {"original", "rectified"}:
        return coord_space
    if not is_interactive:
        raise RuntimeError(
            "Set ALIGNMENT_COORD_SPACE=original for an external alignment file made on the "
            "uploaded source videos, or ALIGNMENT_COORD_SPACE=rectified only for points made "
            "on AMC's rectified media."
        )
    while True:
        answer = input("    Alignment coordinate space [rectified/original]: ").strip().lower()
        if answer in {"original", "rectified"}:
            return answer
        print("    Enter 'rectified' or 'original'.")


def create_project(session: Any, base_url: str, project_name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]{3,50}", project_name):
        raise SystemExit("PROJECT_NAME must be 3-50 characters: letters, digits, underscores, or hyphens.")
    response = session.post(f"{base_url}/create_project", data={"project_name": project_name}, timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()
    payload = response.json()
    project_id = payload.get("project_id")
    if not project_id:
        raise RuntimeError(f"create_project did not return project_id: {payload}")
    return str(project_id)


def check_ready(session: Any, base_url: str) -> None:
    response = session.get(f"{base_url}/ready", timeout=10)
    response.raise_for_status()


def verify_project(session: Any, base_url: str, project_id: str) -> str:
    response = session.post(f"{base_url}/verify_project/{project_id}", timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()
    payload = response.json()
    state = payload.get("project_state") or payload.get("project_info", {}).get("project_state")
    if state != "READY":
        raise RuntimeError(f"Expected READY from verify_project, got {state}: {payload}")
    return str(state)


def stage_linear_media(
    session: Any,
    base_url: str,
    project_id: str,
    *,
    is_interactive: bool = False,
    ui_url: str | None = None,
) -> None:
    """Ensure the project has completed either linear staging or rectification."""
    state = project_info(session, base_url, project_id).get("rectification_state")
    if state == "COMPLETED":
        print("    rectified media already staged")
        return

    mode = os.environ.get("MEDIA_MODE", "").strip().lower()
    if mode == "linear":
        response = session.post(f"{base_url}/linear_media/{project_id}", timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
        state = response.json().get("rectification_state")
        if state != "COMPLETED":
            raise RuntimeError(f"Expected staged linear media, got rectification_state={state}")
        print("    staged already-linear media")
        return

    if mode == "rectified" and is_interactive:
        destination = f" at {ui_url}" if ui_url else ""
        print(f"    Complete Rectification for project {project_id}{destination}, then return here.")
        input("    Press Enter after review/commit completes...")
        state = project_info(session, base_url, project_id).get("rectification_state")
        if state == "COMPLETED":
            print("    rectification completed")
            return
        raise RuntimeError(f"Expected completed rectification, got rectification_state={state}")

    if mode != "rectified":
        raise RuntimeError(
            "Set MEDIA_MODE=linear only after confirming every input is already linear, or set "
            "MEDIA_MODE=rectified for an interactive run after completing AMC Rectification."
        )
    raise RuntimeError(
        "MEDIA_MODE=rectified requires an interactive run so AMC Rectification can be reviewed "
        "and committed for this project."
    )


def project_info(session: Any, base_url: str, project_id: str) -> dict[str, Any]:
    response = session.get(f"{base_url}/get_project_info/{project_id}", timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()
    return response.json().get("project_info", response.json())


def print_log_tail(session: Any, base_url: str, project_id: str, prefix: str = "    ") -> None:
    try:
        response = session.get(f"{base_url}/amc/calibrate/{project_id}/log", timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
        lines = response.text.splitlines()
    except Exception:
        return
    print(f"{prefix}--- last calibration log lines ---")
    for line in lines[-20:]:
        print(f"{prefix}{redact_value(line)}")


def poll_project_completion(session: Any, base_url: str, project_id: str, timeout_seconds: int, state_key: str = "amc_state") -> None:
    start = time.time()
    last_state = ""
    last_beat = 0.0
    while time.time() - start < timeout_seconds:
        info = project_info(session, base_url, project_id)
        state = info.get(state_key)
        mins, secs = divmod(int(time.time() - start), 60)
        if state != last_state or time.time() - last_beat >= 60:
            print(f"    [{mins:>3}m {secs:02d}s] {state}", flush=True)
            last_state = str(state)
            last_beat = time.time()
        if state == "COMPLETED":
            print(f"    completed in {mins}m {secs:02d}s")
            return
        if state == "ERROR":
            if state_key == "amc_state":
                print_log_tail(session, base_url, project_id)
                diagnostics = f"See GET {base_url}/amc/calibrate/{project_id}/log"
            elif state_key == "postprocess_state":
                diagnostics = f"Inspect the AMC-host project_{project_id}/postprocess.log"
            else:
                diagnostics = "Inspect the project logs on the AMC host"
            raise RuntimeError(f"{state_key} failed for project {project_id}. {diagnostics}")
        time.sleep(10)
    raise RuntimeError(f"Calibration still running after {timeout_seconds // 60} min")


def run_postprocess(session: Any, base_url: str, project_id: str) -> bool:
    info = project_info(session, base_url, project_id)
    video_count = int(info.get("video_files_count") or len(info.get("video_files") or []))
    if video_count <= 1:
        print("Skipping layout post-processing: it is available only for multi-camera projects")
        return False
    timeout_seconds = timeout_from_env("POSTPROCESS_TIMEOUT_SECONDS", 1800)
    response = session.post(f"{base_url}/postprocess/{project_id}", timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()
    print("Layout post-processing started")
    poll_project_completion(session, base_url, project_id, timeout_seconds=timeout_seconds, state_key="postprocess_state")
    return True


def maybe_run_vggt(session: Any, base_url: str, project_id: str, run_vggt: bool, is_interactive: bool) -> bool:
    info = project_info(session, base_url, project_id)
    state = info.get("vggt_state", "INIT")
    video_count = int(info.get("video_files_count") or len(info.get("video_files") or []))
    if video_count <= 1:
        print("\nVGGT skipped: it is available only for multi-camera projects")
        return False
    if state == "MODEL_MISSING":
        print("\nVGGT unavailable: the server has no VGGT model installed. Continuing with AMC calibration.")
        return False
    if state == "ERROR":
        print("\nVGGT previously failed. Continuing with AMC calibration; inspect the project logs if needed.")
        return False
    if state == "READY":
        if not run_vggt:
            print("\nVGGT is READY but disabled by RUN_VGGT=false; continuing with AMC-only calibration.")
            return False
        timeout_seconds = timeout_from_env("VGGT_TIMEOUT_SECONDS", 900)
        response = session.post(f"{base_url}/vggt/calibrate/{project_id}", timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
        print("\nIndependent VGGT calibration started")
        try:
            poll_project_completion(session, base_url, project_id, timeout_seconds=timeout_seconds, state_key="vggt_state")
        except RuntimeError:
            if project_info(session, base_url, project_id).get("vggt_state") != "ERROR":
                raise
            print(
                "\nIndependent VGGT calibration failed. Continuing with AMC calibration; "
                "inspect the project logs if needed."
            )
            return False
        info = project_info(session, base_url, project_id)
    elif state == "RUNNING":
        timeout_seconds = timeout_from_env("VGGT_TIMEOUT_SECONDS", 900)
        print("\nIndependent VGGT calibration already running; waiting for completion")
        try:
            poll_project_completion(session, base_url, project_id, timeout_seconds=timeout_seconds, state_key="vggt_state")
        except RuntimeError:
            if project_info(session, base_url, project_id).get("vggt_state") != "ERROR":
                raise
            print(
                "\nIndependent VGGT calibration failed. Continuing with AMC calibration; "
                "inspect the project logs if needed."
            )
            return False
        info = project_info(session, base_url, project_id)
    elif state != "COMPLETED":
        print(f"\nVGGT skipped: project vggt_state={state}")
        return False

    # Each calibration run resets postprocess_state. Always complete VGGT's
    # post-process pass before AMC starts.
    run_postprocess(session, base_url, project_id)
    stats = session.get(f"{base_url}/vggt_results/{project_id}/evaluation_statistics", timeout=DEFAULT_TIMEOUT)
    if stats.status_code == 200:
        print("VGGT evaluation statistics:")
        for key, value in (stats.json().get("statistics") or stats.json()).items():
            print(f"    {key}: {value}")
    return True


def upload_videos(session: Any, base_url: str, project_id: str, videos: list[Path]) -> None:
    if not videos:
        raise SystemExit("Provide at least one video file.")
    handles = []
    files = []
    try:
        for video in videos:
            handle = video.open("rb")
            handles.append(handle)
            files.append(("files", (video.name, handle, "video/mp4")))
        response = session.post(f"{base_url}/upload_video_files/{project_id}", files=files, timeout=300)
        response.raise_for_status()
    finally:
        for handle in handles:
            handle.close()


def upload_json_file(session: Any, url: str, path: Path, field_name: str) -> None:
    with path.open("rb") as handle:
        response = session.post(url, files={field_name: (path.name, handle, "application/json")}, timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()


def upload_binary_file(session: Any, url: str, path: Path, field_name: str, content_type: str, timeout: int = 120) -> None:
    with path.open("rb") as handle:
        response = session.post(url, files={field_name: (path.name, handle, content_type)}, timeout=timeout)
    response.raise_for_status()
