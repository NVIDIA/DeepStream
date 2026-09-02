#!/usr/bin/env python3
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
"""DeepStream Code Agent lab -- the engine behind the Brev launchable notebook.

This module holds ALL the implementation: configuration, the 12-prompt catalog,
every helper function, and the five step entrypoints the notebook calls. The
notebook itself is intentionally thin -- each code cell is a single
``lab.<step>()`` call -- so it SHOWCASES the natural-language prompt + skills ->
generated code flow without ever showing implementation code.

The lab runs a code agent (Claude Code or Codex) INSIDE a long-lived **stock**
DeepStream container, cwd ``/workspace``. The agent CLI is installed into the
running container as a one-time step (no derived image is built). The agent runs
as a NON-ROOT user, generates a DeepStream app from one natural-language prompt,
runs the generated demo, and displays artifacts. Credentials are bring-your-own
and are passed at RUNTIME on every ``docker exec`` (never baked into an image,
never passed at ``docker run``, never printed).

The notebook builds its interactive controls inline in each step cell (see
build_notebook.py) and drives this module through these functions (all also
usable directly from any environment):

  install_agent()                -- install the chosen agent CLI + deps + skills.
  login_start(); login_submit(code)  -- account sign-in (Claude paste-code or Codex device flow).
  check_auth()                   -- verify the agent authenticates (tiny smoke prompt).
  set_selection(agent, prompt_id_or_text)  -- choose agent + prompt.
  generate()                     -- generate the DeepStream app from the prompt.
  run_and_view()                 -- run the generated demo and display artifacts.
  cleanup_containers()           -- stop/remove ds-agent-* containers.
"""

from __future__ import annotations

from pathlib import Path
import contextlib
import json
import os
import re
import select
import shlex
import shutil
import socket
import subprocess
import sys
import textwrap
import time
from urllib.parse import quote, urlparse

from ds_lab_config import *  # config defaults + prompt catalog (ds_lab_config.py)


def klog(msg, kind="step"):
    """Print a KEY-STEP log line in color (ANSI; Jupyter renders it). Raw execution
    output (apt/docker/agent/gstreamer) is left plain so the milestones stand out."""
    c = {
        "step": "\033[1;36m",  # bold cyan  -- a step / phase header
        "ok": "\033[1;32m",  # bold green -- success / ready
        "warn": "\033[1;33m",  # bold yellow
        "err": "\033[1;31m",  # bold red
        "url": "\033[1;35m",
    }.get(kind, "\033[1;36m")  # bold magenta -- action / URL to act on
    print(f"{c}{msg}\033[0m")


def _bool_env(name, default):
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "y")


def derive_custom_selected(text):
    """Build a synthetic catalog entry from a pasted prompt, auto-detecting the scenario.
    Override any guess with NEEDS_* / PREFERRED_ARTIFACT env vars."""
    t = text.lower()
    needs_rtsp = "rtsp" in t
    needs_kafka = "kafka" in t
    needs_nats = "nats" in t
    is_profile = any(
        k in t for k in ("profil", "bottleneck", "nsight", "efficient pipeline", "how many streams")
    )
    needs_vlm = any(k in t for k in ("vlm", "vllm", "multi-modal", "multimodal"))
    needs_model_import = any(
        k in t
        for k in (
            "ultralytics",
            "trtexec",
            " onnx",
            "deepstream-import-vision-model",
            "engine build",
            "benchmark",
        )
    )
    needs_pdf = "pdf" in t and "report" in t
    interactive = "prompt me" in t
    if needs_pdf:
        artifact = "pdf_report"
    elif is_profile:
        artifact = "profile_report"
    elif needs_nats:
        artifact = "nats_json"
    elif needs_kafka:
        artifact = "kafka_text" if needs_vlm else "kafka_json"
    elif "fakesink" in t and ("fps" in t or "count" in t) and "osd" not in t:
        artifact = "logs"
    else:
        artifact = "mp4"
    skill = (
        "deepstream-import-vision-model"
        if "deepstream-import-vision-model" in t
        else "deepstream-profile-pipeline"
        if is_profile
        else "deepstream-dev"
    )
    return {
        "id": "custom",
        "file": "(pasted prompt)",
        "title": "Custom pasted prompt",
        "output_dir": CUSTOM_OUTPUT_DIR,
        "skill": skill,
        "needs_rtsp": _bool_env("NEEDS_RTSP", needs_rtsp),
        "needs_kafka": _bool_env("NEEDS_KAFKA", needs_kafka),
        "needs_nats": _bool_env("NEEDS_NATS", needs_nats),
        "needs_vlm": _bool_env("NEEDS_VLM", needs_vlm),
        "needs_model_import": _bool_env("NEEDS_MODEL_IMPORT", needs_model_import),
        "needs_pdf_report": _bool_env("NEEDS_PDF_REPORT", needs_pdf),
        "interactive": _bool_env("INTERACTIVE", interactive),
        "preferred_artifact": os.environ.get("PREFERRED_ARTIFACT", "").strip()
        or artifact,
        "prompt": text,
    }


def auth_backend_for(agent):
    """Return the active credential backend name (or None) for an agent."""
    if agent == "claude":
        # Treat a partially filled custom endpoint as the endpoint backend too.
        # This prevents an API key (if one was also set in the environment) from
        # masking a missing endpoint token or model during authentication.
        if ANTHROPIC_BASE_URL or ANTHROPIC_AUTH_TOKEN:
            return "anthropic-custom-endpoint"
        if ANTHROPIC_API_KEY:
            return "anthropic-api-key"
    elif agent == "codex":
        if OPENAI_BASE_URL and OPENAI_API_KEY:
            return "openai-custom-endpoint"
        if OPENAI_API_KEY:
            return "openai-api-key"
    return None


# Resolve the initial selection (set_selection() recomputes this in the Generate step).
if CUSTOM_PROMPT:
    selected = derive_custom_selected(CUSTOM_PROMPT)
    SELECTED_PROMPT_ID = "custom"
    catalog_by_id["custom"] = selected
elif SELECTED_PROMPT_ID in catalog_by_id:
    selected = catalog_by_id[SELECTED_PROMPT_ID]
else:
    selected = catalog_by_id[PROMPT_IDS[0]]
    SELECTED_PROMPT_ID = PROMPT_IDS[0]
auth_mode = auth_backend_for(AGENT)

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# ---- Module-level state the dev loop mutates ----
rtsp_urls = []
ctr_prompt_path = None
ctr_run_out = None
# When a combined dropdown option (e.g. "rtvi_vlm_core_app & rtvi_vlm_openapi_spec") is picked,
# this holds the ordered sub-prompt ids; generate() then runs each stage in sequence.
selected_sequence = None

RESULTS_BY_ARTIFACT = {
    "mp4": "embeds & plays the generated out.mp4 (headless filesink output).",
    "kafka_json": "consumes Kafka JSON detection messages and lists them.",
    "kafka_text": "consumes Kafka VLM text summaries and lists them.",
    "pdf_report": "embeds the generated benchmark PDF report inline + a download link.",
    "logs": "shows the demo stdout (buffer counts / FPS) from the run log.",
    "service_code": "curls the FastAPI microservice (advanced) and lists service files.",
    "profile_report": "renders the profiling report (bottleneck, max streams @30 FPS, HW upgrade) + run log.",
    "nats_json": "consumes the NATS subject and lists the published detection messages.",
}


# ---------------------------------------------------------------------------
# Credentials: built fresh on EVERY docker exec (per-exec, runtime only). The
# container starts in the environment/workspace step BEFORE you sign in, so creds are
# NEVER passed at `docker run` -- only here, as -e flags on each exec.
# ---------------------------------------------------------------------------
def build_cred_env():
    """Return the runtime credential `-e` list for the selected agent.

    Auth precedence mirrors auth_backend_for(): a custom Anthropic endpoint
    injects ONLY base URL + auth token (the proxy token wins); otherwise the
    API key. Codex injects the selected API key.
    Codex custom endpoint routing is added by codex_exec_cli() because the
    current CLI requires a model_providers config entry. Keys are never
    printed."""
    env = []
    if AGENT == "claude":
        if ANTHROPIC_BASE_URL or ANTHROPIC_AUTH_TOKEN:
            if ANTHROPIC_BASE_URL:
                env += ["-e", f"ANTHROPIC_BASE_URL={ANTHROPIC_BASE_URL}"]
            if ANTHROPIC_AUTH_TOKEN:
                env += ["-e", f"ANTHROPIC_AUTH_TOKEN={ANTHROPIC_AUTH_TOKEN}"]
        elif ANTHROPIC_API_KEY:
            env += ["-e", f"ANTHROPIC_API_KEY={ANTHROPIC_API_KEY}"]
        # A custom endpoint may only grant specific model ids -- pass them through so
        # Claude Code does not fall back to a default model the key cannot access.
        if ANTHROPIC_MODEL:
            env += ["-e", f"ANTHROPIC_MODEL={ANTHROPIC_MODEL}"]
        if ANTHROPIC_SMALL_FAST_MODEL:
            env += ["-e", f"ANTHROPIC_SMALL_FAST_MODEL={ANTHROPIC_SMALL_FAST_MODEL}"]
    else:
        if OPENAI_API_KEY:
            env += ["-e", f"OPENAI_API_KEY={OPENAI_API_KEY}"]
    return env


def _toml_override(key, value):
    """Return one shell-safe Codex ``-c key=value`` override.

    Codex parses the value as TOML. JSON string encoding is also valid TOML
    basic-string syntax and keeps URLs/models from becoming shell syntax.
    """
    return ["-c", f"{key}={json.dumps(value, ensure_ascii=False)}"]


def codex_exec_cli():
    """Return the Codex CLI prefix used for every non-interactive invocation.

    A third-party OpenAI-compatible endpoint must be registered as a custom
    Codex model provider. Setting OPENAI_BASE_URL alone is not sufficient for
    current Codex releases, so the provider is supplied as ephemeral ``-c``
    overrides and the token remains an injected OPENAI_API_KEY.
    """
    args = ["codex"]
    if auth_backend_for("codex") == "openai-custom-endpoint":
        args += _toml_override("model_provider", "deepstream_custom")
        args += _toml_override("model_providers.deepstream_custom.name", "DeepStream custom endpoint")
        args += _toml_override("model_providers.deepstream_custom.base_url", OPENAI_BASE_URL)
        args += _toml_override("model_providers.deepstream_custom.env_key", "OPENAI_API_KEY")
        # Codex 0.145+ only supports the Responses wire API for custom providers.
        args += _toml_override("model_providers.deepstream_custom.wire_api", "responses")
    if OPENAI_MODEL:
        args += ["-m", OPENAI_MODEL]
    args.append("exec")
    return shlex.join(args)


def dexec_root(inner, **kw):
    """Run a command as root inside the working container (no creds needed)."""
    return subprocess.run(
        ["docker", "exec", "-u", "0", AGENT_CONTAINER, "bash", "-lc", inner],
        text=True,
        **kw,
    )


def dexec(inner, **kw):
    """Run a command as the non-root 'agent' user, injecting creds per-exec.

    Credentials are added as `-e` flags on THIS exec only (never at docker run,
    never baked into an image). All AGENT-facing commands go through here."""
    cmd = [
        "docker",
        "exec",
        "-u",
        "agent",
        *build_cred_env(),
        AGENT_CONTAINER,
        "bash",
        "-lc",
        inner,
    ]
    return subprocess.run(cmd, text=True, **kw)


# Diagnostics from the most recent streamed run (stream_cmd / _stream_agent_json). run_agent reads
# this right after its call to explain WHY a non-zero rc happened: did OUR watchdog fire (timeout),
# or was the process killed/exited by something else (external SIGKILL / host OOM / self-exit)?
_LAST_AGENT_RUN = {}
_LAST_STREAM_TAIL = []


def _compact_text(value, limit=260):
    """Collapse event text to one readable notebook line."""
    text_value = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text_value) > limit:
        return text_value[: limit - 3] + "..."
    return text_value


def _codex_event_summary(line):
    """Return a useful one-line Codex milestone, or None for noisy output.

    Codex ``exec --json`` emits JSONL events. Keep the user-facing stream small
    while the tee'd log retains every event for detailed diagnostics. The plain
    fallback keeps warnings/errors visible if the CLI writes non-JSON text.
    """
    stripped = line.strip()
    if not stripped:
        return None
    try:
        event = json.loads(stripped)
    except (TypeError, ValueError):
        lower = stripped.lower()
        important = (
            "warning",
            "error",
            "fatal",
            "failed",
            "success",
            "succeeded",
            "finished",
            "finalized",
            "traceback",
        )
        if lower.startswith(important) or "traceback" in lower:
            return _compact_text(stripped)
        return None

    if not isinstance(event, dict):
        return None
    event_type = str(event.get("type") or "")
    if event_type == "thread.started":
        return f"Codex session started: {_compact_text(event.get('thread_id'), 80)}"
    if event_type == "turn.started":
        return "Codex turn started"
    if event_type in {"error", "turn.failed"}:
        return "Codex error: " + _compact_text(
            event.get("message") or event.get("error") or event.get("detail")
        )

    item = event.get("item")
    if not isinstance(item, dict):
        if event_type == "turn.completed":
            return "Codex turn completed"
        return None

    item_type = str(item.get("type") or "")
    phase = "completed" if event_type.endswith("completed") else "started"
    if item_type in {"agent_message", "message"}:
        message = item.get("text") or item.get("message") or item.get("content")
        message = _compact_text(message)
        return f"Codex: {message}" if message else None
    if item_type in {"command_execution", "command"}:
        command = item.get("command") or item.get("cmd") or item.get("description")
        status = item.get("status") or event.get("status") or phase
        return f"Codex command {status}: {_compact_text(command, 220)}"
    if item_type in {"file_change", "file_changes"}:
        changes = item.get("changes") or item.get("files") or item.get("path")
        if isinstance(changes, list):
            changes = ", ".join(
                _compact_text(
                    change.get("path") if isinstance(change, dict) else change,
                    80,
                )
                for change in changes[:5]
            )
        return f"Codex files {phase}: {_compact_text(changes, 220)}"
    if item_type in {"mcp_tool_call", "tool_call"}:
        tool = item.get("name") or item.get("tool") or item.get("server")
        return f"Codex tool {phase}: {_compact_text(tool, 180)}"
    return None


def stream_cmd(
    cmd,
    env=None,
    timeout=None,
    quiet=False,
    tail_lines=24,
    ui_flush_interval=None,
    ui_flush_chars=None,
):
    """Run a host command, STREAMING stdout line-by-line through print() so it shows live in
    the notebook Output widget (raw subprocess fd output otherwise bypasses ipywidgets).

    A watchdog kills the child after `timeout` seconds (None = no limit). On Kernel->Interrupt
    the child is killed too, so a user can cleanly abandon a slow step (e.g. a multi-GB VLM
    weight download) instead of reloading the page and leaking the run. Set `quiet=True` for
    verbose agent logs: the process is still drained, only compact milestones are echoed, and the
    last `tail_lines` milestones are retained for failure diagnostics. For verbose but important
    output such as installation, set `ui_flush_interval` and `ui_flush_chars` to batch the
    ipywidgets updates; every line is still forwarded in order, but the browser receives fewer
    larger messages. The full stream is never discarded because the caller tee's it to its
    detailed log. Returns the exit code."""
    global _LAST_STREAM_TAIL
    import threading
    import time

    _LAST_STREAM_TAIL = []
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    fired = {"v": False}  # did OUR watchdog fire?

    def _on_timeout():
        fired["v"] = True
        print(
            f"\n[watchdog] command exceeded the {timeout}s budget -- sending SIGKILL now.",
            flush=True,
        )
        proc.kill()

    timer = threading.Timer(timeout, _on_timeout) if timeout else None
    start = time.monotonic()
    tail = []
    pending_ui = []
    pending_ui_chars = 0
    last_ui_flush = start

    def _append_raw(text):
        active_out = globals().get("_CUR_OUT")
        if active_out is not None and hasattr(active_out, "append_stdout"):
            active_out.append_stdout(text)
        else:
            print(text, end="", flush=True)

    def _flush_raw(force=False):
        nonlocal pending_ui_chars, last_ui_flush
        if not pending_ui:
            return
        now = time.monotonic()
        time_ready = (
            ui_flush_interval is not None
            and now - last_ui_flush >= ui_flush_interval
        )
        size_ready = (
            ui_flush_chars is not None
            and pending_ui_chars >= ui_flush_chars
        )
        if not force and not (time_ready or size_ready):
            return
        _append_raw("".join(pending_ui))
        pending_ui.clear()
        pending_ui_chars = 0
        last_ui_flush = now

    if timer:
        timer.start()
    try:
        for line in proc.stdout:
            if quiet:
                summary = _codex_event_summary(line)
                if summary:
                    tail.append(summary + "\n")
                    del tail[:-tail_lines]
                    active_out = globals().get("_CUR_OUT")
                    if active_out is not None and hasattr(active_out, "append_stdout"):
                        active_out.append_stdout(summary + "\n")
                    else:
                        print(summary, flush=True)
            else:
                if ui_flush_interval is not None or ui_flush_chars is not None:
                    pending_ui.append(line)
                    pending_ui_chars += len(line)
                    _flush_raw()
                else:
                    _append_raw(line)
    except KeyboardInterrupt:
        proc.kill()
        print(
            "\n[interrupted -- subprocess killed; you can pick another prompt and Generate again]"
        )
        raise
    finally:
        if not quiet and (ui_flush_interval is not None or ui_flush_chars is not None):
            _flush_raw(force=True)
        if timer:
            timer.cancel()
        proc.wait()
    _LAST_AGENT_RUN.clear()
    _LAST_AGENT_RUN.update(
        timed_out=fired["v"], elapsed=round(time.monotonic() - start), timeout=timeout
    )
    if quiet:
        _LAST_STREAM_TAIL = tail
    return proc.returncode


def dexec_stream(inner, timeout, quiet=False):
    """Run an agent docker exec, optionally suppressing its verbose stream."""
    cmd = [
        "docker",
        "exec",
        "-u",
        "agent",
        *build_cred_env(),
        AGENT_CONTAINER,
        "bash",
        "-lc",
        inner,
    ]
    return stream_cmd(cmd, timeout=timeout, quiet=quiet)


def _pull(image):
    """docker pull, STREAMED so progress is captured by the active step's `with out:` (and thus
    stays in that cell) rather than leaking to whatever cell is focused."""
    # Pull progress belongs to the prepare/install flow and must remain visible
    # for both agents. Only Generate/Run agent transcripts are compacted.
    rc = stream_cmd(["docker", "pull", image])
    if rc not in (0, None):
        raise SystemExit(f"docker pull {image} failed; see the log above.")


def scenario_plan(item):
    """Return (steps, services) describing what this prompt's scenario exercises."""
    services = []
    if item["needs_rtsp"]:
        services.append("RTSP demo streams (mediamtx + ffmpeg loopers)")
    if item["needs_kafka"]:
        services.append("Kafka broker + topic")
    if item["needs_vlm"]:
        services.append("Local VLM serve (vLLM, advanced)")
    steps = ["Generate code", "Run the generated demo", "Show results"]
    if item["id"] == "rtvi_vlm_openapi_spec":
        steps.insert(2, "Curl the FastAPI microservice (advanced)")
    return steps, services


def set_selection(agent=None, prompt_id_or_text=None):
    """Recompute the global `selected` + `AGENT` for the dev loop.

    `prompt_id_or_text` may be a catalog id (e.g. 'video_infer_app') or pasted prompt
    text (anything not an exact id is treated as pasted, scenario auto-detected)."""
    global AGENT, SELECTED_PROMPT_ID, selected, selected_sequence, auth_mode
    if agent:
        agent = agent.strip().lower()
        if agent not in {"claude", "codex"}:
            raise ValueError("agent must be 'claude' or 'codex'")
        AGENT = agent
    if prompt_id_or_text is not None:
        # A generated result belongs to the prompt that produced it. Invalidate that gate as soon
        # as the user selects or edits another prompt, so Run cannot reuse the previous case.
        state["generated"] = False
        text = prompt_id_or_text.strip()
        if text in PROMPT_SEQUENCES:  # a combined dropdown option -> two-stage sequence
            selected_sequence = list(PROMPT_SEQUENCES[text])
            SELECTED_PROMPT_ID = text
            selected = catalog_by_id[selected_sequence[0]]  # base until generate() cycles the stages
        elif text in catalog_by_id:
            selected_sequence = None
            SELECTED_PROMPT_ID = text
            selected = catalog_by_id[text]
        else:
            selected_sequence = None
            selected = derive_custom_selected(text)
            SELECTED_PROMPT_ID = "custom"
            catalog_by_id["custom"] = selected
    auth_mode = auth_backend_for(AGENT)
    return selected


def select_from_ui(pid, text):
    """Generate-step helper: if the textarea still holds the picked example verbatim, select that
    catalog id (so the run is tracked as that example); if the user edited it, treat the
    text as a custom prompt (scenario auto-detected). Keeps the notebook cell thin."""
    if text.strip() == prompt_text(pid).strip():
        return set_selection(AGENT, pid)
    return set_selection(AGENT, text)


# The RUN-phase contract: how to make a generated app actually run in this headless launchable.
# It is injected ONLY into the Run-phase prompt -- NOT during Generate (Generate stays a pure
# "natural language -> code" step) and NOT as a persistent AGENTS.md (that would be auto-read and
# pollute Generate). All the harness "tweaks" happen here, at run time.
RUN_CONTRACT = """## How this headless launchable runs the app
- It runs HEADLESS (no display): never use EGL / `nveglglessink` / `nv3dsink`. For file output use
  the `nvvideoencfilesinkbin` element -- ONLY this one: it bundles encoder + muxer + filesink and
  handles finalization. Do NOT hand-wire encoder/parser/`mp4mux`/`filesink`. Write an MP4 (e.g.
  `out.mp4`). If the code renders on-screen, switch it to `nvvideoencfilesinkbin`.
- The entrypoint is an executable `run_demo.sh` at the app root. Make it SELF-CONTAINED: do any
  setup (create a venv / install deps / download or export the model / build the TensorRT engine),
  then run the pipeline non-interactively, writing outputs to the artifacts dir below. It MUST
  honor `DEMO_INPUT` (a video path or RTSP URL source) and `DEMO_SECONDS` (a runtime cap) when set.
- Iterate FAST: while debugging, validate with a SHORT run (a few seconds -- just enough to confirm
  frames flow and the pipeline does not crash), NOT a full-length run; do ONE final full
  `DEMO_SECONDS` run only once it works. This keeps the run-and-fix loop quick.
- Cache the TensorRT engine: build it ONCE and REUSE it (`trtexec --saveEngine=...`, or set nvinfer's
  `model-engine-file` so it skips the rebuild when the `.engine` already exists). The engine build is
  the slowest step -- never rebuild it on every run/iteration.
- Keep the artifacts dir CLEAN: write ONLY the final deliverable (e.g. a single `out.mp4`) into the
  artifacts dir below. Put any scratch / test / full-length trial outputs under `/tmp` (or delete
  them) -- do NOT leave extra files like `out_test.mp4` beside the real output, or the result viewer
  picks them up too.
- Finish cleanly so files are valid: send EOS and WAIT for it to reach the sink before quitting so
  `nvvideoencfilesinkbin` finalizes the file (else a 0-byte mp4). FILE source -> run to natural EOS (the
  first run may spend 30-90s building the engine; do not cap that). RTSP/live -> after frames flow,
  stop after about `DEMO_SECONDS`, send EOS, finalize, exit.
- DeepStream gotchas: each `nvinfer` needs a unique `gie-unique-id`; dynamic-shape ONNX needs
  `infer-dims=C;H;W`; tee/dynamic sinks need `async=0`. Use the `deepstream-dev` skill for exact
  API/property details."""


_FORCE_RTSP_IDS = {"multi_stream_tracker"}  # genuinely needs multiple LIVE streams


def _use_rtsp():
    """True only when the prompt genuinely needs live RTSP (multi-stream). Other 'rtsp' prompts
    prefer a FILE input: cleaner EOS (no looping) and no publisher/consumer connect-timing races."""
    return selected["needs_rtsp"] and selected["id"] in _FORCE_RTSP_IDS


def _service_endpoints_ctx():
    """Live service endpoints for the RUN prompt."""
    lines = []
    if _use_rtsp():
        urls = ", ".join(rtsp_urls or [f"{RTSP_BASE}/cam0"])
        lines.append(
            f"- RTSP: an EMPTY mediamtx server is up at {urls} (no publisher yet). Use this PROVEN "
            "order in `run_demo.sh` to avoid the connect-timing flakiness that otherwise needs many "
            f"retries: (1) start one background `ffmpeg` publisher PER path FIRST -- "
            f"`ffmpeg -re -i {SAMPLE_VIDEO} -c copy -rtsp_transport tcp -f rtsp <URL> &` -- (2) sleep "
            "~2-3s so the streams are live, (3) THEN launch the pipeline, reading each URL with "
            "`nvurisrcbin` over TCP with its rtsp-reconnect knobs enabled (see the `deepstream-dev` "
            "skill for the exact property names) and `async=0` on sinks. Each publisher pushes the "
            "file ONCE so the stream ENDS -> clean EOS (no looping). Reuse the running publishers "
            "across code edits; do NOT restart them every iteration."
        )
    if selected["needs_kafka"]:
        lines.append(f"- Kafka: bootstrap `{KAFKA_BOOTSTRAP}`, topic `{KAFKA_TOPIC}`.")
    if selected.get("needs_nats"):
        lines.append(
            f"- NATS: server at `{NATS_URL}` (JetStream enabled). Publish object-detection "
            f"metadata to subject `{NATS_SUBJECT}` (default subject for this run -- use it unless "
            "the prompt names another). A subscriber is ALREADY listening on that subject, so "
            "publishing there is what the result check reads back."
        )
    if selected["needs_vlm"]:
        lines.append(
            f"- VLM (OpenAI-compatible, no API key needed): `{VLM_ENDPOINT}`, model `{VLM_MODEL}`."
        )
    return lines


def build_agent_prompt():
    """Assemble + persist the WRITE-phase prompt (Generate step); set ctr_prompt_path / ctr_run_out."""
    global ctr_prompt_path, ctr_run_out
    run_out_host = WORKSPACE / "agent_outputs" / SELECTED_PROMPT_ID
    run_out_host.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(
            run_out_host, 0o777
        )  # let the non-root container agent write artifacts here
    except OSError:
        pass
    ctr_run_out = f"/workspace/agent_outputs/{SELECTED_PROMPT_ID}"

    # Ordinary Generate = the user's prompt VERBATIM + only one line saying WHERE to put the code.
    # The agent remains free to choose useful validation. We do not inject run_demo.sh / display /
    # endpoint rules here -- those are the formal Run step's job (build_run_prompt).
    # Option A: normal prompts Generate with cwd = their OWN empty app dir (see generate()), so the
    # agent can't read other prompts' code / their run_prompt.md (the headless "never use nv3dsink"
    # RUN_CONTRACT) and copy it -- that cross-prompt bleed was making every app come out fakesink.
    # `generate_in_workspace` prompts run in the shared /workspace instead: import_vision's skill
    # manages models/ itself, and rtvi's microservice stage builds ON TOP of the core app in rtvi_app.
    if selected.get("generate_in_workspace"):
        where = (
            f"\n\n---\n(Write all the code for this under `/workspace/{selected['output_dir']}`; "
            f"cwd is `/workspace`.)"
        )
    else:
        where = (
            f"\n\n---\n(Write all the code for this in the CURRENT working directory -- it is empty "
            f"and dedicated to this app (cwd is `/workspace/{selected['output_dir']}`). Put files "
            f"here; do NOT create a nested `{selected['output_dir']}/` subdirectory.)"
        )
    # If the prompt asks to interactively prompt for inputs, auto-answer 'y' (accept the defaults):
    # the agent runs headless (`-p`) and cannot actually prompt, so without this it stalls/asks
    # instead of generating. The prompt FILE is untouched -- this only appends to the assembled
    # prompt, like the line above.
    _pl = selected["prompt"].lower()
    if "prompt me" in _pl or "reply `y`" in _pl:
        where += ("\n(This runs headless and unattended -- you cannot prompt me. For every input the "
                  "request asks for, treat my answer as 'y': accept the default shown and proceed.)")
    if selected.get("generate_result") == "profile_report":
        where += (
            f"\n(Save the final measured report as `{ctr_run_out}/profiling_report.txt`; include "
            "the bottleneck, maximum 30-FPS stream count, measured per-stream FPS, and hardware "
            "recommendation.)"
        )
    prompt_text = selected["prompt"] + where
    prompt_path_host = run_out_host / "agent_prompt.md"
    prompt_path_host.write_text(prompt_text)
    ctr_prompt_path = f"{ctr_run_out}/agent_prompt.md"
    klog("Generating from your prompt.", "step")
    return prompt_text


def build_run_prompt(app_dir=None):
    """Assemble + persist the RUN-phase prompt (Run step): re-invoke the agent to RUN the app it
    wrote and FIX it until it produces the artifact. Services are live by now. `app_dir` is the
    (copy) directory the agent runs in -- defaults to the generated output_dir."""
    if ctr_run_out is None:
        build_agent_prompt()
    app_dir = app_dir or f"/workspace/{selected['output_dir']}"
    want = {
        "mp4": "a real, non-empty `out.mp4` video",
        "kafka_json": f"JSON detection messages on Kafka topic `{KAFKA_TOPIC}`",
        "kafka_text": f"VLM text-summary messages on Kafka topic `{KAFKA_TOPIC}`",
        "pdf_report": "a PDF report",
        "service_code": "the FastAPI microservice answering on `/openapi.json`",
        "logs": "the demo's stdout / FPS / buffer-count output",
        "profile_report": (
            f"a profiling report SAVED to a file `{ctr_run_out}/profiling_report.txt` (so it "
            "persists on disk and can be displayed -- do NOT only print it to stdout, which is "
            "lost when the run log is filtered out). Also echo it to stdout. It must state the "
            "bottleneck (decode / compute / memory-bandwidth), the max streams this GPU sustains "
            "at 30 FPS, and which HW upgrade helps -- plus the measured per-stream FPS"
        ),
        "nats_json": f"object-detection JSON messages published to NATS subject `{NATS_SUBJECT}`",
    }.get(selected["preferred_artifact"], "the expected output")
    # Prefer a FILE input unless the prompt genuinely needs live RTSP (then the app uses the RTSP
    # URLs it's told about, so no single DEMO_INPUT override).
    demo_input = DEMO_INPUT or ("" if _use_rtsp() else SAMPLE_VIDEO)
    secs = f"DEMO_SECONDS={DEMO_MAX_SECONDS}" + (
        f" DEMO_INPUT={shlex.quote(demo_input)}" if demo_input else ""
    )
    quoted = "> " + selected["prompt"].replace("\n", "\n> ")
    lines = [
        f"# Run & verify: {SELECTED_PROMPT_ID}",
        "",
        f"In a previous step you wrote a DeepStream app under `{app_dir}` for this request:",
        "",
        quoted,
        "",
        "Now make it actually RUN in this headless launchable and produce a result. You MAY modify "
        "the code you wrote (and create/fix `run_demo.sh`). The services it needs are NOW running.",
        "",
        RUN_CONTRACT,
        "",
        "## This run",
        f"1. Ensure an executable `run_demo.sh` exists at `{app_dir}` (create or fix it per the "
        "contract above).",
        f"2. Run it: `cd {app_dir} && {secs} bash run_demo.sh`. READ any error, FIX the code in "
        f"`{app_dir}`, and re-run. Iterate until it genuinely works.",
        f"3. Success = {want}, with the artifact written under `{ctr_run_out}`. Do not stop until "
        "that exists.",
        "",
        "## Paths & live services",
        f"- cwd `/workspace`; app `{app_dir}`; artifacts under `{ctr_run_out}`; default video `{SAMPLE_VIDEO}`.",
    ] + _service_endpoints_ctx()
    if selected["needs_rtsp"] and not _use_rtsp():
        lines.append(
            f"- Input: use the FILE `{SAMPLE_VIDEO}` (DEMO_INPUT is set to it) -- this demo "
            "does NOT need live RTSP; read the file source, not an `rtsp://` URL."
        )
    if selected["id"] == "rtvi_vlm_openapi_spec":
        lines.append(
            f"- Also start the microservice (`run_service.sh`) on port {RTVI_SERVICE_PORT} "
            "and confirm `/openapi.json` responds."
        )
    text = "\n".join(lines)
    (WORKSPACE / "agent_outputs" / SELECTED_PROMPT_ID / "run_prompt.md").write_text(
        text
    )
    return f"{ctr_run_out}/run_prompt.md"


def _container_workspace_host():
    """Host directory the running AGENT_CONTAINER maps at /workspace, or None if not running."""
    try:
        r = subprocess.run(
            ["docker", "inspect", AGENT_CONTAINER, "--format",
             '{{range .Mounts}}{{if eq .Destination "/workspace"}}{{.Source}}{{end}}{{end}}'],
            text=True, capture_output=True,
        )
    except OSError:
        return None
    src = (getattr(r, "stdout", "") or "").strip()
    return src or None


def ensure_install_container():
    """Reuse the existing work container for a standalone Install-step rerun."""
    inspected = subprocess.run(
        ["docker", "inspect", AGENT_CONTAINER, "--format", "{{.State.Running}}"],
        text=True,
        capture_output=True,
    )
    if inspected.returncode != 0:
        detail = (getattr(inspected, "stderr", "") or "").strip()
        raise RuntimeError(
            f"Container '{AGENT_CONTAINER}' does not exist. Run Step 2 once to create it."
            + (f" Docker: {detail}" if detail else "")
        )

    running = (getattr(inspected, "stdout", "") or "").strip().lower()
    if running == "false":
        started = subprocess.run(
            ["docker", "start", AGENT_CONTAINER],
            text=True,
            capture_output=True,
        )
        if started.returncode != 0:
            detail = (getattr(started, "stderr", "") or "").strip()
            raise RuntimeError(
                f"Container '{AGENT_CONTAINER}' exists but could not be started."
                + (f" Docker: {detail}" if detail else "")
            )
        klog(f"Reusing and started existing container '{AGENT_CONTAINER}'.", "ok")
    else:
        klog(f"Reusing existing container '{AGENT_CONTAINER}'.", "ok")
    sync_workspace_to_container()


def sync_workspace_to_container():
    """Make the kernel's WORKSPACE/OUTPUT_ROOT match the host path the EXISTING working container
    actually mounts at /workspace.

    Why: the container is created in Step 2 (with whatever OUTPUT_ROOT was set then) and may be REUSED
    across kernel restarts / later rounds. If the OUTPUT_ROOT default later differs (e.g. a config
    change), the agent would write into the container's mount while the kernel reads a different dir
    -> 'results not found'. The container is where work actually lands, so we treat ITS mount as the
    source of truth. No-op when no container exists (Step 2 will then create it at the current path).
    Rebinds ds_agent_lab's own WORKSPACE/OUTPUT_ROOT globals (a `global` here rebinds this module's
    copy, which every lab.* function uses)."""
    global WORKSPACE, OUTPUT_ROOT
    host = _container_workspace_host()
    if not host:
        return
    host = Path(host).resolve()
    if host != WORKSPACE:
        klog(
            f"Reusing container '{AGENT_CONTAINER}': syncing workspace to its actual mount "
            f"{host} (kernel had {WORKSPACE}). Re-run Step 2 to relocate.",
            "warn",
        )
        WORKSPACE = host
        OUTPUT_ROOT = host.parent


def prepare_services():
    """Start ONLY the RTSP/Kafka/VLM services the selected prompt needs (invisible/auto)."""
    global rtsp_urls
    sync_workspace_to_container()  # reuse-safe: read/write where the (reused) container maps /workspace
    # --- RTSP ---
    rtsp_urls = []
    if (
        _use_rtsp()
    ):  # only the genuinely multi-stream prompt; others test with a file (no RTSP)
        klog(
            "Starting RTSP SERVER (mediamtx) only -- the run agent pushes the sample ONCE per "
            "path from run_demo.sh, so each stream ends and EOS propagates ...",
            "step",
        )
        n_streams = 4 if selected["id"] == "multi_stream_tracker" else 1
        _pull(MEDIAMTX_IMAGE)
        subprocess.run(
            ["docker", "rm", "-f", "ds-agent-mediamtx"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                "ds-agent-mediamtx",
                "--network",
                "host",
                "--restart",
                "unless-stopped",
                MEDIAMTX_IMAGE,
            ],
            check=True,
        )
        for _ in range(15):  # wait until mediamtx accepts TCP on 8554 (~15s)
            try:
                with socket.create_connection(("127.0.0.1", 8554), timeout=1):
                    break
            except OSError:
                time.sleep(1)
        # Kill any leftover pushers from a previous run (`[f]fmpeg` so pkill can't match itself).
        subprocess.run(
            [
                "docker",
                "exec",
                AGENT_CONTAINER,
                "bash",
                "-lc",
                "pkill -f '[f]fmpeg.*rtsp' 2>/dev/null || true",
            ],
            check=False,
        )
        rtsp_urls = [f"{RTSP_BASE}/cam{idx}" for idx in range(n_streams)]
        klog(
            "RTSP server up. Paths (the run agent pushes one round each):\n"
            + "\n".join(rtsp_urls),
            "url",
        )

    # --- Kafka ---
    if selected["needs_kafka"]:
        klog(
            "Starting Kafka broker (apache/kafka, KRaft single node) -- pulling image ...",
            "step",
        )
        _pull(KAFKA_IMAGE)
        subprocess.run(
            ["docker", "rm", "-f", "ds-agent-kafka"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                "ds-agent-kafka",
                "--network",
                "host",
                "--restart",
                "unless-stopped",
                # apache/kafka (KRaft single node) -- KAFKA_* env (NOT bitnami's KAFKA_CFG_*).
                "-e",
                "KAFKA_NODE_ID=1",
                "-e",
                "KAFKA_PROCESS_ROLES=broker,controller",
                "-e",
                "KAFKA_CONTROLLER_QUORUM_VOTERS=1@localhost:9093",
                "-e",
                "KAFKA_LISTENERS=PLAINTEXT://:9092,CONTROLLER://:9093",
                "-e",
                "KAFKA_ADVERTISED_LISTENERS=PLAINTEXT://127.0.0.1:9092",
                "-e",
                "KAFKA_CONTROLLER_LISTENER_NAMES=CONTROLLER",
                "-e",
                "KAFKA_LISTENER_SECURITY_PROTOCOL_MAP=CONTROLLER:PLAINTEXT,PLAINTEXT:PLAINTEXT",
                "-e",
                "KAFKA_INTER_BROKER_LISTENER_NAME=PLAINTEXT",
                "-e",
                "KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR=1",
                "-e",
                "KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR=1",
                "-e",
                "KAFKA_TRANSACTION_STATE_LOG_MIN_ISR=1",
                "-e",
                "KAFKA_AUTO_CREATE_TOPICS_ENABLE=true",
                KAFKA_IMAGE,
            ],
            check=True,
        )
        # Poll the broker until it answers (replaces a fixed sleep); ~60s budget.
        kafka_ready = False
        for _ in range(20):
            probe = subprocess.run(
                [
                    "docker",
                    "exec",
                    "ds-agent-kafka",
                    "/opt/kafka/bin/kafka-topics.sh",
                    "--bootstrap-server",
                    "127.0.0.1:9092",
                    "--list",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if probe.returncode == 0:
                kafka_ready = True
                break
            time.sleep(3)
        if not kafka_ready:
            print(
                "WARNING: Kafka broker did not answer within ~60s; attempting topic create anyway."
            )
        subprocess.run(
            [
                "docker",
                "exec",
                "ds-agent-kafka",
                "/opt/kafka/bin/kafka-topics.sh",
                "--bootstrap-server",
                "127.0.0.1:9092",
                "--create",
                "--if-not-exists",
                "--topic",
                KAFKA_TOPIC,
            ],
            check=True,
        )
        klog(f"Kafka bootstrap {KAFKA_BOOTSTRAP} topic {KAFKA_TOPIC}", "ok")

    # --- NATS ---
    if selected.get("needs_nats"):
        klog(
            "Starting NATS server (JetStream enabled) + a background subscriber on subject "
            f"'{NATS_SUBJECT}' -- pulling images ...",
            "step",
        )
        _pull(NATS_IMAGE)
        _pull(NATS_BOX_IMAGE)
        for _c in ("ds-agent-nats", "ds-agent-nats-sub"):
            subprocess.run(
                ["docker", "rm", "-f", _c],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        # `-js` enables JetStream so the generated adapter's persistent-stream option works; a
        # plain core subscriber still receives JetStream publishes live, which is what we capture.
        subprocess.run(
            [
                "docker", "run", "-d", "--name", "ds-agent-nats", "--network", "host",
                "--restart", "unless-stopped", NATS_IMAGE, "-js",
            ],
            check=True,
        )
        nats_port = urlparse(NATS_URL).port or 4222
        for _ in range(15):  # wait until NATS accepts TCP on its client port (~15s)
            try:
                with socket.create_connection(("127.0.0.1", nats_port), timeout=1):
                    break
            except OSError:
                time.sleep(1)
        # Background subscriber connected BEFORE the run: its `docker logs` then hold every message
        # published on the subject, so show_results can read them back after the bounded run ends
        # (core NATS does not persist, but a connected subscriber receives messages in real time).
        subprocess.run(
            [
                "docker", "run", "-d", "--name", "ds-agent-nats-sub", "--network", "host",
                "--restart", "unless-stopped", NATS_BOX_IMAGE,
                "nats", "sub", NATS_SUBJECT, "--server", NATS_URL,
            ],
            check=True,
        )
        klog(f"NATS up at {NATS_URL}, subject {NATS_SUBJECT} (JetStream on)", "ok")

    # --- VLM (advanced) ---
    if selected["needs_vlm"]:
        klog(
            f"Serving local VLM {VLM_MODEL} on {VLM_ENDPOINT} -- FIRST RUN DOWNLOADS WEIGHTS "
            "(several GB; can take many minutes). Progress streams below; to abandon, use "
            "Kernel -> Interrupt (reloading the page does NOT stop it).",
            "warn",
        )
        vlm_port = str(urlparse(VLM_ENDPOINT).port or 8000)
        rc = stream_cmd(
            ["bash", str(REPO_ROOT / "deploy/brev/scripts/serve_vlm.sh")],
            env={
                **os.environ,
                "SERVE_VLM": "1",
                "VLM_MODEL": VLM_MODEL,
                "VLM_PORT": vlm_port,
                "HF_TOKEN": HF_TOKEN,
            },
            timeout=int(os.environ.get("VLM_TIMEOUT", "1500")),
        )
        if rc not in (0, None):
            raise SystemExit(
                f"VLM did not become ready (serve_vlm.sh exited {rc}); see the log "
                "above. Pick a non-VLM prompt to try the flow without the download."
            )
        klog(f"VLM endpoint ready: {VLM_ENDPOINT}", "ok")


def _print_agent_event(ev):
    """Render one Claude `stream-json` event as a short readable line so the user WATCHES the
    agent work (reads/writes/commands) instead of a frozen screen. Unknown shapes are ignored."""
    if ev.get("type") == "assistant":
        for block in ev.get("message", {}).get("content", []):
            bt = block.get("type")
            if bt == "text" and (block.get("text") or "").strip():
                print(block["text"].strip())
            elif bt == "tool_use":
                inp = block.get("input") or {}
                detail = (
                    inp.get("file_path")
                    or inp.get("command")
                    or inp.get("path")
                    or inp.get("pattern")
                    or ""
                )
                if isinstance(detail, str) and len(detail) > 120:
                    detail = detail[:120] + "..."
                klog(
                    f"  \U0001f527 {block.get('name', 'tool')} {detail}".rstrip(),
                    "step",
                )
    elif ev.get("type") == "result" and ev.get("is_error"):
        print("(agent reported an error result -- see the actions above)")


def _stream_agent_json(cmd, timeout):
    """Popen the claude stream-json exec, print events live (watchdog timeout, interrupt-safe)."""
    import threading
    import time

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    fired = {"v": False}  # did OUR watchdog fire?

    def _on_timeout():
        fired["v"] = True
        print(
            f"\n[watchdog] agent exceeded the {timeout}s budget -- sending SIGKILL to the "
            "`docker exec` client now.",
            flush=True,
        )
        proc.kill()

    timer = threading.Timer(timeout, _on_timeout)
    start = time.monotonic()
    timer.start()
    try:
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                _print_agent_event(json.loads(line))
            except ValueError:
                print(line)  # not JSON (e.g. a stderr line) -- show it raw
    except KeyboardInterrupt:
        proc.kill()
        print("\n[interrupted]")
        raise
    finally:
        timer.cancel()
        proc.wait()
    _LAST_AGENT_RUN.clear()
    _LAST_AGENT_RUN.update(
        timed_out=fired["v"], elapsed=round(time.monotonic() - start), timeout=timeout
    )
    return proc.returncode


def run_agent(
    prompt_path=None,
    log_name="agent_run.log",
    timeout=None,
    workdir="/workspace",
    synchronous=False,
):
    """Run the selected agent headless in the container on the prompt at `prompt_path` (defaults
    to the WRITE prompt). claude uses stream-json so each action shows live; codex streams plain."""
    if prompt_path is None:
        if ctr_prompt_path is None:
            build_agent_prompt()
        prompt_path = ctr_prompt_path
    timeout = timeout or int(
        os.environ.get("AGENT_TIMEOUT", "3000")
    )  # 50 min: a generous BACKSTOP on the whole Generate session (settable in Step 1). Live/RTSP
    # runtime is bounded separately at the Run step (DEMO_SECONDS + RUN_DEMO_TIMEOUT), not here.
    ctr_log = f"{ctr_run_out}/{log_name}"
    # `set -o pipefail` so the agent's real exit status is not masked by `| tee`.
    if AGENT == "claude":
        bash_env = ""
        settings_arg = ""
        if synchronous:
            # `claude -p` owns background Bash processes and kills them when its turn ends. Keep the
            # agent's command unchanged but rewrite that transport flag to foreground, so a long
            # engine build completes before the model continues. This is deterministic: no polling,
            # retry, or guessed sleep duration. The existing Generate watchdog remains the ceiling.
            hook_code = (
                "import json,sys;"
                "event=json.load(sys.stdin);"
                "tool_input=event.get('tool_input',{});"
                "background=tool_input.get('run_in_background') is True;"
                "tool_input['run_in_background']=False;"
                "result={'hookSpecificOutput':{'hookEventName':'PreToolUse',"
                "'permissionDecision':'allow','updatedInput':tool_input}};"
                "print(json.dumps(result)) if background else None"
            )
            settings = {
                "hooks": {
                    "PreToolUse": [
                        {
                            "matcher": "Bash",
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": f"python3 -c {shlex.quote(hook_code)}",
                                }
                            ],
                        }
                    ]
                }
            }
            bash_env = f"BASH_MAX_TIMEOUT_MS={timeout * 1000} "
            settings_arg = f"--settings {shlex.quote(json.dumps(settings))} "
        inner = (
            f"set -o pipefail; cd {shlex.quote(workdir)} && {bash_env}"
            f'claude -p "$(cat {shlex.quote(prompt_path)})" '
            f"--permission-mode bypassPermissions "
            f'--allowedTools "Read,Write,Edit,MultiEdit,Bash" '
            f"{settings_arg}"
            f"--output-format stream-json --verbose "
            f"2>&1 | tee {shlex.quote(ctr_log)}"
        )
        klog(
            f"Running claude in {AGENT_CONTAINER} -- live actions below (timeout {timeout}s):",
            "step",
        )
        cmd = [
            "docker",
            "exec",
            "-u",
            "agent",
            *build_cred_env(),
            AGENT_CONTAINER,
            "bash",
            "-lc",
            inner,
        ]
        rc = _stream_agent_json(cmd, timeout)
    else:
        inner = (
            f"set -o pipefail; cd {shlex.quote(workdir)} && {codex_exec_cli()} "
            f"--skip-git-repo-check --dangerously-bypass-approvals-and-sandbox "
            f"--json "
            f'"$(cat {shlex.quote(prompt_path)})" 2>&1 | tee {shlex.quote(ctr_log)}'
        )
        klog(
            f"Running codex in {AGENT_CONTAINER} (timeout {timeout}s). "
            f"Detailed output is saved to {ctr_log} (JSONL); key milestones are shown here.",
            "step",
        )
        rc = dexec_stream(inner, timeout, quiet=True)
        if rc in (0, None):
            klog(f"Codex finished; detailed output is saved to {ctr_log}.", "ok")
        elif _LAST_STREAM_TAIL:
            detail = "".join(_LAST_STREAM_TAIL).strip()
            if detail:
                print(
                    "\nLast Codex milestones (the full JSONL log is saved in the path above):\n"
                    + detail[-4000:]
                )
    if rc not in (0, None):
        # Classify WHY it stopped so a non-zero rc is diagnosable instead of guessed. rc=-9 alone is
        # ambiguous (our watchdog vs an external SIGKILL/OOM); the watchdog flag disambiguates.
        diag = dict(_LAST_AGENT_RUN)
        elapsed = diag.get("elapsed")
        if diag.get("timed_out"):
            cause = f"OUR {timeout}s watchdog fired and SIGKILLed the docker exec client (timeout)"
        elif rc == -9:
            cause = (
                "SIGKILL (-9) but our watchdog did NOT fire -- an EXTERNAL kill "
                "(host OOM-killer, or something/someone else). Check `dmesg | grep -i oom`"
            )
        elif rc < 0:
            cause = f"killed by signal {-rc} (not our watchdog)"
        else:
            cause = f"the agent process exited on its own with code {rc}"
        print(
            f"\nWARNING: {AGENT} did not finish cleanly (returncode={rc}"
            + (f", ran ~{elapsed}s of the {timeout}s budget" if elapsed is not None else "")
            + f"): {cause}. See the log above. Continuing best-effort."
        )
    return rc


def _find_run_demo():
    app_dir_host = WORKSPACE / selected["output_dir"]
    candidates = []
    direct = app_dir_host / "run_demo.sh"
    if direct.exists():
        candidates.append(direct)
    if app_dir_host.exists():
        candidates += sorted(app_dir_host.rglob("run_demo.sh"))
    if selected["needs_model_import"]:
        preferred = WORKSPACE / "models" / MODEL_NAME / "run_demo.sh"
        if preferred.exists():
            candidates.insert(0, preferred)
        if (WORKSPACE / "models").exists():
            candidates += sorted((WORKSPACE / "models").glob("*/run_demo.sh"))
    seen, ordered = set(), []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered


def _highlight_html(code, filename):
    """Syntax-highlight `code` to HTML with inline styles (pygments); <pre> fallback."""
    try:
        from pygments import highlight
        from pygments.lexers import get_lexer_for_filename, TextLexer
        from pygments.formatters import HtmlFormatter

        try:
            lexer = get_lexer_for_filename(filename)
        except Exception:
            lexer = TextLexer()
        return highlight(code, lexer, HtmlFormatter(noclasses=True, style="friendly"))
    except Exception:
        import html as _html

        return (
            "<pre style='background:#f6f8fa;padding:8px;overflow:auto;max-height:480px'>"
            + _html.escape(code)
            + "</pre>"
        )


# Dependency / build dirs an agent may create inside its app (venv, node deps, caches). Their
# contents are NOT generated code and must never be listed or zipped (a venv is thousands of
# files, e.g. onnx test `data.json`s leaking into the artifact list).
_NOISE_DIRS = {
    ".git",
    "__pycache__",
    "node_modules",
    "venv",
    ".venv",
    "env",
    ".env",
    "site-packages",
    "dist-packages",
    ".cache",
    ".pytest_cache",
    ".mypy_cache",
    ".ipynb_checkpoints",
    ".tox",
    ".ruff_cache",
}


# The launchable's OWN files (written under agent_outputs/<id>/): the agent's action transcripts
# and the prompts we feed it. These are harness internals -- NOT artifacts the user wants to see --
# so result listings must skip them (the "agent_run.log / run_agent.log got dumped as the result"
# noise the user flagged). The agent's tool-call log is never a result.
_HARNESS_FILES = {"agent_prompt.md", "run_prompt.md", "agent_run.log", "run_agent.log"}


def _not_noise(p, root):
    """True if `p` is neither inside a dependency/build dir NOR one of the launchable's own harness
    files (agent prompts / agent action logs) -- so result listings show real artifacts only."""
    if p.name in _HARNESS_FILES:
        return False
    try:
        parts = p.relative_to(root).parts
    except ValueError:
        parts = p.parts
    return not any(part in _NOISE_DIRS for part in parts)


def _jupyter_relative_path(path):
    """Return a path relative to the live Jupyter root, falling back to legacy repo-root."""
    roots = []
    try:
        from jupyter_server.serverapp import list_running_servers

        for server in list_running_servers():
            value = server.get("root_dir") or server.get("notebook_dir")
            if value:
                root = Path(value).expanduser().resolve()
                if root not in roots:
                    roots.append(root)
    except Exception:
        pass
    if REPO_ROOT.resolve() not in roots:
        roots.append(REPO_ROOT.resolve())
    resolved = Path(path).expanduser().resolve()
    for root in roots:
        try:
            return resolved.relative_to(root)
        except ValueError:
            continue
    return None


def _jupyter_file_url(path, download=False):
    """Build a browser-fetchable `/files/` URL relative to the actual Jupyter server root."""
    relative = _jupyter_relative_path(path)
    if relative is None:
        return None
    url = "/files/" + quote(relative.as_posix(), safe="/")
    return url + ("?download=1" if download else "")


def show_generated_code(max_files=6, max_lines=400, show_code=True, exclude_large_mb=None):
    """SHOWCASE the skills -> code result: a working download link + each primary file (code,
    run_demo.sh, and config files) shown collapsible (<details>) with syntax highlighting.

    Per-prompt display knobs (the Generate-phase rule differs by prompt):
      show_code=False     -> file list + download only, NO inline content dump (for prompts whose
                             output is a report + configs, not an app to read).
      exclude_large_mb=N  -> omit files larger than N MB from the download zip (e.g. import_vision's
                             135 MB engine/onnx) so the link stays small; the big files stay on box."""
    from IPython.display import HTML

    # The agent is told to save under output_dir; be robust if it nested under the run dir. Pick
    # the first location that actually holds code (ignoring the prompt file + run logs).
    generated_root = WORKSPACE / selected["output_dir"]
    if selected.get("generate_result") == "pdf_report":
        generated_root = generated_root / MODEL_NAME
    candidates = [
        generated_root,
        WORKSPACE / "agent_outputs" / SELECTED_PROMPT_ID,
    ]
    app_dir_host, all_files = None, []
    for root in candidates:
        if not root.exists():
            continue
        fs = [
            p
            for p in sorted(root.rglob("*"))
            if p.is_file()
            and _not_noise(p, root)
            and p.name != "agent_prompt.md"
            and p.suffix != ".log"
        ]
        if fs:
            app_dir_host, all_files = root, fs
            break
    if not all_files:
        print(
            f"No generated code found yet (looked in {candidates[0]} and the run dir). "
            "Re-run Generate or check the log above."
        )
        return False
    print(f"Generated code for '{SELECTED_PROMPT_ID}' under {app_dir_host}")
    browser_relative = _jupyter_relative_path(app_dir_host)
    browser_path = browser_relative.as_posix() if browser_relative else None
    if browser_path:
        print(f"\U0001f4c2 Open in the left file browser: {browser_path}/")
    if len(all_files) <= 20:
        print("Files: " + ", ".join(str(p.relative_to(app_dir_host)) for p in all_files))
    else:
        print(f"{len(all_files)} files generated (full list in the download).")

    # Download: zip under the repository, then build the URL relative to the live Jupyter root.
    import zipfile

    dl_dir = REPO_ROOT / "deploy/brev/_downloads"
    dl_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dl_dir / f"{SELECTED_PROMPT_ID}_generated.zip"
    skipped = []
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in all_files:
            try:
                if exclude_large_mb and p.stat().st_size > exclude_large_mb * 1048576:
                    skipped.append(p)
                    continue
            except OSError:
                pass
            zf.write(p, p.relative_to(app_dir_host))
    if skipped:
        print(
            f"(download omits {len(skipped)} file(s) >{exclude_large_mb} MB -- kept on the box only: "
            + ", ".join(sorted(p.name for p in skipped))
            + ")"
        )
    href = _jupyter_file_url(zip_path)
    if href:
        emit_display(
            HTML(
                f"<b>Download the generated code:</b> "
                f'<a href="{href}" download>{zip_path.name}</a>'
            )
        )
    else:
        print(f"Download zip: {zip_path}")

    if not show_code:  # download-only: no inline content dump (report+config prompts)
        return True

    pys = [p for p in all_files if p.suffix == ".py"]
    shells = [p for p in all_files if p.name in ("run_demo.sh", "run_service.sh")]
    # C/C++ sources: e.g. the nvds_msgapi protocol-adapter library (the core deliverable for the
    # NATS / custom-broker prompts) -- implementation files first, then headers.
    c_sources = [p for p in all_files if p.suffix in (".c", ".cpp", ".cc", ".cxx", ".h", ".hpp")]
    build_files = [p for p in all_files if p.name in ("Makefile", "CMakeLists.txt")]
    configs = [
        p
        for p in all_files
        if p.suffix in (".yml", ".yaml", ".cfg", ".ini", ".txt")
        and p.name.lower() != "requirements.txt"
    ]  # nvinfer / tracker / etc. configs

    def _py_key(p):
        rank = 0 if p.name.lower() in ("main.py", "app.py", "pipeline.py") else 1
        try:
            return (rank, -p.stat().st_size)
        except OSError:
            return (rank, 0)

    def _c_key(p):  # implementation (.c/.cpp/.cc/.cxx) before headers (.h/.hpp)
        return (0 if p.suffix not in (".h", ".hpp") else 1, str(p))

    primary, seen = [], set()
    for p in (
        sorted(shells)
        + sorted(pys, key=_py_key)
        + sorted(c_sources, key=_c_key)
        + sorted(configs)
        + sorted(build_files)
    ):
        if p not in seen:
            seen.add(p)
            primary.append(p)
    primary = primary[:max_files]

    for p in primary:
        try:
            code = p.read_text(errors="replace")
        except OSError as exc:
            print(f"(could not read {p}: {exc})")
            continue
        lines = code.splitlines()
        if len(lines) > max_lines:
            code = (
                "\n".join(lines[:max_lines])
                + f"\n... ({len(lines) - max_lines} more lines -- see the download)"
            )
        rel = p.relative_to(app_dir_host)
        opn = " open" if p is primary[0] else ""  # first file expanded, the rest folded
        emit_display(
            HTML(
                f"<details{opn}><summary><b>{rel}</b> ({p.stat().st_size} B)</summary>"
                f"{_highlight_html(code, p.name)}</details>"
            )
        )
    return True


_CUR_OUT = None  # the active step's Output widget (set by run_step_threaded's worker)


def emit_display(obj):
    """Rich display into the active step's Output widget. From a worker THREAD neither display()
    NOR the `with out:` context manager render (ipywidgets routes both via a parent message
    header the thread lacks); append_display_data writes to the widget directly and renders from
    any thread. Falls back to display() when not inside a threaded step."""
    if _CUR_OUT is not None:
        _CUR_OUT.append_display_data(obj)
    else:
        from IPython.display import display

        display(obj)


class _StepResult:
    """Mutable carrier so step_status can report success to its (threaded) caller."""

    __slots__ = ("ok",)

    def __init__(self):
        self.ok = False


@contextlib.contextmanager
def step_status(label, result_label=None, status_kind="success"):
    """Show an action result banner and yield whether the body finished cleanly."""
    from IPython.display import HTML

    st = _StepResult()
    try:
        yield st
        st.ok = True
        if status_kind == "info":
            emit_display(
                HTML(
                    f"<div style='margin-top:6px;padding:6px 10px;background:#ddf4ff;"
                    f"border:1px solid #2188ff;border-radius:6px;color:#0969da'>"
                    f"<b>&#8505; {result_label or label}</b></div>"
                )
            )
        else:
            emit_display(
                HTML(
                    f"<div style='margin-top:6px;padding:6px 10px;background:#e6ffed;"
                    f"border:1px solid #2ea44f;border-radius:6px;color:#1a7f37'>"
                    f"<b>&#9989; {result_label or label + ': success'}</b></div>"
                )
            )
    except (Exception, SystemExit) as e:
        import traceback

        traceback.print_exc()
        emit_display(
            HTML(
                f"<div style='margin-top:6px;padding:6px 10px;background:#ffeef0;"
                f"border:1px solid #d1242f;border-radius:6px;color:#cf222e'>"
                f"<b>&#10060; {label}: FAILED &mdash; {e}</b></div>"
            )
        )


# Per-step success flags: a later step is a NO-OP until its prerequisite has succeeded.
state = {"prepared": False, "installed": False, "generated": False}
_STEP_NAME = {
    "prepared": "Check environment & create workspace",
    "installed": "Install agent",
    "generated": "Generate",
}


def _installed_agent_available():
    """True when the existing work container already has the selected agent installed."""
    global AGENT
    candidates = (AGENT,) if AGENT in {"claude", "codex"} else ("claude", "codex")
    for candidate in candidates:
        result = subprocess.run(
            [
                "docker",
                "exec",
                "-u",
                "agent",
                AGENT_CONTAINER,
                "bash",
                "-lc",
                f"command -v {candidate} >/dev/null 2>&1",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode == 0:
            AGENT = candidate
            return True
    return False


def run_step(
    out,
    button,
    label,
    work,
    requires=None,
    success_flag=None,
    controls=(),
    status_kind="success",
    result_label=None,
    clear_output=True,
):
    """Run `work` SYNCHRONOUSLY on the kernel thread so its output renders reliably (background
    threads break Jupyter's output routing). Output STILL streams live as it runs -- the kernel
    is just busy during the step. Disables `button` plus related `controls` for the full operation,
    restores their prior states even after failure, shows the ✅/❌ banner, sets `success_flag` on
    success, and gates the operation until `requires` has succeeded."""
    if clear_output:
        out.clear_output()
        if hasattr(out, "outputs"):
            out.outputs = ()
    if requires == "installed":
        state["installed"] = _installed_agent_available()
    if requires and not state.get(requires, False):
        with out:
            prerequisite = _STEP_NAME.get(requires, requires)
            klog(f"Complete '{prerequisite}' successfully before '{label}'.", "warn")
        return False

    locked = []
    seen = set()
    for control in (button, *controls):
        if control is None or id(control) in seen:
            continue
        seen.add(id(control))
        locked.append((control, control.disabled))
        control.disabled = True
    previous_out = globals().get("_CUR_OUT")
    globals()["_CUR_OUT"] = out
    try:
        with out:
            status = (
                step_status(label)
                if status_kind == "success" and result_label is None
                else step_status(label, result_label=result_label, status_kind=status_kind)
            )
            with status as st:
                work()
            if st.ok and success_flag:
                state[success_flag] = True
            return st.ok
    finally:
        globals()["_CUR_OUT"] = previous_out
        for control, was_disabled in locked:
            control.disabled = was_disabled


def run_demo():
    """Run the agent-generated run_demo.sh inside the container (non-root 'agent').

    Bounded by RUN_DEMO_TIMEOUT (the prompt itself bounds the pipeline to
    DEMO_MAX_SECONDS) so RTSP/streaming demos cannot hang the notebook; on
    timeout we continue so show_results still runs. The demo's stdout+stderr are
    tee'd to a host-visible run_demo.log so the 'logs' scenario has an artifact."""
    ordered = _find_run_demo()
    if not ordered:
        raise SystemExit(
            f"No run_demo.sh found under {WORKSPACE / selected['output_dir']}. "
            "Re-run Generate or inspect generated files."
        )
    run_script = ordered[0]
    # No host-side chmod (the mount may be owned by uid 1000 and not writable by
    # the host user under `set -e`); invoke with `bash run_demo.sh` instead.
    ctr_app_dir = "/workspace/" + str(run_script.parent.relative_to(WORKSPACE))
    ctr_demo_log = f"{ctr_run_out}/run_demo.log"
    klog(
        f"Running demo: {ctr_app_dir}/run_demo.sh (timeout {RUN_DEMO_TIMEOUT}s) -> {ctr_demo_log}",
        "step",
    )
    # Pass the Run-step runtime params to the generated runner via env vars.
    env_prefix = f"DEMO_SECONDS={DEMO_MAX_SECONDS} "
    if DEMO_INPUT:
        env_prefix += f"DEMO_INPUT={shlex.quote(DEMO_INPUT)} "
    inner = (
        f"set -o pipefail; cd {shlex.quote(ctr_app_dir)} && "
        f"{env_prefix}bash run_demo.sh 2>&1 | tee {shlex.quote(ctr_demo_log)}"
    )
    # Codex's plain-text stream is extremely verbose; Claude's existing demo output remains
    # live in the notebook. Keep this choice explicit so the Codex suppression cannot alter the
    # Claude path.
    quiet = AGENT == "codex"
    rc = dexec_stream(inner, RUN_DEMO_TIMEOUT, quiet=quiet)
    if rc not in (0, None):
        if quiet and _LAST_STREAM_TAIL:
            detail = "".join(_LAST_STREAM_TAIL).strip()
            if detail:
                print(
                    "\nLast Codex demo diagnostics (the full log is saved in the path above):\n"
                    + detail[-4000:]
                )
        log_hint = "see the saved log above. " if quiet else "see the log above. "
        print(
            f"\nWARNING: run_demo.sh exited non-zero (returncode={rc}); "
            + log_hint
            + "Continuing to results."
        )
    elif quiet:
        klog(f"Codex demo finished; detailed output is saved to {ctr_demo_log}.", "ok")


def consume_kafka(max_show=5, quiet=False):
    """Consume recent Kafka messages and show them PRETTY: each JSON message formatted (indent=2)
    and syntax-highlighted, instead of one raw line per message. `quiet=True` suppresses the
    "no messages" line -- used where Kafka is an OPTIONAL secondary section (service_code), so an
    empty topic isn't reported as if it were a failure."""
    if not selected["needs_kafka"]:
        return
    from IPython.display import HTML

    cmd = [
        "docker",
        "exec",
        "ds-agent-kafka",
        "/opt/kafka/bin/kafka-console-consumer.sh",
        "--bootstrap-server",
        "127.0.0.1:9092",
        "--topic",
        KAFKA_TOPIC,
        "--from-beginning",
        "--timeout-ms",
        "15000",
        "--max-messages",
        "20",
    ]
    result = subprocess.run(cmd, text=True, capture_output=True)
    # DeepStream nvmsgconv messages are MULTI-LINE pretty JSON, so splitlines() would fragment one
    # message into many "lines" (the "645 messages / first 5 are just `{`,`version`,..." bug).
    # Decode consecutive JSON objects from the stream instead -- each object stays whole.
    text, msgs, i, dec = result.stdout, [], 0, json.JSONDecoder()
    while i < len(text):
        while i < len(text) and text[i] in " \t\r\n":
            i += 1
        if i >= len(text):
            break
        try:
            obj, i = dec.raw_decode(text, i)
            msgs.append(obj)
        except ValueError:
            nl = text.find("\n", i)
            if nl == -1:
                break
            i = nl + 1
    if not msgs:
        if not quiet:
            print(f"No Kafka messages on topic '{KAFKA_TOPIC}' within 15s.")
            if result.stderr.strip():
                print(result.stderr.strip().splitlines()[-1])
        return
    print(f"{len(msgs)} message(s) on topic '{KAFKA_TOPIC}'. Showing the first "
          f"{min(max_show, len(msgs))}, formatted:")
    for obj in msgs[:max_show]:
        emit_display(HTML(_highlight_html(json.dumps(obj, indent=2, ensure_ascii=False), "message.json")))
    if len(msgs) > max_show:
        print(f"... and {len(msgs) - max_show} more.")
    if result.returncode not in (0, 1):
        print(f"(kafka consumer exit {result.returncode})")


def consume_nats(max_show=5):
    """Show recent NATS messages the app published. prepare_services started a background core
    subscriber (ds-agent-nats-sub) on NATS_SUBJECT BEFORE the run, so its `docker logs` hold every
    message that flowed on the subject (whether the app used core NATS or JetStream publish)."""
    if not selected.get("needs_nats"):
        return
    from IPython.display import HTML

    result = subprocess.run(
        ["docker", "logs", "ds-agent-nats-sub"], text=True, capture_output=True
    )
    # The `nats sub` CLI interleaves decoration lines (`[#1] Received on "subj"`) with the JSON
    # payloads (often multi-line pretty JSON). Jump to each `{` and decode one whole object so a
    # pretty-printed message never gets fragmented (the same care as consume_kafka).
    text = (result.stdout or "") + (result.stderr or "")
    msgs, i, dec = [], 0, json.JSONDecoder()
    while i < len(text):
        b = text.find("{", i)
        if b == -1:
            break
        try:
            obj, i = dec.raw_decode(text, b)
            msgs.append(obj)
        except ValueError:
            i = b + 1
    if not msgs:
        print(
            f"No NATS messages captured on subject '{NATS_SUBJECT}' yet. The app must publish to "
            "that subject while the run is live (the subscriber is started before the run)."
        )
        return
    print(
        f"{len(msgs)} message(s) on subject '{NATS_SUBJECT}'. Showing the first "
        f"{min(max_show, len(msgs))}, formatted:"
    )
    for obj in msgs[:max_show]:
        emit_display(
            HTML(_highlight_html(json.dumps(obj, indent=2, ensure_ascii=False), "message.json"))
        )
    if len(msgs) > max_show:
        print(f"... and {len(msgs) - max_show} more.")


def curl_microservice():
    """Show the generated FastAPI microservice's result.

    The Run step already starts the service and saves `/openapi.json` (+ endpoint responses) under
    the run-out dir, so show THAT first -- it's always present when the agent succeeded, and it's the
    proof the service worked. Only if it's missing do we best-effort (re)start `run_service.sh`, and
    crucially from the COPY the agent actually ran in (`run-out/app`), NOT the Generate output
    (`/workspace/rtvi_app`) which never has `run_service.sh` -> the 'No such file or directory' bug."""
    if selected["id"] != "rtvi_vlm_openapi_spec":
        return
    base = f"http://127.0.0.1:{RTVI_SERVICE_PORT}"
    run_out = f"/workspace/agent_outputs/{SELECTED_PROMPT_ID}"
    run_app = f"{run_out}/app"  # where the agent created run_service.sh during Run

    def _show_paths(txt, source):
        print(f"=== /openapi.json (paths) -- {source} ===")
        try:
            for p, methods in json.loads(txt).get("paths", {}).items():
                print(f"  {p}: {', '.join(m.upper() for m in methods)}")
        except Exception:
            print(txt[:800])

    # 1) Prefer the spec the Run step already saved (robust; no fragile re-run needed).
    saved = dexec(
        f"cat {shlex.quote(run_out)}/openapi.json 2>/dev/null || "
        f"cat {shlex.quote(run_app)}/openapi.json 2>/dev/null || true",
        capture_output=True,
    )
    if saved.stdout.strip():
        _show_paths(saved.stdout, "saved by the Run step")
        others = dexec(
            f"ls {shlex.quote(run_out)}/*.json 2>/dev/null | grep -v /openapi.json || true",
            capture_output=True,
        ).stdout.strip()
        if others:
            print("\nEndpoint responses the Run step saved:")
            for ln in others.splitlines():
                print(f"  {ln}")
        return

    # 2) Fallback: start the service from the RUN COPY (where run_service.sh lives) and probe live.
    start = (
        f"cd {shlex.quote(run_app)} && "
        f"nohup bash run_service.sh > {shlex.quote(run_out)}/service.log 2>&1 & echo started"
    )
    subprocess.run(
        ["docker", "exec", "-u", "agent", *build_cred_env(), "-d", AGENT_CONTAINER, "bash", "-lc", start],
        check=False,
    )
    up = False
    for _ in range(30):
        if "up" in dexec(
            f"curl -fsS {base}/openapi.json >/dev/null 2>&1 && echo up || true", capture_output=True
        ).stdout:
            up = True
            break
        time.sleep(3)
    if not up:
        log = dexec(
            f"tail -n 40 {shlex.quote(run_out)}/service.log 2>/dev/null || true", capture_output=True
        )
        print("Microservice did not expose /openapi.json. service.log tail:")
        print(log.stdout)
        return
    _show_paths(dexec(f"curl -fsS {base}/openapi.json", capture_output=True).stdout, "live")


def _serve_file(src, download=False):
    """Return a Jupyter `/files/` URL for `src` so the browser STREAMS it (fast, no base64 bloat).
    Use the live server root when possible; otherwise copy under the repository downloads dir and
    retry. None on failure lets callers retain their existing embed/path fallback."""
    try:
        p = Path(src).resolve()
        direct = _jupyter_file_url(p, download=download)
        if direct:
            return direct
        dst_dir = REPO_ROOT / "deploy/brev/_downloads" / SELECTED_PROMPT_ID
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / p.name
        shutil.copy2(p, dst)
        return _jupyter_file_url(dst, download=download)
    except (OSError, ValueError):
        return None


def show_results():
    """Scenario-branched artifact display + a full fallback listing."""
    from IPython.display import display, HTML, FileLink

    sync_workspace_to_container()  # reuse-safe: read where the (reused) container maps /workspace
    search_roots = [WORKSPACE / "agent_outputs" / SELECTED_PROMPT_ID]
    if selected.get("generate_result") == "pdf_report":
        # The shared models cache may contain reports from other prompts or earlier model imports.
        # Scope this result to the model selected for the current import.
        search_roots.append(WORKSPACE / "models" / MODEL_NAME)
    else:
        search_roots.append(WORKSPACE / selected["output_dir"])
    if selected["needs_model_import"] and selected.get("generate_result") != "pdf_report":
        search_roots.append(WORKSPACE / "models")

    def _collect(patterns):
        found = []
        for root in search_roots:
            if root.exists():
                for pat in patterns:
                    found.extend(f for f in root.rglob(pat) if _not_noise(f, root))
        return sorted(set(found))

    # Result display is classified by the app's OUTPUT TYPE (basic rules):
    #   video                      -> mp4   (play the encoded out.mp4)
    #   kafka / protocol / API     -> JSON  (consume the topic / curl the service, show JSON)
    #   report                     -> PDF   (rasterize + show the pages)
    #   counter/stdout demo        -> the demo's own FPS/buffer-count output (e.g. nvdsdynamicsrcbin)
    # The agent's run logs (agent_run.log / run_agent.log) are harness noise and are NEVER shown.
    artifact = selected["preferred_artifact"]
    klog(f"=== Results for '{SELECTED_PROMPT_ID}' (artifact: {artifact}) ===", "ok")
    browser_relative = _jupyter_relative_path(
        WORKSPACE / "agent_outputs" / SELECTED_PROMPT_ID
    )
    browser_path = browser_relative.as_posix() if browser_relative else None
    if browser_path:
        print(f"\U0001f4c2 Browse these in the left file panel under: {browser_path}/")
    if artifact == "mp4":
        # Scope to THIS prompt's OWN output dirs -- NOT the shared models/ cache, which holds other
        # prompts' artifacts (e.g. import_vision's rtdetr sample videos) and would otherwise be shown
        # as this prompt's result. Skip 0-byte / failed files (a stray empty out.mp4 from an
        # unfinalized sink must never win over the real video). Prefer the canonical out.mp4 (the run
        # contract names it); else the LARGEST remaining real output. Stream through Jupyter's
        # `/files/` endpoint without base64 so large videos do not bloat the notebook document.
        own = (
            str(WORKSPACE / "agent_outputs" / SELECTED_PROMPT_ID),
            str(WORKSPACE / selected["output_dir"]),
        )
        mp4s = [
            m
            for m in _collect(("*.mp4",))
            if m.exists() and m.stat().st_size > 0 and str(m).startswith(own)
        ]
        pick = next((m for m in mp4s if m.name == "out.mp4"), None)
        if pick is None and mp4s:
            pick = max(mp4s, key=lambda p: p.stat().st_size)
        if pick:
            print(f"Playing: {pick}")
            url = _serve_file(pick)
            if url:
                emit_display(HTML(f'<video controls width="900" src="{url}"></video>'))
            else:
                print(
                    "Could not build a Jupyter /files/ URL for this video; "
                    f"open it from the file browser instead: {pick}"
                )
            extra = [m for m in mp4s if m != pick]
            if extra:
                print(f"({len(extra)} other .mp4 not shown: {', '.join(m.name for m in extra)})")
        else:
            print(
                "No non-empty MP4 found yet (a 0-byte file means the encoder never finalized -- "
                "ensure EOS reaches nvvideoencfilesinkbin before exit). Re-run the Run step."
            )
    elif artifact == "pdf_report":
        from IPython.display import Markdown, Image, HTML
        import re
        import tempfile

        # JupyterLab's output sanitizer STRIPS <iframe>/<embed>, and there is no inline
        # application/pdf renderer for cell output -- so the only reliable way to SHOW the PDF is to
        # rasterize each page to PNG (pdftoppm) and display via Image() (image/png is not sanitized,
        # so the formatted report renders exactly as in the PDF). The download uses a plain
        # <a href=/files/...> anchors survive sanitizing. The shared helper maps them against the
        # live server root (repo-root on A40, home-root on Brev) and copies only when necessary.

        pdfs = [
            path
            for path in _collect(("benchmark_report*.pdf",))
            if path.is_file() and path.stat().st_size > 0
        ]
        pdf = pdfs[0] if pdfs else None

        # Download link first (reliable anchor).
        if pdf:
            u = _serve_file(pdf, download=True)
            if u:
                emit_display(
                    HTML(f'<b>PDF report:</b> <a href="{u}" download>{pdf.name}</a> (download)')
                )

        # Show the PDF inline as page images (rasterize with pdftoppm).
        shown = False
        if pdf and shutil.which("pdftoppm"):
            tmp = Path(tempfile.mkdtemp(prefix="pdfpng_"))
            try:
                subprocess.run(
                    ["pdftoppm", "-png", "-r", "120", str(pdf), str(tmp / "page")],
                    check=True,
                    capture_output=True,
                    timeout=180,
                )
                pages = sorted(tmp.glob("page*.png"))
                if pages:
                    print(f"\nPDF report ({pdf.name}) -- {len(pages)} page(s):")
                    for pg in pages:
                        emit_display(Image(filename=str(pg)))  # embeds eagerly at display time
                    shown = True
            except Exception as e:
                print(f"(couldn't rasterize the PDF: {e}; showing report text + charts instead)")
            finally:
                shutil.rmtree(tmp, ignore_errors=True)

        # Fallback (no PDF, or rasterize failed): the benchmark report markdown + charts. The report
        # is reports/benchmark_report.{md,html} -- NOT README.md, which is setup instructions and must
        # never be shown as "the report".
        if not shown:
            mds = _collect(("*.md",))
            reports = [
                m
                for m in mds
                if m.name.lower() != "readme.md"
                and any(k in m.name.lower() for k in ("report", "benchmark", "summary"))
            ]
            if reports:
                txt = reports[0].read_text(errors="replace")
                txt = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", txt)
                txt = re.sub(r"<img[^>]*>", "", txt)
                print(f"Report ({reports[0].name}):")
                emit_display(Markdown(txt))
                shown = True
            # The styled HTML report can't render inline (JupyterLab sanitizes <iframe>/<embed>) --
            # offer it as a download instead.
            for h in _collect(("*.html",)):
                if any(k in h.name.lower() for k in ("report", "benchmark")):
                    u = _serve_file(h, download=True)
                    if u:
                        emit_display(
                            HTML(f'<b>HTML report:</b> <a href="{u}" download>{h.name}</a> (download)')
                        )
                        shown = True
                    break
            # Charts: PNGs in a charts/ dir, or named chart*/benchmark*.
            charts, seen = [], set()
            for c in _collect(("*.png",)):
                low = [p.lower() for p in c.parts]
                if c.name not in seen and (
                    "chart" in c.name.lower() or "benchmark" in c.name.lower() or "charts" in low
                ):
                    seen.add(c.name)
                    charts.append(c)
            if charts:
                print(f"\nCharts ({len(charts)}):")
                for c in charts:
                    emit_display(Image(filename=str(c)))
                shown = True

        # Nothing report-like found -- say so clearly and do NOT pass README.md off as the report.
        # The skill writes reports/benchmark_report.{md,html,pdf} + reports/charts/ only after the
        # engine build + benchmark finish; an interrupted/timed-out Generate leaves none of them.
        if not shown and not pdf:
            print(
                "No benchmark report found. Expected models/<name>/reports/"
                "benchmark_report.{md,html,pdf} + reports/charts/. The import-vision pipeline did not "
                "finish its report stage (often the TensorRT engine build was still running) -- re-run "
                "and let Generate run to completion."
            )
            readmes = [m for m in _collect(("*.md",)) if m.name.lower() == "readme.md"]
            if readmes:
                print(f"\nSetup instructions ({readmes[0].name}) -- NOT the benchmark report:")
                emit_display(Markdown(readmes[0].read_text(errors="replace")))
    elif artifact in ("kafka_json", "kafka_text"):
        kind = "JSON detection" if artifact == "kafka_json" else "VLM text summary"
        print(f"This scenario streams {kind} messages to Kafka topic '{KAFKA_TOPIC}'.")
        consume_kafka()
        jsons = _collect(("*.json",))
        if jsons:
            print("\nRelated JSON artifacts:")
            for jf in jsons[:5]:
                print(f"  {jf}")
    elif artifact == "logs":
        # This scenario's deliverable is the demo's OWN stdout (e.g. nvdsdynamicsrcbin's per-pad
        # buffer count + FPS) -- not the agent's action transcript. _collect() drops the harness
        # logs (agent_run.log / run_agent.log), so only the demo's output log is shown; the agent's
        # tool-call logs are never surfaced as a result.
        demo_logs = _collect(("*.log",))
        if demo_logs:
            print("Demo output (FPS / buffer counts):")
            for lf in demo_logs[:2]:
                print(f"--- {lf.name} (tail) ---")
                print("\n".join(lf.read_text(errors="replace").splitlines()[-40:]))
                print()
        else:
            print(
                "Ran OK -- this scenario prints its FPS / buffer counts to stdout, which streamed "
                "live in the Run step above."
            )
    elif artifact == "service_code":
        # Rule: this is a VLM SUMMARIZATION app fronted by a FastAPI microservice. The DELIVERABLE
        # is the VLM summary of the video (the core app's output) -- show that first: the saved
        # summary/caption JSON content inline + any summaries published to the Kafka topic. The
        # microservice openapi spec is the access layer, shown as a secondary section (not "the
        # result"). The agent's run logs are never shown.
        from IPython.display import HTML

        run_out = WORKSPACE / "agent_outputs" / SELECTED_PROMPT_ID
        shown = 0
        if run_out.exists():
            for p in sorted(run_out.glob("*.json")):
                if p.name == "openapi.json":
                    continue
                try:
                    obj = json.loads(p.read_text(errors="replace"))
                except (OSError, ValueError):
                    continue
                # A summary result has caption/summary/text/transcript content (vs health/metrics).
                if isinstance(obj, dict) and any(
                    obj.get(k)
                    for k in ("caption", "captions", "summary", "summaries", "text", "transcript")
                ):
                    if shown == 0:
                        print("VLM summary (the app's output):")
                    print(f"--- {p.name} ---")
                    emit_display(
                        HTML(_highlight_html(json.dumps(obj, indent=2, ensure_ascii=False), p.name))
                    )
                    shown += 1
                    if shown >= 3:
                        break
        consume_kafka(quiet=True)  # secondary: show Kafka VLM summaries IF any (this variant may
        # serve captions only via the API and not publish to Kafka -- don't report empty as a failure)
        if not shown:
            print("(no saved VLM summary JSON found under the run-out dir yet)")
        print("\n--- microservice API (openapi spec) ---")
        curl_microservice()
        svc_root = WORKSPACE / selected["output_dir"]
        print("\nGenerated files:")
        if svc_root.exists():
            for pat in ("*.py", "Dockerfile*", "*.md", "requirements*.txt"):
                for f in sorted(svc_root.rglob(pat))[:20]:
                    print(f"  {f}")
    elif artifact == "profile_report":
        # FIXED deliverable (same discipline as out.mp4 for video): the Generate request saves the
        # report at exactly `<run_out>/profiling_report.txt`. Read THAT file --
        # no keyword globbing or log tailing. If missing, Generate did not finish the requested job.
        import html as _html

        from IPython.display import HTML

        run_out = WORKSPACE / "agent_outputs" / SELECTED_PROMPT_ID
        report = run_out / "profiling_report.txt"
        if report.exists():
            print(f"Profiling report ({report.name}):")
            # Fixed-width plain text (aligned columns, single-\n line breaks): render verbatim in a
            # <pre> so the layout is preserved (Markdown would collapse the newlines and spaces).
            emit_display(
                HTML(
                    "<pre style='white-space:pre;overflow-x:auto;font-family:monospace;"
                    f"font-size:12px;line-height:1.3'>{_html.escape(report.read_text(errors='replace'))}</pre>"
                )
            )
        else:
            print(
                f"No profiling report at {report}. Re-run Generate so the profiling case can "
                "finish its measured deliverable."
            )
        # Supplementary (not the deliverable): the Nsight capture, for the GUI trace.
        nsys = _collect(("*.nsys-rep",))
        if nsys:
            print("\nNsight Systems captures (open in the Nsight Systems GUI for the full trace):")
            for r in nsys[:3]:
                print(f"  {r}")
    elif artifact == "nats_json":
        print(
            f"This scenario publishes object-detection metadata to NATS subject '{NATS_SUBJECT}'."
        )
        consume_nats()
        jsons = _collect(("*.json",))
        if jsons:
            print("\nRelated JSON artifacts:")
            for jf in jsons[:5]:
                print(f"  {jf}")
    else:
        print("No scenario-specific renderer; see the artifact listing below.")

    all_artifacts = _collect(("*.mp4", "*.pdf", "*.json", "*.log", "*.nsys-rep", "README.md"))
    print("\n=== All artifacts ===")
    print(
        "\n".join(str(p) for p in all_artifacts)
        if all_artifacts
        else "No artifacts found yet. Generate and Run first."
    )


# ===========================================================================
# Plain (no-widgets) step wrappers -- the fallbacks each step widget also calls.
# ===========================================================================


def install_agent():
    """Install step: install ONLY the chosen agent CLI + deps, create the non-root agent user,
    copy both skills, then run a quick auth smoke. Idempotent; re-running is safe.

    Uses the AGENT global; credentials come from account sign-in or env. The
    auth smoke runs as the non-root 'agent' user with creds injected per-exec."""
    if AGENT not in {"claude", "codex"}:
        raise SystemExit(
            "No agent chosen. Set AGENT to 'claude' or 'codex' (the Install step or env), "
            "then call install_agent()."
        )
    ensure_install_container()
    if AGENT == "claude":
        agent_pkg, agent_bin = "@anthropic-ai/claude-code", "claude"
    else:
        agent_pkg, agent_bin = "@openai/codex", "codex"

    install_script = textwrap.dedent(f"""
        set -eux
        export DEBIAN_FRONTEND=noninteractive

        # --- Node.js 20 (NodeSource) + npm, only if node 20+ is not already present ---
        if ! command -v node >/dev/null 2>&1 || [ "$(node -p 'process.versions.node.split(".")[0]' 2>/dev/null || echo 0)" -lt 20 ]; then
            apt-get update
            apt-get install -y --no-install-recommends ca-certificates curl gnupg
            curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
            apt-get install -y --no-install-recommends nodejs
        fi
        node --version
        npm --version

        # --- Install ONLY the selected agent CLI globally ---
        npm i -g {agent_pkg}
        npm_bin="$(npm prefix -g)/bin/{agent_bin}"
        if [ -x "$npm_bin" ] && [ ! -e /usr/local/bin/{agent_bin} ]; then
            ln -sf "$npm_bin" /usr/local/bin/{agent_bin}
        fi
        command -v {agent_bin}

        # --- Shared system deps: Kafka native lib, ffmpeg, jq, curl, git, sudo ---
        apt-get update
        apt-get install -y --no-install-recommends \\
            librdkafka-dev python3-pip jq curl git sudo
        if [ -x /opt/nvidia/deepstream/deepstream/user_additional_install.sh ]; then
            /opt/nvidia/deepstream/deepstream/user_additional_install.sh || true
        fi
        if ! command -v ffmpeg >/dev/null 2>&1; then
            apt-get install -y --no-install-recommends ffmpeg
        fi
        rm -rf /var/lib/apt/lists/*

        # --- DeepStream Python API + Kafka python client ---
        pip install --break-system-packages \\
            /opt/nvidia/deepstream/deepstream/service-maker/python/pyservicemaker*.whl pyyaml
        pip install --break-system-packages confluent-kafka kafka-python || \\
            pip install --break-system-packages kafka-python

        # --- Non-root agent user (passwordless sudo for iterating). Do NOT force uid
        # 1000: the Ubuntu 24.04 base image already has a user at uid 1000 ('ubuntu'),
        # so `useradd -u 1000` fails with "UID not unique". Auto-assign the uid and
        # reference the user BY NAME everywhere (docker exec -u agent, chown agent). ---
        id -u agent >/dev/null 2>&1 || useradd -m -s /bin/bash agent
        echo 'agent ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/agent
        chmod 0440 /etc/sudoers.d/agent
        mkdir -p {AGENT_HOME}/.claude/skills {AGENT_HOME}/.codex/skills

        # --- /workspace is a bind mount shared by BOTH the host (notebook) and the
        # non-root container agent, which have DIFFERENT uids. chown-ing it to the
        # agent locks the host out (breaks artifact display + re-runs), so make it
        # world-rwx and have the agent create world-writable files (umask 000). ---
        chmod -R 0777 /workspace
        grep -q 'umask 000' {AGENT_HOME}/.bashrc 2>/dev/null || echo 'umask 000' >> {AGENT_HOME}/.bashrc

        echo '=== versions ==='
        node --version
        {agent_bin} --version || true
        python3 -c 'import pyservicemaker; print("pyservicemaker import OK")'
    """)
    klog(
        f"Installing '{AGENT}' ({agent_pkg}) + deps into {AGENT_CONTAINER} (idempotent) ...",
        "step",
    )
    # STREAM the install (Popen+print) so its log is captured by the active step's `with out:`
    # and stays in THIS cell -- subprocess.run output is fd-level and leaks to whatever cell is
    # focused (the "log jumps to the prompt cell" bug).
    rc = stream_cmd(
        ["docker", "exec", "-u", "0", AGENT_CONTAINER, "bash", "-lc", install_script],
        timeout=1800,
        quiet=False,  # Install must retain the complete apt/npm/pip transcript.
        # Keep the complete transcript, but coalesce widget messages so apt/dpkg output does not
        # overwhelm the Jupyter WebSocket and browser DOM.
        ui_flush_interval=0.25,
        ui_flush_chars=64 * 1024,
    )
    if rc not in (0, None):
        raise SystemExit(f"Agent install failed (exit {rc}); see the log above.")

    # Copy BOTH repo skills into BOTH agent skill homes via docker cp, then chown. Output is
    # CAPTURED (short, uninteresting) so it can't leak to a focused cell either.
    for name in ("deepstream-dev", "deepstream-import-vision-model", "deepstream-profile-pipeline"):
        src = str(REPO_ROOT / "skills" / name)
        for home_sub in (".claude/skills", ".codex/skills"):
            subprocess.run(["docker", "exec", "-u", "0", AGENT_CONTAINER, "bash", "-lc",
                            f"rm -rf {AGENT_HOME}/{home_sub}/{name}"], check=True, capture_output=True)
            subprocess.run(["docker", "cp", src, f"{AGENT_CONTAINER}:{AGENT_HOME}/{home_sub}/{name}"],
                           check=True, capture_output=True)
    subprocess.run(["docker", "exec", "-u", "0", AGENT_CONTAINER, "bash", "-lc",
                    f"chown -R agent:agent {AGENT_HOME}"], check=True, capture_output=True)

    klog(f"Agent installed: {AGENT} (+ skills).", "ok")


def validate_codex_endpoint(base_url, token, model):
    """Validate the non-secret shape of a Codex third-party endpoint.

    The subsequent check_auth() call performs the real provider/model smoke
    request. This early validation catches the common cases where a user
    pasted an endpoint with credentials in its URL or left a required field
    empty.
    """
    base = (base_url or "").strip().rstrip("/")
    if not base:
        raise ValueError("Custom endpoint base URL is required.")
    parsed = urlparse(base)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Custom endpoint must be an http:// or https:// URL.")
    if parsed.username or parsed.password:
        raise ValueError("Do not put credentials in the endpoint URL; use the token field.")
    if not (token or "").strip():
        raise ValueError("Custom endpoint token is required.")
    if not (model or "").strip():
        raise ValueError("Codex custom endpoint model is required.")
    return base


def validate_claude_endpoint(base_url, token, model):
    """Validate the required fields for a Claude Anthropic-compatible endpoint."""
    base = (base_url or "").strip().rstrip("/")
    if not base:
        raise ValueError("Claude custom endpoint base URL is required.")
    parsed = urlparse(base)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Claude custom endpoint must be an http:// or https:// URL.")
    if parsed.username or parsed.password:
        raise ValueError("Do not put credentials in the endpoint URL; use the token field.")
    if not (token or "").strip():
        raise ValueError("Claude custom endpoint token is required.")
    if not (model or "").strip():
        raise ValueError("Claude custom endpoint model is required.")
    return base


def validate_claude_api_key(api_key):
    """Validate the required field for Claude API-key authentication."""
    key = (api_key or "").strip()
    if not key:
        raise ValueError("Claude API key is required.")
    return key


def clear_claude_credentials():
    """Clear Claude API/custom-endpoint credentials before changing methods."""
    global ANTHROPIC_API_KEY, ANTHROPIC_BASE_URL, ANTHROPIC_AUTH_TOKEN
    ANTHROPIC_API_KEY = ""
    ANTHROPIC_BASE_URL = ""
    ANTHROPIC_AUTH_TOKEN = ""


def clear_codex_credentials():
    """Clear Codex credentials before switching authentication methods."""
    global OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL
    OPENAI_API_KEY = ""
    OPENAI_BASE_URL = ""
    OPENAI_MODEL = ""


def _check_codex_auth():
    """Verify Codex account, API-key, or custom-provider authentication."""
    if OPENAI_BASE_URL:
        # The widget validates this before storing the values. Repeat the same
        # guard for Step 1 environment variables so an empty token/model cannot
        # reach the smoke command and be reported as an opaque auth failure.
        validate_codex_endpoint(OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL)
    if codex_login_in_progress():
        raise RuntimeError(
            "Codex device sign-in is still in progress. Enter the device number, "
            "finish browser verification, then click Check sign-in."
        )
    klog("Verifying codex auth (real provider smoke prompt) ...", "step")
    try:
        smoke = dexec(
            f'cd /workspace && {codex_exec_cli()} '
            '--skip-git-repo-check --sandbox read-only --ephemeral '
            '"Reply with exactly OK. Do not call tools."',
            capture_output=True,
            timeout=180,
        )
        if smoke.returncode != 0:
            detail = "\n".join(
                part
                for part in (
                    getattr(smoke, "stderr", ""),
                    getattr(smoke, "stdout", ""),
                )
                if part
            ).strip()
            detail = _redact_auth_detail(detail)
            if detail:
                print(detail[-4000:])
            raise RuntimeError(
                "Codex authentication check failed. Check the ChatGPT sign-in, API key, "
                "endpoint, and model, then try again."
            )
        klog("Agent ready: codex", "ok")
        return True
    finally:
        _reap_codex_login()


def _redact_auth_detail(detail):
    """Remove known secrets and common query-string credentials from diagnostics."""
    for secret in (OPENAI_API_KEY,):
        if secret:
            detail = detail.replace(secret, "<redacted>")
    return re.sub(
        r"(?i)([?&](?:api[_-]?key|token|access[_-]?token|authorization)=)[^&\s]+",
        r"\1<redacted>",
        re.sub(r"(?i)(https?://)[^/@\s]+:[^/@\s]+@", r"\1<redacted>@", detail),
    )


def check_auth(agent=None, credential_mode=None):
    """Verify the selected agent's credentials without changing Claude's flow."""
    a = (agent or AGENT or "").strip().lower()
    selected = (AGENT or "").strip().lower()
    if a != selected:
        raise ValueError("Step 3 agent changed; rerun Step 4 to rebuild authentication controls.")
    if a == "codex":
        return _check_codex_auth()
    backend = auth_backend_for(a)
    if credential_mode == "api-key":
        validate_claude_api_key(ANTHROPIC_API_KEY)
    elif credential_mode == "custom-endpoint":
        validate_claude_endpoint(
            ANTHROPIC_BASE_URL, ANTHROPIC_AUTH_TOKEN, ANTHROPIC_MODEL
        )
    elif backend == "anthropic-custom-endpoint":
        validate_claude_endpoint(
            ANTHROPIC_BASE_URL, ANTHROPIC_AUTH_TOKEN, ANTHROPIC_MODEL
        )
    elif backend == "anthropic-api-key":
        validate_claude_api_key(ANTHROPIC_API_KEY)
    elif backend is None:
        raise ValueError(
            "Claude credentials are required. Choose account sign-in, enter an API key, "
            "or enter a custom endpoint URL, token, and model."
        )
    klog(f"Verifying {a} auth (tiny headless prompt) ...", "step")
    model_arg = (
        f" --model {shlex.quote(ANTHROPIC_MODEL)}"
        if (credential_mode == "custom-endpoint" or backend == "anthropic-custom-endpoint")
        and ANTHROPIC_MODEL
        else ""
    )
    smoke = dexec(
        'cd /workspace && claude -p "Say OK" --permission-mode bypassPermissions '
        f"--output-format text{model_arg}",
        capture_output=True,
        timeout=180,
    )
    if smoke.returncode != 0:
        print(smoke.stderr or smoke.stdout)
        raise RuntimeError(
            "Claude authentication check failed. Check the API key, endpoint, token, and model, "
            "then try again."
        )
    else:
        klog(f"Agent ready: {a}", "ok")
    return True


# --- Account sign-in (subscription OAuth), as an alternative to an API key ---
_login_proc = None
_login_proc_agent = None
CODEX_DEVICE_URL = ""
CODEX_DEVICE_CODE = ""


def _parse_codex_device_output(output):
    """Extract the device URL and one-time code for a persistent notebook control."""
    clean = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", str(output or ""))
    url_match = re.search(r"https://auth\.openai\.com/codex/device[^\s]*", clean)
    code_match = re.search(
        r"(?im)^\s*([A-Z0-9]{4,}(?:-[A-Z0-9]{4,})+)\s*$", clean
    )
    url = url_match.group(0).rstrip(".,)") if url_match else ""
    code = code_match.group(1) if code_match else ""
    return url, code


def cancel_login():
    """Stop an in-progress account login when the auth method changes."""
    global _login_proc, _login_proc_agent
    proc = _login_proc
    if proc is None:
        return
    try:
        if proc.poll() is None:
            proc.kill()
        proc.communicate(timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        try:
            proc.kill()
            proc.communicate()
        except (OSError, subprocess.TimeoutExpired):
            pass
    finally:
        _login_proc = None
        _login_proc_agent = None


def codex_login_in_progress():
    """Return whether the Codex device-auth process is still waiting for the browser."""
    return (
        AGENT == "codex"
        and _login_proc is not None
        and _login_proc_agent == "codex"
        and _login_proc.poll() is None
    )


def _reap_codex_login():
    """Close the Codex device-auth process and its pipes after verification."""
    global _login_proc, _login_proc_agent
    proc = _login_proc
    if proc is None or _login_proc_agent != "codex":
        return
    try:
        try:
            out, _ = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, _ = proc.communicate()
        if out:
            print(_redact_auth_detail(out), end="")
    finally:
        _login_proc = None
        _login_proc_agent = None


def _drain_login_output(seconds=3, emit=True):
    """Collect currently available login output without blocking the notebook kernel."""
    if _login_proc is None or _login_proc.stdout is None:
        return ""
    fd = _login_proc.stdout.fileno()
    was_blocking = os.get_blocking(fd)
    os.set_blocking(fd, False)
    end = time.monotonic() + seconds
    captured = []
    try:
        while time.monotonic() < end:
            ready, _, _ = select.select([fd], [], [], 0.25)
            if not ready:
                if _login_proc.poll() is not None:
                    break
                continue
            try:
                chunk = os.read(fd, 4096)
            except BlockingIOError:
                continue
            if not chunk:
                break
            text_chunk = chunk.decode(errors="replace")
            captured.append(text_chunk)
            if emit:
                print(text_chunk, end="")
    finally:
        os.set_blocking(fd, was_blocking)
    return "".join(captured)


def login_start(use_console=False):
    """Start account sign-in for the selected agent.

    Claude uses a paste-code flow. Codex uses its device-auth flow so a remote
    Jupyter service does not need a browser callback tunneled into the container;
    after completing the browser step, click the Codex verification button.
    """
    global _login_proc, _login_proc_agent, CODEX_DEVICE_URL, CODEX_DEVICE_CODE
    if AGENT == "codex":
        if _login_proc is not None and _login_proc.poll() is None:
            raise RuntimeError("A Codex sign-in is already in progress; finish it or restart the step.")
        if _login_proc is not None:
            cancel_login()
        CODEX_DEVICE_URL = ""
        CODEX_DEVICE_CODE = ""
        clear_codex_credentials()
        _login_proc = subprocess.Popen(
            [
                "docker",
                "exec",
                "-i",
                "-u",
                "agent",
                AGENT_CONTAINER,
                "bash",
                "-lc",
                "codex login --device-auth",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        _login_proc_agent = "codex"
        output = _drain_login_output(emit=False)
        CODEX_DEVICE_URL, CODEX_DEVICE_CODE = _parse_codex_device_output(output)
        klog(
            "Codex ChatGPT device sign-in started. Use the URL and device number in the controls, "
            "then click 'Check sign-in'.",
            "url",
        )
        if not CODEX_DEVICE_URL or not CODEX_DEVICE_CODE:
            if output:
                print(output, end="")
            else:
                print("If no device URL appeared, confirm Codex is installed and rerun Start sign-in.")
        return True

    if _login_proc is not None and _login_proc.poll() is None:
        raise RuntimeError("A Claude sign-in is already in progress; finish it or restart the step.")
    if _login_proc is not None:
        cancel_login()
    clear_claude_credentials()
    # Claude: run the login WITHOUT any endpoint/key env so it does real account OAuth.
    inner = (
        "env -u ANTHROPIC_BASE_URL -u ANTHROPIC_AUTH_TOKEN -u ANTHROPIC_API_KEY "
        f"claude auth login --{'console' if use_console else 'claudeai'}"
    )
    _login_proc = subprocess.Popen(
        ["docker", "exec", "-i", "-u", "agent", AGENT_CONTAINER, "bash", "-lc", inner],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    _login_proc_agent = "claude"
    url = None
    for line in iter(_login_proc.stdout.readline, ""):
        m = re.search(r"https?://\S*oauth\S*", line)
        if m:
            url = m.group(0)
            break  # the URL line ends in newline; the paste prompt (no newline) follows
    if url:
        klog(
            "Open this URL in your browser, sign in to your Claude account, copy the code it "
            "shows, then paste it into the field below and click Submit:",
            "url",
        )
        print(url)
    else:
        print(
            "Could not capture a sign-in URL. Is the agent installed (the Install step)? Try `claude auth status`."
        )
    return url


def login_submit(code):
    """Finish Claude account sign-in by submitting the code copied from the browser."""
    global _login_proc, _login_proc_agent
    if AGENT != "claude":
        print("Codex uses device sign-in; click 'Check sign-in' after the browser flow.")
        return
    if _login_proc is None or _login_proc.poll() is not None:
        print("No sign-in in progress -- click 'Start sign-in' first.")
        return
    _login_proc.stdin.write(code.strip() + "\n")
    _login_proc.stdin.flush()
    try:
        out, _ = _login_proc.communicate(timeout=90)
        print(out)
    except subprocess.TimeoutExpired:
        print("Sign-in timed out; check `claude auth status`.")
        _login_proc.kill()
    _login_proc = None
    _login_proc_agent = None
    st = dexec("claude auth status 2>&1 || true", capture_output=True)
    print(st.stdout or st.stderr)


def print_plan(item=None):
    item = item or selected
    steps, services = scenario_plan(item)
    print(f"Prompt: {item['id']} ({item['title']})")
    print(f"  skill={item['skill']}  artifact={item['preferred_artifact']}")
    print(
        f"  services: {', '.join(services) if services else '(none -- file-based demo)'}"
    )


def _generate_result_status(item):
    """Return `(complete, missing_message)` for an execute-mode Generate deliverable."""
    result = item.get("generate_result")
    if result == "pdf_report":
        report_dir = WORKSPACE / "models" / MODEL_NAME / "reports"
        complete = any(
            path.is_file() and path.stat().st_size > 0
            for path in report_dir.glob("benchmark_report*.pdf")
        )
        return (
            complete,
            f"no non-empty benchmark PDF was written under {report_dir}",
        )
    if result == "profile_report":
        report = WORKSPACE / "agent_outputs" / item["id"] / "profiling_report.txt"
        try:
            report_text = report.read_text(errors="replace").lower()
        except OSError:
            report_text = ""
        required_conclusions = (
            "bottleneck",
            "stream",
            "fps",
        )
        has_hardware_recommendation = any(
            word in report_text for word in ("hardware", "upgrade", "recommend")
        )
        complete = all(word in report_text for word in required_conclusions)
        complete = complete and has_hardware_recommendation
        return (
            complete,
            f"{report} is missing or does not contain the requested measured conclusions",
        )
    return False, f"unknown Generate result type: {result!r}"


def _generate_one(show_output=True):
    """Generate ONE prompt's code (the body of a single Generate stage). Isolation is decided by
    the per-prompt `generate_in_workspace` flag:
      - normal prompts -> cwd = their OWN empty, cleared app dir, so the agent can't read other
        prompts' generated code / their run_prompt.md (headless RUN_CONTRACT: "never use nv3dsink")
        and mimic it -- that cross-prompt bleed was making every app come out fakesink;
      - flagged prompts run in the shared /workspace WITHOUT clearing: import_vision's skill manages
        models/ itself, and the rtvi microservice stage builds ON TOP of the core app in rtvi_app."""
    klog(
        f"\n=== Generating for prompt: {SELECTED_PROMPT_ID} "
        f"(artifact: {selected['preferred_artifact']}) ===",
        "step",
    )
    print_plan()
    build_agent_prompt()
    in_ws = bool(selected.get("generate_in_workspace"))
    app_dir_ctr = f"/workspace/{selected['output_dir']}"
    if not in_ws:  # clear + (re)create empty & agent-writable -- a clean slate every Generate
        rc = dexec_root(
            f"rm -rf {shlex.quote(app_dir_ctr)} && mkdir -p {shlex.quote(app_dir_ctr)} && "
            f"chown agent:agent {shlex.quote(app_dir_ctr)}",
            capture_output=True,
        )
        if rc.returncode != 0:
            raise SystemExit(
                f"Could not prepare an isolated Generate dir ({app_dir_ctr}): "
                f"{(rc.stderr or '').strip()[-300:]}"
            )
    timeout_used = selected.get("generate_timeout") or int(os.environ.get("AGENT_TIMEOUT", "3000"))
    rc = run_agent(
        timeout=selected.get("generate_timeout"),  # import_vision needs a longer budget
        workdir="/workspace" if in_ws else app_dir_ctr,
        synchronous=bool(selected.get("generate_result")),
    )
    # Show whatever was produced, then judge success by the presence of the Generate DELIVERABLE.
    # On failure RAISE (SystemExit) so the step banner shows FAILED and 'generated' stays unset (which
    # gates the Run step) -- rather than letting the display below falsely imply success. The agent
    # can end its turn early and leave an empty / half-done Generate: most often it defers with a
    # scheduled wake-up, which makes `claude -p` exit 0 expecting an external re-invocation this
    # headless harness never provides, so the remaining work never runs.
    # Per-prompt Generate-phase display rule. Default: dump the generated code inline. Prompts whose
    # output is a report + config files (not an app to read) set generate_display="download" -- we
    # still give the download (minus the huge model/engine binaries), but skip the inline dump; the
    # report itself is rendered at the Run step via show_results().
    if not show_output:
        code_or_files_produced = True
    elif selected.get("generate_display", "code") == "download":
        print(f"\n=== Generated files for '{SELECTED_PROMPT_ID}' (report + configs) ===")
        print(
            "Not dumped inline -- this prompt's output is a report + config files. "
            "The report is rendered at the Run step; download link below."
        )
        code_or_files_produced = show_generated_code(show_code=False, exclude_large_mb=25)
    else:
        print("\n=== Generated code (your prompt -> code) ===")
        # Compact the Codex process transcript in run_agent(), but keep the
        # generated code preview available. show_generated_code() already keeps
        # the first primary file open and folds the remaining files.
        code_or_files_produced = show_generated_code()

    if selected.get("generate_result"):
        # Measurement-driven prompts succeed only when their real report exists. Source files,
        # engines, or an exit code alone are partial progress, not the requested deliverable.
        produced, missing = _generate_result_status(selected)
    else:
        produced = code_or_files_produced
        missing = "no application code was written to the app directory"

    # Success requires BOTH a clean agent exit AND the real deliverable. Checking the deliverable
    # alone is not enough: on a watchdog timeout the agent is SIGKILLed (rc=-9) mid-run, yet partial
    # artifacts it already wrote (e.g. the app .py) sit on disk -- a file-presence-only check would
    # then falsely report success. A non-zero rc means the run did not finish; fail regardless.
    killed = rc not in (0, None)
    if killed or not produced:
        if killed:
            reason = f"the agent process did not exit cleanly (returncode={rc})" + (
                f" -- it hit the {timeout_used}s Generate timeout; raise AGENT_TIMEOUT in Step 1 "
                "and re-run" if rc == -9 else ""
            )
        else:
            reason = (
                f"{missing} -- the agent exited before producing the required final artifact; "
                "inspect agent_run.log for the first incomplete or failed stage"
            )
        raise SystemExit(
            f"Generate did not complete for '{SELECTED_PROMPT_ID}': {reason}. Re-run Generate; "
            "cached artifacts under the app dir are reused, so the retry is faster."
        )


def generate():
    """Generate from the user's prompt while leaving implementation/validation choices to the agent.

    Catalog entries that declare a measured `generate_result` must complete that report in this
    one-shot invocation. Other prompts normally author code here and receive formal runtime
    validation in the Run step, without a Generate-phase prohibition.

    A combined dropdown pick (e.g. `rtvi_vlm_core_app & rtvi_vlm_openapi_spec`) sets
    `selected_sequence`; we then generate each stage IN ORDER on the shared output dir -- the core
    app first (fresh, isolated), then the microservice on top of it (generate_in_workspace) -- and
    leave the selection on the FINAL stage so Run + show_results target the finished software."""
    global selected, SELECTED_PROMPT_ID
    sync_workspace_to_container()  # reuse-safe: write where the (reused) container maps /workspace
    if selected_sequence:
        seq = list(selected_sequence)
        for n, sub_id in enumerate(seq):
            selected = catalog_by_id[sub_id]
            SELECTED_PROMPT_ID = sub_id
            klog(f"\n----- Stage {n + 1}/{len(seq)}: {sub_id} -----", "step")
            _generate_one(show_output=n == len(seq) - 1)
        selected = catalog_by_id[seq[-1]]  # final stage = the deliverable Run/results target
        SELECTED_PROMPT_ID = seq[-1]
    else:
        _generate_one()


_RUN_COPY_EXCLUDE_DIRS = (
    "venv",
    ".venv",
    "node_modules",
    "__pycache__",
    ".cache",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".git",
)


def _run_copy_command(source, destination):
    """Build the container-side command that creates a bounded Run copy.

    Application files are copied by default so prompt-specific formats and resources survive.
    Only non-portable runtime environments, dependency trees, and disposable caches are excluded.
    """
    src = shlex.quote(str(source))
    dst = shlex.quote(str(destination))
    excludes = " ".join(
        f"--exclude={shlex.quote(f'./{name}')} --exclude={shlex.quote(f'*/{name}')}"
        for name in _RUN_COPY_EXCLUDE_DIRS
    )
    return (
        f"set -o pipefail; rm -rf {dst} && mkdir -p {dst} && "
        f"tar -C {src} {excludes} -cf - . | tar -C {dst} -xf -"
    )


def _prepare_run_copy(source, destination):
    """Create an isolated Run copy and fail instead of accepting a partial tree."""
    klog(
        "\nPreparing isolated Run copy (excluding virtual environments and caches) ...",
        "step",
    )
    result = dexec(_run_copy_command(source, destination), capture_output=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        if detail:
            detail = "\n".join(detail.splitlines()[-20:])
        raise SystemExit(
            f"Could not prepare the Run copy from {source} to {destination}."
            + (f"\n{detail}" if detail else "")
        )
    klog("Run copy ready.", "ok")


def run_and_fix():
    """Run-step core: RE-INVOKE the agent with the Run-phase prompt -- it (re)creates run_demo.sh,
    adapts the code for this headless env, runs it, and FIXES until it produces the artifact.

    WYSIWYG: the agent runs+fixes on a COPY of the generated app (under the run-out dir), so the
    code shown in the Generate step stays exactly as generated -- run-time tweaks don't rewrite it."""
    if selected.get("generate_result"):
        # Measurement-driven prompts already produced their final report during Generate.
        klog(
            f"\n{selected['id']} completed its measured report during Generate; nothing to "
            "re-run here -- showing the saved result below.",
            "step",
        )
        return
    if ctr_run_out is None:
        build_agent_prompt()
    src = f"/workspace/{selected['output_dir']}"
    run_dir = (
        f"{ctr_run_out}/app"  # copy the generated app here; agent works on this copy
    )
    _prepare_run_copy(src, run_dir)
    klog(
        "\nRunning ...",
        "step",
    )
    run_agent(
        prompt_path=build_run_prompt(run_dir),
        log_name="run_agent.log",
        timeout=RUN_DEMO_TIMEOUT,
    )


def run_and_view():
    """Run step: start the services, re-invoke the agent to RUN + FIX the app, then show results."""
    klog(
        "\nPreparing services/infra for this prompt (pull image + model, start services) ...",
        "step",
    )
    prepare_services()
    run_and_fix()
    print("\nResults:")
    show_results()  # for the rtvi service_code prompt this also curls the microservice


def cleanup_containers(remove_outputs=None):
    """Stop and remove the ds-agent-* containers (incl. the working container).

    Set remove_outputs=True (or REMOVE_AGENT_OUTPUTS=1) to also wipe OUTPUT_ROOT."""
    # (ffmpeg RTSP loopers live inside ds-agent-work, so removing it stops them too.)
    names = ["ds-agent-work", "ds-agent-kafka", "ds-agent-mediamtx", "ds-agent-vllm",
             "ds-agent-nats", "ds-agent-nats-sub"]
    for name in names:
        subprocess.run(
            ["docker", "rm", "-f", name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    print("Stopped ds-agent-* containers (including the working container).")
    if remove_outputs is None:
        remove_outputs = _bool_env("REMOVE_AGENT_OUTPUTS", False)
    if remove_outputs:
        shutil.rmtree(OUTPUT_ROOT, ignore_errors=True)
        print(f"Removed {OUTPUT_ROOT}.")
    else:
        print(
            f"Keeping {OUTPUT_ROOT}. Pass remove_outputs=True (or REMOVE_AGENT_OUTPUTS=1) to remove it."
        )


# ---------------------------------------------------------------------------
# ipywidgets bootstrap. The notebook's step controls are built inline in each
# cell (see build_notebook.py); this makes ipywidgets importable in the kernel
# before those inline widgets render. Imported LAZILY so importing this module
# never requires ipywidgets to be installed.
# ---------------------------------------------------------------------------
def ensure_ipywidgets():
    """Make ipywidgets importable in THIS kernel so the inline widget cells render. No-op when
    already present; otherwise pip-installs it once (best-effort). Called from each step cell's
    boot block so 'load the kernel' and 'show the controls' are a SINGLE action (no separate
    loader cell)."""
    try:
        import ipywidgets  # noqa: F401

        return True
    except ImportError:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "ipywidgets"], check=False
        )
    try:
        import ipywidgets  # noqa: F401

        return True
    except ImportError:
        print(
            "Could not load ipywidgets; if the step cells show no controls, restart the kernel (Kernel > Restart)."
        )
        return False
