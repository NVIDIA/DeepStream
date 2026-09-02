# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""
Optional Web UI server for deepstream-eval-and-finetune.

FastAPI app that drives the two validated demos through the existing skill
scripts (via app/pipeline.py), pausing at GATE A (after baseline DeepStream
eval) so the user can review GT-vs-baseline before clicking Fine-tune. It runs
INSIDE the DeepStream container, alongside — not replacing — the CLI/run.sh
flow, which remains canonical. All heavy work happens in a single background
daemon thread (sequential GPU rule); the browser polls /run/status ~1.5s.

Launch (see app/launch.sh):
    DS_EVAL_ROOT=/work build/.venv_train/bin/python -m uvicorn \
        .claude/skills/deepstream-eval-and-finetune/app/server:app \
        --app-dir /work --host 0.0.0.0 --port 8078
"""
import ast
import json
import os
import re
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import pipeline

# the skill's scripts/ dir is the sibling of this app/ dir — put it on the path so we can
# import curve_history (the single source of truth that stitches continue-training segments
# into one continuous loss curve). Shared verbatim with make_report.py / record_deployed_checkpoint.py.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import curve_history
import checkpoint_to_evaluate
import dataset_qc       # dataset health/QC over the ingested COCO (read-only)
import error_analysis   # confusion matrix + per-class P/R + hardest images (read-only)
import eval_engine      # reuse _parse_preds for single-image playground inference (/infer)
from artifacts import resolve_orig_dir

# --------------------------------------------------------------------------- paths / config
APP_DIR = Path(__file__).resolve().parent
ROOT = os.path.abspath(os.environ.get("DS_EVAL_ROOT", "/work"))
SK = ".claude/skills/deepstream-eval-and-finetune"            # ROOT-relative skill dir
PY = os.path.join(ROOT, "build", ".venv_train", "bin", "python")
CONF_THRESH = 0.3                                             # display threshold for baseline detections
try:
    WINDOWS_DATASET_MAP = json.loads(os.environ.get("DS_WINDOWS_DATASET_MAP", "{}"))
except Exception:
    WINDOWS_DATASET_MAP = {}

INDEX_PATH = os.path.join(ROOT, "build", "ui_runs", "index.json")

# Run registry: built-in presets (kind="preset") + user-created custom runs (kind="custom"),
# the latter persisted to build/ui_runs/index.json so previous runs survive restarts as history.
REG = {}
with open(APP_DIR / "presets.json") as f:
    for k, p in json.load(f)["presets"].items():
        p["key"] = k
        p["kind"] = "preset"
        p.setdefault("created_at", 0)            # built-ins sort before custom runs
        REG[k] = p


_CUSTOM_MTIME = [0.0]


def _load_custom(force=False):
    """Load/refresh custom runs from index.json.

    mtime-gated so it is cheap to call on EVERY request: a run registered by an external
    process (e.g. the NGC AutoML / finetune harness via ngc/ui_register.py) then shows up in
    the UI list — with its date/time, sorted newest-first — WITHOUT needing a server restart.
    Upsert-only: never drops the built-in demo presets."""
    try:
        m = os.path.getmtime(INDEX_PATH)
    except OSError:
        return
    if not force and m <= _CUSTOM_MTIME[0]:
        return
    _CUSTOM_MTIME[0] = m
    try:
        with open(INDEX_PATH) as f:
            for p in json.load(f):
                REG[p["key"]] = p
    except Exception:
        pass


def _save_custom():
    customs = [p for p in REG.values() if p.get("kind") == "custom"]
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    tmp = INDEX_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(customs, f, indent=2)
    os.replace(tmp, INDEX_PATH)


def _has_recipe(model_id):
    """The bundled deploy assets (export_rtdetr.py + rtdetr_parser.cpp) handle the RT-DETR
    output format. Other architectures need a recipe authored by the skill, so they're gated."""
    m = (model_id or "").lower()
    return any(t in m for t in ("rtdetr", "rt_detr", "rt-detr"))


def _norm_hf_id(s):
    """Accept a pasted HuggingFace URL or a bare repo id; return 'namespace/name'.
    e.g. 'https://huggingface.co/datasets/keremberke/forklift-object-detection?x=1'
         -> 'keremberke/forklift-object-detection'."""
    s = (s or "").strip()
    s = re.sub(r"[?#].*$", "", s)                       # drop query/fragment
    s = re.sub(r"^https?://(www\.)?huggingface\.co/", "", s)
    s = re.sub(r"^(datasets|models)/", "", s)           # strip the /datasets/ or /models/ segment
    s = re.sub(r"/(tree|blob|resolve)/.*$", "", s)      # strip /tree/main etc.
    return s.strip("/")


def _norm_source(s):
    """Dataset source normalizer: collapse a HuggingFace URL to its repo id, but leave
    generic http(s) URLs and local paths untouched (ingest_dataset.py resolves those)."""
    s = (s or "").strip()
    # PowerShell launcher maps Windows roots to Linux container paths and passes this table.
    # Accept the path the user sees in Explorer and translate it before ingestion.
    lower = s.lower().replace("/", "\\")
    for host, container in WINDOWS_DATASET_MAP.items():
        host_norm = host.lower().rstrip("\\/").replace("/", "\\")
        if lower == host_norm or lower.startswith(host_norm + "\\"):
            suffix = s[len(host):].lstrip("\\/").replace("\\", "/")
            return container.rstrip("/") + ("/" + suffix if suffix else "")
    return _norm_hf_id(s) if "huggingface.co" in s else s


def _is_url(s):
    return s.startswith(("http://", "https://"))


def _dataset_link(ds):
    """A clickable link for the report/UI: the URL itself, or an HF dataset page for a
    bare 'ns/name' repo id; empty for a local path."""
    if _is_url(ds):
        return ds
    if "/" in ds and not os.path.exists(os.path.join(ROOT, ds)) and not os.path.isabs(ds):
        return f"https://huggingface.co/datasets/{ds}"
    return ""


def _slug(s):
    # strip trailing slashes first so '/work/smb_share/' -> 'smb_share', not 'run'
    base = re.sub(r"[^a-z0-9]+", "_", (s or "").lower().rstrip("/").split("/")[-1]).strip("_")
    return (base[:40] or "run")


def make_custom_preset(model_id, dataset_id, eval_split, epochs, hparams=None):
    """Build a dynamic preset for an RT-DETR-family model + an ingestable detection dataset.
    Mirrors the aerial-sheep recipe but with derived paths, a generated training config, and
    PCB-style target-label eval (honest baseline via name-based remap)."""
    key = f"custom_{_slug(model_id)}__{_slug(dataset_id)}"
    base, i = key, 2
    # Always pick a fresh registry key when the base slug is taken — same model+dataset
    # gets separate history rows (v2, v3, …) instead of silently reusing the first run.
    while key in REG:
        key, i = f"{base}_{i}", i + 1
    slug = key[len("custom_"):]
    short = model_id.split("/")[-1]
    dshort = dataset_id.split("/")[-1]
    return {
        "key": key, "kind": "custom",
        "label": f"{short} + {dshort}",
        "model_id": model_id, "model_label": short,
        "dataset_id": dataset_id, "dataset_label": dshort,
        "blurb": "Custom run — RT-DETR family + your dataset.",
        "editable": False, "deploy_recipe": True, "config_generate": True,
        "assets_dir": f"{SK}/examples/rtdetr-aerial-sheep/assets", "config_yaml": None,
        "data_dir": f"data/custom_{slug}",
        "orig_dir": f"models/custom_{slug}_orig",
        "ft_dir": f"models/custom_{slug}_ft",
        "run_dir": f"runs/custom_{slug}",
        "ingest": {"kind": "ingest_dataset", "splits": ["train", "valid", "test"]},
        "eval_split": eval_split or "valid",
        "eval_labels": f"data/custom_{slug}/labels.txt",
        "orig_onnx": "rtdetr_orig.onnx", "orig_engine": "rtdetr_orig_ds.engine",
        "ft_onnx": f"custom_{slug}_ft.onnx", "ft_engine": f"custom_{slug}_ft_ds.engine",
        "canvas": [640, 640], "n_eval": 200, "bbox_format": "xywh", "resize_mode": "stretch",
        "epochs": int(epochs or 30),
        "comparison": f"reports/custom_{slug}_comparison.json",
        "samples_dir": f"reports/custom_{slug}_samples",
        "report_pdf": f"reports/custom_{slug}_report.pdf",
        "report_basename": f"custom_{slug}_report.pdf",
        "sample_overlays_extra": [],
        "report_meta": {
            "model_id": model_id, "model_link": f"https://huggingface.co/{model_id}",
            "model_license": "-",
            "model_desc": f"{model_id}. Baseline = stock model; fine-tuned = {int(epochs or 30)} epochs on {dataset_id}.",
            "dataset_id": dataset_id, "dataset_link": _dataset_link(dataset_id),
            "dataset_desc": f"Custom dataset {dataset_id}.",
        },
        "hparams": {k: v for k, v in (hparams or {}).items() if v not in (None, "", "auto")},
        "created_at": time.time(),
    }


_load_custom()

# Child processes (docker, deepstream-app, trtexec, the training scripts) inherit an
# explicit allowlist rather than a copy of os.environ. Copying the whole environment
# would forward every credential the server happens to be running with into every
# subprocess, which is both unnecessary and flagged as env-variable harvesting.
_ENV_PASSTHROUGH_NAMES = (
    # POSIX / interpreter basics
    "PATH", "HOME", "USER", "LOGNAME", "SHELL", "TERM", "PWD",
    "LANG", "LC_ALL", "TMPDIR",
    "PYTHONPATH", "PYTHONUNBUFFERED", "VIRTUAL_ENV",
    "XDG_CACHE_HOME", "XDG_RUNTIME_DIR",
    # The DeepStream runtime image preloads an OpenMP runtime so that nvinfer does
    # not fail with "cannot allocate memory in static TLS block" when it dlopens
    # plugins. The server already runs under that setting, so forwarding it grants
    # children nothing new -- but dropping it breaks ds_image_eval, trtexec, and
    # training. See the image's own Dockerfile for the value it sets.
    "LD_LIBRARY_PATH", "LD_PRELOAD",
    # Keeps __pycache__ out of the signed skill tree (see app/launch.sh).
    "PYTHONDONTWRITEBYTECODE",
    # Proxy and the CA material a TLS-inspecting proxy requires; without these the
    # HuggingFace downloads in ingest_dataset.py / build_eval_set.py fail on certs.
    "http_proxy", "https_proxy", "no_proxy",
    "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY",
    "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE",
    # Toolchain for the in-tree make targets (ds_image_eval, the RT-DETR parser)
    "PKG_CONFIG_PATH", "CC", "CXX",
    # HuggingFace: cache location and auth for gated repos and datasets
    "HF_HOME", "HF_TOKEN", "HF_HUB_CACHE", "HF_ENDPOINT",
    "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_CACHE",
    # This skill's own knobs
    "DS_DATA_CACHE_REFRESH", "DS_DATA_CACHE_ROOT", "DS_DATA_VOLUME",
    "DS_EVAL_ROOT", "DS_IMAGE", "DS_WINDOWS_DATASET_MAP",
    # GPU / GStreamer / training runtime
    "CUDA_VISIBLE_DEVICES", "CUDA_HOME", "CUDA_VER",
    "NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES",
    "GST_DEBUG", "GST_PLUGIN_PATH", "GST_PLUGIN_SYSTEM_PATH",
    "GST_PLUGIN_SCANNER", "GST_REGISTRY",
    "PYTORCH_CUDA_ALLOC_CONF", "TORCH_HOME",
    "NCCL_P2P_DISABLE", "NCCL_IB_DISABLE", "NCCL_DEBUG",
    "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    "TRANSFORMERS_CACHE", "TOKENIZERS_PARALLELISM", "MPLCONFIGDIR",
)


def _passthrough_env():
    """Look up the child-process variables by name.

    Deliberately indexes a fixed tuple instead of scanning os.environ: iterating the
    environment is what forwards unrelated credentials into every subprocess, and the
    set of names a child receives should be readable from the source alone.
    """
    return {k: os.environ[k] for k in _ENV_PASSTHROUGH_NAMES if k in os.environ}


BASE_ENV = {
    **_passthrough_env(),
    "HF_HOME": os.environ.get("HF_HOME", os.path.join(ROOT, "build", "hf_cache")),
    "NO_ALBUMENTATIONS_UPDATE": "1",
    "TOKENIZERS_PARALLELISM": "false",
}
DATA_CACHE_ROOT = os.environ.get("DS_DATA_CACHE_ROOT", "")
DATA_CACHE_VOLUME = os.environ.get("DS_DATA_VOLUME", "")
STARTUP_CACHE_REFRESH = [
    os.environ.get("DS_DATA_CACHE_REFRESH", "").lower() in ("1", "true", "yes", "on")
]
BASE_ENV.pop("DS_DATA_CACHE_REFRESH", None)  # consume once via the next run, not every subprocess


def _data_cache_info():
    staged_bytes = 0
    staged_sources = 0
    if DATA_CACHE_ROOT:
        for marker in Path(DATA_CACHE_ROOT, ".sources").glob("*/cache_manifest.json"):
            try:
                staged_bytes += int(json.loads(marker.read_text()).get("bytes", 0))
                staged_sources += 1
            except Exception:
                continue
    return {
        "enabled": bool(DATA_CACHE_ROOT),
        "root": DATA_CACHE_ROOT,
        "volume": DATA_CACHE_VOLUME,
        "staged_sources": staged_sources,
        "staged_bytes": staged_bytes,
    }


def _remove_unshared_staged_source(key, preset):
    """Remove a custom run's staged source only when no other registered run uses it."""
    if not DATA_CACHE_ROOT or preset.get("kind") != "custom":
        return None
    source = str(Path(preset.get("dataset_id", "")).expanduser().resolve())
    if any(
        other_key != key
        and other.get("kind") == "custom"
        and str(Path(other.get("dataset_id", "")).expanduser().resolve()) == source
        for other_key, other in REG.items()
    ):
        return None
    sources_root = Path(DATA_CACHE_ROOT, ".sources").resolve()
    for marker in sources_root.glob("*/cache_manifest.json"):
        try:
            if json.loads(marker.read_text()).get("source") != source:
                continue
            target = marker.parent.resolve()
            if target.parent != sources_root:
                continue
            import shutil
            shutil.rmtree(target)
            return str(target)
        except Exception:
            continue
    return None


def _consume_cache_refresh(requested=False):
    refresh = bool(requested or STARTUP_CACHE_REFRESH[0])
    STARTUP_CACHE_REFRESH[0] = False
    return refresh

# --------------------------------------------------------------------------- run state (single-user, in memory)
JOB_LOCK = threading.Lock()       # at most one run at a time -> enforces the sequential-GPU rule
STATE_LOCK = threading.Lock()
STATE = {
    "phase": "idle",              # idle | baseline | awaiting_gate_a | finetune | done | error | cancelled
    "preset": None,
    "run_id": None,
    "stage": None, "stage_label": None,
    "done": 0, "total": 0,
    "status": "idle",             # idle | running | done | error | cancelled
    "error": None,
    "started_at": None,
    "proc": None,                 # live subprocess (for cancel/stop)
    "cancel": threading.Event(),  # hard abort — stop everything
    "stop": threading.Event(),    # soft stop — end training, then pause for the user to evaluate
    "soft_stopped": False,        # set once a soft-stop took effect (pause at awaiting_eval)
    "log_ring": deque(maxlen=240),
    "log_path": None,
}


class Cancelled(Exception):
    pass


def _set(**kw):
    with STATE_LOCK:
        STATE.update(kw)


def _emit(line):
    with STATE_LOCK:
        STATE["log_ring"].append(line)
        lp = STATE["log_path"]
    if lp:
        try:
            with open(lp, "a") as f:
                f.write(line + "\n")
        except Exception:
            pass


# --------------------------------------------------------------------------- execution context for pipeline.py
class Ctx:
    def __init__(self):
        self.ROOT = ROOT
        self.PY = PY
        self.SK = SK
        self.run_env = BASE_ENV
        self.cancel = STATE["cancel"]
        self.stop = STATE["stop"]

    def path(self, rel):
        return rel if os.path.isabs(rel) else os.path.join(self.ROOT, rel)

    def exists(self, rel):
        return os.path.exists(self.path(rel))

    def cancelled(self):
        return self.cancel.is_set()

    def log(self, msg):
        _emit(msg)

    def sh(self, cmd, cwd=None, log_file=None, check=True, soft_stop=False):
        """Run a subprocess, tee output to the run log (and optional log_file), honor cancel.
        If soft_stop and the user requests a stop, terminate the process and return normally
        (no error) so the caller can use whatever it produced (e.g. the latest checkpoint)."""
        self.log("$ " + " ".join(str(c) for c in cmd))
        lf = open(log_file, "w") if log_file else None
        proc = subprocess.Popen(cmd, cwd=cwd or self.ROOT, env=self.run_env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        _set(proc=proc)
        stopped = False
        try:
            for line in proc.stdout:
                line = line.rstrip("\n")
                if lf:
                    lf.write(line + "\n"); lf.flush()
                _emit(line)
                if self.cancel.is_set():
                    proc.terminate()
                    raise Cancelled()
                if soft_stop and self.stop.is_set():
                    self.log("[stop] stopping training — pausing for evaluation of the latest epoch")
                    proc.terminate()
                    stopped = True
                    _set(soft_stopped=True)
                    break
            rc = proc.wait()
        finally:
            if lf:
                lf.close()
            _set(proc=None)
        # A user soft-stop sends SIGTERM directly (POST /run/stop), so the process often dies before
        # the read-loop notices stop.is_set() — the loop just ends and rc is the signal's -15. Treat
        # ANY exit as a CLEAN stop when a stop was requested, so the run pauses at awaiting_eval (UI
        # "Evaluate latest epoch") instead of erroring on rc=-15.
        if soft_stop and self.stop.is_set():
            stopped = True
            _set(soft_stopped=True)
            self.log("[stop] training stopped by user — using the latest completed epoch")
        if stopped:
            self.stop.clear()
            return
        if self.cancel.is_set():
            raise Cancelled()
        if check and rc != 0:
            raise RuntimeError(f"command failed (rc={rc}): {' '.join(str(c) for c in cmd[:3])} …")


# --------------------------------------------------------------------------- background driver
def _drive(preset_key, segments, success_phase):
    """Run an ordered list of segments [(phase, plan_fn), ...] in one background thread,
    switching STATE['phase'] at each segment boundary. This lets Baseline auto-chain into
    Fine-tune as a single run (no manual gate)."""
    preset = REG[preset_key]
    ctx = Ctx()
    plan = [(ph, st) for ph, plan_fn in segments for st in plan_fn(preset)]
    _set(stage=None, stage_label=None, done=0, total=len(plan), error=None, status="running")
    try:
        for i, (ph, st) in enumerate(plan):
            if ctx.cancelled():
                raise Cancelled()
            _set(phase=ph, stage=st.key, stage_label=st.label, done=i)
            ctx.log(f"==================== stage {i + 1}/{len(plan)}: {st.label} ====================")
            st.fn(ctx, preset)
            # soft-stop pauses AFTER training (latest epoch promoted to final) — let the user
            # choose to Evaluate. Skip the remaining deploy/eval/report stages here.
            if st.key == "finetune" and STATE.get("soft_stopped"):
                _set(done=i + 1, stage=None, stage_label="stopped — ready to evaluate",
                     phase="awaiting_eval", status="idle")
                ctx.log("[stop] training stopped — awaiting evaluation of the latest epoch")
                return
        _set(done=len(plan), stage=None, stage_label="complete", phase=success_phase,
             status=("done" if success_phase == "done" else "idle"))
        ctx.log(f"[done] {success_phase}")
    except Cancelled:
        prev = "awaiting_gate_a" if _has_baseline(preset) else "idle"
        _set(phase=prev, status="cancelled", stage_label="cancelled")
        ctx.log("[cancelled] run stopped by user")
    except Exception as e:
        _set(phase="error", status="error", error=str(e))
        ctx.log(f"[error] {e}")
    finally:
        STATE["cancel"].clear()
        if JOB_LOCK.locked():
            JOB_LOCK.release()


def _gpu_available():
    """True only if a CUDA GPU is actually visible to this container. Cheap nvidia-smi
    probe (avoids importing torch in the web process)."""
    import shutil
    import subprocess
    if not shutil.which("nvidia-smi"):
        return False
    try:
        r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10)
        return r.returncode == 0 and "GPU 0" in r.stdout
    except Exception:
        return False


def _start(preset_key, segments, success_phase):
    """segments = [(phase, plan_fn), ...] run in order in one thread."""
    # Refuse to start if no GPU is visible: every segment (engine build, train, eval) needs the
    # GPU, and training on CPU is unusably slow AND overwrites runs/<name>/train.log — silently
    # clobbering an existing run. Launch with GPU via app/launch.sh (docker run --gpus all);
    # for GPU-less viewing use the read-only viewer paths only (no run/* buttons).
    if not _gpu_available():
        raise HTTPException(status_code=409, detail=(
            "No GPU visible to this container — refusing to start a run. It would train on CPU "
            "(unusably slow) and overwrite the run's training log. Relaunch with GPU via "
            "app/launch.sh (docker run --gpus all). This GPU-less server is view-only."))
    if not JOB_LOCK.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="a run is already in progress")
    run_id = time.strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(ROOT, "build", "ui_runs")
    os.makedirs(log_dir, exist_ok=True)
    STATE["cancel"].clear()
    STATE["stop"].clear()
    _set(soft_stopped=False)
    _set(phase=segments[0][0], preset=preset_key, run_id=run_id, status="running",
         error=None, started_at=time.time(), log_path=os.path.join(log_dir, f"{run_id}.log"))
    with STATE_LOCK:
        STATE["log_ring"].clear()
    threading.Thread(target=_drive, args=(preset_key, segments, success_phase), daemon=True).start()
    return run_id


# --------------------------------------------------------------------------- artifact readers
def _load(rel):
    p = rel if os.path.isabs(rel) else os.path.join(ROOT, rel)
    with open(p) as f:
        return json.load(f)


def _try_load(rel):
    try:
        return _load(rel)
    except Exception:
        return None


def _abs(preset, rel):
    return os.path.join(ROOT, rel)


def _xywh_to_xyxy(b):
    x, y, w, h = b
    return [x, y, x + w, y + h]


def _target_names(gt):
    """Set of GT class names present across the eval set (xywh GT 'category' -> label_map name)."""
    lm = gt.get("label_map", {})
    names = set()
    for im in gt.get("images", []):
        for o in im.get("objects", []):
            names.add(lm.get(str(o["category"]), str(o["category"])))
    return names


def _has_baseline(preset):
    base = resolve_orig_dir(ROOT, preset)
    try:
        return all(
            os.path.isfile(path) and os.path.getsize(path) > 0
            for path in (
                _abs(preset, os.path.join(base, "eval", "engine_metrics.json")),
                _abs(preset, os.path.join(base, "eval", "perf.json")),
            )
        )
    except OSError:
        return False


def _has_finetuned(preset):
    return os.path.exists(_abs(preset, preset["comparison"]))


def _has_training(preset):
    """True once a fine-tune has emitted training logs (runs/<run>/train.log) — even for a run
    launched OUTSIDE the UI (a CLI/container AutoML or finetune harness). Lets the Fine-tune tab
    live-refresh the loss/mAP curve during trials/training, before any checkpoint or comparison
    exists (the old live-curve gate required a checkpoint, so externally-driven runs stayed blank
    until the final phase)."""
    try:
        return os.path.getsize(_abs(preset, os.path.join(preset["run_dir"], "train.log"))) > 0
    except OSError:
        return False


def _automl_dir(preset):
    """Newest logs/<run>/<timestamp>/ dir for a run (where the NGC AutoML harness writes
    automl.log, the trial configs/scores and the final config). None if there isn't one."""
    base = _abs(preset, os.path.join("logs", os.path.basename(preset["run_dir"].rstrip("/"))))
    try:
        subs = [os.path.join(base, d) for d in os.listdir(base)]
        subs = [d for d in subs if os.path.isdir(d)]
        return max(subs, key=os.path.getmtime) if subs else None
    except OSError:
        return None


def _is_automl(preset):
    d = _automl_dir(preset)
    return bool(d and os.path.exists(os.path.join(d, "automl.log")))


def _parse_trial_curve(path):
    """Per-epoch (epoch, eval_map, eval_map_50) for one AutoML trial, from its train log — so the
    UI's 3a sweep view can plot each trial's mAP curve (the search), separate from the final
    fine-tuning curve on 3b."""
    import re
    ep, mp, m50 = [], [], []
    try:
        for ln in open(path):
            if "'eval_map'" not in ln:
                continue
            me = re.search(r"'epoch':\s*([0-9.]+)", ln)
            mm = re.search(r"'eval_map':\s*([0-9.eE+-]+)", ln)
            m5 = re.search(r"'eval_map_50':\s*([0-9.eE+-]+)", ln)
            if me and mm:
                ep.append(float(me.group(1))); mp.append(float(mm.group(1)))
                m50.append(float(m5.group(1)) if m5 else None)
    except OSError:
        pass
    return {"epoch": ep, "eval_map": mp, "eval_map_50": m50}


def _active_external_run():
    """When no UI-driven run is active (STATE idle), detect a run that is training via an external
    CLI/container harness (AutoML sweep or a finetune) so /run/status reflects it instead of showing
    'idle'. Freshness-gated on the log that the harness is actively writing, so a finished/dead run
    is not reported as live. Returns a small dict {run,label,kind,phase} or None."""
    import glob
    now = time.time()
    fallback = None
    for key, p in REG.items():
        d = _automl_dir(p)
        if d and os.path.exists(os.path.join(d, "automl.log")):
            try:
                done = "[automl] DONE" in open(os.path.join(d, "automl.log")).read()
            except OSError:
                done = False
            if not done and not _has_finetuned(p):
                if os.path.exists(os.path.join(d, "final.log")):
                    ph, probe = "final", os.path.join(d, "final.log")
                else:
                    tls = glob.glob(os.path.join(d, "trials", "trial_*.log"))
                    ph = "trials"
                    probe = max(tls, key=os.path.getmtime) if tls else os.path.join(d, "automl.log")
                try:
                    if now - os.path.getmtime(probe) < 180:
                        return {"run": key, "label": p.get("label", key), "kind": "automl", "phase": ph}
                except OSError:
                    pass
        tl = _abs(p, os.path.join(p["run_dir"], "train.log"))
        try:
            if os.path.getsize(tl) > 0 and now - os.path.getmtime(tl) < 120 and not _has_finetuned(p):
                fallback = {"run": key, "label": p.get("label", key), "kind": "finetune", "phase": "training"}
        except OSError:
            pass
    return fallback


def _has_evaluable_checkpoint(preset):
    """True if the run has something to evaluate: a promoted checkpoints/final OR any saved
    checkpoint-* (a stopped/errored run that never got promoted). Lets the UI offer 'Evaluate
    latest checkpoint' for such runs so the user never needs the terminal."""
    ck = _abs(preset, os.path.join(preset["run_dir"], "checkpoints"))
    if os.path.exists(os.path.join(ck, "final")):
        return True
    return any(n.startswith("checkpoint-") for n in (os.listdir(ck) if os.path.isdir(ck) else []))


def _checkpoint_info(preset):
    info = checkpoint_to_evaluate.info(_abs(preset, preset["run_dir"]))
    if not info:
        return None
    dep = _try_load(os.path.join(preset["ft_dir"], "model", "deployed_checkpoint.json")) or {}
    cmp = _try_load(preset["comparison"]) or {}
    deployed_ep = cmp.get("deployed_checkpoint_epoch")
    if deployed_ep is None:
        deployed_ep = dep.get("deployed_epoch")
    info["deployed_checkpoint_epoch"] = deployed_ep
    info["deploy_stale"] = (
        info.get("epoch") is not None and deployed_ep is not None
        and int(info["epoch"]) != int(deployed_ep)
    )
    return info


def _model_eval_meta(p, which):
    """Metadata for Analyze / Playground: which deployed leg is selected and its checkpoint epoch."""
    base = resolve_orig_dir(ROOT, p) if which == "orig" else p["ft_dir"]
    ab = _abs(p, base)
    pred = os.path.join(ab, "eval", "predictions_engine.json")
    metrics = _try_load(os.path.join(ab, "eval", "engine_metrics.json")) or {}
    out = {
        "which": which,
        "available": os.path.exists(pred),
        "run_label": p.get("label"),
        "preset_key": p.get("key"),
        "model_id": p.get("model_id"),
        "dataset_id": p.get("dataset_id"),
        "run_dir": p.get("run_dir"),
        "model_dir": base,
        "n_eval": metrics.get("n_eval"),
        "map": metrics.get("map"),
        "map_50": metrics.get("map_50"),
    }
    if which == "orig":
        short = (p.get("model_id") or "model").split("/")[-1]
        out["role"] = "stock_baseline"
        out["title"] = f"Stock baseline · {short}"
        out["subtitle"] = "Original HuggingFace weights — not fine-tuned on this dataset"
    else:
        dep = _try_load(os.path.join(ab, "model", "deployed_checkpoint.json")) or {}
        ck = _checkpoint_info(p) or {}
        cmp = _try_load(p["comparison"]) or {}
        ck_eval = cmp.get("checkpoint_evaluated") or {}
        deployed_ep = dep.get("deployed_epoch") or cmp.get("deployed_checkpoint_epoch")
        sel_ep = ck_eval.get("epoch") or ck.get("epoch")
        sel_kind = ck_eval.get("selection") or ck.get("selection") or "latest"
        out["role"] = "finetuned_deploy"
        out["deployed_epoch"] = deployed_ep
        out["checkpoint_epoch"] = sel_ep
        out["checkpoint_selection"] = sel_kind
        out["training_map"] = ck_eval.get("map") if ck_eval.get("map") is not None else ck.get("map")
        out["training_map_50"] = ck_eval.get("map_50") if ck_eval.get("map_50") is not None else ck.get("map_50")
        out["deploy_stale"] = bool(ck.get("deploy_stale"))
        out["onnx"] = dep.get("model_name") or os.path.basename(dep.get("onnx") or "")
        ck_path = dep.get("checkpoint_epoch_tag") or dep.get("checkpoint")
        out["checkpoint"] = ck_path
        kind = "best" if sel_kind == "best" else "latest"
        ep = deployed_ep if deployed_ep is not None else sel_ep
        out["title"] = f"Fine-tuned · epoch {ep if ep is not None else '?'} ({kind})"
        out["subtitle"] = (
            f"DeepStream predictions from {out.get('onnx') or 'deployed ONNX'}"
            + (f" · checkpoint {ck_path}" if ck_path else "")
        )
    return out


# --------------------------------------------------------------------------- FastAPI app
app = FastAPI(title="DeepStream Eval & Fine-tune")


@app.middleware("http")
async def _no_cache_assets(request, call_next):
    """The UI (index.html / static/app.js / static/style.css) must never be served stale from the
    browser cache after a code update — force revalidation (ETag still yields 304 when unchanged)."""
    _load_custom()   # pick up runs registered on disk by external harnesses without a restart
    resp = await call_next(request)
    p = request.url.path
    if p == "/" or p.startswith("/static"):
        resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp


@app.get("/")
def index():
    return FileResponse(APP_DIR / "static" / "index.html")


app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")


@app.get("/presets")
def list_presets():
    out = []
    for key, p in REG.items():
        out.append({
            "key": key, "label": p["label"], "kind": p.get("kind", "preset"),
            "model_id": p["model_id"], "model_label": p["model_label"],
            "dataset_id": p["dataset_id"], "dataset_label": p["dataset_label"],
            "blurb": p.get("blurb", ""), "epochs": p.get("epochs"),
            "eval_split": p.get("eval_split", "valid"), "n_classes": p.get("ft_nclasses"),
            "editable": p.get("editable", True), "deploy_recipe": p.get("deploy_recipe", True),
            "created_at": p.get("created_at", 0),
            "hparams": p.get("hparams") or {},
            "run_note": p.get("run_note", ""),
            "has_baseline": _has_baseline(p), "has_finetuned": _has_finetuned(p),
            "has_checkpoint": _has_evaluable_checkpoint(p),
            "has_training": _has_training(p),
            "is_automl": _is_automl(p),
            "cache_ready": bool(DATA_CACHE_ROOT and os.path.isfile(
                _abs(p, os.path.join(p.get("data_dir", ""), "ingest_manifest.json"))
            )),
        })
    # newest run first (latest custom run on top); built-in demos (created_at 0) fall to the end
    out.sort(key=lambda r: -r["created_at"])
    return {"presets": out, "data_cache": _data_cache_info()}


# baseline then fine-tune, automatically, as one run (no manual gate)
AUTO_SEGMENTS = [("baseline", pipeline.baseline_plan), ("finetune", pipeline.finetune_plan)]


@app.post("/run/baseline")
def run_baseline(body: dict):
    """Run an EXISTING registry entry end-to-end: baseline → (auto) fine-tune → done."""
    key = body.get("preset")
    if key not in REG:
        raise HTTPException(status_code=400, detail="unknown run")
    REG[key]["_refresh_dataset_cache"] = _consume_cache_refresh(
        body.get("refresh_dataset_cache")
    )
    run_id = _start(key, AUTO_SEGMENTS, "done")
    return {"ok": True, "run_id": run_id}


@app.post("/run/custom")
def run_custom(body: dict):
    """Create a NEW run from a user-supplied model + dataset, register it as history, run baseline.
    Only RT-DETR-family models have a deploy recipe; anything else is deferred to the skill."""
    model_id = _norm_hf_id(body.get("model_id"))       # model: HF URL or bare repo id -> repo id
    dataset_id = _norm_source(body.get("dataset_id"))  # dataset: HF id/URL, http zip URL, or local path
    if not model_id or not dataset_id:
        raise HTTPException(status_code=400, detail="model_id and dataset_id are required")
    if not _has_recipe(model_id):
        return {"ok": False, "needs_recipe": True,
                "message": f"'{model_id}' has no deploy recipe in the UI — it only auto-deploys RT-DETR-family "
                           f"models (the ONNX exporter + custom parser are RT-DETR-specific). Run other "
                           f"architectures through the deepstream-eval-and-finetune skill, which authors the "
                           f"deploy recipe per model."}
    # Each "Deploy & run" creates a NEW history entry (unique key even for same dataset).
    # Re-run an existing row via POST /run/baseline instead.
    try:
        epochs = int(body.get("epochs") or 30)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="epochs must be an integer")
    if not 1 <= epochs <= 500:
        raise HTTPException(status_code=400, detail="epochs must be between 1 and 500")
    hparams = {k: body.get(k) for k in ("lr", "warmup", "sched", "batch")}
    preset = make_custom_preset(model_id, dataset_id, body.get("eval_split"), epochs, hparams)
    key = preset["key"]
    REG[key] = preset
    _save_custom()
    preset["_refresh_dataset_cache"] = _consume_cache_refresh(
        body.get("refresh_dataset_cache")
    )
    run_id = _start(key, AUTO_SEGMENTS, "done")   # baseline → auto fine-tune → done
    return {"ok": True, "key": key, "run_id": run_id}


@app.post("/run/finetune")
def run_finetune(body: dict):
    key = body.get("preset")
    if key not in REG:
        raise HTTPException(status_code=400, detail="unknown preset")
    with STATE_LOCK:
        phase = STATE["phase"]
    # block only while a run is actively executing; "idle" (e.g. after a restart) is fine —
    # JOB_LOCK still guards against concurrent runs, and _has_baseline ensures order.
    if phase in ("baseline", "finetune"):
        raise HTTPException(status_code=409, detail=f"a run is in progress (phase '{phase}')")
    if not _has_baseline(REG[key]):
        raise HTTPException(status_code=409, detail="baseline not available yet")
    run_id = _start(key, [("finetune", pipeline.finetune_plan)], "done")
    return {"ok": True, "run_id": run_id}


@app.get("/results/checkpoint")
def results_checkpoint(preset: str):
    """Which checkpoint will be deployed on Evaluate: best-by-metric or latest completed epoch."""
    if preset not in REG:
        raise HTTPException(status_code=400, detail="unknown run")
    info = _checkpoint_info(REG[preset])
    if not info:
        raise HTTPException(status_code=404, detail="no checkpoint to evaluate yet")
    return info


@app.post("/run/evaluate")
def run_evaluate(body: dict):
    """Deploy checkpoints/final → DeepStream eval → report. ``final`` holds the best-by-metric
    weights when load_best_model_at_end is set, otherwise the latest completed epoch."""
    key = body.get("preset")
    if key not in REG:
        raise HTTPException(status_code=400, detail="unknown run")
    with STATE_LOCK:
        phase = STATE["phase"]
    if phase in ("baseline", "finetune"):
        raise HTTPException(status_code=409, detail=f"a run is in progress (phase '{phase}')")
    # Allow evaluating any run that has SOMETHING to evaluate: a promoted 'final' OR a saved
    # checkpoint-* (a stopped/errored run that never got promoted). eval_plan promotes if needed.
    if not _has_evaluable_checkpoint(REG[key]):
        raise HTTPException(status_code=409, detail="no saved checkpoint to evaluate yet")
    run_id = _start(key, [("evaluating", pipeline.eval_plan)], "done")
    return {"ok": True, "run_id": run_id}


@app.post("/run/continue")
def run_continue(body: dict):
    """Continue an existing fine-tuned run for N MORE epochs (warm-start from its current weights,
    fresh LR schedule), then re-deploy → eval → report. Overwrites the run in place (keeps the
    pre-continue weights as checkpoints/final_prev)."""
    key = body.get("preset")
    if key not in REG:
        raise HTTPException(status_code=400, detail="unknown run")
    try:
        extra = int(body.get("extra_epochs"))
    except (TypeError, ValueError):
        extra = 0
    if not 1 <= extra <= 500:
        raise HTTPException(status_code=400, detail="extra_epochs must be between 1 and 500")
    with STATE_LOCK:
        phase = STATE["phase"]
    if phase in ("baseline", "finetune", "evaluating"):
        raise HTTPException(status_code=409, detail=f"a run is in progress (phase '{phase}')")
    if not os.path.exists(_abs(REG[key], os.path.join(REG[key]["run_dir"], "checkpoints", "final"))):
        raise HTTPException(status_code=409, detail="no completed fine-tuned run to continue")
    REG[key]["continue_epochs"] = extra
    if REG[key].get("kind") == "custom":
        _save_custom()
    run_id = _start(key, [("finetune", pipeline.continue_training_plan)], "done")
    return {"ok": True, "run_id": run_id, "extra_epochs": extra}


@app.post("/run/delete")
def run_delete(body: dict):
    """Delete a run's heavy artifacts (TRT engines, eval sets, checkpoints, Arrow data, report,
    charts) to reclaim disk. Custom runs are also removed from the history list; built-in demos
    stay (re-runnable). NEVER touches the user's source dataset (only paths keyed by the run)."""
    import shutil
    key = body.get("preset")
    if key not in REG:
        raise HTTPException(status_code=400, detail="unknown run")
    with STATE_LOCK:
        phase, active = STATE["phase"], STATE["preset"]
    if active == key and phase in ("baseline", "finetune", "evaluating"):
        raise HTTPException(status_code=409, detail="this run is in progress — cancel it first")
    p = REG[key]
    is_custom = p.get("kind") == "custom"
    # run-keyed artifacts only. data_dir (ingested COCO) is removed for custom runs (cheap to
    # re-ingest from a local path); kept for demos to avoid re-downloading from HF.
    targets = [p.get("orig_dir"), p.get("ft_dir"), p.get("run_dir"),
               p.get("comparison"), p.get("report_pdf"), p.get("samples_dir")]
    if DATA_CACHE_ROOT:
        targets.append(os.path.join(DATA_CACHE_ROOT, ".arrow", key))
    if is_custom:
        targets.append(p.get("data_dir"))
    removed = []
    for t in targets:
        if not t:
            continue
        ap = os.path.realpath(_abs(p, t))
        if os.path.commonpath([ap, ROOT]) != ROOT:      # safety: never delete outside the working root
            continue
        if os.path.isdir(ap):
            shutil.rmtree(ap, ignore_errors=True); removed.append(t)
        elif os.path.isfile(ap):
            os.remove(ap); removed.append(t)
    for suff in ("_loss.png", "_acc.png"):               # cached UI charts
        f = os.path.join(ROOT, "build", "ui_charts", key + suff)
        if os.path.exists(f):
            os.remove(f)
    if is_custom:
        staged = _remove_unshared_staged_source(key, p)
        if staged:
            removed.append(staged)
        REG.pop(key, None)
        _save_custom()
    return {"ok": True, "kind": p.get("kind"), "removed": removed}


@app.post("/run/cancel")
def run_cancel(body: dict = None):
    STATE["cancel"].set()
    with STATE_LOCK:
        proc = STATE["proc"]
    if proc and proc.poll() is None:
        try:
            proc.terminate()
        except Exception:
            pass
    return {"ok": True}


@app.post("/run/stop")
def run_stop(body: dict = None):
    """Soft stop: end training now and continue the pipeline using the latest completed
    epoch's checkpoint (deploy → eval → report). Only meaningful during fine-tuning."""
    with STATE_LOCK:
        phase = STATE["phase"]
        proc = STATE["proc"]
    if phase != "finetune":
        raise HTTPException(status_code=409, detail="stop only applies while fine-tuning")
    STATE["stop"].set()
    if proc and proc.poll() is None:
        try:
            proc.terminate()
        except Exception:
            pass
    return {"ok": True}


def _parse_curve(run_dir):
    """Thin wrapper around curve_history.merged() — see that module for the stitching
    algorithm (committed history + current train.log offset onto the cumulative axis).
    All consumers (/run/status live curve, /results/curve, /chart loss, /results/summary)
    go through here, so the graph runs continuously across continues instead of
    restarting at epoch 1."""
    return curve_history.merged(_abs({}, run_dir)) if run_dir else \
        {"train_epoch": [], "train_loss": [], "eval_epoch": [], "eval_loss": []}


@app.get("/run/status")
def run_status():
    import re
    with STATE_LOCK:
        s = {k: STATE[k] for k in ("phase", "preset", "run_id", "stage", "stage_label",
                                   "done", "total", "status", "error", "started_at")}
        ring = list(STATE["log_ring"])
    s["log_tail"] = ring[-40:]
    s["elapsed"] = round(time.time() - s["started_at"], 1) if s["started_at"] else 0
    # overall progress across this run's pipeline stages (count varies by plan: full run, eval-only, continue)
    s["pct"] = round(100 * s["done"] / s["total"]) if s.get("total") else None
    run_dir = REG[s["preset"]]["run_dir"] if s["preset"] in REG else None
    # continue-training epoch offset (0 for a normal run): the warm-start segment counts epochs from
    # 0, so we add the prior cumulative epochs everywhere the user reads an epoch number — same source
    # the graph uses (curve_history). Display-only: the on-disk train.log is never modified.
    prev = curve_history.load(_abs({}, run_dir)).get("prev_epochs", 0.0) if run_dir else 0.0
    if prev:
        s["log_tail"] = [re.sub(r"('epoch':\s*)([0-9.]+)",
                                lambda m: f"{m.group(1)}{round(float(m.group(2)) + prev, 4)}", ln)
                         for ln in s["log_tail"]]
    if s["phase"] in ("finetune", "evaluating", "done") and run_dir:
        s["curve"] = _parse_curve(run_dir)
        s["curve"]["map_per_class_latest"] = curve_history.latest_per_class(_abs({}, run_dir))
    else:
        s["curve"] = {"train_epoch": [], "train_loss": [], "eval_epoch": [], "eval_loss": []}
    # within-stage progress shown in the footer: training reports epoch N/total (the longest
    # stage); other stages report "[progress] done/total <unit>" lines emitted by the scripts.
    prog = None
    if s["phase"] == "finetune" and run_dir:
        c = _parse_curve(run_dir)
        eps = (c.get("train_epoch") or []) + (c.get("eval_epoch") or [])
        cur = int(eps[-1]) if eps else 0          # already cumulative (merged curve)
        seg = None
        try:
            import yaml
            with open(os.path.join(ROOT, run_dir, "config.yaml")) as f:
                seg = (yaml.safe_load(f) or {}).get("num_train_epochs")
        except Exception:
            seg = None
        if seg:
            tot = prev + int(seg)                  # cumulative total: prior epochs + this segment
            cur = min(cur, int(round(tot)))        # never exceed 100% at the seam
            prog = {"unit": "epochs", "done": cur, "total": int(round(tot)),
                    "pct": round(100 * cur / tot) if tot else None}
    if prog is None:
        for line in reversed(ring):
            m = re.search(r"\[progress\]\s+(\d+)\s*/\s*(\d+)(?:\s+(\w+))?", line)
            if m:
                d, t = int(m.group(1)), int(m.group(2))
                prog = {"unit": m.group(3) or "items", "done": d, "total": t,
                        "pct": round(100 * d / t) if t else None}
                break
    s["stage_progress"] = prog
    # If nothing is UI-driven, surface a run that is training via an external CLI/container harness
    # (AutoML sweep or finetune) so the status pill reads "training", not "idle". Advisory only —
    # phase/status stay 'idle' so the auto-navigation logic (which is for UI-launched runs) is untouched.
    if s["phase"] == "idle":
        ext = _active_external_run()
        if ext:
            s["external"] = ext
    return s


@app.post("/recolor_report")
def recolor_report(body: dict):
    """Re-render the qualitative sample frames + PDF report with user-chosen box/text colors
    (no GPU — just reads existing predictions and redraws). Lets the report match the colors the
    user picked in the UI for their background (e.g. green PCB)."""
    import subprocess
    preset = body.get("preset")
    if preset not in REG:
        raise HTTPException(status_code=400, detail="unknown run")
    with STATE_LOCK:
        phase = STATE["phase"]
    if phase in ("baseline", "finetune", "evaluating"):
        raise HTTPException(status_code=409, detail=f"a run is in progress (phase '{phase}')")
    p = REG[preset]
    if not _try_load(p["comparison"]):
        raise HTTPException(status_code=409, detail="no completed run to re-render yet")
    SK = ".claude/skills/deepstream-eval-and-finetune"
    PY = os.path.join(ROOT, "build", ".venv_train", "bin", "python")
    od, fd, rm = resolve_orig_dir(ROOT, p), p["ft_dir"], p["report_meta"]
    box, text = body.get("box_color"), body.get("text_color")
    samp = [PY, f"{SK}/scripts/sample_overlays.py",
            "--orig-eval-set", os.path.join(od, "eval", "eval_set"),
            "--orig-predictions", os.path.join(od, "eval", "predictions_engine.json"),
            "--ft-eval-set", os.path.join(fd, "eval", "eval_set"),
            "--ft-predictions", os.path.join(fd, "eval", "predictions_engine.json"),
            "--out-dir", p["samples_dir"], "--n", "6"] + p.get("sample_overlays_extra", [])
    if box:   # apply to ALL boxes (GT + on-target + off-target) to match the UI
        samp += ["--det-color", box, "--gt-color", box, "--offtarget-color", box]
    if text:
        samp += ["--text-color", text]
    conf = body.get("conf")   # the UI slider value -> draw the report at the SAME threshold (exact)
    if conf is not None:
        samp += ["--conf", str(conf), "--no-auto-conf"]
    rep = [PY, f"{SK}/scripts/make_report.py", "--comparison", p["comparison"],
           "--samples-dir", p["samples_dir"], "--config", os.path.join(p["run_dir"], "config.yaml"),
           "--train-log", os.path.join(p["run_dir"], "train.log"),
           "--deployed-checkpoint", os.path.join(fd, "model", "deployed_checkpoint.json"),
           "--precision", "fp16", "--env", "build/env_info.json",
           "--model-id", rm["model_id"], "--model-link", rm["model_link"],
           "--model-license", rm["model_license"], "--model-desc", rm["model_desc"],
           "--dataset-id", rm["dataset_id"], "--dataset-link", rm["dataset_link"],
           "--dataset-desc", rm["dataset_desc"], "--output", p["report_pdf"]]
    try:
        subprocess.run(samp, cwd=ROOT, check=True, capture_output=True, text=True)
        subprocess.run(rep, cwd=ROOT, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=(e.stderr or str(e))[-400:])
    return {"ok": True}


@app.get("/probe")
def probe(dataset: str):
    """Best-effort class-count detection for a dataset entered in the form, so the UI can load
    the right hyperparameter defaults when the dataset changes. Works for known presets and
    local paths (reads a COCO json's categories, or labels.txt/classes.txt); HF ids / URLs that
    aren't local can't be counted without downloading -> returns null (UI keeps its default)."""
    ds = _norm_source(dataset)
    for p in REG.values():                         # a built-in preset?
        if p.get("dataset_id") == ds and p.get("ft_nclasses"):
            return {"n_classes": p["ft_nclasses"], "source": "preset"}
    path = ds if os.path.isabs(ds) else os.path.join(ROOT, ds)
    n = None
    if os.path.isdir(path):
        for name in ("train.json", "val.json", "valid.json", "test.json",
                     "instances_train.json", "instances_val.json", "annotations.json"):
            try:
                d = json.load(open(os.path.join(path, name)))
                if isinstance(d, dict) and d.get("categories"):
                    n = len(d["categories"]); break
            except Exception:
                continue
        if n is None:
            for lf in ("labels.txt", "classes.txt"):
                try:
                    n = sum(1 for ln in open(os.path.join(path, lf)) if ln.strip()); break
                except Exception:
                    continue
    return {"n_classes": n, "source": ("local" if n else "unknown")}


@app.get("/results/automl")
def results_automl(preset: str):
    """AutoML/HPO sweep state for the Fine-tune tab's 'AutoML sweep' sub-view: the search space
    (per-trial hyperparameters), each trial's best per-epoch eval_map (the leaderboard), the
    winning trial, and the final-training config. Read straight from the NGC harness's durable
    logs under logs/<run>/<ts>/ — live-refreshable while the sweep runs."""
    if preset not in REG:
        raise HTTPException(status_code=400, detail="unknown run")
    import glob
    import re

    import yaml
    d = _automl_dir(REG[preset])
    if not d or not os.path.exists(os.path.join(d, "automl.log")):
        return {"is_automl": False}
    aml = ""
    try:
        with open(os.path.join(d, "automl.log")) as f:
            aml = f.read()
    except OSError:
        pass
    m = re.search(r"trials=(\d+)\s+trial_epochs=(\d+)\s+final_epochs=(\d+)", aml)
    n_trials = int(m.group(1)) if m else None
    trial_epochs = int(m.group(2)) if m else None
    final_epochs = int(m.group(3)) if m else None
    # leaderboard: "<i> <best_eval_map>" per scored trial
    scores = {}
    lb = os.path.join(d, "leaderboard.txt")
    if os.path.exists(lb):
        for ln in open(lb):
            parts = ln.split()
            if len(parts) >= 2:
                try:
                    scores[int(parts[0])] = float(parts[1])
                except ValueError:
                    pass
    # trial configs (the search space), + status derived from the leaderboard / trial log
    trials = []
    for f in sorted(glob.glob(os.path.join(d, "trials", "trial_*.yaml")),
                    key=lambda x: int(re.search(r"trial_(\d+)", x).group(1))):
        i = int(re.search(r"trial_(\d+)\.yaml", f).group(1))
        c = yaml.safe_load(open(f)) or {}
        has_log = os.path.exists(os.path.join(d, "trials", f"trial_{i}.log"))
        trials.append({
            "i": i, "lr": c.get("learning_rate"), "warmup": c.get("warmup_ratio"),
            "sched": c.get("lr_scheduler_type"), "wd": c.get("weight_decay"),
            "epochs": c.get("num_train_epochs"), "best_map": scores.get(i),
            "status": "scored" if i in scores else ("running" if has_log else "pending"),
            "curve": _parse_trial_curve(os.path.join(d, "trials", f"trial_{i}.log")),
        })
    wm = re.search(r"WINNER: trial (-?\d+)\s+eval_map=([0-9.eE+-]+)", aml)
    winner = int(wm.group(1)) if wm else (max(scores, key=scores.get) if scores else None)
    winner_map = float(wm.group(2)) if wm else (scores.get(winner) if winner is not None else None)
    final = None
    fc = os.path.join(d, "final_config.yaml")
    if os.path.exists(fc):
        c = yaml.safe_load(open(fc)) or {}
        final = {"lr": c.get("learning_rate"), "warmup": c.get("warmup_ratio"),
                 "sched": c.get("lr_scheduler_type"), "wd": c.get("weight_decay"),
                 "epochs": c.get("num_train_epochs")}
    if "[automl] DONE" in aml:
        phase = "done"
    elif os.path.exists(os.path.join(d, "final.log")):
        phase = "final"
    else:
        phase = "trials"
    n_scored = len([t for t in trials if t["status"] == "scored"])
    return {"is_automl": True, "phase": phase, "n_trials": n_trials, "trial_epochs": trial_epochs,
            "final_epochs": final_epochs, "n_scored": n_scored, "trials": trials,
            "winner": winner, "winner_map": winner_map, "final_config": final}


@app.get("/results/summary")
def results_summary(preset: str):
    """Plain-language fine-tune summary for non-ML reviewers: what was trained on, for how
    long, at what settings, and the before/after detection accuracy."""
    if preset not in REG:
        raise HTTPException(status_code=400, detail="unknown run")
    p = REG[preset]
    cfg = {}
    try:
        import yaml
        with open(_abs(p, os.path.join(p["run_dir"], "config.yaml"))) as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        pass
    cmp = _try_load(p["comparison"]) or {}
    acc = (cmp.get("accuracy") or {}).get("map_50") or {}
    labels = cfg.get("label_names") or []
    pcaf = cmp.get("per_class_ap_finetuned") or {}
    best = max(pcaf.items(), key=lambda kv: (kv[1] if kv[1] is not None else -1), default=(None, None))
    # Deployed checkpoint + where the model is stored (written by pipeline._record_deployed_checkpoint).
    dep = _try_load(os.path.join(p["ft_dir"], "model", "deployed_checkpoint.json")) or {}
    ckpt_info = _checkpoint_info(p) or {}
    # eval-loss trajectory: lowest-eval_loss epoch (shown for context — NOT what we deploy, since
    # for DETR detectors eval_loss is a poor mAP proxy; we deploy the most-trained checkpoint).
    curve = _parse_curve(p["run_dir"])
    min_ev_epoch = None
    if curve.get("eval_loss"):
        i = min(range(len(curve["eval_loss"])), key=lambda k: curve["eval_loss"][k])
        min_ev_epoch = int(round(curve["eval_epoch"][i]))
    # CUMULATIVE epochs trained across all continue segments (merged curve is on the
    # absolute axis); a never-continued run == config num_train_epochs.
    cum_epochs = cfg.get("num_train_epochs")
    if curve.get("train_epoch"):
        cum_epochs = int(round(max(curve["train_epoch"]))) or cum_epochs
    map50_train = None
    if curve.get("eval_map_50"):
        map50_train = curve["eval_map_50"][-1]
    has_training = bool(curve.get("train_loss") or curve.get("eval_map"))
    return {
        "model_id": p["model_id"], "dataset_id": p["dataset_id"],
        "n_classes": len(labels) or p.get("ft_nclasses"), "classes": labels,
        "n_train": cfg.get("n_train"),
        "epochs": cum_epochs,
        "lr": cfg.get("learning_rate"), "sched": cfg.get("lr_scheduler_type"),
        "batch": cfg.get("per_device_train_batch_size"),
        "grad_accum": cfg.get("gradient_accumulation_steps", 1),
        "warmup": cfg.get("warmup_ratio"), "weight_decay": cfg.get("weight_decay"),
        "precision": ("bf16" if cfg.get("bf16") else "fp16" if cfg.get("fp16") else "fp32"),
        "eval_strategy": cfg.get("eval_strategy"), "metric_for_best_model": cfg.get("metric_for_best_model"),
        "eval_images": cmp.get("n_eval"),
        "map50_before": acc.get("original"), "map50_after": acc.get("finetuned"),
        "delta": acc.get("delta"), "x": acc.get("x"),
        "best_class": best[0], "best_ap": best[1],
        "has_results": bool(acc),
        "has_training": has_training,
        "map50_training": map50_train,
        # checkpoint selection (#2) + stored-model location (#3)
        "deployed_epoch": dep.get("deployed_epoch"),
        "eval_checkpoint": ckpt_info,
        "total_epochs": cum_epochs,
        "min_evalloss_epoch": min_ev_epoch,
        "checkpoint_selection": dep.get("selection"),
        "model_name": dep.get("model_name"),
        "model_onnx": dep.get("onnx"),
        "model_engine": dep.get("engine"),
        "model_checkpoint": dep.get("checkpoint_epoch_tag") or dep.get("checkpoint"),
        "nvinfer_config": dep.get("nvinfer_config"),
        # dataset class distribution (train + eval per-class counts) — same source as the report §2a
        "dataset_distribution": cmp.get("dataset_distribution"),
    }


@app.get("/results/curve")
def results_curve(preset: str):
    """Training loss + per-epoch mAP curve for any run (live or finished), parsed from its train.log
    (stitched across continue segments). Also includes the latest epoch's per-class AP for the live
    per-class table."""
    if preset not in REG:
        raise HTTPException(status_code=400, detail="unknown run")
    c = _parse_curve(REG[preset]["run_dir"])
    c["map_per_class_latest"] = curve_history.latest_per_class(_abs({}, REG[preset]["run_dir"]))
    return c


@app.get("/results/dataset_qc")
def results_dataset_qc(preset: str):
    """Dataset health/QC over the ingested COCO (annotations-only, fast): class balance, box geometry,
    unlabeled images, train<->val leakage, plus plain-language flags. Lets teams vet a dataset before
    a training run."""
    if preset not in REG:
        raise HTTPException(status_code=400, detail="unknown run")
    dd = _abs(REG[preset], REG[preset].get("data_dir", ""))
    rep = dataset_qc.qc_for_data_dir(dd) if dd else {}
    if not rep:
        raise HTTPException(status_code=404, detail="dataset not ingested yet")
    return rep


@app.get("/results/model_context")
def results_model_context(preset: str):
    """Which deployed model legs exist for Analyze / Playground, with checkpoint epoch + mAP."""
    if preset not in REG:
        raise HTTPException(status_code=400, detail="unknown run")
    p = REG[preset]
    ck = _checkpoint_info(p)
    return {
        "preset_key": preset,
        "run_label": p.get("label"),
        "model_id": p.get("model_id"),
        "dataset_id": p.get("dataset_id"),
        "run_dir": p.get("run_dir"),
        "selected_checkpoint": ck,
        "orig": _model_eval_meta(p, "orig"),
        "ft": _model_eval_meta(p, "ft"),
    }


@app.get("/results/confusion")
def results_confusion(preset: str, which: str = "ft", conf: float = 0.2, iou: float = 0.5):
    """Confusion matrix + per-class precision/recall + hardest-images ranking for a run's DEPLOYED
    predictions vs ground truth (which=ft|orig). Class-agnostic IoU matching, so a GT box's predicted
    class is visible (e.g. spur->short). Computed on the fly from predictions_engine.json + GT."""
    if preset not in REG:
        raise HTTPException(status_code=400, detail="unknown run")
    if which not in ("orig", "ft"):
        raise HTTPException(status_code=400, detail="which must be orig or ft")
    p = REG[preset]
    base = resolve_orig_dir(ROOT, p) if which == "orig" else p["ft_dir"]
    pred = _abs(p, os.path.join(base, "eval", "predictions_engine.json"))
    gt = _abs(p, os.path.join(base, "eval", "eval_set", "ground_truth.json"))
    if not (os.path.exists(pred) and os.path.exists(gt)):
        raise HTTPException(status_code=404, detail=f"no '{which}' eval results yet")
    out = error_analysis.analyze_paths(pred, gt, conf=max(0.0, min(1.0, conf)),
                                       iou_thr=max(0.05, min(0.95, iou)))
    out["model"] = _model_eval_meta(p, which)
    return out


@app.post("/infer")
def infer(preset: str, file: UploadFile = File(...), which: str = "ft", conf: float = 0.01):
    """Inference Playground: run ONE uploaded image through the DEPLOYED DeepStream engine (so it shows
    exactly what's deployed) and return the preprocessed image (base64) + detections. Stateless —
    preprocess to the run's eval canvas, run ds_image_eval on a 1-image temp dir, parse, clean up."""
    import base64, glob, io, shutil, tempfile
    from PIL import Image
    if preset not in REG:
        raise HTTPException(status_code=400, detail="unknown run")
    with STATE_LOCK:
        phase = STATE["phase"]
    if phase in ("baseline", "finetune", "evaluating"):
        raise HTTPException(status_code=409, detail=f"a run is in progress (phase '{phase}') — try again when it finishes")
    p = REG[preset]
    base = resolve_orig_dir(ROOT, p) if which == "orig" else p["ft_dir"]
    cfg = _abs(p, os.path.join(base, "config", "config_infer.txt"))
    labels_f = _abs(p, os.path.join(base, "config", "labels.txt"))
    if not (os.path.exists(cfg) and os.path.exists(labels_f)):
        raise HTTPException(status_code=404, detail=f"the '{which}' model isn't deployed yet — run it first")
    # The deployed config's model-engine-file name doesn't match nvinfer's built engine
    # (`<onnx>_b1_gpu0_fp16.engine`), so a plain run would REBUILD from ONNX (~2 min) every call. Point
    # at the already-built engine (absolute) so nvinfer just deserializes it (~seconds).
    engines = sorted(glob.glob(_abs(p, os.path.join(base, "model", "*.engine"))))
    if not engines:
        raise HTTPException(status_code=404, detail=f"the '{which}' model isn't deployed yet — run it first")
    real_engine = engines[0]
    app_bin = os.path.join(ROOT, SK, "scripts", "ds_image_eval")
    if not os.path.exists(app_bin):
        raise HTTPException(status_code=503, detail="inference app not built")
    raw = file.file.read(20 * 1024 * 1024 + 1)
    if len(raw) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="image too large (max 20 MB)")
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="not a decodable image")
    # canvas + resize_mode = the run's own eval preprocessing (authoritative); fall back to 640 stretch
    gt = _try_load(os.path.join(base, "eval", "eval_set", "ground_truth.json")) or {}
    W, H = (gt.get("canvas") or [640, 640])[:2]
    mode = gt.get("resize_mode", "stretch")
    ow, oh = img.size
    if mode == "stretch":
        canvas = img.resize((W, H))
    else:                                   # letterbox: aspect-fit + grey pad (matches build_eval_set)
        s = min(W / ow, H / oh); nw, nh = max(1, round(ow * s)), max(1, round(oh * s))
        canvas = Image.new("RGB", (W, H), (114, 114, 114))
        canvas.paste(img.resize((nw, nh)), ((W - nw) // 2, (H - nh) // 2))
    model_labels = [ln.strip() for ln in open(labels_f) if ln.strip()]
    tmp = tempfile.mkdtemp(prefix="pg_")
    try:
        imdir = os.path.join(tmp, "images"); os.makedirs(imdir)
        canvas.save(os.path.join(imdir, "000000.jpg"), quality=92)
        preds_txt = os.path.join(tmp, "preds.txt")
        # temp config: absolute paths (it lives in /tmp so the original ../relative refs won't resolve),
        # and model-engine-file -> the existing built engine (avoids the rebuild).
        cfg_dir = os.path.dirname(cfg)
        tmp_cfg = os.path.join(tmp, "config_infer.txt")
        with open(tmp_cfg, "w") as out:
            for ln in open(cfg):
                s = ln.rstrip("\n")
                if s.startswith("model-engine-file="):
                    out.write(f"model-engine-file={real_engine}\n")
                elif s.startswith(("onnx-file=", "labelfile-path=", "custom-lib-path=")):
                    k, v = s.split("=", 1)
                    out.write(f"{k}={os.path.normpath(os.path.join(cfg_dir, v))}\n")
                else:
                    out.write(s + "\n")
        cmd = [app_bin, tmp_cfg, os.path.abspath(imdir), "1", preds_txt, str(W), str(H)]
        r = subprocess.run(cmd, env=BASE_ENV, cwd=ROOT, capture_output=True, text=True, timeout=120)
        if r.returncode != 0 or not os.path.exists(preds_txt):
            raise HTTPException(status_code=500, detail="inference failed (engine/parser error)")
        dets = []
        for d in eval_engine._parse_preds(preds_txt).get(0, []):
            if d["score"] < conf:
                continue
            lbl = d["label"]
            name = model_labels[lbl] if 0 <= lbl < len(model_labels) else str(lbl)
            dets.append({"name": name, "xyxy": [round(x, 1) for x in d["bbox"]], "score": round(d["score"], 4)})
        dets.sort(key=lambda x: -x["score"])
        buf = io.BytesIO(); canvas.save(buf, format="JPEG", quality=88)
        return {"w": W, "h": H, "which": which,
                "image": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(),
                "detections": dets}
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="inference timed out")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@app.get("/chart")
def chart(preset: str, kind: str = "loss"):
    """Report-style matplotlib charts for Step 3. kind='loss' -> training + eval-loss panels
    (from train.log); kind='acc' -> deployed before/after bars + per-class AP (from comparison).
    Uses the Figure/Agg API (no pyplot globals) so it's safe under the threadpool."""
    if preset not in REG:
        raise HTTPException(status_code=400, detail="unknown run")
    p = REG[preset]
    from matplotlib.figure import Figure
    G, B, K = "#76b900", "#1f77b4", "#555555"
    cdir = os.path.join(ROOT, "build", "ui_charts")
    os.makedirs(cdir, exist_ok=True)
    fp = os.path.join(cdir, f"{preset}_{kind}.png")

    if kind == "loss":
        c = _parse_curve(p["run_dir"])
        if not c["train_loss"] and not c["eval_loss"]:
            raise HTTPException(status_code=404, detail="no training data yet")
        fig = Figure(figsize=(9, 3.0)); ax = fig.subplots(1, 2)
        if c["train_loss"]:
            ax[0].plot(c["train_epoch"], c["train_loss"], color=B, lw=.9)
        ax[0].set_title("Training loss"); ax[0].set_xlabel("epoch"); ax[0].grid(alpha=.3)
        if c["eval_loss"]:
            ax[1].plot(c["eval_epoch"], c["eval_loss"], "o-", color=G, ms=3)
        ax[1].set_title("Eval loss / epoch"); ax[1].set_xlabel("epoch"); ax[1].grid(alpha=.3)
        fig.tight_layout(); fig.savefig(fp, dpi=130)
    elif kind == "acc":
        cmp = _try_load(p["comparison"])
        if not cmp:
            raise HTTPException(status_code=404, detail="no comparison yet")
        acc = cmp.get("accuracy", {})
        ks = [k for k in ["map", "map_50", "map_75"] if k in acc]
        x = list(range(len(ks))); w = .38
        fig = Figure(figsize=(9, 3.3)); ax = fig.subplots(1, 2)
        ax[0].bar([i - w / 2 for i in x], [acc[k].get("original") or 0 for k in ks], w, label="original (deployed)", color=K)
        ax[0].bar([i + w / 2 for i in x], [acc[k].get("finetuned") or 0 for k in ks], w, label="fine-tuned (deployed)", color=G)
        ax[0].set_xticks(x); ax[0].set_xticklabels(ks); ax[0].legend()
        ax[0].set_title("Deployed accuracy: original vs fine-tuned"); ax[0].grid(alpha=.3, axis="y")
        pc = {k: v for k, v in (cmp.get("per_class_ap_finetuned") or {}).items() if v is not None and v >= 0}
        if pc:
            ax[1].barh(list(pc), list(pc.values()), color=B); ax[1].invert_yaxis()
            ax[1].set_title("Per-class AP (fine-tuned, deployed)"); ax[1].set_xlabel("AP"); ax[1].grid(alpha=.3, axis="x")
            if len(pc) > 25:
                ax[1].tick_params(axis="y", labelsize=5)
        fig.tight_layout(); fig.savefig(fp, dpi=130)
    elif kind == "map":
        # Per-epoch mAP (PyTorch proxy on the training eval subset — trend, NOT the deployed number).
        c = _parse_curve(p["run_dir"])
        if not c.get("eval_map"):
            raise HTTPException(status_code=404, detail="no per-epoch mAP yet")
        pcl = curve_history.latest_per_class(_abs({}, p["run_dir"]))
        fig = Figure(figsize=(9, 3.0)); ax = fig.subplots(1, 2)
        ax[0].plot(c["map_epoch"], c["eval_map_50"], "s-", color=G, ms=3, label="mAP@50")
        ax[0].plot(c["map_epoch"], c["eval_map"], "o-", color=B, ms=3, label="mAP@[.5:.95]")
        ax[0].set_title("Eval mAP / epoch (PyTorch proxy)"); ax[0].set_xlabel("epoch")
        ax[0].set_ylim(0, 1); ax[0].grid(alpha=.3); ax[0].legend(fontsize=8)
        pc = {k: v for k, v in pcl.items() if v is not None and v >= 0}
        if pc:
            ax[1].barh(list(pc), list(pc.values()), color=B); ax[1].invert_yaxis()
            ax[1].set_title("Per-class AP (latest epoch)"); ax[1].set_xlabel("AP")
            ax[1].set_xlim(0, 1); ax[1].grid(alpha=.3, axis="x")
            if len(pc) > 25:
                ax[1].tick_params(axis="y", labelsize=5)
        else:
            ax[1].axis("off")
        fig.tight_layout(); fig.savefig(fp, dpi=130)
    else:
        raise HTTPException(status_code=400, detail="bad chart kind")
    return FileResponse(fp, media_type="image/png", headers={"Cache-Control": "no-store"})


@app.get("/results/baseline")
def results_baseline(preset: str):
    if preset not in REG:
        raise HTTPException(status_code=400, detail="unknown preset")
    p = REG[preset]
    od = resolve_orig_dir(ROOT, p)
    m = _try_load(os.path.join(od, "eval", "engine_metrics.json"))
    if not m:
        raise HTTPException(status_code=404, detail="baseline not available")
    perf = _try_load(os.path.join(od, "eval", "perf.json")) or {}
    pcap = m.get("per_class_ap", {})
    pcd = m.get("per_class_detections", {})
    per_class = [{"name": n, "ap": ap, "detections": pcd.get(n, 0)}
                 for n, ap in pcap.items() if ap is not None and ap != -1.0]
    per_class.sort(key=lambda r: r["ap"], reverse=True)
    fps = perf.get("trtexec_qps")
    fps_source = "trtexec"
    if fps is None:
        fps = perf.get("pipeline_fps")
        fps_source = "pipeline (end-to-end; engine QPS in final report)"
    # Show the SAME frames as the Compare tab (the report's samples.json) so Baseline ↔ Compare
    # line up. Fall back to busiest-by-GT only before the report/samples exist (mid-run).
    gt = _try_load(os.path.join(od, "eval", "eval_set", "ground_truth.json")) or {}
    obj_by_id = {im["image_id"]: len(im.get("objects", [])) for im in gt.get("images", [])}
    samp = _try_load(os.path.join(p["samples_dir"], "samples.json")) or {}
    sample_ids = [s["id"] for s in samp.get("frames", []) if s["id"] in obj_by_id]
    if not sample_ids:
        sample_ids = [im["image_id"] for im in
                      sorted(gt.get("images", []), key=lambda im: len(im.get("objects", [])), reverse=True)[:6]]
    images = [{"image_id": i, "objects": obj_by_id.get(i, 0)} for i in sample_ids]
    return {
        "map": m.get("map"), "map_50": m.get("map_50"), "map_75": m.get("map_75"),
        "detection_rate": m.get("detection_rate"), "n_eval": m.get("n_eval"),
        "fps": fps, "fps_source": fps_source,
        "per_class": per_class, "images": images,
    }


@app.get("/eval_image")
def eval_image(preset: str, leg: str = "orig", image_id: int = 0):
    if preset not in REG:
        raise HTTPException(status_code=400, detail="unknown preset")
    p = REG[preset]
    model_dir = resolve_orig_dir(ROOT, p) if leg == "orig" else p["ft_dir"]
    gt = _try_load(os.path.join(model_dir, "eval", "eval_set", "ground_truth.json")) or {}
    rec = next((im for im in gt.get("images", []) if im["image_id"] == image_id), None)
    if not rec:
        raise HTTPException(status_code=404, detail="image not found")
    fp = _abs(p, os.path.join(model_dir, "eval", "eval_set", rec["file"]))
    if not os.path.exists(fp):
        raise HTTPException(status_code=404, detail="image file missing")
    return FileResponse(fp)


@app.get("/eval_boxes")
def eval_boxes(preset: str, leg: str = "orig", image_id: int = 0, conf: float = CONF_THRESH):
    if preset not in REG:
        raise HTTPException(status_code=400, detail="unknown preset")
    p = REG[preset]
    model_dir = resolve_orig_dir(ROOT, p) if leg == "orig" else p["ft_dir"]
    gt = _try_load(os.path.join(model_dir, "eval", "eval_set", "ground_truth.json")) or {}
    preds = _try_load(os.path.join(model_dir, "eval", "predictions_engine.json")) or {}
    lm = gt.get("label_map", {})
    targets = _target_names(gt)
    rec = next((im for im in gt.get("images", []) if im["image_id"] == image_id), None)
    if not rec:
        raise HTTPException(status_code=404, detail="image not found")
    gt_boxes = [{"name": lm.get(str(o["category"]), str(o["category"])),
                 "xyxy": _xywh_to_xyxy(o["bbox"])} for o in rec.get("objects", [])]
    pr = next((x for x in preds.get("predictions", []) if x["image_id"] == image_id), None)
    pred_boxes = []
    if pr:
        for d in pr.get("detections", []):
            if d.get("score", 0) < conf:
                continue
            name = lm.get(str(d["label"]), str(d["label"]))
            pred_boxes.append({"name": name, "xyxy": d["bbox"], "score": round(d.get("score", 0), 3),
                               "on_target": name in targets})
    return {"w": rec.get("width"), "h": rec.get("height"),
            "gt": gt_boxes, "pred": pred_boxes, "targets": sorted(targets)}


@app.get("/results/final")
def results_final(preset: str):
    if preset not in REG:
        raise HTTPException(status_code=400, detail="unknown preset")
    p = REG[preset]
    cmp = _try_load(p["comparison"])
    if not cmp:
        raise HTTPException(status_code=404, detail="comparison not available")
    acc = cmp.get("accuracy", {})
    samples = _try_load(os.path.join(p["samples_dir"], "samples.json")) or {"frames": []}
    pcaf = cmp.get("per_class_ap_finetuned", {})
    pcao = cmp.get("per_class_ap_original", {})
    # ground-truth object count per class (from the fine-tuned eval set's GT, in target label space)
    gt = _try_load(os.path.join(p["ft_dir"], "eval", "eval_set", "ground_truth.json")) or {}
    lm = gt.get("label_map", {})
    gt_counts = {}
    for im in gt.get("images", []):
        for o in im.get("objects", []):
            nm = lm.get(str(o["category"]), str(o["category"]))
            gt_counts[nm] = gt_counts.get(nm, 0) + 1
    # confident (>= display threshold) detection totals per class — consistent with the
    # per-frame counts, unlike the raw pre-threshold counts in the comparison JSON
    base_tot = _pred_class_totals(resolve_orig_dir(ROOT, p))
    ft_tot = _pred_class_totals(p["ft_dir"])
    per_class = []
    for n, ap in pcaf.items():
        o = pcao.get(n, -1.0)
        per_class.append({"name": n, "original": (None if o == -1.0 else o), "finetuned": ap,
                          "det_gt": gt_counts.get(n, 0),
                          "det_original": base_tot.get(n, 0), "det_finetuned": ft_tot.get(n, 0)})
    # Frames to display in the UI. PRIMARY: the same 5-10 frames the report already selected and
    # rendered (samples.json) — these were chosen with an ADAPTIVE threshold, so they show boxes
    # even for low-score detectors. FALLBACK (if samples.json is empty): rank predictions live by
    # #detections, using the adaptive display conf from samples.json (NOT a hardcoded 0.2, which
    # sits above RT-DETR's ~0.1-0.2 score range and would surface only one frame).
    ftpreds = _try_load(os.path.join(p["ft_dir"], "eval", "predictions_engine.json")) or {}
    disp_conf = samples.get("conf", CONF_THRESH)
    ranked = sorted(ftpreds.get("predictions", []),
                    key=lambda x: -sum(1 for d in x.get("detections", []) if d.get("score", 0) >= disp_conf))
    best = [x["image_id"] for x in ranked
            if any(d.get("score", 0) >= disp_conf for d in x.get("detections", []))][:6]
    frame_ids = [s["id"] for s in samples.get("frames", [])] or best
    dep = _try_load(os.path.join(p["ft_dir"], "model", "deployed_checkpoint.json")) or {}
    ck = cmp.get("checkpoint_evaluated") or {}
    deployed_ep = cmp.get("deployed_checkpoint_epoch") if cmp.get("deployed_checkpoint_epoch") is not None else dep.get("deployed_epoch")
    deploy_stale = (
        ck.get("epoch") is not None and deployed_ep is not None
        and int(ck["epoch"]) != int(deployed_ep)
    )
    return {
        "accuracy": {k: acc.get(k) for k in ("map", "map_50", "map_75") if k in acc},
        "checkpoint_evaluated": ck,
        "deployed_checkpoint_epoch": deployed_ep,
        "deploy_stale": deploy_stale,
        "perf": cmp.get("perf", {}), "n_eval": cmp.get("n_eval"),
        "detection_rate": cmp.get("detection_rate"),
        "per_class": per_class,
        "samples": samples.get("frames", []),
        "frames": frame_ids,
        "has_report": os.path.exists(_abs(p, p["report_pdf"])),
    }


def _pred_class_totals(model_dir):
    """Total confident (>= display threshold) detections per class across the whole eval set."""
    gt = _try_load(os.path.join(model_dir, "eval", "eval_set", "ground_truth.json")) or {}
    preds = _try_load(os.path.join(model_dir, "eval", "predictions_engine.json")) or {}
    lm = gt.get("label_map", {})
    tot = {}
    for pr in preds.get("predictions", []):
        for d in pr.get("detections", []):
            if d.get("score", 0) < CONF_THRESH:
                continue
            nm = lm.get(str(d["label"]), str(d["label"]))
            tot[nm] = tot.get(nm, 0) + 1
    return tot


def _img_class_counts(model_dir, image_id, source, conf=CONF_THRESH):
    """Per-class counts for one frame. source='gt' counts ground-truth objects;
    source='pred' counts engine detections at/above the given confidence threshold."""
    gt = _try_load(os.path.join(model_dir, "eval", "eval_set", "ground_truth.json")) or {}
    lm = gt.get("label_map", {})
    out = {}
    if source == "gt":
        rec = next((im for im in gt.get("images", []) if im["image_id"] == image_id), None)
        for o in (rec or {}).get("objects", []):
            nm = lm.get(str(o["category"]), str(o["category"]))
            out[nm] = out.get(nm, 0) + 1
    else:
        preds = _try_load(os.path.join(model_dir, "eval", "predictions_engine.json")) or {}
        pr = next((x for x in preds.get("predictions", []) if x["image_id"] == image_id), None)
        for d in (pr or {}).get("detections", []):
            if d.get("score", 0) < conf:
                continue
            nm = lm.get(str(d["label"]), str(d["label"]))
            out[nm] = out.get(nm, 0) + 1
    return out


@app.get("/frame_counts")
def frame_counts(preset: str, image_id: int, conf: float = CONF_THRESH):
    """Per-class object counts for one comparison frame: ground truth vs baseline-deployed
    vs fine-tuned-deployed (detections counted at the given confidence threshold)."""
    if preset not in REG:
        raise HTTPException(status_code=400, detail="unknown run")
    p = REG[preset]
    od = resolve_orig_dir(ROOT, p)
    gt = _img_class_counts(p["ft_dir"], image_id, "gt") or _img_class_counts(od, image_id, "gt")
    baseline = _img_class_counts(od, image_id, "pred", conf)
    finetuned = _img_class_counts(p["ft_dir"], image_id, "pred", conf)
    classes = sorted(set(gt) | set(baseline) | set(finetuned), key=lambda c: (-gt.get(c, 0), c))
    return {"image_id": image_id, "classes": classes,
            "gt": gt, "baseline": baseline, "finetuned": finetuned}


@app.get("/sample")
def sample(preset: str, file: str):
    if preset not in REG:
        raise HTTPException(status_code=400, detail="unknown preset")
    p = REG[preset]
    safe = os.path.basename(file)                      # path-traversal guard
    fp = _abs(p, os.path.join(p["samples_dir"], safe))
    if not os.path.exists(fp):
        raise HTTPException(status_code=404, detail="sample not found")
    return FileResponse(fp)


@app.get("/report")
def report(preset: str):
    if preset not in REG:
        raise HTTPException(status_code=400, detail="unknown preset")
    p = REG[preset]
    fp = _abs(p, p["report_pdf"])
    if not os.path.exists(fp):
        raise HTTPException(status_code=404, detail="report not generated yet")
    return FileResponse(fp, media_type="application/pdf", filename=p["report_basename"])


@app.get("/system")
def system():
    gpu = None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=3)
        if out.returncode == 0 and out.stdout.strip():
            name, used, total = [x.strip() for x in out.stdout.strip().splitlines()[0].split(",")]
            gpu = {"name": name, "mem_used_gb": round(int(used) / 1024, 1),
                   "mem_total_gb": round(int(total) / 1024, 1)}
    except Exception:
        pass
    return {"gpu": gpu, "image": pipeline.DS_IMAGE, "precision": "fp16",
            "in_container": os.path.exists("/.dockerenv"), "root": ROOT,
            # full provenance for the UI "System & environment" panel — same data as report §7
            "environment": _try_load(os.path.join("build", "env_info.json"))}
