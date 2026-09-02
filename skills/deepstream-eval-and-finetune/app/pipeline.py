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
Parametrized, stage-by-stage driver for the deepstream-eval-and-finetune Web UI.

This is a faithful translation of the two validated CLI drivers
  - examples/rtdetr-aerial-sheep/run.sh   (preset "aerial-sheep")
  - examples/rtdetr-pcb/run.sh             (preset "pcb")
into two callables the server can run with a pause between them:

  baseline_plan(preset)  -> stages: build C app -> ingest -> export+deploy original -> eval set -> baseline DeepStream eval   [GATE A]
  finetune_plan(preset)  -> stages: fine-tune -> export+deploy fine-tuned -> eval set -> DeepStream eval -> compare+samples+report   [GATE B]

It NEVER imports or edits the skill's scripts — it invokes the exact same
`python scripts/...` commands run.sh issues, parametrized from presets.json.
Every stage is idempotent on its own output (skip-if-exists), matching run.sh.

A "ctx" object (supplied by server.py) provides the execution primitives:
  ctx.ROOT            working root (abs, normally /work)
  ctx.PY              abs path to build/.venv_train/bin/python
  ctx.SK              skill dir, ROOT-relative
  ctx.sh(cmd, cwd=None, log_file=None, check=True)   run a subprocess, stream output, honor cancel
  ctx.log(msg)        emit a line to the run log
  ctx.exists(rel)     ROOT-relative existence test
  ctx.path(rel)       ROOT-relative -> abs path
  ctx.cancelled()     True if the user requested cancel
"""
import json
import os
import shutil
from collections import namedtuple

from artifacts import resolve_orig_dir

# Stage: a unit of work shown as one progress step. fn(ctx, preset) does the work
# (including its own skip-if-exists guard). key/label drive the progress display.
Stage = namedtuple("Stage", ["key", "label", "fn"])

DS_IMAGE = "nvcr.io/nvidia/deepstream:9.1-triton-multiarch"


# ----------------------------------------------------------------------------- helpers
def _manifest_images(ctx, preset, split):
    """Resolve a split's image dir from <data_dir>/ingest_manifest.json (written by the ingest stage)."""
    man = ctx.path(os.path.join(preset["data_dir"], "ingest_manifest.json"))
    with open(man) as f:
        m = json.load(f)
    return m["splits"][split]["images_dir"]


def _train_data_dir(ctx, preset):
    """HF Arrow location: Linux cache on Windows/WSL, legacy run-local path elsewhere."""
    cache_root = os.environ.get("DS_DATA_CACHE_ROOT")
    if cache_root:
        key = preset.get("key") or os.path.basename(preset["run_dir"])
        return os.path.join(cache_root, ".arrow", key)
    return ctx.path(os.path.join(preset["run_dir"], "data"))


def _build_parser(ctx, preset, model_dir):
    """Replicates run.sh build_parser(): cp parser sources, then `make`. Idempotent on the .so."""
    so = os.path.join(model_dir, "parser", "libnvdsinfer_custom_impl.so")
    if ctx.exists(so):
        ctx.log(f"[skip] parser exists: {so}")
        return
    pdir = ctx.path(os.path.join(model_dir, "parser"))
    os.makedirs(pdir, exist_ok=True)
    assets = preset["assets_dir"]
    for fn in ("rtdetr_parser.cpp", "Makefile"):
        shutil.copy(ctx.path(os.path.join(assets, fn)), os.path.join(pdir, fn))
    ctx.sh(["make", "-C", model_dir + "/parser"])


def _count_labels(ctx, model_dir):
    """Number of classes = non-empty lines in the model's exported config/labels.txt."""
    lf = ctx.path(os.path.join(model_dir, "config", "labels.txt"))
    try:
        with open(lf) as f:
            return sum(1 for ln in f if ln.strip())
    except Exception:
        return None


def _write_cfg(ctx, preset, model_dir, onnx, engine, nclasses):
    """Replicates run.sh write_cfg(): sed the nvinfer config template. Idempotent on config_infer.txt.
    The class count is derived from the exported labels.txt (so custom datasets need no hard-coded
    nclasses); the passed value is only a fallback."""
    cfg = os.path.join(model_dir, "config", "config_infer.txt")
    if ctx.exists(cfg):
        ctx.log(f"[skip] config exists: {cfg}")
        return
    derived = _count_labels(ctx, model_dir)
    if derived:
        if nclasses is not None and derived != nclasses:
            ctx.log(f"[cfg] class count from labels.txt = {derived} (preset said {nclasses})")
        nclasses = derived
    tmpl = ctx.path(os.path.join(preset["assets_dir"], "config_infer.template.txt"))
    with open(tmpl) as f:
        text = f.read()
    text = (text.replace("@ONNX@", f"../model/{onnx}")
                .replace("@ENGINE@", f"../model/{engine}")
                .replace("@NCLASSES@", str(nclasses)))
    os.makedirs(ctx.path(os.path.join(model_dir, "config")), exist_ok=True)
    with open(ctx.path(cfg), "w") as f:
        f.write(text)
    ctx.log(f"[cfg] wrote {cfg}")


def _eval_labels(preset, model_dir):
    """build_eval_set --labels: 'model' -> the model's own config/labels.txt; otherwise a fixed target labels path."""
    el = preset.get("eval_labels", "model")
    if el == "model":
        return os.path.join(model_dir, "config", "labels.txt")
    return el


def _export(ctx, preset, checkpoint, onnx_rel, labels_rel):
    """run.sh: export_rtdetr.py --checkpoint <ckpt> --out-onnx <onnx> --out-labels <labels>. Idempotent on the ONNX."""
    if ctx.exists(onnx_rel):
        ctx.log(f"[skip] onnx exists: {onnx_rel}")
        return
    os.makedirs(os.path.dirname(ctx.path(onnx_rel)), exist_ok=True)
    os.makedirs(os.path.dirname(ctx.path(labels_rel)), exist_ok=True)
    ctx.sh([ctx.PY, os.path.join(preset["assets_dir"], "export_rtdetr.py"),
            "--checkpoint", checkpoint, "--out-onnx", onnx_rel, "--out-labels", labels_rel])


def _build_eval_set(ctx, preset, model_dir):
    """run.sh: build_eval_set.py on the eval split. Idempotent on ground_truth.json."""
    gt = os.path.join(model_dir, "eval", "eval_set", "ground_truth.json")
    if ctx.exists(gt):
        ctx.log(f"[skip] eval set exists: {gt}")
        return
    split = preset["eval_split"]
    imgs = _manifest_images(ctx, preset, split)
    ann = os.path.join(preset["data_dir"], f"annotations_{split}.json")
    cw, ch = preset["canvas"]
    ctx.sh([ctx.PY, ctx.SK + "/scripts/build_eval_set.py",
            "--labels", _eval_labels(preset, model_dir),
            "--out", os.path.join(model_dir, "eval", "eval_set"),
            "--local-coco", ann, "--coco-images-root", imgs,
            "--n-eval", str(preset["n_eval"]),
            "--canvas-width", str(cw), "--canvas-height", str(ch),
            "--bbox-format", preset["bbox_format"], "--resize-mode", preset["resize_mode"]])


def _eval_engine(ctx, preset, model_dir):
    """run.sh: gpu_guard then eval_engine.py. Idempotent on engine_metrics.json."""
    metrics = os.path.join(model_dir, "eval", "engine_metrics.json")
    if ctx.exists(metrics):
        ctx.log(f"[skip] metrics exist: {metrics}")
        return
    ctx.sh(["bash", ctx.SK + "/scripts/gpu_guard.sh"], check=False)
    ev = os.path.join(model_dir, "eval")
    ctx.sh([ctx.PY, ctx.SK + "/scripts/eval_engine.py",
            "--app", ctx.SK + "/scripts/ds_image_eval",
            "--engine-config", os.path.join(model_dir, "config", "config_infer.txt"),
            "--eval-set", os.path.join(ev, "eval_set"),
            "--labels", os.path.join(model_dir, "config", "labels.txt"),
            "--predictions", os.path.join(ev, "predictions_engine.json"),
            "--metrics", metrics,
            "--perf-out", os.path.join(ev, "perf.json")])


# ----------------------------------------------------------------------------- baseline stages (-> GATE A)
def _st_build_app(ctx, preset):
    if ctx.exists(ctx.SK + "/scripts/ds_image_eval"):
        ctx.log("[skip] ds_image_eval binary exists")
        return
    ctx.sh(["make", "-C", ctx.SK + "/scripts"])


def _st_ingest(ctx, preset):
    split = preset["eval_split"]
    refresh = bool(preset.pop("_refresh_dataset_cache", False))
    if refresh and os.environ.get("DS_DATA_CACHE_ROOT"):
        # A changed source invalidates every derived result. Clear only this run's known
        # artifact paths so the normal idempotent stages rebuild a coherent before/after.
        for target in (
            preset["data_dir"], _train_data_dir(ctx, preset),
            preset.get("orig_dir"), preset.get("ft_dir"), preset.get("run_dir"),
            preset.get("comparison"), preset.get("report_pdf"), preset.get("samples_dir"),
        ):
            if not target:
                continue
            path = ctx.path(target)
            resolved = os.path.realpath(path)
            if os.path.commonpath([resolved, ctx.ROOT]) != ctx.ROOT:
                raise RuntimeError(f"refusing cache refresh outside working root: {target}")
            if os.path.isdir(path):
                shutil.rmtree(path)
                ctx.log(f"[cache] refresh removed {target}")
            elif os.path.isfile(path):
                os.remove(path)
                ctx.log(f"[cache] refresh removed {target}")
    if ctx.exists(os.path.join(preset["data_dir"], f"annotations_{split}.json")):
        ctx.log("[skip] dataset already ingested")
        return
    kind = preset["ingest"]["kind"]
    old_refresh = ctx.run_env.get("DS_DATA_CACHE_REFRESH")
    if refresh:
        ctx.run_env["DS_DATA_CACHE_REFRESH"] = "1"
    try:
        if kind == "convert_pcb":
            ctx.sh([ctx.PY, ctx.SK + "/examples/rtdetr-pcb/convert_pcb.py",
                    "--repo-id", preset["dataset_id"], "--dest", preset["data_dir"]])
        else:  # ingest_dataset (Roboflow / script-based COCO zips)
            ctx.sh([ctx.PY, ctx.SK + "/scripts/ingest_dataset.py",
                    "--repo-id", preset["dataset_id"], "--dest", preset["data_dir"],
                    "--splits", *preset["ingest"]["splits"]])
    finally:
        if old_refresh is None:
            ctx.run_env.pop("DS_DATA_CACHE_REFRESH", None)
        else:
            ctx.run_env["DS_DATA_CACHE_REFRESH"] = old_refresh


def _st_deploy_orig(ctx, preset):
    od = preset["orig_dir"]
    _export(ctx, preset, preset["model_id"],
            os.path.join(od, "model", preset["orig_onnx"]),
            os.path.join(od, "config", "labels.txt"))
    _build_parser(ctx, preset, od)
    _write_cfg(ctx, preset, od, preset["orig_onnx"], preset["orig_engine"], preset.get("orig_nclasses"))


def _st_eval_set_orig(ctx, preset):
    _build_eval_set(ctx, preset, preset["orig_dir"])


def _st_eval_orig(ctx, preset):
    od = preset["orig_dir"]
    _eval_engine(ctx, preset, od)
    # authoritative BASELINE engine perf from trtexec on the original ONNX (fail-soft), mirroring
    # the fine-tuned leg — so the report compares like-for-like trtexec_qps, not pipeline_fps.
    perf = os.path.join(od, "eval", "perf.json")
    if not ctx.exists(perf) or "trtexec_qps" not in _safe_json(ctx.path(perf)):
        ctx.sh([ctx.PY, ctx.SK + "/scripts/bench_trtexec.py",
                "--onnx", os.path.join(od, "model", preset["orig_onnx"]),
                "--out", perf], check=False)


def baseline_plan(preset):
    return [
        Stage("build_app",    "Build ds_image_eval C app",            _st_build_app),
        Stage("ingest",       "Ingest dataset → COCO",           _st_ingest),
        Stage("deploy_orig",  "Deploy original (export + parser + config)", _st_deploy_orig),
        Stage("eval_set_orig","Build canonical eval set (original)",  _st_eval_set_orig),
        Stage("eval_orig",    "Run DeepStream baseline eval",         _st_eval_orig),
    ]


# ----------------------------------------------------------------------------- finetune stages (-> GATE B)
# Training config template for CUSTOM datasets. Hyperparameters are chosen from THIS dataset's
# properties (see _train_hparams) — never copied from a prior run. train.py reads label_names to
# build id2label + num_labels; n_train/n_eval are informational (train.py loads the full arrow set).
_CONFIG_TEMPLATE = """# Auto-generated by the DeepStream Eval & Fine-tune Web UI for a custom run.
model_id: {model_id}
model_short_name: {short}
task: object-detection
auto_model_class: AutoModelForObjectDetection
ignore_mismatched_sizes: true
dataset_id: {dataset_id}
local_data_dir: {local_data_dir}
label_names: [{label_names}]
n_train: {n_train}
n_eval: {n_eval}
output_dir: ./checkpoints
num_train_epochs: {epochs}
per_device_train_batch_size: {batch}
per_device_eval_batch_size: {batch}
eval_accumulation_steps: 8
gradient_accumulation_steps: 1
learning_rate: {lr}
warmup_ratio: {warmup}
lr_scheduler_type: {sched}
weight_decay: 1.0e-4
bf16: true
dataloader_num_workers: 8
remove_unused_columns: false
eval_strategy: epoch
save_strategy: epoch
save_total_limit: 1
load_best_model_at_end: false
metric_for_best_model: eval_loss
greater_is_better: false
report_to: none
logging_steps: 20
disable_tqdm: true
"""


def _train_hparams(n_classes):
    """Pick training hyperparameters from THIS dataset's class count — not inherited from any
    prior run. RT-DETR's varifocal classification head diverges at lr=1e-4 once there are several
    classes (logits collapse to the focal prior -> near-zero scores, eval_loss climbs, while box
    regression still learns). 1e-4 is fine for 1-2 classes; >=3 classes need the gentler, validated
    2.5e-5 + longer warmup + cosine (the recipe proven on the 6-class PCB demo)."""
    if n_classes <= 2:
        return {"lr": "1.0e-4", "warmup": 0.05, "sched": "linear", "batch": 8}
    return {"lr": "2.5e-5", "warmup": 0.1, "sched": "cosine", "batch": 8}


def _gen_config(ctx, preset, warm_start_from=None, epochs_override=None):
    """Generate run_dir/config.yaml for a custom dataset from labels ingested earlier.
    warm_start_from: a local weights path (relative to run_dir) to start from instead of the HF
    base model — used by 'continue training' to extend a run with a FRESH N-epoch schedule
    (no resume_from_checkpoint, so the LR schedule restarts and the model actually keeps learning)."""
    rd = preset["run_dir"]
    lf = ctx.path(os.path.join(preset["data_dir"], "labels.txt"))
    with open(lf) as f:
        labels = [ln.strip() for ln in f if ln.strip()]
    n_train = 0
    try:
        with open(ctx.path(os.path.join(preset["data_dir"], "annotations_train.json"))) as f:
            n_train = len(json.load(f).get("images", []))
    except Exception:
        pass
    hp = _train_hparams(len(labels))           # auto defaults by class count
    ov = preset.get("hparams") or {}           # user overrides (form Advanced settings); blank = auto
    for k in ("lr", "warmup", "sched", "batch"):
        v = ov.get(k)
        if v not in (None, "", "auto"):
            hp[k] = v
            ctx.log(f"[prep] override {k}={v} (user-set)")
    epochs = epochs_override if epochs_override else preset.get("epochs", 30)
    text = _CONFIG_TEMPLATE.format(
        model_id=(warm_start_from or preset["model_id"]),
        short=os.path.basename(rd),
        dataset_id=preset["dataset_id"],
        local_data_dir=json.dumps(_train_data_dir(ctx, preset)),
        label_names=", ".join(labels),
        n_train=n_train, n_eval=120, epochs=epochs,
        lr=hp["lr"], warmup=hp["warmup"], sched=hp["sched"], batch=hp["batch"])
    if warm_start_from:
        # continue-training: a FRESH N-epoch schedule from the given weights (no resume).
        ctx.log(f"[prep] warm-start from {warm_start_from} for {epochs} epochs (fresh LR schedule)")
    else:
        # Warm-resume: if a prior checkpoint exists (and training isn't already complete), continue
        # from the latest one so bumping `epochs` extends the same run instead of restarting.
        ck = _latest_checkpoint_name(ctx, rd)
        if ck and not ctx.exists(os.path.join(rd, "checkpoints", "final")):
            text += f"resume_from_checkpoint: checkpoints/{ck}\n"
            ctx.log(f"[prep] will resume training from checkpoints/{ck}")
    with open(ctx.path(os.path.join(rd, "config.yaml")), "w") as f:
        f.write(text)
    ctx.log(f"[prep] generated config.yaml ({len(labels)} classes, {epochs} epochs, "
            f"lr={hp['lr']} warmup={hp['warmup']} {hp['sched']})")


def _st_prep_run(ctx, preset):
    rd = preset["run_dir"]
    os.makedirs(ctx.path(rd), exist_ok=True)
    if preset.get("config_generate"):
        _gen_config(ctx, preset)
    else:
        shutil.copy(ctx.path(preset["config_yaml"]), ctx.path(os.path.join(rd, "config.yaml")))
        with open(ctx.path(os.path.join(rd, "config.yaml")), "a") as f:
            f.write(f"\nlocal_data_dir: {json.dumps(_train_data_dir(ctx, preset))}\n")
    shutil.copy(ctx.path(os.path.join(preset["assets_dir"], "train.py")),
                ctx.path(os.path.join(rd, "train.py")))
    ctx.log(f"[prep] config.yaml + train.py -> {rd}")


def _st_data_hf(ctx, preset):
    rd = preset["run_dir"]
    data_dir = _train_data_dir(ctx, preset)
    if ctx.exists(os.path.join(data_dir, "train")):
        ctx.log("[skip] HF arrow data already built")
        return
    dd = preset["data_dir"]
    # train split
    ctx.sh([ctx.PY, ctx.SK + "/scripts/coco_to_hf.py",
            "--coco", os.path.join(dd, "annotations_train.json"),
            "--images", _manifest_images(ctx, preset, "train"),
            "--out", os.path.join(data_dir, "train")])
    # 120-image eval subset from the validation split (in-process; mirrors run.sh's inline python)
    valid_ann = ctx.path(os.path.join(dd, "annotations_valid.json"))
    with open(valid_ann) as f:
        c = json.load(f)
    keep = {im["id"] for im in c["images"][:120]}
    c["images"] = [i for i in c["images"] if i["id"] in keep]
    c["annotations"] = [a for a in c["annotations"] if a["image_id"] in keep]
    subset = ctx.path(os.path.join("build", f"{os.path.basename(rd)}_valid_120.json"))
    os.makedirs(os.path.dirname(subset), exist_ok=True)
    with open(subset, "w") as f:
        json.dump(c, f)
    ctx.sh([ctx.PY, ctx.SK + "/scripts/coco_to_hf.py",
            "--coco", subset, "--images", _manifest_images(ctx, preset, "valid"),
            "--out", os.path.join(data_dir, "eval")])


def _latest_checkpoint_name(ctx, rd):
    """Name of the highest-epoch checkpoint-* dir under run_dir/checkpoints, or None."""
    ckdir = ctx.path(os.path.join(rd, "checkpoints"))
    best, best_ep = None, -1.0
    for name in (os.listdir(ckdir) if os.path.isdir(ckdir) else []):
        if not name.startswith("checkpoint-"):
            continue
        try:
            with open(os.path.join(ckdir, name, "trainer_state.json")) as f:
                ep = json.load(f).get("epoch", 0)
        except Exception:
            continue
        if ep > best_ep:
            best, best_ep = name, ep
    return best


def _checkpoint_name_to_promote(ctx, rd):
    """checkpoint-* dir to promote to final: best-by-metric when configured, else latest epoch."""
    ckdir = ctx.path(os.path.join(rd, "checkpoints"))
    cfg = {}
    try:
        import yaml
        with open(ctx.path(os.path.join(rd, "config.yaml"))) as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        pass
    if cfg.get("load_best_model_at_end"):
        states = []
        for name in (os.listdir(ckdir) if os.path.isdir(ckdir) else []):
            if not name.startswith("checkpoint-"):
                continue
            tsp = os.path.join(ckdir, name, "trainer_state.json")
            if not os.path.exists(tsp):
                continue
            try:
                with open(tsp) as f:
                    ts = json.load(f)
                states.append((name, ts))
            except Exception:
                continue
        if states:
            states.sort(key=lambda x: x[1].get("epoch", 0), reverse=True)
            bcp = states[0][1].get("best_model_checkpoint")
            if bcp:
                return os.path.basename(str(bcp).rstrip("/"))
    return _latest_checkpoint_name(ctx, rd)


def _promote_latest_checkpoint(ctx, rd):
    """After a user soft-stop, train.py never wrote 'final' — promote the configured checkpoint
    (best-by-metric when load_best_model_at_end, else the latest completed epoch)."""
    best = _checkpoint_name_to_promote(ctx, rd)
    if best is None:
        raise RuntimeError("stopped before any epoch completed — no checkpoint to evaluate")
    ckdir = ctx.path(os.path.join(rd, "checkpoints"))
    shutil.copytree(os.path.join(ckdir, best), os.path.join(ckdir, "final"))
    ctx.log(f"[stop] promoted {best} -> checkpoints/final")


def _st_finetune(ctx, preset):
    rd = preset["run_dir"]
    final = os.path.join(rd, "checkpoints", "final")
    if ctx.exists(final):
        ctx.log("[skip] fine-tuned checkpoint already exists")
        return
    # run.sh: ( cd run_dir && /work/build/.venv_train/bin/python train.py --config config.yaml > train.log 2>&1 )
    # log_file -> runs/<name>/train.log so the server's live curve parser and make_report can read it.
    # soft_stop=True: a user "Stop" ends training cleanly and we evaluate the latest epoch below.
    ctx.sh([ctx.PY, "train.py", "--config", "config.yaml"],
           cwd=ctx.path(rd), log_file=ctx.path(os.path.join(rd, "train.log")), soft_stop=True)
    if not ctx.exists(final):
        _promote_latest_checkpoint(ctx, rd)
    # DURABILITY: fold this segment's curve into curve_history.json NOW (the moment training ends),
    # not at the next continue's prep — otherwise a later re-run that truncates train.log would lose
    # it (the bug that made a 40-epoch run display as 22). Then truncate train.log so merged() reads
    # the segment from history exactly once (no double-count). Raw stdout stays in build/ui_runs/<id>.log.
    tl = ctx.path(os.path.join(rd, "train.log"))
    if any("'epoch'" in ln for ln in (open(tl).read().splitlines() if os.path.exists(tl) else [])):
        ctx.sh([ctx.PY, ctx.SK + "/scripts/curve_history.py", "commit", "--run-dir", ctx.path(rd)])
        open(tl, "w").close()
        ctx.log("[curve] segment committed to curve_history.json (durable); train.log cleared")


def _st_deploy_ft(ctx, preset):
    fd = preset["ft_dir"]
    _st_sync_final_for_eval(ctx, preset)
    _export(ctx, preset, os.path.join(preset["run_dir"], "checkpoints", "final"),
            os.path.join(fd, "model", preset["ft_onnx"]),
            os.path.join(fd, "config", "labels.txt"))
    _build_parser(ctx, preset, fd)
    _write_cfg(ctx, preset, fd, preset["ft_onnx"], preset["ft_engine"], preset.get("ft_nclasses"))
    # Record which epoch was deployed + the stored-model paths (shared with the CLI run.sh, so UI
    # and CLI emit an identical deployed_checkpoint.json read by /results/summary and §2b).
    ctx.sh([ctx.PY, ctx.SK + "/scripts/record_deployed_checkpoint.py",
            "--run-dir", preset["run_dir"], "--ft-dir", fd,
            "--onnx-name", preset["ft_onnx"], "--engine-name", preset["ft_engine"]], check=False)


def _st_eval_set_ft(ctx, preset):
    _build_eval_set(ctx, preset, preset["ft_dir"])


def _st_eval_ft(ctx, preset):
    fd = preset["ft_dir"]
    _eval_engine(ctx, preset, fd)
    # authoritative engine perf from trtexec on the ONNX (run.sh allows this to fail soft)
    perf = os.path.join(fd, "eval", "perf.json")
    if not ctx.exists(perf) or "trtexec_qps" not in _safe_json(ctx.path(perf)):
        ctx.sh([ctx.PY, ctx.SK + "/scripts/bench_trtexec.py",
                "--onnx", os.path.join(fd, "model", preset["ft_onnx"]),
                "--out", perf], check=False)


def _embed_checkpoint_metrics(ctx, preset):
    """Attach PyTorch training mAP at the deployed checkpoint epoch to comparison.json."""
    try:
        import checkpoint_to_evaluate
        info = checkpoint_to_evaluate.info(ctx.path(preset["run_dir"]))
        if not info:
            return
        path = ctx.path(preset["comparison"])
        cmp = _safe_json(path)
        if not cmp:
            return
        cmp["checkpoint_evaluated"] = info
        dep = _safe_json(ctx.path(os.path.join(preset["ft_dir"], "model", "deployed_checkpoint.json")))
        if dep.get("deployed_epoch") is not None:
            cmp["deployed_checkpoint_epoch"] = dep["deployed_epoch"]
        with open(path, "w") as f:
            json.dump(cmp, f, indent=2)
        ctx.log(f"[report] checkpoint {info.get('selection')} epoch {info.get('epoch')}: "
                f"training mAP={info.get('map')} mAP@50={info.get('map_50')}")
    except Exception as e:
        ctx.log(f"[report] checkpoint metrics embed skipped: {e}")


def _st_report(ctx, preset):
    # A completed CLI/import-agent baseline may live at a declared alias. Read it for the
    # comparison, while all baseline writers continue to target preset["orig_dir"].
    od, fd = resolve_orig_dir(ctx.ROOT, preset), preset["ft_dir"]
    rm = preset["report_meta"]
    # system/tool provenance (fail-soft, exactly as run.sh)
    ctx.sh([ctx.PY, ctx.SK + "/scripts/collect_env.py",
            "--out", "build/env_info.json", "--image", DS_IMAGE], check=False)
    # merge original-deployed vs fine-tuned-deployed; each leg uses ITS OWN trtexec perf
    ctx.sh([ctx.PY, ctx.SK + "/scripts/compare_runs.py",
            "--original-metrics", os.path.join(od, "eval", "engine_metrics.json"),
            "--finetuned-metrics", os.path.join(fd, "eval", "engine_metrics.json"),
            "--original-perf", os.path.join(od, "eval", "perf.json"),
            "--finetuned-perf", os.path.join(fd, "eval", "perf.json"),
            # train (COCO) + eval (ground_truth) class distribution -> embedded in the comparison JSON
            "--train-coco", os.path.join(preset["data_dir"], "annotations_train.json"),
            "--eval-gt", os.path.join(fd, "eval", "eval_set", "ground_truth.json"),
            "--output", preset["comparison"]])
    _embed_checkpoint_metrics(ctx, preset)
    # qualitative before/after frames (GT | baseline | fine-tuned)
    ctx.sh([ctx.PY, ctx.SK + "/scripts/sample_overlays.py",
            "--orig-eval-set", os.path.join(od, "eval", "eval_set"),
            "--orig-predictions", os.path.join(od, "eval", "predictions_engine.json"),
            "--ft-eval-set", os.path.join(fd, "eval", "eval_set"),
            "--ft-predictions", os.path.join(fd, "eval", "predictions_engine.json"),
            "--out-dir", preset["samples_dir"], "--n", "6", "--conf", "0.2",
            *preset.get("sample_overlays_extra", [])])
    # final before/after PDF
    ctx.sh([ctx.PY, ctx.SK + "/scripts/make_report.py",
            "--comparison", preset["comparison"], "--samples-dir", preset["samples_dir"],
            "--config", os.path.join(preset["run_dir"], "config.yaml"),
            "--train-log", os.path.join(preset["run_dir"], "train.log"),
            "--deployed-checkpoint", os.path.join(fd, "model", "deployed_checkpoint.json"),
            "--precision", "fp16", "--env", "build/env_info.json", "--image", DS_IMAGE,
            "--model-id", rm["model_id"], "--model-link", rm["model_link"],
            "--model-license", rm["model_license"], "--model-desc", rm["model_desc"],
            "--dataset-id", rm["dataset_id"], "--dataset-link", rm["dataset_link"],
            "--dataset-desc", rm["dataset_desc"],
            "--output", preset["report_pdf"]])


def finetune_plan(preset):
    return [
        Stage("prep_run",   "Prepare run dir + config",                  _st_prep_run),
        Stage("data_hf",    "Convert dataset → HF Arrow (train+eval)", _st_data_hf),
        Stage("finetune",   "Fine-tune (live loss curve)",               _st_finetune),
        Stage("deploy_ft",  "Deploy fine-tuned (export + parser + config)", _st_deploy_ft),
        Stage("eval_set_ft","Build canonical eval set (fine-tuned)",     _st_eval_set_ft),
        Stage("eval_ft",    "Run DeepStream eval (fine-tuned) + bench",  _st_eval_ft),
        Stage("report",     "Compare + render samples + PDF report",     _st_report),
    ]


def _st_prep_continue(ctx, preset):
    """Continue-training prep (warm-start): back up the current weights, point the config at them
    for a FRESH N-epoch schedule, and clear the downstream artifacts so deploy/eval/report re-run
    on the continued model. Overwrites the run in place; keeps the pre-continue weights as
    checkpoints/final_prev."""
    rd = preset["run_dir"]
    ck = os.path.join(rd, "checkpoints")
    final, prev = os.path.join(ck, "final"), os.path.join(ck, "final_prev")
    if not ctx.exists(final):
        raise RuntimeError("no checkpoints/final to continue from")
    # PRESERVE the graph: fold the just-finished segment's train.log into the run's persistent
    # curve_history.json (absolute/cumulative epochs) BEFORE the next warm-start segment overwrites
    # train.log. Without this the continued run's loss curve would restart at epoch 1 and the prior
    # history would be lost. Invoked as a subprocess (pipeline invokes scripts, never imports them).
    ctx.sh([ctx.PY, ctx.SK + "/scripts/curve_history.py", "commit", "--run-dir", ctx.path(rd)])
    if ctx.exists(prev):
        shutil.rmtree(ctx.path(prev), ignore_errors=True)
    shutil.move(ctx.path(final), ctx.path(prev))                 # warm-start source + backup
    ckabs = ctx.path(ck)                                          # drop stale per-epoch checkpoints
    for n in (os.listdir(ckabs) if os.path.isdir(ckabs) else []):
        if n.startswith("checkpoint-"):
            shutil.rmtree(os.path.join(ckabs, n), ignore_errors=True)
    os.makedirs(ctx.path(rd), exist_ok=True)
    _gen_config(ctx, preset, warm_start_from=os.path.join("checkpoints", "final_prev"),
                epochs_override=preset.get("continue_epochs"))
    shutil.copy(ctx.path(os.path.join(preset["assets_dir"], "train.py")),
                ctx.path(os.path.join(rd, "train.py")))
    # clear downstream so the (idempotent) deploy/eval/report stages re-run on the continued model
    for t in (preset["ft_dir"], preset["comparison"], preset["report_pdf"], preset["samples_dir"]):
        ap = ctx.path(t)
        if os.path.isdir(ap):
            shutil.rmtree(ap, ignore_errors=True)
        elif os.path.isfile(ap):
            os.remove(ap)
    # NOTE: deliberately do NOT delete build/ui_charts/<key>_*.png here — they regenerate from the
    # MERGED curve (history + new segment), and leaving them avoids a blank graph mid-continue so
    # the page keeps showing the prior training history while the new epochs run.
    ctx.log(f"[continue] warm-start +{preset.get('continue_epochs')} epochs from checkpoints/final_prev "
            f"(prior loss curve preserved in curve_history.json)")


def continue_training_plan(preset):
    """Warm-start an existing fine-tuned run for N MORE epochs, then re-deploy/eval/report.
    Reuses the standard stages — only prep differs (warm-start + clear downstream)."""
    return [
        Stage("build_app",   "Build ds_image_eval C app",                 _st_build_app),
        Stage("prep_run",    "Prepare continue (warm-start +N epochs)",   _st_prep_continue),
        Stage("finetune",    "Fine-tune (+N epochs, warm-start)",          _st_finetune),
        Stage("deploy_ft",   "Deploy fine-tuned (export + parser + config)", _st_deploy_ft),
        Stage("eval_set_ft", "Build canonical eval set (fine-tuned)",      _st_eval_set_ft),
        Stage("eval_ft",     "Run DeepStream eval (fine-tuned) + bench",   _st_eval_ft),
        Stage("report",      "Compare + render samples + PDF report",      _st_report),
    ]


def _st_sync_final_for_eval(ctx, preset):
    """Before deploy/eval: copy the intended checkpoint (best-by-metric or latest epoch) into
    checkpoints/final. Always overwrites stale final — e.g. a root-owned early checkpoint left
    when training could not save at the end, or a prior partial UI evaluate."""
    rd = preset["run_dir"]
    ckdir = ctx.path(os.path.join(rd, "checkpoints"))
    name = _checkpoint_name_to_promote(ctx, rd)
    if not name:
        if not ctx.exists(os.path.join(rd, "checkpoints", "final")):
            raise RuntimeError("no checkpoint to evaluate")
        ctx.log("[promote] using existing checkpoints/final (no checkpoint-* dirs)")
        return
    src, dst = os.path.join(ckdir, name), os.path.join(ckdir, "final")
    if ctx.exists(dst):
        shutil.rmtree(ctx.path(dst), ignore_errors=True)
    shutil.copytree(ctx.path(src), ctx.path(dst))
    ctx.log(f"[promote] checkpoints/final <- {name}")


def _st_clear_ft_deploy(ctx, preset):
    """Drop stale ONNX/engine/metrics so re-evaluate always re-exports and re-scores in DeepStream."""
    fd = preset["ft_dir"]
    removed = []
    for rel in (
        os.path.join("model", preset["ft_onnx"]),
        os.path.join("model", preset["ft_engine"]),
        os.path.join("eval", "engine_metrics.json"),
        os.path.join("eval", "predictions_engine.json"),
        os.path.join("eval", "perf.json"),
    ):
        p = ctx.path(os.path.join(fd, rel))
        if os.path.isfile(p):
            os.remove(p)
            removed.append(rel)
    model_dir = ctx.path(os.path.join(fd, "model"))
    if os.path.isdir(model_dir):
        for n in os.listdir(model_dir):
            if ".engine" in n:
                os.remove(os.path.join(model_dir, n))
                removed.append(f"model/{n}")
    for rel in (preset.get("comparison"), preset.get("report_pdf")):
        if not rel:
            continue
        p = ctx.path(rel)
        if os.path.isfile(p):
            os.remove(p)
            removed.append(rel)
    sd = ctx.path(preset.get("samples_dir", ""))
    if sd and os.path.isdir(sd):
        shutil.rmtree(sd, ignore_errors=True)
        removed.append(preset.get("samples_dir", ""))
    ctx.log(f"[clear] removed {len(removed)} stale deploy/eval artifact(s) for fresh DeepStream run")


def eval_plan(preset):
    """Evaluate-only: deploy the current checkpoints/final + measure + report. Used by the
    'Evaluate latest epoch' control after a soft-stop, OR to evaluate the last saved checkpoint of a
    stopped/errored run (promote_if_needed makes `final` from the latest checkpoint-* when missing)."""
    return [
        Stage("build_app",  "Build ds_image_eval C app",                 _st_build_app),
        Stage("promote",    "Sync checkpoints/final for evaluate",       _st_sync_final_for_eval),
        Stage("clear_ft",   "Clear stale deploy artifacts",              _st_clear_ft_deploy),
        Stage("deploy_ft",  "Deploy fine-tuned (export + parser + config)", _st_deploy_ft),
        Stage("eval_set_ft","Build canonical eval set (fine-tuned)",     _st_eval_set_ft),
        Stage("eval_ft",    "Run DeepStream eval (fine-tuned) + bench",  _st_eval_ft),
        Stage("report",     "Compare + render samples + PDF report",     _st_report),
    ]


def _safe_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}
