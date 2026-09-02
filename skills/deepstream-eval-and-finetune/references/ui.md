<!--
Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-->

# Optional Web UI

## Windows/WSL dataset cache

Windows/WSL UI launches automatically mount a persistent Linux-backed project volume at
`/work/data`. Setup shows its name and staged-source size. **Refresh source cache and rebuild this
run** is unchecked by default: reuse avoids NTFS rescans; checking it atomically restages the local
source and invalidates that run's derived baseline, checkpoint, deploy, evaluation, and report
artifacts before rebuilding. Dataset bytes/splits and all UI result calculations are unchanged.

Checkpoints, models, metrics, logs, reports, samples, and run history remain under the Windows
working root, so closing/restarting the UI does not lose results. See
[windows.md](windows.md) for environment controls and volume cleanup.

An **optional** turnkey web UI (`app/`) is available for the two validated demos
(RT-DETR + aerial-sheep, RT-DETR + PCB). It is a convenience layer only — the
**CLI / `examples/*/run.sh` flow remains the canonical engine and is unchanged**.
The UI runs the *same scripts*, in the *same DeepStream container*, and produces
the *same artifacts*; it just adds a guided, visual front-end with two review gates.

**Launch (inside the DeepStream container, from the working root):**

Before launching, the skill must ask **Start UI** or **Skip UI**. Start only after an explicit
selection; Skip means no launcher, Uvicorn process, or UI container. For Start, confirm the port
(8078 default) and bind mode (`127.0.0.1` default), inspect listeners/containers, and never stop or
replace another session. If occupied, use a user-approved free port. After launch, verify HTTP `/`
and semantic readiness at `/presets`; HTTP 200 alone is insufficient. Dry runs present the choice
but never launch or stop anything.

```bash
bash .claude/skills/deepstream-eval-and-finetune/app/launch.sh   # docker run … -p 127.0.0.1:8078:8078 … uvicorn
```

The launcher binds the published Docker port to `127.0.0.1` by default. Do not expose this control UI on a shared network without authentication. `launch.sh` runs `app/setup.sh` automatically the first time to add FastAPI/Uvicorn to `build/.venv_train`.

**Datasets outside the project root** must be bind-mounted, or the container (which only sees the
project at `/work`) can't read them and ingest fails with *"local dataset path not found inside this
container"*. Pass **`DATASETS_ROOT`** — one or more roots (space-separated), each mounted **read-only at
the same absolute path**, so a dataset path under them (or an in-project symlink to them) resolves
unchanged inside the container. No path is hardcoded:

```bash
DATASETS_ROOT=/mnt/smb_share bash .../app/launch.sh                 # mount one root
DATASETS_ROOT="/mnt/smb_share /data/sets" bash .../app/launch.sh    # mount several
DATASET_MOUNT=/host/src:/container/dst bash .../app/launch.sh       # single explicit mount (legacy)
```

Open **http://localhost:8078**. The UI walks six steps:

1. **Setup** — start a run from any **RT-DETR-family model + ingestable detection
   dataset** (the form), or re-open a previous run from the **Runs — history**
   list (the two built-in demos seed it). The dataset field accepts a HuggingFace
   id or URL, an http(s) link to a `.zip`, or a local path (directory or zip) in a
   YOLO / Roboflow-COCO layout — `scripts/ingest_dataset.py` resolves all of these. Custom runs are registered and persisted
   to `build/ui_runs/index.json`, so they remain as history across restarts; each
   derives its own `data/`, `models/`, `runs/`, and `reports/` paths, generates a
   training config from the ingested labels, and derives the nvinfer class count
   from the exported `labels.txt`. **Training hyperparameters are chosen from the
   dataset's own class count, not copied from any prior run** (`_train_hparams`):
   ≤2 classes use `lr 1e-4`; **≥3 classes use the gentler `lr 2.5e-5` + `warmup 0.1`
   + cosine** — because RT-DETR's varifocal classification head diverges at `1e-4`
   with several classes (scores collapse to ~0, eval_loss climbs, box regression
   still learns), a known failure mode with several classes.
   The form's optional **Advanced training settings** (learning rate, warmup, LR
   scheduler, batch size) override these per custom run; left on `auto`/blank they
   fall back to the class-count rule (each field is independent — override only what
   you want, the rest stay auto). Epochs is always editable.
2. **Baseline** — deploy the original model + run baseline DeepStream eval (per-class
   mAP, FPS, GT-vs-baseline sample frames; click any frame to zoom). A run started from
   the form or a history row **automatically continues into fine-tuning** — no manual
   gate (`/run/baseline` and `/run/custom` run baseline → fine-tune → done as one job;
   the baseline results stay viewable on this step).
3. **Fine-tune** — runs automatically after baseline, then deploys + evals the
   fine-tuned model. **For AutoML/HPO runs (detected via `is_automl` on `/presets` — the
   run has `logs/<run>/<ts>/automl.log`), the step splits into two sub-tabs: `3a AutoML
   sweep` and `3b Fine-tuning`.** `3a` (`/results/automl`) shows the **search space + live
   leaderboard** — one row per trial with its lr / warmup / scheduler / weight-decay, best
   per-epoch eval mAP, and status (pending → running → scored → ★ winner), the winning trial,
   and the winning config that feeds the final full-length training. It also plots **all trials'
   per-epoch eval-mAP curves overlaid on one canvas** (winner bold, color-keyed legend) — parsed
   per trial from `logs/<run>/<ts>/trials/trial_*.log` and returned as `curve` per trial — so you
   can *see* the search, not just the numbers. These trial curves live only on 3a; 3b stays the
   final fine-tuning curve. `3b` is the normal training
   view (below). Non-AutoML runs skip the sub-tabs and show `3b` directly. The sweep table
   live-refreshes on the same 20s poll as the curve. Shows a **plain-language summary**
   (`/results/summary`: images
   trained/evaluated, epochs, learning rate, before→after mAP@50, best class) for
   non-ML reviewers, plus a **dataset class-distribution table** (per-class train + eval
   instance/image counts — greyed rows = classes absent from the eval slice), alongside
   the live + report charts. Shows **both** a real-time **live loss curve** (canvas, redrawn
   each poll) and the **report-style charts** (the same matplotlib training-curve and
   accuracy figures embedded in the PDF, served via `/chart`).
   Also shows a **live per-epoch mAP curve** (`eval_map` / `eval_map_50`) + a **per-class AP
   table for the latest epoch**, so you can watch accuracy rise/fall *during* training and catch
   a regression without waiting for the deploy. `train.py` wires its `compute_metrics`
   (torchmetrics) into the Trainer, logging `eval_map`/`eval_map_50`/`eval_cls_<name>` each epoch;
   it's **one growing graph** (a new point per epoch, redrawn each poll — not a file per epoch).
   This is a **PyTorch proxy on the 120-image training eval subset** (a *trend* signal), **not**
   the deployed DeepStream mAP, and it is labelled as such in the UI. It computes from the eval
   loop's existing predictions — **no DeepStream / ONNX export / engine build per epoch**.
   **Checkpoint selection is unchanged** (still deploys the most-trained checkpoint;
   `load_best_model_at_end: false`); these metrics are informational. A metric error is caught
   (try/except) so it can never kill a run. Viewable again after
   completion for any run via `/results/curve`. Controls during training: **Stop
   training** (`/run/stop` — halts and promotes the latest *completed* epoch to
   `checkpoints/final`, then **pauses** at phase `awaiting_eval`) and **Cancel**
   (`/run/cancel` — hard abort, no evaluation). When paused, an **Evaluate latest
   epoch** button (`/run/evaluate` → `eval_plan`: deploy → eval → report on `final`)
   runs the measurement. Natural completion at N epochs auto-evaluates. Stop and
   evaluate are deliberately separate steps so you can halt, inspect, then choose.
   A Stop is treated as a **clean** stop even though it SIGTERMs the trainer (so it pauses
   instead of erroring), and **any run that has a saved checkpoint is evaluable from the UI** —
   `eval_plan` auto-promotes the latest `checkpoint-*` to `final` if needed, and the Evaluate
   control also appears for a re-opened stopped/errored run (`has_checkpoint`), so the last
   epoch is always evaluable without the terminal.
   **After a run completes**, a **Continue training: +N epochs** control
   (`/run/continue` → `continue_training_plan`) extends it **without redoing the
   first N epochs**: it **warm-starts** from the existing `checkpoints/final`
   (moved to `checkpoints/final_prev` as both the source and a backup) and trains
   N *more* epochs with a **fresh LR schedule** — deliberately a warm-start, not a
   true `resume_from_checkpoint`, because resuming reloads the already-decayed
   cosine LR (~0) so the model would barely learn. `train.py` needs no change
   (`from_pretrained` accepts the local `final_prev` path). The run is **overwritten
   in place** (same history key, no duplicate) and the deploy/eval/report/charts
   re-run on the new weights. Use this when a class is under-trained — e.g. a class
   that stays at AP 0 (localized but misclassified as another class) until more epochs —
   a UI **+N** continue is the fix, not a from-scratch restart.
   The continue **PRESERVES and EXTENDS the loss graph** rather than restarting it:
   because `train.log` is truncated each run and the warm-start segment counts epochs
   from 0, each completed segment is folded into a persistent
   `runs/<name>/curve_history.json` (absolute/cumulative epochs) by
   `scripts/curve_history.py` — the single source of truth shared by the server,
   `make_report.py` and `record_deployed_checkpoint.py`. Every curve consumer shows
   `history + (current train.log offset by the prior total)`, so the live canvas, the
   `/chart` PNGs, the PDF training curve, the summary's epoch count and the deployed
   epoch all run continuously (e.g. `1 … 20 | 21 … 50`). A never-continued run has no
   history file and behaves exactly as before. The commit happens once per continue in
   `_st_prep_continue` (via the `curve_history.py commit` CLI, before the next segment
   overwrites `train.log`), and the cached charts are deliberately **not** deleted so
   the page keeps the prior curve on screen while the new epochs run.
4. **Compare** — two labeled sections: **① Training eval** (PyTorch mAP at the selected/best checkpoint on the training validation subset) and **② DeepStream deployed** (baseline → fine-tuned through ONNX → TensorRT → nvinfer on the KPI eval set). Tiles use **mAP first, then mAP@50** (mixed-case labels). A yellow **stale** banner appears when the deployed engine epoch ≠ the selected checkpoint; **Re-evaluate** clears stale ONNX/engine/metrics before re-deploy. Per-class AP table spans training epoch vs both DeepStream legs. Live
   `GT | baseline | fine-tuned` sample frames (click to zoom). A **real-time
   confidence slider** filters the drawn boxes client-side (boxes are fetched once
   per frame, then filtered instantly — no per-drag fetch), and under each frame a
   **box list** shows GT classes and fine-tuned detections with **class + confidence**
   plus a per-class count table. The displayed frames are chosen to **showcase where
   the fine-tuned model actually detects** (ranked by detection count). Step 2
   (Baseline) shows the **same frames** as Compare (same `samples.json` selection) and
   has the same slider + box list. Then the PDF report download.

   **Box/text colors** — below the slider, a palette + custom pickers set the box and
   label colors (applied to GT + baseline + fine-tuned, with an auto dark text halo) so
   detections stay visible on any background (e.g. green PCB); the choice is remembered.
   **"Apply colors + conf to report"** re-renders the PDF + frames at the **exact** UI
   colors and confidence, so the report matches what you see (the auto-report otherwise
   uses defaults — conf 0.2). **Confidence default is 0.2** (drag down to ~0.02 for
   low-score heads like RT-DETR). A **bottom progress bar** shows overall stage % +
   per-stage progress (epochs / images). Frames render the photo as a native `<img>`
   with a transparent box-overlay canvas (no `drawImage` — avoids blank/black frames),
   and Baseline/Compare use distinct element-id namespaces (no cross-panel collisions).
5. **Analyze** — error analysis for the deployed model (`error_analysis.py` via `/results/confusion`):
   a **model context banner** shows which run, checkpoint epoch (e.g. best epoch 21), DeepStream mAP,
   and training mAP at that epoch; the baseline/fine-tuned dropdown labels include run name + epoch.
   Predictions always come from the **currently selected run** in the sidebar (v2/v3/etc. each have
   their own deploy). A yellow warning appears if deployed epoch ≠ selected checkpoint (stale).
   Then: a **plain-language summary** (how-to-read + auto insights: overall P/R, biggest class confusion,
   weakest/most-missed class), a **confusion matrix** (rows = GT class, cols = predicted incl.
   *(missed)*/*(background)*; diagonal green, class-confusion red, misses amber), **per-class
   precision/recall/TP/FP/FN**, and a **hardest-images gallery** (the frames with the most FP+misses,
   GT | predicted — deliberately *different* from the Compare tab's detection-count "showcase" frames).
   Self-contained controls — baseline/fine-tuned toggle, confidence + IoU sliders, and a **Show labels**
   toggle (shared with the other tabs). Surfaces *which classes the model confuses* with no manual
   digging. Read-only over the run's predictions + GT.
6. **Playground** — `POST /infer`: drop/upload any image → it's preprocessed to the run's eval canvas
   and run through the **deployed DeepStream FP16 engine** (so it's exactly what's deployed) → the
   processed image (base64) + detections are returned and drawn (model toggle baseline/fine-tuned,
   confidence slider that filters client-side, shared Show-labels, and a **box-color picker** —
   defaults to red for contrast on green PCBs). Stateless (temp dir per request).
   Reuses `ds_image_eval` + `eval_engine._parse_preds`; **loads the already-built engine** by writing a
   temp config that points `model-engine-file` at it (the deploy config's engine name otherwise mismatches
   nvinfer's `<onnx>_b1_gpu0_fp16.engine`, which would trigger a ~2-min rebuild). Refuses while a heavy
   run is active (GPU) and 404s until the model is deployed. ~1 s/call after the first load.

**Runs & navigation:** the Step-1 **Runs — history** list shows every run (built-in
demos + your custom runs), **newest first with each run's date/time** (built-in demos
have no timestamp and fall to the bottom); the list is sorted by `created_at` both
server-side (`/presets`) and again client-side in `renderRuns()` so the order + dates are
guaranteed. Custom runs persist to `build/ui_runs/index.json` and survive restarts.
**The server hot-reloads that index on every request (mtime-gated `_load_custom()` in the
HTTP middleware), so a run registered on disk by an external harness** — e.g. a
CLI/container `run_ngc_finetune.sh` / `run_ngc_automl.sh` calling `ngc/ui_register.py` —
**appears in the list (with its date, correctly sorted) WITHOUT restarting the UI.** The
NGC finetune harness mirrors the live training log into `runs/<run>/train.log` every 20s so the
loss + per-epoch mAP curve populates live even for runs the UI didn't launch itself. **For an
AutoML run, `train.log` is written ONLY for the final full-length training** (the "3b Fine-tuning"
curve) — the trial runs are the hyperparameter *search*, so they appear on the **"3a AutoML sweep"**
leaderboard (`/results/automl`), never in `train.log`; that way 3b never shows trial data, and a
run is reachable while the sweep runs via the `is_automl` flag even though `train.log` is still
empty. AutoML registration preserves the exact `orig_dir`, and both NGC runners refuse to start
unless its baseline `engine_metrics.json` and `perf.json` are non-empty. Consequently the Baseline
step remains available throughout AutoML; an invalid legacy sweep is labeled
`blocked: baseline missing` instead of being presented as a valid training run. Each row has a **delete/clear**
action (`POST /run/delete`): for **custom** runs it removes the TRT engines,
checkpoints, Arrow data, ingested dataset, eval and report **and** drops the run from
history; for **built-in demos** it clears those artifacts but keeps the (re-runnable)
row. It only deletes **run-keyed paths** under the working root — never the user's
source dataset — and refuses while that run is in progress. The 6-step header is a
**clickable stepper** — any step with results for the selected run is reachable, so you
can move freely across Setup → Baseline → Fine-tune → Compare → Analyze → Playground without re-running. A
context bar shows which run you're viewing; **+ New run** returns to Setup. A header
**System & environment** icon (`/system`) opens a panel with the CPU/GPU/driver/CUDA/
DeepStream/TensorRT + package versions (the same provenance as report §7). A header
**Dataset QC** icon (`/results/dataset_qc`) opens a modal with the dataset health check
(class balance, box geometry, unlabeled images, train↔valid leakage, verdict) for the selected
run — so teams can vet a dataset in seconds before training.

A read-only **viewer** path works without a GPU (visualizes artifacts already on
disk): run uvicorn on the host with `DS_EVAL_ROOT="$PWD"` (see the comment in
`app/launch.sh`), then open a preset's *"view baseline" / "view comparison"* links.

**Skill vs. outputs (deployment hygiene):** the skill directory holds **source only**
and is never written to by a run. All generated data — TRT engines, eval sets,
predictions/metrics, fine-tune checkpoints, the run-history index, training/accuracy
charts, per-run logs, and reports — is written under the **working root**
(`models/`, `runs/` — including each run's `curve_history.json` continue-training curve,
`reports/`, `data/`, `build/ui_runs/`, `build/ui_charts/`),
resolved from `DS_EVAL_ROOT`. `launch.sh` sets `PYTHONDONTWRITEBYTECODE=1` so no
`__pycache__` lands in the skill, and `.gitignore` excludes the locally-built
`ds_image_eval` binary — so the shipped skill stays pristine.

**CLI artifact adoption:** built-in presets may declare `orig_dir_aliases` for baseline artifacts
produced by the canonical CLI/import-agent workflow under a model-derived directory. The UI adopts
an alias only when **both** hold:

1. `eval/engine_metrics.json` exists — so a partial deploy never appears ready; and
2. the directory's eval set targets this preset's label space — `ground_truth.json`'s
   `category_map` size must match the preset's `ft_nclasses`.

Check 2 exists because both bundled demos use the same base model (`PekingU/rtdetr_r50vd`), so a
model-derived directory like `models/rtdetr_r50vd` is **ambiguous between them** — whichever
workload ran last owns it. Without the check, opening the PCB preset after an aerial-sheep run
would report sheep's non-zero mAP as PCB's baseline, whose honest value is 0. The check fails open
when either side is unknown, so it only ever rejects a proven mismatch.

Once adopted, that directory is used consistently for baseline views, analysis, playground
inference, samples, and reports. UI runs still write to `orig_dir`, and clear/delete never removes
an alias. After launching against existing artifacts, verify `/presets` reports the expected
`has_baseline` and `has_finetuned` values; HTTP 200 alone is only a server-health check.

**Scope:** the UI auto-deploys **RT-DETR-family** models only (the bundled ONNX
exporter + custom parser handle the RT-DETR output format) against any dataset
`ingest_dataset.py` can ingest. A non-RT-DETR model is flagged *"needs deploy
recipe — run via the skill"* and is **not** auto-deployed — run those through this
skill directly, which authors the deploy recipe per model. Port **8078**.

| File | Purpose |
|------|---------|
| `app/launch.sh` / `app/setup.sh` | boot the server in-container (port 8078, `PYTHONDONTWRITEBYTECODE=1`) / install UI deps into `build/.venv_train` |
| `app/server.py` | FastAPI app: run registry + history (**hot-reloaded from `index.json` every request so externally-launched runs appear without a restart**), two-phase driver, polling, custom-run creation, artifact readers, `/chart` + `/frame_counts` + `/results/*` (incl. dataset distribution + **`/results/automl`** sweep leaderboard) + `/recolor_report` + `/run/delete` + `/system` endpoints; `/presets` exposes **`has_training`** + **`is_automl`** flags |
| `app/pipeline.py` | parametrized `run.sh` logic split into `baseline_plan` / `finetune_plan`; generates a training config for custom datasets; calls the existing scripts and never edits them |
| `app/presets.json` | the two built-in demos' paths, stage args, and report metadata |
| `app/static/` | NVIDIA-themed single-page UI (no build step): clickable stepper, zoom lightbox, live + report charts |
| `.gitignore` | excludes machine-local build artifacts (`__pycache__`, `ds_image_eval`) so deployment ships source only |
