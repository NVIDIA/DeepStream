<!--
Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License").
-->

# Scripts reference

Every script the skill drives, and what it does. All live under the skill directory; all run
outputs go to the working root (`models/`, `runs/`, `reports/`, `data/`, `logs/`, `build/`).

## Install / bootstrap / environment

| Script | Purpose |
|--------|---------|
| `install.sh` (skill root) | **Install into a project** — copies the skill into `<target>/.claude/skills/` and installs the tao-skill-bank plugin via the `claude` CLI. `bash install.sh --target <proj>` |
| `setup.sh` (skill root) | **Fresh-machine bootstrap.** Create `build/.venv_train`, install `scripts/requirements.txt`, build `ds_image_eval`. Idempotent |
| `dataset_cache.py` | Stage a Windows/WSL local dataset once into the persistent Linux-backed `/work/data/.sources/<hash>` cache. Parallel copy, disk-space check, atomic promotion, manifest, safe explicit refresh |
| `scripts/requirements.txt` | Pinned Python deps (torch, transformers, datasets, torchmetrics, pycocotools, reportlab, matplotlib, …). Optional web-UI deps live in `app/setup.sh` |
| `preflight.sh` | **Run first (after setup).** Verify docker + DeepStream image + GPU + venv packages before any run |
| `check_tao_skills.sh` | **Run after preflight.** Detector (never installs) — verifies the tao-skill-bank plugin is installed (checks `tao-launch-workflow` + `tao-finetune-huggingface-model` + `tao-run-automl` resolve); on `INSTALL-NEEDED` points to `install_tao_skills.sh` (+ `/plugin` manual fallback) for https://github.com/NVIDIA-TAO/tao-skill-bank |
| `install_tao_skills.sh` | **Manual fallback** (normally unneeded — `install.sh` already installs the plugin). `claude plugin marketplace add` + `install tao-skills@tao-skill-bank --scope project`, then re-runs the detector; start a new session so the enabled plugin loads. Degrades to `/plugin` instructions if the `claude` CLI is absent. Idempotent |
| `gpu_guard.sh` | Wait for a free GPU before a heavy stage (sequential-run guard) |

## Fine-tune / AutoML (tao NGC PyTorch container)

| Script | Purpose |
|--------|---------|
| `ngc/run_ngc_finetune.sh` | **Reproducible Stage-3 finetune in the tao NGC container.** Hard-requires the run's non-empty baseline `engine_metrics.json` + `perf.json` before any training side effect. Installs deps, runs `train_ngc.py` with **per-epoch eval always on** (curves populate), writes **durable timestamped host logs** to `logs/<run>/<ts>/`, and refreshes `runs/<run>/train.log` **live every 20s**. Auto-registers the run in the UI (`ui_register.py`) and self-snapshots to survive mid-run edits. Set `ORIG_DIR` for a non-default baseline path. Args: `<run> <config.yaml> <train_coco> <eval_coco> <images_dir> [MAX_STEPS]` |
| `ngc/run_ngc_automl.sh` | **HF AutoML / HPO sweep in the tao NGC container** (`tao-run-automl` targets TAO-Toolkit networks, not HF models). Hard-requires the run's non-empty baseline `engine_metrics.json` + `perf.json` before any AutoML side effect. Runs N trials on a data subset over LR/warmup/scheduler/weight-decay, scores each by best per-epoch `eval_map`, picks the winner, trains a final model on the full data. Trial curves stay on the "3a AutoML sweep" leaderboard (NOT in `train.log`, which is reserved for the final training = "3b Fine-tuning"). Set `ORIG_DIR` for a non-default baseline path. Args: `<run> <base_config> <full_train_coco> <subset_train_coco> <eval_coco> <images_dir> [N_TRIALS] [TRIAL_EPOCHS] [FINAL_EPOCHS]` |
| `ngc/train_ngc.py` | Self-contained RT-DETR trainer (plain torch Dataset over local COCO + images; **no HF `datasets` dep** so it installs cleanly on NGC). Per-epoch eval logs `eval_loss`/`eval_map`/per-class AP |
| `ngc/ui_register.py` | Upsert a run into the UI registry (`build/ui_runs/index.json`, visible-by-default) so a row appears the moment training starts; preserves existing artifact paths and can enforce the two-file baseline gate via `--require-baseline`; called by both NGC runners |

## Data

| Script | Purpose |
|--------|---------|
| `ingest_dataset.py` | YOLO-zip **or** Roboflow/script-based COCO-zip → COCO + labels.txt (auto-detects `_annotations.coco.json`) |
| `coco_to_hf.py` | COCO → HF Arrow (for the tao-skill-bank finetune skill) |
| `build_eval_set.py` | Canonical letterboxed eval set + GT; records label `overlap` (informational; never routes/blocks) |
| `dataset_qc.py` | **Dataset health/QC over the ingested COCO** (annotations-only → fast + SMB-safe): class balance + imbalance ratio, box geometry (tiny/huge/out-of-bounds/degenerate/extreme-AR), unlabeled images, train↔valid leakage, plain-language flags + green/amber/red verdict. Served by `/results/dataset_qc` |

## Deploy-eval + report

| Script | Purpose |
|--------|---------|
| `ds_image_eval.c` + `Makefile` | C DeepStream app: direct JPEG-decode → nvinfer → detection probe |
| `eval_engine.py` | Drive `ds_image_eval`; score deployed mAP; collect perf |
| `bench_trtexec.py` | Authoritative engine FPS + GPU latency via `trtexec` (`--loadEngine`, or `--onnx` — nvinfer-serialized engines don't reload in trtexec, so bench from ONNX) |
| `compute_map.py` | Shared torchmetrics core |
| `eval_hf.py` | Optional PyTorch reference mAP (fidelity check) |
| `checkpoint_to_evaluate.py` | Resolve which checkpoint `checkpoints/final` represents (best-by-metric vs latest) + attach PyTorch training mAP at that epoch for the Compare tab / evaluate button |
| `record_deployed_checkpoint.py` | Record which epoch's checkpoint was deployed (+ stored-model paths) → `deployed_checkpoint.json` + a `final_epochN` tag. Continue-aware: offsets the epoch by `curve_history.prev_epochs` for the cumulative deployed epoch |
| `compare_runs.py` | Merge original-deployed vs fine-tuned-deployed (accuracy + perf) → report JSON. With `--train-coco` + `--eval-gt` also embeds **`dataset_distribution`** (per-class train + eval instance/image counts) |
| `curve_history.py` | **Single source of truth for the fine-tuning loss + per-epoch mAP curves across continue-training segments** (`load`/`merged`/`parse_log`/`seed`/`latest_per_class`); carries `eval_loss` + `eval_map`/`eval_map_50` (+ latest-epoch per-class AP). A `commit` CLI folds a finished segment into `runs/<name>/curve_history.json` (absolute epochs) the moment training ends, so a re-run that overwrites `train.log` can't drop a segment |
| `error_analysis.py` | **Confusion matrix + per-class precision/recall + hardest-images ranking** from deployed predictions vs GT (no re-inference). Class-agnostic greedy IoU match. Served by `/results/confusion` |
| `sample_overlays.py` | Render 5–10 before/after frames (GT \| baseline \| fine-tuned) → `reports/samples/`. **Auto-adapts the draw threshold** to the model's score range (RT-DETR sigmoid head maxes ~0.1–0.2 even at mAP@50 ~0.8) so boxes still render |
| `collect_env.py` | Capture system + tool provenance (CPU, GPU, driver/CUDA, DeepStream, TensorRT, package versions) → `env_info.json` for report §7 |
| `make_report.py` | Before/after deployed PDF report (sequence + roles + improvement + per-class table + qualitative samples + §2a dataset classes & distribution + §7 system & environment) |
