<!--
Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License").
-->

# Fine-tune (DEFT) + before/after report

After the original model is deployed and measured (`references/accuracy-eval.md`),
fine-tune it on the KPI dataset, re-deploy, and report the deployed before/after.
**There is no Go/No-Go gate** — the deliverable is the measured improvement.

## Source of the fine-tune + automl lanes (tao-skill-bank)

The fine-tune and automl skills — and every other tao skill — are **not in this repo**;
they ship in the **`tao-skill-bank`** plugin (**https://github.com/NVIDIA-TAO/tao-skill-bank**),
installed by the skill's `install.sh` via `claude plugin install`, so enabled plugins load at
session start. Once installed, each bank skill is invocable by its own name:

| Bank skill | Role |
|------------|------|
| `tao-launch-workflow` | launch intake / router — the Stage-3 entry point |
| `tao-finetune-huggingface-model` | HuggingFace fine-tune (DEFT) — the launch target |
| `tao-run-automl` | hyperparameter search for TAO-Toolkit networks (not HF — HF AutoML uses `scripts/ngc/run_ngc_automl.sh`) |
| `tao-list-capabilities`, `tao-train-*`, … | discover / use any other tao skill |

**AutoML is optional** — for the HF RT-DETR path, a hyperparameter sweep runs via the skill's own
`scripts/ngc/run_ngc_automl.sh` harness (the bank's `tao-run-automl` targets TAO-Toolkit networks,
not HF models). The default single-shot path is the finetune skill.

## DEFT lane = `tao-launch-workflow` → `tao-finetune-huggingface-model`

Invoke `tao-launch-workflow` (active agent skill runtime) with `platform=local-docker`, targeting the
HuggingFace object-detection finetune (`tao-finetune-huggingface-model`) on the KPI dataset.
Export `TAO_SKILL_BANK_PATH=~/.claude/plugins/cache/tao-skill-bank/tao-skills/<version>` first
so the launcher's helper scripts find the packaged manifests. Inputs:
- `task: object-detection`, the KPI dataset (HF `dataset_id`, or local COCO via
  `local_dataset_format: coco` from `ingest_dataset.py`), `label_names` from the
  dataset's classes.
- Produces `checkpoints/final`.

**Container: ALWAYS the tao NGC PyTorch image** that `tao-launch-workflow` resolves from
packaged metadata (e.g. `nvcr.io/nvidia/pytorch:25.03-py3`). **Never fine-tune in the
DeepStream container.** The tao skill mounts `TAO_SKILL_BANK_PATH` + dataset + output and runs
its own generated pipeline, so **no manual staging is needed**. The DeepStream image is used
only for the deploy + DeepStream-eval stages.

On Windows/WSL, also mount the project-scoped volume printed by the UI launcher:
`-v "$DS_DATA_VOLUME:/work/data" -e DS_DATA_CACHE_ROOT=/work/data`. This keeps the local COCO images
and generated Arrow data on Linux storage for both single fine-tuning and AutoML; result/checkpoint
paths remain on the normal `/work` Windows bind mount.

**Before either training lane, require the deployed baseline.** Verify that the original model's
Stage-2 `eval/engine_metrics.json` and `eval/perf.json` are non-empty and belong to this exact dataset/run.
If not, return to Stages 1–2. AutoML is a Stage-3 alternative, not an alternate end-to-end path,
and must never start before the baseline. Both bundled NGC runners fail before creating logs,
installing dependencies, or training when either artifact is missing or empty. They resolve the
baseline as `models/<run>_orig` or reuse the run's existing UI-registry `orig_dir`; set
`ORIG_DIR=models/<exact-baseline>` for a new run with a non-default path.

**Ask for the epoch count before launching:** offer **30 epochs (default)** or a
**user-defined positive integer**. Use the resolved value as `num_train_epochs` for a single
fine-tune or `FINAL_EPOCHS` for AutoML's final full-dataset run. AutoML's shorter trial epochs are
selected separately as part of its scope tier.

**Then ask for the training lane (user-input prompt) — single fine-tune vs AutoML:**
default = single fixed-config `tao-finetune-huggingface-model`; option = an **AutoML sweep** via
the skill's `scripts/ngc/run_ngc_automl.sh` harness — many trials over LR/warmup/scheduler/
weight-decay that picks the best config (much longer; use when you want tuning rather than a
known-good recipe).

**Known compat (apply in the container):**
- `transformers==4.49`, `numpy<2`, `albumentations<2.0`.
- albumentations 1.4.x has no `filter_invalid_bboxes` → **pre-filter degenerate
  boxes** in the transform (drop w/h≤0, x≥W, y≥H).
- `--shm-size=16g`; venvs via `virtualenv`.
- **`load_best_model_at_end: false`** (or eval mAP on the last checkpoint) — for
  detectors, eval_loss can overfit after a few epochs and pick a near-untrained
  "best", so select the most-trained checkpoint.
- **Model choice matters for speed/quality:** DETR / YOLOS are DETR-family =
  slow-converging (YOLOS is *not* a real YOLO). **RT-DETR converges far faster** —
  prefer it for a strong result on a budget. Few distinct classes + ~30 epochs is
  the fast, strong recipe.

If Docker/GPU is unavailable, **don't fail mid-loop** — emit a ready-to-run
handoff (config + the `tao-launch-workflow` / `tao-finetune-huggingface-model` command)
and stop; the user runs the fine-tune on a GPU host, then re-enters at the
deploy-fine-tuned stage.

## Re-deploy + before/after

1. Re-deploy `checkpoints/final` via `deepstream-import-vision-model` (FP16) →
   fine-tuned engine + nvinfer config + parser.
2. `build_eval_set.py` (reuse the same KPI images / canvas) → `eval_engine.py`
   (drives `ds_image_eval`) → fine-tuned `engine_metrics.json` (+ `perf.json`).
3. `compare_runs.py --original-metrics ... --finetuned-metrics ...` → merge into a
   deployed before/after JSON (accuracy + per-class AP + per-class detection counts + perf).
4. `sample_overlays.py --orig-eval-set ... --orig-predictions ... --ft-eval-set ...
   --ft-predictions ... --out-dir reports/samples --n 6` → 5–10 rendered before/after
   frames (GT | baseline | fine-tuned), boxes drawn above a confidence threshold.
5. `collect_env.py --out build/env_info.json --image <DS image>` → capture system/tool
   provenance (CPU, GPU, driver/CUDA, DeepStream, TensorRT, package versions).
6. `make_report.py --comparison <json> --train-log <train.log> --samples-dir reports/samples
   --env build/env_info.json --model-* --dataset-*` → the narrative PDF: definitions, where
   DeepStream is used, the step table, the **original-deployed vs fine-tuned-deployed** results,
   the qualitative samples (§5e), and **§7 system & environment** (reproducibility). No verdict.
   (If `--env` is omitted, make_report collects it best-effort at render time.)

## What the report shows (recap)

- **Original (deployed)** mAP+perf — the default-state baseline; may be low or 0
  on KPI classes the model can't output yet (reported per-class as "no object detected").
- **Fine-tuned (deployed)** mAP+perf — the improved model, same DeepStream path.
- **Improvement** = fine-tuned − original, plus per-class AP and FPS/detection-rate.
- **Qualitative samples (§5e)** = 5–10 rendered frames, ground truth vs baseline vs
  fine-tuned detections — the visual proof that complements the mAP delta.

Optional fidelity check (not the headline): `eval_hf.py` gives the fine-tuned
model's PyTorch mAP on the same canonical images — comparing it to the deployed
mAP confirms the engine is faithful (FP16 ≈ 0 loss).
