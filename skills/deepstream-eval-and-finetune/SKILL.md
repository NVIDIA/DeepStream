---
name: deepstream-eval-and-finetune
description: >
  Evaluate and improve an object detector in NVIDIA DeepStream. Use for deployed mAP,
  FPS and latency measurement, TAO Skill Bank fine-tuning or AutoML, redeployment, and
  before/after reporting on HuggingFace, NGC, ONNX, or local models and KPI datasets.
  Object detection only; do not use for classification or non-vision models.
license: Apache-2.0
metadata:
  author: "Tushar Khinvasara <tkhinvasara@nvidia.com>"
  owner: "Tushar Khinvasara <tkhinvasara@nvidia.com>"
  service: "deepstream"
  version: "0.6.1"
  reviewed: "2026-08-04"
  team: deepstream-sdk
  tags:
    - deepstream
    - object-detection
    - map
    - fine-tuning
    - automl
    - tensorrt
  languages:
    - bash
    - python
    - javascript
  domain: computer-vision
---
<!--
Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License").
-->

# DeepStream Eval & Finetune

## Purpose

Deploy an object detector, measure its deployed mAP and performance on the user's KPI data,
fine-tune it, redeploy it, and produce a before/after report. Both accuracy legs must run through
the real DeepStream pipeline. Report measured results without a Go/No-Go verdict.

This skill orchestrates `deepstream-import-vision-model` for deployment and the TAO Skill Bank
router `tao-launch-workflow` (target `tao-finetune-huggingface-model`) for single-run fine-tuning.
HF AutoML uses this skill's NGC harness. Read the applicable direct reference before each stage.

## Prerequisites

- Linux, or Windows with Docker Desktop, WSL2, and an NVIDIA GPU; GPU compute is unsupported on
  macOS. Docker stages use `nvcr.io/nvidia/deepstream:9.1-triton-multiarch` with `--gpus all` and
  `--shm-size=16g`.
- The `tao-skill-bank` plugin from `https://github.com/NVIDIA-TAO/tao-skill-bank`. Confirm it with
  `scripts/check_tao_skills.sh`; install only when it reports `INSTALL-NEEDED`, then start a new
  session.
- `scripts/preflight.sh` must report `PASS` before deployment.

## Instructions

Follow [the end-to-end runbook](references/run-flow.md). The fixed sequence is:

1. Preflight and confirm TAO Skill Bank.
2. Deploy the original model through `deepstream-import-vision-model` using TensorRT FP16.
3. Evaluate the original model in DeepStream and preserve baseline mAP, per-class results,
   `eval/engine_metrics.json`, and `eval/perf.json`.
4. Run the selected fine-tune lane in the TAO NGC PyTorch container, never the DeepStream image.
5. Deploy and evaluate the fine-tuned model on the identical canonical images and canvas.
6. Generate performance, qualitative-overlay, and before/after PDF artifacts.

After intake, run unattended except for the `tao-launch-workflow` launch review in the single-run
lane. Run GPU stages sequentially and call `scripts/gpu_guard.sh` before each heavy stage.

### When no answer can arrive

The intake gates below assume an interactive caller. When there is none — an automated or
evaluation harness, a batch invocation, a request that states the task with no way to reply —
**do not stall.** Waiting for input that cannot arrive fails the task outright, which is a worse
outcome than proceeding on a stated assumption.

For each unanswered gate: announce it, apply the **recommended default**, record that value in the
run config exactly as if it had been chosen, and continue.

| Gate | Default when unanswerable |
|------|---------------------------|
| Workload | infer from the request when it names a model and dataset; otherwise **1. PCB Defects** |
| Epochs | **30** |
| Fine-tune lane | **single fine-tune**, unless the request asks for AutoML, HPO, or a sweep |
| AutoML scope (when AutoML) | **standard** |
| UI launch | **B. Skip the UI** — never start a server nobody asked for |

State the assumptions in one block before Stage 0, e.g. *"No interactive input available;
proceeding with PCB Defects, 30 epochs, single fine-tune, UI skipped."* This is a fallback for an
absent caller, not licence to skip a gate when someone can answer — with a live caller, ask.

## Inputs

Required inputs come from the selected preset or explicit user answers:

- `model_id`: HuggingFace/NGC model, ONNX path, or fine-tuned checkpoint; object detection only.
- Dataset source: HF dataset id, local COCO/Arrow, YOLO/Roboflow archive, or URL.
- `epochs`: explicit default or custom choice below.
- Fine-tune lane: single fine-tune or AutoML.
- UI choice: start or skip.

Optional inputs default to `split=validation`, `n_eval=200`, `canvas=[640,640]`, and
`precision=fp16`. Explicit user answers override presets; presets override these defaults. Ingest
non-loadable HF/script/zip datasets with `scripts/ingest_dataset.py` before evaluation.

## Workload choice — always offer three options

Before preflight or execution, present exactly these ordered choices. If the prompt already selects
one, confirm it instead of asking again.

### 1. PCB Defects (recommended)

```yaml
model_id: PekingU/rtdetr_r50vd
dataset_id: itsyoboieltr/pcb
split: test
n_eval: 200
canvas: [640, 640]
precision: fp16
epochs: 30
```

The six PCB classes are absent from COCO, so an honest deployed stock baseline near mAP 0 is
expected; fine-tuning creates the target-class capability. This workload takes longer.

### 2. Aerial Sheep (alternate validated demo)

```yaml
model_id: PekingU/rtdetr_r50vd
dataset_id: keremberke/aerial-sheep-object-detection
split: valid
n_eval: 200
canvas: [640, 640]
precision: fp16
epochs: 30
```

COCO already contains sheep, so the stock baseline is non-zero. Ingest this script-based HF dataset
first, then use the resulting local COCO input.

### 3. Custom model and dataset

Collect `model_id`, dataset source, split, eval-image count, canvas, and precision. Keep defaults of
200 images, 640x640, and FP16 unless overridden. Route non-RT-DETR deployment through
`deepstream-import-vision-model` as needed.

After workload and epoch selection, ask separately for single fine-tune or AutoML. Explain that a
single run is faster and fixed-config; AutoML searches several configurations and costs multiple
times the GPU time. Offer quick, standard, and thorough AutoML scope tiers.

## Epoch choice — always ask

Present exactly these ordered choices before execution:

1. **Default — 30 epochs (recommended).**
2. **Custom epoch count** — collect a positive integer.

Never infer epochs from a preset, prior run, or AutoML tier. Record the answer in the run config and
launch review. Use it as `num_train_epochs` for one fine-tune or `FINAL_EPOCHS` for AutoML's final
full-data training; trial epochs belong to the separate AutoML scope. A dry run shows the resolved
value without training.

## UI launch choice — always ask

Do not infer the answer from prior runs or an existing UI.

### A. Start the UI

Confirm port (default `8078`) and bind mode (default loopback-only `127.0.0.1`). Check listeners and
containers first. For another session's port/container, never stop, replace, or rebind it; use a
user-approved free port. Launch with `PORT=<port> bash <skill-root>/app/launch.sh`, then verify HTTP
health and semantic readiness from `/presets` (`has_baseline` and `has_finetuned`). Bind `0.0.0.0` only after
the user requests network access and acknowledges that the UI is unauthenticated and can start GPU
jobs or delete artifacts.

### B. Skip the UI

Do not launch Uvicorn or a UI container. Continue through the CLI/chat workflow.

For a dry run, present the choice but never launch or stop anything regardless of the answer.

## One path — always run, report per category

Deploy and score both original and fine-tuned models on every KPI class. Classes a stock model
cannot output score 0 and appear as "no object detected"; never hide or special-case them. See
[accuracy evaluation](references/accuracy-eval.md).

## Training invariant

Baseline is a hard precondition for single fine-tune and AutoML. For this exact dataset/run, require
non-empty original `eval/engine_metrics.json` and `eval/perf.json`; stale or mismatched artifacts
return execution to original deployment/evaluation. AutoML must never skip or bypass the baseline.
Preserve baseline artifacts for comparison. Pass `ORIG_DIR=models/<exact-baseline>` when using a
non-default baseline directory.

Create a new run key for every configuration; never overwrite another run. In the agent/CLI lane —
the one this skill drives — run training only in the TAO NGC PyTorch image: single fine-tune invokes
`tao-launch-workflow`, HF AutoML invokes `scripts/ngc/run_ngc_automl.sh`. See
[fine-tuning and reporting](references/finetune-and-report.md).

**Known divergence — the optional web UI.** `app/pipeline.py::_st_finetune` runs `train.py` with
`build/.venv_train` *inside the DeepStream container*, not in the NGC image. Its results are
therefore not guaranteed bit-equivalent to the NGC lane, because the CUDA/torch stack differs.
Treat UI-produced checkpoints as interactive/demo output; reproduce through the NGC lane before
reporting or publishing a result. Do not cite the invariant above as though the UI honoured it.

On Windows/WSL, mount the same project cache volume at `/work/data` in every DeepStream and training
container and set `DS_DATA_CACHE_ROOT=/work/data`.

## Windows and dataset persistence

The PowerShell and WSL launchers create a project-scoped Docker data volume to avoid repeated
NTFS-to-WSL image reads. Reuse an existing matching source cache by default; offer explicit refresh
and rebuild. `DS_DATA_CACHE=off` disables it, `DS_DATA_VOLUME` overrides its name, and
`DS_DATA_CACHE_REFRESH=1` refreshes it. Never delete or modify the source dataset.

Checkpoints, engines, metrics, logs, reports, and UI history remain under the Windows project root
and persist across UI/container restarts. Only cached dataset/Arrow content lives at `/work/data`.
See [Windows operation](references/windows.md).

## Output Format

Return paths and a concise before/after summary containing:

- Original and fine-tuned deployed mAP, per-class results, TensorRT FPS, and latency.
- Zero/unavailable stock classes labeled `no object detected`.
- The exact model, dataset/split, eval images, canvas, precision, epochs, lane, and environment.
- PDF report and overlay/contact-sheet paths; durable logs and run-registry paths.

Canonical artifacts are under `models/{name}/`, `models/{name}_ft/`, `reports/`, `logs/{run}/`, and
`runs/{run}/`. `runs/{run}/train.log` contains final training only; AutoML trial curves remain in the
sweep leaderboard.

## Available Scripts

| Script | Purpose | Key arguments or outputs |
|--------|---------|--------------------------|
| `scripts/preflight.sh` | Validate Docker, image, GPU, and environment | Must return `PASS` |
| `scripts/ingest_dataset.py` | Normalize non-standard datasets | HF/local/archive source to COCO/Arrow |
| `scripts/build_eval_set.py` | Build canonical evaluation images | Dataset, split, count, canvas |
| `scripts/eval_engine.py` | Run deployed detection evaluation | Engine/config to mAP and per-class JSON |
| `scripts/bench_trtexec.py` | Measure comparable engine performance | FPS and latency JSON |
| `scripts/ngc/run_ngc_automl.sh` | Run HF AutoML and final training | Baseline, scope, `FINAL_EPOCHS` |
| `scripts/compare_runs.py` | Compare both deployed legs | Before/after JSON |
| `scripts/make_report.py` | Produce final report | PDF output |

See [the complete script catalogue](references/scripts.md) for commands and all arguments. Execute
scripts through the shell or Python from the skill/working root; do not copy their logic inline.

## Reliability rules

- Keep `ds_image_eval` canvas identical to streammux width/height and reuse the same canonical images
  for both legs.
- Measure performance with `bench_trtexec.py`, not `eval_engine.py` pipeline FPS.
- Build `ds_image_eval` per machine/DeepStream version with `make -C scripts`.
- Always render 5-10 ground-truth/baseline/fine-tuned overlays.
- Match RT-DETR parsing to HF global top-k sigmoid post-processing; see the accuracy reference.
- Validate UI artifact readiness through `/presets`, not HTTP health alone.
- Never expose credentials, overwrite durable logs, disturb another UI session, or run concurrent
  GPU stages.

## Limitations

- Object detection only; reject classification, NLP, and incompatible vision tasks.
- Every compute stage requires an NVIDIA GPU. macOS supports only host helpers/read-only viewing.
- UI network exposure has no authentication and requires explicit acknowledgement.
- The workflow reports evidence and improvement; it does not decide whether a model should ship.

## Troubleshooting

| Error | Cause | Resolution |
|-------|-------|------------|
| Preflight fails | Missing image, GPU access, or environment | Apply the reported fix and rerun preflight |
| Training gate refuses | Baseline artifacts missing/stale/mismatched | Rerun original deploy and evaluation |
| UI port occupied | Another process/session owns it | Preserve it and obtain approval for a free port |
| Windows training is slow | Dataset is read across NTFS/WSL | Reuse or explicitly refresh the Linux data cache |
| Fine-tune skill missing | TAO Skill Bank is not loaded | Install from the canonical repository and start a new session |

## References

| Document | Read when |
|----------|-----------|
| [Run flow](references/run-flow.md) | Executing the full stage sequence and launch gate |
| [Accuracy evaluation](references/accuracy-eval.md) | Deploying, scoring, parser rules, and container commands |
| [Fine-tuning and reporting](references/finetune-and-report.md) | Running single/AutoML training and building reports |
| [Web UI](references/ui.md) | Launching, exposing, inspecting, or operating the UI |
| [Script catalogue](references/scripts.md) | Resolving script commands and arguments |
| [Windows operation](references/windows.md) | Running through Windows/WSL and managing the data cache |
