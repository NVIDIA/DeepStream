<!--
Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License").
-->

# End-to-end run flow (runbook)

How a full run proceeds once a user prompt launches `deepstream-eval-and-finetune`.
This is the operational companion to the `SKILL.md` "End-to-end run flow" diagram —
the **why** and the **exact inputs** behind each step, plus the point where a human is
required.

## Trigger

The **agent skill runtime loads this skill** when a user prompt matches its description
(e.g. "deploy my HF detector to DeepStream, measure mAP, fine-tune it on my dataset, and
show me the before/after"). The user brings a **model** (HF/NGC/ONNX object detector) and
a **KPI dataset**; the skill drives the rest.

## Intake — choose the workload

> **No interactive caller?** If the gates in this section cannot be answered — an automated or
> evaluation harness, a batch run, a request that states the task with no way to reply — announce
> each gate, apply its recommended default (PCB Defects · 30 epochs · single fine-tune · skip the
> UI), record the value in the run config as though it had been chosen, and continue. Stalling on
> input that cannot arrive fails the task, which is worse than proceeding on a stated assumption.
> With a live caller, always ask.

Before preflight or execution, present exactly three ordered choices:

1. **PCB Defects (recommended):** `PekingU/rtdetr_r50vd` + `itsyoboieltr/pcb`,
   test split, 200 images, 640×640 FP16. Its six target categories are absent from COCO,
   so the deployed stock baseline is expected to be zero.
2. **Aerial Sheep (alternate validated demo):** `PekingU/rtdetr_r50vd` +
   `keremberke/aerial-sheep-object-detection`, valid split, 200 images, 640×640 FP16.
   COCO already has sheep, so the deployed stock baseline is non-zero.
3. **Custom model and dataset:** collect the object detector, dataset source, split, eval count,
   canvas, and precision; default to 200 images, 640×640, and FP16.

Do not collapse the intake to only the recommended demo. If the user already selected one choice,
confirm it and continue.

Then always ask the **epoch choice**:

1. **Default — 30 epochs (recommended).**
2. **Custom epoch count:** collect a positive integer.

Do not derive this answer silently from the workload or AutoML tier. The resolved value drives
`num_train_epochs` for a single fine-tune and `FINAL_EPOCHS` for AutoML's final full-data training;
AutoML trial epochs are a separate scope/budget setting. Ask single fine-tune versus AutoML
separately before Stage 3.

Then always ask the **UI launch choice**:

- **Start UI:** confirm the port (default 8078) and bind mode (default `127.0.0.1`), check for
  conflicts without disturbing another session, launch it, and verify both `/` and `/presets`.
  Network exposure on `0.0.0.0` requires an explicit request plus acknowledgement of the
  unauthenticated job/delete controls.
- **Skip UI:** launch no UI/Uvicorn container and continue with the canonical CLI/chat workflow.

For a dry run, show this choice but perform neither launch nor stop actions.

On Windows/WSL, dataset caching is automatic. When the selected local source already has a staged
copy, offer **reuse cache (default)** or **refresh source cache and rebuild this run**. Refresh
invalidates derived baseline/fine-tuned artifacts so results can never mix dataset versions.

## Sequence

| Stage | Command / skill | Output |
|-------|-----------------|--------|
| **0 Preflight** | `bash scripts/preflight.sh` | PASS on docker + DeepStream image + GPU + venv; else FAIL with the exact fix |
| **0b Confirm plugin** | `bash scripts/check_tao_skills.sh` | tao-skill-bank plugin resolves (installed by `install.sh`) |
| **1 Deploy original** | Skill: `deepstream-import-vision-model` (FP16) | TRT engine + nvinfer config + parser + `labels.txt` |
| **2 Eval original** | `make -C scripts` · `ingest_dataset.py` · `build_eval_set.py` · `eval_engine.py` | baseline per-class mAP + `perf.json` (FPS/latency) |
| **3 Fine-tune (DEFT)** | Skill: `tao-launch-workflow` → `tao-finetune-huggingface-model` | `checkpoints/final` |
| **4 Deploy fine-tuned** | Skill: `deepstream-import-vision-model` on `checkpoints/final` | fine-tuned engine + config |
| **5 Eval fine-tuned** | `build_eval_set.py` (same images) · `eval_engine.py` | fine-tuned mAP + perf |
| **6 Report** | `compare_runs.py` → `sample_overlays.py` → `collect_env.py` → `make_report.py` | before/after deployed PDF |

**Mandatory ordering invariant:** Stage 3 has two possible lanes, but neither lane may begin until
Stage 2 has produced the original deployed model's non-empty `eval/engine_metrics.json` and
`eval/perf.json` for the selected workload. Missing, stale, or mismatched artifacts return execution to
Stages 1–2. AutoML changes only Stage 3; it never bypasses the baseline.

## Stage 0b — confirm the tao-skill-bank plugin

The fine-tune / automl lanes are **not** in this repo; they come from the **tao-skill-bank**
plugin (https://github.com/NVIDIA-TAO/tao-skill-bank).

**Normally this is already done at install time.** The skill's `install.sh` installs
`tao-skills@tao-skill-bank` through the `claude` CLI, and **enabled plugins load at session
start** — the tao skills are invocable by name from the first prompt, with no mid-session install
and no reload (the manual-install fallback below is the one exception — it needs a new session).

1. `check_tao_skills.sh` (detector, no side effects) confirms three representative bank
   skills resolve: `tao-launch-workflow`, `tao-finetune-huggingface-model`,
   `tao-run-automl`. `PASS` → continue.
2. Only on `INSTALL-NEEDED` (the skill was copied in without the plugin), run
   `install_tao_skills.sh` (in the **host** Claude session, where the `claude` CLI lives):
   ```bash
   claude plugin marketplace add https://github.com/NVIDIA-TAO/tao-skill-bank --scope project
   claude plugin install tao-skills@tao-skill-bank --scope project
   ```
   The CLI persists the marketplace + plugin in its own configuration. **Start a new session**
   so the enabled plugin loads, then re-run the detector.
3. If the `claude` CLI is unavailable (e.g. attempted inside the DeepStream container), the
   installer degrades to printing the manual `/plugin` slash commands.

## Stage 3 — fine-tune via tao-launch-workflow

`tao-launch-workflow` is the bank's model-agnostic **launch intake**. Before invoking it,
export the bank path so its helper scripts find the packaged manifests:

```bash
export TAO_SKILL_BANK_PATH=~/.claude/plugins/cache/tao-skill-bank/tao-skills/<version>
```

**Container: ALWAYS the tao NGC PyTorch image** `tao-launch-workflow` resolves (e.g.
`nvcr.io/nvidia/pytorch:25.03-py3`) — **never the DeepStream image for fine-tuning**. The tao
skill mounts everything + generates its own pipeline → **no manual staging**.

**Before launching, confirm the resolved epoch count and ask the user for the lane:** a single fixed-config
fine-tune (`tao-finetune-huggingface-model` via `tao-launch-workflow`, default), or an **AutoML
sweep** via the skill's own HPO harness `scripts/ngc/run_ngc_automl.sh` (many trials over
LR/warmup/scheduler/weight-decay, best config; much longer). Offer AutoML whenever the user wants
tuning over a known-good recipe. *(`tao-run-automl` from the bank targets TAO-Toolkit networks,
not HF models, so HF RT-DETR AutoML uses this skill's harness.)*

Before either invocation, check the Stage-2 baseline artifacts for this exact run and dataset.
Do not invoke `tao-launch-workflow` or `run_ngc_automl.sh` when
`eval/engine_metrics.json` or `eval/perf.json` is missing, empty, stale, or from another workload.

Then invoke `tao-launch-workflow` targeting the chosen lane, pre-answering the launch intake so
its **Non-Negotiable Launch Gate** clears in one pass:

- **platform** = `local-docker` (GPU host)
- **dataset** = the KPI dataset spec (the ingested COCO / HF Arrow from Stage 2), `task: object-detection`, `label_names` from the dataset classes
- **container image** = the tao NGC PyTorch image (never DeepStream)
- **compute shape / monitoring** = defaults (monitor in chat)

**Human touchpoint — launch review.** The gate shows a final review (image, platform,
datasets, compute shape, expected runtime, config changes) and waits for the user to
confirm before it launches side-effecting training. After confirmation it trains and
produces `checkpoints/final`. *(For AutoML instead of a single fine-tune, run the skill's
`scripts/ngc/run_ngc_automl.sh` HPO harness — see SKILL.md Stage 3.)*

If Docker/GPU is unavailable, don't fail mid-loop — emit a ready-to-run handoff (config +
the `tao-launch-workflow` / `tao-finetune-huggingface-model` command) and stop; the user
runs it on a GPU host and re-enters at Stage 4.

## The only mid-run human touchpoint

**Launch review** — the `tao-launch-workflow` confirmation before training (Stage 3).

Everything else — preflight, deploy, eval, redeploy, re-eval, report — runs unattended. The
tao-skill-bank plugin is installed **with** the eval skill (`install.sh`) and loads at
**session start**, so there's no mid-session install and no reload (except the manual-install
fallback, which needs one new session).
