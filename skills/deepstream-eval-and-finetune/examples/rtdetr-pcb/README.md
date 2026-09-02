<!--
Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License").
-->

# Demo (alternate): RT-DETR → PCB defects → fine-tune → deploy → eval

The second validated worked example for `deepstream-eval-and-finetune`. It runs the same
*measure → improve* loop as [rtdetr-aerial-sheep](../rtdetr-aerial-sheep/README.md), but tells the
opposite story: a **capability the stock model does not have at all**.

Why this example exists: **none of the six PCB-defect categories appear in COCO-80**, so stock
RT-DETR cannot emit them — its predictions remap to none of the targets and it scores an
**honest mAP 0**. That is not a broken run; it is the skill's "report per category, never hide a
zero" rule doing its job. Fine-tuning is what *creates* the target-class capability, so the
before/after shows capability being added rather than accuracy being sharpened.

Use aerial-sheep when you want a genuine non-zero baseline and a shorter run; use this one when you
want the zero-baseline / new-capability case. This example can take longer because it trains six
target classes instead of one.

| Item | Value |
|------|-------|
| Model | `PekingU/rtdetr_r50vd` — RT-DETR R50vd, Apache-2.0, COCO-trained |
| Dataset | `itsyoboieltr/pcb` — PKU/HRIPCB, 6 defect classes |
| Classes | `mouse_bite`, `spur`, `missing_hole`, `short`, `open_circuit`, `spurious_copper` |
| Size | train 6,370; eval on 200 **test** images |
| Canvas | 640×640, **stretch** resize (RT-DETR square-resize processor) |
| Precision | FP16 |
| Epochs | 30 |

## Expected result (deployed, same 200 test images)

| Metric | Baseline (stock RT-DETR) | Fine-tuned (30 ep) |
|--------|--------------------------|--------------------|
| **mAP@50** | **0** *(by construction — no PCB class in COCO-80)* | measured by the run |

The baseline zero is structural and reproducible, not a measurement to be tuned away. The
fine-tuned figures are whatever your run reports; this table is deliberately not pre-filled with
numbers from a machine you did not run on. `reports/rtdetr_pcb_report.pdf` carries the measured
before/after for your hardware.

> **Heads-up:** `itsyoboieltr/pcb` ships a custom HF schema, not COCO. It is converted once with
> `convert_pcb.py` (step 0). **Both** eval legs are built against the 6-class *target* label space
> so ground truth is preserved across the comparison; only `eval_engine --labels` differs
> (stock COCO-80 for the baseline, the 6 fine-tuned labels after).

All stages run in the DeepStream container (`nvcr.io/nvidia/deepstream:9.1-triton-multiarch`,
`--gpus all --shm-size=16g`); GPU stages run **sequentially** (`gpu_guard.sh` before each). Paths
below use `SK=.claude/skills/deepstream-eval-and-finetune` and the shared venv `build/.venv_train`.

## Quick start (one command)

From the working root, inside the container:

```bash
docker run --rm --gpus all --shm-size=16g -v "$PWD":/work -w /work \
  --entrypoint bash nvcr.io/nvidia/deepstream:9.1-triton-multiarch \
  .claude/skills/deepstream-eval-and-finetune/examples/rtdetr-pcb/run.sh
# -> reports/rtdetr_pcb_report.pdf
```

`run.sh` is idempotent — every stage guards on its own output, so a re-run resumes rather than
redoing finished work.

## Stages

| # | Stage | Key output |
|---|-------|------------|
| 0 | Convert `itsyoboieltr/pcb` → COCO + `labels.txt` (6 classes) — `convert_pcb.py` | `data/pcb_defect/annotations_{train,valid,test}.json` |
| 1–2 | Deploy **stock** RT-DETR (80 COCO labels) + DeepStream eval | `models/rtdetr_pcb_orig/eval/engine_metrics.json` (mAP 0) |
| 3 | Fine-tune RT-DETR on PCB defects, 30 epochs, 6 classes | `runs/rtdetr_pcb/checkpoints/final` |
| 4 | Deploy **fine-tuned** + DeepStream eval + trtexec perf | `models/rtdetr_pcb_ft/eval/engine_metrics.json` |
| 5 | Compare → samples → before/after report | `reports/rtdetr_pcb_report.pdf` |

The RT-DETR assets (ONNX export, custom bbox parser, nvinfer config template, `train.py`) are
shared with the aerial-sheep example and read from `../rtdetr-aerial-sheep/assets/`; only
`convert_pcb.py` and `config_pcb.yaml` are PCB-specific.

## Files

| File | Purpose |
|------|---------|
| `run.sh` | the full end-to-end loop (stages 0–5) |
| `convert_pcb.py` | `itsyoboieltr/pcb` custom HF schema → COCO + `labels.txt` |
| `config_pcb.yaml` | fine-tune config consumed by the shared `train.py` |
| `stratify_test.py` | build a class-balanced test subset |
| `diag.py` | inspect converted annotations when a stage looks wrong |

## Web UI

The bundled UI ships a matching `pcb` preset (`app/presets.json`), so this workload can also be
driven from the browser — same scripts, same artifacts. See [../../references/ui.md](../../references/ui.md).

Note that both built-in demos use the **same base model**, so a model-derived baseline directory is
ambiguous between them. The UI adopts a baseline produced outside its own `orig_dir` only when that
directory's eval set targets this workload's label space, which keeps an aerial-sheep baseline from
being reported as this demo's. See the *CLI artifact adoption* section of `references/ui.md`.

## Notes / gotchas

- **mAP 0 at baseline is correct.** If you see a non-zero baseline here, the eval set was probably
  built against the wrong label space — both legs must use the 6-class target labels.
- **`--shm-size=16g` is required** — the default 64 MB `/dev/shm` deadlocks DataLoader workers.
- **Run GPU stages sequentially.** Never run this concurrently with another `--gpus all` job.
- The per-epoch eval subset is 120 validation images (built in stage 3) to keep epochs fast; the
  reported comparison always uses the full 200-image test eval set.
