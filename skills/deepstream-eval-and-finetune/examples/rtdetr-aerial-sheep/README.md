<!--
Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License").
-->

# Demo (default): RT-DETR → aerial-sheep → fine-tune → deploy → eval

The **recommended default** worked example for `deepstream-eval-and-finetune`. It
exercises every stage and tells the full *measure → improve* story with a
**real, non-zero baseline** — no contrived under-training.

Why this is the default: COCO already has a `sheep` class, so **stock RT-DETR
produces a genuine baseline** on this data (it really does try to find sheep).
But the dataset is aerial/drone imagery with tiny, dense sheep (~31 per image),
a large domain shift from COCO's ground-level sheep — so there is **real
fine-tuning headroom**. That combination (honest baseline + big legitimate gain)
is exactly what the skill is meant to demonstrate.

| Item | Value |
|------|-------|
| Model | `PekingU/rtdetr_r50vd` — RT-DETR R50vd, Apache-2.0, COCO-trained |
| Dataset | `keremberke/aerial-sheep-object-detection` — Roboflow, 1 class (`sheep`) |
| Size | 600×600; train 3,609 / valid 350 / test 174; ~31 boxes/image |
| Canvas | 640×640, **stretch** resize (RT-DETR square-resize processor) |
| Precision | FP16 |

## Validated result (deployed, same 200 valid images)

| Metric | Baseline (stock RT-DETR) | Fine-tuned (12 ep) | Δ |
|--------|--------------------------|--------------------|---|
| **mAP@50** | **0.1875** | **0.8735** | **+0.686 (4.7×)** |
| mAP@[.5:.95] | 0.0704 | 0.4884 | +0.418 |
| mAP@75 | 0.0377 | 0.4863 | +0.449 |
| Engine FPS (trtexec, b1) | ~605 | ~605 | same arch |

> **Heads-up:** aerial-sheep is a **script-based** HF dataset, which `datasets`
> >=3 rejects ("Dataset scripts are no longer supported"). It bundles its data as
> Roboflow COCO zips (`data/{train,valid,test}.zip`), so it is ingested to COCO once
> (step 1) and the loop is driven with `--local-coco`, never `--dataset-id`.

All stages run in the DeepStream container
(`nvcr.io/nvidia/deepstream:9.1-triton-multiarch`, `--gpus all --shm-size=16g`);
GPU stages run **sequentially** (call `gpu_guard.sh` before each). Paths below use
`SK=.claude/skills/deepstream-eval-and-finetune` and the shared venv
`build/.venv_train`.

## Quick start (one command)

`run.sh` drives the whole loop (ingest → deploy stock → eval → fine-tune → deploy
fine-tuned → eval → report) using the bundled `assets/` (export script, RT-DETR
parser, nvinfer config template, train.py). Run it from the working root inside
the DeepStream container, after `preflight.sh` passes:

```bash
docker run --rm --gpus all --shm-size=16g -v "$PWD":/work -w /work \
  --entrypoint bash nvcr.io/nvidia/deepstream:9.1-triton-multiarch \
  .claude/skills/deepstream-eval-and-finetune/examples/rtdetr-aerial-sheep/run.sh
# -> reports/rtdetr_sheep_report.pdf
```
Stages are idempotent (skip if their output exists; delete to redo). The sections
below are the same steps, broken out for manual / step-by-step runs.

## 1. Ingest the dataset → COCO + labels.txt
```bash
python $SK/scripts/ingest_dataset.py \
  --repo-id keremberke/aerial-sheep-object-detection \
  --dest data/aerial_sheep --splits train valid test
# COCO passthrough (auto-detects _annotations.coco.json per split):
#   data/aerial_sheep/annotations_{train,valid,test}.json + labels.txt (1 class)
# images live alongside each split's json (see ingest_manifest.json -> images_dir)
```

## 2. Baseline — deploy STOCK RT-DETR + DeepStream eval
Reuse a stock RT-DETR deploy (COCO-80; it has `sheep` at index 18). Build the eval
set against the model's COCO-80 `labels.txt` so `sheep` maps by name, then score:
```bash
M=models/rtdetr_sheep_orig
python $SK/scripts/build_eval_set.py --labels $M/config/labels.txt --out $M/eval/eval_set \
  --local-coco data/aerial_sheep/annotations_valid.json \
  --coco-images-root <valid images dir> \
  --n-eval 200 --canvas-width 640 --canvas-height 640 --bbox-format xywh --resize-mode stretch
bash $SK/scripts/gpu_guard.sh
python $SK/scripts/eval_engine.py --app $SK/scripts/ds_image_eval \
  --engine-config $M/config/config_infer.txt --eval-set $M/eval/eval_set \
  --labels $M/config/labels.txt --predictions $M/eval/predictions_engine.json \
  --metrics $M/eval/engine_metrics.json --perf-out $M/eval/perf.json
# -> baseline map_50 ~0.19 (only the 'sheep' GT contributes; other COCO preds drop out)
```

## 3. Fine-tune (RT-DETR)
This worked example runs the bundled `train.py` directly; the full skill routes the same
fine-tune through `tao-finetune-huggingface-model` (tao-skill-bank).
```bash
python $SK/scripts/coco_to_hf.py --coco data/aerial_sheep/annotations_train.json \
  --images <train images dir> --out runs/rtdetr_sheep/data/train
python $SK/scripts/coco_to_hf.py --coco <120-image valid subset json> \
  --images <valid images dir> --out runs/rtdetr_sheep/data/eval
# then, from runs/rtdetr_sheep/ (config.yaml is in this example dir):
cd runs/rtdetr_sheep && python train.py --config config.yaml   # 30 epochs -> checkpoints/final
```

## 4. Deploy fine-tuned + DeepStream eval
Export `checkpoints/final` → ONNX, reuse the RT-DETR parser (`.so` reads the class
count from the logits dim — works for 1 class), write a `config_infer.txt` with
`num-detected-classes=1`, then build the eval set against the **fine-tuned**
`labels.txt` (`sheep`, same 200 images) and score:
```bash
F=models/rtdetr_sheep_ft
python $SK/scripts/build_eval_set.py --labels $F/config/labels.txt --out $F/eval/eval_set \
  --local-coco data/aerial_sheep/annotations_valid.json --coco-images-root <valid images dir> \
  --n-eval 200 --canvas-width 640 --canvas-height 640 --bbox-format xywh --resize-mode stretch
bash $SK/scripts/gpu_guard.sh
python $SK/scripts/eval_engine.py --app $SK/scripts/ds_image_eval \
  --engine-config $F/config/config_infer.txt --eval-set $F/eval/eval_set \
  --labels $F/config/labels.txt --predictions $F/eval/predictions_engine.json \
  --metrics $F/eval/engine_metrics.json --perf-out $F/eval/perf.json
python $SK/scripts/bench_trtexec.py --onnx $F/model/rtdetr_sheep_ft.onnx --out $F/eval/perf.json
# -> fine-tuned map_50 ~0.87
```

## 5. Qualitative samples + report
```bash
python $SK/scripts/compare_runs.py \
  --original-metrics models/rtdetr_sheep_orig/eval/engine_metrics.json \
  --finetuned-metrics models/rtdetr_sheep_ft/eval/engine_metrics.json \
  --original-perf models/rtdetr_sheep_ft/eval/perf.json \
  --finetuned-perf models/rtdetr_sheep_ft/eval/perf.json \
  --output reports/sheep_comparison.json
# 5–10 rendered before/after frames (GT | baseline | fine-tuned)
python $SK/scripts/sample_overlays.py \
  --orig-eval-set models/rtdetr_sheep_orig/eval/eval_set --orig-predictions models/rtdetr_sheep_orig/eval/predictions_engine.json \
  --ft-eval-set models/rtdetr_sheep_ft/eval/eval_set --ft-predictions models/rtdetr_sheep_ft/eval/predictions_engine.json \
  --out-dir reports/samples --n 6 --conf 0.3
python $SK/scripts/make_report.py --comparison reports/sheep_comparison.json --samples-dir reports/samples \
  --config runs/rtdetr_sheep/config.yaml --train-log runs/rtdetr_sheep/train.log \
  --precision fp16 --model-id "PekingU/rtdetr_r50vd (RT-DETR)" \
  --dataset-id "keremberke/aerial-sheep-object-detection" \
  --output reports/rtdetr_sheep_report.pdf
# -> reports/samples/sample_*.png + reports/rtdetr_sheep_report.pdf (§5e samples)
```

## Notes / gotchas
- **Single class** trips torchmetrics (`map_per_class` is a 0-dim scalar) — handled
  in `compute_map.py`.
- **Per-class report rows** key off classes with real AP (≥0); a stock model scored
  in COCO-80 space lists only `sheep`, not 80 rows — handled in `make_report.py`.
- **eval_loss ≠ mAP**: eval_loss rose every epoch yet deployed mAP@50 hit 0.87.
  Keep `load_best_model_at_end: false`.
- **mAP@75 ≪ mAP@50** (0.49 vs 0.87) is expected for ~20 px boxes — boxes are
  found but not pixel-tight; push input resolution / epochs if tight boxes matter.
