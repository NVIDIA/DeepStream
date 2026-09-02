#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Stages 4-5: deploy fine-tuned RT-DETR, eval in DeepStream, generate report
set -euo pipefail

SK=.claude/skills/deepstream-eval-and-finetune
EX=$SK/examples/rtdetr-aerial-sheep
PY=build/.venv_train/bin/python
export HF_HOME=/work/build/hf_cache NO_ALBUMENTATIONS_UPDATE=1 TOKENIZERS_PARALLELISM=false
DATA=data/aerial_sheep
ORIG=models/rtdetr_sheep_orig
FT=models/rtdetr_sheep_ft

VALID_IMG=$($PY -c "import json; print(json.load(open('$DATA/ingest_manifest.json'))['splits']['valid']['images_dir'])")
echo "valid images: $VALID_IMG"

echo "==================== Stage 4: deploy FINE-TUNED RT-DETR ===================="
if [ ! -f $FT/eval/engine_metrics.json ]; then
  $PY $EX/assets/export_rtdetr.py \
    --checkpoint runs/rtdetr_sheep/checkpoints/final \
    --out-onnx $FT/model/rtdetr_sheep_ft.onnx \
    --out-labels $FT/config/labels.txt

  mkdir -p $FT/parser
  cp $EX/assets/rtdetr_parser.cpp $EX/assets/Makefile $FT/parser/
  make -C $FT/parser >/dev/null

  sed -e "s#@ONNX@#../model/rtdetr_sheep_ft.onnx#" \
      -e "s#@ENGINE@#../model/rtdetr_sheep_ft_ds.engine#" \
      -e "s#@NCLASSES@#1#" \
      $EX/assets/config_infer.template.txt > $FT/config/config_infer.txt

  $PY $SK/scripts/record_deployed_checkpoint.py \
    --run-dir runs/rtdetr_sheep --ft-dir $FT \
    --onnx-name rtdetr_sheep_ft.onnx \
    --engine-name rtdetr_sheep_ft_ds.engine || true

  echo "==================== Stage 5a: build eval set (same 200 images) ===================="
  $PY $SK/scripts/build_eval_set.py \
    --labels $FT/config/labels.txt \
    --out $FT/eval/eval_set \
    --local-coco $DATA/annotations_valid.json \
    --coco-images-root "$VALID_IMG" \
    --n-eval 200 \
    --canvas-width 640 --canvas-height 640 \
    --bbox-format xywh --resize-mode stretch

  echo "==================== Stage 5b: DeepStream eval (fine-tuned) ===================="
  bash $SK/scripts/gpu_guard.sh || true

  $PY $SK/scripts/eval_engine.py \
    --app $SK/scripts/ds_image_eval \
    --engine-config $FT/config/config_infer.txt \
    --eval-set $FT/eval/eval_set \
    --labels $FT/config/labels.txt \
    --predictions $FT/eval/predictions_engine.json \
    --metrics $FT/eval/engine_metrics.json \
    --perf-out $FT/eval/perf.json

  echo "--- Fine-tuned metrics ---"
  cat $FT/eval/engine_metrics.json

  echo "==================== Stage 5c: perf benchmark (trtexec) ===================="
  $PY $SK/scripts/bench_trtexec.py \
    --onnx $FT/model/rtdetr_sheep_ft.onnx \
    --out $FT/eval/perf.json || true
else
  echo "[skip] fine-tuned eval already exists"
fi

echo "==================== Stage 6: before/after report ===================="
mkdir -p reports

$PY $SK/scripts/collect_env.py \
  --out build/env_info.json \
  --image nvcr.io/nvidia/deepstream:9.1-triton-multiarch || true

$PY $SK/scripts/compare_runs.py \
  --original-metrics $ORIG/eval/engine_metrics.json \
  --finetuned-metrics $FT/eval/engine_metrics.json \
  --original-perf $FT/eval/perf.json \
  --finetuned-perf $FT/eval/perf.json \
  --train-coco $DATA/annotations_train.json \
  --eval-gt $FT/eval/eval_set/ground_truth.json \
  --output reports/sheep_comparison.json

$PY $SK/scripts/sample_overlays.py \
  --orig-eval-set $ORIG/eval/eval_set \
  --orig-predictions $ORIG/eval/predictions_engine.json \
  --ft-eval-set $FT/eval/eval_set \
  --ft-predictions $FT/eval/predictions_engine.json \
  --out-dir reports/samples --n 6 --conf 0.2

$PY $SK/scripts/make_report.py \
  --comparison reports/sheep_comparison.json \
  --samples-dir reports/samples \
  --config runs/rtdetr_sheep/config.yaml \
  --train-log runs/rtdetr_sheep/train.log \
  --precision fp16 \
  --deployed-checkpoint $FT/model/deployed_checkpoint.json \
  --env build/env_info.json \
  --image nvcr.io/nvidia/deepstream:9.1-triton-multiarch \
  --model-id "PekingU/rtdetr_r50vd (RT-DETR)" \
  --model-link "https://huggingface.co/PekingU/rtdetr_r50vd" \
  --model-license "Apache-2.0" \
  --model-desc "RT-DETR R50vd. Baseline = stock COCO-80 model; fine-tuned = 30 epochs on aerial-sheep (1 class)." \
  --dataset-id "keremberke/aerial-sheep-object-detection" \
  --dataset-link "https://huggingface.co/datasets/keremberke/aerial-sheep-object-detection" \
  --dataset-desc "Aerial/drone sheep detection, 1 class; dense small objects. eval on 200 valid images." \
  --output reports/rtdetr_sheep_report.pdf

echo "==================== DONE -> reports/rtdetr_sheep_report.pdf ===================="
