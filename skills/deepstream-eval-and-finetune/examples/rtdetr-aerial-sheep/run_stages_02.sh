#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Stages 0-2: ingest aerial-sheep, deploy stock RT-DETR, baseline DeepStream eval
set -euo pipefail

SK=.claude/skills/deepstream-eval-and-finetune
EX=$SK/examples/rtdetr-aerial-sheep
PY=build/.venv_train/bin/python
export HF_HOME=/work/build/hf_cache NO_ALBUMENTATIONS_UPDATE=1 TOKENIZERS_PARALLELISM=false
DATA=data/aerial_sheep
ORIG=models/rtdetr_sheep_orig

echo "==================== build ds_image_eval ===================="
make -C $SK/scripts

echo "==================== Stage 0: ingest aerial-sheep dataset ===================="
if [ ! -f $DATA/annotations_valid.json ]; then
  $PY $SK/scripts/ingest_dataset.py \
    --repo-id keremberke/aerial-sheep-object-detection \
    --dest $DATA --splits train valid test
else
  echo "[skip] already ingested"
fi

VALID_IMG=$($PY -c "import json; print(json.load(open('$DATA/ingest_manifest.json'))['splits']['valid']['images_dir'])")
echo "valid images: $VALID_IMG"

echo "==================== Stage 1: deploy STOCK RT-DETR + build parser ===================="
if [ ! -f $ORIG/eval/engine_metrics.json ]; then
  $PY $EX/assets/export_rtdetr.py \
    --checkpoint PekingU/rtdetr_r50vd \
    --out-onnx $ORIG/model/rtdetr_orig.onnx \
    --out-labels $ORIG/config/labels.txt

  mkdir -p $ORIG/parser
  cp $EX/assets/rtdetr_parser.cpp $EX/assets/Makefile $ORIG/parser/
  make -C $ORIG/parser >/dev/null

  sed -e "s#@ONNX@#../model/rtdetr_orig.onnx#" \
      -e "s#@ENGINE@#../model/rtdetr_orig_ds.engine#" \
      -e "s#@NCLASSES@#80#" \
      $EX/assets/config_infer.template.txt > $ORIG/config/config_infer.txt

  echo "==================== Stage 2: eval STOCK RT-DETR in DeepStream ===================="
  $PY $SK/scripts/build_eval_set.py \
    --labels $ORIG/config/labels.txt \
    --out $ORIG/eval/eval_set \
    --local-coco $DATA/annotations_valid.json \
    --coco-images-root "$VALID_IMG" \
    --n-eval 200 \
    --canvas-width 640 --canvas-height 640 \
    --bbox-format xywh --resize-mode stretch

  bash $SK/scripts/gpu_guard.sh || true

  $PY $SK/scripts/eval_engine.py \
    --app $SK/scripts/ds_image_eval \
    --engine-config $ORIG/config/config_infer.txt \
    --eval-set $ORIG/eval/eval_set \
    --labels $ORIG/config/labels.txt \
    --predictions $ORIG/eval/predictions_engine.json \
    --metrics $ORIG/eval/engine_metrics.json \
    --perf-out $ORIG/eval/perf.json
else
  echo "[skip] baseline eval already exists"
fi

echo "==================== Stages 0-2 DONE ===================="
echo "--- Baseline metrics ---"
cat $ORIG/eval/engine_metrics.json
