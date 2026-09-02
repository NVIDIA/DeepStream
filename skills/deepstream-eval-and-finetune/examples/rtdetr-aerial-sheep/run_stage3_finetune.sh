#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Stage 3: fine-tune RT-DETR on aerial-sheep (30 epochs, 1 class)
set -euo pipefail

SK=.claude/skills/deepstream-eval-and-finetune
EX=$SK/examples/rtdetr-aerial-sheep
PY=build/.venv_train/bin/python
export HF_HOME=/work/build/hf_cache NO_ALBUMENTATIONS_UPDATE=1 TOKENIZERS_PARALLELISM=false
DATA=data/aerial_sheep

echo "==================== Stage 3: fine-tune RT-DETR ===================="
TRAIN_IMG=$($PY -c "import json; print(json.load(open('$DATA/ingest_manifest.json'))['splits']['train']['images_dir'])")
VALID_IMG=$($PY -c "import json; print(json.load(open('$DATA/ingest_manifest.json'))['splits']['valid']['images_dir'])")
echo "train images: $TRAIN_IMG"
echo "valid images: $VALID_IMG"

mkdir -p runs/rtdetr_sheep
cp $EX/config.yaml runs/rtdetr_sheep/config.yaml
cp $EX/assets/train.py runs/rtdetr_sheep/train.py

if [ ! -d runs/rtdetr_sheep/data/train ]; then
  echo "[3a] converting train split to HF Arrow..."
  $PY $SK/scripts/coco_to_hf.py \
    --coco $DATA/annotations_train.json \
    --images "$TRAIN_IMG" \
    --out runs/rtdetr_sheep/data/train

  echo "[3b] building 120-image eval subset for per-epoch eval..."
  $PY -c "
import json
c = json.load(open('$DATA/annotations_valid.json'))
k = {im['id'] for im in c['images'][:120]}
c['images'] = [i for i in c['images'] if i['id'] in k]
c['annotations'] = [a for a in c['annotations'] if a['image_id'] in k]
json.dump(c, open('/tmp/sheep_valid_120.json', 'w'))
print('wrote 120-image eval subset')
"
  $PY $SK/scripts/coco_to_hf.py \
    --coco /tmp/sheep_valid_120.json \
    --images "$VALID_IMG" \
    --out runs/rtdetr_sheep/data/eval
else
  echo "[skip] HF Arrow datasets already exist"
fi

if [ ! -d runs/rtdetr_sheep/checkpoints/final ]; then
  echo "[3c] training (30 epochs) — logging to runs/rtdetr_sheep/train.log"
  cd runs/rtdetr_sheep
  /work/$PY train.py --config config.yaml 2>&1 | tee train.log
  echo "==================== Stage 3 DONE ===================="
else
  echo "[skip] checkpoints/final already exists"
fi
