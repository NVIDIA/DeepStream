#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
#
# Reproducible end-to-end driver for the RT-DETR + aerial-sheep default demo.
# Encodes the validated commands: ingest -> deploy stock (baseline) -> eval ->
# fine-tune -> deploy fine-tuned -> eval -> before/after report.
#
# RUN IT INSIDE the DeepStream container, from the working root (where models/,
# runs/, reports/, build/.venv_train live):
#
#   docker run --rm --gpus all --shm-size=16g -v "$PWD":/work -w /work \
#     --entrypoint bash nvcr.io/nvidia/deepstream:9.1-triton-multiarch \
#     .claude/skills/deepstream-eval-and-finetune/examples/rtdetr-aerial-sheep/run.sh
#
# Preflight (scripts/preflight.sh) should PASS first. Stages are idempotent: a
# stage is skipped if its main output already exists (delete it to force a redo).
set -euo pipefail

SK=.claude/skills/deepstream-eval-and-finetune
EX=$SK/examples/rtdetr-aerial-sheep
PY=build/.venv_train/bin/python
export HF_HOME=${HF_HOME:-/work/build/hf_cache} NO_ALBUMENTATIONS_UPDATE=1 TOKENIZERS_PARALLELISM=false
DATA=data/aerial_sheep
ORIG=models/rtdetr_sheep_orig
FT=models/rtdetr_sheep_ft
banner(){ echo; echo "==================== $* ===================="; }

banner "build the deployed-eval C app (ds_image_eval) — rebuilt per environment"
make -C $SK/scripts            # binary is NOT shipped; build it for this machine/DeepStream version

banner "0. ingest dataset (Roboflow COCO-zip -> COCO)"
if [ ! -f $DATA/annotations_valid.json ]; then
  $PY $SK/scripts/ingest_dataset.py --repo-id keremberke/aerial-sheep-object-detection \
    --dest $DATA --splits train valid test
fi
VALID_IMG=$($PY -c "import json;print(json.load(open('$DATA/ingest_manifest.json'))['splits']['valid']['images_dir'])")
TRAIN_IMG=$($PY -c "import json;print(json.load(open('$DATA/ingest_manifest.json'))['splits']['train']['images_dir'])")
echo "valid images: $VALID_IMG"

build_parser(){  # $1 = model dir
  mkdir -p "$1/parser"
  cp $EX/assets/rtdetr_parser.cpp $EX/assets/Makefile "$1/parser/"
  make -C "$1/parser" >/dev/null
}
write_cfg(){  # $1=model dir  $2=onnx name  $3=engine name  $4=nclasses
  mkdir -p "$1/config"
  sed -e "s#@ONNX@#../model/$2#" -e "s#@ENGINE@#../model/$3#" -e "s#@NCLASSES@#$4#" \
    $EX/assets/config_infer.template.txt > "$1/config/config_infer.txt"
}

banner "1-2. deploy STOCK RT-DETR (baseline) + DeepStream eval"
if [ ! -f $ORIG/eval/engine_metrics.json ]; then
  $PY $EX/assets/export_rtdetr.py --checkpoint PekingU/rtdetr_r50vd \
    --out-onnx $ORIG/model/rtdetr_orig.onnx --out-labels $ORIG/config/labels.txt
  build_parser $ORIG
  write_cfg $ORIG rtdetr_orig.onnx rtdetr_orig_ds.engine 80
  # baseline eval set: model's own COCO-80 labels (so "sheep" maps by name -> idx 18)
  $PY $SK/scripts/build_eval_set.py --labels $ORIG/config/labels.txt --out $ORIG/eval/eval_set \
    --local-coco $DATA/annotations_valid.json --coco-images-root "$VALID_IMG" \
    --n-eval 200 --canvas-width 640 --canvas-height 640 --bbox-format xywh --resize-mode stretch
  bash $SK/scripts/gpu_guard.sh || true
  $PY $SK/scripts/eval_engine.py --app $SK/scripts/ds_image_eval \
    --engine-config $ORIG/config/config_infer.txt --eval-set $ORIG/eval/eval_set \
    --labels $ORIG/config/labels.txt --predictions $ORIG/eval/predictions_engine.json \
    --metrics $ORIG/eval/engine_metrics.json --perf-out $ORIG/eval/perf.json
fi

banner "3. fine-tune RT-DETR on aerial-sheep (30 epochs, 1 class)"
mkdir -p runs/rtdetr_sheep
cp $EX/config.yaml runs/rtdetr_sheep/config.yaml
cp $EX/assets/train.py runs/rtdetr_sheep/train.py
if [ ! -d runs/rtdetr_sheep/data/train ]; then
  $PY $SK/scripts/coco_to_hf.py --coco $DATA/annotations_train.json --images "$TRAIN_IMG" \
    --out runs/rtdetr_sheep/data/train
  # 120-image eval subset for fast per-epoch eval during training
  $PY -c "import json;c=json.load(open('$DATA/annotations_valid.json'));k={im['id'] for im in c['images'][:120]};c['images']=[i for i in c['images'] if i['id'] in k];c['annotations']=[a for a in c['annotations'] if a['image_id'] in k];json.dump(c,open('/tmp/sheep_valid_120.json','w'))"
  $PY $SK/scripts/coco_to_hf.py --coco /tmp/sheep_valid_120.json --images "$VALID_IMG" \
    --out runs/rtdetr_sheep/data/eval
fi
if [ ! -d runs/rtdetr_sheep/checkpoints/final ]; then
  ( cd runs/rtdetr_sheep && /work/$PY train.py --config config.yaml > train.log 2>&1 )
fi

banner "4. deploy FINE-TUNED + DeepStream eval"
if [ ! -f $FT/eval/engine_metrics.json ]; then
  $PY $EX/assets/export_rtdetr.py --checkpoint runs/rtdetr_sheep/checkpoints/final \
    --out-onnx $FT/model/rtdetr_sheep_ft.onnx --out-labels $FT/config/labels.txt
  build_parser $FT
  write_cfg $FT rtdetr_sheep_ft.onnx rtdetr_sheep_ft_ds.engine 1
  $PY $SK/scripts/record_deployed_checkpoint.py --run-dir runs/rtdetr_sheep --ft-dir $FT \
    --onnx-name rtdetr_sheep_ft.onnx --engine-name rtdetr_sheep_ft_ds.engine || true
  $PY $SK/scripts/build_eval_set.py --labels $FT/config/labels.txt --out $FT/eval/eval_set \
    --local-coco $DATA/annotations_valid.json --coco-images-root "$VALID_IMG" \
    --n-eval 200 --canvas-width 640 --canvas-height 640 --bbox-format xywh --resize-mode stretch
  bash $SK/scripts/gpu_guard.sh || true
  $PY $SK/scripts/eval_engine.py --app $SK/scripts/ds_image_eval \
    --engine-config $FT/config/config_infer.txt --eval-set $FT/eval/eval_set \
    --labels $FT/config/labels.txt --predictions $FT/eval/predictions_engine.json \
    --metrics $FT/eval/engine_metrics.json --perf-out $FT/eval/perf.json
  $PY $SK/scripts/bench_trtexec.py --onnx $FT/model/rtdetr_sheep_ft.onnx --out $FT/eval/perf.json || true
fi

banner "5. before/after report"
mkdir -p reports
# system/tool provenance for the report's §7 (GPU, driver, CUDA, DeepStream, TensorRT, package versions)
$PY $SK/scripts/collect_env.py --out build/env_info.json --image nvcr.io/nvidia/deepstream:9.1-triton-multiarch || true
$PY $SK/scripts/compare_runs.py \
  --original-metrics $ORIG/eval/engine_metrics.json --finetuned-metrics $FT/eval/engine_metrics.json \
  --original-perf $FT/eval/perf.json --finetuned-perf $FT/eval/perf.json \
  --train-coco $DATA/annotations_train.json --eval-gt $FT/eval/eval_set/ground_truth.json \
  --output reports/sheep_comparison.json
# qualitative before/after frames (GT | baseline | fine-tuned)
$PY $SK/scripts/sample_overlays.py \
  --orig-eval-set $ORIG/eval/eval_set --orig-predictions $ORIG/eval/predictions_engine.json \
  --ft-eval-set $FT/eval/eval_set --ft-predictions $FT/eval/predictions_engine.json \
  --out-dir reports/samples --n 6 --conf 0.2
$PY $SK/scripts/make_report.py --comparison reports/sheep_comparison.json --samples-dir reports/samples \
  --config runs/rtdetr_sheep/config.yaml --train-log runs/rtdetr_sheep/train.log --precision fp16 \
  --deployed-checkpoint $FT/model/deployed_checkpoint.json \
  --env build/env_info.json --image nvcr.io/nvidia/deepstream:9.1-triton-multiarch \
  --model-id "PekingU/rtdetr_r50vd (RT-DETR)" --model-link "https://huggingface.co/PekingU/rtdetr_r50vd" \
  --model-license "Apache-2.0" \
  --model-desc "RT-DETR R50vd. Baseline = stock COCO-80 model (has a 'sheep' class); fine-tuned = 30 epochs on aerial-sheep (1 class)." \
  --dataset-id "keremberke/aerial-sheep-object-detection" \
  --dataset-link "https://huggingface.co/datasets/keremberke/aerial-sheep-object-detection" \
  --dataset-desc "Aerial/drone sheep detection, 1 class; dense small objects. eval on 200 valid images." \
  --output reports/rtdetr_sheep_report.pdf
banner "DONE -> reports/rtdetr_sheep_report.pdf"
