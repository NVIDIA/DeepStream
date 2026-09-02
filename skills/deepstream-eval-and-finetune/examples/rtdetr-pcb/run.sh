#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# End-to-end eval-and-improve loop for RT-DETR + itsyoboieltr/pcb (PKU/HRIPCB 6-class defects).
# Mirrors the rtdetr-aerial-sheep run.sh but:
#   - dataset is converted from a custom HF schema to COCO first (convert_pcb.py)
#   - NO PCB defect class exists in COCO-80, so BOTH eval-set legs are built against
#     the 6-class TARGET labels (GT preserved). The baseline only differs in
#     eval_engine --labels (stock COCO-80) -> COCO preds remap to nothing -> honest mAP 0.
#
# Run INSIDE the DeepStream container from the working root:
#   docker run --rm --gpus all --shm-size=16g -v "$PWD":/work -w /work \
#     --entrypoint bash nvcr.io/nvidia/deepstream:9.1-triton-multiarch \
#     .claude/skills/deepstream-eval-and-finetune/examples/rtdetr-pcb/run.sh
set -euo pipefail

SK=.claude/skills/deepstream-eval-and-finetune
EX=$SK/examples/rtdetr-aerial-sheep          # reuse RT-DETR assets (export, parser, cfg template, train.py)
PA=$SK/examples/rtdetr-pcb                   # pcb-specific assets (convert, config) — now inside the skill
PY=build/.venv_train/bin/python
export HF_HOME=${HF_HOME:-/work/build/hf_cache} NO_ALBUMENTATIONS_UPDATE=1 TOKENIZERS_PARALLELISM=false
DATA=data/pcb_defect
TGT=$DATA/labels.txt                          # 6-class TARGET label space (both eval legs)
ORIG=models/rtdetr_pcb_orig
FT=models/rtdetr_pcb_ft
banner(){ echo; echo "==================== $* ===================="; }

banner "build the deployed-eval C app (ds_image_eval)"
make -C $SK/scripts

banner "0. convert itsyoboieltr/pcb -> COCO + labels.txt (6 classes)"
if [ ! -f $DATA/annotations_test.json ]; then
  $PY $PA/convert_pcb.py --repo-id itsyoboieltr/pcb --dest $DATA
fi
TEST_IMG=$($PY -c "import json;print(json.load(open('$DATA/ingest_manifest.json'))['splits']['test']['images_dir'])")
TRAIN_IMG=$($PY -c "import json;print(json.load(open('$DATA/ingest_manifest.json'))['splits']['train']['images_dir'])")
VALID_IMG=$($PY -c "import json;print(json.load(open('$DATA/ingest_manifest.json'))['splits']['valid']['images_dir'])")
echo "test images: $TEST_IMG"

build_parser(){ mkdir -p "$1/parser"; cp $EX/assets/rtdetr_parser.cpp $EX/assets/Makefile "$1/parser/"; make -C "$1/parser" >/dev/null; }
write_cfg(){ mkdir -p "$1/config"; sed -e "s#@ONNX@#../model/$2#" -e "s#@ENGINE@#../model/$3#" -e "s#@NCLASSES@#$4#" \
    $EX/assets/config_infer.template.txt > "$1/config/config_infer.txt"; }
# build the canonical eval set against the 6-class TARGET labels (GT preserved) on the TEST split, cap 200
build_eval(){  # $1 = model dir
  $PY $SK/scripts/build_eval_set.py --labels $TGT --out $1/eval/eval_set \
    --local-coco $DATA/annotations_test.json --coco-images-root "$TEST_IMG" \
    --n-eval 200 --canvas-width 640 --canvas-height 640 --bbox-format xywh --resize-mode stretch
}

banner "1-2. deploy STOCK RT-DETR (baseline) + DeepStream eval (honest mAP~0; no PCB class in COCO)"
if [ ! -f $ORIG/eval/engine_metrics.json ]; then
  $PY $EX/assets/export_rtdetr.py --checkpoint PekingU/rtdetr_r50vd \
    --out-onnx $ORIG/model/rtdetr_orig.onnx --out-labels $ORIG/config/labels.txt   # 80 COCO labels
  build_parser $ORIG
  write_cfg $ORIG rtdetr_orig.onnx rtdetr_orig_ds.engine 80
  build_eval $ORIG
  bash $SK/scripts/gpu_guard.sh || true
  # deployed model labels = stock COCO-80 -> COCO preds remap to none of the 6 targets -> mAP 0
  $PY $SK/scripts/eval_engine.py --app $SK/scripts/ds_image_eval \
    --engine-config $ORIG/config/config_infer.txt --eval-set $ORIG/eval/eval_set \
    --labels $ORIG/config/labels.txt --predictions $ORIG/eval/predictions_engine.json \
    --metrics $ORIG/eval/engine_metrics.json --perf-out $ORIG/eval/perf.json
fi

banner "3. fine-tune RT-DETR on PCB defects (30 epochs, 6 classes)"
mkdir -p runs/rtdetr_pcb
cp $PA/config_pcb.yaml runs/rtdetr_pcb/config.yaml
cp $EX/assets/train.py runs/rtdetr_pcb/train.py
if [ ! -d runs/rtdetr_pcb/data/train ]; then
  $PY $SK/scripts/coco_to_hf.py --coco $DATA/annotations_train.json --images "$TRAIN_IMG" \
    --out runs/rtdetr_pcb/data/train
  # 120-image eval subset from validation for fast per-epoch eval
  $PY -c "import json;c=json.load(open('$DATA/annotations_valid.json'));k={im['id'] for im in c['images'][:120]};c['images']=[i for i in c['images'] if i['id'] in k];c['annotations']=[a for a in c['annotations'] if a['image_id'] in k];json.dump(c,open('/tmp/pcb_valid_120.json','w'))"
  $PY $SK/scripts/coco_to_hf.py --coco /tmp/pcb_valid_120.json --images "$VALID_IMG" \
    --out runs/rtdetr_pcb/data/eval
fi
if [ ! -d runs/rtdetr_pcb/checkpoints/final ]; then
  ( cd runs/rtdetr_pcb && /work/$PY train.py --config config.yaml > train.log 2>&1 )
fi

banner "4. deploy FINE-TUNED + DeepStream eval"
if [ ! -f $FT/eval/engine_metrics.json ]; then
  $PY $EX/assets/export_rtdetr.py --checkpoint runs/rtdetr_pcb/checkpoints/final \
    --out-onnx $FT/model/rtdetr_pcb_ft.onnx --out-labels $FT/config/labels.txt    # 6 labels
  build_parser $FT
  write_cfg $FT rtdetr_pcb_ft.onnx rtdetr_pcb_ft_ds.engine 6
  $PY $SK/scripts/record_deployed_checkpoint.py --run-dir runs/rtdetr_pcb --ft-dir $FT \
    --onnx-name rtdetr_pcb_ft.onnx --engine-name rtdetr_pcb_ft_ds.engine || true
  build_eval $FT
  bash $SK/scripts/gpu_guard.sh || true
  $PY $SK/scripts/eval_engine.py --app $SK/scripts/ds_image_eval \
    --engine-config $FT/config/config_infer.txt --eval-set $FT/eval/eval_set \
    --labels $FT/config/labels.txt --predictions $FT/eval/predictions_engine.json \
    --metrics $FT/eval/engine_metrics.json --perf-out $FT/eval/perf.json
  $PY $SK/scripts/bench_trtexec.py --onnx $FT/model/rtdetr_pcb_ft.onnx --out $FT/eval/perf.json || true
fi

banner "5. before/after report"
mkdir -p reports
$PY $SK/scripts/collect_env.py --out build/env_info.json --image nvcr.io/nvidia/deepstream:9.1-triton-multiarch || true
$PY $SK/scripts/compare_runs.py \
  --original-metrics $ORIG/eval/engine_metrics.json --finetuned-metrics $FT/eval/engine_metrics.json \
  --original-perf $FT/eval/perf.json --finetuned-perf $FT/eval/perf.json \
  --train-coco $DATA/annotations_train.json --eval-gt $FT/eval/eval_set/ground_truth.json \
  --output reports/pcb_comparison.json
$PY $SK/scripts/sample_overlays.py \
  --orig-eval-set $ORIG/eval/eval_set --orig-predictions $ORIG/eval/predictions_engine.json \
  --ft-eval-set $FT/eval/eval_set --ft-predictions $FT/eval/predictions_engine.json \
  --out-dir reports/samples_pcb --n 6 --conf 0.2 --min-gt 1 --max-gt 12 --max-boxes 12
$PY $SK/scripts/make_report.py --comparison reports/pcb_comparison.json --samples-dir reports/samples_pcb \
  --config runs/rtdetr_pcb/config.yaml --train-log runs/rtdetr_pcb/train.log --precision fp16 \
  --deployed-checkpoint $FT/model/deployed_checkpoint.json \
  --env build/env_info.json --image nvcr.io/nvidia/deepstream:9.1-triton-multiarch \
  --model-id "PekingU/rtdetr_r50vd (RT-DETR)" --model-link "https://huggingface.co/PekingU/rtdetr_r50vd" \
  --model-license "Apache-2.0" \
  --model-desc "RT-DETR R50vd. Baseline = stock COCO-80 model (no PCB-defect class -> honest mAP 0); fine-tuned = 30 epochs on PKU/HRIPCB (6 defect classes)." \
  --dataset-id "itsyoboieltr/pcb (PKU/HRIPCB)" \
  --dataset-link "https://huggingface.co/datasets/itsyoboieltr/pcb" \
  --dataset-desc "PCB surface-defect detection, 6 classes (mouse_bite, spur, missing_hole, short, open_circuit, spurious_copper); 6370 train. Eval on 200 test images." \
  --output reports/rtdetr_pcb_report.pdf
banner "DONE -> reports/rtdetr_pcb_report.pdf"
