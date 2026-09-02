#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved. Apache-2.0.
#
# Reproducible RT-DETR fine-tune in the tao NGC PyTorch container (Stage 3), with:
#   • per-epoch eval ALWAYS on -> training-loss + eval-loss + per-epoch mAP curves populate
#   • DURABLE host-side logging: everything under logs/<run>/<timestamp>/ on the WORKING ROOT
#     (mounted into the container), NEVER overwritten — so re-runs can't clobber a prior log
#     and the logs survive the ephemeral --rm container for later reference / user queries.
#
# Reads a LOCAL COCO json + images dir (copy remote/SMB datasets local first — SMB per-file
# latency kills training; a parallel copy defeats it). No HF `datasets` dep (installs clean on NGC).
#
# Run INSIDE the tao NGC PyTorch image, from the project working root:
#   docker run --rm --gpus all --shm-size=16g -v "$PWD":/work -w /work \
#     nvcr.io/nvidia/pytorch:25.03-py3 \
#     bash .claude/skills/deepstream-eval-and-finetune/scripts/ngc/run_ngc_finetune.sh \
#       <run_name> <config.yaml> <train_coco.json> <eval_coco.json> <images_dir> [MAX_STEPS]
# Windows/WSL: also add `-v "$DS_DATA_VOLUME":/work/data -e DS_DATA_CACHE_ROOT=/work/data`
# using the project volume printed by app/launch.ps1 or app/launch.sh.
#
# Outputs: runs/<run_name>/checkpoints/final  +  logs/<run_name>/<timestamp>/{finetune.log,versions.txt,config.yaml}
# and runs/<run_name>/train.log (the report/UI curve source, refreshed from the durable log).
set -euo pipefail
. "${NGC_FT_HERE:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}/snapshot_reexec.sh"
ngc_snapshot_reexec NGC_FT "${BASH_SOURCE[0]}" "$@"
RUN="${1:?run name}"; CFG="${2:?config.yaml}"; TR="${3:?train coco json}"; EV="${4:?eval coco json}"; IMG="${5:?images dir}"; MAXS="${6:-}"
HERE="${NGC_FT_HERE:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
export HF_HOME="${HF_HOME:-/work/build/hf_cache}" NO_ALBUMENTATIONS_UPDATE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1

# Hard Stage-2 gate. Registration preserves an existing non-standard orig_dir; set ORIG_DIR
# when this is a new external run whose baseline does not use models/<run>_orig.
REGISTER=(python "$HERE/ui_register.py" --run "$RUN" --config "$CFG"
          --blurb "NGC/tao fine-tune (running)" --require-baseline)
[ -n "${ORIG_DIR:-}" ] && REGISTER+=(--orig-dir "$ORIG_DIR")
"${REGISTER[@]}"

TS="$(date +%Y%m%d_%H%M%S)"
LOGDIR="logs/${RUN}/${TS}"
mkdir -p "$LOGDIR" "runs/${RUN}"
FTLOG="${LOGDIR}/finetune.log"
log(){ echo "$@" | tee -a "$FTLOG"; }
log "[ngc] run=$RUN  logdir=$LOGDIR  started=$(date -u +%FT%TZ)"
log "[ngc] config=$CFG  train=$TR  eval=$EV  images=$IMG  max_steps=${MAXS:-<full>}"

log "[ngc] installing deps (transformers + albumentations + torchmetrics; no HF datasets/dill)"
. "$HERE/pip_env.sh"
ngc_pip_install transformers==5.14.1 'albumentations==1.4.24' torchmetrics==1.9.0 accelerate==1.14.0 timm==1.0.28 >>"$FTLOG" 2>&1 \
  || { log "[ngc] DEP INSTALL FAILED — see $FTLOG"; exit 1; }
python -c "import transformers,albumentations,torchmetrics,torch;print('transformers',transformers.__version__,'albumentations',albumentations.__version__,'torchmetrics',torchmetrics.__version__,'torch',torch.__version__,'cuda',torch.cuda.is_available())" \
  | tee "${LOGDIR}/versions.txt" | tee -a "$FTLOG"
cp "$CFG" "${LOGDIR}/config.yaml" 2>/dev/null || true

STEP=""; [ -n "$MAXS" ] && STEP="--max-steps $MAXS"
# LIVE UI curve: while training runs, mirror the durable log -> runs/<run>/train.log every 20s so
# the web UI /results/curve shows progressive loss + per-epoch mAP (not only at completion). The
# authoritative final write still happens after training (below). Self-terminates on script exit.
( while :; do grep -E "\{'loss':|'eval_|'train_runtime'" "$FTLOG" > "runs/${RUN}/train.log.tmp" 2>/dev/null \
    && mv "runs/${RUN}/train.log.tmp" "runs/${RUN}/train.log"; sleep 20; done ) &
CURVE_PID=$!; trap 'kill "$CURVE_PID" 2>/dev/null || true' EXIT
log "[ngc] training RT-DETR (per-epoch eval ON) …  (live curve -> runs/${RUN}/train.log every 20s)"
set -o pipefail
python "$HERE/train_ngc.py" --config "$CFG" --train-coco "$TR" --eval-coco "$EV" --images "$IMG" $STEP 2>&1 | tee -a "$FTLOG"
kill "$CURVE_PID" 2>/dev/null || true

# The report + UI parse runs/<run>/train.log for the curves — refresh it from the durable log
# (training-loss dicts + eval_* dicts + final summary). The durable copy is the source of truth.
grep -E "\{'loss':|'eval_|'train_runtime'" "$FTLOG" > "runs/${RUN}/train.log" || true
log "[ngc] DONE  finished=$(date -u +%FT%TZ)"
log "[ngc] durable log: $FTLOG"
log "[ngc] report/UI curve log: runs/${RUN}/train.log ($(grep -c "'loss'" "runs/${RUN}/train.log" 2>/dev/null || echo 0) loss pts, $(grep -c 'eval_map' "runs/${RUN}/train.log" 2>/dev/null || echo 0) eval pts)"
[ -z "$MAXS" ] && { log "[ngc] final checkpoint:"; ls -la "runs/${RUN}/checkpoints/final" 2>&1 | tee -a "$FTLOG"; }
