#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved. Apache-2.0.
#
# AutoML / HPO sweep for the HuggingFace RT-DETR fine-tune (the tao-run-automl AutoMLRunner is
# for TAO Toolkit networks, not HF models — so this sweeps OUR NGC trainer instead). It runs N
# trials on a data SUBSET with different hyperparameters (learning rate / warmup / LR scheduler
# / weight decay), scores each by the best per-epoch eval_map (which train_ngc.py logs), picks
# the winner, and runs a FINAL full-length training at the best config. Durable, timestamped,
# never-overwritten host logs under logs/<run>/<timestamp>/.
#
# Run INSIDE the tao NGC PyTorch image, from the project working root:
#   docker run --rm --gpus all --shm-size=16g -v "$PWD":/work -w /work \
#     nvcr.io/nvidia/pytorch:25.03-py3 \
#     bash .claude/skills/deepstream-eval-and-finetune/scripts/ngc/run_ngc_automl.sh \
#       <run> <base_config.yaml> <full_train_coco> <subset_train_coco> <eval_coco> <images> \
#       [N_TRIALS=6] [TRIAL_EPOCHS=8] [FINAL_EPOCHS=20]
# Windows/WSL: also add `-v "$DS_DATA_VOLUME":/work/data -e DS_DATA_CACHE_ROOT=/work/data`
# using the project volume printed by app/launch.ps1 or app/launch.sh.
set -euo pipefail
. "${NGC_AML_HERE:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}/snapshot_reexec.sh"
ngc_snapshot_reexec NGC_AML "${BASH_SOURCE[0]}" "$@"
RUN="${1:?run}"; BASE="${2:?base config}"; FULL="${3:?full train coco}"; SUB="${4:?subset train coco}"; EV="${5:?eval coco}"; IMG="${6:?images}"
NT="${7:-6}"; TE="${8:-8}"; FE="${9:-20}"
HERE="${NGC_AML_HERE:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
export HF_HOME="${HF_HOME:-/work/build/hf_cache}" NO_ALBUMENTATIONS_UPDATE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1

# Hard Stage-2 gate. Registration preserves an existing non-standard orig_dir; set ORIG_DIR
# when this is a new external run whose baseline does not use models/<run>_orig.
REGISTER=(python "$HERE/ui_register.py" --run "$RUN" --config "$BASE"
          --blurb "AutoML/HPO sweep (running)" --require-baseline)
[ -n "${ORIG_DIR:-}" ] && REGISTER+=(--orig-dir "$ORIG_DIR")
"${REGISTER[@]}"

TS="$(date +%Y%m%d_%H%M%S)"; LOGDIR="logs/${RUN}/${TS}"; TRIALS="${LOGDIR}/trials"
mkdir -p "$TRIALS" "runs/${RUN}"
AML="${LOGDIR}/automl.log"
log(){ echo "$@" | tee -a "$AML"; }
log "[automl] run=$RUN logdir=$LOGDIR trials=$NT trial_epochs=$TE final_epochs=$FE started=$(date -u +%FT%TZ)"

# Discard one trial's checkpoint dir — trials keep only the score, not the weights. train_ngc.py
# runs them with save_strategy="no", so this is normally already empty; it is a disk-safety net.
# Containment-checked and non-recursive-force: resolves the path and refuses anything that is not
# strictly under $TRIALS, so a bad/empty variable can never widen the delete.
discard_trial_ckpt(){
  local target="$1" base resolved
  base="$(cd "$TRIALS" 2>/dev/null && pwd)" || return 0
  [ -d "$target" ] || return 0
  resolved="$(cd "$(dirname "$target")" 2>/dev/null && pwd)/$(basename "$target")" || return 0
  case "$resolved" in
    "$base"/*) ;;
    *) log "[automl] refusing to discard $resolved (not under $base)"; return 0 ;;
  esac
  find "$resolved" -mindepth 1 -delete 2>/dev/null || true
  rmdir "$resolved" 2>/dev/null || true
}

log "[automl] installing deps (transformers + albumentations + torchmetrics)"
. "$HERE/pip_env.sh"
ngc_pip_install transformers==5.14.1 'albumentations==1.4.24' torchmetrics==1.9.0 accelerate==1.14.0 timm==1.0.28 >>"$AML" 2>&1 \
  || { log "[automl] DEP INSTALL FAILED"; exit 1; }

# --- generate N trial configs (search space; first N taken) ------------------------------------
python - "$BASE" "$TRIALS" "$NT" "$TE" <<'PY' | tee -a "$AML"
import sys, yaml, os
base_path, trials_dir, nt, te = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
base = yaml.safe_load(open(base_path))
# (lr, warmup, scheduler, weight_decay)
SPACE = [
 (2.5e-5, 0.10, "cosine", 1e-4),
 (1.0e-5, 0.10, "cosine", 1e-4),
 (5.0e-5, 0.10, "cosine", 1e-4),
 (2.5e-5, 0.05, "cosine", 1e-4),
 (2.5e-5, 0.10, "linear", 1e-4),
 (5.0e-5, 0.05, "cosine", 1e-4),
 (1.0e-5, 0.05, "cosine", 1e-4),
 (5.0e-5, 0.10, "linear", 1e-4),
 (2.5e-5, 0.10, "cosine", 5e-4),
 (7.5e-5, 0.10, "cosine", 1e-4),
]
for i,(lr,wu,sch,wd) in enumerate(SPACE[:nt]):
    c = dict(base)
    c.update(learning_rate=float(lr), warmup_ratio=float(wu), lr_scheduler_type=sch,
             weight_decay=float(wd), num_train_epochs=te,
             output_dir=f"{trials_dir}/trial_{i}/ckpt", save_strategy="no")
    yaml.safe_dump(c, open(f"{trials_dir}/trial_{i}.yaml","w"))
    print(f"[automl] trial {i}: lr={lr} warmup={wu} sched={sch} wd={wd}")
PY

# --- run each trial on the SUBSET, capture best eval_map --------------------------------------
# NOTE: trials are the AutoML SEARCH, not the fine-tune — their per-epoch curves belong to the
# UI's "3a AutoML sweep" leaderboard (fed by /results/automl from leaderboard.txt + trial logs),
# NOT to runs/<run>/train.log. train.log is reserved for the FINAL full-length training (the UI's
# "3b Fine-tuning" curve), written by the final-phase mirror below — so 3b never shows trial data.
best_i=-1; best_map="-1"
for i in $(seq 0 $((NT-1))); do
  tl="${TRIALS}/trial_${i}.log"
  log "[automl] === trial $i / $((NT-1)) (subset, ${TE} ep) ==="
  set +e
  python "$HERE/train_ngc.py" --config "${TRIALS}/trial_${i}.yaml" --train-coco "$SUB" --eval-coco "$EV" --images "$IMG" >"$tl" 2>&1
  rc=$?; set -e
  bm=$(grep -oE "'eval_map': [0-9.eE+-]+" "$tl" 2>/dev/null | sed "s/'eval_map': //" | sort -g | tail -1)
  bm="${bm:-0}"
  log "[automl] trial $i rc=$rc best eval_map=$bm"
  discard_trial_ckpt "${TRIALS}/trial_${i}/ckpt"
  awk -v i="$i" -v m="$bm" 'BEGIN{print i" "m}' >> "${LOGDIR}/leaderboard.txt"
  if awk -v a="$bm" -v b="$best_map" 'BEGIN{exit !(a+0>b+0)}'; then best_i=$i; best_map=$bm; fi
done
log "[automl] leaderboard:"; sort -k2 -g -r "${LOGDIR}/leaderboard.txt" | tee -a "$AML"
log "[automl] WINNER: trial $best_i  eval_map=$best_map"

# --- final full-length training at the winning config ----------------------------------------
python - "${TRIALS}/trial_${best_i}.yaml" "$BASE" "${LOGDIR}/final_config.yaml" "$FE" "$RUN" <<'PY'
import sys, yaml
win = yaml.safe_load(open(sys.argv[1])); base = yaml.safe_load(open(sys.argv[2]))
c = dict(base)
for k in ("learning_rate","warmup_ratio","lr_scheduler_type","weight_decay"): c[k]=win[k]
c.update(num_train_epochs=int(sys.argv[4]), save_strategy="epoch",
         output_dir=f"runs/{sys.argv[5]}/checkpoints")   # per-run output — never overwrite another run
yaml.safe_dump(c, open(sys.argv[3],"w"))
print("[automl] final config:", {k:c[k] for k in ("learning_rate","warmup_ratio","lr_scheduler_type","weight_decay","num_train_epochs","output_dir")})
PY
cp "${LOGDIR}/final_config.yaml" "runs/${RUN}/config.yaml"
log "[automl] === FINAL full training (${FE} ep, full data) at winning config ==="
# LIVE UI curve: mirror the final-training log -> runs/<run>/train.log every 20s so the web UI
# shows progressive loss + per-epoch mAP during the final run (not only at completion).
( while :; do grep -E "\{'loss':|'eval_|'train_runtime'" "${LOGDIR}/final.log" > "runs/${RUN}/train.log.tmp" 2>/dev/null \
    && mv "runs/${RUN}/train.log.tmp" "runs/${RUN}/train.log"; sleep 20; done ) &
CURVE_PID=$!; trap 'kill "$CURVE_PID" 2>/dev/null || true' EXIT
set -o pipefail
python "$HERE/train_ngc.py" --config "${LOGDIR}/final_config.yaml" --train-coco "$FULL" --eval-coco "$EV" --images "$IMG" 2>&1 | tee "${LOGDIR}/final.log"
kill "$CURVE_PID" 2>/dev/null || true

# report/UI curve source from the durable final log
grep -E "\{'loss':|'eval_|'train_runtime'" "${LOGDIR}/final.log" > "runs/${RUN}/train.log" || true
log "[automl] DONE finished=$(date -u +%FT%TZ)  durable logs: $LOGDIR"
log "[automl] final checkpoint:"; ls -la "runs/${RUN}/checkpoints/final" 2>&1 | tee -a "$AML"
