# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
#
# Record WHICH epoch's checkpoint was deployed and WHERE the model is stored, so the report
# (§2b) and the UI summary can show it. The skill deploys the MOST-TRAINED checkpoint
# ('checkpoints/final'), NOT the lowest-eval_loss one: for DETR-family detectors eval_loss is a
# poor mAP proxy (it often rises after a few epochs while detection mAP keeps improving), so the
# most-trained checkpoint is used and validated by the deployed mAP. This is shared by the CLI
# run.sh and the Web-UI pipeline so both paths emit an identical deployed_checkpoint.json.
import argparse
import ast
import json
import os


def _prev_epochs(run_dir):
    """Cumulative epochs already completed in EARLIER continue segments (0 for a normal run).
    The current train.log / final checkpoint count epochs from 0 within the latest warm-start
    segment, so add this to report the absolute deployed epoch (e.g. 50, not 30)."""
    try:
        import curve_history
        return curve_history.load(run_dir).get("prev_epochs", 0.0) or 0.0
    except Exception:
        return 0.0


def deployed_epoch(run_dir):
    """Cumulative epoch of the deployed 'final' checkpoint (continue-aware). We deploy the
    MOST-TRAINED checkpoint, i.e. the last epoch, so the deployed epoch == the max epoch of the
    full stitched curve (curve_history). _st_finetune commits each segment durably before deploy,
    so curve_history holds the complete cumulative curve. Fallbacks (no curve) read trainer_state /
    train.log / config and add prev_epochs."""
    # 0) PRIMARY: max epoch of the durable stitched curve (== deployed/most-trained epoch)
    try:
        import curve_history
        te = curve_history.merged(run_dir).get("train_epoch") or []
        if te:
            return int(round(max(te)))
    except Exception:
        pass
    off = _prev_epochs(run_dir)
    # 1) trainer_state.json inside final/ (soft-stop path)
    try:
        with open(os.path.join(run_dir, "checkpoints", "final", "trainer_state.json")) as f:
            return int(round(json.load(f).get("epoch", 0) + off))
    except Exception:
        pass
    # 2) last 'epoch' logged in train.log (each line is a python dict repr)
    try:
        last = 0.0
        with open(os.path.join(run_dir, "train.log")) as f:
            for line in f:
                i, j = line.find("{"), line.rfind("}")
                if i < 0 or j <= i:
                    continue
                try:
                    d = ast.literal_eval(line[i:j + 1])
                except Exception:
                    continue
                if isinstance(d, dict) and "epoch" in d:
                    last = max(last, float(d["epoch"]))
        if last:
            return int(round(last + off))
    except Exception:
        pass
    # 3) configured epoch count (assumes the run completed)
    try:
        import yaml
        with open(os.path.join(run_dir, "config.yaml")) as f:
            n = (yaml.safe_load(f) or {}).get("num_train_epochs")
        return int(round(n + off)) if n is not None else None
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="runs/<name> (has checkpoints/, train.log, config.yaml)")
    ap.add_argument("--ft-dir", required=True, help="models/<name>_ft (deployed fine-tuned model dir)")
    ap.add_argument("--onnx-name", required=True)
    ap.add_argument("--engine-name", required=True)
    ap.add_argument("--nvinfer-config", default="config/config_infer.txt", help="relative to --ft-dir")
    args = ap.parse_args()

    rd, fd = args.run_dir, args.ft_dir
    ep = deployed_epoch(rd)
    try:
        import checkpoint_to_evaluate
        ci = checkpoint_to_evaluate.info(rd)
        if ci and ci.get("epoch") is not None:
            ep = ci["epoch"]
    except Exception:
        pass
    # human-readable epoch tag beside checkpoints/final (symlink — no extra disk)
    tag = None
    if ep is not None:
        tag = os.path.join(rd, "checkpoints", f"final_epoch{ep}")
        try:
            if not os.path.lexists(tag):
                os.symlink("final", tag)
        except Exception:
            tag = None
    manifest = {
        "deployed_epoch": ep,
        "selection": "best-by-eval_map checkpoint when load_best_model_at_end else most-trained epoch",
        "checkpoint": os.path.join(rd, "checkpoints", "final"),
        "checkpoint_epoch_tag": tag,
        "model_name": args.onnx_name,
        "onnx": os.path.join(fd, "model", args.onnx_name),
        "engine": os.path.join(fd, "model", args.engine_name),
        "nvinfer_config": os.path.join(fd, args.nvinfer_config),
    }
    out = os.path.join(fd, "model", "deployed_checkpoint.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[record_deployed_checkpoint] deployed epoch {ep}; model {manifest['onnx']} -> {out}")


if __name__ == "__main__":
    main()
