# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Describe which checkpoint ``checkpoints/final`` represents (best-by-metric vs latest
completed epoch). Used by the Web UI evaluate button and deploy summary."""
import json
import os

import curve_history


def _load_cfg(run_dir):
    path = os.path.join(run_dir, "config.yaml") if run_dir else None
    if not path or not os.path.exists(path):
        return {}
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        pass
    # Minimal fallback when PyYAML is unavailable (e.g. bare docker python).
    out = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or ":" not in line:
                    continue
                k, v = line.split(":", 1)
                k, v = k.strip(), v.strip()
                if k == "load_best_model_at_end":
                    out[k] = v.lower() in ("true", "yes", "1")
                elif k == "num_train_epochs":
                    out[k] = int(float(v))
                elif k in ("metric_for_best_model",):
                    out[k] = v
    except Exception:
        pass
    return out


def _trainer_states(ckdir):
    """All checkpoint-*/trainer_state.json as (name, state) sorted by epoch desc."""
    out = []
    if not os.path.isdir(ckdir):
        return out
    for name in os.listdir(ckdir):
        if not name.startswith("checkpoint-"):
            continue
        tsp = os.path.join(ckdir, name, "trainer_state.json")
        if not os.path.exists(tsp):
            continue
        try:
            with open(tsp) as f:
                ts = json.load(f)
            out.append((name, ts))
        except Exception:
            continue
    out.sort(key=lambda x: x[1].get("epoch", 0), reverse=True)
    return out


def _segment_start(run_dir, prev, seg_epochs):
    """Absolute epoch index where the current (or last committed) segment begins."""
    cur = curve_history.parse_log(run_dir)
    h = curve_history.load(run_dir)
    has_current = bool(cur.get("eval_epoch") or cur.get("train_epoch"))
    if not has_current:
        return max(0, int(round(prev - seg_epochs))) if prev and seg_epochs else 0
    # train.log not cleared after commit — same segment already folded into history
    if (prev > 0 and cur.get("eval_map") and h.get("eval_map")
            and max(cur.get("eval_epoch") or [0]) <= prev
            and abs(h["eval_map"][-1] - cur["eval_map"][-1]) < 1e-5):
        return max(0, int(round(prev - seg_epochs)))
    return int(round(prev))


def _eval_line_at_segment_epoch(run_dir, seg_epoch):
    """Full eval dict from train.log at ``seg_epoch`` (segment-local), if logged."""
    import ast
    path = os.path.join(run_dir, "train.log") if run_dir else None
    if not path or not os.path.exists(path):
        return {}
    best = {}
    try:
        with open(path) as f:
            for line in f:
                i, j = line.find("{"), line.rfind("}")
                if i < 0 or j <= i:
                    continue
                try:
                    d = ast.literal_eval(line[i:j + 1])
                except Exception:
                    continue
                if not isinstance(d, dict) or "eval_map" not in d or "epoch" not in d:
                    continue
                if abs(float(d["epoch"]) - float(seg_epoch)) < 0.01:
                    best = d
    except Exception:
        pass
    return best


def _metrics_at_segment_epoch(run_dir, seg_epoch):
    """PyTorch training-eval mAP / mAP@50 / mAP@75 at ``seg_epoch``."""
    ev = _eval_line_at_segment_epoch(run_dir, seg_epoch)
    if ev:
        return ev.get("eval_map"), ev.get("eval_map_50"), ev.get("eval_map_75")
    c = curve_history.merged(run_dir)
    me, em, em50 = c.get("map_epoch") or [], c.get("eval_map") or [], c.get("eval_map_50") or []
    em75 = c.get("eval_map_75") or []
    for i, ep in enumerate(me):
        if abs(float(ep) - float(seg_epoch)) < 0.01:
            m = em[i] if i < len(em) else None
            m50 = em50[i] if i < len(em50) else None
            m75 = em75[i] if i < len(em75) else None
            return m, m50, m75
    return None, None, None


def info(run_dir):
    """Return a dict for UI/API, or None when no checkpoint exists.

    Keys: selection ('best'|'latest'), epoch (absolute), segment_epoch, metric,
    metric_value, total_epochs.
    """
    if not run_dir:
        return None
    ckdir = os.path.join(run_dir, "checkpoints")
    final = os.path.join(ckdir, "final")
    states = _trainer_states(ckdir)
    if not states and not os.path.isdir(final):
        return None

    cfg = _load_cfg(run_dir)
    load_best = bool(cfg.get("load_best_model_at_end", False))
    metric = cfg.get("metric_for_best_model") or "eval_loss"
    seg_epochs = int(cfg.get("num_train_epochs") or 0)
    prev = curve_history.load(run_dir).get("prev_epochs", 0.0) or 0.0
    seg_start = _segment_start(run_dir, prev, seg_epochs)

    latest_name, latest_ts = states[0] if states else (None, {})
    latest_seg = int(round(latest_ts.get("epoch", 0))) if latest_ts else seg_epochs

    best_seg = latest_seg
    metric_value = latest_ts.get("best_metric")
    if load_best and latest_ts.get("best_model_checkpoint"):
        bname = os.path.basename(str(latest_ts["best_model_checkpoint"]).rstrip("/"))
        bts = os.path.join(ckdir, bname, "trainer_state.json")
        if os.path.exists(bts):
            try:
                with open(bts) as f:
                    b = json.load(f)
                best_seg = int(round(b.get("epoch", best_seg)))
                metric_value = latest_ts.get("best_metric", b.get("best_metric"))
            except Exception:
                pass

    selection = "best" if load_best else "latest"
    seg_ep = best_seg if load_best else latest_seg
    total = int(round(prev)) if prev and not curve_history.parse_log(run_dir).get("train_epoch") else seg_start + latest_seg
    if curve_history.parse_log(run_dir).get("train_epoch"):
        te = curve_history.parse_log(run_dir).get("train_epoch") or []
        h = curve_history.load(run_dir)
        dup = (prev > 0 and h.get("eval_map") and curve_history.parse_log(run_dir).get("eval_map")
               and abs(h["eval_map"][-1] - curve_history.parse_log(run_dir)["eval_map"][-1]) < 1e-5)
        if not dup and te:
            total = seg_start + int(round(max(te)))
        elif prev:
            total = int(round(prev))

    map_ep, map50_ep, map75_ep = _metrics_at_segment_epoch(run_dir, seg_ep)
    ev = _eval_line_at_segment_epoch(run_dir, seg_ep)
    per_class = {k[len("eval_cls_"):]: v for k, v in ev.items() if k.startswith("eval_cls_")}

    return {
        "selection": selection,
        "epoch": seg_start + seg_ep,
        "segment_epoch": seg_ep,
        "metric": metric,
        "metric_value": metric_value,
        "map": map_ep,
        "map_50": map50_ep,
        "map_75": map75_ep,
        "per_class_ap": per_class,
        "total_epochs": total,
        "load_best_model_at_end": load_best,
    }
