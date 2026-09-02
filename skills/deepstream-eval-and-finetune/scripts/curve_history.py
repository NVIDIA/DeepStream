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

"""Single source of truth for the fine-tuning loss/eval curve, so "continue training"
PRESERVES and EXTENDS the graph instead of restarting it.

The live ``runs/<name>/train.log`` only ever holds the CURRENT training segment (a
warm-start trains a FRESH schedule, so its epochs run 0..N — not 20..20+N), and the log
is opened in truncate mode each run. So on its own the log can never show the full
history across continues. We keep that history in ``runs/<name>/curve_history.json`` in
ABSOLUTE (cumulative) epoch x-values, and present:

    display = committed history  +  (current train.log offset by prev_epochs)

A never-continued run has no history file, so ``merged()`` == the raw current segment and
behaves exactly as before this module existed.

Used as a LIBRARY by the web server, make_report.py and record_deployed_checkpoint.py
(read paths: ``load`` / ``merged``), and as a CLI by the pipeline (``commit`` — invoked as
a subprocess so app/pipeline.py keeps to its "invoke scripts, don't import them" rule)."""
import argparse
import ast
import json
import math
import os

HIST_NAME = "curve_history.json"
# map_epoch/eval_map/eval_map_50 are a SEPARATE parallel triple from the eval_loss arrays: not every
# eval line has mAP (older runs / recovered history logged only eval_loss), so the mAP curve simply
# starts wherever per-epoch mAP logging began, independent of the loss curve.
_KEYS = ("train_epoch", "train_loss", "eval_epoch", "eval_loss",
         "map_epoch", "eval_map", "eval_map_50")


def _empty():
    return {"prev_epochs": 0.0, "train_epoch": [], "train_loss": [],
            "eval_epoch": [], "eval_loss": [],
            "map_epoch": [], "eval_map": [], "eval_map_50": []}


def _num(v):
    """Coerce a logged metric to float, or None when it isn't numeric.

    transformers >= 5 renders Trainer log dicts with STRING values
    (`{'loss': '12.53', 'epoch': '7.967'}`); v4 rendered floats. Without this,
    parse_log returned strings and merged() raised
    `TypeError: can only concatenate str (not "float") to str` on `e + off`,
    which 500s /results/curve + /results/summary and crashes make_report.py.
    Accepting both shapes keeps old and new logs working."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def parse_log(run_dir):
    """Current training segment from runs/<name>/train.log. Each line is a python dict
    repr (single quotes) — parse with ast.literal_eval, NOT json.loads. Epochs are
    segment-local (0..N). Values are coerced to float (see _num)."""
    cur = {k: [] for k in _KEYS}
    path = os.path.join(run_dir, "train.log") if run_dir else None
    if not path or not os.path.exists(path):
        return cur
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
                if not isinstance(d, dict) or "epoch" not in d:
                    continue
                ep = _num(d.get("epoch"))
                if ep is None:
                    continue
                if "eval_loss" in d:
                    cur["eval_epoch"].append(ep); cur["eval_loss"].append(_num(d["eval_loss"]))
                    if "eval_map" in d:   # per-epoch mAP (only when compute_metrics is wired in)
                        cur["map_epoch"].append(ep); cur["eval_map"].append(_num(d["eval_map"]))
                        cur["eval_map_50"].append(_num(d.get("eval_map_50")))
                elif "loss" in d:
                    cur["train_epoch"].append(ep); cur["train_loss"].append(_num(d["loss"]))
    except Exception:
        pass
    return cur


def load(run_dir):
    """Committed prior-segment history (absolute epochs), or an empty skeleton."""
    p = os.path.join(run_dir, HIST_NAME) if run_dir else None
    if p and os.path.exists(p):
        try:
            with open(p) as f:
                h = json.load(f)
            base = _empty()
            for k in base:
                if k in h:
                    base[k] = h[k]
            return base
        except Exception:
            pass
    return _empty()


def merged(run_dir):
    """The full stitched curve to display: committed history followed by the current
    train.log segment offset onto the cumulative epoch axis. Same dict shape as
    parse_log (four arrays) so existing consumers are drop-in."""
    h = load(run_dir)
    cur = parse_log(run_dir)
    off = h.get("prev_epochs", 0.0) or 0.0
    # Segment committed to history but train.log not cleared (manual CLI / recovery): same
    # eval tail as history → skip the duplicate current segment instead of doubling epochs.
    if (off > 0 and cur["eval_map"] and h["eval_map"]
            and max(cur["eval_epoch"]) <= off
            and abs(h["eval_map"][-1] - cur["eval_map"][-1]) < 1e-5):
        cur = {k: [] for k in _KEYS}
    return {
        "train_epoch": list(h["train_epoch"]) + [e + off for e in cur["train_epoch"]],
        "train_loss": list(h["train_loss"]) + list(cur["train_loss"]),
        "eval_epoch": list(h["eval_epoch"]) + [e + off for e in cur["eval_epoch"]],
        "eval_loss": list(h["eval_loss"]) + list(cur["eval_loss"]),
        "map_epoch": list(h["map_epoch"]) + [e + off for e in cur["map_epoch"]],
        "eval_map": list(h["eval_map"]) + list(cur["eval_map"]),
        "eval_map_50": list(h["eval_map_50"]) + list(cur["eval_map_50"]),
    }


def _per_class_from_log(run_dir):
    """Last eval line's eval_cls_* keys from train.log (if any)."""
    out = {}
    path = os.path.join(run_dir, "train.log") if run_dir else None
    if not path or not os.path.exists(path):
        return out
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
                if isinstance(d, dict) and any(k.startswith("eval_cls_") for k in d):
                    out = {k[len("eval_cls_"):]: v for k, v in d.items() if k.startswith("eval_cls_")}
    except Exception:
        pass
    return out


def commit(run_dir, segment_epochs=None):
    """Fold the CURRENT train.log segment into the persistent history (offset onto the
    cumulative axis) and bump prev_epochs by the segment length. Call this ONCE per
    continue, BEFORE the next segment overwrites train.log. ``segment_epochs`` is the
    just-finished config's num_train_epochs (exact intent); falls back to the max logged
    epoch when not given."""
    h = load(run_dir)
    cur = parse_log(run_dir)
    pc = _per_class_from_log(run_dir)
    if pc:
        h["per_class_latest"] = pc
    off = h.get("prev_epochs", 0.0) or 0.0
    h["train_epoch"] = list(h["train_epoch"]) + [e + off for e in cur["train_epoch"]]
    h["train_loss"] = list(h["train_loss"]) + list(cur["train_loss"])
    h["eval_epoch"] = list(h["eval_epoch"]) + [e + off for e in cur["eval_epoch"]]
    h["eval_loss"] = list(h["eval_loss"]) + list(cur["eval_loss"])
    h["map_epoch"] = list(h["map_epoch"]) + [e + off for e in cur["map_epoch"]]
    h["eval_map"] = list(h["eval_map"]) + list(cur["eval_map"])
    h["eval_map_50"] = list(h["eval_map_50"]) + list(cur["eval_map_50"])
    seg = segment_epochs
    if not seg:
        seg = math.ceil(max(cur["train_epoch"])) if cur["train_epoch"] else 0
    h["prev_epochs"] = off + float(seg)
    with open(os.path.join(run_dir, HIST_NAME), "w") as f:
        json.dump(h, f)
    return h


def seed(run_dir, train_epoch, train_loss, eval_epoch, eval_loss, prev_epochs,
         map_epoch=None, eval_map=None, eval_map_50=None):
    """Write a history file directly (one-time recovery of a segment whose train.log was
    already overwritten — reconstructed from the original UI run log). mAP arrays are optional
    (older runs logged only eval_loss)."""
    h = {"prev_epochs": float(prev_epochs), "train_epoch": list(train_epoch),
         "train_loss": list(train_loss), "eval_epoch": list(eval_epoch),
         "eval_loss": list(eval_loss), "map_epoch": list(map_epoch or []),
         "eval_map": list(eval_map or []), "eval_map_50": list(eval_map_50 or [])}
    with open(os.path.join(run_dir, HIST_NAME), "w") as f:
        json.dump(h, f)
    return h


def latest_per_class(run_dir):
    """Per-class AP for the MOST RECENT eval epoch: live train.log first, else committed
    history (after train.log was cleared post-commit)."""
    out = _per_class_from_log(run_dir)
    if out:
        return out
    h = load(run_dir)
    return dict(h.get("per_class_latest") or {})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("commit", help="fold current train.log segment into curve_history.json")
    c.add_argument("--run-dir", required=True, help="runs/<name> (has train.log)")
    c.add_argument("--segment-epochs", type=float, default=None,
                   help="num_train_epochs of the just-finished segment")
    args = ap.parse_args()
    if args.cmd == "commit":
        h = commit(args.run_dir, args.segment_epochs)
        print(f"[curve_history] committed segment; prev_epochs={h['prev_epochs']} "
              f"({len(h['train_epoch'])} train pts, {len(h['eval_epoch'])} eval pts)")


if __name__ == "__main__":
    main()
