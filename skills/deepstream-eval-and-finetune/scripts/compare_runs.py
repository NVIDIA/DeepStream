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

"""Merge the ORIGINAL-deployed and FINE-TUNED-deployed runs into a before/after.

Both inputs are deployed-engine metrics (from eval_engine.py) — same DeepStream
path, same images — so the comparison is a true deployed before/after. There is
NO pass/fail gate: this just quantifies the improvement (accuracy + perf) for the
report. The original (baseline) accuracy may be low — even 0 on KPI classes the
model cannot output yet; that is recorded per-class, not judged.
"""
import argparse
import json
from pathlib import Path

ACC = ["map", "map_50", "map_75", "map_small", "map_medium", "map_large"]


def _load(p):
    return json.loads(Path(p).read_text()) if p and Path(p).exists() else {}


_PK = ("trtexec_qps", "gpu_compute_ms_mean", "gpu_compute_ms_p99", "pipeline_fps")


# --- dataset class distribution: per-class instance + image counts (train vs eval) ----------
def _coco_dist(path):
    """{name: {instances, images}} + n_images from a COCO annotations file. `categories` gives
    id->name (so 0/1-based ids are handled); classes with no annotations still appear (count 0)."""
    d = _load(path)
    if not d or "categories" not in d:
        return {}, 0
    id2name = {c["id"]: c["name"] for c in d.get("categories", [])}
    inst, imgs = {}, {}
    for a in d.get("annotations", []):
        nm = id2name.get(a.get("category_id"), str(a.get("category_id")))
        inst[nm] = inst.get(nm, 0) + 1
        imgs.setdefault(nm, set()).add(a.get("image_id"))
    names = list(dict.fromkeys(list(id2name.values()) + list(inst.keys())))
    return ({nm: {"instances": inst.get(nm, 0), "images": len(imgs.get(nm, ()))} for nm in names},
            len(d.get("images", [])))


def _gt_dist(path):
    """{name: {instances, images}} + n_images from an eval ground_truth.json (label_map + objects)."""
    d = _load(path)
    if not d or "images" not in d:
        return {}, 0
    lm = d.get("label_map", {})
    inst, imgs = {}, {}
    for im in d.get("images", []):
        seen = set()
        for o in im.get("objects", []):
            nm = lm.get(str(o.get("category")), str(o.get("category")))
            inst[nm] = inst.get(nm, 0) + 1
            seen.add(nm)
        for nm in seen:
            imgs[nm] = imgs.get(nm, 0) + 1
    names = list(dict.fromkeys(list(lm.values()) + list(inst.keys())))
    return ({nm: {"instances": inst.get(nm, 0), "images": imgs.get(nm, 0)} for nm in names},
            len(d.get("images", [])))


def _build_distribution(train_coco, eval_gt):
    """Join train (COCO) + eval (ground_truth) per-class counts BY NAME (the stable key across
    0/1-based id schemes). Returns None if neither source is available."""
    tr, n_tr = _coco_dist(train_coco) if train_coco else ({}, 0)
    ev, n_ev = _gt_dist(eval_gt) if eval_gt else ({}, 0)
    if not tr and not ev:
        return None
    names = list(tr.keys())
    for nm in ev:
        if nm not in names:
            names.append(nm)
    names.sort(key=lambda n: (-(tr.get(n, {}).get("instances", 0)), n))  # busiest train class first
    classes = [{"name": nm,
                "train_instances": tr.get(nm, {}).get("instances", 0),
                "train_images": tr.get(nm, {}).get("images", 0),
                "eval_instances": ev.get(nm, {}).get("instances", 0),
                "eval_images": ev.get(nm, {}).get("images", 0)} for nm in names]
    return {"classes": classes, "n_train_images": n_tr, "n_eval_images": n_ev,
            "train_total_instances": sum(c["train_instances"] for c in classes),
            "eval_total_instances": sum(c["eval_instances"] for c in classes)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original-metrics", required=True, help="engine_metrics.json (original = baseline, deployed)")
    ap.add_argument("--finetuned-metrics", required=True, help="engine_metrics.json (fine-tuned, deployed)")
    ap.add_argument("--original-perf"); ap.add_argument("--finetuned-perf")
    ap.add_argument("--train-coco", help="COCO annotations_train.json -> training class distribution")
    ap.add_argument("--eval-gt", help="eval ground_truth.json -> eval-set class distribution")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    # ONE report shape: original-deployed (the baseline) vs fine-tuned-deployed.
    a, b = _load(args.original_metrics), _load(args.finetuned_metrics)
    ap_, bp = _load(args.original_perf), _load(args.finetuned_perf)

    accuracy = {}
    for k in ACC:
        if k in a or k in b:
            av, bv = a.get(k), b.get(k)
            row = {"original": av, "finetuned": bv}
            if av is not None and bv is not None:
                row["delta"] = round(bv - av, 4)
                row["x"] = round(bv / av, 1) if av and av > 1e-6 else None
            accuracy[k] = row

    perf = {}
    for src, d in [("original", ap_ or a), ("finetuned", bp or b)]:
        perf[src] = {kk: d.get(kk) for kk in _PK if d.get(kk) is not None}

    report = {
        "mode": "deployed-before-after",
        "accuracy": accuracy,
        "per_class_ap_original": a.get("per_class_ap", {}),
        "per_class_ap_finetuned": b.get("per_class_ap", {}),
        "per_class_detections_original": a.get("per_class_detections", {}),
        "per_class_detections_finetuned": b.get("per_class_detections", {}),
        "perf": perf,
        "n_eval": b.get("n_eval") or a.get("n_eval"),
        "detection_rate": {"original": a.get("detection_rate"), "finetuned": b.get("detection_rate")},
    }
    dist = _build_distribution(args.train_coco, args.eval_gt)
    if dist:
        report["dataset_distribution"] = dist
        print(f"[compare_runs] dataset distribution: {len(dist['classes'])} classes, "
              f"{dist['train_total_instances']} train / {dist['eval_total_instances']} eval instances")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2))

    m = accuracy.get("map_50", {})
    print(f"[compare_runs] map_50 baseline={m.get('original')} -> finetuned={m.get('finetuned')}"
          + (f"  ({m.get('x')}x)" if m.get("x") else "") + f"  -> {args.output}")


if __name__ == "__main__":
    main()
