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

"""Dataset health / QC over the ingested COCO annotations — FAST and SMB-safe (no image decode; uses
only the COCO json: image width/height + bbox xywh + category). Lets a team catch dataset problems
(imbalance, bad boxes, unlabeled images, train<->val leakage) in seconds, BEFORE spending a training
run. Importable (server calls qc_for_data_dir) and runnable as a CLI."""
import argparse, json, os, statistics
from collections import defaultdict

TINY_AREA = 0.001     # < 0.1% of image area
HUGE_AREA = 0.5       # > 50% of image area
EXTREME_AR = 20.0     # w/h or h/w beyond this


def _load(path):
    with open(path) as f:
        return json.load(f)


def _split_stats(coco):
    """Per-split metrics from one COCO dict (no image decode)."""
    id2name = {c["id"]: c["name"] for c in coco.get("categories", [])}
    imgs = {im["id"]: im for im in coco.get("images", [])}
    anns = coco.get("annotations", [])
    per_class = defaultdict(lambda: {"instances": 0, "images": set()})
    boxes_per_image = defaultdict(int)
    geo = {"tiny": 0, "huge": 0, "out_of_bounds": 0, "degenerate": 0, "extreme_aspect": 0}
    for a in anns:
        cid = a.get("category_id")
        name = id2name.get(cid, str(cid))
        per_class[name]["instances"] += 1
        per_class[name]["images"].add(a.get("image_id"))
        boxes_per_image[a.get("image_id")] += 1
        b = a.get("bbox") or [0, 0, 0, 0]
        x, y, w, h = (list(b) + [0, 0, 0, 0])[:4]
        if w <= 0 or h <= 0:
            geo["degenerate"] += 1; continue
        im = imgs.get(a.get("image_id"), {})
        W, H = im.get("width") or 0, im.get("height") or 0
        if W and H:
            if (w * h) / (W * H) < TINY_AREA: geo["tiny"] += 1
            if (w * h) / (W * H) > HUGE_AREA: geo["huge"] += 1
            if x < -1 or y < -1 or x + w > W + 1 or y + h > H + 1: geo["out_of_bounds"] += 1
        ar = (w / h) if h else 0
        if ar and (ar > EXTREME_AR or ar < 1.0 / EXTREME_AR): geo["extreme_aspect"] += 1
    n_images = len(imgs)
    labeled = {a.get("image_id") for a in anns}
    bpi = list(boxes_per_image.values())
    return {
        "n_images": n_images, "n_annotations": len(anns),
        "no_label_images": n_images - len(labeled & set(imgs)),
        "boxes_per_image_avg": round(statistics.mean(bpi), 2) if bpi else 0,
        "boxes_per_image_median": int(statistics.median(bpi)) if bpi else 0,
        "per_class": {k: {"instances": v["instances"], "images": len(v["images"])}
                      for k, v in per_class.items()},
        "geometry": geo,
        "_filenames": [im.get("file_name") for im in coco.get("images", []) if im.get("file_name")],
    }


def analyze(splits, labels=None):
    """splits = {split_name: coco_dict}. Returns the QC report + plain-language flags/verdict."""
    out = {"splits": {}, "flags": []}
    for sp, coco in splits.items():
        out["splits"][sp] = _split_stats(coco)
    classes = list(labels) if labels else sorted(
        {c for s in out["splits"].values() for c in s["per_class"]})
    out["classes"] = classes
    # combined per-class train/val counts
    tr, va = out["splits"].get("train", {}), out["splits"].get("valid", {})
    comb = {}
    for c in classes:
        comb[c] = {"train_instances": (tr.get("per_class", {}).get(c) or {}).get("instances", 0),
                   "valid_instances": (va.get("per_class", {}).get(c) or {}).get("instances", 0),
                   "train_images": (tr.get("per_class", {}).get(c) or {}).get("images", 0),
                   "valid_images": (va.get("per_class", {}).get(c) or {}).get("images", 0)}
    out["combined_per_class"] = comb
    inst = {c: comb[c]["train_instances"] for c in classes}
    nz = [v for v in inst.values() if v > 0]
    out["imbalance_ratio"] = round(max(nz) / min(nz), 1) if len(nz) > 1 else 1.0
    # train<->val leakage by exact file name (cheap; no hashing of bytes)
    tn, vn = set(tr.get("_filenames", [])), set(va.get("_filenames", []))
    dups = sorted(tn & vn)
    out["leakage"] = {"count": len(dups), "examples": dups[:15]}
    for s in out["splits"].values():
        s.pop("_filenames", None)

    # ---- flags / verdict ----
    f = out["flags"]
    if dups:
        f.append({"level": "red", "msg": f"{len(dups)} image(s) appear in BOTH train and valid "
                  f"(leakage) — this inflates eval mAP. e.g. {', '.join(dups[:3])}"})
    mx = max(inst.values()) if inst else 0
    for c in classes:
        if inst.get(c, 0) == 0:
            f.append({"level": "red", "msg": f"class '{c}' has 0 training instances — it cannot be learned"})
        elif mx and (inst[c] < 25 or inst[c] < 0.05 * mx):
            f.append({"level": "amber", "msg": f"class '{c}' is under-represented "
                      f"({inst[c]} train instances vs {mx} for the largest) — expect weak AP"})
    if out["imbalance_ratio"] >= 10:
        f.append({"level": "amber", "msg": f"class imbalance ratio {out['imbalance_ratio']}:1 "
                  f"(largest vs smallest class) — consider balancing/weighting"})
    for sp, s in out["splits"].items():
        if s["no_label_images"]:
            f.append({"level": "amber", "msg": f"{s['no_label_images']} {sp} image(s) have no labels "
                      f"(ignored in training)"})
        g = s["geometry"]
        if g["degenerate"]:
            f.append({"level": "red", "msg": f"{g['degenerate']} {sp} box(es) are zero/negative area (corrupt labels)"})
        if g["out_of_bounds"]:
            f.append({"level": "amber", "msg": f"{g['out_of_bounds']} {sp} box(es) extend outside the image bounds"})
        if g["tiny"] and s["n_annotations"]:
            pct = round(100 * g["tiny"] / s["n_annotations"])
            if pct >= 20:
                f.append({"level": "amber", "msg": f"{pct}% of {sp} boxes are tiny (<0.1% of image area) "
                          f"— small objects need higher input resolution"})
    out["verdict"] = ("red" if any(x["level"] == "red" for x in f)
                      else "amber" if f else "green")
    return out


def qc_for_data_dir(data_dir):
    """Find annotations_<split>.json + labels.txt under data_dir and analyze. Returns {} if no COCO."""
    splits = {}
    for sp in ("train", "valid"):
        p = os.path.join(data_dir, f"annotations_{sp}.json")
        if os.path.exists(p):
            splits[sp] = _load(p)
    if not splits:
        return {}
    labels = None
    lf = os.path.join(data_dir, "labels.txt")
    if os.path.exists(lf):
        with open(lf) as f:
            labels = [ln.strip() for ln in f if ln.strip()]
    return analyze(splits, labels)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True, help="dir with annotations_{train,valid}.json + labels.txt")
    ap.add_argument("--out", help="write the QC json here (else stdout)")
    args = ap.parse_args()
    rep = qc_for_data_dir(args.data_dir)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rep, f, indent=2)
        print(f"[dataset_qc] verdict={rep.get('verdict')} flags={len(rep.get('flags', []))} -> {args.out}")
    else:
        print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
