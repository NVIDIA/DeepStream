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

"""Detection error analysis from a run's deployed predictions + ground truth (no re-inference).
Greedy IoU matching (CLASS-AGNOSTIC, so we can see what a GT box got predicted AS) yields a
confusion matrix + per-class precision/recall/TP/FP/FN, and a per-image error ranking ("hardest
images"). This is the spur->short style diagnosis, productized. Importable by the server."""
import argparse, json, os
from collections import defaultdict

MISSED = "(missed)"        # GT box with no overlapping detection -> a column
BACKGROUND = "(background)"  # detection with no overlapping GT -> a row (false positive)


def _xywh2xyxy(b):
    return [b[0], b[1], b[0] + b[2], b[1] + b[3]]


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def analyze(predictions, gt, conf=0.2, iou_thr=0.5):
    """predictions: list of {image_id, detections:[{bbox xyxy, score, label int}]}.
    gt: ground_truth.json dict (label_map, images:[{image_id, file, objects:[{bbox xywh, category}]}]).
    Class-agnostic greedy match (highest-score det first) at IoU>=iou_thr."""
    lm = {int(k): v for k, v in gt.get("label_map", {}).items()}
    names = [lm[i] for i in sorted(lm)]
    gt_by_img, file_by_img = {}, {}
    for im in gt.get("images", []):
        gt_by_img[im["image_id"]] = [(o["category"], _xywh2xyxy(o["bbox"])) for o in im.get("objects", [])]
        file_by_img[im["image_id"]] = im.get("file")
    preds_by_img = {p["image_id"]: p.get("detections", []) for p in predictions}

    confusion = defaultdict(lambda: defaultdict(int))   # row(GT/background) -> col(pred/missed) -> n
    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)
    per_image = []
    for iid, gts in gt_by_img.items():
        dets = [d for d in preds_by_img.get(iid, []) if d.get("score", 0) >= conf]
        dets.sort(key=lambda d: -d.get("score", 0))
        used = [False] * len(gts)
        img_fp = img_fn = 0
        for d in dets:
            dname = lm.get(int(d.get("label", -1)), str(d.get("label")))
            best, best_iou = -1, iou_thr
            for gi, (gcat, gbox) in enumerate(gts):
                if used[gi]:
                    continue
                v = _iou(d["bbox"], gbox)
                if v >= best_iou:
                    best, best_iou = gi, v
            if best >= 0:
                used[best] = True
                gname = lm.get(gts[best][0], str(gts[best][0]))
                confusion[gname][dname] += 1
                if gname == dname:
                    tp[dname] += 1
                else:                       # localized but wrong class: FP for pred, FN for GT
                    fp[dname] += 1; fn[gname] += 1; img_fp += 1; img_fn += 1
            else:                            # detection over nothing -> false positive
                confusion[BACKGROUND][dname] += 1
                fp[dname] += 1; img_fp += 1
        for gi, (gcat, gbox) in enumerate(gts):
            if not used[gi]:                 # GT never detected -> missed (false negative)
                gname = lm.get(gcat, str(gcat))
                confusion[gname][MISSED] += 1
                fn[gname] += 1; img_fn += 1
        per_image.append({"image_id": iid, "file": file_by_img.get(iid),
                          "fp": img_fp, "fn": img_fn, "errors": img_fp + img_fn})

    per_class = {}
    for n in names:
        t, f_, m = tp[n], fp[n], fn[n]
        per_class[n] = {"tp": t, "fp": f_, "fn": m,
                        "precision": round(t / (t + f_), 4) if (t + f_) else None,
                        "recall": round(t / (t + m), 4) if (t + m) else None}
    rows = names + [BACKGROUND]
    cols = names + [MISSED]
    totals = {"tp": sum(tp.values()), "fp": sum(fp.values()), "fn": sum(fn.values())}
    summary = _summarize(names, per_class, confusion, totals, tp, fp, fn)
    return {
        "conf": conf, "iou": iou_thr, "classes": names,
        "rows": rows, "cols": cols,
        "confusion": {r: {c: confusion[r][c] for c in cols if confusion[r][c]} for r in rows},
        "per_class": per_class,
        "totals": totals,
        "summary": summary,
        "hardest": sorted(per_image, key=lambda x: -x["errors"]),
        "n_images": len(gt_by_img),
    }


def _summarize(names, per_class, confusion, totals, tp, fp, fn):
    """Plain-language interpretation of the confusion matrix (also embeddable in the PDF)."""
    gt_count = {n: tp[n] + fn[n] for n in names}
    n_gt = sum(gt_count.values())
    mp = totals["tp"] / (totals["tp"] + totals["fp"]) if (totals["tp"] + totals["fp"]) else None
    mr = totals["tp"] / (totals["tp"] + totals["fn"]) if (totals["tp"] + totals["fn"]) else None
    pct = lambda v: "n/a" if v is None else f"{round(100 * v)}%"
    # biggest class->class confusions (off-diagonal)
    conf_pairs = sorted(
        ({"gt": r, "pred": c, "count": confusion[r][c],
          "pct": round(100 * confusion[r][c] / gt_count[r]) if gt_count.get(r) else 0}
         for r in names for c in names if r != c and confusion[r][c]),
        key=lambda x: -x["count"])
    missed = sorted(((n, confusion[n][MISSED]) for n in names if confusion[n][MISSED]),
                    key=lambda x: -x[1])
    have_gt = [(n, per_class[n]["recall"]) for n in names if gt_count[n]]
    worst = sorted(have_gt, key=lambda x: (x[1] if x[1] is not None else -1))[0] if have_gt else None
    best = max(have_gt, key=lambda x: (x[1] if x[1] is not None else -1)) if have_gt else None
    ins = [f"Across {n_gt} ground-truth boxes: {totals['tp']} correct, {totals['fp']} false positives, "
           f"{totals['fn']} missed (overall precision {pct(mp)}, recall {pct(mr)})."]
    if conf_pairs:
        c0 = conf_pairs[0]
        ins.append(f"Biggest class confusion: '{c0['gt']}' is predicted as '{c0['pred']}' "
                   f"{c0['count']}× ({c0['pct']}% of all '{c0['gt']}' instances).")
    if worst:
        wn, wr = worst
        ins.append(f"Weakest class: '{wn}' (recall {pct(wr)}, {fn[wn]} of {gt_count[wn]} missed)"
                   + (" — effectively never detected correctly." if (wr or 0) == 0 else "."))
    if missed:
        ins.append(f"Most undetected: '{missed[0][0]}' has {missed[0][1]} ground-truth boxes the model never found.")
    if best and (best[1] or 0) > 0:
        ins.append(f"Best class: '{best[0]}' (recall {pct(best[1])}).")
    return {"n_gt": n_gt, "precision": mp, "recall": mr,
            "top_confusions": conf_pairs[:5], "most_missed": [{"class": m[0], "count": m[1]} for m in missed[:5]],
            "best_class": best[0] if best else None, "worst_class": worst[0] if worst else None,
            "insights": ins}


def analyze_paths(predictions_json, gt_json, conf=0.2, iou_thr=0.5):
    with open(predictions_json) as f:
        preds = json.load(f).get("predictions", [])
    with open(gt_json) as f:
        gt = json.load(f)
    return analyze(preds, gt, conf, iou_thr)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--ground-truth", required=True)
    ap.add_argument("--conf", type=float, default=0.2)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--out")
    args = ap.parse_args()
    rep = analyze_paths(args.predictions, args.ground_truth, args.conf, args.iou)
    s = json.dumps(rep, indent=2)
    if args.out:
        open(args.out, "w").write(s); print(f"[error_analysis] wrote {args.out}")
    else:
        print(s)


if __name__ == "__main__":
    main()
