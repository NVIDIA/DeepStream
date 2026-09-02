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

"""Shared detection-metric core for the deepstream-eval-and-finetune skill.

Both eval_engine.py (deployed TRT engine) and eval_hf.py (PyTorch reference)
import `score()` from here so the two numbers are produced by *identical* metric
code and are directly comparable. The implementation mirrors tao-finetune-huggingface-model's
detection run_eval.py (torchmetrics MeanAveragePrecision, box_format="xyxy"),
so engine vs HF vs finetune numbers all live on the same scale.

Predictions / targets are passed as plain Python dicts of lists (boxes in
pixel xyxy, integer labels in the *model* label space) so callers never need to
import torch themselves. CLI mode scores a predictions JSON against a
ground_truth.json (the format emitted by build_eval_set.py).
"""
import argparse
import json
from pathlib import Path


def _to_tensor_lists(items):
    import torch

    out = []
    for it in items:
        boxes = it.get("boxes", [])
        labels = it.get("labels", [])
        scores = it.get("scores")
        d = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
            "labels": torch.tensor(labels, dtype=torch.long) if labels else torch.zeros((0,), dtype=torch.long),
        }
        if scores is not None:
            d["scores"] = torch.tensor(scores, dtype=torch.float32) if scores else torch.zeros((0,))
        out.append(d)
    return out


def score(predictions, targets, id2label=None):
    """Compute COCO-style detection metrics.

    predictions: list (one per image) of {"boxes": [[x1,y1,x2,y2],...],
                 "scores": [...], "labels": [...]}  (pixel xyxy, model labels)
    targets:     list (one per image) of {"boxes": [[x1,y1,x2,y2],...],
                 "labels": [...]}
    id2label:    optional {int: name} to label the per-class AP table.

    Returns a dict with map, map_50, map_75, map_small/medium/large,
    per_class_ap, and accuracy (== map, the primary metric).
    """
    from torchmetrics.detection.mean_ap import MeanAveragePrecision

    metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
    metric.update(_to_tensor_lists(predictions), _to_tensor_lists(targets))
    m = metric.compute()

    result = {
        "map": float(m["map"].item()),
        "map_50": float(m["map_50"].item()),
        "map_75": float(m["map_75"].item()),
        "map_small": float(m["map_small"].item()),
        "map_medium": float(m["map_medium"].item()),
        "map_large": float(m["map_large"].item()),
    }
    if "classes" in m and "map_per_class" in m:
        id2label = id2label or {}
        # With a single class torchmetrics returns 0-dim scalars; .tolist() then
        # yields a bare int/float, not a list. Normalise both to lists first.
        cls = m["classes"].tolist();  ap = m["map_per_class"].tolist()
        if not isinstance(cls, list): cls = [cls]
        if not isinstance(ap, list):  ap = [ap]
        result["per_class_ap"] = {
            id2label.get(int(c), str(int(c))): float(v) for c, v in zip(cls, ap)
        }
    result["accuracy"] = result["map"]  # primary metric, matches tao-finetune-huggingface-model convention
    return result


def _load_gt(gt_path):
    """ground_truth.json -> (targets list aligned by image_id, id2label)."""
    gt = json.loads(Path(gt_path).read_text())
    id2label = {int(k): v for k, v in gt.get("label_map", {}).items()}
    by_id = {}
    for img in gt["images"]:
        boxes, labels = [], []
        for o in img["objects"]:
            x, y, w, h = o["bbox"]
            boxes.append([x, y, x + w, y + h])  # COCO xywh -> xyxy
            labels.append(int(o["category"]))   # already mapped to model label index
        by_id[img["image_id"]] = {"boxes": boxes, "labels": labels}
    return by_id, id2label


def _load_preds(pred_path):
    """predictions JSON -> {image_id: {"boxes","scores","labels"}} (xyxy)."""
    data = json.loads(Path(pred_path).read_text())
    by_id = {}
    for p in data["predictions"]:
        by_id[p["image_id"]] = {
            "boxes": [d["bbox"] for d in p["detections"]],  # xyxy pixel
            "scores": [d["score"] for d in p["detections"]],
            "labels": [int(d["label"]) for d in p["detections"]],
        }
    return by_id


def main():
    ap = argparse.ArgumentParser(description="Score a predictions JSON against ground_truth.json")
    ap.add_argument("--ground-truth", required=True, help="eval_set/ground_truth.json")
    ap.add_argument("--predictions", required=True, help="predictions_*.json (engine or hf)")
    ap.add_argument("--output", required=True, help="metrics JSON to write")
    ap.add_argument("--source", default="unknown", help="tag stored under 'source' (engine|hf)")
    args = ap.parse_args()

    targets_by_id, id2label = _load_gt(args.ground_truth)
    preds_by_id = _load_preds(args.predictions)

    # Align on the ground-truth image order; images with no prediction entry
    # contribute an empty prediction (correctly penalised as missed detections).
    ids = sorted(targets_by_id)
    targets = [targets_by_id[i] for i in ids]
    predictions = [preds_by_id.get(i, {"boxes": [], "scores": [], "labels": []}) for i in ids]

    result = score(predictions, targets, id2label)
    result["source"] = args.source
    result["n_eval"] = len(ids)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(f"[compute_map:{args.source}] map={result['map']:.4f} "
          f"map_50={result['map_50']:.4f} n={len(ids)} -> {args.output}")


if __name__ == "__main__":
    main()
