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

"""Deployed-engine accuracy — the unique value of this skill.

Stage 3 (engine leg). Scores the model AS DEPLOYED: the real TRT engine, the
compiled custom bbox parser, and nvinfer's preprocessing. We get this by driving
the C app `ds_image_eval`, which decodes the canonical JPEGs DIRECTLY (no MJPEG
container, no re-encode) and writes one detection line per object. No
re-implementation of any architecture's decode -> no drift from the C++ parser.

This script is a thin driver:
  1. Run ds_image_eval <nvinfer_config> <eval_set/images> <N> <preds.txt> W H.
  2. Parse preds.txt: "frame_num class_id conf left top width height" (canvas
     pixel space) -> predictions_engine.json (xyxy).
  3. Score against ground_truth.json (already canvas space) via compute_map.

The canonical images and GT come from build_eval_set.py, which letterboxed both
into the same W x H canvas. eval_hf.py consumes the same images, so engine vs HF
differ only by the deployment path.
"""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path


def _parse_preds(preds_txt):
    """ds_image_eval output -> {image_id: [{bbox xyxy, score, label}]}."""
    by_id = {}
    for line in Path(preds_txt).read_text().splitlines():
        p = line.split()
        if len(p) != 7:
            continue
        frame, cls, conf, left, top, w, h = p
        image_id = int(frame)
        left, top, w, h = float(left), float(top), float(w), float(h)
        by_id.setdefault(image_id, []).append(
            {"bbox": [left, top, left + w, top + h], "score": float(conf), "label": int(cls)})
    return by_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--app", required=True, help="path to the compiled ds_image_eval binary")
    ap.add_argument("--engine-config", required=True, help="nvinfer config_infer_primary_*.txt")
    ap.add_argument("--eval-set", required=True)
    ap.add_argument("--labels", required=True, help="DEPLOYED model labels.txt (predicted ids index into this)")
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--perf-out", help="optional perf.json (pipeline fps/elapsed)")
    ap.add_argument("--preds-txt", help="raw ds_image_eval output (default: <work>/engine_preds.txt)")
    args = ap.parse_args()

    eval_set = Path(args.eval_set)
    gt = json.loads((eval_set / "ground_truth.json").read_text())
    W, H = gt.get("canvas", [1280, 720])
    n = gt["n_images"]
    # Deployed model's OWN label order (predicted class ids index into this).
    model_labels = [l.strip() for l in Path(args.labels).read_text().splitlines() if l.strip()]
    # TARGET (GT) label space from the eval set. Predictions are remapped into it
    # by NAME so original (e.g. COCO) and fine-tuned models score in one space.
    id2label = {int(k): v for k, v in gt.get("label_map", {}).items()}
    _norm = lambda s: re.sub(r"[^a-z0-9]", "", str(s).lower())
    target_norm = {_norm(v): k for k, v in id2label.items()}
    model_to_target = {i: target_norm.get(_norm(nm)) for i, nm in enumerate(model_labels)}

    preds_txt = Path(args.preds_txt) if args.preds_txt else Path(args.predictions).with_suffix(".raw.txt")

    # 1: run the real pipeline (compiled parser + nvinfer preprocessing).
    cmd = [args.app, str(Path(args.engine_config).resolve()),
           str((eval_set / "images").resolve()), str(n),
           str(preds_txt.resolve()), str(W), str(H)]
    print(f"[eval_engine] {' '.join(cmd)}")
    t0 = time.time()
    r = subprocess.run(cmd)
    elapsed = time.time() - t0
    if r.returncode != 0 or not preds_txt.exists():
        sys.exit(f"ERROR: ds_image_eval failed (exit {r.returncode}). "
                 "Confirm it is built (make -C scripts) and the engine config is valid.")

    # 2: parse predictions (model-label space), then remap into the TARGET space by
    # name. Drop predictions whose class isn't a target class. Fine-tuned model
    # (labels == target) -> identity; a model that can't output the target classes
    # maps to nothing -> ~0 mAP (the honest 'before').
    preds_by_id = _parse_preds(preds_txt)
    for i, dets in list(preds_by_id.items()):
        keep = []
        for d in dets:
            t = model_to_target.get(d["label"])
            if t is None:
                continue
            d["label"] = t
            keep.append(d)
        preds_by_id[i] = keep
    # Per-class detection counts (target space). Always reported, so categories the
    # model produced NOTHING for show up as "no object detected" instead of vanishing.
    from collections import Counter
    _dc = Counter(d["label"] for dets in preds_by_id.values() for d in dets)
    per_class_detections = {id2label.get(cid, str(cid)): int(_dc.get(cid, 0)) for cid in id2label}
    ids = [im["image_id"] for im in gt["images"]]
    predictions_out = [{"image_id": i, "detections": preds_by_id.get(i, [])} for i in ids]
    Path(args.predictions).parent.mkdir(parents=True, exist_ok=True)
    Path(args.predictions).write_text(json.dumps(
        {"engine_config": args.engine_config, "canvas": [W, H],
         "predictions": predictions_out}, indent=2))

    # 3: GT is already canvas-space; score via the shared core.
    import compute_map
    targets_by_id, _ = compute_map._load_gt(eval_set / "ground_truth.json")
    pred_lists = [{"boxes": [d["bbox"] for d in preds_by_id.get(i, [])],
                   "scores": [d["score"] for d in preds_by_id.get(i, [])],
                   "labels": [d["label"] for d in preds_by_id.get(i, [])]} for i in ids]
    result = compute_map.score(pred_lists, [targets_by_id[i] for i in ids], id2label)

    frames_with_dets = sum(1 for i in ids if preds_by_id.get(i))
    pipeline_fps = round(len(ids) / elapsed, 2) if elapsed > 0 else None
    result.update(source="engine", engine_config=args.engine_config, n_eval=len(ids),
                  frames_with_detections=frames_with_dets,
                  detection_rate=round(frames_with_dets / max(1, len(ids)), 4),
                  per_class_detections=per_class_detections, pipeline_fps=pipeline_fps)
    Path(args.metrics).write_text(json.dumps(result, indent=2))
    if args.perf_out:
        # Rough end-to-end pipeline FPS (includes nvinfer engine build on first run).
        # Authoritative engine throughput/latency comes from trtexec (engine build log).
        Path(args.perf_out).write_text(json.dumps(
            {"pipeline_fps": pipeline_fps, "elapsed_s": round(elapsed, 2),
             "n_frames": len(ids), "note": "end-to-end DS pipeline; trtexec qps is authoritative engine perf"}, indent=2))
    n_undet = sum(1 for v in per_class_detections.values() if v == 0)
    print(f"[eval_engine] map={result['map']:.4f} map_50={result['map_50']:.4f} "
          f"n={len(ids)} det_rate={result['detection_rate']:.0%} "
          f"no_detection_classes={n_undet}/{len(per_class_detections)}")


if __name__ == "__main__":
    main()
