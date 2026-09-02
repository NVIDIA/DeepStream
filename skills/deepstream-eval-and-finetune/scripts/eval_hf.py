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

"""PyTorch reference eval — the deployment-drop denominator.

Stage 3 (HF leg) of the deepstream-eval-and-finetune skill. Runs the original
HuggingFace model in PyTorch over the SAME eval_set images that the deployed
TRT engine sees, so eval_engine vs eval_hf differ only by the deployment path
(FP16 quantisation + custom parser + nvinfer preprocessing).

Adapted from a HuggingFace detection eval example, but reads
the eval_set/ground_truth.json built by build_eval_set.py and emits predictions
in the shared schema consumed by compute_map.py. Labels are emitted in the model
label space (== labels.txt order, which the import skill derives from the model's
own config.id2label).
"""
import argparse
import json
from pathlib import Path


def main():
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="HF model id or local checkpoint dir")
    ap.add_argument("--eval-set", required=True, help="eval_set dir (has images/ + ground_truth.json)")
    ap.add_argument("--predictions", required=True, help="output predictions_hf.json")
    ap.add_argument("--metrics", required=True, help="output hf_metrics.json")
    ap.add_argument("--threshold", type=float, default=0.0,
                    help="score floor for post-processing (0.0 = let mAP integrate all)")
    ap.add_argument("--image-size", type=int, default=None,
                    help="force a square resize (shortest=longest=N) so the HF model sees the "
                         "SAME network input as the deployed engine (apples-to-apples drop)")
    ap.add_argument("--revision", default="main",
                    help="Hub revision (branch, tag, or commit SHA). Pin a SHA for reproducible builds.")
    args = ap.parse_args()

    eval_set = Path(args.eval_set)
    gt = json.loads((eval_set / "ground_truth.json").read_text())

    # Auth for gated repos is resolved by huggingface_hub from the environment
    # (HF_TOKEN) or the stored login — never read or forwarded explicitly here.
    ip_kwargs = {}
    if args.image_size:
        ip_kwargs.update(do_resize=True,
                         size={"shortest_edge": args.image_size, "longest_edge": args.image_size})
    ip = AutoImageProcessor.from_pretrained(args.checkpoint, revision=args.revision, **ip_kwargs)
    model = AutoModelForObjectDetection.from_pretrained(
        args.checkpoint, revision=args.revision).cuda().eval()

    predictions = []
    with torch.inference_mode():
        for img_rec in gt["images"]:
            img = Image.open(eval_set / img_rec["file"]).convert("RGB")
            inputs = ip(images=img, return_tensors="pt").to("cuda")
            outputs = model(**inputs)
            h, w = img_rec["height"], img_rec["width"]
            post = ip.post_process_object_detection(
                outputs, threshold=args.threshold,
                target_sizes=torch.tensor([[h, w]]).cuda())[0]
            dets = []
            for box, sc, lb in zip(post["boxes"].cpu().tolist(),
                                   post["scores"].cpu().tolist(),
                                   post["labels"].cpu().tolist()):
                dets.append({"bbox": [float(v) for v in box],  # xyxy pixel
                             "score": float(sc), "label": int(lb)})
            predictions.append({"image_id": img_rec["image_id"], "detections": dets})

    Path(args.predictions).parent.mkdir(parents=True, exist_ok=True)
    Path(args.predictions).write_text(json.dumps({"checkpoint": args.checkpoint,
                                                  "predictions": predictions}, indent=2))

    # Score via the shared core so the number matches eval_engine exactly.
    import compute_map
    targets_by_id, id2label = compute_map._load_gt(eval_set / "ground_truth.json")
    preds_by_id = compute_map._load_preds(args.predictions)
    ids = sorted(targets_by_id)
    result = compute_map.score(
        [preds_by_id.get(i, {"boxes": [], "scores": [], "labels": []}) for i in ids],
        [targets_by_id[i] for i in ids], id2label)
    result.update(source="hf", checkpoint=args.checkpoint, n_eval=len(ids))
    Path(args.metrics).write_text(json.dumps(result, indent=2))
    print(f"[eval_hf] map={result['map']:.4f} map_50={result['map_50']:.4f} n={len(ids)}")


if __name__ == "__main__":
    main()
