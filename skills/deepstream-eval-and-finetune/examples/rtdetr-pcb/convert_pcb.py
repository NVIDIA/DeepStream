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

# Convert itsyoboieltr/pcb (HF custom schema) -> COCO json + images on disk + labels.txt.
#
# Source schema: row = {image, label:{name, bboxes:[{object_class:int, bbox:[cx,cy,w,h] normalized}]}}
# object_class mapping (verified by sampling filenames):
#   0=mouse_bite 1=spur 2=missing_hole 3=short 4=open_circuit 5=spurious_copper
# COCO output: category_id == object_class (0-based, matches tao-finetune-huggingface-model label_names index),
# bbox in absolute pixel xywh (top-left). Mirrors ingest_dataset.py's COCO + labels.txt layout.
import argparse, json
from pathlib import Path

CLASSES = ["mouse_bite", "spur", "missing_hole", "short", "open_circuit", "spurious_copper"]


def convert_split(ds, split, dest, hf_split_name):
    img_dir = dest / f"images_{split}"
    img_dir.mkdir(parents=True, exist_ok=True)
    images, annotations = [], []
    ann_id = 0
    n_box = 0
    for i, row in enumerate(ds):
        img = row["image"].convert("RGB")
        W, H = img.width, img.height
        fname = f"{i:06d}.jpg"
        img.save(img_dir / fname, quality=95)
        images.append({"id": i, "file_name": fname, "width": W, "height": H})
        for b in row["label"]["bboxes"]:
            c = int(b["object_class"])
            cx, cy, bw, bh = [float(v) for v in b["bbox"]]   # normalized cxcywh
            x = (cx - bw / 2.0) * W
            y = (cy - bh / 2.0) * H
            w = bw * W
            h = bh * H
            # clamp into image, drop degenerate
            x = max(0.0, min(x, W - 1.0)); y = max(0.0, min(y, H - 1.0))
            w = max(0.0, min(w, W - x));   h = max(0.0, min(h, H - y))
            if w < 1.0 or h < 1.0:
                continue
            annotations.append({"id": ann_id, "image_id": i, "category_id": c,
                                 "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0})
            ann_id += 1; n_box += 1
    coco = {"images": images,
            "annotations": annotations,
            "categories": [{"id": idx, "name": n} for idx, n in enumerate(CLASSES)]}
    out_json = dest / f"annotations_{split}.json"
    out_json.write_text(json.dumps(coco))
    print(f"[convert_pcb] {split}: {len(images)} imgs, {n_box} boxes -> {out_json}")
    return {"images_dir": str(img_dir), "annotations": str(out_json), "n_images": len(images)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="itsyoboieltr/pcb")
    ap.add_argument("--dest", required=True)
    ap.add_argument("--revision", default="main",
                    help="Hub revision (branch, tag, or commit SHA). Pin a SHA for reproducible builds.")
    args = ap.parse_args()
    from datasets import load_dataset

    dest = Path(args.dest); dest.mkdir(parents=True, exist_ok=True)
    (dest / "labels.txt").write_text("\n".join(CLASSES) + "\n")

    dsd = load_dataset(args.repo_id, revision=args.revision)
    print(f"[convert_pcb] splits available: {list(dsd.keys())}")
    # HF split names: train / validation / test
    name_map = {"train": "train", "valid": "validation", "test": "test"}
    manifest = {"repo_id": args.repo_id, "classes": CLASSES, "splits": {}}
    for local, hf_name in name_map.items():
        if hf_name not in dsd:
            continue
        manifest["splits"][local] = convert_split(dsd[hf_name], local, dest, hf_name)
    (dest / "ingest_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[convert_pcb] manifest -> {dest/'ingest_manifest.json'}")


if __name__ == "__main__":
    main()
