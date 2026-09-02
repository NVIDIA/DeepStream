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

"""Convert a COCO annotations.json (+ images dir) into a HuggingFace Arrow dataset.

Bridges `ingest_dataset.py`'s COCO output to the `tao-finetune-huggingface-model` DETR recipe,
which expects `load_from_disk` dirs with columns: image, image_id, width, height,
and `objects` = {bbox (COCO xywh), category, area, id}. Boxes are kept in COCO
xywh (what the recipe's albumentations `format="coco"` expects).
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", required=True, help="COCO annotations.json")
    ap.add_argument("--images", required=True, help="image dir (file_name is relative to this)")
    ap.add_argument("--out", required=True, help="output dir for save_to_disk")
    args = ap.parse_args()

    from datasets import Dataset, Features, Image, Sequence, Value

    coco = json.loads(Path(args.coco).read_text())
    root = Path(args.images)

    # Remap COCO category ids -> contiguous 0-based labels, ordered by category id — the same
    # order ingest_dataset.py writes labels.txt in, and the order train.py's id2label expects.
    # Identity for already-0-indexed datasets (the demos); fixes 1-indexed COCO (e.g. HRIPCB),
    # where a raw label == num_classes would trigger a CUDA device-side assert in the loss.
    cats = sorted(coco.get("categories", []), key=lambda c: c["id"])
    cat_index = {c["id"]: i for i, c in enumerate(cats)}
    if cats and list(cat_index.values()) != list(cat_index.keys()):
        print(f"[coco_to_hf] remapping category ids {list(cat_index.keys())} -> 0..{len(cats) - 1}")

    by_img = defaultdict(list)
    for a in coco["annotations"]:
        by_img[a["image_id"]].append(a)

    records = []
    for im in coco["images"]:
        anns = by_img.get(im["id"], [])
        records.append({
            "image": str(root / im["file_name"]),
            "image_id": int(im["id"]),
            "width": int(im.get("width", 0)),
            "height": int(im.get("height", 0)),
            "objects": {
                "id": [int(a.get("id", i)) for i, a in enumerate(anns)],
                "category": [cat_index.get(int(a["category_id"]), int(a["category_id"])) for a in anns],
                "bbox": [[float(v) for v in a["bbox"]] for a in anns],   # COCO xywh
                "area": [float(a.get("area", a["bbox"][2] * a["bbox"][3])) for a in anns],
            },
        })

    features = Features({
        "image": Image(),
        "image_id": Value("int64"),
        "width": Value("int64"),
        "height": Value("int64"),
        "objects": Sequence({
            "id": Value("int64"),
            "category": Value("int64"),
            "bbox": Sequence(Value("float32"), length=4),
            "area": Value("float32"),
        }),
    })
    ds = Dataset.from_list(records, features=features)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    import os
    nproc = max(1, min(8, (os.cpu_count() or 2)))
    # The image bytes are read+encoded HERE (save_to_disk), not in the records loop above — this is
    # the slow part over a network/SMB mount. Parallelize across procs (latency-bound reads benefit
    # a lot) and bracket with [progress] markers so the footer shows the image count for this stage.
    print(f"[coco_to_hf] encoding {len(ds)} images from {root} into Arrow "
          f"(reads every image; parallelized across {nproc} procs — slow over a network mount)…", flush=True)
    print(f"[progress] 0/{len(ds)} images", flush=True)
    try:
        ds.save_to_disk(args.out, num_proc=nproc)
    except TypeError:                       # older datasets without num_proc
        ds.save_to_disk(args.out)
    print(f"[progress] {len(ds)}/{len(ds)} images", flush=True)
    print(f"[coco_to_hf] {len(ds)} rows -> {args.out} (cols: {ds.column_names})")


if __name__ == "__main__":
    main()
