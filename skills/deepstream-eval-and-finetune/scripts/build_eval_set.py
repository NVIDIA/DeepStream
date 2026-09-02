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

"""Build a canonical eval set + validate the label space.

Stage 2 of the deepstream-eval-and-finetune skill. Sources, in priority order:
  --local-coco <annotations.json>   (e.g. produced by ingest_dataset.py)
  --dataset-id <hf id>  /  --local-arrow <load_from_disk dir>

For every image it:
  1. Letterboxes into a fixed W x H CANONICAL canvas (aspect preserved, grey pad)
     and transforms GT boxes into that same pixel space. Both eval legs
     (ds_image_eval and eval_hf.py) consume these canonical images, so the
     deployment-drop reflects only FP16/parser/preprocessing.
  2. Maps every dataset category to the MODEL label index (labels.txt) and FAILS
     LOUDLY when the spaces don't overlap — a near-zero mAP from mismatched
     labels is a labelling bug, not model quality. The silent index-identity
     fallback is gated behind --assume-index-identity.

Emits eval_set/images/NNNNNN.jpg + eval_set/ground_truth.json (categories already
in the model label space; boxes in canvas-pixel xywh).
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


def _norm(name):
    """Normalise a class name for tolerant matching (case/punct/space-insensitive)."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _load_labels(labels_txt):
    return [l.strip() for l in Path(labels_txt).read_text().splitlines() if l.strip()]


# --- Source loaders: return a normalized in-memory structure (no images held) ---
#   items: list of {"ref": ("hf", idx) | ("path", filepath),
#                   "width": int|None, "height": int|None,
#                   "objs": [(x, y, w, h, dataset_cat_id), ...]}   # COCO xywh pixels
#   cat_names: {dataset_cat_id: name} | None

def _load_coco(annotations_json, images_root):
    coco = json.loads(Path(annotations_json).read_text())
    root = Path(images_root) if images_root else Path(annotations_json).parent
    by_img = defaultdict(list)
    for a in coco["annotations"]:
        x, y, w, h = a["bbox"]
        by_img[a["image_id"]].append((float(x), float(y), float(w), float(h), int(a["category_id"])))
    items = []
    for im in coco["images"]:
        items.append({"ref": ("path", str(root / im["file_name"])),
                      "width": im.get("width"), "height": im.get("height"),
                      "objs": by_img.get(im["id"], [])})
    cat_names = {int(c["id"]): c["name"] for c in coco.get("categories", [])} or None
    return items, cat_names


def _hf_category_names(ds, objects_column):
    try:
        seq = ds.features[objects_column]
        inner = seq.feature["category"] if hasattr(seq, "feature") else seq["category"]
        names = getattr(inner, "names", None)
        if names:
            return {i: n for i, n in enumerate(names)}
    except Exception:
        pass
    return None


def _load_hf(ds, image_column, objects_column):
    # Read only the objects column (no image decode) to enumerate boxes/categories.
    obj_col = ds[objects_column]
    items = []
    for i, o in enumerate(obj_col):
        objs = [(float(b[0]), float(b[1]), float(b[2]), float(b[3]), int(c))
                for b, c in zip(o["bbox"], o["category"])]
        items.append({"ref": ("hf", i), "width": None, "height": None, "objs": objs})
    return items, _hf_category_names(ds, objects_column)


def _open_image(ref, ds, image_column):
    kind, val = ref
    if kind == "path":
        from PIL import Image
        return Image.open(val).convert("RGB")
    return ds[val][image_column].convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="model labels.txt from the import skill")
    ap.add_argument("--out", required=True, help="eval_set output dir")
    # sources (exactly one)
    ap.add_argument("--local-coco", help="COCO annotations.json (e.g. from ingest_dataset.py)")
    ap.add_argument("--coco-images-root", help="image dir for --local-coco (default: json's dir)")
    ap.add_argument("--dataset-id", help="HF dataset id (object detection)")
    ap.add_argument("--local-arrow", help="path passed to datasets.load_from_disk")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--image-column", default="image")
    ap.add_argument("--objects-column", default="objects")
    ap.add_argument("--category-map", help="optional JSON {dataset_cat_id_or_name: model_label_name}")
    ap.add_argument("--min-overlap", type=float, default=0.5,
                    help="fail if < this fraction of dataset categories map to model labels")
    ap.add_argument("--canvas-width", type=int, default=1280,
                    help="letterbox canvas width; MUST match ds_image_eval/streammux width")
    ap.add_argument("--canvas-height", type=int, default=720)
    ap.add_argument("--bbox-format", choices=["xywh", "xyxy"], default="xywh",
                    help="GT bbox format in the source. xywh = COCO standard (cppe-5, most). "
                         "xyxy = corner format (e.g. detection-datasets/coco) — converted to xywh.")
    ap.add_argument("--resize-mode", choices=["letterbox", "stretch"], default="letterbox",
                    help="letterbox (aspect-preserve+pad; YOLO-style) | stretch (resize to WxH ignoring "
                         "aspect; matches RT-DETR / square-resize processors). MUST match the model's "
                         "training preprocessing AND the nvinfer maintain-aspect-ratio setting.")
    ap.add_argument("--assume-index-identity", action="store_true",
                    help="when the dataset exposes no class names, assume dataset category id == "
                         "model label index. ONLY pass this when you have VERIFIED both sides share "
                         "the same ordering. Otherwise the skill fails loudly rather than guess.")
    ap.add_argument("--revision", default="main",
                    help="Hub revision (branch, tag, or commit SHA). Pin a SHA for reproducible builds.")
    args = ap.parse_args()

    model_labels = _load_labels(args.labels)
    model_norm = {_norm(n): i for i, n in enumerate(model_labels)}
    print(f"[build_eval_set] model label space ({len(model_labels)}): {model_labels[:10]}"
          f"{' ...' if len(model_labels) > 10 else ''}")

    # --- Load the source into normalized items + cat_names ----------------
    ds = None
    if args.local_coco:
        items, cat_names = _load_coco(args.local_coco, args.coco_images_root)
    elif args.local_arrow or args.dataset_id:
        from datasets import load_dataset, load_from_disk
        if args.local_arrow:
            ds = load_from_disk(args.local_arrow)
            if hasattr(ds, "keys"):
                ds = ds[args.split]
        else:
            # Auth for gated datasets is resolved by huggingface_hub from the
            # environment (HF_TOKEN) or the stored login.
            ds = load_dataset(args.dataset_id, split=f"{args.split}[:{args.n_eval}]",
                              revision=args.revision)
        items, cat_names = _load_hf(ds, args.image_column, args.objects_column)
    else:
        sys.exit("ERROR: pass --local-coco, --dataset-id, or --local-arrow")

    if len(items) > args.n_eval:
        items = items[:args.n_eval]

    # Normalize bbox to COCO xywh. Most detection datasets are xywh, but some
    # (e.g. detection-datasets/coco) store xyxy — converting wrong would put
    # x2,y2 where w,h belong and tank mAP for ANY model.
    if args.bbox_format == "xyxy":
        for it in items:
            it["objs"] = [(x, y, x2 - x, y2 - y, c) for (x, y, x2, y2, c) in it["objs"]]

    # --- Resolve dataset category -> model label index (loud guard) -------
    user_map = json.loads(Path(args.category_map).read_text()) if args.category_map else None

    def resolve(cat_id):
        if user_map:
            target = user_map.get(str(cat_id))
            if target is None and cat_names:
                target = user_map.get(cat_names.get(cat_id, ""))
            if target is not None:
                return model_norm.get(_norm(target)), target
        if cat_names and cat_id in cat_names:
            name = cat_names[cat_id]
            return model_norm.get(_norm(name)), name
        if args.assume_index_identity and cat_id < len(model_labels):
            return cat_id, f"<identity:{model_labels[cat_id]}>"
        return None, str(cat_id)

    present_cats = {o[4] for it in items for o in it["objs"]}
    cat_to_model, mapped_names, unmatched = {}, {}, []
    for cid in sorted(present_cats):
        idx, name = resolve(cid)
        if idx is None:
            unmatched.append(name)
        else:
            cat_to_model[cid] = idx
            mapped_names[name] = model_labels[idx]

    overlap = len(cat_to_model) / max(1, len(present_cats))
    # Informational only (NOT a router, NOT a blocker). The skill ALWAYS builds the
    # eval set and ALWAYS runs DeepStream; KPI categories the model cannot name simply
    # score 0 and are reported per-class as "no object detected" — never special-cased.
    print(f"[build_eval_set] mapped {len(cat_to_model)}/{len(present_cats)} categories "
          f"(overlap={overlap:.0%}); unmatched={unmatched[:10]}")
    if overlap < args.min_overlap:
        print("[build_eval_set] NOTE: the model's labels overlap few KPI classes — the baseline "
              "will show low/zero AP on the unmatched classes (reported per-category as "
              "'no object detected'); fine-tuning is what closes that gap.")

    # --- Materialise CANONICAL letterboxed images + GT (canvas space) -----
    from PIL import Image

    out = Path(args.out)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    W, H = args.canvas_width, args.canvas_height

    images = []
    print(f"[build_eval_set] materialising {len(items)} canonical {W}x{H} images "
          f"(reads + resizes each source image)…", flush=True)
    for i, it in enumerate(items):
        if i % 25 == 0 or i == len(items) - 1:
            print(f"[progress] {i + 1}/{len(items)} images", flush=True)  # footer progress
        img = _open_image(it["ref"], ds, args.image_column)
        ow, oh = img.width, img.height
        if args.resize_mode == "stretch":
            # Resize ignoring aspect (match RT-DETR / square-resize processors).
            sx, sy, px, py = W / ow, H / oh, 0.0, 0.0
            canvas = img.resize((W, H))
        else:
            # Letterbox: aspect-preserve + grey pad (match YOLO-style preprocessing).
            sx = sy = min(W / ow, H / oh)
            nw, nh = max(1, round(ow * sx)), max(1, round(oh * sy))
            px, py = (W - nw) / 2.0, (H - nh) / 2.0
            canvas = Image.new("RGB", (W, H), (114, 114, 114))
            canvas.paste(img.resize((nw, nh)), (round(px), round(py)))
        fname = f"images/{i:06d}.jpg"
        canvas.save(out / fname, quality=95)

        objs = []
        for (x, y, bw, bh, c) in it["objs"]:
            if c not in cat_to_model:
                continue
            objs.append({"bbox": [x * sx + px, y * sy + py, bw * sx, bh * sy],
                         "category": cat_to_model[c]})
        images.append({"image_id": i, "file": fname, "width": W, "height": H,
                       "orig_size": [ow, oh], "objects": objs})

    ground_truth = {
        "label_map": {str(i): n for i, n in enumerate(model_labels)},
        "category_map": {str(k): v for k, v in cat_to_model.items()},
        "canvas": [W, H],
        "resize_mode": args.resize_mode,
        "overlap": round(overlap, 4),
        "n_images": len(images),
        "images": images,
    }
    (out / "ground_truth.json").write_text(json.dumps(ground_truth, indent=2))
    n_boxes = sum(len(im["objects"]) for im in images)
    print(f"[build_eval_set] wrote {len(images)} canonical {W}x{H} images, "
          f"{n_boxes} boxes -> {out/'ground_truth.json'}")


if __name__ == "__main__":
    main()
