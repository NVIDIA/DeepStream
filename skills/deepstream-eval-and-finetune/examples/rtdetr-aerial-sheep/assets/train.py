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

"""DETR fine-tune on CPPE-5 (adapted from HF repo run_object_detection.py)."""
import argparse, os
from functools import partial
from pathlib import Path
from typing import Any

import albumentations as A
import numpy as np
import torch, yaml
from datasets import load_from_disk
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from transformers import (
    AutoImageProcessor, AutoModelForObjectDetection,
    Trainer, TrainingArguments,
)


def format_image_annotations_as_coco(image_id, categories, areas, bboxes):
    """Format (image_id, categories, areas, bboxes) as a COCO annotation dict for DETR preprocessing."""
    annotations = [
        {"image_id": image_id, "category_id": cat, "iscrowd": 0, "area": area,
         "bbox": list(bbox)}  # expected: [x, y, w, h]
        for cat, area, bbox in zip(categories, areas, bboxes)
    ]
    return {"image_id": image_id, "annotations": annotations}


def augment_and_transform_batch(examples, transform, image_processor):
    pixel_values, labels = [], []
    for img_id, img, objs in zip(examples["image_id"], examples["image"], examples["objects"]):
        image = np.array(img.convert("RGB"))
        H, W = image.shape[:2]
        fb, fc = [], []   # pre-filter degenerate boxes (albumentations 1.4.x has no filter_invalid_bboxes)
        for (x, y, w, h), c in zip(objs["bbox"], objs["category"]):
            if w <= 0 or h <= 0 or x >= W or y >= H:
                continue
            x = max(0.0, min(float(x), W - 1.0)); y = max(0.0, min(float(y), H - 1.0))
            w = min(float(w), W - x); h = min(float(h), H - y)
            if w < 1.0 or h < 1.0:
                continue
            fb.append([x, y, w, h]); fc.append(c)
        out = transform(image=image, bboxes=fb, category_ids=fc)
        formatted = format_image_annotations_as_coco(
            img_id, out["category_ids"], [b[2] * b[3] for b in out["bboxes"]], out["bboxes"])
        encoded = image_processor(images=out["image"], annotations=formatted, return_tensors="pt")
        pixel_values.append(encoded["pixel_values"][0])
        labels.append(encoded["labels"][0])
    return {"pixel_values": pixel_values, "labels": labels}


def make_collate_fn(image_processor):
    """Pad a batch to a common canvas and emit the pixel mask alongside it.

    Written against torch rather than ImageProcessor.pad(): that method took the whole batch
    plus return_tensors in transformers 4.x, but pads a single image in 5.x.
    """
    def collate_fn(batch):
        pixel_values = [torch.as_tensor(b["pixel_values"]) for b in batch]
        canvas_h = max(px.shape[-2] for px in pixel_values)
        canvas_w = max(px.shape[-1] for px in pixel_values)
        padded, pixel_mask = [], []
        for px in pixel_values:
            height, width = px.shape[-2], px.shape[-1]
            padded.append(torch.nn.functional.pad(px, (0, canvas_w - width, 0, canvas_h - height)))
            keep = torch.zeros((canvas_h, canvas_w), dtype=torch.long)
            keep[:height, :width] = 1
            pixel_mask.append(keep)
        labels = [{k: torch.as_tensor(v) for k, v in b["labels"].items()} for b in batch]
        return {"pixel_values": torch.stack(padded), "pixel_mask": torch.stack(pixel_mask),
                "labels": labels}
    return collate_fn


@torch.no_grad()
def _pick_logits_boxes(predictions, ncls):
    """Robustly pull (logits, pred_boxes) out of RT-DETR's prediction tuple (12+ tensors: also
    last_hidden_state, init_reference, enc-topk logits/boxes, 8400-anchor outputs, 4-D intermediates,
    and dict/list aux outputs). logits & pred_boxes are the FIRST detection fields, so take the
    EARLIEST 3-D tensor whose last dim == num_classes (logits) and == 4 (boxes). Keying on ncls
    handles any class count incl. 1 (where last-dim 1 broke the old `>4` heuristic)."""
    preds = predictions if isinstance(predictions, (tuple, list)) else (predictions,)
    logits = boxes = None
    for p in preds:
        if not hasattr(p, "ndim") or p.ndim != 3:   # skip dicts/lists/None and 4-D intermediates
            continue
        if logits is None and p.shape[-1] == ncls:
            logits = p
        elif boxes is None and p.shape[-1] == 4:
            boxes = p
        if logits is not None and boxes is not None:
            break
    return logits, boxes


def preprocess_logits_for_metrics(predictions, labels, ncls):
    """Trainer hook: keep ONLY (logits, pred_boxes) from the model output before they're gathered for
    metrics — removes ambiguity for compute_metrics AND avoids accumulating the big/irrelevant tensors
    (256-dim hidden states, 8400-anchor outputs) over the eval set."""
    lg, bx = _pick_logits_boxes(predictions, ncls)
    return (lg, bx) if (lg is not None and bx is not None) else predictions


def compute_metrics(eval_pred, image_processor, id2label, threshold=0.0):
    """Replicate HF repo compute_metrics: post-process + torchmetrics MeanAveragePrecision.
    Logged each epoch by the Trainer (prefixed 'eval_') so the UI can show a live mAP curve:
    -> eval_map, eval_map_50, eval_map_75 (overall) and eval_cls_<name> (per-class AP).
    Wrapped in try/except: a metric hiccup must NEVER kill a training run (returns {})."""
    try:
        predictions, targets = eval_pred.predictions, eval_pred.label_ids
        ncls = len(id2label)
        # With eval_do_concat_batches=False the Trainer hands predictions & label_ids as LISTS over
        # eval batches (object-detection labels are variable-size dicts that can't be concatenated).
        # Process PER BATCH so per-image counts always match (this is the canonical HF OD pattern;
        # the earlier "target sizes != logits batch" error came from forcing a flat layout).
        pred_batches = predictions if isinstance(predictions, list) else [predictions]
        tgt_batches = targets if isinstance(targets, list) else [targets]

        def _img_iter(b):   # a batch's targets may be a list of dicts, or a dict of arrays
            if isinstance(b, dict):
                n = len(b["orig_size"])
                return [{k: v[i] for k, v in b.items()} for i in range(n)]
            return list(b)

        post_targets, sizes_per_batch = [], []
        for batch in tgt_batches:
            sizes = []
            for t in _img_iter(batch):
                h, w = int(t["orig_size"][0]), int(t["orig_size"][1])
                sizes.append([h, w])
                tb = torch.as_tensor(t["boxes"], dtype=torch.float32).reshape(-1, 4)
                if tb.numel():
                    cx, cy, bw, bh = tb.unbind(-1)   # [cx,cy,w,h] normalized -> xyxy pixels
                    xyxy = torch.stack([(cx - bw / 2) * w, (cy - bh / 2) * h,
                                        (cx + bw / 2) * w, (cy + bh / 2) * h], dim=-1)
                else:
                    xyxy = torch.zeros((0, 4))
                post_targets.append({"boxes": xyxy,
                                     "labels": torch.as_tensor(t["class_labels"]).reshape(-1).long()})
            sizes_per_batch.append(torch.tensor(sizes) if sizes else torch.zeros((0, 2), dtype=torch.long))

        post_preds = []
        for batch, sizes in zip(pred_batches, sizes_per_batch):
            logits, boxes = _pick_logits_boxes(batch, ncls)
            if logits is None or boxes is None or sizes.numel() == 0:
                continue
            outputs = type("O", (), {"logits": torch.as_tensor(logits), "pred_boxes": torch.as_tensor(boxes)})()
            post_preds.extend(image_processor.post_process_object_detection(
                outputs, threshold=threshold, target_sizes=sizes))

        if not post_preds or not post_targets:
            return {"map": 0.0}
        metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
        metric.update(post_preds, post_targets)
        m = metric.compute()
        out = {"map": float(m["map"].item()), "map_50": float(m["map_50"].item()),
               "map_75": float(m["map_75"].item())}
        # per-class AP -> 'cls_<name>' (unambiguous vs map_50/map_75; logged as eval_cls_<name>).
        # torch.atleast_1d: for a SINGLE class torchmetrics returns scalar classes/map_per_class, so
        # .tolist() would give an int and zip() would throw — keep them 1-D. Guarded so a per-class
        # hiccup never drops the overall map.
        try:
            if "classes" in m and "map_per_class" in m:
                classes = torch.atleast_1d(m["classes"]).tolist()
                aps = torch.atleast_1d(m["map_per_class"]).tolist()
                for cls_i, ap in zip(classes, aps):
                    out[f"cls_{id2label.get(int(cls_i), cls_i)}"] = float(ap)
        except Exception as e:
            print(f"[compute_metrics] per-class AP skipped (non-fatal): {e}")
        return out
    except Exception as e:
        print(f"[compute_metrics] skipped (non-fatal): {e}")
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max_steps", type=int, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    # Auth for gated repos is resolved by huggingface_hub from the environment
    # (HF_TOKEN) or the stored login — never read or forwarded explicitly here.

    data_root = cfg.get("local_data_dir", "data")
    ds_tr = load_from_disk(os.path.join(data_root, "train"))
    ds_ev = load_from_disk(os.path.join(data_root, "eval"))
    label_names = cfg["label_names"]
    id2label = {i: n for i, n in enumerate(label_names)}
    label2id = {n: i for i, n in id2label.items()}

    # Pin the Hub revision so a repo update cannot silently change the weights mid-project.
    revision = cfg.get("revision", "main")
    ip = AutoImageProcessor.from_pretrained(cfg["model_id"], revision=revision, do_resize=True,
                                            size={"height": 640, "width": 640})

    # Albumentations transforms (COCO format bboxes); filter_invalid_bboxes drops zero-area
    # boxes that clipping can collapse, which CPPE-5 has a handful of.
    train_tx = A.Compose([
        A.Perspective(p=0.1),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.1),
    ], bbox_params=A.BboxParams(format="coco", label_fields=["category_ids"], clip=True,
                                min_area=1))
    eval_tx = A.Compose([A.NoOp()],
        bbox_params=A.BboxParams(format="coco", label_fields=["category_ids"], clip=True,
                                 min_area=1))

    ds_tr = ds_tr.with_transform(partial(augment_and_transform_batch,
                                          transform=train_tx, image_processor=ip))
    ds_ev = ds_ev.with_transform(partial(augment_and_transform_batch,
                                          transform=eval_tx, image_processor=ip))

    model = AutoModelForObjectDetection.from_pretrained(
        cfg["model_id"], num_labels=len(label_names),
        id2label=id2label, label2id=label2id,
        ignore_mismatched_sizes=cfg.get("ignore_mismatched_sizes", True),
        revision=revision,
    )

    os.environ.setdefault("WANDB_PROJECT", "tao-finetune-huggingface-model-5tasks")
    if args.smoke: os.environ["WANDB_MODE"] = "disabled"

    kw = dict(
        output_dir=cfg["output_dir"], remove_unused_columns=cfg.get("remove_unused_columns", False),
        eval_strategy=cfg.get("eval_strategy", "epoch"),
        save_strategy=cfg.get("save_strategy", "epoch"),
        save_total_limit=cfg.get("save_total_limit", 1),
        learning_rate=cfg["learning_rate"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        eval_accumulation_steps=cfg.get("eval_accumulation_steps", 8),  # offload eval preds periodically (OOM guard for per-epoch mAP)
        eval_do_concat_batches=False,  # object-detection labels are lists of variable-size dicts — keep per-batch (canonical HF object-detection pattern), so compute_metrics gets matched preds/targets
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        num_train_epochs=cfg["num_train_epochs"],
        warmup_ratio=cfg.get("warmup_ratio", 0.1),
        weight_decay=cfg.get("weight_decay", 1e-4),
        bf16=cfg.get("bf16", True),
        dataloader_num_workers=cfg.get("dataloader_num_workers", 4),
        load_best_model_at_end=cfg.get("load_best_model_at_end", True),
        metric_for_best_model=cfg.get("metric_for_best_model", "eval_map"),
        greater_is_better=cfg.get("greater_is_better", True),
        report_to=("none" if args.smoke else cfg.get("report_to", "wandb")),
        run_name=cfg.get("model_short_name", "run"),
        logging_steps=cfg.get("logging_steps", 10),
        logging_first_step=cfg.get("logging_first_step", True),
        logging_strategy=cfg.get("logging_strategy", "steps"),
        disable_tqdm=cfg.get("disable_tqdm", True),
        push_to_hub=False,
        label_names=["labels"],
    )
    if args.max_steps is not None:
        kw["max_steps"] = args.max_steps
        kw["eval_strategy"] = "no"; kw["save_strategy"] = "no"; kw["load_best_model_at_end"] = False

    # Per-epoch mAP: compute_metrics logs eval_map/eval_map_50/eval_cls_<name> each epoch for the
    # live UI curve (a PyTorch trend proxy, NOT the deployed DeepStream mAP). Checkpoint SELECTION is
    # unchanged — generated configs keep load_best_model_at_end=false + metric_for_best_model=eval_loss,
    # so we still deploy the most-trained checkpoint; these metrics are informational only.
    # Disabled in --smoke (max_steps sets eval_strategy="no", so eval/metrics don't run anyway).
    trainer = Trainer(
        model=model, args=TrainingArguments(**kw),
        data_collator=make_collate_fn(ip),
        train_dataset=ds_tr, eval_dataset=ds_ev,
        processing_class=ip,
        compute_metrics=partial(compute_metrics, image_processor=ip, id2label=id2label),
        preprocess_logits_for_metrics=partial(preprocess_logits_for_metrics, ncls=len(id2label)),
    )
    # Warm-resume: continue from a prior checkpoint (optimizer + LR + epoch state) so a run
    # can be extended to more epochs without retraining from scratch.
    resume = cfg.get("resume_from_checkpoint")
    if resume:
        print(f"[train] resuming from {resume}")
        trainer.train(resume_from_checkpoint=resume)
    else:
        trainer.train()

    if not args.smoke:
        final = Path(cfg["output_dir"]) / "final"
        trainer.save_model(str(final)); ip.save_pretrained(str(final))
        print(f"[train] final checkpoint -> {final}")


if __name__ == "__main__":
    main()
