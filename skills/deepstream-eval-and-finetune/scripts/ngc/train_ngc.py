# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Persistent RT-DETR object-detection fine-tune for the tao NGC PyTorch container.
# Reads a local COCO json + images dir with a plain torch Dataset (NO HF `datasets` lib, so
# it installs cleanly in the NGC image). Per-epoch eval is ALWAYS on: each epoch logs
# eval_loss + eval_map + eval_map_50 + per-class eval_cls_<name>, so the report's
# fine-tuning curves (training-loss, eval-loss, per-epoch mAP) always populate.
# Deployed mAP is still measured separately in DeepStream.
import argparse, json, os
from collections import defaultdict
from functools import partial
from pathlib import Path

import albumentations as A
import numpy as np
import torch, yaml
from PIL import Image
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from transformers import AutoImageProcessor, AutoModelForObjectDetection, Trainer, TrainingArguments


class CocoDetDataset(torch.utils.data.Dataset):
    def __init__(self, coco_json, images_dir, transform, image_processor):
        c = json.load(open(coco_json))
        self.imgs = {im["id"]: im for im in c["images"]}
        self.by_img = defaultdict(list)
        for a in c["annotations"]:
            self.by_img[a["image_id"]].append(a)
        self.ids = list(self.imgs.keys())
        self.dir = images_dir; self.tf = transform; self.ip = image_processor

    def __len__(self): return len(self.ids)

    def __getitem__(self, i):
        iid = self.ids[i]; meta = self.imgs[iid]
        img = np.array(Image.open(os.path.join(self.dir, meta["file_name"])).convert("RGB"))
        H, W = img.shape[:2]
        bboxes, cats = [], []
        for a in self.by_img[iid]:
            x, y, w, h = [float(v) for v in a["bbox"]]
            if w <= 0 or h <= 0 or x >= W or y >= H:
                continue
            x = max(0.0, min(x, W - 1.0)); y = max(0.0, min(y, H - 1.0))
            w = min(w, W - x); h = min(h, H - y)
            if w < 1.0 or h < 1.0:
                continue
            bboxes.append([x, y, w, h]); cats.append(int(a["category_id"]))
        out = self.tf(image=img, bboxes=bboxes, category_ids=cats)
        anns = [{"image_id": iid, "category_id": c, "iscrowd": 0, "area": b[2] * b[3], "bbox": list(b)}
                for c, b in zip(out["category_ids"], out["bboxes"])]
        enc = self.ip(images=out["image"], annotations={"image_id": iid, "annotations": anns}, return_tensors="pt")
        return {"pixel_values": enc["pixel_values"][0], "labels": enc["labels"][0]}


def make_collate_fn(ip):
    # transformers 5.x changed ImageProcessor.pad() from a batch operation
    # (pad(list, return_tensors="pt") -> pixel_values + pixel_mask) to a single-image one
    # (pad(image, padded_size, ...)). Batching and the mask are built here instead, which
    # also removes the dependency on that API entirely.
    def collate(batch):
        pv = [torch.as_tensor(b["pixel_values"]) for b in batch]
        mh, mw = max(t.shape[-2] for t in pv), max(t.shape[-1] for t in pv)
        stacked, masks = [], []
        for t in pv:
            h, w = t.shape[-2], t.shape[-1]
            stacked.append(torch.nn.functional.pad(t, (0, mw - w, 0, mh - h)))
            mask = torch.zeros((mh, mw), dtype=torch.long)
            mask[:h, :w] = 1
            masks.append(mask)
        labels = [{k: torch.as_tensor(v) for k, v in b["labels"].items()} for b in batch]
        return {"pixel_values": torch.stack(stacked), "pixel_mask": torch.stack(masks),
                "labels": labels}
    return collate


def _pick_logits_boxes(predictions, ncls):
    preds = predictions if isinstance(predictions, (tuple, list)) else (predictions,)
    logits = boxes = None
    for p in preds:
        if not hasattr(p, "ndim") or p.ndim != 3:
            continue
        if logits is None and p.shape[-1] == ncls:
            logits = p
        elif boxes is None and p.shape[-1] == 4:
            boxes = p
        if logits is not None and boxes is not None:
            break
    return logits, boxes


def preprocess_logits_for_metrics(predictions, labels, ncls):
    lg, bx = _pick_logits_boxes(predictions, ncls)
    return (lg, bx) if (lg is not None and bx is not None) else predictions


def compute_metrics(eval_pred, image_processor, id2label, threshold=0.0):
    try:
        predictions, targets = eval_pred.predictions, eval_pred.label_ids
        ncls = len(id2label)
        pred_batches = predictions if isinstance(predictions, list) else [predictions]
        tgt_batches = targets if isinstance(targets, list) else [targets]

        def _img_iter(b):
            if isinstance(b, dict):
                n = len(b["orig_size"]); return [{k: v[i] for k, v in b.items()} for i in range(n)]
            return list(b)

        post_targets, sizes_per_batch = [], []
        for batch in tgt_batches:
            sizes = []
            for t in _img_iter(batch):
                h, w = int(t["orig_size"][0]), int(t["orig_size"][1]); sizes.append([h, w])
                tb = torch.as_tensor(t["boxes"], dtype=torch.float32).reshape(-1, 4)
                if tb.numel():
                    cx, cy, bw, bh = tb.unbind(-1)
                    xyxy = torch.stack([(cx - bw/2)*w, (cy - bh/2)*h, (cx + bw/2)*w, (cy + bh/2)*h], dim=-1)
                else:
                    xyxy = torch.zeros((0, 4))
                post_targets.append({"boxes": xyxy, "labels": torch.as_tensor(t["class_labels"]).reshape(-1).long()})
            sizes_per_batch.append(torch.tensor(sizes) if sizes else torch.zeros((0, 2), dtype=torch.long))

        post_preds = []
        for batch, sizes in zip(pred_batches, sizes_per_batch):
            logits, boxes = _pick_logits_boxes(batch, ncls)
            if logits is None or boxes is None or sizes.numel() == 0:
                continue
            outs = type("O", (), {"logits": torch.as_tensor(logits), "pred_boxes": torch.as_tensor(boxes)})()
            post_preds.extend(image_processor.post_process_object_detection(outs, threshold=threshold, target_sizes=sizes))

        if not post_preds or not post_targets:
            return {"map": 0.0}
        metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
        metric.update(post_preds, post_targets); m = metric.compute()
        out = {"map": float(m["map"]), "map_50": float(m["map_50"]), "map_75": float(m["map_75"])}
        try:
            if "classes" in m and "map_per_class" in m:
                for ci, ap in zip(torch.atleast_1d(m["classes"]).tolist(), torch.atleast_1d(m["map_per_class"]).tolist()):
                    out[f"cls_{id2label.get(int(ci), ci)}"] = float(ap)
        except Exception as e:
            print(f"[train_ngc] per-class AP skipped: {e}", flush=True)
        return out
    except Exception as e:
        print(f"[train_ngc] compute_metrics skipped (non-fatal): {e}", flush=True)
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--train-coco", required=True)
    ap.add_argument("--eval-coco", default=None)
    ap.add_argument("--images", required=True)
    ap.add_argument("--max-steps", type=int, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    # Auth for gated repos is resolved by huggingface_hub from the environment
    # (HF_TOKEN) or the stored login — never read or forwarded explicitly here.

    label_names = cfg["label_names"]
    id2label = {i: n for i, n in enumerate(label_names)}
    label2id = {n: i for i, n in id2label.items()}

    # Pin the Hub revision so a repo update cannot silently change the weights mid-project.
    revision = cfg.get("revision", "main")
    ip = AutoImageProcessor.from_pretrained(cfg["model_id"], revision=revision, do_resize=True,
                                            size={"height": 640, "width": 640})
    train_tx = A.Compose([A.HorizontalFlip(p=0.5), A.RandomBrightnessContrast(p=0.5), A.HueSaturationValue(p=0.1)],
                         bbox_params=A.BboxParams(format="coco", label_fields=["category_ids"], clip=True, min_area=1))
    eval_tx = A.Compose([A.NoOp()], bbox_params=A.BboxParams(format="coco", label_fields=["category_ids"], clip=True, min_area=1))

    ds_tr = CocoDetDataset(args.train_coco, args.images, train_tx, ip)
    ds_ev = CocoDetDataset(args.eval_coco, args.images, eval_tx, ip) if args.eval_coco else None
    print(f"[train_ngc] train {len(ds_tr)} imgs, eval {len(ds_ev) if ds_ev else 0} imgs, {len(label_names)} classes {label_names}", flush=True)

    model = AutoModelForObjectDetection.from_pretrained(
        cfg["model_id"], num_labels=len(label_names), id2label=id2label, label2id=label2id,
        ignore_mismatched_sizes=cfg.get("ignore_mismatched_sizes", True), revision=revision)

    eval_on = ds_ev is not None and args.max_steps is None
    kw = dict(
        output_dir=cfg["output_dir"], remove_unused_columns=False,
        eval_strategy=("epoch" if eval_on else "no"), save_strategy=cfg.get("save_strategy", "epoch"),
        save_total_limit=cfg.get("save_total_limit", 1),
        eval_do_concat_batches=False, eval_accumulation_steps=8,
        learning_rate=cfg["learning_rate"], per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg.get("per_device_eval_batch_size", 8),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 1), num_train_epochs=cfg["num_train_epochs"],
        warmup_ratio=cfg.get("warmup_ratio", 0.1), weight_decay=cfg.get("weight_decay", 1e-4),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "linear"), bf16=cfg.get("bf16", True),
        dataloader_num_workers=cfg.get("dataloader_num_workers", 8), load_best_model_at_end=False,
        report_to="none", run_name=cfg.get("model_short_name", "run"), logging_steps=cfg.get("logging_steps", 50),
        logging_first_step=True, disable_tqdm=cfg.get("disable_tqdm", True), push_to_hub=False, label_names=["labels"])
    if args.max_steps is not None:
        kw["max_steps"] = args.max_steps
        if args.eval_coco:
            kw["eval_strategy"] = "steps"; kw["eval_steps"] = max(1, args.max_steps // 2)
        kw["save_strategy"] = "no"

    trainer = Trainer(model=model, args=TrainingArguments(**kw), data_collator=make_collate_fn(ip),
                      train_dataset=ds_tr, eval_dataset=ds_ev, processing_class=ip,
                      compute_metrics=partial(compute_metrics, image_processor=ip, id2label=id2label),
                      preprocess_logits_for_metrics=partial(preprocess_logits_for_metrics, ncls=len(label_names)))
    trainer.train()

    if args.max_steps is None:
        final = Path(cfg["output_dir"]) / "final"
        trainer.save_model(str(final)); ip.save_pretrained(str(final))
        print(f"[train_ngc] final checkpoint -> {final}", flush=True)


if __name__ == "__main__":
    main()
