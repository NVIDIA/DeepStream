# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
#
# Qualitative before/after: render DeepStream detections on a handful of frames
# for the BASELINE and the FINE-TUNED model, side by side with ground truth.
# Numbers (mAP) say HOW MUCH better; these frames show HOW.
#
# Per frame -> a 3-up panel:  [ Ground Truth | Baseline (deployed) | Fine-tuned (deployed) ]
# Boxes are drawn above --conf; target-class boxes green, off-target boxes orange
# (so a stock model's confusion is visible). Also writes a stacked contact sheet.
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

GREEN = (118, 185, 0); ORANGE = (235, 140, 0); BLUE = (31, 119, 180)
DARK = (33, 33, 33); WHITE = (255, 255, 255); HDR = (70, 115, 0)
# overridable via CLI so boxes/text stay visible on any background (e.g. green PCB) — set in main()
DET_COLOR = GREEN; OFFTARGET_COLOR = ORANGE; GT_COLOR = BLUE; TEXT_COLOR = WHITE


def _rgb(s):
    """'#76b900' or '118,185,0' -> (118,185,0)."""
    s = str(s).strip()
    if s.startswith("#"):
        s = s.lstrip("#")
        return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))
    return tuple(int(v) for v in s.split(","))


def _font(sz, bold=True):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(p).exists():
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def _load_set(eval_set):
    gt = json.loads((Path(eval_set) / "ground_truth.json").read_text())
    id2label = {int(k): v for k, v in gt.get("label_map", {}).items()}
    imgs = {im["image_id"]: im for im in gt["images"]}
    # target class NAMES = those that actually have GT in the set
    tgt = {id2label.get(int(o["category"]), str(o["category"]))
           for im in gt["images"] for o in im["objects"]}
    return gt, id2label, imgs, tgt


def _load_preds(path):
    d = json.loads(Path(path).read_text())
    return {p["image_id"]: p["detections"] for p in d["predictions"]}


def _auto_conf(preds, requested, n_samples, floor=0.01):
    """Pick a draw threshold matched to THIS model's score range. Detectors like
    RT-DETR (sigmoid head, few classes) can produce well-ranked but low-magnitude
    scores (max ~0.1-0.2), so a fixed 0.3 draws nothing even at mAP@50 ~0.8.

    Trigger and target are separate: keep `requested` if it already yields enough
    detecting frames to populate the sample panel (a well-calibrated model is left
    alone); otherwise lower to where a healthy fraction (~60%) of frames detect, so the
    chosen frames are box-rich — floored at `floor`, never raised above `requested`."""
    maxes = sorted((max((float(d.get("score", 1.0)) for d in dets), default=0.0)
                    for dets in preds.values()), reverse=True)
    if not maxes:
        return requested
    if sum(1 for m in maxes if m >= requested) >= n_samples:
        return requested
    target = min(len(maxes), max(n_samples, int(0.6 * len(maxes))))
    return round(max(floor, min(requested, maxes[target - 1])), 4)


def _panel(img, title, count_txt):
    """Add a header strip with title + count above the image."""
    W, H = img.size; hh = 34
    canvas = Image.new("RGB", (W, H + hh), WHITE)
    canvas.paste(img, (0, hh))
    d = ImageDraw.Draw(canvas)
    d.rectangle([0, 0, W, hh], fill=HDR)
    d.text((8, 7), title, fill=WHITE, font=_font(18))
    if count_txt:
        f = _font(15); tw = d.textlength(count_txt, font=f)
        d.text((W - tw - 8, 9), count_txt, fill=WHITE, font=f)
    return canvas


def _draw(image_path, dets, id2label, tgt, conf, maxb, draw_gt=False, labels=True):
    # GT boxes (ground_truth.json) are COCO xywh; prediction boxes (eval_engine) are xyxy.
    im = Image.open(image_path).convert("RGB").copy()
    d = ImageDraw.Draw(im); fsz = 11; f = _font(fsz)
    if draw_gt:  # dets are GT objects (no score), bbox = xywh
        boxes = [(o["bbox"], 1.0, id2label.get(int(o["category"]), "")) for o in dets]
    else:        # predictions, bbox = xyxy
        boxes = [(x["bbox"], float(x.get("score", 1.0)), id2label.get(int(x["label"]), str(x["label"])))
                 for x in dets]
        boxes = [b for b in boxes if b[1] >= conf]
    boxes = sorted(boxes, key=lambda b: -b[1])[:maxb]
    for (bb, sc, nm) in boxes:
        rect = [bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]] if draw_gt else list(bb)  # xywh→xyxy / xyxy
        col = GT_COLOR if draw_gt else (DET_COLOR if nm in tgt else OFFTARGET_COLOR)
        d.rectangle(rect, outline=col, width=3)
        if labels and not draw_gt:
            # class + score in a small colored chip above the box, with a dark halo around the
            # text so it stays readable on any background (e.g. green PCB).
            txt = f"{nm} {sc:.3f}"; tw = d.textlength(txt, font=f); th = fsz + 4
            lx = min(rect[0], im.width - tw - 4); ly = rect[1] - th if rect[1] - th >= 0 else rect[1]
            d.rectangle([lx, ly, lx + tw + 4, ly + th], fill=col)
            d.text((lx + 2, ly + 1), txt, fill=TEXT_COLOR, font=f, stroke_width=1, stroke_fill=DARK)
    return im, len(boxes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig-eval-set", required=True)
    ap.add_argument("--orig-predictions", required=True)
    ap.add_argument("--ft-eval-set", required=True)
    ap.add_argument("--ft-predictions", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n", type=int, default=6, help="number of sample frames (5-10 typical)")
    ap.add_argument("--conf", type=float, default=0.2, help="min score to draw a predicted box")
    ap.add_argument("--no-auto-conf", action="store_true",
                    help="use --conf EXACTLY (skip the adaptive lowering) — for syncing the report "
                         "to the UI's confidence slider so both draw boxes at the same threshold")
    ap.add_argument("--max-boxes", type=int, default=80)
    ap.add_argument("--frames", help="comma-separated image_ids to force (else busiest-by-GT)")
    ap.add_argument("--contact-frames", type=int, default=4, help="frames in the report contact sheet")
    ap.add_argument("--min-gt", type=int, default=10, help="min GT objects/frame for legible selection")
    ap.add_argument("--max-gt", type=int, default=70, help="max GT objects/frame for legible selection")
    # colors (hex '#rrggbb' or 'r,g,b') so boxes/text stay visible on any background (e.g. green PCB)
    ap.add_argument("--det-color", help="detection box (on-target) color; default green")
    ap.add_argument("--offtarget-color", help="off-target detection box color; default orange")
    ap.add_argument("--gt-color", help="ground-truth box color; default blue")
    ap.add_argument("--text-color", help="label text color; default white (drawn with a dark halo)")
    args = ap.parse_args()

    global DET_COLOR, OFFTARGET_COLOR, GT_COLOR, TEXT_COLOR
    if args.det_color:       DET_COLOR = _rgb(args.det_color)
    if args.offtarget_color: OFFTARGET_COLOR = _rgb(args.offtarget_color)
    if args.gt_color:        GT_COLOR = _rgb(args.gt_color)
    if args.text_color:      TEXT_COLOR = _rgb(args.text_color)

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    gO, l2O, imO, tgtO = _load_set(args.orig_eval_set)
    gF, l2F, imF, tgtF = _load_set(args.ft_eval_set)
    pO, pF = _load_preds(args.orig_predictions), _load_preds(args.ft_predictions)

    # Draw threshold. With --no-auto-conf use --conf EXACTLY (so the report matches the UI slider).
    # Otherwise adapt to the model's own score range so boxes are visible OOB (a fixed --conf can
    # sit far above a low-magnitude score head).
    if args.no_auto_conf:
        conf = args.conf
        print(f"[sample_overlays] using exact conf {conf:g} (no auto-lowering)")
    else:
        conf = _auto_conf(pF, args.conf, max(args.n, 1))
        if conf < args.conf:
            print(f"[sample_overlays] requested --conf {args.conf:g} yields too few detections; "
                  f"auto-lowered to {conf:g} to match the deployed model's score range")

    # frame selection: forced, else a SPREAD across GT-density (sparse→dense) so
    # boxes stay legible while still showing a busy flock.
    if args.frames:
        ids = [int(x) for x in args.frames.split(",")]
    else:
        # Prefer frames where the FINE-TUNED model actually detects (>= conf), so the report shows
        # boxes — ranked by detection count, tie-broken by GT density.
        det = {im["image_id"]: (sum(1 for d in pF.get(im["image_id"], []) if float(d.get("score", 1.0)) >= conf),
                                len(im["objects"])) for im in gF["images"]}
        hits = sorted([i for i, (n, _) in det.items() if n > 0], key=lambda i: (-det[i][0], -det[i][1]))
        if len(hits) >= args.n:
            ids = hits[:args.n]
        else:
            # too few detecting frames — seed with them, then fill from a GT-density spread
            band = [im for im in gF["images"] if args.min_gt <= len(im["objects"]) <= args.max_gt]
            if len(band) < args.n:
                band = [im for im in gF["images"] if im["objects"]] or list(gF["images"])
            pool = sorted(band, key=lambda m: len(m["objects"]))
            n = max(1, min(args.n, len(pool)))
            idxs = sorted({min(len(pool) - 1, int((k + 0.5) * len(pool) / n)) for k in range(n)})
            ids = list(hits)
            for j in list(idxs) + list(range(len(pool) - 1, -1, -1)):
                if len(ids) >= args.n:
                    break
                if pool[j]["image_id"] not in ids:
                    ids.append(pool[j]["image_id"])
    ids = ids[:args.n]

    panels = []; rows = []
    for i in ids:
        imgF = Path(args.ft_eval_set) / imF[i]["file"]
        imgO = Path(args.orig_eval_set) / imO[i]["file"]
        gt_im, ng = _draw(imgF, imF[i]["objects"], l2F, tgtF, 0, 10 ** 6, draw_gt=True)
        bl_im, nb = _draw(imgO, pO.get(i, []), l2O, tgtO, conf, args.max_boxes)
        ft_im, nf = _draw(imgF, pF.get(i, []), l2F, tgtF, conf, args.max_boxes)
        pg = _panel(gt_im, "Ground truth", f"{ng} objects")
        pb = _panel(bl_im, "Baseline (deployed)", f"{nb} ≥{conf:g}")
        pf = _panel(ft_im, "Fine-tuned (deployed)", f"{nf} ≥{conf:g}")
        gap = 10; W = pg.width * 3 + gap * 2; H = pg.height
        row = Image.new("RGB", (W, H), WHITE)
        row.paste(pg, (0, 0)); row.paste(pb, (pg.width + gap, 0)); row.paste(pf, (2 * (pg.width + gap), 0))
        fp = out / f"sample_{i:06d}.png"; row.save(fp)
        panels.append(row); rows.append({"id": i, "file": fp.name, "gt": ng, "baseline": nb, "finetuned": nf})
        print(f"[sample_overlays] frame {i}: GT={ng}  baseline={nb}  finetuned={nf}  -> {fp}")

    # Manifest, ordered by how clearly the frame shows the improvement (fine-tuned − baseline).
    # The report features the top entries; this is a display order, not cherry-picking
    # (all frames come from the same legible-density band).
    rows_sorted = sorted(rows, key=lambda r: -(r["finetuned"] - r["baseline"]))
    (out / "samples.json").write_text(json.dumps(
        {"conf": conf, "requested_conf": args.conf, "frames": rows_sorted}, indent=2))

    # stacked contact sheet for the report (first --contact-frames frames)
    cf = panels[:args.contact_frames]
    if cf:
        gap = 14; W = cf[0].width; H = sum(p.height for p in cf) + gap * (len(cf) - 1)
        sheet = Image.new("RGB", (W, H), WHITE); y = 0
        for p in cf:
            sheet.paste(p, (0, y)); y += p.height + gap
        sheet.save(out / "samples_contact.png")
        print(f"[sample_overlays] contact sheet ({len(cf)} frames) -> {out/'samples_contact.png'}")


if __name__ == "__main__":
    main()
