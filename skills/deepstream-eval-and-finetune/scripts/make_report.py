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

"""Render a deploy→adapt→re-deploy run into a narrative PDF.

Headline = DEPLOYED before/after: ORIGINAL model deployed vs FINE-TUNED model
deployed (both measured in DeepStream, same images) — accuracy + perf. No
pass/fail verdict. Reads the comparison.json from compare_runs.py.
"""
import argparse
import json
import re
import tempfile
from pathlib import Path


def _load(p):
    return json.loads(Path(p).read_text()) if p and Path(p).exists() else None


def _parse_train_log(path):
    if not path or not Path(path).exists():
        return [], [], [], []
    txt = Path(path).read_text()
    ev = re.findall(r"'eval_loss': ([0-9.]+).*?'epoch': ([0-9.]+)", txt)
    tr = re.findall(r"'loss': ([0-9.]+), 'grad_norm'.*?'epoch': ([0-9.]+)", txt)
    return ([float(e) for _, e in ev], [float(l) for l, _ in ev],
            [float(e) for _, e in tr], [float(l) for l, _ in tr])


def _charts(tmp, cmp, ep, el, tep, tl, map_ep=None, mapv=None, map50=None):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    G, B, K = "#76b900", "#1f77b4", "#555555"
    out = {}
    # per-epoch mAP (PyTorch proxy) — only when the run logged it (compute_metrics wired in)
    if map_ep:
        fig, ax = plt.subplots(1, 1, figsize=(9, 2.8))
        if map50: ax.plot(map_ep, map50, "s-", color=G, ms=3, label="mAP@50")
        if mapv: ax.plot(map_ep, mapv, "o-", color=B, ms=3, label="mAP@[.5:.95]")
        ax.set_title("Eval mAP / epoch (PyTorch proxy on the eval subset — not the deployed mAP)")
        ax.set_xlabel("epoch"); ax.set_ylim(0, 1); ax.grid(alpha=.3); ax.legend(fontsize=8)
        fig.tight_layout(); out["map"] = f"{tmp}/map.png"; fig.savefig(out["map"], dpi=130); plt.close(fig)
    if tep or ep:
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.0))
        if tep: ax[0].plot(tep, tl, color=B, lw=.8); ax[0].set_title("Training loss"); ax[0].set_xlabel("epoch"); ax[0].grid(alpha=.3)
        if ep: ax[1].plot(ep, el, "o-", color=G, ms=3); ax[1].set_title("Eval loss / epoch"); ax[1].set_xlabel("epoch"); ax[1].grid(alpha=.3)
        fig.tight_layout(); out["loss"] = f"{tmp}/loss.png"; fig.savefig(out["loss"], dpi=130); plt.close(fig)
    acc = (cmp or {}).get("accuracy", {})
    if acc:
        ks = [k for k in ["map", "map_50", "map_75"] if k in acc]; x = range(len(ks)); w = .38
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
        ax[0].bar([i-w/2 for i in x], [acc[k].get("original") or 0 for k in ks], w, label="original (deployed)", color=K)
        ax[0].bar([i+w/2 for i in x], [acc[k].get("finetuned") or 0 for k in ks], w, label="fine-tuned (deployed)", color=G)
        ax[0].set_xticks(list(x)); ax[0].set_xticklabels(ks); ax[0].legend()
        ax[0].set_title("Deployed accuracy: original (baseline) vs fine-tuned"); ax[0].grid(alpha=.3, axis="y")
        pc = {k: v for k, v in (cmp.get("per_class_ap_finetuned") or {}).items() if v is not None and v >= 0}
        if pc:
            ax[1].barh(list(pc), list(pc.values()), color=B); ax[1].invert_yaxis()
            ax[1].set_title("Per-class AP (fine-tuned, deployed)"); ax[1].set_xlabel("AP"); ax[1].grid(alpha=.3, axis="x")
            if len(pc) > 25:  # COCO-80 etc.: thin labels so they stay legible
                ax[1].tick_params(axis="y", labelsize=5)
        fig.tight_layout(); out["acc"] = f"{tmp}/acc.png"; fig.savefig(out["acc"], dpi=130); plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", default="DeepStream Eval & Finetune Report")
    ap.add_argument("--config"); ap.add_argument("--comparison", required=True)
    ap.add_argument("--train-log"); ap.add_argument("--output", required=True)
    ap.add_argument("--model-id", default="-"); ap.add_argument("--model-link", default="")
    ap.add_argument("--model-license", default="-"); ap.add_argument("--model-desc", default="")
    ap.add_argument("--dataset-id", default="-"); ap.add_argument("--dataset-link", default="")
    ap.add_argument("--dataset-license", default="-"); ap.add_argument("--dataset-desc", default="")
    ap.add_argument("--precision", default="fp16")
    ap.add_argument("--deployed-checkpoint", help="deployed_checkpoint.json: deployed epoch + "
                    "stored-model paths (renders §2b). Written by the pipeline at deploy time.")
    ap.add_argument("--samples-dir", help="dir of sample_*.png from sample_overlays.py (qualitative §5e)")
    ap.add_argument("--env", help="env_info.json from collect_env.py (system/tool provenance, §7). "
                                  "If omitted, collected best-effort at report time.")
    ap.add_argument("--image", default="nvcr.io/nvidia/deepstream:9.1-triton-multiarch",
                    help="DeepStream container image, for the §7 provenance table")
    args = ap.parse_args()

    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable, PageBreak)

    cmp = _load(args.comparison) or {}
    env = _load(args.env)
    if env is None:  # no env_info.json supplied — collect best-effort at report time
        try:
            import collect_env
            env = collect_env.collect(image=args.image)
        except Exception:
            env = None
    # Loss curve from the STITCHED history (curve_history.json + current train.log) so a run that was
    # continued shows the full continuous curve (e.g. 1..50), not just the last warm-start segment.
    if args.train_log:
        import curve_history
        try:
            mc = curve_history.merged(str(Path(args.train_log).resolve().parent))
        except Exception as e:
            # A curve-parsing failure (e.g. a trainer log-format change) must never destroy the
            # whole deliverable — render the report without the training-curve figure instead.
            print(f"[report] WARN: training-curve parse failed ({e}); rendering without curves")
            mc = {k: [] for k in ("eval_epoch", "eval_loss", "train_epoch", "train_loss",
                                  "map_epoch", "eval_map", "eval_map_50")}
        ep, el, tep, tl = mc["eval_epoch"], mc["eval_loss"], mc["train_epoch"], mc["train_loss"]
        map_ep, mapv, map50 = mc.get("map_epoch", []), mc.get("eval_map", []), mc.get("eval_map_50", [])
    else:
        ep, el, tep, tl = [], [], [], []
        map_ep, mapv, map50 = [], [], []
    cfg = Path(args.config).read_text() if args.config and Path(args.config).exists() else ""
    epochs = (re.search(r"^num_train_epochs:\s*(.+)$", cfg, re.M) or [None, "-"])[1] if cfg else "-"
    if tep:  # report the CUMULATIVE epochs trained across all continue segments
        epochs = str(int(round(max(tep))))

    ss = getSampleStyleSheet()
    H1 = ParagraphStyle("H1", parent=ss["Heading1"], spaceBefore=8)
    H2 = ParagraphStyle("H2", parent=ss["Heading2"], textColor=colors.HexColor("#3b6e00"))
    body = ParagraphStyle("body", parent=ss["BodyText"], fontSize=9.5, leading=13)
    cell = ParagraphStyle("cell", parent=body, fontSize=8.5, leading=11)
    cellb = ParagraphStyle("cellb", parent=cell, textColor=colors.white, fontName="Helvetica-Bold")
    keyc = ParagraphStyle("keyc", parent=cell, fontName="Helvetica-Bold")
    small = ParagraphStyle("small", parent=body, fontSize=8, textColor=colors.HexColor("#666"))
    GREEN = colors.HexColor("#76b900"); HEADBG = colors.HexColor("#5a8f00"); DS = colors.HexColor("#eaf3d6")

    story = [Paragraph(args.title, H1),
             Paragraph("NVIDIA DeepStream — deploy → measure → adapt → re-deploy", small),
             HRFlowable(width="100%", color=GREEN, thickness=2, spaceAfter=8)]

    # 1. definitions
    story.append(Paragraph("1. What this report shows", H2))
    story.append(Paragraph("The workflow deploys the user's model to DeepStream, measures it on the user's "
                           "KPI dataset (the <b>baseline</b>), fine-tunes it, re-deploys, and reports the "
                           "<b>before/after improvement</b> — both models measured <b>as deployed in DeepStream</b>. "
                           "There is no pass/fail verdict.", body))
    defs = [["Term", "Meaning"],
        ["Original (deployed) = baseline", "The user's model exported to ONNX, built to a TensorRT engine, and run "
                                "through the DeepStream nvinfer pipeline on the KPI images. This IS the default-state "
                                "baseline — mAP + FPS + latency, exactly as shipped. Every model is measured this way; "
                                "KPI categories the model does not yet output simply score 0 (shown per-class as "
                                "'no object detected')."],
        ["Fine-tuned (deployed)", "The same model after fine-tuning on the KPI dataset, then re-deployed the same way."],
        ["Improvement", "Fine-tuned (deployed) minus Original (deployed) — the measured gain on the KPI data."]]
    t = Table([[Paragraph(c, cellb if i == 0 else (keyc if j == 0 else cell)) for j, c in enumerate(r)] for i, r in enumerate(defs)],
              colWidths=[1.5*inch, 5.2*inch])
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),HEADBG),("GRID",(0,0),(-1,-1),.5,colors.grey),
                           ("VALIGN",(0,0),(-1,-1),"TOP"),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white, colors.HexColor("#f2f7e8")])]))
    story += [t, Spacer(1, 6),
              Paragraph("<b>Where DeepStream is used:</b> the model is deployed and measured in DeepStream "
                        "<i>twice</i> — once original (baseline), once fine-tuned — via the custom nvinfer parser + the "
                        "<font face='Courier'>ds_image_eval</font> pipeline "
                        "(<font face='Courier'>multifilesrc → jpegdec → nvstreammux → nvinfer → probe</font>). "
                        f"Precision: <b>{args.precision.upper()}</b>"
                        + (" (no calibration needed)." if args.precision == "fp16"
                           else " (calibration cache auto-generated from the KPI dataset)."), body)]

    # 2. model & dataset
    story.append(Paragraph("2. Model &amp; dataset", H2))
    md = [["Model", Paragraph(f"<b>{args.model_id}</b><br/>{args.model_desc}"
            + (f"<br/><font color='#1f77b4'>{args.model_link}</font>" if args.model_link else "")
            + f"<br/>License: {args.model_license}", cell)],
          ["Dataset", Paragraph(f"<b>{args.dataset_id}</b><br/>{args.dataset_desc}"
            + (f"<br/><font color='#1f77b4'>{args.dataset_link}</font>" if args.dataset_link else "")
            + f"<br/>License: {args.dataset_license}", cell)]]
    t = Table(md, colWidths=[1.0*inch, 5.7*inch])
    t.setStyle(TableStyle([("GRID",(0,0),(-1,-1),.5,colors.grey),("VALIGN",(0,0),(-1,-1),"TOP"),
                           ("BACKGROUND",(0,0),(0,-1),colors.HexColor("#eef5dd")),("FONTNAME",(0,0),(0,-1),"Helvetica-Bold")]))
    story.append(t)

    # 2a. dataset classes & distribution — per-class instance + image counts (train vs eval).
    # Explains class imbalance + which classes the deployed mAP actually covers.
    dist = cmp.get("dataset_distribution")
    if dist and dist.get("classes"):
        story.append(Spacer(1, 6))
        story.append(Paragraph("2a. Dataset classes &amp; distribution", H2))
        rows = [[Paragraph(h, cellb) for h in ("Class", "Train (inst / imgs)", "Eval (inst / imgs)")]]
        for c in dist["classes"]:
            rows.append([Paragraph(c["name"], cell),
                         Paragraph(f"{c['train_instances']} / {c['train_images']}", cell),
                         Paragraph(f"{c['eval_instances']} / {c['eval_images']}", cell)])
        rows.append([Paragraph("<b>Total</b>", cell),
                     Paragraph(f"<b>{dist.get('train_total_instances',0)}</b> inst / {dist.get('n_train_images',0)} imgs", cell),
                     Paragraph(f"<b>{dist.get('eval_total_instances',0)}</b> inst / {dist.get('n_eval_images',0)} imgs", cell)])
        td = Table(rows, colWidths=[2.6*inch, 2.05*inch, 2.05*inch])
        td.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),HEADBG),("GRID",(0,0),(-1,-1),.5,colors.grey),
                                ("VALIGN",(0,0),(-1,-1),"TOP"),
                                ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white, colors.HexColor("#f2f7e8")])]))
        story.append(td)
        nzero = sum(1 for c in dist["classes"] if c["eval_instances"] == 0)
        cap = ("<b>inst</b> = annotated objects, <b>imgs</b> = images containing the class. "
               + (f"{nzero} class(es) have <b>0 eval instances</b> — not present in the KPI eval slice, so "
                  "deployed mAP is averaged only over the classes that are."
                  if nzero else "All classes appear in the eval slice."))
        story.append(Paragraph(cap, small))

    # 2b. deployed checkpoint & stored model (which epoch was deployed + where the model lives)
    dep = _load(args.deployed_checkpoint) if args.deployed_checkpoint else None
    if dep:
        min_ev = int(round(ep[min(range(len(el)), key=lambda k: el[k])])) if el else None
        de = dep.get("deployed_epoch")
        note = (f"Deployed the <b>most-trained</b> checkpoint (epoch <b>{de}</b> of {epochs}). "
                + (f"Lowest eval-loss was epoch {min_ev}, but for DETR-family detectors eval-loss is a poor "
                   "accuracy proxy — it often rises while detection mAP keeps improving — so the most-trained "
                   "checkpoint is deployed and validated by the deployed mAP below."
                   if (min_ev is not None and min_ev != de) else
                   "For DETR-family detectors eval-loss is a poor accuracy proxy, so the most-trained checkpoint "
                   "is deployed and validated by the deployed mAP below."))
        rows = [["Checkpoint", Paragraph(note, cell)],
                ["Stored model", Paragraph(
                    f"<b>{dep.get('model_name','-')}</b><br/>"
                    f"checkpoint: <font face='Courier'>{dep.get('checkpoint_epoch_tag') or dep.get('checkpoint','-')}</font><br/>"
                    f"ONNX: <font face='Courier'>{dep.get('onnx','-')}</font><br/>"
                    f"TRT engine: <font face='Courier'>{dep.get('engine','-')}</font><br/>"
                    f"nvinfer config: <font face='Courier'>{dep.get('nvinfer_config','-')}</font>", cell)]]
        story.append(Spacer(1, 6))
        story.append(Paragraph("2b. Deployed checkpoint &amp; stored model", H2))
        tb = Table(rows, colWidths=[1.0*inch, 5.7*inch])
        tb.setStyle(TableStyle([("GRID",(0,0),(-1,-1),.5,colors.grey),("VALIGN",(0,0),(-1,-1),"TOP"),
                               ("BACKGROUND",(0,0),(0,-1),colors.HexColor("#eef5dd")),("FONTNAME",(0,0),(0,-1),"Helvetica-Bold")]))
        story.append(tb)

    # 2c. Training configuration — the actual fine-tune hyperparameters + how each was chosen.
    def _cv(key, default="—"):
        m = re.search(rf"^{re.escape(key)}:\s*(.+?)\s*$", cfg, re.M)
        return m.group(1).strip() if m else default
    if cfg:
        lab = _cv("label_names", "")
        ncls = (lab.count(",") + 1) if lab.strip("[] ") else "—"
        trows = [["Parameter", "Value", "How it was set"],
            ["Learning rate", _cv("learning_rate"), "auto by class count (≤2 → 1e-4; ≥3 → 2.5e-5) · UI override"],
            ["LR scheduler", _cv("lr_scheduler_type"), "auto (≤2 → linear; ≥3 → cosine)"],
            ["Warmup ratio", _cv("warmup_ratio"), "auto (≤2 → 0.05; ≥3 → 0.1)"],
            ["Epochs", epochs, "user-set on the Setup form (default 12)"],
            ["Batch size", _cv("per_device_train_batch_size"), "fixed default 8 · UI override"],
            ["Grad accumulation", _cv("gradient_accumulation_steps", "1"), "fixed"],
            ["Weight decay", _cv("weight_decay"), "fixed"],
            ["Precision", "bf16" if _cv("bf16", "") == "true" else "fp32", "fixed (mixed precision)"],
            ["Eval / best by", f'{_cv("eval_strategy","epoch")} · {_cv("metric_for_best_model","eval_loss")}', "fixed"]]
        story.append(Spacer(1, 6))
        story.append(Paragraph("2c. Training configuration", H2))
        tc = Table([[Paragraph(c, cellb if i == 0 else (keyc if j == 0 else cell)) for j, c in enumerate(r)] for i, r in enumerate(trows)],
                   colWidths=[1.5*inch, 1.5*inch, 3.7*inch])
        tc.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),HEADBG),("GRID",(0,0),(-1,-1),.5,colors.grey),
                                ("VALIGN",(0,0),(-1,-1),"TOP"),
                                ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white, colors.HexColor("#f2f7e8")])]))
        story.append(tc)
        story.append(Paragraph(
            f"Hyperparameters are chosen from <b>this dataset's class count</b> ({ncls} classes), not copied from any "
            "prior run: RT-DETR's varifocal classification head diverges at lr=1e-4 once there are several classes "
            "(logits collapse to the focal prior), so <b>≥3 classes</b> use the gentler <b>2.5e-5 / cosine / warmup "
            "0.1</b> recipe and <b>≤2 classes</b> use <b>1e-4 / linear</b>. UI <i>Advanced settings</i> override the "
            "auto values; the remaining values are fixed.", small))

    # 3. skills used
    story.append(Paragraph("3. Skills used", H2))
    sk = [["Skill", "Role", "Steps"],
          ["deepstream-eval-and-finetune", "Orchestrator — build eval set, run deployed eval (ds_image_eval), bench perf, compare, report", "3,6,7"],
          ["deepstream-import-vision-model", "Import Agent — ONNX export + TensorRT engine + nvinfer parser/config (deploys both models)", "2,5"],
          ["tao-finetune-huggingface-model", "DEFT — fine-tunes the model on the KPI dataset (tao-skill-bank, via tao-launch-workflow)", "4"]]
    t = Table([[Paragraph(c, cellb if i == 0 else (keyc if j == 0 else cell)) for j, c in enumerate(r)] for i, r in enumerate(sk)],
              colWidths=[2.1*inch, 4.0*inch, 0.6*inch])
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),HEADBG),("GRID",(0,0),(-1,-1),.5,colors.grey),("VALIGN",(0,0),(-1,-1),"TOP")]))
    story.append(t)

    # 4. workflow
    story.append(Paragraph("4. Workflow — step by step", H2))
    steps = [["#", "Step", "Skill / tool", "What it does", "Why", False],
        ["1","Model download","HF Hub (transformers)","Pull the model + processor.","Starting point — the user's model.",False],
        ["2","Deploy original","deepstream-import-vision-model","Validate + ONNX + TRT engine + nvinfer config + parser.","Make the user's model runnable in DeepStream.",True],
        ["3","Eval original (deployed)","deepstream-eval-and-finetune · ds_image_eval","Stream KPI images through nvinfer; score mAP (per-category); bench perf.","Baseline AS DEPLOYED — per-class mAP; 0 for classes not yet output.",True],
        ["4","Fine-tune","tao-finetune-huggingface-model","Train on the KPI dataset.","Improve accuracy on the user's classes.",False],
        ["5","Deploy fine-tuned","deepstream-import-vision-model","Re-deploy the fine-tuned checkpoint (ONNX→TRT→nvinfer).","Make the improved model runnable in DeepStream.",True],
        ["6","Eval fine-tuned (deployed)","deepstream-eval-and-finetune · ds_image_eval","Same images, same pipeline; score mAP + perf.","The 'after', AS DEPLOYED.",True],
        ["7","Report","deepstream-eval-and-finetune · compare_runs + make_report","Merge before/after; render this PDF.","Show the deployed improvement (accuracy + perf).",False]]
    rows = [[Paragraph(str(c), cellb if i == 0 else cell) for c in r[:5]] for i, r in enumerate(steps)]
    t = Table(rows, colWidths=[0.28*inch,1.0*inch,1.85*inch,1.97*inch,1.6*inch], repeatRows=1)
    tstyle = [("BACKGROUND",(0,0),(-1,0),HEADBG),("GRID",(0,0),(-1,-1),.5,colors.grey),("VALIGN",(0,0),(-1,-1),"TOP")]
    for i, r in enumerate(steps[1:], 1):
        if r[5]: tstyle.append(("BACKGROUND",(0,i),(-1,i),DS))
    t.setStyle(TableStyle(tstyle))
    story += [t, Paragraph("Rows shaded green are the DeepStream stages.", small)]

    # 4. results
    with tempfile.TemporaryDirectory() as tmp:
        ch = _charts(tmp, cmp, ep, el, tep, tl, map_ep, mapv, map50)
        story.append(Paragraph("5. Results", H2))
        acc = cmp.get("accuracy", {})
        if acc:
            m50 = acc.get("map_50", {})
            ck = cmp.get("checkpoint_evaluated") or {}
            head = "5a. Deployed accuracy — original (baseline) vs fine-tuned (DeepStream, same images)"
            if m50.get("x"):
                head += f" — <b>{m50['x']}× mAP@50</b>"
            story.append(Paragraph(head, body))
            if ck.get("epoch") is not None and (ck.get("map") is not None or ck.get("map_50") is not None):
                sel = "best" if ck.get("selection") == "best" else "latest"
                tr = f"Checkpoint deployed: <b>{sel} training epoch {ck['epoch']}</b>"
                if ck.get("map") is not None:
                    tr += f" — training mAP <b>{ck['map']:.3f}</b>"
                if ck.get("map_50") is not None:
                    tr += f", training mAP@50 <b>{ck['map_50']:.3f}</b>"
                tr += " (PyTorch eval during training; tiles below are DeepStream deployed)."
                story.append(Paragraph(tr, small))
            rr = [["Metric", "Original (baseline)", "Fine-tuned (deployed)", "Δ"]]
            for k in ["map", "map_50", "map_75"]:
                if k in acc:
                    a = acc[k]
                    rr.append([k, f"{a.get('original',0):.4f}", f"{a.get('finetuned',0):.4f}",
                               f"{a.get('delta',0):+.4f}" if a.get("delta") is not None else "—"])
            tb = Table(rr, colWidths=[1.0*inch, 1.7*inch, 1.7*inch, 1.0*inch])
            tb.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),HEADBG),("TEXTCOLOR",(0,0),(-1,0),colors.white),
                                    ("GRID",(0,0),(-1,-1),.5,colors.grey),("FONTSIZE",(0,0),(-1,-1),9)]))
            story.append(tb)
            if ch.get("acc"): story.append(Image(ch["acc"], width=6.7*inch, height=2.46*inch))

        # perf
        perf = cmp.get("perf", {}); dr = cmp.get("detection_rate", {})
        po, pf = perf.get("original", {}), perf.get("finetuned", {})
        def pv(d, k, suf=""):
            return f"{d[k]}{suf}" if d.get(k) is not None else "—"
        if po or pf:
            rr = [["Perf (TRT FP16, batch 1)", "Original", "Fine-tuned"],
                  ["engine FPS (trtexec qps)", pv(po, "trtexec_qps"), pv(pf, "trtexec_qps")],
                  ["GPU latency mean (ms)", pv(po, "gpu_compute_ms_mean"), pv(pf, "gpu_compute_ms_mean")],
                  ["GPU latency p99 (ms)", pv(po, "gpu_compute_ms_p99"), pv(pf, "gpu_compute_ms_p99")],
                  ["detection rate", f"{(dr.get('original') or 0)*100:.0f}%", f"{(dr.get('finetuned') or 0)*100:.0f}%"]]
            tb = Table(rr, colWidths=[2.2*inch, 1.5*inch, 1.5*inch])
            tb.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),HEADBG),("TEXTCOLOR",(0,0),(-1,0),colors.white),
                                    ("GRID",(0,0),(-1,-1),.5,colors.grey),("FONTSIZE",(0,0),(-1,-1),9)]))
            story += [Spacer(1, 6), Paragraph("5b. Perf — authoritative engine throughput/latency (trtexec)", body), tb,
                      Paragraph("Engine FPS/latency are from a trtexec benchmark of the serialized engine "
                                "(GPU-only, batch 1) — real throughput, independent of the one-time nvinfer build. "
                                "Original and fine-tuned share the architecture, so perf is essentially identical.", small)]
        if ch.get("loss"):
            story += [Spacer(1, 6), Paragraph("5c. Fine-tuning curves", body), Image(ch["loss"], width=6.7*inch, height=2.23*inch)]
        if ch.get("map"):
            story += [Spacer(1, 6), Paragraph("5d. Per-epoch eval mAP (PyTorch proxy — trend, not the deployed mAP)", body),
                      Image(ch["map"], width=6.7*inch, height=2.08*inch)]

        # 5d. Per-class detail — EVERY KPI category, including any with no detections.
        pco = cmp.get("per_class_ap_original", {}) or {}
        pcf = cmp.get("per_class_ap_finetuned", {}) or {}
        dco = cmp.get("per_class_detections_original", {}) or {}
        dcf = cmp.get("per_class_detections_finetuned", {}) or {}
        # Rows = the KPI/target classes that actually have ground-truth AP (>=0).
        # torchmetrics returns -1 for classes with no GT (e.g. the other 79 COCO
        # classes when a stock model is scored), and detection counts may include
        # extra classes in a wider label space — neither should drive the rows.
        def _valid(d, c):
            v = d.get(c)
            return isinstance(v, (int, float)) and v >= 0
        classes = sorted({c for c in (set(pco) | set(pcf)) if _valid(pco, c) or _valid(pcf, c)})
        if classes:
            def apfmt(d, dets, c):
                v = d.get(c)
                if c in dets and dets.get(c) == 0:
                    return "no object detected"
                if v is None or v < 0:
                    return "n/a"
                return f"{v:.4f}"
            rr = [["KPI class", "Baseline AP", "Fine-tuned AP"]]
            for c in classes:
                rr.append([c, apfmt(pco, dco, c), apfmt(pcf, dcf, c)])
            tb = Table(rr, colWidths=[2.6*inch, 1.9*inch, 1.9*inch])
            tb.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),HEADBG),("TEXTCOLOR",(0,0),(-1,0),colors.white),
                                    ("GRID",(0,0),(-1,-1),.5,colors.grey),("FONTSIZE",(0,0),(-1,-1),8.5),
                                    ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white, colors.HexColor("#f2f7e8")])]))
            story += [Spacer(1, 6), Paragraph("5d. Per-class accuracy (every KPI category)", body), tb,
                      Paragraph("Categories the deployed model produced no detections for are marked "
                                "'no object detected' (AP 0). 'n/a' = no ground-truth instances of that "
                                "class in the eval sample.", small)]

        # 5e. Qualitative samples — rendered deployed detections (GT | baseline | fine-tuned).
        if args.samples_dir and Path(args.samples_dir).is_dir():
            sdir = Path(args.samples_dir); man = sdir / "samples.json"
            conf = 0.3
            if man.exists():  # feature the clearest comparisons first
                mj = json.loads(man.read_text()); conf = mj.get("conf", 0.3)
                feat = mj.get("frames", [])
                frames = [(sdir / f["file"], f) for f in feat[:3] if (sdir / f["file"]).exists()]
            else:
                frames = [(p, None) for p in sorted(sdir.glob("sample_*.png"))[:3]]
            if frames:
                story += [PageBreak(), Paragraph("5e. Qualitative samples — deployed detections", H2),
                          Paragraph("The same frames, scored by each deployed model. Left → right: "
                                    "<b>ground truth</b> · <b>baseline</b> (original, deployed) · "
                                    f"<b>fine-tuned</b> (deployed). Boxes drawn above a {conf:g} confidence "
                                    "threshold — target-class boxes green, off-target (a stock model's "
                                    "mis-detections) orange.", body)]
                from PIL import Image as _PILImage
                for fp, meta in frames:
                    iw, ih = _PILImage.open(fp).size
                    w = 7.1 * inch; h = w * ih / iw
                    story.append(Image(str(fp), width=w, height=h))
                    if meta:
                        story.append(Paragraph(f"Frame {meta['id']}: {meta['gt']} ground-truth object(s) — "
                            f"baseline found {meta['baseline']}, fine-tuned found {meta['finetuned']}.",
                            small))
                    story.append(Spacer(1, 6))

        # 6. Detailed summary (prose recap, end of report).
        story.append(Paragraph("6. Detailed summary", H2))
        m50 = acc.get("map_50", {}); mm = acc.get("map", {})
        n_eval = cmp.get("n_eval", "-")
        und = [c for c in classes if dcf.get(c) == 0]
        def _g(d, k):
            v = d.get(k); return f"{v:.4f}" if isinstance(v, (int, float)) else "-"
        po2, pf2 = perf.get("original", {}), perf.get("finetuned", {})
        lines = [f"<b>Model:</b> {args.model_id} ({args.precision.upper()}), deployed to DeepStream via "
                 f"ONNX &#8594; TensorRT &#8594; nvinfer with a custom output parser.",
                 f"<b>Dataset:</b> {args.dataset_id} &#8212; {n_eval} KPI images, scored through the real "
                 f"DeepStream pipeline (identical images for baseline and fine-tuned)."]
        if m50:
            dx = m50.get("delta"); xf = f" ({m50['x']}&#215;)" if m50.get("x") else ""
            lines.append(f"<b>Accuracy (mAP@50, as deployed):</b> baseline {_g(m50,'original')} &#8594; "
                         f"fine-tuned {_g(m50,'finetuned')}"
                         + (f", a gain of {dx:+.4f}{xf}." if dx is not None else "."))
        if mm:
            lines.append(f"<b>Accuracy (mAP@[.5:.95]):</b> baseline {_g(mm,'original')} &#8594; "
                         f"fine-tuned {_g(mm,'finetuned')}.")
        if pf2.get("trtexec_qps"):
            lines.append(f"<b>Performance (TRT {args.precision.upper()}, trtexec, batch 1):</b> "
                         f"~{pf2.get('trtexec_qps')} FPS, {pf2.get('gpu_compute_ms_mean','-')} ms mean GPU "
                         f"latency &#8212; essentially unchanged by fine-tuning (same architecture).")
        if classes:
            lines.append(f"<b>Per-class coverage:</b> {len(classes)} KPI categories evaluated; "
                         + (f"{len(und)} still produced no detections after fine-tuning "
                            f"({', '.join(und[:8])}{'&#8230;' if len(und)>8 else ''})." if und
                            else "all categories produced detections."))
        lines.append("<b>Takeaway:</b> this quantifies, end-to-end and exactly as deployed in DeepStream, how "
                     "much fine-tuning on the KPI data moved the needle. There is no pass/fail gate &#8212; the "
                     "developer decides whether the deployed accuracy and perf clear their bar.")
        for ln in lines:
            story += [Paragraph(ln, body), Spacer(1, 3)]

        # 7. System & environment — provenance so the run is reproducible.
        if env:
            story += [Spacer(1, 6), Paragraph("7. System &amp; environment", H2),
                      Paragraph("The hardware and software stack these deployed numbers were produced on "
                                "&#8212; recorded for reproducibility.", body)]
            er = [["Component", "Detail"],
                  ["GPU", env.get("gpu", "-")],
                  ["GPU driver", env.get("gpu_driver", "-")],
                  ["CUDA (driver)", env.get("cuda_driver", "-")],
                  ["CPU", env.get("cpu", "-")],
                  ["System memory", env.get("ram", "-")],
                  ["OS / kernel", env.get("os", "-")],
                  ["DeepStream", env.get("deepstream", "-")],
                  ["TensorRT", env.get("tensorrt", "-")],
                  ["Container image", env.get("container_image") or "-"],
                  ["Precision", args.precision.upper()]]
            tb = Table([[Paragraph(c, cellb if i == 0 else (keyc if j == 0 else cell)) for j, c in enumerate(r)]
                        for i, r in enumerate(er)], colWidths=[1.7*inch, 5.0*inch])
            tb.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),HEADBG),("GRID",(0,0),(-1,-1),.5,colors.grey),
                                    ("VALIGN",(0,0),(-1,-1),"TOP"),
                                    ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white, colors.HexColor("#f2f7e8")])]))
            story.append(tb)
            pkgs = env.get("packages") or {}
            if pkgs:
                order = ["torch", "torchvision", "transformers", "datasets", "torchmetrics",
                         "onnx", "onnxscript", "pycocotools", "albumentations", "numpy", "reportlab"]
                shown = [k for k in order if k in pkgs] + [k for k in pkgs if k not in order]
                pkg_str = " · ".join(f"{k} {pkgs[k]}" for k in shown)
                story += [Spacer(1, 4), Paragraph("<b>Key tool / package versions</b> (shared venv):", body),
                          Paragraph(pkg_str, small)]

        SimpleDocTemplate(args.output, pagesize=LETTER, topMargin=.6*inch, bottomMargin=.6*inch,
                          leftMargin=.7*inch, rightMargin=.7*inch).build(story)
    print(f"[make_report] wrote {args.output}")


if __name__ == "__main__":
    main()
