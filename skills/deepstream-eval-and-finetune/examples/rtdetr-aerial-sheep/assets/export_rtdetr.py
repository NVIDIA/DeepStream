# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
#
# Export an RT-DETR checkpoint (stock HF id OR a fine-tuned dir) to ONNX +
# labels.txt for DeepStream. The custom parser reads the class count from the
# logits dim, so the same .so works for 80 (stock) or 1 (fine-tuned) classes.
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForObjectDetection


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="HF id (PekingU/rtdetr_r50vd) or local checkpoint dir")
    ap.add_argument("--out-onnx", required=True)
    ap.add_argument("--out-labels", required=True)
    ap.add_argument("--revision", default="main",
                    help="Hub revision (branch, tag, or commit SHA). Pin a SHA for reproducible builds.")
    args = ap.parse_args()

    m = AutoModelForObjectDetection.from_pretrained(args.checkpoint, revision=args.revision).eval()
    id2label = m.config.id2label
    labels = [id2label[i] for i in range(len(id2label))]
    Path(args.out_labels).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_labels).write_text("\n".join(labels) + "\n")

    class Wrap(torch.nn.Module):
        def __init__(s, m): super().__init__(); s.m = m
        def forward(s, x): o = s.m(pixel_values=x); return o.logits, o.pred_boxes

    Path(args.out_onnx).parent.mkdir(parents=True, exist_ok=True)
    # dynamo=False on purpose. RT-DETR specializes the batch dimension under the dynamo
    # exporter ("you marked batch as dynamic but your code specialized it to a constant"),
    # producing a static-batch graph; the TorchScript path keeps it dynamic. Verified against
    # transformers 5.14.1 in the DeepStream 9.1 container — export yields ['b', 3, 640, 640].
    torch.onnx.export(
        Wrap(m), torch.randn(1, 3, 640, 640), args.out_onnx,
        input_names=["pixel_values"], output_names=["logits", "pred_boxes"],
        dynamic_axes={"pixel_values": {0: "b"}, "logits": {0: "b"}, "pred_boxes": {0: "b"}},
        opset_version=17, do_constant_folding=True, dynamo=False)
    print(f"[export_rtdetr] {len(labels)} classes -> {args.out_onnx}")


if __name__ == "__main__":
    main()
