<!--
Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License").
-->

# Deploy + measure accuracy in DeepStream

This skill measures a model's accuracy **as deployed** — by running the real
DeepStream pipeline over the KPI images — for *both* the original and the
fine-tuned model, so the report is a true deployed before/after. Read this
before the deploy/eval stages.

## The deployed-eval pipeline

`eval_engine.py` drives the C app **`ds_image_eval`**, which decodes the KPI
JPEGs directly (no MJPEG, no pyds, no re-encode):

```
multifilesrc (NNNNNN.jpg) → jpegparse → nvv4l2decoder → nvvideoconvert
  → "video/x-raw(memory:NVMM),format=RGBA" → nvstreammux (W×H, b=1)
  → nvinfer (the import skill's config + custom parser) → fakesink
```

A buffer probe on **nvinfer's src pad** reads `NvDsObjectMeta`
(`class_id`, `confidence`, `rect_params`) and writes one line per detection:
`frame_num class_id confidence left top width height`. `frame_num` == image index
(deterministic). `eval_engine.py` parses those, remaps labels (below), and scores
mAP with `compute_map.py` (`torchmetrics`). Build the app: `make -C scripts`.

## Canonical images + coordinate rule

`build_eval_set.py` resizes every KPI image into one fixed **W×H canvas** (script
default **1280×720**; the skill **recommends 640×640** for RT-DETR — see the
recommended-default inputs) and writes `ground_truth.json` (boxes in canvas
pixels). `ds_image_eval`'s `nvstreammux` W×H MUST equal that canvas (so it passes
frames through 1:1 and `rect_params` and GT share one pixel space). Pick the
`--resize-mode` to match the model's preprocessing — **stretch** for RT-DETR's
square-resize processor, **letterbox** for YOLO-style. The SAME canonical images
are used for the original and fine-tuned runs.

## Prediction remap + per-category scoring (one path, no Case A/B)

`ground_truth.json` is in the **KPI/target** class space. `ds_image_eval` emits
`class_id` in the **deployed model's own** label order (its `labels.txt`).
`eval_engine.py` therefore **remaps predicted class ids → target ids by name**,
dropping predictions whose class isn't a target class:

- **Fine-tuned model:** its labels == target → identity remap → real mAP.
- **Original model whose classes overlap the target:** remap by name → real
  baseline mAP on the overlapping classes.
- **Original model that can't output the target classes** (e.g. a COCO model on
  PPE): nothing maps → mAP ≈ 0. This is the honest "before" — recorded, not judged.

There is **no Case A/B routing**. The skill ALWAYS builds the eval set and ALWAYS
runs DeepStream. `eval_engine.py` records `per_class_detections`, so every KPI
category is reported individually; categories with zero detections show as
**"no object detected"** (AP 0) in report §5d rather than being skipped.
`build_eval_set.py` records label `overlap` for information only — it never routes
or blocks.

## Precision: FP16

**FP16** (default and only validated precision): `trtexec --fp16` / nvinfer
`network-mode=2`. No calibration needed; ~2× faster than FP32 and near-lossless — which is
why the deployed mAP tracks the PyTorch reference closely.

## Per-architecture nvinfer specifics (from the import skill)

- **DETR:** softmax; background = LAST class (skip it); ImageNet norm →
  `net-scale-factor≈0.0175` + `offsets=123.675;116.28;103.275`.
- **RT-DETR:** **sigmoid**; **no background class**; `do_normalize=False` →
  `net-scale-factor=1/255`, no offsets; ~300 queries; set-prediction →
  `cluster-mode=4`.
- Parser post-process: match HF `post_process_object_detection(use_focal_loss=True)` —
  **global top-k over flattened (query × class) sigmoid scores**, not per-query
  argmax. Per-query argmax skews class balance and can drop mAP ~6 pts vs PyTorch.
- Parser threshold: emit all top-k queries (thresh ≈ 0) so mAP integrates by score.

## Container

Stages run in the DeepStream container (always `triton-multiarch`, latest):
```
docker run --rm --gpus all --shm-size=16g -v "$(pwd)":/work -w /work \
  nvcr.io/nvidia/deepstream:9.1-triton-multiarch bash -lc '<stage>'
```
`--shm-size=16g` (or `--ipc=host`) is REQUIRED (default `/dev/shm` 64 MB deadlocks
DataLoader workers). Make venvs with `virtualenv` (container python lacks ensurepip).
