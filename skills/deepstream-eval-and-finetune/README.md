<!--
Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-->

# DeepStream Eval & Fine-tune

Where `deepstream-import-vision-model` tells you how **fast** a model runs in DeepStream, this
skill adds **accuracy**: it deploys your object-detection model, measures **mAP + perf** on *your*
KPI data by running the **real DeepStream pipeline**, fine-tunes the model, re-deploys it, and
reports the **before/after improvement** — original-deployed vs fine-tuned-deployed, both measured
in DeepStream. **Object detection only.**

It is an **orchestrator**: it invokes `deepstream-import-vision-model` (deploy) and the
`tao-skill-bank` fine-tune / AutoML lanes (train), and leaves them unmodified. There is **no
pass/fail verdict** — both models are run through DeepStream and the report presents the measured
improvement; the developer decides what to do with it.

Runtime note: Claude Code and Codex require their respective TAO Skill Bank plugin installation. Cursor can run the bundled CLI/UI workflow, but cannot directly invoke TAO plugin skills.

## The loop

```
preflight → deploy ORIGINAL (TRT FP16) → measure baseline mAP+perf in DeepStream
          → fine-tune (single run or AutoML sweep) → deploy FINE-TUNED
          → re-measure in DeepStream → before/after PDF report
```

Classes the original model cannot output yet simply score 0 and are reported per category as
"no object detected" — an honest default-state number, never hidden; fine-tuning is what raises it.

## Quickstart

Prerequisites: Docker, an NVIDIA GPU, and the DeepStream image
(`docker pull nvcr.io/nvidia/deepstream:9.1-triton-multiarch`).

```bash
# 1. one-time bootstrap (venv + ds_image_eval C app), from your working root
docker run --rm -it --gpus all --shm-size=16g -v "$PWD":/work -w /work \
  --entrypoint bash nvcr.io/nvidia/deepstream:9.1-triton-multiarch .claude/skills/deepstream-eval-and-finetune/setup.sh

# 2. run a validated end-to-end example (inside the container)
bash .claude/skills/deepstream-eval-and-finetune/examples/rtdetr-aerial-sheep/run.sh        # RT-DETR + aerial-sheep
bash .claude/skills/deepstream-eval-and-finetune/examples/rtdetr-pcb/run.sh                 # RT-DETR + PKU/HRIPCB defects

# 3. or launch the optional guided web UI (host, port 8078)
bash .claude/skills/deepstream-eval-and-finetune/app/launch.sh
```

Validated result (RT-DETR + aerial-sheep, FP16): deployed **mAP@50 0.19 → 0.87 (~4.7×)**, ~605 FPS.

## Layout

| Path | What |
|------|------|
| `SKILL.md` | the agent playbook — read the relevant `references/*.md` before each stage |
| `references/` | deploy+eval mechanics, run-flow, fine-tune+report, the web UI |
| `scripts/` | eval-set build, `ds_image_eval` C app, deployed-eval, compare, report; `scripts/ngc/` fine-tune + AutoML harnesses |
| `app/` | optional web UI (same scripts, same container, same artifacts) |
| `examples/` | two validated worked examples (aerial-sheep, PCB) |
| `evals/` | skill evaluation cases |

See **`SKILL.md`** for the full stage-by-stage workflow and critical rules.
