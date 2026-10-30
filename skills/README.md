<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DeepStream Coding Agent

A project showcasing how to leverage **AI coding assistants** (Cursor, Claude Code, etc.) for accelerated [NVIDIA DeepStream SDK](https://developer.nvidia.com/deepstream-sdk) application development using a curated agentic skill and structured prompts.

> **Disclaimer:** Code generated with AI coding assistants is intended as a development starting point. All generated code must undergo your full software development lifecycle (SDLC) — including code review, testing, and security validation — before production use.

---

## Prerequisites

### For code generation (using the skill and prompts)

- **AI coding assistant** that supports agentic skills (e.g., Cursor, Claude Code)

No GPU, SDK, or special hardware is required — the skill and example prompts work on any system.

### For running generated Service Maker code

The following are required on the target execution environment:

- **NVIDIA DeepStream SDK** — installed locally or available via [NVIDIA NGC container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/deepstream)
- **Python 3.12+** with the `pyservicemaker` package
- **NVIDIA GPU** with driver version 595 or later
- **CUDA 13.2** and **TensorRT 10.16.0.72**
- **Supported OS:** Ubuntu 24.04 (x86_64 or ARM64/Jetson)

> Individual command-line skills may support an earlier Python version. Follow each skill's own prerequisites.

> The `deepstream-import-vision-model` and `deepstream-eval-and-finetune` skills run their toolchains inside `nvcr.io/nvidia/deepstream:9.1-triton-multiarch`; the host needs Docker, NVIDIA Container Toolkit, and a compatible NVIDIA driver. Their setup and preflight scripts create and verify the mounted Python environments and runtime tools.

> For detailed environment setup, refer to the [DeepStream SDK Developer Guide](https://docs.nvidia.com/deepstream/9.2).

### DeepStream Version Compatibility

Some skills may document a specific DeepStream SDK version or container image that is not the latest DeepStream release version. Use the version documented by the skill unless you have a reason to test with a newer DeepStream image.

Skills that mention older DeepStream base images may still work out of the box with newer DeepStream versions. If you need to use a newer image, update the image tag and run the skill's setup, test, or benchmark steps to confirm the workflow.

---

## Project Structure

This README sits under `skills/` inside the DeepStream mono-repo. Layout of this subtree, plus the related top-level `example_prompts/` directory:

```
deepstream/                                 # mono-repo root
├── skills/                                 # Agentic skills for guided DeepStream development
│   ├── README.md                           # This file
│   ├── deepstream-dev/                     # DeepStream development skill with condensed references
│   ├── deepstream-inspect-stack/            # Read-only DeepStream installation and compatibility diagnostic skill
│   ├── deepstream-generate-pipeline/       # Interactive gst-launch pipeline builder (BM25 retrieval over 270+ pipelines)
│   ├── deepstream-generate-config/         # deepstream-app / deepstream-test5 config generator (spec → render → validate)
│   ├── deepstream-audit-config/            # Audit/validate deepstream-app & deepstream-test5 configs
│   ├── deepstream-profile-pipeline/        # Nsight Systems profiling & config derivation skill
│   ├── deepstream-inspect-rtsp/            # RTSP ingest reliability diagnostic skill
│   ├── deepstream-import-vision-model/     # Unified detection/classification/segmentation/2D-pose/depth import skill
│   ├── deepstream-eval-and-finetune/       # DeepStream accuracy evaluation + AutoML fine-tuning skill
│   ├── deepstream-run-mv3dt/               # MV3DT reference-app operations skill
│   ├── deepstream-sop/                     # DeepStream SOP microservice skill (step-sequence compliance via GEBD + VLM)
│   ├── deepstream-setup-on-wsl/            # DeepStream-in-Docker setup and verification for WSL2
│   ├── deepstream-build-compute-stack/     # NVIDIA driver/CUDA/cuDNN/TensorRT host provisioning
│   ├── inference-builder/                   # Declarative GPU inference pipeline generation skill
│   ├── amc-run-auto-calibration/           # AutoMagicCalib setup, calibration, and custom-dataset tuning skill
│   ├── rtvi-cv-customize-model/            # Swap CV detection model in RTVI-CV (VSS Alerts Blueprint)
│   ├── rtvi-vlm-customize-model/           # Swap VLM in the VSS Alerts Blueprint
│   └── rtvi-cv-scaffold-vss-service/       # Scaffold a custom RTVI CV microservice for VSS
└── example_prompts/                        # Pre-built prompts for code generation
```

---

## Purpose

This project provides the tooling and reference material needed to:
1. **Supply domain-specific context** to AI coding assistants through a curated agentic skill
2. **Generate production-ready DeepStream code** using well-structured prompts
3. **Accelerate development** of video analytics pipelines with AI assistance

---

## Agentic Skills

An **agentic skill** is a structured knowledge package that an AI coding assistant can automatically discover and activate during code generation. It contains domain-specific rules, reference documentation, and guardrails that guide the AI agent to produce accurate, idiomatic code — without the developer needing to manually reference files in every conversation.

Each subdirectory under `skills/` contains a DeepStream agentic skill that follows the standard `SKILL.md` convention supported by AI coding assistants such as Cursor, Claude Code, and others.

This project ships **eighteen complementary skills**:

| Skill | Mode | Use when you want to… |
|-------|------|----------------------|
| [`deepstream-dev`](deepstream-dev/) | Reference-rich (you write code, the agent consults docs) | Hand-author or refine a `pyservicemaker` / GStreamer DeepStream pipeline with the agent answering API questions correctly. |
| [`deepstream-inspect-stack`](deepstream-inspect-stack/) | Read-only compatibility diagnosis + rerouting | Diagnose DeepStream installation, version, container, JetPack/CUDA/TensorRT/driver, GPU/Jetson, WSL2, and Python-binding mismatches; reroutes pipeline building and model import without generating implementation code. |
| [`deepstream-generate-pipeline`](deepstream-generate-pipeline/) | Interactive questionnaire + retrieval | Generate a ready-to-run `gst-launch-1.0` pipeline by answering a few questions; the agent retrieves and adapts from 270+ verified pipelines and validates the result. |
| [`deepstream-generate-config`](deepstream-generate-config/) | Interactive Q&A / spec → render → validate | Generate or repair a `deepstream-app` / `deepstream-test5-app` config (sources, streammux, GIE, tracker, sinks, RTSP/msgbroker) from quickstart, YAML spec, natural language, or guided Q&A — then validate and print the run command. |
| [`deepstream-audit-config`](deepstream-audit-config/) | Offline validate + approval-gated probes | Audit or sanity-check an existing `deepstream-app` / `deepstream-test5` config (sources, streammux, GIE/tracker, sinks, msgbroker, nvmultiurisrcbin, RTSP/NTP policy) and get a Go ahead / conditions / Do not use yet verdict. |
| [`deepstream-profile-pipeline`](deepstream-profile-pipeline/) | Measure-then-derive (Nsight Systems) | Build an efficient/performant pipeline or benchmark, tune, and measure FPS — the agent profiles with `nsys`, derives configs from the measured inference plateau batch and HW ceiling, and reports per-plugin NVTX timings. |
| [`deepstream-inspect-rtsp`](deepstream-inspect-rtsp/) | Diagnostic triage | Diagnose DeepStream and GStreamer RTSP ingest packet loss, latency, disconnects, frame drops, decoder saturation, artifacts, and camera/network faults without generating a production application. |
| [`deepstream-import-vision-model`](deepstream-import-vision-model/) | Unified category-aware orchestration | Preserve the native end-to-end object-detection workflow and route image classification, semantic segmentation, 2D pose, or monocular depth through fail-closed type adapters, parity gates, structured output, and visual validation. |
| [`deepstream-eval-and-finetune`](deepstream-eval-and-finetune/) | Accuracy evaluation + fine-tuning orchestration | Deploy an object-detection model to DeepStream (TensorRT FP16), measure mAP and performance on a KPI dataset through the real pipeline, fine-tune it with a single run or AutoML sweep, re-deploy, and produce a before/after report. |
| [`deepstream-run-mv3dt`](deepstream-run-mv3dt/) | Reference-app operations (the agent sets up and runs MV3DT) | Run the DeepStream Multi-View 3D Tracking reference app on shipped samples or synchronized MP4 datasets, including calibration handoff, Kafka metadata, and OSD/BEV outputs. |
| [`deepstream-sop`](deepstream-sop/) | Microservice scaffold + evaluate-and-fix loop | Build, deploy, evaluate, debug, or measure latency for a DeepStream SOP (Standard Operating Procedure) inference microservice — GPU-accelerated operator step-sequence compliance on industrial video via GEBD + VLM (Cosmos Reason 1/2), with file / RTSP / Basler camera inputs and SSE / Kafka output. |
| [`deepstream-setup-on-wsl`](deepstream-setup-on-wsl/) | Consent-first setup and verification | Set up and verify DeepStream in Docker inside WSL2 on a supported local or remote Windows workstation, with secure remote access and GPU compatibility gates. |
| [`deepstream-build-compute-stack`](deepstream-build-compute-stack/) | Version-pinned host provisioning | Provision an Ubuntu 24.04 x86_64 or ARM SBSA host with the NVIDIA open driver, CUDA, cuDNN, TensorRT, and NVIDIA Docker; optionally install DeepStream OS dependencies. |
| [`inference-builder`](inference-builder/) | Declarative pipeline generation | Generate deployable GPU inference pipelines and containers from YAML for DeepStream, Triton, vLLM, TensorRT-LLM, or PyTorch backends. |
| [`amc-run-auto-calibration`](amc-run-auto-calibration/) | AMC operations runbook | Launch the AutoMagicCalib service/UI, run sample, MP4, or RTSP calibration, and tune settings for unfamiliar or poor-quality custom datasets using release paths under `tools/auto-magic-calib`. |
| [`rtvi-cv-customize-model`](rtvi-cv-customize-model/) | Model swap + redeployment | Swap the CV detection model in RTVI-CV (VSS Alerts Blueprint verification mode) — covers ONNX staging, custom bbox parser, nvinfer config, runtime TRT engine build, and redeployment. |
| [`rtvi-vlm-customize-model`](rtvi-vlm-customize-model/) | VLM endpoint / deployment swap | Swap the VLM in the VSS Alerts Blueprint across all three consumers (`rtvi-vlm`, `vlm-as-verifier`, `vss-agent`) via OpenAI-compatible endpoint or in-container vLLM. |
| [`rtvi-cv-scaffold-vss-service`](rtvi-cv-scaffold-vss-service/) | Microservice scaffold | Scaffold a custom RTVI CV microservice (YOLO26 reference) that publishes detection metadata to VSS via Kafka `mdx-raw`. |

Skip ahead to [Skill: deepstream-import-vision-model](#skill-deepstream-import-vision-model) for the model-onboarding workflow, or [Skill: deepstream-run-mv3dt](#skill-deepstream-run-mv3dt) for the MV3DT reference-app workflow.

### Skill: deepstream-inspect-rtsp

`deepstream-inspect-rtsp` is a read-only diagnostic skill for DeepStream and GStreamer RTSP ingest reliability. It isolates the first failing stage—camera/network, jitterbuffer, decoder, streammux, sink/output, or lifecycle—before suggesting bounded configuration changes.

#### Installing the skill

```bash
# Example: Cursor user-level
cp -r skills/deepstream-inspect-rtsp ~/.cursor/skills/

# Example: Claude Code user-level
cp -r skills/deepstream-inspect-rtsp ~/.claude/skills/

# Example: Codex user-level
cp -r skills/deepstream-inspect-rtsp ~/.codex/skills/
```

Restart or reopen the coding assistant after copying the skill.

#### Example prompts

```text
My DeepStream RTSP cameras have rising latency and frame drops. Diagnose the first failing stage and give me a minimal test.
```

```text
DeepStream RTSP output is choppy in VLC from another workstation, but local fakesink FPS is healthy. What should I check first?
```

```text
My DeepStream app dynamically adds and removes RTSP sources and latency rises below the configured streammux batch size. Diagnose it without writing application code.
```

### Skill: deepstream-dev

This skill targets NVIDIA DeepStream SDK development using the Python `pyservicemaker` API. When activated, it instructs the AI agent to consult bundled reference documents before generating any code, significantly reducing inaccuracies and ensuring correct API usage.

**Bundled reference topics:**

| Reference | Coverage |
|-----------|----------|
| `gstreamer_plugins.md` | GStreamer plugin properties |
| `service_maker_api.md` | Pipeline/Flow API, metadata access, probes |
| `use_cases_pipelines.md` | Pipeline patterns: playback, multi-inference, cascaded GIE |
| `kafka_messaging.md` | Kafka/message broker setup and configuration |
| `best_practices.md` | Design patterns, pitfalls, anti-patterns |
| `buffer_apis.md` | BufferProvider/Feeder and BufferRetriever/Receiver |
| `media_extractor_advanced.md` | MediaExtractor, MediaChunk, FrameSampler |
| `utilities_config.md` | PerfMonitor, EngineFileMonitor, SourceConfig |
| `nvinfer_config.md` | nvinfer config file format and all parameters |
| `tracker_config.md` | nvtracker config (NvDCF, IOU, DeepSORT, NvSORT) |
| `troubleshooting.md` | Error messages and solutions |
| `rest_api_dynamic.md` | REST API, dynamic source management |
| `metamux_config.md` | nvdsmetamux config, parallel multi-model inference, metadata merging |
| `docker_containers.md` | Docker images, Dockerfile examples, pyservicemaker install, container run commands |
| `streaming_sources.md` | Streaming ingest sources: HTTP progressive, HLS, MPEG-DASH, RTSP |
| `nvds_msgapi_adapter.md` | nvds_msgapi protocol adapter interface for message brokers |

### Installing the Skill

Copy the `deepstream-dev` skill directory (including its `references/` subdirectory) into the skills folder recognized by your AI coding assistant. You can install it at the **user level** (available across all projects) or at the **workspace level** (scoped to a single project).

| Tool | User-level path | Workspace-level path |
|------|----------------|---------------------|
| Cursor | `~/.cursor/skills/deepstream-dev/` | `<workspace>/.cursor/skills/deepstream-dev/` |
| Claude Code | `~/.claude/skills/deepstream-dev/` | `<workspace>/.claude/skills/deepstream-dev/` |
| Codex | `~/.codex/skills/deepstream-dev/` | `<workspace>/.codex/skills/deepstream-dev/` |
| Other tools | Consult your tool's documentation for the skills directory location |

#### Step 1: Create the Skills Directory

```bash
# Example: Cursor user-level
mkdir -p ~/.cursor/skills/

# Example: Claude Code user-level
mkdir -p ~/.claude/skills/

# Example: Codex user-level
mkdir -p ~/.codex/skills/

# Example: workspace-level (replace .cursor with your tool's directory)
mkdir -p <workspace>/.cursor/skills/
```

#### Step 2: Copy the Skill

```bash
# User-level (replace path with your tool's skills directory)
cp -r skills/deepstream-dev ~/.cursor/skills/

# Example: Claude Code user-level
cp -r skills/deepstream-dev ~/.claude/skills/

# Example: Codex user-level
cp -r skills/deepstream-dev ~/.codex/skills/

# Or workspace-level
cp -r skills/deepstream-dev <workspace>/.cursor/skills/
```

After copying, the directory structure should look like:

```
<skills-directory>/
└── deepstream-dev/
    ├── SKILL.md              # Skill definition with rules and quick references
    └── references/           # Condensed reference documents
        ├── best_practices.md
        ├── buffer_apis.md
        ├── gstreamer_plugins.md
        ├── kafka_messaging.md
        ├── media_extractor_advanced.md
        ├── nvinfer_config.md
        ├── rest_api_dynamic.md
        ├── service_maker_api.md
        ├── tracker_config.md
        ├── troubleshooting.md
        ├── use_cases_pipelines.md
        ├── utilities_config.md
        ├── metamux_config.md
        ├── docker_containers.md
        ├── streaming_sources.md
        └── nvds_msgapi_adapter.md
```

#### Step 3: Verify the Installation

1. Open (or restart) your AI coding assistant.
2. Open the agent / chat panel.
3. Ask a DeepStream-related question, for example:
   ```
   Create a DeepStream pipeline that reads a video file and runs object detection using ResNet18.
   ```
4. The agent should automatically activate the `deepstream-dev` skill and consult its reference documents before generating code.

> **Tip:** The skill is most effective in **Agent mode**. In agent mode, the AI assistant automatically selects and activates relevant skills based on the task context — no manual file referencing needed.

---

### Skill: deepstream-generate-pipeline

`deepstream-generate-pipeline` builds ready-to-run `gst-launch-1.0` pipelines interactively. It collects pipeline requirements through a short questionnaire (input source, stream count, inference, tracker, sink, platform, extras), then assembles the pipeline using a standalone **BM25 retrieval backend** with structural metadata boosting over **270+ verified pipelines** — pure Python stdlib, zero external dependencies. The result is validated (syntax, elements, properties, live parse) before it is presented.

**Supported configurations:**

| Parameter | Options |
|-----------|---------|
| Input | Local video (.mp4/.h264/.h265), local image (.jpg/.png), RTSP, USB camera, test pattern |
| Inference | None, primary (nvinfer), primary+secondary, with preprocessor, Triton (nvinferserver) |
| Tracker | None, NvDCF, IOU, NvSORT, DeepSORT |
| Sink | Display (dGPU/Jetson), save (JPG/PNG/MP4/H264), RTSP out, fakesink |
| Platform | x86 dGPU (T4, A100, L40, RTX, …) or aarch64 — Jetson (Orin, Xavier, Nano) / SBSA (Grace, GH200) |
| Extras | Resize, rotate/flip, crop, color-format conversion |

**Bundled scripts:**

| Script | Purpose |
|--------|---------|
| `scripts/generate_pipeline.py` | BM25 retrieval engine — scores/ranks pipelines from `data/data.csv` |
| `scripts/validate_pipeline.py` | 4-stage validator: syntax, elements, properties, live parse |
| `scripts/lint_data.py` | Data-quality linter for the pipeline CSV (`--fix` to auto-repair) |

**Bundled references:** `assembly-rules.md`, `output-format.md`, `requirement-extraction.md`, `security-and-limitations.md`.

**Install** (same paths as `deepstream-dev`):

```bash
cp -r skills/deepstream-generate-pipeline ~/.cursor/skills/   # or ~/.claude/skills/ , ~/.codex/skills/
```

**Try it** — the skill activates from natural pipeline phrasing, no `@`-mention needed:

```text
detect and track on 4 rtsp streams and display on jetson
give me a pipeline to infer on an image
build a pipeline
```

> **Tip:** Works best in **Agent mode**. Run the bundled test suite with `python3 -m unittest discover -s <skill-path>/tests -v`.

---

### Skill: deepstream-generate-config

`deepstream-generate-config` generates ready-to-run **`deepstream-app` / `deepstream-test5-app` config files** from requirements. It drives a **spec → `render.py` → `validate.py` → run command** loop (never hand-writes configs in chat). Default mode is interactive Q&A; it also supports quickstart, YAML spec, and natural-language → spec.

**Requirements:** Python 3.10+ with the PyYAML version pinned in `scripts/requirements.txt`; DeepStream SDK 9.2.x is required to run the generated config.

**Modes:**

| Mode | When | Behavior |
|------|------|----------|
| A — Quickstart | "quickstart", "starter", "just run", "I'm new" | Renders bundled `assets/templates/quickstart_spec.yaml` |
| B — Spec file | `--spec`, pasted YAML, or a spec path | Renders the supplied spec |
| C — Natural language | Detailed prose with ≥3 inferable fields | Builds YAML per schema, confirms assumptions, then renders |
| D — Interactive Q&A (default) | Everything else | Guided questions from `references/interactive_qa.md` |

**Supported profiles** (overlay applied by `render.py`; invalid values are rejected):

| Profile | Goal |
|---------|------|
| `latency` | Lowest steady-state per-frame latency (FakeSink + nvdslogger defaults) |
| `perf` | Throughput / max FPS measurement |
| `display` | On-screen tiled display path |
| `custom` | No profile overlay — spec values only |

**Typical building blocks:** file / RTSP / V4L2 / CSI / `source-list` (`nvmultiurisrcbin`), streammux, primary/secondary GIE, NvDCF / IOU / NvSORT tracker, FakeSink / EGL / nv3dsink / file / RTSP-out / msgbroker sinks.

**Bundled scripts:**

| Script | Purpose |
|--------|---------|
| `scripts/render.py` | Spec → config (`--spec`, `--quickstart`, `--out`, `--print-resolved-spec`) |
| `scripts/validate.py` | Config linter before run |
| `scripts/spec_schema.yaml` | Spec field reference |
| `scripts/render_infer.py` / `scripts/render_tracker.py` | Infer / tracker fragment helpers |

**Bundled references:** `profiles.md`, `interactive_qa.md`, `natural_language_mapping.md`, `reference.md`, `examples.md`, `validator_pitfalls.md`.

**Dependency:** install the pinned PyYAML version with
`python3 -m pip install -r skills/deepstream-generate-config/scripts/requirements.txt`.

**Install** (same paths as `deepstream-dev`):

```bash
cp -r skills/deepstream-generate-config ~/.cursor/skills/   # or ~/.claude/skills/ , ~/.codex/skills/
```

**Try it:**

```text
generate a deepstream-app config for 4 RTSP cameras, YOLO detect, NvDCF tracker, latency profile
quickstart deepstream-app config
deepstream-app config from this spec.yaml
```

```bash
python3 scripts/render.py --quickstart --out ./quickstart_generated.txt
python3 scripts/validate.py ./quickstart_generated.txt
deepstream-app -c ./quickstart_generated.txt
```

> **Tip:** Always render via `scripts/render.py` and validate before running. Prefer a user-writable `--out` path (e.g. `./generated_config.txt`); do not write into root-owned sample dirs unless the user asks.

---

### Skill: deepstream-audit-config

`deepstream-audit-config` audits existing `deepstream-app` / `deepstream-test5` configs and proposed design changes. `deepstream-generate-config` owns creation and its render-time validation; activate the auditor afterward only for an explicitly requested independent audit or when the user supplies the config later for review. It translates plain-language requests ("validate this config for 4 RTSP cameras and Kafka") into `scripts/validate_config.py`, defaults to **offline/static** checks, and runs runtime/network probes (`--probe-*`) only after explicit approval.

**Do not use** when you need a brand-new config from requirements (`deepstream-generate-config`); do not load the auditor or run `validate_config.py` in that creation turn unless the user separately requests an independent audit. The repository has no bundled ONVIF-discovery skill; use `deepstream-inspect-rtsp` only to diagnose a known RTSP URI. Route nsys profiling to `deepstream-profile-pipeline`. Also do not use the auditor for NVBug fixes, custom GStreamer element code, or merely running a config that already passed validation (`deepstream-app -c`).

**Workflow:**

1. Collect the config path, pasted snippet, or design proposal.
2. Infer use case from conversation + config (`--intent`).
3. Run offline validation first (no `--probe-*`).
4. Ask for approval before any runtime/network probes.
5. Report **Go ahead**, **Go ahead with conditions**, or **Do not use yet** (2–3 lines); every ERROR/WARN gets a concrete fix.
6. Offer safe edits only after approval, then re-validate.

**Bundled script:**

| Script | Purpose |
|--------|---------|
| `scripts/validate_config.py` | Config audit + approval-gated runtime probes (`--intent`, `--usecase`, `--target-platform`, `--json`, `--check-runtime`, `--probe-*`, `--timeout-sec`) |

**Bundled references:**

| Reference | Coverage |
|-----------|----------|
| `rules.md` | Parser-backed validation rules (do not invent keys) |
| `probe_flags.md` | Runtime/network `--probe-*` flags |
| `rtp_ntp_policy.md` | RTSP / NTP timestamp policy |
| `config_edits.md` | Safe config edit patterns |
| `design_review.md` | Design / code-change review guidance |
| `examples.md` | Example findings (RTSP, msgbroker/test5, demux, tracker, probes) |

**Install** (same paths as `deepstream-dev`):

```bash
cp -r skills/deepstream-audit-config ~/.cursor/skills/   # or ~/.claude/skills/ , ~/.codex/skills/
```

**Try it:**

```text
validate this config for 4 RTSP cameras and Kafka
audit my deepstream-test5 config for nvmultiurisrcbin + REST stream add
review this streammux / source setup before I run it
```

See the [complete static-audit prompt](../example_prompts/deepstream_audit_config.md).

> **Tip:** Complements `deepstream-generate-config` — generate creates configs; audit validates them. Prefer fixing the generator/spec and re-rendering over hand-editing generated output in a loop.

---

### Skill: deepstream-profile-pipeline

`deepstream-profile-pipeline` replaces performance guesswork with measurement. When the user wants an **efficient / fast / performant** pipeline — or asks to **benchmark, tune, or measure FPS** — this skill profiles the pipeline with **Nsight Systems** (`nsys profile` + `nsys stats`, fully headless, no GUI), establishes two measured numbers (the **inference plateau batch** and the **HW ceiling**), and derives every other config from them. It then reports per-plugin **NVTX** timings for the end-to-end pipeline.

The skill is **model- and pipeline-agnostic**: it assumes only that the inference element is `nvinfer` or `nvinferserver`, and reads the user's actual config to discover model dims, target FPS, and source properties. It works for detection (with or without tracker), classification, segmentation, VLM, and embedding pipelines, with file / RTSP / USB-camera sources.

**Bundled references:**

| Reference | Coverage |
|-----------|----------|
| `nvtx-coverage.md` | Per-plugin NVTX range coverage and interpretation |
| `hw-ceiling-formulas.md` | HW ceiling derivation formulas |
| `config-derivation-rules.md` | Deriving pipeline configs from measured numbers |
| `nsys-cli-recipes.md` | `nsys profile` / `nsys stats` CLI recipes |
| `boundedness-rules.md` | Compute- vs memory- vs IO-boundedness rules |

> **Container requirement:** Run from the `nvcr.io/nvidia/deepstream:9.0-triton-multiarch` dev image. The slimmer `samples-multiarch` variant strips the nsys NVTX injector and produces empty per-plugin traces — do not use it for profiling. Requires `nsys` (Nsight Systems 2024+) and `nvidia-smi` on `PATH`.

**Install** (same paths as `deepstream-dev`):

```bash
cp -r skills/deepstream-profile-pipeline ~/.cursor/skills/   # or ~/.claude/skills/ , ~/.codex/skills/
```

**Try it:**

```text
build an efficient pipeline for this GPU
how many streams can this GPU handle at 30 FPS?
profile and tune my deepstream pipeline
```

> **Tip:** Works best in **Agent mode** — it activates when a prompt carries efficiency/benchmarking intent.

---

### Skill: deepstream-import-vision-model

`deepstream-import-vision-model` is an **autonomous** skill: instead of helping you write code, it executes a complete model bring-up pipeline and hands you back a benchmarked TensorRT engine plus a publication-ready PDF report.

**Pipeline (runs unattended):**

1. **Model Acquire** — parses a HuggingFace or NVIDIA NGC URL, downloads ONNX (or exports SafeTensors → ONNX via `optimum-cli`), extracts labels.
2. **Engine Build** — builds a dynamic TensorRT engine via `trtexec`, with iterative batch-size scaling and warm-cache reuse.
3. **DeepStream Pipeline** — generates a custom `nvinfer` bbox parser, builds the `.so`, runs single-stream KITTI validation, then a multi-stream sweep.
4. **Report** — produces 5 benchmark charts and a Markdown / HTML / PDF report under `models/<model_name>/reports/`.

**Supported model scope:** Object detection models only. Classification, segmentation, pose estimation, and other vision tasks are rejected up front (architecture detected from `config.json` — downloaded for HuggingFace models, extracted from the archive for NGC models).

The following detection architecture families are supported:

- **Transformer-based detectors** — query-based encoder-decoder designs (e.g., DETR and RT-DETR family, including TAO Transformer variants from NGC)
- **One-stage grid-based detectors** — single-pass, anchor-based or anchor-free designs that predict boxes directly from spatial feature grids (e.g., YOLO family)
- **Open-vocabulary / zero-shot detectors** — vision-language models that localize objects described by free-form text queries at inference time (e.g., GroundingDINO / OWL-ViT)

Models that fall outside these families are untested; custom bbox parsers may need manual adjustment for novel output tensor layouts.

#### Bundled references

| Reference | Coverage |
|-----------|----------|
| `model-acquire.md` | HF / NGC URL parsing, ONNX vs SafeTensors detection, optimum export, label extraction |
| `engine-build.md` | `trtexec` flags, dynamic shapes, batch-size scaling, timing-cache reuse, PEAK_GPU_STREAMS derivation |
| `pipeline-run.md` | Custom nvinfer bbox parser (with the mandatory `obj = {}` zero-init), single-stream KITTI validation, multi-stream sweep |
| `report-generation.md` | `benchmark_data.json` schema, 5-chart generation, 12-section Markdown report, HTML + PDF render via `wkhtmltopdf` |

#### Installing the skill

Same install paths as `deepstream-dev`:

```bash
# Example: Cursor user-level
cp -r skills/deepstream-import-vision-model ~/.cursor/skills/

# Example: Claude Code user-level
cp -r skills/deepstream-import-vision-model ~/.claude/skills/

# Example: Codex user-level
cp -r skills/deepstream-import-vision-model ~/.codex/skills/

# Or workspace-level
cp -r skills/deepstream-import-vision-model <workspace>/.cursor/skills/
```

After copying:

```text
<skills-directory>/
└── deepstream-import-vision-model/
    ├── SKILL.md            # Top-level skill definition + critical rules
    ├── references/         # 4 phase references (model-acquire, engine-build, pipeline-run, report-generation)
    └── scripts/            # Helpers: model/, engine/, deepstream/, report/
```

#### Verifying the installation

1. Open (or restart) your AI coding assistant on a workspace where you want bring-up artifacts to land (the skill writes to `models/<model_name>/` relative to the project root).
2. Ask:

   ```text
   Use deepstream-import-vision-model to onboard and benchmark this detection model
   end-to-end, and produce the PDF benchmark report:
   https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/rtdetr_2d_warehouse

   Use the default sample video at
   /opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4.
   ```

   In Cursor:

   ```text
   @deepstream-import-vision-model onboard and benchmark this detection model end-to-end,
   and produce the PDF benchmark report:
   https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/rtdetr_2d_warehouse

   Use the default sample video at
   /opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4.
   ```

   For an interactive variant that prompts for inputs with defaults, see [`example_prompts/import_vision_model_detection_pipeline.md`](../example_prompts/import_vision_model_detection_pipeline.md).

3. The agent should activate `deepstream-import-vision-model`, run the containerized setup and preflight checks (Docker, GPU, DeepStream 9.1, Python environment, TensorRT, and report tooling), and proceed through Steps 1–8 without further prompting.

#### Output structure

```text
models/<model_name>/
  model/          # ONNX file(s)
  parser/         # custom nvinfer bbox parser (.cpp, .so)
  config/         # nvinfer config, ds-app config, labels.txt
  benchmarks/     # TRT engines + trtexec / DS logs
  reports/        # benchmark_report.md / .html / .pdf + charts/
  samples/        # output .mp4, KITTI detections, test frames
```

The final PDF (`reports/benchmark_report_<model_name>.pdf`) being **>500 KB** is the skill's own success signal that charts were embedded correctly.

> **Tip:** Like `deepstream-dev`, this skill works best in **Agent mode**. Manual `@`-mention is not required after install — the assistant picks it up from the URL pattern in your prompt.

---

### Skill: deepstream-run-mv3dt

`deepstream-run-mv3dt` operates the DeepStream Multi-View 3D Tracking reference app in `src/apps/reference_apps/deepstream-tracker-3d-multi-view`. It supports setup checks, shipped 4-camera and 12-camera sample runs, synchronized MP4 datasets, AutoMagicCalib handoff for missing calibration, display or headless execution, OSD/BEV outputs, and Kafka metadata inspection.

#### Installing the skill

Same install paths as `deepstream-dev`:

```bash
# Example: Cursor user-level
cp -r skills/deepstream-run-mv3dt ~/.cursor/skills/

# Example: Claude Code user-level
cp -r skills/deepstream-run-mv3dt ~/.claude/skills/

# Example: Codex user-level
cp -r skills/deepstream-run-mv3dt ~/.codex/skills/

# Or workspace-level
cp -r skills/deepstream-run-mv3dt <workspace>/.cursor/skills/
```

#### Verifying the installation

Ask for an MV3DT workflow, for example:

```text
Run the DeepStream MV3DT 4-camera sample headlessly and save output artifacts.
```

The agent should activate `deepstream-run-mv3dt`, resolve the MV3DT reference-app directory, run preflight checks, request approval before privileged Docker launch commands, and verify current-run artifacts rather than stale files.

#### Example prompts

```text
Deploy MV3DT on the default sample dataset.
```

```text
Run the MV3DT 12-camera sample headlessly with RTDETR and save videos.
```

```text
Run MV3DT on synchronized MP4s under /path/to/my-dataset using PeopleNetTransformer.
```

```text
My MV3DT videos under /path/to/my-dataset do not have calibration. Use AutoMagicCalib with resnet, then run MV3DT.
```

---

### Skill: deepstream-sop

`deepstream-sop` is a domain-specific skill for building, deploying, evaluating, debugging, and measuring latency on the **DeepStream SOP (Standard Operating Procedure) Inference Microservice** — a GPU-accelerated FastAPI service that combines GEBD (Generic Event Boundary Detection, e.g. DDM) with VLM-based step classification (Cosmos Reason 1/2 via embedded vLLM) to verify operator step sequence on industrial video. It supports file, RTSP, and Basler GigE camera inputs, and emits results over SSE and/or Kafka.

The skill ships an 18-section progressive-disclosure reference layout: the agent loads only the sections relevant to the current task (generate, evaluate, latency-measurement, etc.), not the whole thing.

**Bundled reference topics:**

| Reference | Coverage |
|-----------|----------|
| `skill_01_fastapi_endpoints.md` | FastAPI endpoints, server init, Prometheus metrics |
| `skill_02_pydantic_schemas.md` | Request/response Pydantic models |
| `skill_03_deepstream_pipeline.md` | DeepStream pyservicemaker pipeline + tensor parser |
| `skill_04_config_templates.md` | `nvdspreprocess` / `nvinferserver` templates |
| `skill_05_triton_ddm_model.md` | Triton model repo, `config.pbtxt`, GEBD model swap |
| `skill_05b_custom_postprocess.md` | C++ postprocess plugin (DeepStream `IOptions` API) |
| `skill_06_sop_process_manager.md` | `SOPProcessManager`, `SOPVideoProcessor`, VLM wiring |
| `skill_06b_sop_checker.md` | Cycle detection, missing / mis-ordered step compliance |
| `skill_07_sse_streaming.md` | SSE generator, dummy test mode |
| `skill_08_basler_camera.md` | Basler GigE camera, Pylon SDK, camera emulation |
| `skill_09_docker_build_deploy.md` | Docker build, deploy, `.env` configuration |
| `skill_10_test_suite.md` | Test suite coverage, assertions |
| `skill_11_env_variables.md` | All environment variables |
| `skill_12_evaluation_workflow.md` | Generate-evaluate-iterate workflow with watchdog |
| `skill_13_verification_curl.md` | Verification steps + curl examples |
| `skill_14_implementation_checklist.md` | File copy map + Docker prereqs + post-build verification |
| `skill_15_latency_measurement.md` | File-input TTFC and C2C latency measurement |
| `skill_16_message_schema.md` | Kafka message schema (JSON / NvProto) |
| `skill_17_camera_latency_measurement.md` | Live-stream `chunk_e2e` latency measurement |
| `skill_18_rtsp_streaming_output.md` | RTSP streaming output as a natively generated feature |

#### Example prompts

Generate the microservice scaffold:

```text
Please follow @references/example_sop_prompt.md to generate a SOP microservice
in a folder named ds_sop_microservice.
```

Evaluate the microservice and fix issues until the test suite is green:

```text
Please follow @references/eval_sop_prompt.md to evaluate the SOP microservice
in ds_sop_microservice. Fix any issues found.
```

Live-camera question — the skill activates without being named:

```text
How do I send a chat completion request using a Basler GigE camera with serial 40748152?
```

Measure file-input latency:

```text
Measure TTFC and C2C latency on /path/to/test_video.mp4 — follow the deepstream-sop skill.
```

> **Tip:** Like the other skills, `deepstream-sop` works best in **Agent mode** — the assistant picks it up automatically when a prompt matches its trigger list (SOP / step-sequence / VLM-on-video / Basler camera / SOP-checker, etc.).

---

### Skill: AutoMagicCalib

`amc-run-auto-calibration` is the single DeepStream skill for AutoMagicCalib setup, calibration, and custom-dataset tuning workflows. It uses the release AMC checkout under `tools/auto-magic-calib/` and routes details to reference files for service setup, the SDG sample dataset, local MP4 calibration, RTSP/VIOS capture, bounded parameter tuning, and troubleshooting.

This is a runtime-operation skill. Running setup or calibration requires a host with an NVIDIA GPU with NVENC, NVIDIA driver, Docker without sudo, NVIDIA Container Toolkit, and NGC image access. Optional independent VGGT calibration also requires HuggingFace access and the VGGT model. In restricted or no-GPU sandboxes, use the skill for planning and command preparation only.

| Skill | What it does |
|-------|--------------|
| [`amc-run-auto-calibration`](amc-run-auto-calibration/SKILL.md) | Launches the AMC service/UI, verifies `/v1/ready`, runs the bundled sample, calibrates user `cam_*.mp4` videos, handles RTSP capture through VIOS, and runs approved sequential experiments to recommend custom-dataset settings with per-attempt evidence. |

#### Installing the AMC skill

Copy the AMC skill directory into your assistant's skill directory. Examples:

```bash
# Claude Code user-level
mkdir -p ~/.claude/skills
cp -r skills/amc-run-auto-calibration ~/.claude/skills/

# Codex user-level
mkdir -p ~/.codex/skills
cp -r skills/amc-run-auto-calibration ~/.codex/skills/
```

Restart or reopen the coding assistant after copying the skill.

#### Example prompts

```text
launch auto calibration
```

```text
calibrate sample dataset
```

```text
calibrate these videos - /path/to/NV-Warehouse-4Cam/videos
```

```text
calibrate these streams - rtsp://<host>:<port>/<path>/cam_00.mp4, rtsp://<host>:<port>/<path>/cam_01.mp4, rtsp://<host>:<port>/<path>/cam_02.mp4. Use resnet; I will upload alignment/layout/settings in the UI.
```

---

### Skill: rtvi-cv-customize-model

`rtvi-cv-customize-model` covers swapping the DeepStream CV detection model in the VSS Alerts Blueprint verification (`2d_cv`) mode. It walks through ONNX staging, custom bbox parser build and symbol alignment, nvinfer config, runtime TRT engine build via `trtexec`, and redeployment of `perception-alerts`. YOLO11 is the reference example; the same steps apply to any ONNX-format detector.

#### Installing the skill

```bash
# Claude Code user-level
cp -r skills/rtvi-cv-customize-model ~/.claude/skills/

# Codex user-level
cp -r skills/rtvi-cv-customize-model ~/.codex/skills/
```

#### Example prompts

```text
Replace the VSS Alerts Blueprint verification-mode detector with a custom ONNX model and redeploy perception-alerts.
```

```text
How do I use a different detection model with the VSS Alerts Blueprint?
```

---

### Skill: rtvi-vlm-customize-model

`rtvi-vlm-customize-model` covers swapping the VLM in the VSS Alerts Blueprint. It handles all three VLM consumers (`rtvi-vlm`, `vlm-as-verifier`, `vss-agent`), both deployment methods (OpenAI-compatible endpoint and in-container vLLM), and health checks.

#### Installing the skill

```bash
# Claude Code user-level
cp -r skills/rtvi-vlm-customize-model ~/.claude/skills/

# Codex user-level
cp -r skills/rtvi-vlm-customize-model ~/.codex/skills/
```

#### Example prompts

```text
Point the VSS Alerts Blueprint at a host-side NIM on http://host.docker.internal:30082 for all RTVI-VLM calls.
```

```text
Run rtvi-vlm standalone with Qwen3-VL-8B-Instruct served inside the container.
```

---

### Skill: rtvi-cv-scaffold-vss-service

`rtvi-cv-scaffold-vss-service` scaffolds a deployable custom RTVI CV microservice that plugs into VSS Search and Alerts profiles via Kafka `mdx-raw`. The shipped scaffold is a YOLO26 reference implementation and generates a runnable repo with a Dockerfile, DeepStream pipeline configs, compose definition, Python adapter, tests, and a smoke-test consumer.

#### Installing the skill

```bash
# Claude Code user-level
cp -r skills/rtvi-cv-scaffold-vss-service ~/.claude/skills/

# Codex user-level
cp -r skills/rtvi-cv-scaffold-vss-service ~/.codex/skills/
```

#### Example prompts

```text
Create a custom RTVI CV microservice that runs YOLO26 in DeepStream and publishes object metadata so VSS Search and Alerts can consume it.
```

```text
Scaffold a custom perception microservice for VSS Alerts using my YOLO26 ONNX and wire it into mdx-raw.
```

---

## Using Example Prompts

The `example_prompts/` directory contains pre-built prompts for generating DeepStream applications. Each prompt file provides a complete specification that an AI agent can follow to produce working code.

> **Getting started?** Begin with `video_infer_app.md` for a minimal single-stream inference example, then progress to `multi_stream_tracker.md` for multi-stream and tracking capabilities.

### Available Prompts

| Prompt File | Purpose |
|-------------|---------|
| `multi_stream_tracker.md` | Multi-stream RTSP app with tracker and 2x2 tiled display |
| `rtvi_vlm_core_app.md` | Complete RTSP video processing app with VLM integration |
| `rtvi_vlm_openapi_spec.md` | FastAPI microservice with OpenAPI specification. Should be used after the core app is generated using @rtvi_vlm_core_app.md |
| `video_infer_app.md` | Basic video file inference with bounding box display |
| `video_object_count.md` | Video inference with object detection counting |
| `video_parallel_infer_app.md` | Parallel multi-model inference with demux stream selection and metadata merging |
| `yolov26s_detection.md` | YOLOv26s model download, ONNX export, and custom parsing library |
| `nvdsdynamicsrcbin_app.md` | Use of nvdsdynamicsrcbin plugin |
| `nvdsanalytics_config_sample.md` | Video inference with nvdsanalytics — ROI filtering, line-crossing, overcrowding, and direction detection |
| `msgconv_kafka.md` | Video inference with message converter sending detection results to Kafka |
| `msgbroker_nats.md` | Publish detection results to NATS via a custom NATS protocol adapter (JetStream, auth, TLS) |
| `ds_profiling_efficient_pipeline.md` | Build and profile an efficient multi-stream RTSP pipeline; report bottleneck and max sustainable streams (pairs with `deepstream-profile-pipeline`) |
| `import_vision_model_detection_pipeline.md` | Interactive detection-model onboarding & benchmark prompt (pairs with `deepstream-import-vision-model`) |
| `tiled_inference_accuracy.md` | Compare full-frame and tiled detection throughput and detection-count proxies |
| `single_view_3d_tracker.md` | The single-view 3D tracking. Given the camera matrix and human model of a static camera, estimates and keeps tracking of object states in the 3D physical world |

---

### Step-by-Step Guide: Using Prompts

#### Step 1: Open the AI Chat / Agent Panel

Open the agent or chat panel in your AI coding assistant. Most tools provide a keyboard shortcut or sidebar icon for this.

#### Step 2: Reference the Prompt File

Use your tool's file-referencing feature (e.g., `@` mentions) to include the prompt file:

```
@example_prompts/rtvi_vlm_core_app.md
```

Or simply type `@` and start typing the filename to search.

#### Step 3: Execute the Prompt

**Option A: Direct execution**

Reference the file in the chat and instruct the agent to follow it:

```
Follow the instructions in @example_prompts/rtvi_vlm_core_app.md to generate the application.
```

**Option B: Incremental execution**

For complex prompts, break them into smaller steps:

```
Based on @example_prompts/rtvi_vlm_core_app.md, first implement the vLLM backend module.
```

Then follow up with:
```
Now implement the frame selection logic as described in the prompt.
```

#### Step 4: Review and Iterate

1. Review the generated code in the diff view.
2. Accept or reject individual changes.
3. Ask follow-up questions for refinements:
   ```
   Can you optimize the GPU memory usage in the generated stream_processor.py?
   ```

---

### Example Workflow: Generating the RTVI Application

Here's a complete workflow for generating the RTVI VLM application:

**1. Generate Core Application**
```
@example_prompts/rtvi_vlm_core_app.md

Generate the complete application following these instructions.
```

**2. Add FastAPI Microservice**
```
@example_prompts/rtvi_vlm_openapi_spec.md

Create the FastAPI server with all endpoints shown in @rtvi_vlm_openapi_spec.png
```

---

## Best Practices for AI-Assisted Development

### Writing Effective Prompts

1. **Be specific** — Include exact requirements, constraints, and expected outputs
2. **Reference context** — Use `@` mentions to include relevant files and documents
3. **Break down complex tasks** — Divide large features into smaller, focused prompts
4. **Include examples** — Show expected input/output formats when applicable
5. **Specify the deployment target** — Mention whether the application targets dGPU (x86_64) or Jetson (ARM64), as pipeline elements and sink choices may differ

### Iterating on Generated Code

1. **Review before accepting** — Always inspect generated pipelines for correct element linking and property values
2. **Test incrementally** — Run the pipeline after each major change rather than building the entire application at once
3. **Use the troubleshooting reference** — If a pipeline fails, ask the agent to consult `troubleshooting.md` for known error patterns
4. **Provide error output** — When debugging, paste the full GStreamer or DeepStream error log into the chat for more accurate fixes

---

## Demo Video

<a href="https://www.youtube.com/watch?v=ZQTX7MeN7mI" target="_blank" rel="noopener noreferrer">
  <picture>
    <img src="https://img.youtube.com/vi/ZQTX7MeN7mI/maxresdefault.jpg" alt="Build Vision AI Pipelines with DeepStream Coding Agents" width="560" style="border-radius:8px">
  </picture>
</a>
