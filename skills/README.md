# DeepStream Coding Agent

A project showcasing how to leverage **AI coding assistants** (Cursor, Claude Code, etc.) for accelerated [NVIDIA DeepStream SDK](https://developer.nvidia.com/deepstream-sdk) application development using a curated agentic skill and structured prompts.

> **Disclaimer:** Code generated with AI coding assistants is intended as a development starting point. All generated code must undergo your full software development lifecycle (SDLC) — including code review, testing, and security validation — before production use.

---

## Prerequisites

### For code generation (using the skill and prompts)

- **AI coding assistant** that supports agentic skills (e.g., Cursor, Claude Code)

No GPU, SDK, or special hardware is required — the skill and example prompts work on any system.

### For running the generated code

The following are required on the target execution environment:

- **NVIDIA DeepStream SDK 9.0** — installed locally or available via [NVIDIA NGC container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/deepstream)
- **Python 3.12+** with the `pyservicemaker` package
- **NVIDIA GPU** with driver version 590 or later
- **CUDA 13.1** and **TensorRT 10.14.1.48**
- **Supported OS:** Ubuntu 24.04 (x86_64 or ARM64/Jetson)

> The `deepstream-import-vision-model` skill needs a few extra runtime tools (`trtexec`, `wkhtmltopdf`, `mediainfo`, `deepstream-app`, an `optimum`-capable Python venv). They are listed and auto-checked by the pre-flight script in [`skills/deepstream-import-vision-model/SKILL.md`](deepstream-import-vision-model/SKILL.md#pre-flight-checks).

> For detailed environment setup, refer to the [DeepStream SDK Developer Guide](https://docs.nvidia.com/metropolis/deepstream/dev-guide/).

---

## Project Structure

This README sits under `skills/` inside the DeepStream mono-repo. Layout of this subtree, plus the related top-level `example_prompts/` directory:

```
deepstream/                                 # mono-repo root
├── skills/                                 # Agentic skills for guided DeepStream development
│   ├── README.md                           # This file
│   ├── deepstream-dev/                     # DeepStream development skill with condensed references
│   ├── deepstream-generate-pipeline/       # Interactive gst-launch pipeline builder (BM25 retrieval over 270+ pipelines)
│   ├── deepstream-profile-pipeline/        # Nsight Systems profiling & config derivation skill
│   ├── deepstream-import-vision-model/     # Autonomous vision-model onboarding & benchmarking pipeline skill
│   ├── deepstream-sop/                     # DeepStream SOP microservice skill (step-sequence compliance via GEBD + VLM)
│   ├── amc-setup-calibration-stack/        # AutoMagicCalib MS + UI launch skill
│   ├── amc-run-sample-calibration/         # AutoMagicCalib bundled-sample validation skill
│   └── amc-run-video-calibration/          # AutoMagicCalib user-video calibration skill
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

This project ships **eight complementary skills**:

| Skill | Mode | Use when you want to… |
|-------|------|----------------------|
| [`deepstream-dev`](deepstream-dev/) | Reference-rich (you write code, the agent consults docs) | Hand-author or refine a `pyservicemaker` / GStreamer DeepStream pipeline with the agent answering API questions correctly. |
| [`deepstream-generate-pipeline`](deepstream-generate-pipeline/) | Interactive questionnaire + retrieval | Generate a ready-to-run `gst-launch-1.0` pipeline by answering a few questions; the agent retrieves and adapts from 270+ verified pipelines and validates the result. |
| [`deepstream-profile-pipeline`](deepstream-profile-pipeline/) | Measure-then-derive (Nsight Systems) | Build an efficient/performant pipeline or benchmark, tune, and measure FPS — the agent profiles with `nsys`, derives configs from the measured inference plateau batch and HW ceiling, and reports per-plugin NVTX timings. |
| [`deepstream-import-vision-model`](deepstream-import-vision-model/) | Autonomous orchestration (the agent runs an end-to-end pipeline) | Take any HuggingFace or NGC object-detection model and produce a TensorRT engine, a DeepStream multi-stream benchmark, and a PDF report — fully unattended. |
| [`deepstream-sop`](deepstream-sop/) | Microservice scaffold + evaluate-and-fix loop | Build, deploy, evaluate, debug, or measure latency for a DeepStream SOP (Standard Operating Procedure) inference microservice — GPU-accelerated operator step-sequence compliance on industrial video via GEBD + VLM (Cosmos Reason 1/2), with file / RTSP / Basler camera inputs and SSE / Kafka output. |
| [`amc-setup-calibration-stack`](amc-setup-calibration-stack/) | Deployment runbook | Launch the AutoMagicCalib microservice and web UI from NGC release images via Docker Compose. |
| [`amc-run-sample-calibration`](amc-run-sample-calibration/) | Validation runbook + script | Verify a running AMC stack with the bundled synthetic sample dataset. |
| [`amc-run-video-calibration`](amc-run-video-calibration/) | Calibration runbook + script | Calibrate a camera rig from user-provided pre-recorded MP4 files via the AMC REST API. |

Skip ahead to [Skill: deepstream-import-vision-model](#skill-deepstream-import-vision-model) for the model-onboarding workflow.

### Skill: deepstream-dev

This skill targets NVIDIA DeepStream SDK 9.0 development using the Python `pyservicemaker` API. When activated, it instructs the AI agent to consult bundled reference documents before generating any code, significantly reducing inaccuracies and ensuring correct API usage.

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
        └── docker_containers.md
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

3. The agent should activate `deepstream-import-vision-model`, run pre-flight checks (`nvidia-smi`, `trtexec`, `wkhtmltopdf`, `mediainfo`, `deepstream-app`), and proceed through Steps 1–8 without further prompting.

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

### AutoMagicCalib Skills

The AMC skills provide guided AutoMagicCalib setup and calibration workflows. When these skills mention the root `README.md`, `assets/`, `compose/`, `projects/`, or `models/`, they mean the resolved `auto-magic-calib` checkout. The setup skill finds an existing checkout or asks before cloning one.

| Skill | What it does |
|-------|--------------|
| [`amc-setup-calibration-stack`](amc-setup-calibration-stack/SKILL.md) | Launches the AMC microservice and UI from NGC release images, handles NGC login, resolves ports, configures `compose/.env`, verifies `/v1/ready`, and reports MS, Swagger, and UI URLs. |
| [`amc-run-sample-calibration`](amc-run-sample-calibration/SKILL.md) | Runs the shipped sample dataset (`assets/sdg_08_2_sample_data_010926.zip`) through the AMC REST API using the bundled Python script or a documented Swagger UI path. |
| [`amc-run-video-calibration`](amc-run-video-calibration/SKILL.md) | Runs calibration for user-supplied `cam_*.mp4` videos through the AMC REST API, with local config/alignment/layout detection, UI fallback, optional GT, focal lengths, detector selection, and optional VGGT refinement. |

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
| `msgconv_kafka.md` | Video inference with message converter sending detection results to Kafka |
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

