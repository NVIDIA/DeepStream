# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Configuration + prompt catalog for the DeepStream code-agent launchable.

Pure data/defaults: paths, container/image names, service endpoints, timeouts, credential
defaults (overwritten at runtime by ds_agent_lab in the Install step), and the PROMPT_CATALOG (1:1 with
example_prompts/*.md) plus its accessors. ds_agent_lab does
`from ds_lab_config import *`, so these names are part of the `lab.*` surface; runtime mutation
happens on ds_agent_lab's own bindings (a `global X` there rebinds its copy, not this default).
"""
from pathlib import Path
import os


# ---------------------------------------------------------------------------
# Image / paths / container (override via env; sane defaults).
#
# REPO_ROOT is derived from THIS module's own location (deploy/brev/scripts/),
# never from cwd, so skills/example_prompts always resolve regardless of where
# the notebook kernel is launched.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[3]

DEEPSTREAM_IMAGE = os.environ.get(
    "DEEPSTREAM_IMAGE", "nvcr.io/nvidia/deepstream:9.1-triton-multiarch"
)
AGENT_CONTAINER = os.environ.get("AGENT_CONTAINER", "ds-agent-work")
# Outputs default to a dir NEXT TO the notebook (deploy/brev/outputs) -- still under the repo root
# (= the Jupyter server's root_dir), so the generated code and run results appear in the left-hand
# FILE BROWSER and survive a page refresh (real files, not just widget output). Override OUTPUT_ROOT
# to relocate elsewhere (e.g. outside the tree); the `outputs/` dir is gitignored so the tree stays clean.
OUTPUT_ROOT = Path(
    os.environ.get("OUTPUT_ROOT", REPO_ROOT / "deploy" / "brev" / "outputs")
).resolve()
WORKSPACE = OUTPUT_ROOT / "workspace"

# Container-side paths (constants reused by the helpers).
CTR_WORKSPACE = "/workspace"
CTR_AGENT_OUTPUTS = "/workspace/agent_outputs"
CTR_EXAMPLE_PROMPTS = "/workspace/example_prompts"

# ---- Agent + prompt selection (set by the step widgets or fallbacks) ----
AGENT = os.environ.get("AGENT", "").strip().lower()  # "claude" | "codex"
SELECTED_PROMPT_ID = os.environ.get("SELECTED_PROMPT_ID", "video_infer_app")
CUSTOM_PROMPT = os.environ.get(
    "CUSTOM_PROMPT", ""
).strip()  # pasted text overrides the catalog id
CUSTOM_OUTPUT_DIR = os.environ.get("CUSTOM_OUTPUT_DIR", "custom_app")

# ---- Demo + service settings ----
KAFKA_TOPIC = os.environ.get("KAFKA_TOPIC", "deepstream-agent-demo")
KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "127.0.0.1:9092")
NATS_URL = os.environ.get("NATS_URL", "nats://127.0.0.1:4222")
NATS_SUBJECT = os.environ.get("NATS_SUBJECT", "deepstream.detections")
MODEL_NAME = os.environ.get("MODEL_NAME", "rtdetr_2d_warehouse")
MODEL_IMPORT_URL = os.environ.get(
    "MODEL_IMPORT_URL",
    "https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/rtdetr_2d_warehouse",
)
SAMPLE_VIDEO = os.environ.get(
    "SAMPLE_VIDEO", "/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4"
)

# ---- Service container images (pinned defaults; override via env) ----
# (No ffmpeg image: RTSP loopers run inside the DeepStream work container, which ships ffmpeg.)
MEDIAMTX_IMAGE = os.environ.get("MEDIAMTX_IMAGE", "bluenviron/mediamtx:1.11.3")
KAFKA_IMAGE = os.environ.get("KAFKA_IMAGE", "apache/kafka:3.9.1")
NATS_IMAGE = os.environ.get("NATS_IMAGE", "nats:2.10-alpine")
NATS_BOX_IMAGE = os.environ.get("NATS_BOX_IMAGE", "natsio/nats-box:0.14.5")

# ---- Bounded-demo controls so RTSP/streaming prompts never hang forever ----
DEMO_MAX_SECONDS = int(os.environ.get("DEMO_MAX_SECONDS", "30"))
RUN_DEMO_TIMEOUT = int(
    os.environ.get("RUN_DEMO_TIMEOUT", "1800")
)  # 30 min: run_demo.sh now also does first-run setup (venv/deps/model/engine)
# import_vision runs its ENTIRE end-to-end skill (download -> engine build -> multi-stream
# benchmark sweep -> PDF report) during Generate, so it needs a far larger budget than the
# write-only prompts (AGENT_TIMEOUT=1200s) -- otherwise it times out mid-benchmark and fails.
IMPORT_VISION_TIMEOUT = int(os.environ.get("IMPORT_VISION_TIMEOUT", "5400"))  # 90 min

# Optional runtime input the user types in the Run step (a video file path or RTSP URL).
# Empty = let the generated run_demo.sh use its own default (sample video / RTSP).
DEMO_INPUT = os.environ.get("DEMO_INPUT", "")

# ---- Local service endpoints injected into the agent prompt (avoid port clashes) ----
RTSP_BASE = os.environ.get(
    "RTSP_BASE", "rtsp://127.0.0.1:8554"
)  # streams: 8554/cam0 .. 8554/cam3
VLM_ENDPOINT = os.environ.get(
    "VLM_ENDPOINT", "http://127.0.0.1:8000/v1"
)  # vLLM OpenAI-compatible
VLM_MODEL = os.environ.get(
    "VLM_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct"
)  # qwen2_5_vl: needs vLLM >= v0.7.3 (see VLM_IMAGE in serve_vlm.sh); ~7GB, fits A40 (46GB)
RTVI_SERVICE_PORT = int(
    os.environ.get("RTVI_SERVICE_PORT", "8080")
)  # rtvi FastAPI (8000 is the VLM)

# ---- Non-root agent user created INSIDE the running container by the Install step ----
AGENT_UID = int(os.environ.get("AGENT_UID", "1000"))
AGENT_GID = int(os.environ.get("AGENT_GID", "1000"))
AGENT_HOME = os.environ.get("AGENT_HOME", "/home/agent")

# ---- Credentials (RUNTIME only; never baked into an image, never at `docker run`) ----
# Defaults come from the environment; the Install step / account sign-in overwrites these globals.
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_BASE_URL = os.environ.get(
    "ANTHROPIC_BASE_URL", ""
)  # optional bring-your-own endpoint
ANTHROPIC_AUTH_TOKEN = os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
# Custom Anthropic-compatible endpoints (e.g. a Bedrock/Azure gateway) often require
# explicit model ids the key is allowed to use. Bring-your-own; blank by default.
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "")
ANTHROPIC_SMALL_FAST_MODEL = os.environ.get("ANTHROPIC_SMALL_FAST_MODEL", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
# Codex custom providers are configured per invocation by ds_agent_lab.  The
# base URL is intentionally not passed as OPENAI_BASE_URL: current Codex CLI
# versions ignore that environment variable and require a model_providers
# entry instead.
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "")
NGC_API_KEY = os.environ.get(
    "NGC_API_KEY", ""
)  # only if the DeepStream image is private
HF_TOKEN = os.environ.get("HF_TOKEN", "")  # only for gated VLM weights

# ---- Prompt catalog: 1:1 with example_prompts/*.md (validator enforces this) ----
PROMPT_CATALOG = [
    {
        "id": "import_vision_model_detection_pipeline",
        "file": "example_prompts/import_vision_model_detection_pipeline.md",
        "title": "Import object detection model end-to-end",
        "output_dir": "models",
        "skill": "deepstream-import-vision-model",
        "needs_rtsp": False,
        "needs_kafka": False,
        "needs_vlm": False,
        "needs_model_import": True,
        "needs_pdf_report": True,
        "interactive": True,
        "preferred_artifact": "pdf_report",
        # The requested deliverable is measured output, not just authored code. Generate must
        # finish it in the current one-shot agent invocation.
        "generate_result": "pdf_report",
        # End-to-end onboard: build engine + benchmark + report all happen at Generate.
        "generate_timeout": IMPORT_VISION_TIMEOUT,
        # Generate output is a report + config files (not an app) -> download only, no inline dump.
        "generate_display": "download",
        # Generate runs in the shared /workspace (no isolated/cleared dir): the skill manages models/.
        "generate_in_workspace": True,
    },
    {
        "id": "ds_profiling_efficient_pipeline",
        "file": "example_prompts/ds_profiling_efficient_pipeline.md",
        "title": "Profile & build an efficient multi-stream RTSP pipeline",
        "output_dir": "efficient_pipeline_app",
        "skill": "deepstream-profile-pipeline",
        # RTSP-style prompt, but the lab profiles with a FILE source (no 16 live streams to
        # spin up): _use_rtsp() stays False, so the Run contract tells the agent to read the
        # sample file. The profiling skill still derives the HW ceiling / max-streams answer.
        "needs_rtsp": True,
        "needs_kafka": False,
        "needs_vlm": False,
        "needs_nats": False,
        "needs_model_import": False,
        "needs_pdf_report": False,
        "interactive": False,
        # Deliverable is the saved Nsight-derived profiling report (bottleneck, max streams @30 FPS,
        # HW upgrade); show_results renders the fixed report path after Generate.
        "preferred_artifact": "profile_report",
        # Profiling conclusions must come from this GPU's real measurements, so the report is a
        # Generate deliverable. The normal Run button only renders the completed report.
        "generate_result": "profile_report",
    },
    {
        "id": "msgbroker_nats",
        "file": "example_prompts/msgbroker_nats.md",
        "title": "Publish object detections to NATS (JetStream adapter)",
        "output_dir": "msgbroker_nats_app",
        "skill": "deepstream-dev",
        "needs_rtsp": False,
        "needs_kafka": False,
        "needs_vlm": False,
        "needs_nats": True,
        "needs_model_import": False,
        "needs_pdf_report": False,
        "interactive": False,
        "preferred_artifact": "nats_json",
    },
    {
        "id": "msgconv_kafka",
        "file": "example_prompts/msgconv_kafka.md",
        "title": "DeepStream msgconv Kafka output",
        "output_dir": "msgconv_kafka",
        "skill": "deepstream-dev",
        "needs_rtsp": False,
        "needs_kafka": True,
        "needs_vlm": False,
        "needs_nats": False,
        "needs_model_import": False,
        "needs_pdf_report": False,
        "interactive": False,
        "preferred_artifact": "kafka_json",
    },
    {
        "id": "multi_stream_tracker",
        "file": "example_prompts/multi_stream_tracker.md",
        "title": "Four RTSP streams with tracker and tiled render",
        "output_dir": "multi_stream_tracker_app",
        "skill": "deepstream-dev",
        "needs_rtsp": True,
        "needs_kafka": False,
        "needs_vlm": False,
        "needs_model_import": False,
        "needs_pdf_report": False,
        "interactive": False,
        "preferred_artifact": "mp4",
    },
    {
        "id": "nvdsanalytics_config_sample",
        "file": "example_prompts/nvdsanalytics_config_sample.md",
        "title": "nvdsanalytics ROI, line crossing, overcrowding, direction",
        "output_dir": "deepstream_nvdsanalytics_test_app",
        "skill": "deepstream-dev",
        "needs_rtsp": False,
        "needs_kafka": False,
        "needs_vlm": False,
        "needs_model_import": False,
        "needs_pdf_report": False,
        "interactive": False,
        "preferred_artifact": "mp4",
    },
    {
        "id": "nvdsdynamicsrcbin_app",
        "file": "example_prompts/nvdsdynamicsrcbin_app.md",
        "title": "nvdsdynamicsrcbin dynamic source demo",
        "output_dir": "nvdsdynamicsrcbin_app",
        "skill": "deepstream-dev",
        "needs_rtsp": False,
        "needs_kafka": False,
        "needs_vlm": False,
        "needs_model_import": False,
        "needs_pdf_report": False,
        "interactive": False,
        "preferred_artifact": "logs",
    },
    {
        "id": "rtvi_vlm_core_app",
        "file": "example_prompts/rtvi_vlm_core_app.md",
        "title": "RTVI VLM core app with RTSP and Kafka summaries",
        "output_dir": "rtvi_app",
        "skill": "deepstream-dev",
        "needs_rtsp": True,
        "needs_kafka": True,
        "needs_vlm": True,
        "needs_model_import": False,
        "needs_pdf_report": False,
        "interactive": False,
        "preferred_artifact": "kafka_text",
    },
    {
        "id": "rtvi_vlm_openapi_spec",
        "file": "example_prompts/rtvi_vlm_openapi_spec.md",
        "title": "FastAPI microservice for RTVI app",
        "output_dir": "rtvi_app",
        "skill": "deepstream-dev",
        "needs_rtsp": False,
        "needs_kafka": True,
        "needs_vlm": True,
        "needs_model_import": False,
        "needs_pdf_report": False,
        "interactive": False,
        "preferred_artifact": "service_code",
        # Stage 2 of the rtvi pair: builds the microservice ON TOP of the core app already in
        # rtvi_app, so Generate runs in /workspace WITHOUT clearing (see PROMPT_SEQUENCES).
        "generate_in_workspace": True,
    },
    {
        "id": "single_view_3d_tracker",
        "file": "example_prompts/single_view_3d_tracker.md",
        "title": "Single-view 3D tracker with MP4 output",
        "output_dir": "single_view_3d_tracker_app",
        "skill": "deepstream-dev",
        "needs_rtsp": True,
        "needs_kafka": False,
        "needs_vlm": False,
        "needs_model_import": False,
        "needs_pdf_report": False,
        "interactive": False,
        "preferred_artifact": "mp4",
    },
    {
        "id": "video_infer_app",
        "file": "example_prompts/video_infer_app.md",
        "title": "File inference with TrafficCamNet and OSD",
        "output_dir": "video_infer_app",
        "skill": "deepstream-dev",
        "needs_rtsp": False,
        "needs_kafka": False,
        "needs_vlm": False,
        "needs_model_import": False,
        "needs_pdf_report": False,
        "interactive": False,
        "preferred_artifact": "mp4",
    },
    {
        "id": "video_object_count",
        "file": "example_prompts/video_object_count.md",
        "title": "Video object count app",
        "output_dir": "video_object_count_app",
        "skill": "deepstream-dev",
        "needs_rtsp": False,
        "needs_kafka": False,
        "needs_vlm": False,
        "needs_model_import": False,
        "needs_pdf_report": False,
        "interactive": False,
        "preferred_artifact": "mp4",
    },
    {
        "id": "video_parallel_infer_app",
        "file": "example_prompts/video_parallel_infer_app.md",
        "title": "Parallel inference branches with metadata merge",
        "output_dir": "video_parallel_infer_app",
        "skill": "deepstream-dev",
        "needs_rtsp": False,
        "needs_kafka": False,
        "needs_vlm": False,
        "needs_model_import": True,
        "needs_pdf_report": False,
        "interactive": False,
        "preferred_artifact": "mp4",
    },
    {
        "id": "yolov26s_detection",
        "file": "example_prompts/yolov26s_detection.md",
        "title": "YOLO26s conversion and DeepStream app",
        "output_dir": "yolo_detection",
        "skill": "deepstream-dev",
        "needs_rtsp": False,
        "needs_kafka": False,
        "needs_vlm": False,
        "needs_model_import": True,
        "needs_pdf_report": False,
        "interactive": False,
        "preferred_artifact": "mp4",
    },
]


def load_prompt_text(item):
    """Load the source .md and normalize attachment references for /workspace."""
    text = (REPO_ROOT / item["file"]).read_text()
    if item["id"] == "rtvi_vlm_openapi_spec":
        # The agent runs with cwd /workspace, so point the attachment at the
        # mounted example_prompts copy.
        text = text.replace(
            "@rtvi_vlm_openapi_spec.png", "@example_prompts/rtvi_vlm_openapi_spec.png"
        )
    return text


PROMPT_CATALOG = [
    item for item in PROMPT_CATALOG if (REPO_ROOT / item["file"]).is_file()
]
if not PROMPT_CATALOG:
    raise RuntimeError(f"No configured prompt files exist under {REPO_ROOT}")

for item in PROMPT_CATALOG:
    item["prompt"] = load_prompt_text(item)

catalog_by_id = {item["id"]: item for item in PROMPT_CATALOG}
PROMPT_IDS = [item["id"] for item in PROMPT_CATALOG]

# Combined dropdown options: some prompts are one software in stages. The dropdown offers the group
# as a SINGLE pick (label = ids joined by " & "); Generate runs the stages in order on the shared
# output dir, Run runs the finished result. The catalog itself stays one entry per prompt -- this is
# purely a menu/flow grouping. (rtvi: core VLM app, then the FastAPI microservice on top of it.)
PROMPT_SEQUENCES = {
    "rtvi_vlm_core_app & rtvi_vlm_openapi_spec": [
        "rtvi_vlm_core_app",
        "rtvi_vlm_openapi_spec",
    ],
}
_available_prompt_ids = set(PROMPT_IDS)
PROMPT_SEQUENCES = {
    label: available_members
    for label, members in PROMPT_SEQUENCES.items()
    if len(
        available_members := [
            member for member in members if member in _available_prompt_ids
        ]
    )
    > 1
}

# Dropdown ids = catalog ids with each sequence's members collapsed into their combined label
# (the label appears at the position of the group's first member).
_SEQ_MEMBERS = {m for members in PROMPT_SEQUENCES.values() for m in members}
_SEQ_BY_FIRST = {members[0]: label for label, members in PROMPT_SEQUENCES.items()}
MENU_PROMPT_IDS = []
for _pid in PROMPT_IDS:
    if _pid in _SEQ_BY_FIRST:
        MENU_PROMPT_IDS.append(_SEQ_BY_FIRST[_pid])
    elif _pid not in _SEQ_MEMBERS:
        MENU_PROMPT_IDS.append(_pid)


def prompt_text(prompt_id):
    """Return the catalog prompt text for an id (used by the Generate-step dropdown<->textarea link).
    For a combined option, show each stage's prompt with a header (informational; Generate uses the
    individual catalog prompts per stage, not this concatenation)."""
    if prompt_id in PROMPT_SEQUENCES:
        return "\n\n".join(
            f"# Stage {i + 1}: {sid}\n{catalog_by_id[sid]['prompt']}"
            for i, sid in enumerate(PROMPT_SEQUENCES[prompt_id])
        )
    return catalog_by_id[prompt_id]["prompt"]
