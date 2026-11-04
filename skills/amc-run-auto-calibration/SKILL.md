---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
name: amc-run-auto-calibration
description: Run or tune NVIDIA AutoMagicCalib from DeepStream using sample data, local MP4s, or RTSP streams. Use for AMC setup and calibration artifacts. Do not use for general calibration theory, target-based stereo calibration, or runtime analytics.
metadata:
  author: "Harshal Nishar <hnishar@nvidia.com>"
  version: "3.3.0"
  tags: [amc, calibration, deepstream]
---

# AutoMagicCalib Auto-Calibration

## Purpose

Operate AutoMagicCalib (AMC) from a DeepStream checkout. Produce camera-calibration artifacts from sample data, local MP4s, or live RTSP streams without exposing credentials or modifying user source videos. Keep all source paths rooted at `<DEEPSTREAM_REPO>/...` or `$DEEPSTREAM_REPO/...`.

## Prerequisites

1. Identify the user's workflow:
   - setup or launch AMC service stack -> read [references/setup.md](references/setup.md)
   - run the SDG sample dataset -> read [references/sample-dataset.md](references/sample-dataset.md)
   - calibrate local MP4 files -> read [references/video-files.md](references/video-files.md)
   - calibrate live RTSP streams -> read [references/rtsp.md](references/rtsp.md)
   - tune or optimize custom-dataset settings -> read [references/tuning.md](references/tuning.md)
   - diagnose failures or verify outputs -> read [references/troubleshooting.md](references/troubleshooting.md)
2. Resolve `<DEEPSTREAM_REPO>` from the current workspace root. If the user is not in the DeepStream repo, ask for the repo path before running commands.
3. Set the AMC root to `$DEEPSTREAM_REPO/tools/auto-magic-calib`. This tracked DeepStream directory contains `compose/`, `assets/`, `models/`, and `projects/`. If it is missing, restore or update the DeepStream checkout; do not run a submodule command.
4. If the SDG sample archive is missing from the tracked AMC directory, fetch the public AMC repository `https://github.com/NVIDIA-AI-IOT/auto-magic-calib` into a repo-rooted work directory such as `$DEEPSTREAM_REPO/data/amc/source/auto-magic-calib`, then copy `assets/sdg_08_2_sample_data_010926.zip` to `$DEEPSTREAM_REPO/data/amc/sdg_08_2_sample_data_010926.zip`, the helper's fallback archive path. Follow the checksum verification in the sample reference. Do not hard-code private/internal repository URLs.
5. When credentials are needed, ask the user to provide them through their normal shell or secret store. Never print, persist, or invent an NGC API key, HuggingFace token, RTSP password, VIOS token, or service credential.

## Usage

Use these variables in commands:

```bash
export DEEPSTREAM_REPO="/path/to/DeepStream"
export AMC_ROOT="$DEEPSTREAM_REPO/tools/auto-magic-calib"
```

Primary release paths:

```text
$DEEPSTREAM_REPO/tools/auto-magic-calib/compose/compose.yml
$DEEPSTREAM_REPO/tools/auto-magic-calib/assets/sdg_08_2_sample_data_010926.zip
$DEEPSTREAM_REPO/tools/auto-magic-calib/models/
$DEEPSTREAM_REPO/tools/auto-magic-calib/projects/
```

Do not use paths relative to this skill directory unless the requested file is actually part of the skill. The runtime helper scripts below are part of this skill and are intended for deterministic REST execution after setup:

```text
$DEEPSTREAM_REPO/skills/amc-run-auto-calibration/scripts/run_sample_calibration.py
$DEEPSTREAM_REPO/skills/amc-run-auto-calibration/scripts/run_video_calibration.py
$DEEPSTREAM_REPO/skills/amc-run-auto-calibration/scripts/run_rtsp_calibration.py
```

### Validate → Prepare → Execute → Verify → Report

For setup requests, launch or verify the AMC microservice stack, then show health and logs from [references/setup.md](references/setup.md).

For sample requests, ensure the SDG sample dataset is available under `$AMC_ROOT/assets/`. If it is missing, restore or update the DeepStream checkout; only use the public AMC repository URL as a fallback source for the archive. Prefer `scripts/run_sample_calibration.py` for the end-to-end REST run unless the user asks for Swagger UI or manual commands.

For MP4 calibration requests, validate that every provided path is readable, upload all `*.mp4` files in explicit sorted camera order, and submit the job through the AMC REST API. Filenames need not follow `cam_*.mp4`. Prefer `scripts/run_video_calibration.py` when the user provides or approves environment variables for the run.

For RTSP calibration requests, validate AMC and VIOS reachability without logging credentials, start one bounded capture session with all streams together, poll automatic ingest until `INGESTED`, then run the same staged-media/verify/calibrate flow. Prefer `scripts/run_rtsp_calibration.py` when RTSP inputs are available.

Linear-media gate: calibration cannot start until source videos are staged. For already-linear/pinhole media, call `POST /v1/linear_media/{project_id}`. For distorted media, use UI Step 4 (Rectification): select Auto, Manual, or Videos Are Rectified; review estimates; then click Generate Rectified Videos. Auto supports `simple_divisional` (default), `simple_radial`, and `radial`; Manual permits per-camera `model`, `k1`, and `k2` for `radial`. `READY_FOR_REVIEW` is not complete—continue only when `rectification_state == COMPLETED`. Re-rectification invalidates verification, calibration, and post-process results. The helper scripts support this interactive path with `MEDIA_MODE=rectified`; use `MEDIA_MODE=linear` only for confirmed linear inputs. Both paths produce `rectified.mp4` and `rectified.jpg`. External alignment files normally use `coord_space=original`; use `rectified` only when points were created on AMC's rectified media.

For normal non-tuning multi-camera projects, VGGT runs first by default when `vggt_state == READY`; `MODEL_MISSING` and a reported VGGT failure are safe AMC-only fallbacks. Resume and wait when VGGT is already `RUNNING`. Run `POST /v1/postprocess/{project_id}` after every completed VGGT run, including a previously completed run, then run AMC and post-process again because each calibration resets `postprocess_state`. Evaluate `vggt_state`, `amc_state`, and `postprocess_state` independently; do not use aggregate `project_state` as the pipeline-success signal. When both methods complete, compare their evaluation metrics and Results-page overlays, then select the more accurate calibration for export.

`verify_project` returning `project_state == READY` is the pre-calibration input-readiness gate, not a claim that any calibration completed. Keep that check; after starting workloads, use their independent terminal states. The reference endpoints describe the AMC 3.3.0 contract. Confirm the running OpenAPI schema before mutations; stop and resolve a version mismatch rather than inventing a replacement payload.

AMC UI sequence: Step 1 Project Setup, Step 2 Video Configuration, Step 3 Parameters (including layout pixels-per-meter), Step 4 Rectification, Step 5 Manual Alignment, Step 6 Execute, Step 7 Results.

For all workflows, return:

- the exact command sequence run
- the service URL and health state
- the job ID or equivalent run identifier
- the output directory and generated calibration files
- any warnings that affect calibration quality, synchronization, or camera coverage
- rectification path/state and post-process state
- for tuning, every attempt, configuration delta, AMC/VGGT status, preserved artifacts, recommendation, and retained project IDs

### Tuning

For an explicit tuning request, follow [references/tuning.md](references/tuning.md) instead of invoking a single-run helper end-to-end. It owns setup prerequisites, detector comparison, project naming, repeated attempts, result preservation, and selection. Read [references/calibration-parameters.md](references/calibration-parameters.md) before adjusting a parameter. Do not ask the user to choose the tuning detector sequence or separately approve the default five-attempt budget.

After a normal custom-dataset calibration, offer tuning as an optional next step, including after a failure supported by tunable evidence. Do not start repeated runs until the user accepts; input and infrastructure problems must be resolved first.

Tuning changes service-wide settings. Before a new normal run that should use defaults, compare the active `GET /v1/config` tunable values with `GET /v1/config/defaults`. If they differ, use the REST/UI path to apply the complete defaults plus that dataset's confirmed scale and inputs before calibration; see [Configuration Handoff](references/tuning.md#configuration-handoff). Preserve explicitly selected file/UI settings and resumed projects.

## Reference

Use repo-rooted work areas for agent-created inputs and exported artifacts:

```bash
$DEEPSTREAM_REPO/data/amc/<run-name>/
$DEEPSTREAM_REPO/output/amc/<run-name>/
```

AMC server projects persist under:

```bash
$DEEPSTREAM_REPO/tools/auto-magic-calib/projects/project_<project_id>/
```

Do not write outputs into source-controlled sample or config directories unless the user explicitly asks.

Read only reference needed for requested workflow:

| Request | Reference |
|---|---|
| Service setup, launch, stop | [references/setup.md](references/setup.md) |
| Bundled SDG sample | [references/sample-dataset.md](references/sample-dataset.md) |
| Local MP4 files | [references/video-files.md](references/video-files.md) |
| Live RTSP through VIOS | [references/rtsp.md](references/rtsp.md) |
| Custom-dataset tuning | [references/tuning.md](references/tuning.md) |
| Tuning parameter meanings and adjustment limits | [references/calibration-parameters.md](references/calibration-parameters.md) |
| Failed setup, capture, or calibration | [references/troubleshooting.md](references/troubleshooting.md) |

## Available Scripts

Run these with `python3` after resolving the checkout and the required inputs in the linked mode reference. The common module is imported by the entry points.

| Script | Purpose | Inputs |
|---|---|---|
| [run_sample_calibration.py](scripts/run_sample_calibration.py) | Bundled SDG sample | `DEEPSTREAM_REPO`, AMC URL/port; optional `RUN_VGGT` |
| [run_video_calibration.py](scripts/run_video_calibration.py) | User MP4s | `VIDEO_DIR`, `DETECTOR_TYPE`, confirmed `MEDIA_MODE`; optional asset paths |
| [run_rtsp_calibration.py](scripts/run_rtsp_calibration.py) | One all-camera capture through VIOS | `RTSP_URLS` or `STREAMS_JSON`, `VIOS_BASE_URL`, duration, detector, media decision |
| [amc_common.py](scripts/amc_common.py) | Shared validation, API, polling, and post-processing helpers | Imported module; not an executable workflow |

## Limitations

Requires a supported Ubuntu/x86_64 GPU host, matching AMC service, and prepared calibration inputs. Live capture also requires VIOS. VGGT needs model access and is inapplicable to single-camera projects. No-GT tuning is heuristic and requires visual review. A sandbox without these services can explain prerequisites but cannot claim a successful calibration.

## Inputs

Resolve the checkout and service URL, then collect mode-specific media, detector/settings, media geometry, alignment, layout/scale, and optional GT as described in the selected reference. Explicit user settings take precedence over inferred defaults; never infer linear media or layout scale. Contact only the user-designated AMC/VIOS endpoints and model registry; keep camera data and credentials within that intended deployment.

Helpers accept positive integer `VGGT_TIMEOUT_SECONDS` (default `900`) and `POSTPROCESS_TIMEOUT_SECONDS` (default `1800`) to extend observation of larger datasets. These are client polling limits, not server cancellation: on timeout inspect/resume the same project, never assume it stopped or start a competing workload.

## Troubleshooting

Stop current stage on an unavailable service, failed validation, terminal job error, or missing linear media. Report stage, redacted endpoint or stream identifier, project/session ID, safe log tail, and retry condition. Outside explicit tuning, do not retry calibration or capture automatically: duplicate runs create new server projects and capture sessions. During tuning, follow the bounded retry rules in [references/tuning.md](references/tuning.md); never recapture for each attempt.

## Completion Criteria

A workflow is complete only when:

- the AMC service is healthy, or the setup failure is reported with logs
- the calibration job reached a terminal success or failure state
- generated calibration files are listed with repo-rooted paths
- the user has enough commands to reproduce or clean up the run
- tuning reports all attempts and identifies the recommended configuration as GT-backed or heuristic

## Related Use

Downstream VSS workflows may consume this DeepStream skill when they need AMC calibration artifacts.

## Examples

- “Launch AMC and verify GPU/service readiness.” → Read [references/setup.md](references/setup.md).
- “Calibrate these four synchronized MP4 files.” → Read [references/video-files.md](references/video-files.md); validate, explicitly stage/rectify, calibrate, post-process, report.
- “Capture RTSP cameras through VIOS.” → Read [references/rtsp.md](references/rtsp.md); use one bounded capture and poll automatic ingest.
- “Tune calibration for this custom dataset.” → Read [references/tuning.md](references/tuning.md); prepare AMC/VGGT, freeze inputs, and run bounded evidence-driven attempts.
