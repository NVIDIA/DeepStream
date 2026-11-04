<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# AMC Local Video Calibration

Use this workflow when the user provides pre-recorded local video files and asks to calibrate them with AutoMagicCalib.

For explicit tuning, use this reference only for input discovery, ordering, and upload; [tuning.md](tuning.md) owns the repeated-run workflow and project names. After a normal run, offer tuning as an optional next step without starting it automatically.

## Inputs

Collect:

- AMC microservice URL, normally `http://<HOST_IP>:<AUTO_MAGIC_CALIB_MS_PORT>/v1`
- project name
- one readable MP4 per camera view, time-synchronized; filenames may be arbitrary
- optional settings JSON: `settings.json`, `config.json`, or `calibration_config.json`
- alignment JSON: `alignment_data.json`, or confirmation that the user will create alignment in the UI
- layout PNG: `layout.png`, or confirmation that the user will upload it in the UI
- optional `GT.zip` and focal lengths
- detector type: `resnet` or `transformer`; do not silently choose a detector for a custom dataset when no settings file pins one

Use repo-rooted working directories:

```bash
$DEEPSTREAM_REPO/data/amc/<run-name>/videos/
$DEEPSTREAM_REPO/output/amc/<run-name>/
```

If source MP4s are outside the repo, stage symlinks or copies into `$DEEPSTREAM_REPO/data/amc/<run-name>/videos/` only with the user's approval. Do not rename or overwrite the user's original videos.

## Preflight

Validate:

- every file exists and is readable
- video names are sorted in the intended camera order; upload order defines camera indices
- MP4 resolution is `1920x1080` as expected by the UI documentation, with a supported codec and pixel format
- there is enough moving person/object tracklet content; static scenes usually fail calibration
- clips represent the same time interval for multi-camera calibration
- the backend `/v1/ready` endpoint returns success

`1920x1080` is the documented reference input, not permission to resize footage automatically. For another resolution, report the dimensions, check the running version's supported inputs, and confirm the plan with the user before upload. Reject unreadable/unsupported media; otherwise preserve original pixels and use alignment/focal lengths appropriate to that resolution. Do not silently crop, resize, or reuse alignment from another image geometry.

Use `ffprobe` if available:

```bash
ffprobe -hide_banner "$DEEPSTREAM_REPO/data/amc/<run-name>/videos/<first-camera>.mp4"
```

## Local Asset Resolution

Scan only the videos directory, its first-level subdirectories, and its parent for:

| File | Candidate names | Fallback |
|---|---|---|
| Settings | `settings.json`, `config.json`, `calibration_config.json` | AMC UI Step 3: Parameters |
| Alignment | `alignment_data.json` | AMC UI Step 5: Manual Alignment |
| Layout | `layout.png` | AMC UI Step 5: Manual Alignment |
| Ground truth | `GT.zip`, `gt.zip` | skip metrics |

If a settings file has `detector` or `detector_type` set to `resnet` or `transformer`, pass that same detector to `/v1/calibrate/{project_id}`. Posting settings to `/v1/config/{project_id}` does not replace the calibrate detector parameter.

## Scripted Run

The bundled helper script creates the AMC project, uploads all readable `*.mp4` files in sorted camera order, auto-detects colocated settings/alignment/layout/GT assets, supports UI fallback for missing settings or alignment, requires explicit detector selection for custom data, confirms the plan, runs calibration, polls completion, and prints output paths.

```bash
export DEEPSTREAM_REPO="/path/to/DeepStream"
export VIDEO_DIR="$DEEPSTREAM_REPO/data/amc/<run-name>/videos"
export PROJECT_NAME="<run-name>"
export DETECTOR_TYPE="<resnet-or-transformer>"
export MEDIA_MODE=linear  # only when every uploaded video is already linear
export ALIGNMENT_COORD_SPACE=original  # external file made against uploaded source videos
export ASSUME_YES=true
python3 "$DEEPSTREAM_REPO/skills/amc-run-auto-calibration/scripts/run_video_calibration.py"
```

For distorted input, run interactively with `MEDIA_MODE=rectified`; complete Rectification in the AMC UI, review/commit it, then return to the helper when prompted.

In UI Step 4, choose Auto, Manual, or Videos Are Rectified. Auto supports `simple_divisional` (default), `simple_radial`, and `radial`; Manual supports per-camera `model`, `k1`, and `k2` for `radial`. Review previews, then click **Generate Rectified Videos**; do not continue from `READY_FOR_REVIEW`.

Optional environment variables:

```bash
export BASE_URL="http://<HOST_IP>:<AUTO_MAGIC_CALIB_MS_PORT>/v1"
export CONFIG_FILE="$DEEPSTREAM_REPO/data/amc/<run-name>/settings.json"
export ALIGNMENT_JSON="$DEEPSTREAM_REPO/data/amc/<run-name>/alignment_data.json"
export LAYOUT_PNG="$DEEPSTREAM_REPO/data/amc/<run-name>/layout.png"
export GT_ZIP="$DEEPSTREAM_REPO/data/amc/<run-name>/GT.zip"
export FOCAL_LENGTHS=1200.0,1201.5,1199.8,1200.2
export PROJECTS_DIR="$DEEPSTREAM_REPO/tools/auto-magic-calib/projects"
export RUN_VGGT=true  # default; set false only for an explicit AMC-only run
```

Use the REST sequence below when the user needs manual control, Swagger UI steps, or endpoint-level debugging.

## REST Sequence

Normalize the service URL:

```bash
export DEEPSTREAM_REPO="/path/to/DeepStream"
export AMC_ROOT="$DEEPSTREAM_REPO/tools/auto-magic-calib"
: "${BASE_URL:?set the user-confirmed AMC service URL before probing}"
BASE_URL="${BASE_URL%/}"
case "$BASE_URL" in */v1) ;; *) BASE_URL="$BASE_URL/v1" ;; esac
curl -fsS "$BASE_URL/ready"
```

Then submit in this order:

1. `POST /v1/create_project` with form field `project_name`; save `project_id`.
2. `POST /v1/upload_video_files/{project_id}` with all readable `*.mp4` files in explicit sorted camera order. Filenames need not start with `cam_`.
3. Stage media before calibration: use `POST /v1/linear_media/{project_id}` only for confirmed linear videos; otherwise use Rectification UI/API, review/commit parameters, and wait for `rectification_state == COMPLETED`.
4. `POST /v1/config/{project_id}` with settings JSON if provided.
5. `POST /v1/upload_alignment/{project_id}?coord_space=original` for an external file made on uploaded source videos. Use `coord_space=rectified` only for points created on AMC rectified media.
6. Upload layout, GT, and focal lengths if provided.
7. If alignment or layout is missing, use UI Step 5 (Manual Alignment). In UI Step 3, set `layout_px_per_m` directly or measure two known points; rerun post-process after scale or alignment changes.
8. `POST /v1/verify_project/{project_id}` and require `rectification_state == COMPLETED` plus `project_state == READY`.
9. Run or resume VGGT first: start it from `READY`, wait from `RUNNING`, and post-process every `COMPLETED` run. `MODEL_MISSING` or `ERROR` means report the condition and continue with AMC.
10. Confirm calibration plan, then `POST /v1/calibrate/{project_id}`. Poll `amc_state` to terminal state.
11. Post-process again and poll `postprocess_state` to terminal state.
12. Fetch each method's evaluation statistics when GT exists. Compare VGGT and AMC metrics plus Results-page overlays, then select the more accurate calibration for export. On error, fetch the relevant AMC, VGGT, post-process, or rectification log.

Use the running OpenAPI schema for field details rather than inventing payloads:

```bash
curl -fsS "${BASE_URL%/v1}/openapi.yaml" >/tmp/amc-openapi.yaml
```

## Validate

Check:

- terminal job status
- output calibration files
- per-camera success/failure
- `rectification_state == COMPLETED` and `postprocess_state == COMPLETED`
- warnings about motion, target visibility, insufficient overlap, timestamp mismatch, or unsupported video format

Final AMC camera parameters are normally under:

```text
$DEEPSTREAM_REPO/tools/auto-magic-calib/projects/project_<project_id>/output/multi_view_results/BA_output/results_ba/refined/camInfo_XX.yaml
```

The MV3DT-format export endpoint is:

```text
GET /v1/result/{project_id}/mv3dt_result?result_type=amc
GET /v1/result/{project_id}/mv3dt_result?result_type=vggt
```

Use `result_type=vggt` only when independent VGGT calibration completed and the user wants the VGGT result. Report the project ID, UI URL, log path, final files, and any cameras that need new clips.
