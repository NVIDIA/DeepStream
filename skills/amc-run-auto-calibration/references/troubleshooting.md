<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# AMC Troubleshooting

Use this reference when setup, ingestion, or calibration fails.

## Service Issues

Check container state and logs with the DeepStream release compose file:

```bash
export DEEPSTREAM_REPO="/path/to/DeepStream"
export AMC_ROOT="$DEEPSTREAM_REPO/tools/auto-magic-calib"
docker compose -f "$AMC_ROOT/compose/compose.yml" ps
docker compose -f "$AMC_ROOT/compose/compose.yml" logs --tail=200 auto-magic-calib-ms
curl -fsS "http://localhost:<AUTO_MAGIC_CALIB_MS_PORT>/v1/ready"
```

Common failures:

- NGC authentication missing or expired
- GPU runtime unavailable to Docker
- port already in use
- required bind mount missing
- service started but health endpoint is not ready yet
- `HOST_IP` empty or set to `localhost`, preventing the UI container or browser from reaching the backend
- `projects/` or `models/` not writable by UID/GID `1000`

## Dataset Issues

Check:

- archive exists and is not a Git LFS pointer stub
- archive extracts successfully
- files are readable by the container user
- videos use supported codecs
- camera count matches the expected workflow
- sample dataset path is `$DEEPSTREAM_REPO/tools/auto-magic-calib/assets/sdg_08_2_sample_data_010926.zip`, or a fetched copy under `$DEEPSTREAM_REPO/data/amc/`

If the SDG archive is missing from the DeepStream checkout, restore or update the tracked `tools/auto-magic-calib` directory; do not run a submodule command. If the archive is still unavailable, follow the [sample archive fallback](sample-dataset.md#dataset): fetch the public AMC repository into `$DEEPSTREAM_REPO/data/amc/source/auto-magic-calib`, then copy its `assets/sdg_08_2_sample_data_010926.zip` to `$DEEPSTREAM_REPO/data/amc/sdg_08_2_sample_data_010926.zip`. The helper searches that exact fallback file, not the fetched repository directory. Verify the release checksum before use. Do not hard-code private repository URLs.

## Calibration Quality Issues

Watch for:

- insufficient overlapping views
- too little scene motion or visible calibration signal
- unsynchronized clips
- dropped frames or unstable RTSP streams
- incorrect camera-to-video mapping
- MP4 files uploaded in the wrong camera order
- focal length count not matching camera count
- `rectification_state != COMPLETED`; calibration requires explicitly staged linear media

Do not edit generated calibration files by hand. For valid inputs with a tunable failure or poor quality, summarize the evidence and offer [tuning.md](tuning.md); begin only after the user accepts tuning. Request corrected clips, alignment, or infrastructure instead when those are the cause.

## REST API Issues

Prefer the running service OpenAPI schema over memory:

```bash
curl -fsS "http://localhost:<AUTO_MAGIC_CALIB_MS_PORT>/openapi.yaml" >/tmp/amc-openapi.yaml
```

If `/openapi.yaml` is unavailable, inspect `/docs` or service logs to establish supported routes.

Core routes used by this skill:

```text
GET  /v1/ready
POST /v1/create_project
POST /v1/upload_video_files/{project_id}
POST /v1/config/{project_id}
POST /v1/upload_alignment/{project_id}
POST /v1/upload_layout/{project_id}
POST /v1/upload_gt_file/{project_id}
POST /v1/upload_focal_length/{project_id}
POST /v1/verify_project/{project_id}
POST /v1/linear_media/{project_id}
POST /v1/rectification/{project_id}
GET  /v1/rectification/{project_id}
POST /v1/calibrate/{project_id}
POST /v1/postprocess/{project_id}
GET  /v1/get_project_info/{project_id}
GET  /v1/amc/calibrate/{project_id}/log
GET  /v1/result/{project_id}/evaluation_statistics
GET  /v1/result/{project_id}/mv3dt_result?result_type=amc
POST /v1/vggt/calibrate/{project_id}
GET  /v1/vggt_results/{project_id}/evaluation_statistics
POST /v1/rtsp/capture/{project_id}
GET  /v1/rtsp/capture/{project_id}/{session_id}
POST /v1/rtsp/capture/{project_id}/{session_id}/stop
```

## RTSP Issues

Common RTSP and VIOS failures:

| Symptom | Action |
|---|---|
| VIOS `/vst/api/v1/sensor/list` fails | Ask the user for a reachable VIOS deployment URL before capture. |
| Capture returns `503` or "VIOS not configured" | Relaunch `auto-magic-calib-ms` with `VIOS_BASE_URL` in a temporary compose override; do not edit checked-in compose files. |
| Session stuck in `STARTING` | Check VIOS sensors and wait after sensor restart. |
| Session stuck in `RECORDING` past duration | Call `POST /v1/rtsp/capture/{project_id}/{session_id}/stop`, then poll automatic ingest. |
| Session stuck in `INGESTING` | Inspect AMC/VIOS ingest logs and disk space. Preserve the project/session IDs and resume polling that same session after recovery. Do not stop an already-ingesting session or recapture automatically. |
| Capture reaches `ERROR`/`CANCELLED` or the observer times out | Report the redacted state and IDs. A timeout does not terminate the server workload; inspect the same session before an explicitly approved retry. |
| Ingest says no clip available | The recording window may not overlap the VIOS timeline; start a new capture after sensors are online. |
| Logs include full RTSP URLs | Redact credentials in user-facing output and avoid printing secrets again. |

## VGGT Issues

`vggt_state` is independent of `amc_state`. Start VGGT from `READY`, resume/wait from `RUNNING`, and always post-process `COMPLETED` before AMC. `MODEL_MISSING` or `ERROR` means report the condition and continue with AMC. After AMC, post-process again. If a model is expected, verify the configured VGGT model path, license acceptance, mount path, and MS logs.

## Reporting Failures

Return:

- failing command
- redacted service URL or stream ID
- compose file path
- relevant log excerpt
- whether the issue is setup, input, API, or calibration quality
- next concrete remediation

Do not claim a service is healthy, a calibration completed, or metrics were generated unless the command output proves it.
