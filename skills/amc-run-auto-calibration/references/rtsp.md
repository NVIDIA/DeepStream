<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# AMC RTSP Calibration

Use this workflow when the user provides live RTSP camera streams and asks to calibrate them with AutoMagicCalib.

For explicit tuning, [tuning.md](tuning.md) owns project names and post-ingest attempts. Capture once and reuse the frozen clips; do not repeat live capture for each attempt or invoke the end-to-end single-run helper for tuning.

## Inputs

Collect:

- RTSP URL for each camera
- camera names or stable IDs
- desired recording duration, from `60` to `1200` seconds; prefer `120` to `180` seconds or more for sparse motion
- output run name
- AMC microservice URL, normally `http://<HOST_IP>:<AUTO_MAGIC_CALIB_MS_PORT>/v1`
- VIOS base URL, if RTSP capture is not already configured in the AMC service
- calibration asset source for this RTSP scene: explicit settings/alignment/layout files, a repo-rooted asset directory, or user confirmation that they will use the AMC UI
- detector type when settings do not pin `resnet` or `transformer`
- whether credentials are embedded in URLs or provided separately

Do not print RTSP credentials in logs or final responses. Redact URLs as needed:

```text
rtsp://<user>:<redacted>@host/path
```

## Preflight

Validate:

- the AMC backend `/v1/ready` endpoint returns success
- VIOS responds at `/vst/api/v1/sensor/list`
- `VIOS_BASE_URL` is present in the running AMC microservice environment before capture starts
- all streams are reachable from the VIOS host
- source stream URLs are used (not UI/proxy URLs), cameras have overlapping fields of view, and clips contain synchronized motion
- one capture session will include all streams together; do not run staggered captures for a multi-camera calibration
- calibration assets are from this RTSP scene, not from the bundled sample dataset or old projects

Probe AMC and VIOS:

```bash
export DEEPSTREAM_REPO="/path/to/DeepStream"
export AMC_ROOT="$DEEPSTREAM_REPO/tools/auto-magic-calib"
: "${BASE_URL:?set the user-confirmed AMC service URL before probing}"
BASE_URL="${BASE_URL%/}"
case "$BASE_URL" in */v1) ;; *) BASE_URL="$BASE_URL/v1" ;; esac
curl -fsS "$BASE_URL/ready"
curl -fsS "${VIOS_BASE_URL}/vst/api/v1/sensor/list"
```

If the AMC microservice is missing `VIOS_BASE_URL`, do not edit checked-in compose files. Relaunch the microservice with a temporary override:

```bash
export DEEPSTREAM_REPO="/path/to/DeepStream"
export AMC_ROOT="$DEEPSTREAM_REPO/tools/auto-magic-calib"
export VIOS_BASE_URL="http://<VIOS_HOST>:30888"
OVERRIDE_FILE="${TMPDIR:-/tmp}/amc-vios.override.yml"
cat > "$OVERRIDE_FILE" <<'YAML'
services:
  auto-magic-calib-ms:
    environment:
      - VIOS_BASE_URL=${VIOS_BASE_URL}
YAML

docker compose -f "$AMC_ROOT/compose/compose.yml" -f "$OVERRIDE_FILE" up -d auto-magic-calib-ms
docker exec auto-magic-calib-ms-1 printenv VIOS_BASE_URL
```

## Scripted Run

The bundled helper script validates AMC and VIOS reachability, redacts RTSP URLs in console output, starts one bounded capture containing all streams, polls capture status, ingests clips, uploads optional calibration assets, supports UI fallback for missing settings/alignment, verifies the project, requires explicit detector selection when needed, calibrates, polls completion, and reports project outputs.

```bash
export DEEPSTREAM_REPO="/path/to/DeepStream"
read -r -s -p 'Comma-separated RTSP URLs (hidden input): ' RTSP_URLS
printf '\n'
export RTSP_URLS
export CAMERA_NAMES='cam_00,cam_01'
export DURATION_SECONDS=180
export VIOS_BASE_URL="http://<VIOS_HOST>:30888"
export DETECTOR_TYPE="<resnet-or-transformer>"
export MEDIA_MODE=linear  # only when captured clips are already linear
export ALIGNMENT_COORD_SPACE=original  # external file made against captured source clips
export ASSUME_YES=true
python3 "$DEEPSTREAM_REPO/skills/amc-run-auto-calibration/scripts/run_rtsp_calibration.py"
unset RTSP_URLS
```

Set `ASSUME_YES=true` only after the user approves the capture/calibration plan; it skips those interactive prompts and does not authorize cleanup. Have the operator enter credential-bearing URLs privately on the AMC host, never in chat, command literals, or shell history. In automation, use the approved secret provider to populate only the helper's environment; do not print it or export it to unrelated subprocesses. The helper needs these values in memory; environment access remains possible to privileged or same-user processes, so use a trusted host and unset them afterward. Redact stream URLs in reports. Confirm AMC and VIOS URLs belong to the intended deployment before contacting them.

For JSON stream definitions or local calibration assets:

```bash
export STREAMS_JSON='[{"rtsp_url":"rtsp://<user>:<password>@<host0>/stream","camera_name":"cam_00"}]'
export CALIB_ASSET_DIR="$DEEPSTREAM_REPO/data/amc/<run-name>/assets"
export CONFIG_FILE="$DEEPSTREAM_REPO/data/amc/<run-name>/assets/settings.json"
export ALIGNMENT_JSON="$DEEPSTREAM_REPO/data/amc/<run-name>/assets/alignment_data.json"
export LAYOUT_PNG="$DEEPSTREAM_REPO/data/amc/<run-name>/assets/layout.png"
export GT_ZIP="$DEEPSTREAM_REPO/data/amc/<run-name>/assets/GT.zip"
export FOCAL_LENGTHS=1200.0,1201.5
export VIOS_TOKEN="<VIOS_TOKEN_FROM_SECRET_STORE>"
export SSL_VERIFY=true
export RUN_VGGT=true  # default; set false only for an explicit AMC-only run
```

Keep `SSL_VERIFY=true` for VIOS HTTPS capture. For private CAs, configure trust on the AMC backend that contacts VIOS. The helper rejects `SSL_VERIFY=false` unless the user explicitly approves an isolated test network and sets `AMC_ALLOW_INSECURE=1`. This opt-out exposes VIOS tokens to interception and emits a warning; do not suggest it for a production deployment.

Use the REST sequence below when the user needs manual control, Swagger UI steps, or endpoint-level debugging.

## RTSP Capture Sequence

Normalize the service URL, create a project, and start one capture with all streams:

```text
POST /v1/create_project
POST /v1/rtsp/capture/{project_id}
{
  "streams": [
    {"rtsp_url": "rtsp://...", "camera_name": "cam_00", "sensor_id": null},
    {"rtsp_url": "rtsp://...", "camera_name": "cam_01", "sensor_id": null}
  ],
  "duration_seconds": 180,
  "vios_token": null,
  "ssl_verify": true
}
```

Save `session.session_id` from the response. Poll:

```text
GET /v1/rtsp/capture/{project_id}/{session_id}
```

Session lifecycle:

```text
STARTING -> RECORDING -> STOPPING -> INGESTING -> INGESTED
Any state -> ERROR
STARTING -> CANCELLED (no usable recording window)
```

AMC automatically ingests clips. Poll until `INGESTED`; do not call a separate ingest endpoint. On `ERROR` or `CANCELLED`, stop and report redacted diagnostics with the existing IDs. If polling exceeds the helper's capture duration plus 600-second ingest allowance, report the last state and inspect AMC/VIOS logs and disk space. An observation timeout does not cancel the server session: resume polling the same IDs after recovery and do not start another capture. Do not call stop on an already `INGESTING` session.

To stop early after at least 60 seconds, then continue polling until automatic ingest reaches `INGESTED`:

```text
POST /v1/rtsp/capture/{project_id}/{session_id}/stop
```

List sessions without modifying captured data:

```text
GET /v1/rtsp/sessions/{project_id}
```

Retain sessions after capture and calibration. Report the exact project/session pair and preserve required clips and artifacts outside the server tree. If the user requests removal, explain that it permanently removes server-side session data and guide the user to inspect the inactive session in AMC UI/Swagger and perform the cleanup there. This calibration workflow does not submit deletion requests. Record retention status as user-managed; do not claim cleanup occurred without checking the resulting session list.

## Calibration Sequence

After automatic ingest, stage source clips before calibration. Use `POST /v1/linear_media/{project_id}` only when clips are known linear. Otherwise, for an interactive helper run set `MEDIA_MODE=rectified`, complete AMC rectification in the UI, review/commit it, and return to the helper. Then use local-video calibration flow without uploading videos again:

1. Upload settings via `POST /v1/config/{project_id}` if provided.
2. Upload alignment via `POST /v1/upload_alignment/{project_id}` if provided; pass the `coord_space` query parameter as `original` for external alignment, and `rectified` only for points made on AMC rectified media.
3. Upload layout via `POST /v1/upload_layout/{project_id}` if provided; this upload does not take `coord_space`.
4. Upload optional `GT.zip` and focal lengths if provided.
5. If settings/alignment/layout are missing, send the user to the AMC UI for this project. For RTSP projects, videos are already ingested; do not re-upload videos.
6. Verify `POST /v1/verify_project/{project_id}` returns `READY` and `rectification_state == COMPLETED`.
7. Confirm detector, stream count, duration, settings source, alignment/layout source, optional GT, and focal lengths.
8. If `vggt_state == READY`, run VGGT first and post-process it; `MODEL_MISSING` is a safe AMC-only fallback.
9. Calibrate with `POST /v1/calibrate/{project_id}` and `{"detector_type": "<resnet-or-transformer>"}`.
10. Poll `amc_state`, then post-process again and poll `postprocess_state`.
11. When both methods completed, compare VGGT and AMC metrics plus Results-page overlays, then select the more accurate calibration for export.
12. Fetch logs from `GET /v1/amc/calibrate/{project_id}/log` on AMC error.

## Validate

Check:

- each RTSP camera produced usable input
- capture session reached `INGESTED`
- ingest added expected videos to the project
- job status reached success or a reported terminal failure
- calibration files were generated under `$DEEPSTREAM_REPO/tools/auto-magic-calib/projects/project_<project_id>/output/`
- logs do not expose camera credentials

Report redacted stream identifiers, project ID, session ID, UI URL, final files, and any streams that need a new recording.
