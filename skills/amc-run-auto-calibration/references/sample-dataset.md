<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# AMC Sample Dataset Calibration

Use this workflow when the user asks to run AMC on the shipped SDG sample dataset or verify the AMC installation with sample data.

## Dataset

Expected sample archive in the DeepStream checkout:

```text
$DEEPSTREAM_REPO/tools/auto-magic-calib/assets/sdg_08_2_sample_data_010926.zip
```

The archive contains:

```text
alignment_data/alignment_data.json
alignment_data/layout.png
GT.zip
videos/cam_00.mp4
videos/cam_01.mp4
videos/cam_02.mp4
videos/cam_03.mp4
```

This archive is tracked under DeepStream `tools/auto-magic-calib`. If the directory or archive is missing, restore or update the DeepStream checkout; do not run a submodule command.

If the archive is still missing, fetch it from the public AMC repository into a repo-rooted working directory. Do not hard-code private/internal repository URLs.

Example shape:

```bash
export DEEPSTREAM_REPO="/path/to/DeepStream"
mkdir -p "$DEEPSTREAM_REPO/data/amc/source" "$DEEPSTREAM_REPO/data/amc"
git clone https://github.com/NVIDIA-AI-IOT/auto-magic-calib "$DEEPSTREAM_REPO/data/amc/source/auto-magic-calib"
cp "$DEEPSTREAM_REPO/data/amc/source/auto-magic-calib/assets/sdg_08_2_sample_data_010926.zip" \
  "$DEEPSTREAM_REPO/data/amc/"
```

Prefer the tracked copy under `$DEEPSTREAM_REPO/tools/auto-magic-calib/assets/` whenever it is present.

Before extracting any tracked or fallback copy, verify SHA-256 `0dceb0cc8324f5775b0c2007efe7a3e7c36fda10c5964b88e20712b002d98bdb`, the Git LFS object ID recorded for this named release archive. The helper enforces this check. A mismatch or a Git LFS pointer is not a usable ZIP: restore the matching release data; do not bypass the check or invent a new digest.

## Scripted Run

The bundled helper script performs the sample workflow end to end: locate or extract the archive, create the project, upload the sample videos and assets, verify readiness, run VGGT first when available, post-process it, run AMC with the release sample detector, post-process again, and print metrics.

```bash
export DEEPSTREAM_REPO="/path/to/DeepStream"
python3 "$DEEPSTREAM_REPO/skills/amc-run-auto-calibration/scripts/run_sample_calibration.py"
```

Optional environment variables:

```bash
export BASE_URL="http://<HOST_IP>:<AUTO_MAGIC_CALIB_MS_PORT>/v1"
export SAMPLE_DIR="$DEEPSTREAM_REPO/data/amc/sample/sdg_08_2_sample_data_010926"
export PROJECT_NAME="<sample-run-name>"
export MEDIA_MODE=linear
export RUN_VGGT=true
```

For a distorted variant of the dataset, run interactively with `MEDIA_MODE=rectified`; complete Rectification in the AMC UI, review/commit it, then return to the helper when prompted.

Use the REST sequence below when the user asks for a manual or Swagger UI walkthrough.

## Preflight

Ensure the AMC backend is reachable and extract the sample into a repo-rooted cache:

```bash
export DEEPSTREAM_REPO="/path/to/DeepStream"
export AMC_ROOT="$DEEPSTREAM_REPO/tools/auto-magic-calib"
SAMPLE_ZIP="$AMC_ROOT/assets/sdg_08_2_sample_data_010926.zip"
test -f "$SAMPLE_ZIP" || SAMPLE_ZIP="$DEEPSTREAM_REPO/data/amc/sdg_08_2_sample_data_010926.zip"
test -f "$SAMPLE_ZIP" || { echo "ERROR: missing sdg_08_2_sample_data_010926.zip"; exit 1; }

SAMPLE_DIR="$DEEPSTREAM_REPO/data/amc/sample/sdg_08_2_sample_data_010926"
printf '%s  %s\n' '0dceb0cc8324f5775b0c2007efe7a3e7c36fda10c5964b88e20712b002d98bdb' "$SAMPLE_ZIP" | sha256sum -c - || exit 1
mkdir -p "$SAMPLE_DIR"
unzip -q -n "$SAMPLE_ZIP" -d "$SAMPLE_DIR"
find "$SAMPLE_DIR" -maxdepth 3 -type f | sort

# Check the intended service after verifying and extracting the local archive.
BASE_URL="${BASE_URL:-http://localhost:${AUTO_MAGIC_CALIB_MS_PORT:-8000}/v1}"
curl -fsS "$BASE_URL/ready"
```

## REST Sequence

Use the OpenAPI contract from the running service when available:

```bash
curl -fsS "${BASE_URL%/v1}/openapi.yaml" >/tmp/amc-openapi.yaml
```

Submit in this order:

1. `POST /v1/create_project` with form field `project_name`.
2. `POST /v1/upload_video_files/{project_id}` with all `videos/cam_*.mp4`, sorted alphabetically.
3. `POST /v1/linear_media/{project_id}` only after confirming sample videos are already linear. For distorted data, instead complete rectification in the UI.
4. `POST /v1/upload_alignment/{project_id}?coord_space=original` with `alignment_data/alignment_data.json`, which is authored against the uploaded source videos.
5. `POST /v1/upload_layout/{project_id}` with `alignment_data/layout.png`.
6. `POST /v1/upload_gt_file/{project_id}` with `GT.zip`.
7. `POST /v1/verify_project/{project_id}`; require `rectification_state == COMPLETED` and `project_state == READY`.
8. If `vggt_state == READY`, run `POST /v1/vggt/calibrate/{project_id}` first, poll `vggt_state`, then post-process. Skip safely when it is `MODEL_MISSING`.
9. `POST /v1/calibrate/{project_id}` with `{"detector_type": "resnet"}`, then poll `amc_state` to `COMPLETED` or `ERROR`.
10. `POST /v1/postprocess/{project_id}`, then poll `postprocess_state` to `COMPLETED` or `ERROR`.
11. Fetch AMC and VGGT evaluation statistics when available, compare their metrics and Results-page overlays, then select the more accurate calibration for export.

The Swagger UI path is `http://<HOST_IP>:<AUTO_MAGIC_CALIB_MS_PORT>/docs`. If the user asks for a UI walkthrough or no-Python path, give the endpoint order above and do not run shell or Python tooling.

## Validate

Confirm:

- the job reached a terminal success state
- the number of uploaded videos is `4`
- evaluation statistics were returned because `GT.zip` was uploaded
- representative metrics are in the expected range: average L2 distance below about `1.5 m`, reprojection error below about `10 px`
- final camera files exist under `$DEEPSTREAM_REPO/tools/auto-magic-calib/projects/project_<project_id>/output/multi_view_results/BA_output/results_ba/refined/camInfo_XX.yaml`
- `rectification_state` and `postprocess_state` are `COMPLETED`
- logs do not show dropped inputs, missing frames, or unsupported codecs

Report the project ID, UI URL, metrics, calibration log path, and final `camInfo_XX.yaml` files. Do not fabricate metrics if the backend or dataset is unavailable.

## Additional Dataset

The AMC README documents the real-world `nv_warehouse_071026.zip` dataset on NGC as `nvidia/amc-nv-warehouse`. It has no ground truth. Use the local-video workflow with `nv_warehouse_config.json` uploaded through `/v1/config/{project_id}` and set detector type to `transformer`.
