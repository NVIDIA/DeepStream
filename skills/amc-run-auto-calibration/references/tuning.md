<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# AMC Custom Dataset Tuning

Use this workflow when the user explicitly asks to tune or optimize AMC settings, does not know which configuration fits a custom dataset, or accepts the tuning offer after a normal calibration. This workflow owns tuning project names, repeated attempts, artifact preservation, and result selection. Do not invoke the single-run helper scripts end-to-end for an explicit tuning request; use this workflow to own each attempt.

Use the applicable input-mode reference only for source-specific discovery, ordering, capture, and upload semantics:

- Local videos: [video-files.md](video-files.md)
- RTSP streams: [rtsp.md](rtsp.md); capture and ingest once, then tune the same frozen clips
- Bundled sample: [sample-dataset.md](sample-dataset.md)

Do not use numeric tuning to repair missing or invalid inputs, camera ordering, unsynchronized clips, bad alignment, unresolved layout scale, incomplete rectification, or service, GPU, disk, and permission failures.

## Enter Tuning

For an explicit request such as "tune configs", "tune calibration", "optimize settings", or "find the best configuration":

- Treat the request as approval for up to five sequential AMC configuration attempts.
- Do not ask the user to choose `resnet`, `transformer`, or whether to include VGGT.
- Run both supported AMC detector baselines by default.
- For eligible multi-camera projects, run VGGT in every attempt project. VGGT does not consume an AMC attempt.
- Before starting, state the bounded plan and explain that up to five AMC attempts plus VGGT and post-processing per eligible attempt can take several hours. Runtime depends on clip duration, camera count, detector, settings, and hardware. Refine the remaining-time estimate after the baseline; an estimate is not a timeout. Respect any user-specified time budget without asking again to approve the default plan.

Without GT, the two detector baselines are not normally enough to finish tuning. Run at least one safe, evidence-driven follow-up after them when the logs and artifacts support a documented change. Continue through attempt 5 while safe, meaningful changes remain.

Stop early only when:

- a GT-backed target is met;
- an input or infrastructure problem prevents valid comparison; or
- no safe evidence-driven change remains.

Never submit a sixth AMC attempt without asking the user.

## Prepare and Freeze Inputs

1. Probe the configured `/v1/ready` endpoint and UI root and reuse a healthy stack. If unavailable, follow [setup.md](setup.md): host preflight, NGC authentication, required VGGT assets, then Compose launch and verification. Present only the current setup action; never batch credential and dataset questions. Report verified backend, Swagger, and UI URLs before asking the user to perform UI steps. Docker access is needed to launch or repair the stack, not merely to use a healthy API.
2. For multi-camera tuning, require VGGT support in the running service. Explicit tuning counts as a VGGT request; resolve `MODEL_MISSING` through [VGGT Prerequisite for Tuning](setup.md#vggt-prerequisite-for-tuning) rather than silently switching to AMC-only tuning. VGGT is not available for single-camera input; record that and continue with AMC.
3. Discover the running schema at `/openapi.yaml` on the service root (without `/v1`); use `/docs` or service logs if it is unavailable. Do not guess API paths or payloads when the contract cannot be established.
4. Fetch `GET /v1/config/defaults` and retain the complete runtime `config_params` object as the attempt-1 baseline.
5. Resolve and freeze:
   - synchronized videos in confirmed overlapping-FOV order, consistent with the alignment data;
   - `layout.png` and exact `layout_px_per_m`;
   - alignment data and its coordinate space;
   - focal lengths, when provided;
   - optional `GT.zip`;
   - already-linear staging or completed rectification.

A settings file is optional and is not required for tuning. Start from the running service defaults; do not auto-discover or inherit a nearby settings file.

If `layout_px_per_m` is unknown, create attempt 1, upload its videos and layout, and direct the user to UI Step 3. They may enter an exact value or calculate it from two layout points and a known real-world distance, then save it. Read back and freeze the exact saved value. Do not infer it from another dataset or image dimensions.

If alignment is missing, finish media preparation, then direct the user to UI Step 5 to create and save it against the staged rectified frames. Freeze and reuse the saved alignment. External alignment made on original videos uses `coord_space=original`; UI alignment made on rectified frames uses `coord_space=rectified`.

Use the same videos, layout, scale, alignment, camera order, focal lengths, GT, and media preparation for every attempt. Changing any of them silently invalidates the comparison.

## Configuration Isolation

`GET /v1/config` returns mutable service-wide settings, even though settings are submitted through `POST /v1/config/<project_id>`. It is a snapshot of the current service configuration, not a readback of settings committed to a particular project.

Reserve the backend for the tuning session before applying settings. No other UI session, API client, or agent may change configuration or start workloads on that backend during the session. Viewing results and logs is safe. If exclusive use cannot be established, pause tuning or use a separate backend; separate project IDs do not isolate global configuration.

Keep the complete submitted `config.json` for each attempt. Under exclusive use, compare the service-wide snapshot immediately after applying settings and immediately before calibration submission. On a mismatch, pause and resolve the interference before continuing. Preserve the project's generated configuration under `output/config_AutoMagicCalib/` when available and compare its effective tunable values with the intended configuration. A current global snapshot cannot establish what a completed run used; exclude a run from configuration comparisons if its settings cannot be attributed reliably.

For agents sharing the AMC host, have its operator designate one shared writable lock path per backend, outside the container-owned `projects/` and `models/` directories. Set `AMC_TUNING_LOCK` to that exact host-side path. Start this holder in a dedicated tool process session and retain the live process handle while other tool calls perform tuning:

```bash
: "${AMC_TUNING_LOCK:?set the operator-designated shared backend-host lock path}"
flock --help | grep -q -- '--no-fork' || { echo "ERROR: install util-linux flock with --no-fork support on the AMC host"; exit 1; }
exec flock --no-fork --nonblock "$AMC_TUNING_LOCK" sleep infinity
```

A lock failure exits the holder immediately: wait or choose a separate backend. Verify the holder remains running before mutation; recheck its handle before every attempt. `--no-fork` makes the held process own the lock, so terminating that exact tool process at session end releases it. This does not require commands to share a shell. Do not delete the lock file, kill an unrelated PID, or expire a long-running holder by timestamp. All clients must use the same designated path on the AMC host (via an SSH process handle when needed). The lock cannot exclude UI/API clients that ignore it, so also reserve the backend with its operator. For API-only access without that reservation, pause or use a dedicated backend. Configuration snapshots are interference checks, not an atomic server lock.

If a tool connection is lost while its holder survives, have the backend owner identify that exact process and verify its session and workloads have ended before terminating it. A lock-file timestamp alone is not evidence of an abandoned holder.

## Reuse a Normal Project

When tuning begins after a normal calibration, reuse that project as attempt 1 only when its frozen inputs, exact effective configuration, and detector can be recovered and verified.

- Do not rename or duplicate the server project.
- Record its actual name and ID plus its canonical attempt label in the report.
- Recover the configuration from the project's persisted calibration output, such as `project_<id>/output/config_AutoMagicCalib/`, or a per-project readback endpoint exposed by the running schema. Recover the detector from project metadata or `calibration.log`. Do not use the service-wide `GET /v1/config` alone as evidence of what an earlier project used.
- Normalize and compare every AMC-tunable field with the current runtime defaults plus the same frozen `layout_px_per_m` and explicit dataset-required overrides. Reuse the project only when they are equivalent; apply that exact recovered configuration unchanged for the alternate-detector baseline.
- Preserve its completed AMC artifacts before launching another workload.
- If VGGT already completed, preserve and reuse that result.
- If VGGT is `READY`, preserve AMC first, then run VGGT and post-processing in the same project and preserve VGGT separately.
- Count the completed normal AMC baseline as attempt 1; do not rerun it only to enter tuning.

If the effective configuration or detector cannot be recovered exactly, differs from the intended baseline, or the project otherwise cannot support a valid comparison, retain its result only as an external alternative and create a fresh attempt 1 from runtime defaults.

## Project Names and Output Labels

Derive one stable `dataset_slug` before creating any tuning project: lowercase the dataset name, replace non-alphanumeric runs with underscores, trim separators, and retain readable words.

Use this canonical label for every new server project, output directory, report row, and JSON-summary entry:

```text
<dataset_slug>_attempt_<NN>_<type>
```

Examples:

```text
nv_warehouse_attempt_01_resnet
nv_warehouse_attempt_02_transformer
nv_warehouse_attempt_03_resnet_sf5
```

Use two-digit attempt numbers. Keep the detector and relevant configuration delta readable. Do not switch to opaque abbreviations.

Determine the planned suffixes first, then resolve project-name `maxLength` from the running create-project schema. If it is absent, use a conservative limit of 32 characters. Reserve every attempt/type suffix in full and deterministically shorten only the dataset slug until all names fit. Reuse that exact slug throughout tuning and record the original-name-to-slug mapping.

If project creation returns HTTP 422:

1. Report the validation response body.
2. Revalidate the same logical name against the schema.
3. Retry once with a schema-compliant name while keeping the logical dataset slug stable.
4. If the response proves the service has a smaller limit, rebuild only the not-yet-submitted names with one deterministic compact slug and update the recorded mapping.

A project-creation, upload, configuration, or verification error before an accepted `POST /v1/calibrate/<project_id>` does not consume an AMC attempt.

Write tuning output outside the generated server project tree:

```text
$DEEPSTREAM_REPO/output/amc/<dataset_slug>/<canonical-label>/
```

Apply these naming rules through the tuning REST workflow; do not change the normal single-run helpers or their project-name behavior. Keep the dataset slug fixed when selecting later evidence-driven suffixes within the reserved name budget.

For a reused normal project, keep the actual project name unchanged and use the canonical label only for output and reporting.

## Default Attempt Plan

Use this order unless the running schema exposes different supported detectors or the user explicitly narrows the experiment:

1. AMC attempt 1: `resnet` with runtime defaults plus the frozen dataset inputs.
2. AMC attempt 2: `transformer` with identical settings and inputs.
3. AMC attempt 3: one small change or tightly coupled group justified by baseline logs and artifacts.
4. Attempts 4 and 5: continue only while safe evidence supports another documented change.

For a reused normal baseline, keep its recovered detector as attempt 1 and use the other supported detector for attempt 2.

In every eligible multi-camera attempt project, run and preserve VGGT before submitting AMC. VGGT uses the same project ID but is reported separately and consumes zero AMC attempts.

Maintain an AMC incumbent after every completed attempt. A failed or degraded later experiment cannot replace it. Preserve the best completed baseline as an alternative even when a tuned result wins.

Do not copy a successful configuration from another dataset. Detector and parameter decisions must come from the current dataset's runtime contract, logs, metrics, and artifacts.

## Run One Fresh Attempt

Use the API paths exposed by the running schema. `<MS_URL>` below includes the `/v1` prefix. For each new tuning attempt:

1. Create the canonical project and upload the frozen assets using the applicable input-mode reference.
2. For confirmed-linear media, call `POST <MS_URL>/linear_media/<project_id>`. Otherwise complete Rectification through the UI or running rectification API. Poll project/rectification state and require `rectification_state == COMPLETED`.
3. Build the complete attempt configuration from runtime defaults, the frozen `layout_px_per_m`, explicit dataset-required overrides, and only this attempt's intentional delta. Save that exact JSON as `config.json`.
4. With [exclusive configuration access](#configuration-isolation), apply it with `POST <MS_URL>/config/<project_id>`. Require a successful response, then compare `GET <MS_URL>/config`'s normalized `config_params` with every submitted field. Save this service-wide snapshot separately from `config.json`; it is not project-specific validation. Stop before submission if any value differs or the current settings cannot be checked.
5. Call `POST <MS_URL>/verify_project/<project_id>`, then `GET <MS_URL>/get_project_info/<project_id>` and require the project and result-specific states to be ready.
6. For eligible multi-camera input, call `POST <MS_URL>/vggt/calibrate/<project_id>` and poll `GET <MS_URL>/get_project_info/<project_id>` until `vggt_state` reaches a terminal state. Fetch the VGGT calibration log while running or on failure.
7. When VGGT completes, call `POST <MS_URL>/postprocess/<project_id>` and poll project info until `postprocess_state == COMPLETED`.
8. Immediately preserve the complete VGGT result under `artifacts/vggt/` using the artifact mappings below.
9. Reapply the same complete attempt configuration and repeat the service-wide snapshot check immediately before AMC, with no intervening configuration changes.
10. Call `POST <MS_URL>/calibrate/<project_id>` with `{"detector_type":"<detector>"}`. Increment the AMC-attempt count only after this request is accepted.
11. Poll `GET <MS_URL>/get_project_info/<project_id>` until `amc_state` reaches a terminal state. Fetch `GET <MS_URL>/amc/calibrate/<project_id>/log` while running and on failure.
12. When multi-camera AMC completes, call post-processing again because AMC resets the shared `postprocess_state`; require it to reach `COMPLETED`. For single-camera AMC, review the single-view outputs; do not call shared multi-camera post-processing or require its state to complete.
13. Preserve the complete AMC result under `artifacts/amc/` without overwriting VGGT.
14. Compare the result with the incumbent before planning another attempt.

If VGGT fails, preserve its logs, terminal state, and failure reason. Continue to AMC only when the failure does not indicate a shared input or infrastructure problem.

Create a separate VGGT project only for an explicit isolation request, a demonstrated API limitation, an unsafe attempt project, or an input/media difference that makes the comparison invalid. Report the concrete reason and preserve both actual project IDs; do not create a reference project merely for experimental isolation.

Never run VGGT, AMC, rectification, post-processing, or separate attempts concurrently. Wait for each result-specific calibration and post-processing state to become terminal before starting the next workload.

Create the attempt directory before AMC submission and preserve available results before starting another workload:

```text
<canonical-label>/
  config.json
  project_id.txt
  artifacts/
    vggt/
      calibration.log
      postprocess.log
      project_state.json
      calibration.json
      evaluation_metrics.json
      evaluation_statistics.json
      refined-camera-yaml/
      scaled-world-output/
      overlays/
      mv3dt-camera-map/
    amc/
      calibration.log
      postprocess.log
      project_state.json
      calibration.json
      evaluation_metrics.json
      evaluation_statistics.json
      tracklet-overlays/
      virtual-gt-overlays/
      refined-camera-yaml/
      scaled-world-output/
      mv3dt-camera-map/
```

Preserve only artifacts that exist and record missing artifacts explicitly. For a failed run, keep the calibration log, partial multi-view log, generated camera-pair overlays, exact terminal state, and failure reason.

Use these result-specific mappings when the running schema exposes them, and save their response bodies with clear filenames:

| Result | API or project source |
|---|---|
| State | `GET <MS_URL>/get_project_info/<project_id>` |
| Effective AMC configuration | Copy the project's `output/config_AutoMagicCalib/` when available; retain the intended request and service-wide snapshots separately. |
| AMC log | `GET <MS_URL>/amc/calibrate/<project_id>/log` |
| VGGT log | `GET <MS_URL>/vggt/calibrate/<project_id>/log` |
| AMC evaluation | `GET <MS_URL>/result/<project_id>/evaluation_metrics` and `/evaluation_statistics` |
| VGGT evaluation | `GET <MS_URL>/vggt_results/<project_id>/evaluation_metrics` and `/evaluation_statistics` |
| AMC overlay | `GET <MS_URL>/result/<project_id>/overlay_image` |
| VGGT overlay | `GET <MS_URL>/vggt_results/<project_id>/overlay_image` when available |
| MV3DT export | `GET <MS_URL>/result/<project_id>/mv3dt_result?result_type=amc` or `result_type=vggt` |
| Complete snapshot | Copy the available result-specific files from `$AMC_ROOT/projects/project_<id>/output/` before another calibration or post-process can replace shared outputs. |

Also preserve camera-parameter and full-export responses exposed by the running OpenAPI schema. Treat a documented endpoint returning not found as a missing artifact, not permission to invent a replacement path.

AMC trajectory overlays and VGGT calibration visualizations are separate result types. VGGT is not automatically composited onto AMC trajectory overlays, so seeing only AMC trajectories does not mean VGGT failed. Evaluate each result through its own state, post-processing output, camera parameters, metrics, overlays, and exports.

A VGGT-only result may have no trajectory overlay because that visualization requires AMC tracklet artifacts. Review available virtual-object overlays, camera parameters, and exports instead. After AMC and the subsequent shared post-processing complete, preserve any newly available VGGT visualizations separately without overwriting the earlier VGGT snapshot. GT evaluation metrics are absent when no GT was supplied.

## Diagnose and Adjust

Read [calibration-parameters.md](calibration-parameters.md) before choosing a numeric change.

For single-camera input, use single-view evidence and controls; cross-camera matching and multi-view bundle-adjustment controls do not apply.

1. Identify the failing stage and weakest camera pair from logs and artifacts.
2. Select only a documented parameter whose stage matches that evidence.
3. Prefer an explicit rejected boundary reported by the current run. Otherwise use the parameter reference's midpoint-to-policy-bound rule.
4. Change one parameter or one tightly coupled group and record the hypothesis, old and new values, expected effect, and risk.
5. Compare pair coverage, filtered outliers, reprojection distribution, bundle-adjustment cost, artifacts, and runtime against the incumbent.

Prefer the alternate detector before broadly loosening matching thresholds. Do not use a later-stage threshold to compensate for an earlier filtering or input failure. Stop when no supported change remains inside the documented guardrails.

Never tune `layout_px_per_m`, camera order, synchronization, source files, alignment points, layout orientation, or VGGT controls in response to AMC logs. Treat rectification changes as a separate experiment requiring a fresh project and revalidated alignment.

## Select and Report

For every completed AMC and VGGT result, record what is available:

- pair-by-pair match counts and filtered-outlier counts;
- reprojection mean, standard deviation, and maximum for each camera;
- bundle-adjustment final costs;
- calibration, post-process, and export states;
- runtime, warnings, failures, and missing artifacts.

Select the AMC incumbent using complete evidence, in this order:

1. Completed calibration and, for multi-camera input, completed shared post-processing.
2. GT-backed metrics when GT exists.
3. Camera-pair coverage, outliers, L2 metrics, and reprojection distribution.
4. Bundle-adjustment cost, valid artifacts, and warnings.
5. Visual review of overlays, projections, and layout placement.

Do not select on mean reprojection error alone. Disclose when mean error improves but maximum error, variance, or the weakest camera pair worsens. Without GT, label the recommendation heuristic and require visual review in the AMC UI.

Save a concise Markdown report and machine-readable JSON summary containing:

- frozen inputs and runtime defaults;
- original dataset name and stable slug mapping;
- AMC attempt count and per-project VGGT status;
- canonical label, actual project name, and project ID;
- detector, configuration delta, terminal states, metrics, runtime, and artifacts;
- incumbent selection, trade-offs, and preserved baseline alternative;
- recommended configuration JSON and manual next action when no result is acceptable.

Do not fabricate unavailable metrics or overwrite a user's existing configuration. Ask before using the recommendation for a separate production calibration.

## Configuration Handoff

Report the service-wide settings left active separately from the recommended configuration; the last attempt need not be the winner. Never reset a completed project merely to change the service-wide settings.

For a subsequent normal run requesting defaults, compare the active `GET /v1/config` tunable values with `GET /v1/config/defaults`. If they differ, do not inherit the last experiment: prepare the complete runtime-default `config_params` object plus the new dataset's confirmed `layout_px_per_m` and explicit dataset-required overrides. Use the existing helper's `CONFIG_FILE` input where supported, or apply it through `POST /v1/config/<new_project_id>` in the REST/UI workflow. Resolve unknown scale in UI Step 3; never substitute the global/default scale. Do not overwrite a user's settings file, explicit UI choices, or a resumed project's settings.

Apply settings only on an idle backend with exclusive configuration access, and compare the service-wide snapshot before calibration. Leave preserved results untouched; a global snapshot is not evidence of the settings used by a completed project.

## Project Retention

Retain projects by default so the user can review them in the UI. In the final report, list the session's project IDs, external artifact locations, and any failed or superseded projects eligible for cleanup. Keep the selected result, preserved baseline alternative, and any reused normal project unless the user explicitly includes them in a cleanup request.

Maintain a session ledger of IDs and names returned by successful project-creation calls. Offer a retention review of failed or superseded session projects after preserving their available configurations, logs, results, and exports outside the server tree. Exclude the selected result, baseline alternative, reused normal projects, and unrelated projects from suggested removal.

If cleanup is requested, have the user identify each intended ID from that ledger and inspect its name and idle state in AMC UI/Swagger. Explain that removal permanently destroys the server-side project and have the user perform the deletion there. Tuning does not submit project deletion requests, and `ASSUME_YES` does not authorize cleanup. Record what remains and verify any user-reported removal by querying the project list; never infer successful cleanup from the request alone.
