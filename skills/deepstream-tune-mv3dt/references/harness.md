# Harness commands

[Inspect](#1-prepare-and-inspect) | [Basic candidate](#2-establish-a-basic-candidate) | [Run](#3-supply-a-foreground-job) | [Evaluate](#4-evaluate-and-inspect) | [Finish](#5-select-clean-and-confirm)

Run helpers from any directory using absolute skill paths. Use the MV3DT sample's Python environment for YAML, OpenCV, NumPy and pinned TrackEval. The inspection and campaign lifecycle helpers otherwise use the Python standard library. FFprobe is needed for input inspection. Reuse the sample's setup scripts instead of introducing another environment or silently changing global packages.

```bash
SKILL="$DS_DIR/skills/deepstream-tune-mv3dt"
APP="$DS_DIR/src/apps/reference_apps/deepstream-tracker-3d-multi-view"
PYTHON="$APP/mv3dt_venv/bin/python"
```

## 1. Prepare and inspect

Copy `assets/dataset.json` into the experiment workspace and replace its example paths, camera IDs, physical frame rate, class and annotation frames. Remove `ground_truth` and `evaluation` if GT is absent. Add detector, cache, model, source patch and adapter files to `artifacts` so their hashes are frozen. Video and calibration paths are relative to the manifest, not the shell directory.

Stage videos and calibrations under a conventional `videos/` and `camInfo/` dataset for the sample configurator. Rename copies or use read-only mounts with an explicit mapping. The configurator can fall back to sorted calibration filenames, silently swapping valid camera matrices. Give staged videos/calibrations matching camera names and verify each generated matrix against its intended source. Store timestamps and crop/resize mappings with the adapter. The provided converter handles sample nine-field world GT and native trajectory rows. Use a tested adapter for other formats. The generic scorer is currently world-space, one class and one sequence per invocation. Use the pinned TrackEval MOTChallenge interface separately for 2D-only GT.

```bash
"$PYTHON" "$SKILL/scripts/campaign.py" inspect \
    --manifest "$WORKSPACE/dataset.json" --output "$WORKSPACE/campaign"
```

Inspection performs a bounded full video scan, rejecting decoder errors instead of trusting headers from a damaged file. Increase --probe-timeout explicitly for longer recordings if needed. Review timing_warnings, especially variable timestamps, missing PTS and different camera start times. These need an explicit timing/mapping decision, not an assumption that a successful probe proves synchronization.

The helper calculates budget_seconds = sum(camera_duration_seconds) × 60 and stores seconds internally. Read its converted wall-clock hours. Ask the user to accept that allowance or provide custom hours. Only after their answer:

```bash
"$PYTHON" "$SKILL/scripts/campaign.py" approve \
    --campaign "$WORKSPACE/campaign" --hours 24 \
    --note 'User accepted the proposed 24-hour budget'
```

Use the actual approved value and a factual approval note. Do not run this example as an automatic approval. For a user-supplied end time or a child campaign, use `--deadline` with a timezone-qualified ISO-8601 timestamp instead of `--hours`. This preserves the parent's exact deadline. Re-running approve is rejected, preventing an accidental deadline reset. Extensions require a new explicit user decision and a separately recorded continuation campaign carrying the previous results and frozen inputs.

## 2. Establish a basic candidate

Use `config_tracker_NvSORT.yml` from the open tracker as the algorithm starting point. Generate a dataset-specific MV3DT base with the reference-app configurator, then disable optional modules. Review the generated state-estimator, class models, projection, communication and camera paths. The candidate helper requires those essential MV3DT sections to exist before applying the basic overlay.

```bash
"$PYTHON" "$SKILL/scripts/campaign.py" candidate \
    --base "$WORKSPACE/base/config_tracker.yml" --basic \
    --output "$WORKSPACE/basic.yml"

# Later candidates apply a small nested YAML delta to the selected parent.
"$PYTHON" "$SKILL/scripts/campaign.py" candidate \
    --base "$WORKSPACE/basic.yml" --overlay "$WORKSPACE/lifetime-delta.yml" \
    --output "$WORKSPACE/lifetime-01.yml"
```

The helper does not invent search ranges or decide which source parameters are active. That is the agent's job, using the priorities and diagnostics references. Unknown keys may be silently ignored by the tracker, so verify the parser and effective startup config. The geometric baseline must use estimator type 3 or 4. Its overlay disables inherited box-height-scaled covariance weights and 2D predicted-box association. Review fixed world-noise values for the declared coordinate units.

For a native-dump evaluation, add a separate diagnostic overlay before the first baseline and keep its output settings fixed throughout comparisons. For the reference app's container mount convention:

```yaml
DataAssociator:
  debugVerboseLevel: 1
TargetManagement:
  debugVerboseLevel: 1
TrajectoryManagement:
  debugVerboseLevel: 1
  dumpTrajectoryIntoFile: 1
  trajectoryDumpFileName: /workspace/experiments/trajDumps/trajDump_Stream_
```

Check each field in the selected source and determine whether online versus retrospective native dumps match the desired evaluation. Do not enable tentative or shadow export merely to improve a score. Prefer metadata and selected review windows during exploration. Full video encoding on every candidate changes throughput and storage cost, so keep that choice fixed and enable it only when useful.

## 3. Supply a foreground job

Create one job JSON for the frozen launch protocol. `prepare`, `run` and optional `collect` are argv arrays executed with shell=False. Placeholders are literal substitutions, not shell expressions:

- `{config}`: isolated candidate YAML.
- `{run_dir}`: new experiment directory.
- `{manifest}`: frozen dataset JSON.
- `{container}`: unique container name owned by this run.

Use `prepare` to call `python -m utils.deepstream_auto_configurator` from the sample app directory with the dataset, selected detector profile, `--tracker-config {config}`, `--output-dir {run_dir}` and output options. The seed must already have MV3DT sections, so the configurator does not silently enable visual tracking. For replay, pass the recorded detector cache. Do not use the SCOUT or WILDTRACK profile merely to inherit its tuned tracker settings. Verify actual primary inference enablement for live runs, since a replay-oriented profile can leave it disabled.

The harness keeps a byte-preserved source-config.yml and writes a normalized candidate.yml before preparation, recording both hashes. This preserves four-space cameraModelFilepath list indentation required by the current line-based sample configurator. Ordinary PyYAML indentless lists are valid YAML but can break that configurator.

Preparation recursively rejects added, changed or removed candidate fields. If a path must be rewritten, declare its exact qualified leaf key and final string or list of strings in the frozen job's config_rewrites mapping, such as ObjectModelProjection.cameraModelFilepath. Whole sections, model mappings and module selectors cannot be rewritten. Prefer a fully prepared seed so no rewrites are needed. When communicatorType is already 1, do not pass the redundant --use_debug_communicator flag: the current configurator replaces that entire section and can remove pubSubInfoConfigPath. Preserve explicit camera IDs and peer topology.

Use `run` for the deployment command. A direct Docker/Podman command must include `--rm --name {container}` and run in the foreground. Bind only required model, dataset and run directories. Use the networking verified by the existing run skill. A local wrapper must stay in the foreground and propagate termination to its children, including any containers. Detached services are prepared separately and explicitly owned, not hidden behind the job.

Use `collect` for a format adapter or GT scoring when useful. Do not give shell syntax to argv. If a shell script is necessary, create a reviewed script in the workspace and include it in `artifacts`. Never place credentials in job arguments or the recorded job JSON. Refer to protected credential files or environment variables as supported by the deployment.

Example shape, to fill with discovered paths and verified settings:

```json
{
  "cwd": "/path/to/deepstream-tracker-3d-multi-view",
  "scope": "live detector, all cameras concurrent, external service off",
  "pipeline_config": "config_deepstream.txt",
  "detector_mode": "live",
  "prepare": ["/path/to/mv3dt_venv/bin/python", "-m", "utils.deepstream_auto_configurator", "--profile", "mtmc_4cam_pnt", "--dataset-dir", "/path/to/prepared", "--tracker-config", "{config}", "--output-dir", "{run_dir}", "--enable-file-output"],
  "run": ["docker", "run", "-t", "--rm", "--name", "{container}", "--gpus", "all", "-e", "NVIDIA_DRIVER_CAPABILITIES=compute,utility,video", "-v", "/path/to/models:/workspace/models", "-v", "/path/to/prepared:/workspace/inputs:ro", "-v", "{run_dir}:/workspace/experiments", "-w", "/workspace/experiments", "ghcr.io/nvidia/deepstream:9.2-triton-multiarch", "deepstream-test5-app", "-c", "config_deepstream.txt"],
  "success_text": "App run successful",
  "timeout_seconds": 3600
}
```

The example omits replay and communication details deliberately: adapt them to the declared scope, including adding `--replay` and changing detector_mode to replay for a replay run, and configuring memory versus MQTT for the actual topology. With pipeline_config supplied, the harness checks reference-app inference/replay enablement and legacy crop alignment before launch. For another pipeline format, implement the equivalent check in its adapter. A timeout is a per-trial ceiling inside the campaign budget. Refine it after baseline measurement. Do not interpret a slow engine build as an accuracy failure.

```bash
"$PYTHON" "$SKILL/scripts/campaign.py" run \
    --campaign "$WORKSPACE/campaign" --name basic-r1 \
    --config "$WORKSPACE/basic.yml" --job "$WORKSPACE/job.json" \
    --gpu-lock "$MV3DT_GPU_LOCK" \
    --hypothesis 'Measure the geometric baseline before adding optional modules'
```

Set `MV3DT_GPU_LOCK` to the shared lock file agreed for this GPU, not a private lock that other campaigns cannot see. The command refuses a busy lock, changed frozen input, duplicate run name or expired budget. Its adjacent .mv3dt-owner.json records the owned process start identity and container so a killed controller cannot silently release the GPU to a duplicate job. Inspect an existing live owner before resuming, do not delete its lease to force a launch. The helper records submitted and generated config hashes and rejects candidate mutation or changed module selectors during preparation. The agent must also inspect effective detector, tracker, broker and runtime settings, especially primary inference enablement versus replay.

The harness checks successful process exit, the completion marker, unchanged effective configs and unchanged frozen inputs after collection. Verify full frame coverage in saved metadata before scoring and promoting. Zero detections can still produce a successful app exit. Do not start another worker just because a heartbeat is stale. Check its exact PID/container and logs first.

## 4. Evaluate and inspect

Convert observations into the canonical CSV and validate them. Use frozen dataset.json and one predictions CSV for the entire camera set:

```bash
"$PYTHON" "$SKILL/scripts/evaluate.py" score \
    --manifest "$WORKSPACE/campaign/dataset.json" \
    --predictions "$WORKSPACE/campaign/runs/basic-r1/predictions.csv" \
    --output "$WORKSPACE/campaign/runs/basic-r1/metrics.json"

"$PYTHON" "$SKILL/scripts/evaluate.py" diagnostics \
    --predictions "$WORKSPACE/campaign/runs/basic-r1/predictions.csv" \
    --fps 10 --output "$WORKSPACE/campaign/runs/basic-r1/diagnostics.json"

"$PYTHON" "$SKILL/scripts/review_frames.py" \
    --manifest "$WORKSPACE/campaign/dataset.json" \
    --predictions "$WORKSPACE/campaign/runs/basic-r1/predictions.csv" \
    --frames 1,100,500 --output "$WORKSPACE/campaign/runs/basic-r1/review"
```

Replace example FPS and frames with the actual normalized time axis and relevant windows. The image helper produces per-frame contact sheets. Open those images and inspect short sequences around events before deciding whether an association is correct. Use the diagnostics reference for richer cylinders/BEV and event-specific logging.

## 5. Select, clean and confirm

```bash
"$PYTHON" "$SKILL/scripts/campaign.py" summarize --campaign "$WORKSPACE/campaign"
"$PYTHON" "$SKILL/scripts/campaign.py" export \
    --config "$WORKSPACE/selected.yml" --output "$WORKSPACE/final.yml"
```

Summary retains per-run results and groups identical candidate/effective config hashes. Its confirmed ranking and Pareto set use HOTA/MOTA/IDF1 means after at least three runs, with observed ranges reported. A lucky individual run does not outrank a better repeated mean. It rejects mixed protocols or launch scopes, and does not rank no-GT diagnostics as accuracy. Apply user constraints, visual judgment and uncertainty separately. Export disables diagnostic settings but does not assert success. Run final.yml again before handing it off.

If native trajectory dumping supplied the evaluation, confirm the winner with that frozen exporter first. For the clean run, use an equivalent independently validated capture adapter if available. Otherwise report online-output accuracy separately from retrospective dump accuracy and verify production completion and throughput. Do not claim the diagnostic score was reproduced by a clean config under different output semantics. Do not silently turn debug dumps back on in the delivered config.

Keep campaign data, run artifacts and final configs outside the skill package. No large media, engines or results belong in the skill source tree.
