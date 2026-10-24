<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# DeepStream Container: Step-by-step Instructions

This page provides detailed step-by-step instructions for running MV3DT using the DeepStream Container. For quick start scripts, see the [main README](../README.md#option-1-running-mv3dt-using-deepstream-container).

Complete the [prerequisites](../README.md#prerequisites) first and use the `DS_DIR` set there.

```bash
MV3DT_DIR="$DS_DIR/src/apps/reference_apps/deepstream-tracker-3d-multi-view"
cd "$MV3DT_DIR"
```

## Sample 1: 4-camera dataset

1. Set paths reused by the commands below
    ```bash
    DATASET_DIR="$MV3DT_DIR/datasets/mtmc_4cam"
    EXPERIMENT_DIR="$MV3DT_DIR/experiments/deepstream/4cam"
    ```
2. Generate DeepStream configuration files using the auto-configurator
   
   The auto-configurator automatically generates all necessary configuration files based on your dataset. It supports various output options (OSD display, video file output, Kafka streaming) and can work with both 2D and 3D tracker configurations. 
   
   **About Override Files:** The `--config-overrides` parameter allows you to apply dataset-specific settings. For example, `config_templates/common/override_tracker_4cam.yml` is optimized for the sample 4-camera dataset (which uses feet as world coordinate units). You can create custom override files for your own datasets.
   
   For more info on the auto-configurator, see [`utils/README.md`](../utils/README.md#deepstream_auto_configuratorpy).
   
    ```bash
    # Activate the Python environment
    source mv3dt_venv/bin/activate
    
    # Generate configs with 4-camera overrides
    python -m utils.deepstream_auto_configurator \
        --profile=mtmc_4cam_pnt \
        --dataset-dir="$DATASET_DIR" \
        --enable-msg-broker \
        --enable-osd \
        --config-overrides=override_tracker_4cam.yml \
        --output-dir="$EXPERIMENT_DIR"
    
    # [Expected output] You should see
    #   Generated files:
    #     - config_deepstream.txt (main pipeline config)
    #     - config_tracker.yml (3D tracker config)
    #     - config_msgconv.txt (message converter config)
    #     - pub_sub_info_config_0.yml (communication config)

    ```
3. (Optional) Launch real-time BEV visualization
    
    Before launching the main MV3DT pipeline, optionally start the bird's-eye view visualizer to see real-time 3D tracking results. Please keep it running in a separate terminal window or add `&` to the end of the command to run it in the background.
    
    ```bash
    # Start BEV visualization
    python utils/kafka_bev_visualizer.py \
        --dataset-path="$DATASET_DIR" \
        --msgconv-config="$EXPERIMENT_DIR/config_msgconv.txt" \
        --average-multi-cam \
        --show-ids 

    # [Expected output] You should see a window named "Bird-Eye View of Multi-View 3D Tracking" pop up and will display the live tracking results.
    # Select the window and press 'q' to quit.
    ```

4. Launch MV3DT

    The following command mounts the necessary folders into the DeepStream container and starts the `deepstream-test5-app` with MV3DT configs. 

    ```bash
    sudo xhost + # give container access to display

    docker run -t --privileged --rm --net=host --runtime=nvidia \
        -v "$MV3DT_DIR/models:/workspace/models" \
        -v "$DATASET_DIR:/workspace/inputs" \
        -v "$EXPERIMENT_DIR:/workspace/experiments" \
        -v /tmp/.X11-unix/:/tmp/.X11-unix \
        -e DISPLAY=$DISPLAY \
        -w /workspace/experiments \
        ghcr.io/nvidia/deepstream:9.2-triton-multiarch \
        deepstream-test5-app -c config_deepstream.txt

    # [Expected output] You should see a window named "DeepStreamTest5App" pop up and will display 4 camera views in a grid.
    # Select the window and press 'q' to quit early.
    # The pipeline will quit automatically with "App run successful" as the last line from the logs.
    ```

Select `mtmc_4cam_pn2` for PeopleNet 2.6.3 or `mtmc_4cam_detr` for RT-DETR. Keep the same override file for all three 4-camera profiles.

## Sample 2: 12-camera dataset

The steps are the same as for the 4-camera dataset, except setting `DATASET_DIR` and `EXPERIMENT_DIR` to the 12-camera directories. The auto-configurator automatically detects the number of cameras in your dataset and generates required config files for 12-camera dataset.

```bash
DATASET_DIR="$MV3DT_DIR/datasets/mtmc_12cam"
EXPERIMENT_DIR="$MV3DT_DIR/experiments/deepstream/12cam"

python -m utils.deepstream_auto_configurator \
    --profile=mtmc_12cam_pnt \
    --dataset-dir="$DATASET_DIR" \
    --enable-msg-broker \
    --enable-osd \
    --output-dir="$EXPERIMENT_DIR"
```

Select `mtmc_12cam_detr` for the tuned RT-DETR profile or `mtmc_12cam_pn2` for the PeopleNet 2.6.3 baseline. Each profile supplies `config_pgie.txt` and `config_tracker.yml`.

## Sample 3: WILDTRACK

WILDTRACK has seven camera views, each with 196 frames encoded at 25 FPS. The example detects people with [PeopleNet Transformer](https://catalog.ngc.nvidia.com/orgs/nvidia/tao/models/peoplenet_transformer_v2/deployable_v1.0) and uses geometric 3D tracking. Pose, visual tracking, local ReID, and the ReID service are disabled.

First generate the dataset with `bash scripts/setup_datasets.sh --dataset wildtrack --download`, see [Datasets](../README.md#datasets-for-scout-and-wildtrack). Quick start: `bash scripts/test_wildtrack_ds.sh`, which replays the recorded detections and writes to `experiments/deepstream/wildtrack_run1`. Set `EXPERIMENT_DIR` to a new directory for each additional run, and see the [launcher notes](../README.md#scout-and-wildtrack) for `REPLAY`. The manual steps follow.

```bash
cd "$MV3DT_DIR"
source mv3dt_venv/bin/activate
DATASET_DIR="$MV3DT_DIR/datasets/wildtrack"
EXPERIMENT_DIR="$MV3DT_DIR/experiments/deepstream/wildtrack_run1"

python -m utils.deepstream_auto_configurator \
    --profile wildtrack \
    --dataset-dir "$DATASET_DIR" \
    --output-dir "$EXPERIMENT_DIR" \
    --replay "$DATASET_DIR/detections/PeopleNetTransformer" \
    --enable-file-output
```

Then use the [Docker launch command](#launch-scout-or-wildtrack) below. The native profile is `config_templates/wildtrack`. Peer communication is in-memory. Its pipeline preserves video timestamps and the verified 1920 by 1088 tracker resolution.

The command replays the recorded detections. Remove the `--replay` line to run live. A live run also detects people outside the annotated region, which count as false positives, so lower scores are expected. Annotation alignment uses source frame numbers, not the encoded 25 FPS.

## Sample 4: SCOUT

SCOUT has eight camera views, each with 5,980 frames at 10 FPS, from the second half of the official `sequence_1`. The default profile detects people with [PeopleNet Transformer](https://catalog.ngc.nvidia.com/orgs/nvidia/tao/models/peoplenet_transformer_v2/deployable_v1.0) and uses local CLIP ReID.

First generate the dataset with `bash scripts/setup_datasets.sh --dataset scout`, see [Datasets](../README.md#datasets-for-scout-and-wildtrack). Quick start: `bash scripts/test_scout_ds.sh`, which replays the recorded detections and writes to `experiments/deepstream/scout_pnt_run1`. Select PeopleNet 2.6.3 with `DETECTOR_MODEL=PeopleNet2.6.3` (output `scout_pn2_run1`). Set `EXPERIMENT_DIR` to a new directory for each additional run, and see the [launcher notes](../README.md#scout-and-wildtrack) for `REPLAY`. The manual steps follow.

```bash
cd "$MV3DT_DIR"
source mv3dt_venv/bin/activate
DATASET_DIR="$MV3DT_DIR/datasets/scout"
EXPERIMENT_DIR="$MV3DT_DIR/experiments/deepstream/scout_pnt_run1"

python -m utils.deepstream_auto_configurator \
    --profile scout_pnt \
    --dataset-dir "$DATASET_DIR" \
    --output-dir "$EXPERIMENT_DIR" \
    --replay "$DATASET_DIR/detections/PeopleNetTransformer" \
    --enable-file-output
```

Then use the [Docker launch command](#launch-scout-or-wildtrack) below. Select `--profile scout_pn2` for [PeopleNet 2.6.3](https://catalog.ngc.nvidia.com/orgs/nvidia/tao/models/peoplenet/deployable_quantized_onnx_v2.6.3/file-browser).

The command replays the recorded detections, and `--profile scout_pn2` needs `detections/PeopleNet2.6.3` instead. Remove the `--replay` line to run live. Replay replaces only detector inference, so local CLIP ReID stays on in the Transformer profile.

See the [main README](../README.md) for the reference replay results. With local CLIP ReID enabled, repeated runs can vary slightly. Evaluate each run separately with the [evaluation command](#evaluation).

### Optional ReID service

Deploy the service and its dependencies using the public [VSS standalone ReID instructions](https://docs.nvidia.com/vss/system-components/vision-microservices/object-detection-and-tracking/mv-3-dt-standalone-deployment#33-optional-reid-service). Use the public `nvcr.io/nvidia/vss-core/reid-embed:3.3.0` image. In the VSS standalone `docker/.env`, set the following values to match this sample:

```bash
KAFKA_BOOTSTRAP=localhost:9092
RAW_TOPIC=mv3dt
REID_INPUT_TOPIC=mv3dt
REID_SERVICE_HOST=127.0.0.1
REID_SERVICE_PORT=8000
REID_DIMENSION=1280
REID_ENABLE_COMPRESSION=False
REID_ENABLE_SECONDARY_EMBEDDING=False
REID_EMBED_IMAGE=nvcr.io/nvidia/vss-core/reid-embed
REID_EMBED_TAG=3.3.0
```

After preparing the VSS environment and model directory, start only Kafka, the ReID service, and its dependencies from the VSS standalone `docker/` directory:

```bash
docker compose --profile kafka --profile reid up -d kafka-topic-init reid-embed
```

The guide's default service port and topic differ from this sample. The service must be ready at `http://127.0.0.1:8000` and consume protobuf messages from `mv3dt`. SCOUT publishes its own CLIP embeddings, so no secondary embedding model or VIOS server is needed.

**Reset the dedicated service before every independent recording.** Stop other producers and let Kafka ingestion finish first. Reset deletes the service's stored identities, so do not reset a shared service.

```bash
curl --fail --silent --show-error http://127.0.0.1:8000/health/ready
curl --fail --silent --show-error -X POST http://127.0.0.1:8000/reset
REID_SERVICE=1 bash scripts/test_scout_ds.sh
```

For PeopleNet 2.6.3, add `DETECTOR_MODEL=PeopleNet2.6.3` to the launcher command. Live inference uses `REPLAY=0`. Service-enabled runs default to `scout_pnt_reid_run1` or `scout_pn2_reid_run1`. The launcher checks readiness and uses host networking to reach local Kafka and HTTP endpoints. It does not deploy, reset, or stop the service.

For manual configuration generation, add `--enable-reid-service` to the SCOUT command above and use `--network host` in the Docker launch below. The flag merges `common/override_reid_service.yml` into the selected tracker and enables the Kafka sink. Other association settings and in-memory peer communication are preserved. After the run, check `curl --fail --silent --show-error http://127.0.0.1:8000/count` and the service logs for embedding ingestion and `/search` requests. Readiness alone does not verify the Kafka path, and service availability does not imply an accuracy improvement.

## Launch SCOUT or WILDTRACK

Use a new experiment directory for each run. The same auto-configurator used by Samples 1 and 2 prepares the native configs and output directories. These two profiles automatically use all-to-all in-memory peers. Kafka and the external ReID service are disabled unless explicitly enabled for SCOUT above. GT is used only for evaluation.

Models are mounted from `models/`. Replace the host path in that mount if using another prepared model tree. Models are mounted writable so TensorRT can cache engines, as in the existing samples. The dataset mount is read-only.

```bash
set -o pipefail
docker run -t --rm --gpus all --network bridge \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    -v "$MV3DT_DIR/models:/workspace/models" \
    -v "$DATASET_DIR:/workspace/inputs:ro" \
    -v "$EXPERIMENT_DIR:/workspace/experiments" \
    -w /workspace/experiments \
    "${DEEPSTREAM_IMAGE:-ghcr.io/nvidia/deepstream:9.2-triton-multiarch}" \
    deepstream-test5-app -c config_deepstream.txt \
    2>&1 | tee "$EXPERIMENT_DIR/run_deepstream_app.log"
grep -q 'App run successful' "$EXPERIMENT_DIR/run_deepstream_app.log"
```

This command is headless. Docker's `-t` prevents repeated blank lines from the app's keyboard polling at EOF. `--enable-file-output` saves `outVideos/tiled_display_raw.mp4`. For live display, generate with `--enable-osd` and provide an authorized X11 session, `DISPLAY`, and the X11 socket mount as in Sample 1.

Outputs include `config_deepstream.txt`, `config_pgie.txt`, `config_tracker.yml`, `config_msgconv.txt`, `pub_sub_info_config_0.yml`, the run log, `tracker-kitti-dump/`, `infer-kitti-dump/`, and `trajDumps/`. See [configuration profiles](../config_templates/README.md) for profile contents.

## Evaluation

Evaluation uses pinned public [TrackEval](https://github.com/JonathonLuiten/TrackEval), installed into `mv3dt_venv` during prerequisite setup.

```bash
# Evaluate WILDTRACK
mv3dt_venv/bin/python -m utils.evaluate_tracking --dataset wildtrack --experiment experiments/deepstream/wildtrack_run1
# Evaluate SCOUT
mv3dt_venv/bin/python -m utils.evaluate_tracking --dataset scout --experiment experiments/deepstream/scout_pnt_run1
```

Pass the experiment directory containing the generated native configs, trajectory dumps, and saved per-frame KITTI files. Use your own run name, for example `scout_pnt_run1`. List available runs with `find experiments/deepstream -name config_tracker.yml`. Use `--output-dir PATH` to evaluate a run again without overwriting its existing results. When KITTI files are unavailable, pass the saved source-frame count with `--frames N`. A saved run log must contain `App run successful`, and without a log the report says completion is not verified.

The wrapper reads saved trajectory dumps and evaluates ground-plane XY positions. `--fusion-method closest` is the default. `median` and `mean` are also supported. Aggregation is applied independently to GT and predictions for each frame and ID. Tracker IDs are preserved.

Camera-distance filtering applies to GT and predictions before aggregation. The default cutoff is 30 world units. Override it with `--max-dist <value>` or disable it with `--max-dist inf`. Distance is measured from the camera center to the assumed 3D box center.

GT rows contain `camera_id person_id frame_id bbox_left bbox_top bbox_width bbox_height world_x world_y`.

Similarity is `max(0, 1 - XY_distance / zero_distance)`, with `--zero-distance 2` by default in the world units of the dataset's `camInfo` and ground truth. CLEAR and Identity use a similarity threshold of 0.5, corresponding to distance 1 at the default setting. HOTA averages over its standard similarity thresholds. Native units and ROI settings must match before comparing different datasets or calibration versions. `--region XMIN XMAX YMIN YMAX` overrides the sample's region.

Results go to `evaluation/` inside the experiment, or `--output-dir PATH`. Re-evaluation replaces `metrics.json`, `metrics.csv`, and the converted GT/prediction MOT files, leaving other files untouched. Reported columns are HOTA, DetA, AssA, MOTA, IDF1, GT_Dets, CLR_TP, CLR_FP, CLR_FN, and IDSW. HOTA, DetA, AssA, MOTA, and IDF1 are multiplied by 100 and print with two decimals. DetA and AssA are averaged across HOTA's similarity thresholds. MOTA can be negative. GT_Dets counts evaluated GT person-frames after filtering and fusion, equal to CLR_TP + CLR_FN. CLR_TP, CLR_FP, CLR_FN, and IDSW are CLEAR counts.

TrackEval setup replaces deprecated NumPy type aliases in the installed HOTA and Identity modules for Python 3.12 compatibility. Metric formulas are unchanged. Run the evaluation tests with:

```bash
mv3dt_venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```
