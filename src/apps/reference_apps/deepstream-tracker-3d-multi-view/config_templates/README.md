# Configuration profiles

Select a folder with `--profile`. Each profile contains native DeepStream detector and tracker files. Shared pipeline, messaging, and Inference Builder templates are in `common/`. A profile can override a shared template by providing the same filename.

| Profile | Dataset | Default detector |
|---|---|---|
| `scout_pnt` | `datasets/scout` | PeopleNet Transformer |
| `scout_pn2` | `datasets/scout` | PeopleNet 2.6.3 |
| `wildtrack` | `datasets/wildtrack` | PeopleNet Transformer |
| `mtmc_4cam_pnt` | `datasets/mtmc_4cam` | PeopleNet Transformer |
| `mtmc_4cam_pn2` | `datasets/mtmc_4cam` | PeopleNet 2.6.3 |
| `mtmc_4cam_detr` | `datasets/mtmc_4cam` | RT-DETR |
| `mtmc_12cam_pnt` | `datasets/mtmc_12cam` | PeopleNet Transformer, tuned |
| `mtmc_12cam_detr` | `datasets/mtmc_12cam` | RT-DETR, tuned |
| `mtmc_12cam_pn2` | `datasets/mtmc_12cam` | PeopleNet 2.6.3, baseline |

Every profile supplies `config_pgie.txt` and `config_tracker.yml`. The 4-camera profiles retain the sample's shared tracker settings. Their launchers apply `common/override_tracker_4cam.yml`.

WILDTRACK also supplies `config_deepstream.txt` to preserve video timestamps and its verified 1920 by 1088 tracker resolution. Explicit tracker dimensions in a profile-specific pipeline take precedence over automatic sizing. Its tracker sets person height 1.8 and radius 0.15 without modifying dataset calibration files. Pose, visual tracking, local ReID, and the ReID service are disabled.

`scripts/setup_models.sh` prepares detector, pose, and CLIP ReID models for these profiles. All PeopleNet 2.6.3 detector configs use `PeopleNet2.6.3/resnet34_peoplenet.onnx` and the provided `detector_labels.txt`.

The 4-camera and 12-camera launchers select a profile through `DETECTOR_MODEL`. DeepStream uses each profile's default tracker. The 12-camera Inference Builder RT-DETR launcher uses `config_tracker_ib.yml`. Other Inference Builder launchers use the default tracker. The optional shared `override_tracker_12cam.yml` is not applied by default.

Run from the app directory:

```bash
cd "$DS_DIR/src/apps/reference_apps/deepstream-tracker-3d-multi-view"
python -m utils.deepstream_auto_configurator \
    --profile mtmc_12cam_detr \
    --dataset-dir datasets/mtmc_12cam \
    --output-dir experiments/deepstream/12cam
```

For Inference Builder, use `python -m utils.inference_builder_auto_configurator`. Config filename arguments are resolved in the selected profile, then `common/`. Explicit paths are also accepted. The default profile is `mtmc_4cam_pnt`.

For DeepStream detection replay, add `--replay datasets/<dataset>/detections/<model>`. The folder must be inside the selected dataset and contain `000.txt`, `001.txt`, and so on in source order. Profiles using in-memory communication automatically get all-to-all peer configurations. The configurator prepares files only. Launch the container using the sample instructions.

For optional SCOUT service-backed ReID, add `--enable-reid-service`. It merges `common/override_reid_service.yml` into the selected profile, enabling CLIP embeddings and remote queries, and enables the protobuf Kafka sink. In-memory peer communication and other tracker settings are unchanged. Deploy the external service and use host networking as described in the [sample instructions](../docs/step-by-step-deepstream.md#optional-reid-service).

Generated experiment files retain their conventional names, including `config_deepstream.txt`, `config_pgie.txt`, `config_tracker.yml`, `config_msgconv.txt`, `config_mqtt.txt`, and the Inference Builder YAML files. The shared `config_tracker_2d.yml` remains available for the 2D-to-MV3DT conversion example.
