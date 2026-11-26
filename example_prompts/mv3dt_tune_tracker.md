# Tune the MV3DT tracker

Use $deepstream-tune-mv3dt to tune the MV3DT tracker on the 4-camera sample in
`src/apps/reference_apps/deepstream-tracker-3d-multi-view/datasets/mtmc_4cam`.

- Tuning budget: 30 minutes. Use it without asking again.
- The sample has no ground truth, so use diagnostics and visual review. Do not report HOTA, MOTA or IDF1.
- Begin with geometric tracking, then test optional modules one at a time.
- Report the best tested configuration, the evidence for it, and any visible failures.
- Deliver an ordinary DeepStream tracker config that runs without the tuning harness.

Set up the reference app first, as described in the MV3DT run skill.

## Variants

These notes are for the person using the prompt, not part of the task.

- With ground truth: generate the WILDTRACK sample with
  `bash scripts/setup_datasets.sh --dataset wildtrack --download`, then replace the dataset with
  `datasets/wildtrack`, replay the recorded detections in
  `datasets/wildtrack/detections/PeopleNetTransformer`, use HOTA as the primary metric and a
  1 hour budget, and drop the line about ground truth.
- Your own recordings: replace the dataset path and the budget. Without a stated budget the skill
  proposes a long one (60 times the camera-seconds) and asks you to confirm it.
