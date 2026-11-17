# SCOUT transformed calibration and positions

These files hold the camera calibration and the ground-truth person positions of SCOUT `sequence_1`, transformed once from the official SCOUT files into the world frame of the sample. They contain no images or 2D boxes. `utils/prepare_scout.py` loads them.

- `transformed_calibration.json`: intrinsics K, rotation R and translation T of cameras 0 to 7 in the transformed frame, plus one similarity transform used to draw `map.png`.
- `transformed_positions.json`: ground plane position of every (person, frame) pair of `sequence_1` in the transformed frame, columnar, x and y in integer centimeters.
- `ground_plane_correspondences.json`: hand collected ground plane points (undistorted camera pixel and mesh pixel, 19 to 30 per camera). Only `--transform-calib` reads it.

The two transformed files were produced once by the bundle adjustment in `utils/transform_scout.py`, the same procedure that `prepare_scout.py --transform-calib` runs. Recomputing them needs the annotations of all `sequence_1` frames for cameras 0 to 7, not only the second half. Install the extra packages and give a new folder for the files, outside this one:

```bash
pip install torch kornia
python3 utils/prepare_scout.py --source <official SCOUT dir> --output <output dir> --transform-calib --transformed-out <new transformed dir>
```

`--transform-calib` uses the new files instead of the recorded ones and records the calibration as `recomputed` (the evaluation report keeps it) and the sha256 of the correspondences in `manifest.json`. Existing JSON files in `--transformed-out` are overwritten. The run takes up to `--max-iterations` (default 20000) and is faster on a GPU.

The official SCOUT files are still required, see [Datasets](../../README.md#datasets-for-scout-and-wildtrack).
