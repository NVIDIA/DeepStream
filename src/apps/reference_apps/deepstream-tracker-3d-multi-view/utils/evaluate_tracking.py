# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Evaluate saved SCOUT or WILDTRACK trajectories with pinned public TrackEval."""

import csv
import json
import math
import yaml
import hashlib
import argparse
import trackeval
import configparser

import numpy as np

from pathlib import Path
from tabulate import tabulate
from importlib.metadata import distribution

APP = Path(__file__).resolve().parents[1]
SCOUT_REGION = [0, 26, 0, 140]
TRACKEVAL_REVISION = '12c8791b303e0a0b50f753af204249e622d0281a'
METRICS = ('HOTA', 'DetA', 'AssA', 'MOTA', 'IDF1', 'GT_Dets', 'CLR_TP', 'CLR_FP', 'CLR_FN', 'IDSW')
DATASETS = {
    'scout': {
        'ground_truth_source': 'SCOUT annotations converted by prepare_scout.py',
        'camera_id_base': 1,
        'tracker_frame_start': 1, 'tracker_frame_step': 1, 'region': SCOUT_REGION,
        'max_dist': 30.0,
        # The manifest must record this preparation. Its max_dist is the 50 m GT conversion range, not the evaluation cutoff.
        'protocol': {'sequence': 'sequence_1', 'split': 'test', 'gt_alignment': 'exact',
                     'cameras': [0, 1, 2, 3, 4, 5, 6, 7], 'region': SCOUT_REGION, 'fps': 10, 'max_dist': 50.0,
                     'frames_per_video': 5980, 'first_frame': 5980, 'last_frame': 11959},
        # Manifest files required, in tracker camera order.
        'inventory': {'cameras': [0, 1, 2, 3, 4, 5, 6, 7], 'video': 'videos/cam_{}.mp4', 'camera': 'camInfo/cam_{}.yml'},
    },
    'wildtrack': {
        'ground_truth_source': 'WILDTRACK annotations converted by prepare_wildtrack.py',
        'camera_id_base': 0, 'annotated_frames': 40,
        'tracker_frame_start': 1, 'tracker_frame_step': 5, 'region': None,
        'max_dist': 30.0,
        'protocol': {'first_official_frame': 1800, 'frames': 196, 'cameras': 7, 'fps': 25},
        'inventory': {'cameras': [0, 1, 2, 3, 4, 5, 6], 'video': 'videos/{:03d}.mp4', 'camera': 'camInfo/{:03d}.yml'},
    },
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def camera_geometry(path):
    values = yaml.safe_load('\n'.join(line for line in path.read_text().splitlines()
                                      if not line.startswith('%YAML:')))
    projection = np.asarray(values['projectionMatrix_3x4_w2p'], dtype=float).reshape(3, 4)
    center = -np.linalg.solve(projection[:, :3], projection[:, 3])
    models = values['modelInfo']
    if isinstance(models, list):
        models = next((model for model in models if model.get('classID') == 1), models[0])
    height = float(models['height'])
    if not np.isfinite(center).all() or not math.isfinite(height) or height <= 0:
        raise ValueError(f'Invalid camera geometry: {path}')
    return center, height


def add_observation(objects, frame, identity, camera, x, y, geometry, region, max_dist=math.inf):
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError('Nonfinite world position')
    if region and not (region[0] < x < region[1] and region[2] < y < region[3]):
        return
    center, height = geometry[camera]
    distance = (x-center[0])**2 + (y-center[1])**2 + (height/2-center[2])**2
    if distance > max_dist*max_dist:
        return
    candidate = (distance, camera, x, y)
    key = (frame, identity)
    objects.setdefault(key, []).append(candidate)


def fuse(objects, method):
    fused = {}
    for key, views in objects.items():
        if method == 'closest':
            fused[key] = min(views, key=lambda row: row[:2])
        else:
            points = np.asarray([row[2:] for row in views])
            x, y = np.median(points, axis=0) if method == 'median' else np.mean(points, axis=0)
            fused[key] = (0, -1, float(x), float(y))
    return fused


def read_ground_truth(path, frames, camera_base, geometry, region, max_dist=math.inf):
    objects = {}
    with path.open() as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = line.split()
            if len(row) != 9:
                raise ValueError(f'{path}:{number}: expected nine GT fields')
            camera, identity, frame = map(int, row[:3])
            camera -= camera_base
            if camera not in geometry:
                raise ValueError(f'{path}:{number}: unknown camera')
            if 1 <= frame <= frames:
                add_observation(objects, frame, identity, camera, float(row[7]), float(row[8]), geometry, region, max_dist)
    return objects


def read_predictions(paths, frame_map, geometry, region, max_dist=math.inf):
    objects = {}
    for camera, path in enumerate(paths):
        with path.open() as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = line.split(',', 9)
                if len(row) < 9:
                    raise ValueError(f'{path}:{number}: expected at least nine trajectory fields')
                frame = int(row[0])
                if frame in frame_map:
                    add_observation(objects, frame_map[frame], int(row[1]), camera,
                                    float(row[7]), float(row[8]), geometry, region, max_dist)
    return objects


def sequence_data(ground_truth, predictions, frames, zero_distance):
    data = {'num_timesteps': frames, 'gt_ids': [], 'tracker_ids': [], 'similarity_scores': []}
    for name, objects in [('gt', ground_truth), ('tracker', predictions)]:
        identities = sorted({identity for _, identity in objects})
        mapping = {identity: index for index, identity in enumerate(identities)}
        data[f'num_{name}_ids'] = len(identities)
        data[f'num_{name}_dets'] = len(objects)
        grouped = [[] for _ in range(frames)]
        for (frame, identity), (_, _, x, y) in sorted(objects.items()):
            grouped[frame-1].append((mapping[identity], x, y))
        data[f'{name}_dets'] = []
        for rows in grouped:
            data[f'{name}_ids'].append(np.asarray([row[0] for row in rows], dtype=int))
            data[f'{name}_dets'].append(np.asarray([row[1:] for row in rows], dtype=float).reshape(-1, 2))
    for gt, pred in zip(data['gt_dets'], data['tracker_dets']):
        distances = np.linalg.norm(gt[:, None, :] - pred[None, :, :], axis=2)
        data['similarity_scores'].append(np.maximum(0, 1-distances/zero_distance))
    return data


def score(ground_truth, predictions, frames, zero_distance=2.0):
    data = sequence_data(ground_truth, predictions, frames, zero_distance)
    hota = trackeval.metrics.HOTA({'PRINT_CONFIG': False}).eval_sequence(data)
    clear = trackeval.metrics.CLEAR({'PRINT_CONFIG': False, 'THRESHOLD': .5}).eval_sequence(data)
    identity = trackeval.metrics.Identity({'PRINT_CONFIG': False, 'THRESHOLD': .5}).eval_sequence(data)
    return {'HOTA': 100*float(np.mean(hota['HOTA'])),
            'DetA': 100*float(np.mean(hota['DetA'])), 'AssA': 100*float(np.mean(hota['AssA'])),
            'MOTA': 100*float(clear['MOTA']),
            'IDF1': 100*float(identity['IDF1']), 'GT_Dets': data['num_gt_dets'], 'CLR_TP': int(clear['CLR_TP']),
            'CLR_FP': int(clear['CLR_FP']), 'CLR_FN': int(clear['CLR_FN']), 'IDSW': int(clear['IDSW'])}


def write_mot(path, objects):
    with path.open('w') as stream:
        for (frame, identity), (_, _, x, y) in sorted(objects.items()):
            stream.write(f'{frame} {identity} -1 -1 -1 -1 1 {x:.9f} {y:.9f} -1\n')


def run_path(experiment, value):
    path = Path(value)
    if path.is_relative_to('/workspace/experiments'):
        return experiment / path.relative_to('/workspace/experiments')
    return path if path.is_absolute() else experiment / path


def experiment_inputs(experiment, dataset, frames=None):
    tracker_path = experiment / 'config_tracker.yml'
    tracker = yaml.safe_load('\n'.join(line for line in tracker_path.read_text().splitlines() if not line.startswith('%YAML:')))
    cameras = [dataset / 'camInfo' / Path(name).name for name in tracker['ObjectModelProjection']['cameraModelFilepath']]
    if not cameras or any(not path.is_file() for path in cameras):
        raise ValueError('Tracker camera files do not match the selected dataset')
    prefix = run_path(experiment, tracker['TrajectoryManagement']['trajectoryDumpFileName'])
    paths = [Path(f'{prefix}{index}.txt') for index in range(len(cameras))]
    main = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=('#',))
    main.read(experiment / 'config_deepstream.txt')
    folder = run_path(experiment, main.get('application', 'kitti-track-output-dir', fallback='tracker-kitti-dump'))
    observed = [set() for _ in cameras]
    for path in folder.glob('*.txt'):
        fields = path.stem.split('_')
        if len(fields) != 3 or not all(field.isdigit() for field in fields):
            continue
        camera, frame = int(fields[1]), int(fields[2])
        if camera >= len(cameras):
            raise ValueError(f'Unexpected KITTI stream index: {path}')
        observed[camera].add(frame)
    if any(observed):
        counts = [len(values) for values in observed]
        if len(set(counts)) != 1 or any(values != set(range(counts[0])) for values in observed):
            raise ValueError('KITTI output has missing frames or inconsistent camera coverage')
        if frames is not None and frames > counts[0]:
            raise ValueError('--frames exceeds the saved frame coverage')
        frames = counts[0] if frames is None else frames
    if frames is None or frames < 1:
        raise ValueError('No frame coverage found. Pass --frames N when per-frame KITTI output is unavailable.')
    log = experiment / 'run_deepstream_app.log'
    completed = None
    if log.is_file():
        completed = 'App run successful' in log.read_text(errors='replace')
        if not completed:
            raise ValueError('The saved DeepStream log does not show App run successful')
    return cameras, paths, frames, completed


def verify_dataset_manifest(dataset: Path, name: str, settings: dict) -> dict:
    """Check that the dataset was prepared for this evaluation protocol and still matches its manifest."""
    manifest_path = dataset / 'manifest.json'
    if not manifest_path.is_file():
        raise ValueError(f'Missing {manifest_path}. Generate the dataset with scripts/setup_datasets.sh before evaluating.')
    regenerate = 'Regenerate the dataset with scripts/setup_datasets.sh.'
    try:
        manifest = json.loads(manifest_path.read_text())
        files = manifest['files']
        recorded = files['ground_truth.txt']['sha256']
        annotated = int(manifest['annotated_frames'])
        prepared_for = str(manifest['dataset']).lower()
    except (KeyError, TypeError, ValueError, AttributeError, json.JSONDecodeError) as error:
        raise ValueError(f'{manifest_path} is incomplete. {regenerate}') from error
    prepared = manifest.get('protocol', {})
    if not isinstance(files, dict) or not isinstance(prepared, dict):
        raise ValueError(f'{manifest_path} is incomplete. {regenerate}')
    if prepared_for != name:
        raise ValueError(f'{manifest_path} was prepared for {prepared_for}, not {name}. {regenerate}')
    if 'annotated_frames' in settings and annotated != settings['annotated_frames']:
        raise ValueError(f'{manifest_path} records {annotated} annotated frames but the {name} protocol uses '
                         f'{settings["annotated_frames"]}. {regenerate}')
    if annotated < 1:
        raise ValueError(f'{manifest_path} records no annotated frames. {regenerate}')
    for key, expected in settings.get('protocol', {}).items():
        actual = prepared.get(key)
        if actual != expected:
            raise ValueError(f'{manifest_path} was prepared with {key}={actual!r} but the {name} protocol needs '
                             f'{key}={expected!r}. {regenerate}')
    inventory = settings.get('inventory')
    if inventory:
        required = [item.format(index) for index in inventory['cameras'] for item in (inventory['video'], inventory['camera'])]
        absent = [item for item in required if item not in files]
        if absent:
            raise ValueError(f'{manifest_path} does not list {", ".join(absent[:3])}'
                             f'{" and more" if len(absent) > 3 else ""}. {regenerate}')
    if 'first_frame' in prepared and (prepared['last_frame'] - prepared['first_frame'] + 1 != prepared['frames_per_video']
                                      or annotated > prepared['frames_per_video']):
        raise ValueError(f'{manifest_path} has an inconsistent frame span or annotated count. {regenerate}')
    if sha256(dataset / 'ground_truth.txt') != recorded:
        raise ValueError(f'GT hash does not match manifest.json. {regenerate}')
    for relative, entry in files.items():
        if relative in ('ground_truth.txt', 'map.png'):
            continue
        path = dataset / relative
        if not path.is_file():
            raise ValueError(f'{relative} listed in manifest.json is missing. {regenerate}')
        # Videos are large, so only their size is checked.
        field = 'bytes' if relative.startswith('videos/') else 'sha256'
        if not isinstance(entry, dict) or field not in entry:
            raise ValueError(f'{relative} has no {field} in manifest.json. {regenerate}')
        if (path.stat().st_size if field == 'bytes' else sha256(path)) != entry[field]:
            raise ValueError(f'{relative} does not match manifest.json. {regenerate}')
    return manifest


def evaluate(args):
    experiment = args.experiment.resolve()
    if not experiment.is_dir():
        raise ValueError(f'Experiment not found: {experiment}. Pass an existing completed run directory containing config_tracker.yml.')
    direct_url = json.loads(distribution('trackeval').read_text('direct_url.json') or '{}')
    if direct_url.get('vcs_info', {}).get('commit_id') != TRACKEVAL_REVISION:
        raise ValueError('Run bash scripts/setup_evaluation.sh to install the pinned evaluator')
    dataset = args.dataset_dir.resolve(strict=True)
    settings = dict(DATASETS[args.dataset])
    cameras, paths, run_frames, completed = experiment_inputs(experiment, dataset, args.frames)
    inventory = settings.get('inventory')
    if inventory and [path.name for path in cameras] != [Path(inventory['camera'].format(index)).name for index in inventory['cameras']]:
        raise ValueError(f'Experiment cameras {[path.name for path in cameras]} do not match the {args.dataset} dataset cameras in order')
    ground_truth_path = dataset / 'ground_truth.txt'
    manifest = verify_dataset_manifest(dataset, args.dataset, settings)
    settings['annotated_frames'] = manifest['annotated_frames']
    output = (args.output_dir or experiment / 'evaluation').resolve()
    geometry = {}
    for camera, path in enumerate(cameras):
        geometry[camera] = camera_geometry(path)
    start, step = settings['tracker_frame_start'], settings['tracker_frame_step']
    frames = min(settings['annotated_frames'], (run_frames-start)//step + 1)
    if frames < 1:
        raise ValueError('Run has no annotated frames')
    frame_map = {start+index*step: index+1 for index in range(frames)}
    region = args.region if args.region is not None else settings['region']
    if region and (not all(math.isfinite(value) for value in region) or
                   region[0] >= region[1] or region[2] >= region[3]):
        raise ValueError('Region must have finite, increasing X and Y bounds')
    max_dist = settings['max_dist'] if args.max_dist is None else args.max_dist
    if math.isnan(max_dist) or max_dist <= 0:
        raise ValueError('--max-dist must be positive or inf')
    ground_truth = read_ground_truth(ground_truth_path, frames, settings['camera_id_base'], geometry, region, max_dist)
    predictions = read_predictions(paths, frame_map, geometry, region, max_dist)
    collisions = sum(len(views)-len({view[1] for view in views}) for views in predictions.values())
    ground_truth = fuse(ground_truth, args.fusion_method)
    predictions = fuse(predictions, args.fusion_method)
    if not ground_truth:
        raise ValueError('No GT observations in the selected evaluation scope')
    metrics = score(ground_truth, predictions, frames, args.zero_distance)
    report = {'dataset': args.dataset, 'experiment': str(experiment), 'metrics': metrics,
              'metric_scale': 'HOTA, DetA, AssA, MOTA and IDF1 are multiplied by 100. MOTA can be negative. CLR_TP, CLR_FP, CLR_FN and IDSW are CLEAR counts.',
              'trackeval_revision': TRACKEVAL_REVISION,
              'trackeval_metric_sha256': {name: sha256(Path(distribution('trackeval').locate_file(f'trackeval/metrics/{name}.py')))
                                         for name in ('hota', 'clear', 'identity')},
              'trackeval_compatibility': 'Deprecated numpy.float and numpy.int type aliases replaced with float and int.',
              'ground_truth_source': settings['ground_truth_source'],
              'ground_truth_sha256': manifest['files']['ground_truth.txt']['sha256'],
              'dataset_manifest_sha256': sha256(dataset / 'manifest.json'), 'dataset_protocol': manifest.get('protocol', {}),
              'calibration_sha256': {path.name: sha256(path) for path in cameras},
              'config_sha256': {path.name: sha256(path) for path in experiment.glob('config*') if path.is_file()},
              'trajectory_sha256': {path.name: sha256(path) for path in paths},
              'frames_evaluated': frames, 'annotated_frames': settings['annotated_frames'],
              'partial_sequence': frames < settings['annotated_frames'],
              'tracker_frames_evaluated': list(frame_map), 'run_frames': run_frames,
              'completion_verified_from_log': completed,
              'ground_truth_camera_id_base': settings['camera_id_base'],
              'same_camera_id_collision_rows': collisions,
              'region': region, 'fusion_method': args.fusion_method, 'zero_distance': args.zero_distance,
              'max_dist': max_dist if math.isfinite(max_dist) else 'inf',
              'distance_filter': 'Camera center to object center at model height/2, applied to GT and predictions before fusion, in the world units of the dataset.',
              'matching': 'max(0, 1 - XY_distance / zero_distance), in the world units of the dataset',
              'clear_identity_distance_threshold': args.zero_distance/2,
              'postprocessing': 'None beyond ROI and camera-distance filtering and the selected multi-camera aggregation. Tracker identities are preserved.'}
    output.mkdir(parents=True, exist_ok=True)
    write_mot(output / 'ground_truth_mot.txt', ground_truth)
    write_mot(output / 'predictions_mot.txt', predictions)
    (output / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    with (output / 'metrics.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(METRICS)
        writer.writerow([f'{metrics[key]:.2f}' if key in METRICS[:5] else metrics[key] for key in METRICS])
    print(f'\n{args.dataset.upper()}: {experiment.name}')
    print(tabulate([[metrics[key] for key in METRICS]], headers=METRICS,
                   floatfmt='.2f', intfmt=',', numalign='right'))
    print(f'{frames}/{settings["annotated_frames"]} annotated frames evaluated. Results: {output}')
    if completed is None:
        print('No completion log was supplied. Results cover only the declared or saved frame range.')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('scout', 'wildtrack'), required=True)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--dataset-dir', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--frames', type=int, help='Explicit saved-frame count when KITTI frame files are unavailable')
    parser.add_argument('--zero-distance', type=float, default=2.0)
    parser.add_argument('--max-dist', type=float, help='Maximum camera-to-object distance in world units (default: 30). Use inf to disable.')
    parser.add_argument('--fusion-method', choices=('closest', 'median', 'mean'), default='closest')
    parser.add_argument('--region', type=float, nargs=4, metavar=('XMIN', 'XMAX', 'YMIN', 'YMAX'))
    args = parser.parse_args()
    args.dataset_dir = args.dataset_dir or APP / 'datasets' / args.dataset
    if not math.isfinite(args.zero_distance) or args.zero_distance <= 0:
        parser.error('--zero-distance must be finite and positive')
    if args.frames is not None and args.frames < 1:
        parser.error('--frames must be positive')
    try:
        evaluate(args)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f'ERROR: {error}\n')


if __name__ == '__main__':
    main()
