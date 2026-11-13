#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adapt observations, evaluate world-space tracks, and flag review intervals."""

import csv
import json
import math
import hashlib
import argparse
import importlib.util

from pathlib import Path
from itertools import pairwise
from collections import defaultdict
from importlib.metadata import distribution

FIELDS = ('camera', 'frame', 'id', 'class', 'x', 'y', 'left', 'top', 'width', 'height')
APP = Path(__file__).resolve().parents[3] / 'src/apps/reference_apps/deepstream-tracker-3d-multi-view'


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def load_rows(path):
    with Path(path).open() as handle:
        reader = csv.DictReader(handle)
        if not {'camera', 'frame', 'id', 'class'}.issubset(reader.fieldnames or []):
            raise ValueError('Canonical CSV needs camera, frame, id and class columns')
        if len(set(reader.fieldnames)) != len(reader.fieldnames) or any(not field for field in reader.fieldnames):
            raise ValueError(f'{path}: CSV columns must have unique nonempty names')
        rows = []
        for number, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f'{path}:{number}: row does not match its CSV columns')
            result = {key: row.get(key, '') for key in FIELDS}
            result['frame'] = int(result['frame'])
            if result['frame'] < 1 or not result['id'] or not result['camera'] or not result['class']:
                raise ValueError(f'{path}:{number}: invalid frame, camera or identity')
            for key in FIELDS[4:]:
                result[key] = float(result[key]) if result[key] != '' else None
                if result[key] is not None and not math.isfinite(result[key]):
                    raise ValueError(f'{path}:{number}: nonfinite {key}')
            rows.append(result)
        return rows


def convert(args):
    mapping = json.loads(Path(args.frame_map).read_text()) if args.frame_map else None
    rows = []
    with Path(args.input).open() as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip() or line.startswith('#'):
                continue
            fields = [field.strip() for field in line.split(',')] if args.format == 'deepstream-traj' else line.split()
            if len(fields) < 9:
                raise ValueError(f'{args.input}:{number}: expected at least nine fields')
            if args.format == 'deepstream-traj':
                if args.camera is None:
                    raise ValueError('Specify --camera for each trajectory dump')
                if len(fields) < 11:
                    raise ValueError(f'{args.input}:{number}: native trajectory requires its class field at index 10')
                frame, identity = fields[0:2]
                camera, box, position = args.camera, fields[2:6], fields[7:9]
                class_id = fields[10]
                if args.class_id is not None and class_id != str(args.class_id):
                    continue
            else:
                if len(fields) != 9:
                    raise ValueError(f'{args.input}:{number}: nine-field GT requires exactly nine fields')
                if args.class_id is None:
                    raise ValueError('Specify --class-id for unlabelled nine-field GT')
                camera, identity, frame = fields[:3]
                box, position = fields[3:7], fields[7:9]
                class_id = str(args.class_id)
            original = int(frame)
            if mapping is not None:
                table = mapping[str(camera)]
                if str(original) not in table:
                    raise ValueError(f'Unmapped annotation frame {camera}:{original}')
                frame = table[str(original)]
                if type(frame) is not int:
                    raise ValueError('Mapped video frame must be an integer')
            else:
                frame = original - args.frame_base + 1
            if frame < 1:
                raise ValueError('Converted video frame must be one-based')
            rows.append([camera, frame, identity, class_id, *position, *box])
    with Path(args.output).open('x', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(FIELDS)
        writer.writerows(rows)
    load_rows(args.output)
    print(f'Converted {len(rows)} observations. Verify units, camera mapping and sample rows.')


def frame_list(settings):
    frames = settings['frames']
    if isinstance(frames, dict):
        start, stop, step = frames['start'], frames['stop'], frames.get('step', 1)
        if any(type(value) is not int for value in (start, stop, step)) or start < 1 or step < 1 or stop < start:
            raise ValueError('Frame range requires positive integer start/step and an inclusive stop >= start')
        frames = list(range(start, stop + 1, step))
    if not frames or frames != sorted(set(frames)) or any(type(f) is not int or f < 1 for f in frames):
        raise ValueError('Evaluation frames must be explicit, sorted, unique positive video-frame indices')
    return frames


def metric_core(path):
    spec = importlib.util.spec_from_file_location('mv3dt_metric_core', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    installed = json.loads(distribution('trackeval').read_text('direct_url.json') or '{}')
    if installed.get('vcs_info', {}).get('commit_id') != module.TRACKEVAL_REVISION:
        raise ValueError('Install the pinned TrackEval with the MV3DT sample setup_evaluation.sh')
    return module


def geometry(camera, class_id):
    import yaml

    import numpy as np
    path = Path(camera['calibration'])
    data = yaml.safe_load('\n'.join(line for line in path.read_text().splitlines() if not line.startswith('%YAML:')))
    projection = np.asarray(data['projectionMatrix_3x4_w2p'], dtype=float).reshape(3, 4)
    center = -np.linalg.solve(projection[:, :3], projection[:, 3])
    model = data['modelInfo']
    if isinstance(model, list):
        candidates = [m for m in model if str(m.get('classID')) == str(class_id)]
        if len(candidates) != 1:
            raise ValueError(f'{path}: need one explicit model for class {class_id}')
        model = candidates[0]
    height = float(model['height'])
    if not np.isfinite(center).all() or not math.isfinite(height) or height <= 0:
        raise ValueError(f'Invalid camera model: {path}')
    return center, height


def score(args):
    manifest_path = Path(args.manifest).resolve()
    data = json.loads(manifest_path.read_text())
    settings = data['evaluation']
    class_id = str(settings['class_id'])
    if not data.get('world_units'):
        raise ValueError('Declare world_units before world-space evaluation')
    frames = frame_list(settings)
    indices = {frame: i + 1 for i, frame in enumerate(frames)}
    method = settings['fusion']
    if method not in ('closest', 'median', 'mean'):
        raise ValueError('Use closest, median or mean with the sample metric core')
    zero = float(settings['zero_distance'])
    maximum = settings.get('max_distance')
    maximum = math.inf if maximum is None else float(maximum)
    if not math.isfinite(zero) or zero <= 0 or maximum <= 0 or math.isnan(maximum):
        raise ValueError('Invalid matching or camera-distance settings')
    region = settings.get('region')
    if region is not None and (len(region) != 4 or not all(math.isfinite(x) for x in region)
                               or region[0] >= region[1] or region[2] >= region[3]):
        raise ValueError('Invalid evaluation region')
    core = metric_core(Path(args.evaluator).resolve())
    cameras = {str(c['id']): c for c in data['cameras']}
    if len(cameras) != len(data['cameras']):
        raise ValueError('Camera IDs must be unique')
    for camera in cameras.values():
        path = Path(camera['calibration'])
        camera['calibration'] = str(path if path.is_absolute() else manifest_path.parent / path)
    geo = {name: geometry(c, class_id) for name, c in cameras.items()}
    fingerprints = {name: {'path': c['calibration'], 'sha256': sha256(c['calibration'])}
                    for name, c in cameras.items()}
    truth_path = Path(data['ground_truth'])
    if not truth_path.is_absolute():
        truth_path = manifest_path.parent / truth_path
    protocol = {'evaluation': settings, 'frames': frames, 'world_units': data['world_units'],
                'output_semantics': settings.get('output_semantics', 'unspecified'),
                'gt_sha256': sha256(truth_path), 'calibrations': fingerprints,
                'evaluator_sha256': sha256(args.evaluator), 'adapter_sha256': sha256(__file__),
                'trackeval_revision': core.TRACKEVAL_REVISION}
    protocol['package_versions'] = {name: distribution(name).version for name in ('numpy', 'scipy', 'PyYAML')}
    metric_root = Path(distribution('trackeval').locate_file('trackeval'))
    protocol['trackeval_metric_sha256'] = {str(path.relative_to(metric_root)): sha256(path)
                                         for path in sorted(metric_root.rglob('*.py'))}
    if not protocol['trackeval_metric_sha256']:
        raise ValueError('Cannot fingerprint installed TrackEval metric code')
    objects = []
    counts = []
    for path in (truth_path, Path(args.predictions)):
        views, seen = {}, set()
        eligible = 0
        for row in load_rows(path):
            if row['camera'] not in cameras:
                raise ValueError(f'Unknown camera in {path}: {row["camera"]}')
            if str(row['class']) != class_id or row['frame'] not in indices:
                continue
            key = (row['camera'], row['frame'], row['id'])
            if key in seen:
                raise ValueError(f'Duplicate camera/frame/identity in {path}: {key}')
            seen.add(key)
            if row['x'] is None or row['y'] is None:
                raise ValueError('World-space evaluation requires x and y for every eligible observation')
            eligible += 1
            core.add_observation(views, indices[row['frame']], row['id'], row['camera'],
                                 row['x'], row['y'], geo, region, maximum)
        counts.append({'eligible_observations': eligible, 'admitted_observations': sum(map(len, views.values()))})
        objects.append(core.fuse(views, method))
    if not objects[0]:
        raise ValueError('No admitted GT observations. Use diagnostics for an unlabelled or entirely empty recording.')
    metrics = core.score(*objects, len(frames), zero)
    signature = hashlib.sha256(json.dumps(protocol, sort_keys=True, allow_nan=False).encode()).hexdigest()
    result = {'metrics': metrics, 'protocol_sha256': signature, 'protocol': protocol,
              'prediction_file': str(Path(args.predictions).resolve()),
              'prediction_sha256': sha256(args.predictions), 'gt_counts': counts[0],
              'prediction_counts': counts[1], 'annotation_steps': len(frames)}
    with Path(args.output).open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps(metrics, indent=2))


def diagnostics(args):
    rows = load_rows(args.predictions)
    class_id = getattr(args, 'class_id', None)
    if class_id is not None:
        rows = [row for row in rows if row['class'] == str(class_id)]
    if len({row['class'] for row in rows}) > 1 and (args.speed_limit is not None or args.disagreement is not None):
        raise ValueError('Select --class-id before applying physical thresholds to mixed classes')
    fps = float(args.fps)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('Supply the physical frame rate of the normalized time axis')
    for value in (args.speed_limit, args.disagreement, args.short_seconds):
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError('Diagnostic thresholds must be finite and positive')
    if args.limit < 1:
        raise ValueError('Review event limit must be positive')
    tracks, simultaneous = defaultdict(list), defaultdict(list)
    seen, collisions = set(), []
    for row in rows:
        key = (row['camera'], row['class'], row['id'])
        tracks[key].append(row)
        simultaneous[(row['frame'], row['class'], row['id'])].append(row)
        identity = (*key, row['frame'])
        if identity in seen:
            collisions.append({'camera': row['camera'], 'frame': row['frame'], 'id': row['id']})
        seen.add(identity)
    events, short, gaps = [], 0, 0
    speed_pairs, cross_view_pairs, speed_flags, disagreement_flags = 0, 0, 0, 0
    for (camera, cls, identity), track in tracks.items():
        track.sort(key=lambda x: x['frame'])
        if (track[-1]['frame'] - track[0]['frame'] + 1) / fps < args.short_seconds:
            short += 1
        for a, b in pairwise(track):
            dt = (b['frame'] - a['frame']) / fps
            if dt <= 0:
                continue
            if b['frame'] - a['frame'] > 1:
                gaps += 1
            if all(r[k] is not None for r in (a, b) for k in ('x', 'y')):
                speed_pairs += 1
                speed = math.hypot(b['x'] - a['x'], b['y'] - a['y']) / dt
                if args.speed_limit is not None and speed > args.speed_limit:
                    speed_flags += 1
                    events.append({'kind': 'speed', 'camera': camera, 'class': cls, 'id': identity,
                                   'frame': b['frame'], 'value': speed})
    for (frame, cls, identity), views in simultaneous.items():
        points = [v for v in views if v['x'] is not None and v['y'] is not None]
        for i, a in enumerate(points):
            for b in points[i + 1:]:
                if a['camera'] == b['camera']:
                    continue
                cross_view_pairs += 1
                distance = math.hypot(a['x'] - b['x'], a['y'] - b['y'])
                if args.disagreement is not None and distance > args.disagreement:
                    disagreement_flags += 1
                    events.append({'kind': 'cross_view_disagreement', 'camera': a['camera'],
                                   'peer': b['camera'], 'class': cls, 'id': identity,
                                   'frame': frame, 'value': distance})
    events.sort(key=lambda x: x['value'], reverse=True)
    result = {'scope': 'Proxy diagnostics, not GT accuracy', 'observations': len(rows),
              'class_id': class_id,
              'camera_tracks': len(tracks), 'short_tracks': short, 'gap_events': gaps,
              'speed_comparisons': speed_pairs, 'speed_flags': speed_flags,
              'shared_id_cross_view_pairs': cross_view_pairs, 'disagreement_flags': disagreement_flags,
              'speed_flag_rate': speed_flags / speed_pairs if speed_pairs and args.speed_limit is not None else None,
              'shared_id_disagreement_rate': disagreement_flags / cross_view_pairs
                  if cross_view_pairs and args.disagreement is not None else None,
              'thresholds': {'speed_limit': args.speed_limit, 'disagreement': args.disagreement},
              'duplicate_rows': len(collisions), 'duplicate_examples': collisions[:50],
              'flagged_events': len(events), 'review_events': events[:args.limit],
              'prediction_sha256': sha256(args.predictions), 'physical_fps': fps}
    with Path(args.output).open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({k: v for k, v in result.items() if not isinstance(v, list)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(required=True)
    p = sub.add_parser('convert')
    p.add_argument('--format', choices=('nine-field-gt', 'deepstream-traj'), required=True)
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--camera')
    p.add_argument('--class-id', help='Required label for nine-field GT, optional class filter for native trajectories')
    p.add_argument('--frame-base', type=int, choices=(0, 1), default=1)
    p.add_argument('--frame-map', help='JSON: camera ID -> original frame -> normalized video frame')
    p.set_defaults(func=convert)
    p = sub.add_parser('score')
    p.add_argument('--manifest', required=True)
    p.add_argument('--predictions', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--evaluator', default=str(APP / 'utils/evaluate_tracking.py'))
    p.set_defaults(func=score)
    p = sub.add_parser('diagnostics')
    p.add_argument('--predictions', required=True)
    p.add_argument('--class-id', help='Select one class for class-specific physical thresholds')
    p.add_argument('--output', required=True)
    p.add_argument('--fps', type=float, required=True)
    p.add_argument('--speed-limit', type=float, help='Physical world units per second, chosen for the class')
    p.add_argument('--disagreement', type=float, help='World distance requiring visual review')
    p.add_argument('--short-seconds', type=float, default=.3)
    p.add_argument('--limit', type=int, default=100)
    p.set_defaults(func=diagnostics)
    args = parser.parse_args()
    try:
        args.func(args)
    except (OSError, ValueError, KeyError, ImportError) as error:
        parser.exit(1, f'ERROR: {error}\n')


if __name__ == '__main__':
    main()
