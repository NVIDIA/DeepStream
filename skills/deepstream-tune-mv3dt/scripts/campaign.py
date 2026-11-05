#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inspect, budget and record isolated MV3DT tuning experiments."""

import os
import json
import math
import time
import uuid
import fcntl
import shutil
import signal
import hashlib
import argparse
import tempfile
import subprocess
import statistics
import configparser

from pathlib import Path
from datetime import datetime
from fractions import Fraction
from contextlib import contextmanager

SKILL = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')
        temporary = Path(handle.name)
    temporary.replace(path)


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def verify_hashes(files):
    for path, expected in files.items():
        if sha256(path) != expected:
            raise ValueError(f'Frozen input changed: {path}')


def number(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')
    return value


def label(value):
    if not value or any(not (c.isascii() and (c.isalnum() or c in '_-')) for c in value):
        raise ValueError('Use letters, digits, underscores or hyphens for names')
    return value


def resolve(path, base):
    result = Path(path).expanduser()
    return (base / result).resolve() if not result.is_absolute() else result.resolve()


def probe_number(value, positive=True):
    try:
        result = float(Fraction(value))
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return result if math.isfinite(result) and (result > 0 or not positive) else None


def scan_video(path, timeout):
    count = timestamp_count = 0
    first = last = previous = tail = None
    minimum_step, maximum_step = math.inf, -math.inf
    with tempfile.TemporaryFile(mode='w+') as output:
        result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_frames',
                                 '-show_entries', 'frame=best_effort_timestamp_time,duration_time,pkt_duration_time',
                                 '-of', 'compact=p=1:nk=0', str(path)], check=True, text=True,
                                stdout=output, stderr=subprocess.PIPE, timeout=timeout)
        if result.stderr.strip():
            raise ValueError(f'FFprobe reported video decode errors for {path}: {result.stderr[:500]}')
        output.seek(0)
        for line in output:
            if not line.startswith('frame|'):
                continue
            count += 1
            fields = dict(part.split('=', 1) for part in line.strip().split('|')[1:] if '=' in part)
            timestamp = probe_number(fields.get('best_effort_timestamp_time'), positive=False)
            if timestamp is None:
                continue
            timestamp_count += 1
            if previous is not None:
                step = timestamp - previous
                minimum_step, maximum_step = min(minimum_step, step), max(maximum_step, step)
            previous = timestamp
            first = timestamp if first is None else min(first, timestamp)
            if last is None or timestamp >= last:
                last = timestamp
                tail = probe_number(fields.get('duration_time') or fields.get('pkt_duration_time'))
    number(count, 'decoded video frame count')
    complete = timestamp_count == count
    duration = last - first + tail if complete and tail is not None else None
    return {'frame_count': count, 'frame_count_basis': 'decoded presented frames',
            'duration_seconds': duration, 'first_pts': first, 'last_pts': last,
            'pts_complete': complete, 'pts_monotonic': minimum_step > 0 if complete and count > 1 else None,
            'pts_variable': maximum_step - minimum_step > max(.000002, minimum_step * .01)
            if complete and count > 1 else None}


def probe(path, timeout=120):
    result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                             '-show_entries', 'stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration,time_base,start_time:format=duration',
                             '-of', 'json', str(path)], check=True, text=True,
                            capture_output=True, timeout=timeout)
    data = json.loads(result.stdout)
    if not data.get('streams'):
        raise ValueError(f'No video stream: {path}')
    stream = data['streams'][0]
    duration = probe_number(stream.get('duration'))
    fps = probe_number(stream.get('avg_frame_rate'))
    count = int(stream['nb_frames']) if str(stream.get('nb_frames', '')).isdigit() else None
    metadata = {**stream, 'duration_seconds': duration, 'duration_basis': 'video stream metadata',
                'average_fps': fps, 'frame_count': count, 'frame_count_basis': 'stream metadata',
                'container_duration': probe_number(data.get('format', {}).get('duration'))}
    scanned = scan_video(path, timeout)
    if scanned['duration_seconds'] is None:
        scanned.pop('duration_seconds')
    else:
        metadata['duration_basis'] = 'decoded video timestamps'
    metadata.update(scanned)
    if fps is None and metadata['duration_seconds']:
        metadata['average_fps'] = scanned['frame_count'] / metadata['duration_seconds']
    return metadata


def proposed_seconds(cameras):
    return sum(number(c['duration_seconds'], 'selected duration') for c in cameras) * 60


@contextmanager
def lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f'Resource is in use: {path}') from None
        yield


def inspect(args):
    source = Path(args.manifest).resolve()
    data = read_json(source)
    cameras = data.get('cameras', [])
    if any(not isinstance(c.get('id'), (str, int)) or isinstance(c['id'], bool)
           or not str(c['id']).strip() for c in cameras):
        raise ValueError('Camera IDs must be nonempty strings or integers')
    if not cameras or len({str(c['id']) for c in cameras}) != len(cameras):
        raise ValueError('Provide cameras with unique IDs')
    files = {}
    for camera in cameras:
        for key in ('video', 'calibration'):
            if not camera.get(key):
                continue
            path = resolve(camera[key], source.parent)
            if not path.is_file():
                raise ValueError(f'Missing {key}: {path}')
            camera[key] = str(path)
            files[str(path)] = sha256(path)
        metadata = probe(camera['video'], timeout=number(getattr(args, 'probe_timeout', 120), 'probe timeout'))
        camera['probe'] = metadata
        duration = metadata['duration_seconds']
        camera['duration_basis'] = metadata['duration_basis']
        if data.get('physical_fps') is not None:
            fps = number(data['physical_fps'], 'physical FPS')
            count = number(metadata['frame_count'], 'decoded video frame count for physical duration')
            duration = count / fps
            camera['duration_basis'] = 'decoded presented frames / verified physical FPS'
        if duration is None:
            raise ValueError(f'Unknown video timing for {camera["id"]}. Prepare a timestamped copy with verified capture cadence.')
        selected = number(camera.get('duration_seconds', duration), 'selected duration')
        if abs(selected - duration) > .1:
            raise ValueError(f'Trim a prepared copy before selecting a different duration: {camera["id"]}')
        camera['duration_seconds'] = selected
    durations = [c['duration_seconds'] for c in cameras]
    rates = [c['probe']['average_fps'] for c in cameras if c['probe']['average_fps']]
    cadence = data.get('physical_fps') or min(rates, default=None)
    if max(durations) - min(durations) > max(.1, 1 / cadence if cadence else 0):
        raise ValueError('Camera durations differ. Prepare synchronized copies covering the same interval.')
    warnings = []
    for camera in cameras:
        metadata = camera['probe']
        if metadata.get('pts_variable') or metadata.get('pts_monotonic') is False:
            warnings.append(f'{camera["id"]}: irregular decoded timestamps. Verify capture cadence and normalization before approval.')
        if metadata.get('pts_complete') is False:
            warnings.append(f'{camera["id"]}: incomplete encoded timestamps. Verify physical FPS against the recording source.')
        if metadata['average_fps'] != probe_number(metadata.get('r_frame_rate')):
            warnings.append(f'{camera["id"]}: average and nominal FPS differ or are unknown. Inspect timestamps.')
    starts = [probe_number(c['probe'].get('start_time'), positive=False) for c in cameras]
    if len({value for value in starts if value is not None}) > 1:
        warnings.append('Encoded start times differ. Equal durations do not establish cross-camera synchronization.')
    data['timing_warnings'] = warnings
    for key in ('ground_truth',):
        if data.get(key):
            path = resolve(data[key], source.parent)
            data[key] = str(path)
            files[str(path)] = sha256(path)
    artifacts = []
    for value in data.get('artifacts', []):
        path = resolve(value, source.parent)
        files[str(path)] = sha256(path)
        artifacts.append(str(path))
    data['artifacts'] = artifacts
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / 'dataset.json', data)
    files[str(root / 'dataset.json')] = sha256(root / 'dataset.json')
    write_json(root / 'inputs.json', files)
    seconds = proposed_seconds(cameras)
    hours = seconds / 3600
    write_json(root / 'state.json', {'status': 'awaiting_budget', 'proposed_seconds': seconds,
                                   'proposed_hours': hours,
                                   'created_at': time.time(), 'deadline': None})
    print(json.dumps({'campaign': str(root), 'cameras': len(cameras),
                      'camera_minutes': sum(c['duration_seconds'] for c in cameras) / 60,
                      'proposed_wall_seconds': seconds, 'proposed_wall_hours': hours,
                      'timing_warnings': warnings,
                      'next': 'Ask the user to accept this budget or provide custom hours.'}, indent=2))


def approve(args):
    root = Path(args.campaign).resolve()
    with lock(root / '.lock'):
        state = read_json(root / 'state.json')
        if state.get('deadline') is not None:
            raise ValueError('Budget already approved. A resume must retain its deadline.')
        approved_at = time.time()
        if getattr(args, 'deadline', None):
            deadline = datetime.fromisoformat(args.deadline)
            if deadline.tzinfo is None:
                raise ValueError('Deadline must include its timezone')
            seconds = number(deadline.timestamp() - approved_at, 'remaining budget seconds')
            hours = seconds / 3600
        else:
            hours = number(args.hours, 'approved hours')
            seconds = hours * 3600
        if not args.note.strip():
            raise ValueError('Record the actual user budget decision')
        state.update(status='ready', approved_hours=hours, approved_seconds=seconds,
                     approved_at=approved_at, deadline=approved_at + seconds, approval_note=args.note)
        write_json(root / 'state.json', state)
        print(f'Approved {hours:g} wall-clock hours. Deadline: {state["deadline"]:.0f}')


def yaml_load(path):
    import yaml
    config = yaml.safe_load('\n'.join(line for line in Path(path).read_text().splitlines()
                                    if not line.startswith('%YAML:')))
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise TypeError(f'Tracker config must be a YAML mapping: {path}')
    return config


def merge(base, update):
    for key, value in update.items():
        if isinstance(value, dict):
            base[key] = merge(base.get(key, {}), value)
        else:
            base[key] = value
    return base


def yaml_write(path, config):
    import yaml
    class ConfigDumper(yaml.SafeDumper):
        def increase_indent(self, flow=False, indentless=False):
            return super().increase_indent(flow, False)
    with Path(path).open('x') as handle:
        yaml.dump(config, handle, Dumper=ConfigDumper, sort_keys=False)


def candidate(args):
    config = yaml_load(args.base)
    for path in args.overlay:
        config = merge(config, yaml_load(path))
    if args.basic:
        if not all(isinstance(config.get(key), dict) for key in ('ObjectModelProjection', 'StateEstimator', 'Communicator')):
            raise ValueError('Prepare the base MV3DT geometry/state/communication sections before --basic')
        if config['StateEstimator'].get('stateEstimatorType') not in (3, 4):
            raise ValueError('Basic MV3DT requires geometric stateEstimatorType 3 or 4')
        config = merge(config, yaml_load(SKILL / 'assets/basic-overlay.yml'))
        config['TrajectoryManagement'].pop('enableArchiveReAssoc', None)
        config['ObjectModelProjection'].pop('outputVisibility', None)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    yaml_write(target, config)
    print(target)


def argv(template, substitutions):
    if not isinstance(template, list) or not template or not all(isinstance(x, str) for x in template):
        raise ValueError('Commands must be nonempty JSON arrays of string arguments')
    result = []
    for item in template:
        for key, value in substitutions.items():
            item = item.replace('{' + key + '}', value)
        result.append(item)
    return result


def verify_config(expected, actual, rewrites, prefix=''):
    if not isinstance(actual, dict):
        raise TypeError(f'Preparation malformed section {prefix}')
    added = set(actual) - set(expected)
    if added:
        raise ValueError(f'Preparation added unrequested fields under {prefix}: {sorted(added)}')
    for key, value in expected.items():
        field = f'{prefix}.{key}' if prefix else str(key)
        if key not in actual:
            raise ValueError(f'Preparation removed {field}')
        if isinstance(value, dict):
            verify_config(value, actual[key], rewrites, field)
        elif actual[key] != value:
            path_key = str(key).lower().endswith(('file', 'filepath', 'filename', 'path', 'dir', 'directory', 'fileprefix'))
            replacement = rewrites.get(field)
            path_value = isinstance(replacement, str) or (isinstance(replacement, list)
                         and all(isinstance(item, str) for item in replacement))
            if not path_key or not path_value or actual[key] != replacement:
                raise ValueError(f'Preparation changed {field}. Only exact path-leaf rewrites are allowed.')


def check_pipeline(path, tracker, mode=None):
    pipeline = configparser.ConfigParser(interpolation=None)
    with path.open() as handle:
        pipeline.read_file(handle)
    if mode is not None:
        if mode not in ('live', 'replay'):
            raise ValueError('detector_mode must be live or replay')
        enabled = pipeline.getint('primary-gie', 'enable', fallback=0)
        replay = pipeline.getint('replay', 'enable', fallback=0)
        if (mode == 'live' and (enabled != 1 or replay != 0)) or (mode == 'replay' and (replay != 1 or enabled != 0)):
            raise ValueError(f'Prepared detector enablement does not match {mode} mode')
    width = pipeline.getint('tracker', 'tracker-width', fallback=0)
    height = pipeline.getint('tracker', 'tracker-height', fallback=0)
    if width <= 0 or height <= 0:
        raise ValueError('Declare positive tracker-width and tracker-height in the prepared pipeline')
    legacy = tracker.get('VisualTracker', {}).get('visualTrackerType') == 1
    for section, selector in (('PoseEstimator', 'poseEstimatorType'), ('ReID', 'reidType')):
        config = tracker.get(section, {})
        if not config.get(selector, 0):
            continue
        models = {key: value for key, value in config.items() if str(key).startswith('model')
                  and str(key)[5:].isdigit() and isinstance(value, dict)}
        selected = list(models.values()) if models else [config]
        if models and config.get('enableModelId', 0) >= 0:
            key = f'model{config.get("enableModelId", 0)}'
            if key not in models:
                raise ValueError(f'Enabled ReID model is missing: {key}')
            selected = [models[key]]
        for value in selected:
            value = {**config, **value}
            if 'useVPICropScaler' not in value:
                raise ValueError(f'Declare {section} useVPICropScaler explicitly for the enabled model')
            if value['useVPICropScaler'] == 0:
                legacy = True
    if legacy and (width % 32 or height % 32):
        raise ValueError('Legacy visual/crop backend requires tracker dimensions divisible by 32. Check VPI cropping before resizing geometry.')


def group_alive(group):
    for path in Path('/proc').iterdir():
        if not path.name.isdecimal():
            continue
        try:
            fields = (path / 'stat').read_text().rpartition(')')[2].split()
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if fields[0] != 'Z' and fields[2] == str(group) and fields[3] == str(group):
            return True
    return False


def terminate(process):
    if group_alive(process.pid):
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        pass
    if group_alive(process.pid):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait(timeout=5)
    until = time.monotonic() + 2
    while group_alive(process.pid) and time.monotonic() < until:
        time.sleep(.05)
    if group_alive(process.pid):
        raise RuntimeError(f'Owned process group did not terminate: {process.pid}')


def process_stamp(pid):
    try:
        fields = Path(f'/proc/{int(pid)}/stat').read_text().rpartition(')')[2].split()
        return fields[19] if fields[0] != 'Z' else None
    except FileNotFoundError:
        return None


def check_worker(path):
    if not path.exists():
        return
    worker = read_json(path)
    stamp = process_stamp(worker['pid']) if worker.get('pid') else None
    if stamp is not None and (worker.get('process_start_ticks') is None or stamp == worker['process_start_ticks']):
        raise RuntimeError(f'An existing worker is still running: {path}. Inspect it before resuming.')
    if stamp is None and worker.get('pid') and group_alive(worker['pid']):
        raise RuntimeError(f'An orphan worker process group is still active: {path}')
    if worker.get('runtime') and worker.get('container'):
        result = subprocess.run([worker['runtime'], 'ps', '--filter', f'name={worker["container"]}',
                                 '--format', '{{.Names}}'], capture_output=True, text=True, check=True, timeout=15)
        if worker['container'] in result.stdout.splitlines():
            raise RuntimeError(f'An owned container is still running: {worker["container"]}')


def execute(command, cwd, log, deadline, status_path, lease_path=None, owner=None):
    if time.time() >= deadline:
        raise TimeoutError('Budget expired before starting a process')
    with log.open('x') as stream:
        process = subprocess.Popen(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        worker = {**(owner or {}), 'pid': process.pid, 'process_start_ticks': process_stamp(process.pid),
                  'controller_pid': os.getpid(), 'deadline': deadline, 'log': str(log), 'active': True}
        paths = [status_path] + ([lease_path] if lease_path is not None else [])
        last = 0
        try:
            for path in paths:
                write_json(path, worker)
            while process.poll() is None:
                now = time.time()
                if now >= deadline:
                    raise TimeoutError('Approved campaign budget or trial timeout exhausted')
                if now - last >= 30:
                    worker['heartbeat'] = now
                    for path in paths:
                        write_json(path, worker)
                    last = now
                time.sleep(.2)
            if process.returncode:
                raise RuntimeError(f'Command exited {process.returncode}. See {log}')
        finally:
            terminate(process)
            worker.update(active=False, finished_at=time.time(), exit_code=process.returncode)
            for path in paths:
                write_json(path, worker)


def run(args):
    root = Path(args.campaign).resolve()
    if not (root / 'state.json').is_file():
        raise ValueError('Inspect the inputs to create a campaign first')
    gpu_lock = Path(args.gpu_lock).resolve()
    lease_path = gpu_lock.with_name(gpu_lock.name + '.mv3dt-owner.json')
    with lock(root / '.lock'), lock(gpu_lock):
        state = read_json(root / 'state.json')
        if state.get('deadline') is None:
            raise ValueError('Ask for the budget and record approval before launching')
        if time.time() >= state['deadline']:
            raise TimeoutError('Campaign budget expired. Do not restart automatically.')
        check_worker(root / 'worker.json')
        check_worker(lease_path)
        inputs = read_json(root / 'inputs.json')
        verify_hashes(inputs)
        job_path = Path(args.job).resolve()
        job = read_json(job_path)
        if not isinstance(job.get('config_rewrites', {}), dict):
            raise TypeError('config_rewrites must map qualified path-leaf keys to final values')
        if not job.get('run'):
            raise ValueError('Job is missing its run command')
        commands = [job[k] for k in ('prepare', 'run', 'collect') if job.get(k)]
        for command in commands:
            argv(command, {})
        if not any('{config}' in x for cmd in commands for x in cmd):
            raise ValueError('Job must consume {config}, not a fixed tracker config')
        timeout = number(job.get('timeout_seconds', 86400), 'trial timeout')
        effective_name = Path(job.get('effective_config', 'config_tracker.yml'))
        if effective_name.is_absolute() or '..' in effective_name.parts:
            raise ValueError('Effective config must be relative to the isolated run directory')
        pipeline_name = Path(job['pipeline_config']) if job.get('pipeline_config') else None
        if pipeline_name is not None and (pipeline_name.is_absolute() or '..' in pipeline_name.parts):
            raise ValueError('Pipeline config must be relative to the isolated run directory')
        expected = yaml_load(args.config)
        folder = root / 'runs' / label(args.name)
        folder.mkdir(parents=True, exist_ok=False)
        config = folder / 'candidate.yml'
        source_config = folder / 'source-config.yml'
        shutil.copyfile(args.config, source_config)
        yaml_write(config, expected)
        name = 'mv3dt-tune-' + uuid.uuid4().hex[:12]
        substitutions = {'config': str(config), 'run_dir': str(folder),
                         'manifest': str(root / 'dataset.json'), 'container': name}
        cwd = resolve(job.get('cwd', '.'), job_path.parent)
        deadline = min(state['deadline'], time.time() + timeout)
        record = {'name': args.name, 'hypothesis': args.hypothesis, 'started_at': time.time(),
                  'status': 'running', 'candidate_sha256': sha256(config),
                  'source_config_sha256': sha256(source_config),
                  'job_sha256': sha256(job_path), 'container': name,
                  'scope': job.get('scope', 'unspecified'), 'job': job}
        write_json(folder / 'run.json', record)
        state.update(status='running', last_run=args.name)
        write_json(root / 'state.json', state)
        runtime = None
        try:
            for phase in ('prepare', 'run', 'collect'):
                if not job.get(phase):
                    continue
                if phase == 'run' and pipeline_name is not None:
                    pipeline = (folder / pipeline_name).resolve()
                    if not pipeline.is_relative_to(folder):
                        raise ValueError('Pipeline config resolved outside the isolated run directory')
                    effective_tracker = yaml_load(folder / effective_name) if job.get('prepare') else expected
                    check_pipeline(pipeline, effective_tracker, job.get('detector_mode'))
                    record['pipeline_config_sha256'] = sha256(pipeline)
                command = argv(job[phase], substitutions)
                if Path(command[0]).name in ('docker', 'podman') and 'run' in command:
                    at = command.index('--name') if '--name' in command else -1
                    if at < 0 or at + 1 >= len(command) or command[at + 1] != name or '--rm' not in command:
                        raise ValueError('Container runs must use --rm --name {container}')
                    if any(value.startswith('--detach') or (value.startswith('-') and not value.startswith('--')
                           and 'd' in value[1:]) for value in command):
                        raise ValueError('Run the trial in the foreground')
                    runtime = command[0]
                execute(command, cwd, folder / f'{phase}.log', deadline, root / 'worker.json', lease_path,
                        {'campaign': str(root), 'run': args.name, 'container': name, 'runtime': runtime})
                if sha256(config) != record['candidate_sha256']:
                    raise ValueError('Submitted candidate config changed during execution')
                if phase == 'prepare':
                    effective = (folder / effective_name).resolve()
                    if not effective.is_relative_to(folder):
                        raise ValueError('Effective config resolved outside the isolated run directory')
                    if not effective.is_file():
                        raise ValueError('Preparation did not save the effective tracker config')
                    actual = yaml_load(effective)
                    trajectory = actual.get('TrajectoryManagement', {})
                    if (isinstance(trajectory, dict) and 'enableArchiveReAssoc' in trajectory
                            and trajectory.get('enable3DReAssoc', trajectory['enableArchiveReAssoc']) != trajectory['enableArchiveReAssoc']):
                        raise ValueError('Conflicting canonical and legacy re-association controls')
                    verify_config(expected, actual, job.get('config_rewrites', {}))
                    record['effective_config_sha256'] = sha256(effective)
                if 'effective_config_sha256' in record:
                    verify_hashes({folder / effective_name: record['effective_config_sha256']})
                if 'pipeline_config_sha256' in record:
                    verify_hashes({folder / pipeline_name: record['pipeline_config_sha256']})
            expected = job.get('success_text', 'App run successful')
            if expected and expected not in (folder / 'run.log').read_text(errors='replace'):
                raise ValueError(f'Run log does not contain {expected!r}')
            verify_hashes(inputs)
            record['status'] = 'completed'
        except BaseException as error:
            record.update(status='failed', error=str(error) or 'interrupted')
            raise
        finally:
            if runtime and record['status'] != 'completed':
                try:
                    subprocess.run([runtime, 'stop', '--time', '10', name], capture_output=True, timeout=30, check=False)
                except (OSError, subprocess.TimeoutExpired) as error:
                    record['cleanup_error'] = str(error)
            record['finished_at'] = time.time()
            write_json(folder / 'run.json', record)
            state.update(status='ready' if time.time() < state['deadline'] else 'budget_expired')
            write_json(root / 'state.json', state)
    print(folder)


def dominates(a, b):
    keys = ('HOTA', 'MOTA', 'IDF1')
    return all(a[k] >= b[k] for k in keys) and any(a[k] > b[k] for k in keys)


def configuration_summary(rows):
    groups = {}
    for row in rows:
        candidate = row.get('candidate_sha256')
        effective = row.get('effective_config_sha256')
        key = (candidate, effective) if candidate else ('unverified', row['name'])
        groups.setdefault(key, []).append(row)
    summaries = []
    for (candidate, effective), runs in groups.items():
        values = [row['evaluation']['metrics'] for row in runs]
        means = {key: statistics.mean(row[key] for row in values) for key in ('HOTA', 'MOTA', 'IDF1')}
        ranges = {key: max(row[key] for row in values) - min(row[key] for row in values) for key in means}
        summaries.append({'candidate_sha256': candidate, 'effective_config_sha256': effective,
                          'representative_run': runs[0]['name'], 'runs': [row['name'] for row in runs],
                          'count': len(runs), 'confirmed': candidate != 'unverified' and len(runs) >= 3,
                          'mean': means, 'range': ranges})
    return sorted(summaries, key=lambda row: row['mean']['HOTA'], reverse=True)


def summarize(args):
    root = Path(args.campaign)
    dataset = read_json(root / 'dataset.json')
    rows = []
    for path in sorted((root / 'runs').glob('*/run.json')):
        row = read_json(path)
        score = path.parent / 'metrics.json'
        if score.is_file() and row['status'] == 'completed':
            evaluation = read_json(score)
            if not dataset.get('ground_truth'):
                raise ValueError('Cannot accuracy-rank a no-GT campaign')
            prediction = Path(evaluation.get('prediction_file', '')).resolve()
            if not prediction.is_relative_to(path.parent.resolve()):
                raise ValueError(f'Prediction artifact must belong to its run: {path.parent}')
            if not prediction.is_file() or sha256(prediction) != evaluation.get('prediction_sha256'):
                raise ValueError(f'Prediction artifact hash changed or is missing: {prediction}')
            protocol = evaluation.get('protocol', {})
            if not protocol.get('output_semantics') or protocol['output_semantics'] == 'unspecified':
                raise ValueError(f'Declare evaluation.output_semantics before ranking: {score}')
            signature = hashlib.sha256(json.dumps(protocol, sort_keys=True, allow_nan=False).encode()).hexdigest()
            if signature != evaluation.get('protocol_sha256'):
                raise ValueError(f'Invalid evaluation protocol hash: {score}')
            truth = resolve(dataset['ground_truth'], root)
            if protocol.get('evaluation') != dataset.get('evaluation') or protocol.get('gt_sha256') != sha256(truth):
                raise ValueError(f'Evaluation differs from the frozen dataset protocol: {score}')
            if evaluation['metrics'].get('GT_Dets', 0) <= 0:
                raise ValueError(f'No evaluated GT observations: {score}')
            if not all(math.isfinite(evaluation['metrics'][key]) for key in ('HOTA', 'MOTA', 'IDF1')):
                raise ValueError(f'Nonfinite accuracy metrics: {score}')
            row['evaluation'] = evaluation
        rows.append(row)
    scored = [r for r in rows if 'metrics' in r.get('evaluation', {})]
    protocols = {r['evaluation'].get('protocol_sha256') for r in scored}
    if None in protocols or len(protocols) > 1:
        raise ValueError('Cannot rank different or unspecified evaluation protocols')
    scopes = {(r.get('scope'), r.get('job_sha256')) for r in scored}
    if len(scopes) > 1:
        raise ValueError('Cannot rank different launch/output scopes. Report separate comparison groups.')
    scored.sort(key=lambda r: r['evaluation']['metrics']['HOTA'], reverse=True)
    frontier = [r['name'] for r in scored if not any(dominates(s['evaluation']['metrics'],
                r['evaluation']['metrics']) for s in scored)]
    groups = configuration_summary(scored)
    confirmed = [group for group in groups if group['confirmed']]
    result = {'runs': rows, 'hota_order': [r['name'] for r in scored], 'pareto_runs': frontier,
              'configurations': groups, 'confirmed_hota_order': [group['representative_run'] for group in confirmed],
              'confirmed_pareto_runs': [group['representative_run'] for group in confirmed
                  if not any(dominates(other['mean'], group['mean']) for other in confirmed)],
              'note': 'Promote from repeated config means and ranges, not the best individual run. Apply constraints and visual review separately.'}
    write_json(root / 'summary.json', result)
    print(json.dumps({'runs': len(rows), 'confirmed_hota_order': result['confirmed_hota_order'],
                      'confirmed_pareto_runs': result['confirmed_pareto_runs'],
                      'provisional_configs': sum(not group['confirmed'] for group in groups)}, indent=2))


def export(args):
    config = yaml_load(args.config)
    temporary = {'debugVerboseLevel', 'debugVerboseLevel_gpumemory', 'debugVerboseLevel_cufft',
                 'showGpuTiming', 'saveFrameViz', 'saveLog2File', 'dumpTrajectoryIntoFile',
                 'dumpDetectorBboxes', 'dumpTrajectoryTentative', 'dumpTrajectoryQuasiActive',
                 'dumpTrajectoryInactive'}
    def clean(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in temporary:
                    node[key] = 0
                else:
                    clean(value)
        elif isinstance(node, list):
            for value in node:
                clean(value)
    clean(config)
    target = Path(args.output)
    yaml_write(target, config)
    print(f'{target}: diagnostics disabled. Rerun this exact config before delivery.')


def main():
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('inspect')
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--probe-timeout', type=float, default=120, help='Per-video FFprobe timeout in seconds')
    p.set_defaults(func=inspect)
    p = commands.add_parser('approve')
    p.add_argument('--campaign', required=True)
    budget = p.add_mutually_exclusive_group(required=True)
    budget.add_argument('--hours', type=float)
    budget.add_argument('--deadline', help='ISO-8601 absolute deadline with timezone, inherited unchanged')
    p.add_argument('--note', required=True, help='Record the user decision, never invent acceptance')
    p.set_defaults(func=approve)
    p = commands.add_parser('candidate')
    p.add_argument('--base', required=True)
    p.add_argument('--overlay', action='append', default=[])
    p.add_argument('--basic', action='store_true')
    p.add_argument('--output', required=True)
    p.set_defaults(func=candidate)
    p = commands.add_parser('run')
    for name in ('campaign', 'job', 'config', 'name', 'hypothesis', 'gpu-lock'):
        p.add_argument('--' + name, required=True)
    p.set_defaults(func=run)
    p = commands.add_parser('summarize')
    p.add_argument('--campaign', required=True)
    p.set_defaults(func=summarize)
    p = commands.add_parser('export')
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    p.set_defaults(func=export)
    args = parser.parse_args()
    try:
        args.func(args)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, TimeoutError, configparser.Error, subprocess.SubprocessError) as error:
        parser.exit(1, f'ERROR: {error}\n')


if __name__ == '__main__':
    main()
