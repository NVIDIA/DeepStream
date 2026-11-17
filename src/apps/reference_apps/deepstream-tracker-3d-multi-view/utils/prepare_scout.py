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


"""Build the packaged SCOUT sample from the official SCOUT dataset.

SCOUT is the multi-view people tracking dataset of EPFL CVLab ("Unified People Tracking with Graph
Neural Networks", arXiv 2507.08494). It is released for academic use only and must not be
redistributed. Download it yourself from https://scout.epfl.ch/download. This script never downloads
anything. It reads the official files in place and writes the converted dataset below --output.

Reads from the official release (the full release is about 350 GB, only this subset is needed):
  calibrations/cam_0 to cam_7 (json or yaml) and timestamps/timestamps_sequence_1.json
  annotations/individual_format/cam_K and images/sequence_1/cam_K for the frames of the split
      (the test split is frames 5980 to 11959, about 3.2 GB of images per camera)
  meshes/mesh.ply and the three scene_dense_textured_cleanup PNG textures (map.png only)

Writes below --output: videos/cam_K.mp4 (undistorted, H.264, yuv420p, 10 fps), camInfo/cam_K.yml,
ground_truth.txt, transforms.yml, map.png and manifest.json. Other folders next to them, for
example detections, are never touched. Generated camera files of cameras outside --cameras are
deleted on every run.

The transformed files in assets/scout_transformed (transformed_calibration.json and
transformed_positions.json) hold the camera calibration and the person positions in the world frame
of the sample. They are recorded (shipped) numbers and the script applies them to the official files.
There is no interactive step and torch is never imported.

Options: --split (test, train or all), --region, --cameras, --jobs, --no-map (skip the mesh),
--check-only (validate the source layout and exit) and --force (rebuild and ignore the manifest).
Opt in with --transform-calib to recompute the transformed files instead: it reruns the bundle
adjustment of utils/transform_scout.py with --correspondences, writes new transformed files to
--transformed-out and uses them instead of the shipped ones. It needs torch and kornia and the
annotations of all frames. The manifest then records the calibration as recomputed.

Default protocol: sequence_1, test split, cameras 0 to 7, recorded calibration, ground truth region
0 26 0 140 meters in the transformed frame, maximum camera distance 50 meters, gt_alignment exact.

Ground truth frame ids are aligned with the video frames, the first video frame is id 1. Every
official annotation row takes its person position from the transformed files and its box moves with
the projected foot point from the official camera to the transformed camera (the size is kept, boxes
are not clipped). A row is kept when its box is valid (width and height above one pixel) and
intersects the image, the position is inside the region and the camera is closer than the maximum
distance. annotated_frames in the manifest counts the distinct ids that carry a row (5971 for the
default test split).

Reuse: manifest.json records the protocol, the source subset, a source fingerprint, the sha256 of the
transformed files and the hashes of the generated files. A second run into the same directory refuses
to continue when any of them differs (unless --force) and reuses a video only when it matches its
recorded hash. The fingerprint is a sha256 over the sorted lines relative path, size and mtime in
nanoseconds of every selected input file. It is not a content hash, so touching or copying the
inputs changes it and --force accepts that. The manifest is written last, so an interrupted run can
be repeated with the same options.
"""

import os
import re
import sys
import json
import gzip
import time
import yaml
import hashlib
import argparse
import subprocess
import contextlib

import cv2
import numpy as np

from pathlib import Path
from collections import deque
from types import SimpleNamespace
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor

APP = Path(__file__).resolve().parents[1]
RECORDED_DIR = APP / 'assets' / 'scout_transformed'
RECORDED_CALIBRATION = 'transformed_calibration.json'
RECORDED_POSITIONS = 'transformed_positions.json'
CORRESPONDENCES = RECORDED_DIR / 'ground_plane_correspondences.json'
EXIT_SOURCE, EXIT_MISMATCH, EXIT_OUTPUT, EXIT_MISSING_PACKAGES = 2, 3, 4, 5
SEQUENCE = 'sequence_1'
CAMERAS = tuple(range(8))
FRAME_SIZE = (1920, 1080)
FPS = 10
TRAIN_FRACTION = 0.5
REGION = (0.0, 26.0, 0.0, 140.0)
MAX_DIST = 50.0
SHIFT_HEIGHT = 1.34
PERSON_HEIGHT = 1.7
PERSON_RADIUS = 0.3
CRF = 20
PRESET = 'medium'
ENCODING = {'codec': 'libx264', 'crf': CRF, 'preset': PRESET}
THREADS = 2
WINDOW_FACTOR = 4
MAP_SIZE = (300, 1560)
MAP_TRANSFORM = ((-10.0, 0.0, 220.0), (0.0, 10.0, 80.0), (0.0, 0.0, 1.0))
MAP_SUPERSAMPLE = 4
MESH_RANGES = {'x': (0.0, 50.0), 'y': (30.0, 190.0), 'z_max': (-2.5, 0.5)}
TEXTURES = tuple(f'scene_dense_textured_cleanup{index}.png' for index in range(3))
DOWNLOAD_URL = 'https://scout.epfl.ch/download'
PAPER = 'arXiv 2507.08494'
FINGERPRINT_NOTE = 'metadata fingerprint (relative path, size, mtime_ns of every selected input), not a content hash'
KEY_SCALE = 100000
GT_ALIGNMENT = 'exact'


class SourceError(Exception):
    """The official SCOUT files or the transformed files are missing or inconsistent."""


class ExitError(Exception):
    """A run that stops with a message and an exit code."""

    def __init__(self, message, code):
        super().__init__(str(message))
        self.code = code


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def load_calibration(path):
    """Read one official camera calibration (json or yaml) with keys K, R, T and dist."""
    with Path(path).open() as stream:
        values = yaml.safe_load(stream)
    missing = [key for key in ('K', 'R', 'T', 'dist') if key not in values]
    if missing:
        raise SourceError(f'{path}: missing calibration keys {missing}')
    calibration = {'K': np.asarray(values['K'], dtype=float).reshape(3, 3),
                   'R': np.asarray(values['R'], dtype=float).reshape(3, 3),
                   'T': np.asarray(values['T'], dtype=float).reshape(3, 1),
                   'dist': np.asarray(values['dist'], dtype=float).reshape(-1)}
    if calibration['dist'].size not in (4, 5, 8, 12, 14):
        raise SourceError(f'{path}: unsupported distortion length {calibration["dist"].size}')
    return calibration


def find_calibration_file(source, camera):
    for suffix in ('json', 'yaml', 'yml'):
        path = Path(source) / 'calibrations' / f'cam_{camera}.{suffix}'
        if path.is_file():
            return path
    return None


def undistorted_intrinsics(calibration):
    """Pinhole intrinsics of the undistorted image, keeping only valid pixels (alpha 0)."""
    matrix, _ = cv2.getOptimalNewCameraMatrix(calibration['K'], calibration['dist'], FRAME_SIZE, alpha=0, newImgSize=FRAME_SIZE)
    return matrix


def recorded_files(directory=None):
    """Paths of the two transformed files (the positions may be stored gzip compressed)."""
    directory = Path(directory or RECORDED_DIR)
    positions = directory / RECORDED_POSITIONS
    if not positions.is_file() and positions.with_name(RECORDED_POSITIONS + '.gz').is_file():
        positions = positions.with_name(RECORDED_POSITIONS + '.gz')
    return {RECORDED_CALIBRATION: directory / RECORDED_CALIBRATION, RECORDED_POSITIONS: positions}


def read_json(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt') as stream:
        return json.load(stream)


class Recorded:
    """The transformed files: cameras, person positions and the map frame."""

    def __init__(self, cameras, keys, positions, map_frame, hashes):
        self.cameras, self.map_frame, self.hashes = cameras, map_frame, hashes
        order = np.argsort(keys, kind='stable')
        self.keys, self.positions = keys[order], positions[order]

    def lookup(self, persons, frames):
        """Transformed positions (n, 3) in meters of (person, frame) pairs, an error when one was not recorded."""
        wanted = np.asarray(persons, dtype=np.int64)*KEY_SCALE + np.asarray(frames, dtype=np.int64)
        index = np.searchsorted(self.keys, wanted).clip(0, len(self.keys) - 1)
        absent = self.keys[index] != wanted
        if absent.any():
            first = int(wanted[absent][0])
            raise SourceError(f'{int(absent.sum())} official annotation rows have no position in the transformed files, for example '
                              f'person {first//KEY_SCALE} frame {first % KEY_SCALE}. The official files do not match the '
                              f'transformed files of sequence_1, download SCOUT again from {DOWNLOAD_URL}.')
        return self.positions[index]


def decode_positions(values):
    """Columnar positions to (keys, positions in meters). See assets/scout_transformed/README.md."""
    count = int(values['count'])
    counts = np.asarray(values['per_frame'], dtype=np.int64)
    frames = np.repeat(np.asarray(values['frames'], dtype=np.int64), counts)
    deltas = np.asarray(values['person'], dtype=np.int64)
    x, y = np.asarray(values['x_cm'], dtype=np.int64), np.asarray(values['y_cm'], dtype=np.int64)
    if not (len(frames) == len(deltas) == len(x) == len(y) == count) or counts.sum() != count:
        raise SourceError('The transformed positions are inconsistent, restore assets/scout_transformed from the repository')
    starts = np.cumsum(counts) - counts
    running = np.cumsum(deltas)
    persons = running - np.repeat(running[starts] - deltas[starts], counts)
    z = np.zeros(count, dtype=np.int64)
    z[np.asarray(values['z_index'], dtype=np.int64)] = np.asarray(values['z_cm'], dtype=np.int64)
    return persons*KEY_SCALE + frames, np.column_stack([x, y, z])/100.0


def load_recorded(directory=None):
    """Load and validate the transformed JSON files."""
    files = recorded_files(directory)
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise SourceError(f'Missing transformed files {", ".join(missing)}. Restore assets/scout_transformed from the repository.')
    calibration = read_json(files[RECORDED_CALIBRATION])
    if calibration.get('format') != 'scout-transformed-calibration-1':
        raise SourceError(f'{files[RECORDED_CALIBRATION]}: unexpected format {calibration.get("format")}')
    cameras = {}
    for camera in CAMERAS:
        entry = calibration['cameras'].get(str(camera))
        if entry is None:
            raise SourceError(f'{files[RECORDED_CALIBRATION]}: camera {camera} is not recorded')
        cameras[camera] = {'K': np.asarray(entry['K'], dtype=float).reshape(3, 3), 'R': np.asarray(entry['R'], dtype=float).reshape(3, 3),
                           'T': np.asarray(entry['T'], dtype=float).reshape(3, 1)}
    positions = read_json(files[RECORDED_POSITIONS])
    if positions.get('format') != 'scout-transformed-positions-1':
        raise SourceError(f'{files[RECORDED_POSITIONS]}: unexpected format {positions.get("format")}')
    keys, meters = decode_positions(positions)
    frame = calibration['map_frame']
    map_frame = WorldFrame(frame['scale'], frame['yaw_deg'], (frame['translation'][0], frame['translation'][1], 0.0))
    return Recorded(cameras, keys, meters, map_frame, {name: sha256_file(path) for name, path in files.items()})


class WorldFrame:
    """Similarity transform of world points, used to draw the official mesh in the transformed frame."""

    def __init__(self, scale, yaw_deg, translation):
        yaw = np.radians(yaw_deg)
        self.scale = float(scale)
        self.rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
        self.translation = np.asarray(translation, dtype=float)

    def apply(self, points):
        return self.scale*np.asarray(points, dtype=float) @ self.rotation.T + self.translation


def project_points(points, matrix, rotation, translation, dist=None):
    """Project world points with OpenCV, optionally with lens distortion."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 1, 3)
    if len(points) == 0:
        return np.zeros((0, 2))
    rvec, _ = cv2.Rodrigues(np.asarray(rotation, dtype=np.float64))
    dist = np.zeros(5) if dist is None else np.asarray(dist, dtype=np.float64)
    pixels, _ = cv2.projectPoints(points, rvec, np.asarray(translation, dtype=np.float64).reshape(3, 1),
                                  np.asarray(matrix, dtype=np.float64), dist)
    return pixels.reshape(-1, 2)


def projection_matrix(matrix, rotation, translation):
    """The 3x4 world to pixel matrix K [R | T], normalized so its last element is one."""
    full = np.asarray(matrix, dtype=float) @ np.hstack([rotation, np.asarray(translation, dtype=float).reshape(3, 1)])
    if abs(full[2, 3]) < 1e-12:
        raise ValueError('Projection matrix cannot be normalized, camera center at the world origin plane')
    return full/full[2, 3]


def caminfo_text(projection):
    lines = ['projectionMatrix_3x4_w2p:'] + [f'- {value:.6f}' for value in np.asarray(projection).reshape(-1)]
    lines += ['', 'modelInfo:', '  - classID: 1 # person', f'    height: {PERSON_HEIGHT}', f'    radius: {PERSON_RADIUS}']
    return '\n'.join(lines) + '\n'


def frame_ids_in_directory(directory, suffix):
    pattern = re.compile(rf'image_(\d+){re.escape(suffix)}$')
    found = set()
    if Path(directory).is_dir():
        for name in os.listdir(directory):
            match = pattern.match(name)
            if match:
                found.add(int(match.group(1)))
    return found


def total_frames(source, sequence=SEQUENCE):
    path = Path(source) / 'timestamps' / f'timestamps_{sequence}.json'
    if not path.is_file():
        raise SourceError(f'Missing {path}. Download the SCOUT timestamps from {DOWNLOAD_URL}.')
    with path.open() as stream:
        return len(json.load(stream))


def split_frames(count, split):
    """Frame ids of a split. The test split is the second half of the sequence."""
    start = int(count*TRAIN_FRACTION)
    ranges = {'train': range(0, start), 'test': range(start, count), 'all': range(0, count)}
    return list(ranges[split])


def read_annotation_file(path):
    rows = []
    with Path(path).open() as stream:
        for line in stream:
            fields = line.strip().split(',')
            if len(fields) < 9:
                continue
            rows.append([int(fields[0]), *map(float, fields[1:5]), *map(float, fields[6:9])])
    return rows


def load_camera_annotations(source, camera, frames):
    """Rows [frame, person, x, y, w, h, X, Y, Z] of one camera for the given frames."""
    folder = Path(source) / 'annotations' / 'individual_format' / f'cam_{camera}'
    rows = []
    for frame in frames:
        path = folder / f'image_{frame}.txt'
        if not path.is_file():
            raise SourceError(f'Missing annotation file {path}. Download the SCOUT individual_format annotations.')
        rows.extend([frame, row[0], *row[1:]] for row in read_annotation_file(path))
    return np.asarray(rows, dtype=float).reshape(-1, 9)


def camera_ground_truth(annotations, calibration, matrix_new, rotation, translation, positions, region):
    """Ground truth rows of one camera as [frame, person, x, y, w, h, X, Y].

    The box keeps its size and its top left corner moves by the displacement of the projected foot
    point between the official distorted camera and the transformed camera. Boxes are not clipped.
    A row is kept when its box is valid (width and height above one pixel), intersects the image,
    the position is inside the region and the camera is closer than MAX_DIST.
    """
    foot_original = project_points(annotations[:, 6:9], calibration['K'], calibration['R'], calibration['T'],
                                   calibration['dist'])
    foot_new = project_points(positions, matrix_new, rotation, translation)
    boxes = annotations[:, 2:6].copy()
    boxes[:, :2] += foot_new - foot_original
    center = (-np.asarray(rotation).T @ np.asarray(translation).reshape(3, 1)).ravel()
    distance = np.linalg.norm(positions + np.array([0.0, 0.0, SHIFT_HEIGHT/2]) - center, axis=1)
    x1, y1, w, h = boxes.T
    keep = ((x1 + w > 1) & (x1 < FRAME_SIZE[0] - 2) & (w > 1) & (y1 + h > 1) & (y1 < FRAME_SIZE[1] - 2) & (h > 1) &
            (distance < MAX_DIST) & (positions[:, 0] > region[0]) & (positions[:, 0] < region[1]) &
            (positions[:, 1] > region[2]) & (positions[:, 1] < region[3]))
    rows = np.column_stack([annotations[:, 0], annotations[:, 1], boxes, positions[:, :2]])
    return rows[keep]


def write_ground_truth(path, rows):
    """Rows are [camera (one based), person, frame, x, y, w, h, X, Y]. Boxes truncate to integers."""
    with Path(path).open('w') as stream:
        for row in rows:
            stream.write('%d %d %d %d %d %d %d %.2f %.2f\n' % tuple(row))


def matrix_text(name, values):
    values = np.asarray(values)
    if np.all(values == np.rint(values)):
        values = values.astype(int)
    with np.printoptions(precision=3, suppress=True):
        lines = str(values).replace('[[', '').replace(' [', '').replace(']', '').split('\n')
    pad = ' '*(len(name) + 4)
    return '\n'.join([f'#{pad}|{lines[0]}|', f'# {name} = |{lines[1]}|', f'#{pad}|{lines[2]}|'])


def transforms_text(transform=MAP_TRANSFORM):
    transform = np.asarray(transform, dtype=float)
    head = ['# Include the transforms as a list of elements in row-major order',
            '# For example for a transform with elements:', '#', '#     | a  b  c |', '# T = | d  e  f |',
            '#     | g  h  i |', '#', '# The yaml config node will be:', '#', '# T: ',
            '#   - ' + '\n#   - '.join('abcdefghi'), '']
    body = ''
    for name, caption in (('T_ov2px', 'world coordinates to pixel coordinates'),
                          ('T_gt2px', 'ground truth coordinates to pixel coordinates')):
        spacing = '\n' if name == 'T_ov2px' else '\n\n'
        body += (f'{spacing}# The transform from {caption}\n#\n{matrix_text(name, transform)}\n#\n'
                 f'# to be entered as below:\n{name}:\n' + ''.join(f'  - {value:.6f}\n' for value in transform.reshape(-1)))
    return '\n'.join(head) + body


def missing_listing(label, paths, limit=3):
    paths = list(paths)
    if not paths:
        return []
    shown = ', '.join(str(path) for path in paths[:limit])
    more = f' and {len(paths) - limit} more' if len(paths) > limit else ''
    return [f'{label}: {len(paths)} missing, for example {shown}{more}']


def validate_source(source, sequence, cameras, frames, annotation_frames, need_map):
    """Return a list of actionable problems, empty when the layout is complete."""
    source = Path(source)
    problems = []
    if not source.is_dir():
        return [f'Source directory {source} does not exist. Download SCOUT from {DOWNLOAD_URL} and pass its folder.']
    missing = [f'calibrations/cam_{camera}.json' for camera in cameras if find_calibration_file(source, camera) is None]
    problems += missing_listing('Calibration files (official calibrations folder)', missing)
    if not (source / 'timestamps' / f'timestamps_{sequence}.json').is_file():
        problems.append(f'Missing timestamps/timestamps_{sequence}.json (official timestamps folder)')
    for camera in cameras:
        images = source / 'images' / sequence / f'cam_{camera}'
        present = frame_ids_in_directory(images, '.jpg')
        gone = [images / f'image_{frame}.jpg' for frame in frames if frame not in present]
        problems += missing_listing(f'Images of camera {camera} (official images/{sequence} subset)', gone)
        annotations = source / 'annotations' / 'individual_format' / f'cam_{camera}'
        present = frame_ids_in_directory(annotations, '.txt')
        gone = [annotations / f'image_{frame}.txt' for frame in annotation_frames if frame not in present]
        problems += missing_listing(f'Annotations of camera {camera} (official individual_format)', gone)
    if need_map:
        for name in ('mesh.ply', *TEXTURES):
            if not (source / 'meshes' / name).is_file():
                problems.append(f'Missing meshes/{name} (official 3D scene meshes, needed for map.png)')
    return problems


def required_subset_text(sequence, cameras, frames, annotation_frames, need_map):
    text = [f'Needed from the official SCOUT release (sequence {sequence}, cameras {cameras[0]} to {cameras[-1]}):',
            f'  calibrations/cam_K, timestamps_{sequence}.json',
            f'  images/{sequence}/cam_K/image_F.jpg for F in {frames[0]}..{frames[-1]} ({len(frames)} frames per camera)',
            f'  annotations/individual_format/cam_K/image_F.txt for F in {annotation_frames[0]}..{annotation_frames[-1]}']
    if need_map:
        text.append('  meshes/mesh.ply and ' + ', '.join(TEXTURES))
    return '\n'.join(text)


def ffmpeg_command(output):
    return ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{FRAME_SIZE[0]}x{FRAME_SIZE[1]}',
            '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-preset', PRESET, '-crf', str(CRF), '-pix_fmt', 'yuv420p',
            '-threads', str(THREADS), '-map_metadata', '-1', '-fflags', '+bitexact', '-flags:v', '+bitexact',
            '-movflags', '+faststart', str(output)]


def undistort_frame(path, maps):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise SourceError(f'Cannot read image {path}. The file may be missing or corrupt, download it again.')
    return cv2.remap(image, maps[0], maps[1], cv2.INTER_LINEAR)


def ordered_window(function, items, pool, window):
    """Yield function(item) in input order with at most window items pulled and unconsumed at once."""
    pending = deque()
    for item in items:
        pending.append(pool.submit(function, item))
        if len(pending) >= window:
            yield pending.popleft().result()
    while pending:
        yield pending.popleft().result()


def encode_camera(job):
    """Undistort the frames of one camera and pipe them to ffmpeg. Runs in a worker process."""
    source, sequence, camera, frames, output = job
    calibration = load_calibration(find_calibration_file(source, camera))
    maps = cv2.initUndistortRectifyMap(calibration['K'], calibration['dist'], None, undistorted_intrinsics(calibration), FRAME_SIZE,
                                       cv2.CV_32FC1)
    folder = Path(source) / 'images' / sequence / f'cam_{camera}'
    temporary = Path(str(output) + '.part.mp4')
    process = subprocess.Popen(ffmpeg_command(temporary), stdin=subprocess.PIPE)
    try:
        try:
            with ThreadPoolExecutor(max_workers=THREADS) as pool:
                work = lambda number: undistort_frame(folder / f'image_{number}.jpg', maps)
                for frame in ordered_window(work, frames, pool, THREADS*WINDOW_FACTOR):
                    process.stdin.write(frame.tobytes())
        except BrokenPipeError:
            pass
        except BaseException:
            process.kill()
            raise
        finally:
            with contextlib.suppress(BrokenPipeError):
                process.stdin.close()
            status = process.wait()
        if status != 0:
            raise RuntimeError(f'ffmpeg failed for camera {camera} with exit status {status}, see its message above')
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return camera


def probe_video(path):
    command = ['ffprobe', '-v', 'error', '-count_packets', '-select_streams', 'v:0', '-show_entries',
               'stream=codec_name,width,height,r_frame_rate,nb_read_packets,pix_fmt', '-of', 'json', str(path)]
    stream = json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)['streams'][0]
    return {'codec': stream['codec_name'], 'width': stream['width'], 'height': stream['height'],
            'fps': stream['r_frame_rate'], 'frames': int(stream['nb_read_packets']), 'pix_fmt': stream['pix_fmt']}


def video_reusable(path, recorded, expected):
    """A video is reused only when its stream properties and its recorded bytes and sha256 all match."""
    path = Path(path)
    if not path.is_file() or not recorded or path.stat().st_size != recorded.get('bytes'):
        return False
    try:
        if probe_video(path) != expected:
            return False
    except (subprocess.CalledProcessError, KeyError, ValueError, IndexError):
        return False
    return sha256_file(path) == recorded.get('sha256')


def expected_video(frames_count):
    return {'codec': 'h264', 'width': FRAME_SIZE[0], 'height': FRAME_SIZE[1], 'fps': f'{FPS}/1', 'frames': frames_count,
            'pix_fmt': 'yuv420p'}


def build_videos(source, sequence, cameras, frames, output, workers, reusable):
    """Encode every camera unless reusable (camera to recorded manifest file entry) validates the existing video."""
    todo = []
    (Path(output) / 'videos').mkdir(parents=True, exist_ok=True)
    for camera in cameras:
        target = Path(output) / 'videos' / f'cam_{camera}.mp4'
        if reusable and video_reusable(target, reusable.get(camera), expected_video(len(frames))):
            print(f'  videos/cam_{camera}.mp4 matches its manifest entry, reusing')
            continue
        todo.append((str(source), sequence, camera, frames, target))
    if todo:
        with Pool(min(workers, len(todo))) as pool:
            for camera in pool.imap_unordered(encode_camera, todo):
                print(f'  videos/cam_{camera}.mp4 done')


_RECORDED = None


def set_recorded(recorded):
    global _RECORDED
    _RECORDED = recorded


def camera_job(job):
    """Ground truth rows of one camera. Runs in a worker process that holds the transformed files."""
    source, camera, frames, ids, region = job
    calibration = load_calibration(find_calibration_file(source, camera))
    annotations = load_camera_annotations(source, camera, frames)
    annotations = annotations[np.isin(annotations[:, 0], list(ids))]
    transformed = _RECORDED.cameras[camera]
    positions = _RECORDED.lookup(annotations[:, 1], annotations[:, 0])
    rows = camera_ground_truth(annotations, calibration, transformed['K'], transformed['R'], transformed['T'], positions, region)
    rows[:, 0] = [ids[int(value)] for value in rows[:, 0]]
    return camera, rows


def build_ground_truth(source, cameras, frames, ids, region, workers, output, recorded):
    jobs = [(str(source), camera, frames, ids, region) for camera in cameras]
    with Pool(min(workers, len(jobs)), initializer=set_recorded, initargs=(recorded,)) as pool:
        results = dict(pool.map(camera_job, jobs))
    tables = []
    for camera in cameras:
        part = results[camera]
        tables.append(np.column_stack([np.full(len(part), camera + 1), part[:, 1], part[:, 0], part[:, 2:]]))
    table = np.vstack(tables)
    write_ground_truth(output, table)
    return table


def write_calibration(recorded, cameras, output):
    folder = Path(output) / 'camInfo'
    folder.mkdir(parents=True, exist_ok=True)
    for camera in cameras:
        transformed = recorded.cameras[camera]
        (folder / f'cam_{camera}.yml').write_text(caminfo_text(projection_matrix(transformed['K'], transformed['R'], transformed['T'])))


def read_ply_mesh(path):
    """Read the textured binary SCOUT mesh: vertices, triangles, per corner texture coordinates, texture ids."""
    with Path(path).open('rb') as stream:
        lines = []
        while True:
            raw = stream.readline()
            if not raw:
                break
            lines.append(raw.decode('ascii', 'replace').strip())
            if lines[-1] == 'end_header':
                break
        offset = stream.tell()
    counts = {}
    for line in lines:
        parts = line.split()
        if parts[:1] == ['element']:
            counts[parts[1]] = int(parts[2])
    expected = ['format binary_little_endian 1.0', 'property float x', 'property float y', 'property float z',
                'property list uchar int vertex_indices', 'property list uchar float texcoord', 'property int texnumber', 'end_header']
    if any(line not in lines for line in expected) or set(counts) != {'vertex', 'face'}:
        raise SourceError(f'{path}: unexpected mesh layout, expected the official binary little endian mesh.ply')
    vertices = np.fromfile(path, dtype='<f4', count=counts['vertex']*3, offset=offset).reshape(-1, 3)
    face_type = np.dtype([('n', 'u1'), ('v', '<i4', 3), ('m', 'u1'), ('uv', '<f4', 6), ('tex', '<i4')])
    faces = np.fromfile(path, dtype=face_type, count=counts['face'], offset=offset + counts['vertex']*12)
    if (faces['n'] != 3).any() or (faces['m'] != 6).any():
        raise SourceError(f'{path}: the mesh contains faces that are not triangles')
    return vertices, faces['v'], faces['uv'].reshape(-1, 3, 2), faces['tex']


def render_map(source, frame):
    """Top down view of the textured official mesh in the transformed world frame.

    Face corners and centers are splatted in file order on a supersampled grid, small holes are
    inpainted and the grid is reduced with area averaging. It approximates the hand rectified map.
    """
    vertices, triangles, texcoords, texture_ids = read_ply_mesh(Path(source) / 'meshes' / 'mesh.ply')
    corners = vertices[triangles]
    keep = np.ones(len(triangles), dtype=bool)
    for axis, key in ((0, 'x'), (1, 'y')):
        mean = corners[:, :, axis].mean(axis=1)
        keep &= (mean >= MESH_RANGES[key][0]) & (mean <= MESH_RANGES[key][1])
    top = corners[:, :, 2].max(axis=1)
    keep &= (top >= MESH_RANGES['z_max'][0]) & (top <= MESH_RANGES['z_max'][1])
    corners, texcoords, texture_ids = corners[keep], texcoords[keep], texture_ids[keep]
    width, height = MAP_SIZE[0]*MAP_SUPERSAMPLE, MAP_SIZE[1]*MAP_SUPERSAMPLE
    affine = np.asarray(MAP_TRANSFORM, dtype=float)[:2]*MAP_SUPERSAMPLE
    image = np.zeros((height, width, 3), dtype=np.uint8)
    covered = np.zeros((height, width), dtype=bool)
    for index in range(int(texture_ids.max()) + 1 if len(texture_ids) else 0):
        selected = texture_ids == index
        if not selected.any():
            continue
        texture = cv2.imread(str(Path(source) / 'meshes' / TEXTURES[index]), cv2.IMREAD_COLOR)
        if texture is None:
            raise SourceError(f'Cannot read mesh texture {TEXTURES[index]}')
        tex_height, tex_width = texture.shape[:2]
        face_corners, face_uvs = corners[selected], texcoords[selected]
        points, colors = [], []
        for corner in range(4):
            if corner < 3:
                xyz, uv = face_corners[:, corner], face_uvs[:, corner]
            else:
                xyz, uv = face_corners.mean(axis=1), face_uvs.mean(axis=1)
            world = frame.apply(xyz)
            pixel = np.rint(world[:, :2] @ affine[:, :2].T + affine[:, 2] - 0.5).astype(np.int64)
            column = np.clip(np.rint(uv[:, 0]*(tex_width - 1)), 0, tex_width - 1).astype(np.int64)
            row = np.clip(np.rint((1 - uv[:, 1])*(tex_height - 1)), 0, tex_height - 1).astype(np.int64)
            inside = (pixel[:, 0] >= 0) & (pixel[:, 0] < width) & (pixel[:, 1] >= 0) & (pixel[:, 1] < height)
            points.append(pixel[inside])
            colors.append(texture[row[inside], column[inside]])
        points, colors = np.concatenate(points), np.concatenate(colors)
        image[points[:, 1], points[:, 0]] = colors
        covered[points[:, 1], points[:, 0]] = True
    closed = cv2.morphologyEx(covered.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((2*MAP_SUPERSAMPLE + 1,)*2, np.uint8))
    holes = (closed > 0) & ~covered
    image = cv2.inpaint(image, holes.astype(np.uint8), MAP_SUPERSAMPLE, cv2.INPAINT_TELEA)
    return cv2.resize(image, MAP_SIZE, interpolation=cv2.INTER_AREA)


def source_fingerprint(source, sequence, cameras, frames, annotation_frames, need_map):
    """Metadata fingerprint of every selected official input (sha256 over sorted relative path, size and mtime_ns lines).

    No file is read and it is not a content hash. Returns the hex digest and the number of files.
    """
    source = Path(source)
    paths = [source / 'timestamps' / f'timestamps_{sequence}.json']
    for camera in cameras:
        paths.append(find_calibration_file(source, camera))
        folder = source / 'annotations' / 'individual_format' / f'cam_{camera}'
        paths.extend(folder / f'image_{frame}.txt' for frame in annotation_frames)
        folder = source / 'images' / sequence / f'cam_{camera}'
        paths.extend(folder / f'image_{frame}.jpg' for frame in frames)
    if need_map:
        paths.extend(source / 'meshes' / name for name in ('mesh.ply', *TEXTURES))
    lines = []
    for path in paths:
        try:
            info = path.stat()
        except OSError as error:
            raise SourceError(f'Cannot stat selected input {path}: {error.strerror}')
        lines.append(f'{path.relative_to(source).as_posix()} {info.st_size} {info.st_mtime_ns}')
    lines.sort()
    return hashlib.sha256('\n'.join(lines).encode()).hexdigest(), len(lines)


def expected_outputs(cameras, with_map):
    """Relative paths of the generated files that belong to the dataset, the only ones the manifest lists."""
    names = [f'videos/cam_{camera}.mp4' for camera in cameras] + [f'camInfo/cam_{camera}.yml' for camera in cameras]
    return names + ['ground_truth.txt', 'transforms.yml'] + (['map.png'] if with_map else [])


def remove_stale_camera_files(output, cameras):
    """Delete generated cam_K.mp4 and cam_K.yml of cameras outside the selection, nothing else is touched."""
    pattern = re.compile(r'cam_(\d+)\.(mp4|yml)$')
    removed = []
    for folder in ('videos', 'camInfo'):
        for path in sorted((Path(output) / folder).glob('cam_*')):
            match = pattern.match(path.name)
            if match and path.is_file() and int(match.group(1)) not in cameras:
                path.unlink()
                removed.append(f'{folder}/{path.name}')
    return removed


def count_annotation_files(source, cameras, frames):
    """Frames of the selected span with an official annotation file for every camera, empty files included."""
    present = None
    for camera in cameras:
        found = frame_ids_in_directory(Path(source) / 'annotations' / 'individual_format' / f'cam_{camera}', '.txt')
        present = found if present is None else present & found
    return sum(1 for frame in frames if frame in present)


def annotated_frame_ids(table):
    """Distinct ground truth frame ids that carry at least one row, sorted."""
    return np.unique(table[:, 2]).astype(int) if len(table) else np.zeros(0, dtype=int)


def protocol_of(args, frames):
    return {'sequence': SEQUENCE, 'split': args.split, 'calibration': 'recomputed' if args.transform_calib else 'recorded',
            'gt_alignment': GT_ALIGNMENT, 'cameras': list(args.cameras), 'region': [float(v) for v in args.region],
            'max_dist': MAX_DIST, 'fps': FPS, 'frames_per_video': len(frames), 'first_frame': frames[0],
            'last_frame': frames[-1]}


def subset_of(source, cameras, frames, annotation_frames, count):
    hashes = {f'cam_{camera}': sha256_file(find_calibration_file(source, camera)) for camera in cameras}
    return {'timestamp_frames': count, 'image_frames': [frames[0], frames[-1]],
            'annotation_frames': [min(annotation_frames), max(annotation_frames)], 'calibration_sha256': hashes}


def manifest_differences(old, protocol, subset, encoding, recorded_hashes, fingerprint, correspondences=None):
    """Human readable list of the differences between an existing manifest and this run.

    recorded_hashes is None when the transformed files are regenerated by this run, they are not compared then.
    """
    differences = []
    before_source = old.get('source', {})
    pairs = [('protocol', old.get('protocol'), protocol), ('source.subset', before_source.get('subset'), subset),
             ('source.fingerprint', before_source.get('fingerprint'), fingerprint),
             ('source.correspondences_sha256', before_source.get('correspondences_sha256'), correspondences),
             ('encoding', old.get('encoding'), encoding)]
    if recorded_hashes is not None:
        pairs.append(('source.transformed_sha256', before_source.get('transformed_sha256'), recorded_hashes))
    for label, before, after in pairs:
        if not isinstance(before, dict) or not isinstance(after, dict):
            if before != after:
                differences.append(f'{label}: {before} -> {after}')
            continue
        for key in sorted(set(before) | set(after)):
            if before.get(key) != after.get(key):
                differences.append(f'{label}.{key}: {before.get(key)} -> {after.get(key)}')
    return differences


def output_problems(source, output, force):
    """Reasons to refuse the output folder, empty when it is safe to write there."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        return [f'--output {output} and --source {source} overlap, pick an output folder outside the official data']
    if output.is_dir() and not force and not (output / 'manifest.json').is_file():
        existing = [path for path in output.rglob('*') if (path.is_file() or path.is_symlink()) and not path.name.endswith('.part.mp4')]
        if existing:
            return [f'--output {output} is not empty and has no manifest.json from this tool, use an empty folder or pass --force '
                    f'(also when a first run was interrupted before it wrote the manifest, for example {existing[0].name})']
    return []


def read_manifest(output):
    path = Path(output) / 'manifest.json'
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except ValueError:
        return {}


def write_manifest(output, args, frames, table, timings, extra, subset, recorded_hashes, files_in_span, fingerprint,
                   correspondences=None):
    """Write manifest.json through a temporary name, so an interrupted run never leaves a half written manifest."""
    files = {}
    for name in expected_outputs(args.cameras, not args.no_map):
        path = Path(output) / name
        if path.is_file():
            files[name] = {'sha256': sha256_file(path), 'bytes': path.stat().st_size}
    videos = {f'cam_{camera}': probe_video(Path(output) / 'videos' / f'cam_{camera}.mp4') for camera in args.cameras}
    ids = annotated_frame_ids(table)
    source = {'download': DOWNLOAD_URL, 'paper': PAPER,
              'subset': subset, 'fingerprint': fingerprint[0], 'fingerprint_inputs': fingerprint[1],
              'fingerprint_note': FINGERPRINT_NOTE, 'transformed_sha256': recorded_hashes}
    if correspondences:
        source['correspondences_sha256'] = correspondences
    manifest = {'dataset': 'scout', 'source': source, 'protocol': protocol_of(args, frames),
                'annotated_frames': int(len(ids)), 'encoding': ENCODING,
                'counts': {'ground_truth_lines': int(len(table)), 'ground_truth_last_frame_id': int(ids[-1]) if len(ids) else 0,
                           'annotation_files_in_span': files_in_span},
                'videos': videos, 'files': files, 'extra': extra,
                'run': {'workers': args.workers, 'stage_seconds': timings}}
    temporary = Path(output, 'manifest.json.part')
    temporary.write_text(json.dumps(manifest, indent=2) + '\n')
    os.replace(temporary, Path(output, 'manifest.json'))
    return manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Build the packaged SCOUT sample from the official SCOUT dataset.')
    parser.add_argument('--source', default='datasets/raw/scout', help=f'Official SCOUT directory, see {DOWNLOAD_URL}')
    parser.add_argument('--output', default='datasets/scout', help='Directory for the converted dataset')
    parser.add_argument('--split', choices=('test', 'train', 'all'), default='test')
    parser.add_argument('--region', type=float, nargs=4, metavar=('XMIN', 'XMAX', 'YMIN', 'YMAX'), default=list(REGION),
                        help='Ground truth region in meters in the transformed frame (default 0 26 0 140)')
    parser.add_argument('--cameras', type=int, nargs='+', default=list(CAMERAS))
    parser.add_argument('--jobs', '--workers', dest='workers', type=int, default=min(8, os.cpu_count() or 1),
                        help='Process pool size')
    parser.add_argument('--transform-calib', action='store_true',
                        help='Rerun the bundle adjustment (needs torch and kornia) instead of loading the transformed files in assets/scout_transformed')
    parser.add_argument('--correspondences', default=str(CORRESPONDENCES),
                        help='Ground plane correspondences json used by --transform-calib (default assets/scout_transformed)')
    parser.add_argument('--transformed-out', help='Folder for the new transformed_calibration.json and transformed_positions.json '
                                                 '(required with --transform-calib, the shipped files are never overwritten)')
    parser.add_argument('--max-iterations', type=int, default=20000, help='Bundle adjustment iterations for --transform-calib (the loss flattens after about 20000)')
    parser.add_argument('--no-map', action='store_true', help='Skip map.png (no mesh needed)')
    parser.add_argument('--check-only', action='store_true', help='Validate the source layout and exit')
    parser.add_argument('--force', action='store_true', help='Rebuild everything and ignore an existing manifest')
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error('--workers must be at least one')
    if any(camera not in CAMERAS for camera in args.cameras):
        parser.error('--cameras must be chosen from 0 to 7, the transformed files cover these cameras')
    args.cameras = sorted(set(args.cameras))
    if args.region[0] >= args.region[1] or args.region[2] >= args.region[3]:
        parser.error('--region needs XMIN < XMAX and YMIN < YMAX')
    if args.transform_calib:
        if not args.transformed_out:
            parser.error('--transform-calib needs --transformed-out DIR for the new json files')
        if args.cameras != list(CAMERAS):
            parser.error('--transform-calib adjusts all eight cameras together, do not pass --cameras')
        if args.max_iterations < 1:
            parser.error('--max-iterations must be at least one')
    return args


def plan_frames(args, count):
    """Frames to encode, annotation frames to read and the frame id mapping of the ground truth."""
    frames = split_frames(count, args.split)
    ids = {frame: frame - frames[0] + 1 for frame in frames}
    return frames, (list(range(count)) if args.transform_calib else sorted(ids)), ids


@contextlib.contextmanager
def timed(timings, name, workers=None):
    start = time.perf_counter()
    print(f'[{name}] started')
    try:
        yield
    finally:
        timings[name] = {'seconds': round(time.perf_counter() - start, 1)}
        if workers:
            timings[name]['workers'] = workers
        print(f'[{name}] finished in {timings[name]["seconds"]} s')


def transform_inputs(source, cameras, count, workers):
    """Official calibrations, undistorted intrinsics and the annotations of all frames for the bundle adjustment."""
    official = {camera: load_calibration(find_calibration_file(source, camera)) for camera in cameras}
    matrices = {camera: undistorted_intrinsics(official[camera]) for camera in cameras}
    with Pool(min(workers, len(cameras))) as pool:
        loaded = pool.starmap(load_camera_annotations, [(str(source), camera, list(range(count))) for camera in cameras])
    return official, matrices, dict(zip(cameras, loaded))


def import_transform():
    """The optional transform module. Importing it does not import torch or kornia."""
    if __package__:
        from . import transform_scout
    else:
        import transform_scout
    return transform_scout


def transform_cameras(args, source, cameras, count):
    """Run the bundle adjustment, write the json files to --transformed-out and return the loaded results."""
    transform = import_transform()
    correspondences = transform.load_correspondences(args.correspondences, cameras)
    official, matrices, annotations = transform_inputs(source, cameras, count, args.workers)
    result = transform.transform_calibration(official, matrices, annotations, correspondences, args.transformed_out, region=REGION,
                                       names=(RECORDED_CALIBRATION, RECORDED_POSITIONS), height=SHIFT_HEIGHT,
                                       frame_size=FRAME_SIZE, max_iter=args.max_iterations)
    recorded = load_recorded(args.transformed_out)
    return recorded, {'best_loss': result['best_loss'], 'first_loss': result['first_loss'], 'final_loss': result['final_loss'],
                      'iterations': args.max_iterations, 'transformed_out': str(args.transformed_out)}


def check_transform_calib(args):
    """Raise ExitError when --transform-calib cannot run, before anything is read or written."""
    transform = import_transform()
    missing = transform.missing_packages()
    if missing:
        raise ExitError(f'--transform-calib needs {" and ".join(missing)}, which {"is" if len(missing) == 1 else "are"} not installed. '
                        f'Install them with: {transform.INSTALL_HINT}', EXIT_MISSING_PACKAGES)
    if not Path(args.correspondences).is_file():
        raise ExitError(f'correspondences file {args.correspondences} does not exist', EXIT_SOURCE)


def plan_source(args):
    """Validate the official files and the transformed files and return the plan of the run."""
    source, cameras = Path(args.source), tuple(args.cameras)
    need_map = not args.no_map
    try:
        count = total_frames(source)
        frames, annotation_frames, ids = plan_frames(args, count)
        print(required_subset_text(SEQUENCE, cameras, frames, annotation_frames, need_map))
        problems = validate_source(source, SEQUENCE, cameras, frames, annotation_frames, need_map)
        if problems:
            raise SourceError('The source layout is incomplete:\n  ' + '\n  '.join(problems) +
                              f'\nFollow the official download page {DOWNLOAD_URL} and place the files in this layout.')
        recorded = None if args.transform_calib else load_recorded()
    except SourceError as error:
        raise ExitError(error, EXIT_SOURCE) from error
    return SimpleNamespace(source=source, output=Path(args.output), cameras=cameras, need_map=need_map, count=count, frames=frames,
                           annotation_frames=annotation_frames, ids=ids, recorded=recorded)


def check_output(args, plan):
    """Refuse an unsafe output folder or a manifest from another run. Returns the videos that may be reused, or None."""
    problems = output_problems(plan.source, plan.output, args.force)
    if args.transform_calib and Path(args.transformed_out).resolve().is_relative_to(plan.source.resolve()):
        problems.append(f'--transformed-out {args.transformed_out} is inside the official data, pick another folder')
    if problems:
        raise ExitError('\n  '.join(problems), EXIT_OUTPUT)
    plan.protocol = protocol_of(args, plan.frames)
    plan.subset = subset_of(plan.source, plan.cameras, plan.frames, plan.annotation_frames, plan.count)
    try:
        plan.fingerprint = source_fingerprint(plan.source, SEQUENCE, plan.cameras, plan.frames, plan.annotation_frames, plan.need_map)
    except SourceError as error:
        raise ExitError(error, EXIT_SOURCE) from error
    plan.correspondences_hash = sha256_file(args.correspondences) if args.transform_calib else None
    old = read_manifest(plan.output)
    if old is None or args.force:
        return None
    differences = (manifest_differences(old, plan.protocol, plan.subset, ENCODING, None if args.transform_calib else plan.recorded.hashes,
                                        plan.fingerprint[0], plan.correspondences_hash) if old else ['manifest unreadable'])
    if differences:
        raise ExitError(f'{plan.output / "manifest.json"} was made with a different protocol, source or transformed files:\n  ' +
                        '\n  '.join(differences) + '\nThe source fingerprint is a metadata fingerprint (path, size, mtime), not a content hash.\n'
                        'Use another --output or pass --force to rebuild everything.', EXIT_MISMATCH)
    return {camera: old['files'].get(f'videos/cam_{camera}.mp4') for camera in plan.cameras}


def build_outputs(args, plan, reusable, started):
    """Write the videos, camera files, ground truth, transforms, map and finally the manifest."""
    output, recorded, extra, timings = plan.output, plan.recorded, {}, {}
    output.mkdir(parents=True, exist_ok=True)
    for path in output.glob('videos/*.part.mp4'):
        path.unlink()
    for name in remove_stale_camera_files(output, plan.cameras):
        print(f'  removed stale {name}')
    if args.transform_calib:
        with timed(timings, 'transform_calib', args.workers):
            try:
                recorded, extra['transform_calib'] = transform_cameras(args, plan.source, plan.cameras, plan.count)
            except (SourceError, ValueError) as error:
                raise ExitError(error, EXIT_SOURCE) from error
    with timed(timings, 'undistort_and_encode', args.workers):
        build_videos(plan.source, SEQUENCE, plan.cameras, plan.frames, output, args.workers, reusable)
    with timed(timings, 'calibration'):
        write_calibration(recorded, plan.cameras, output)
    with timed(timings, 'ground_truth', args.workers):
        try:
            table = build_ground_truth(plan.source, plan.cameras, plan.annotation_frames, plan.ids, args.region, args.workers,
                                       output / 'ground_truth.txt', recorded)
        except SourceError as error:
            raise ExitError(error, EXIT_SOURCE) from error
    ground_ids = annotated_frame_ids(table)
    if len(ground_ids) and (ground_ids[0] != 1 or ground_ids[-1] != len(ground_ids)):
        print(f'warning: ground truth frame ids are not contiguous from 1 ({len(ground_ids)} ids, last {ground_ids[-1]})')
    (output / 'transforms.yml').write_text(transforms_text())
    map_path = output / 'map.png'
    if plan.need_map:
        with timed(timings, 'map'):
            cv2.imwrite(str(map_path), render_map(plan.source, recorded.map_frame))
    else:
        map_path.unlink(missing_ok=True)
    timings['total'] = {'seconds': round(time.perf_counter() - started, 1)}
    manifest = write_manifest(output, args, plan.frames, table, timings, extra, plan.subset, recorded.hashes,
                              count_annotation_files(plan.source, plan.cameras, plan.frames), plan.fingerprint,
                              plan.correspondences_hash)
    print(json.dumps({key: manifest[key] for key in ('protocol', 'annotated_frames', 'counts', 'run')}))


def main(argv=None):
    args = parse_args(argv)
    started = time.perf_counter()
    try:
        if args.transform_calib:
            check_transform_calib(args)
        plan = plan_source(args)
        if args.check_only:
            print('Source layout is complete.')
            return 0
        reusable = check_output(args, plan)
        build_outputs(args, plan, reusable, started)
    except ExitError as error:
        print(f'error: {error}', file=sys.stderr)
        return error.code
    return 0


if __name__ == '__main__':
    sys.exit(main())
