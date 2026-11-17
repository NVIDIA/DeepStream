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

"""Build the MV3DT WILDTRACK sample layout from the official WILDTRACK release.

The official data is distributed by EPFL under its own license and is never shipped with this
repository. This script converts the files the user downloaded into the layout the reference
application expects: videos/000.mp4 ... camInfo/000.yml ... ground_truth.txt, transforms.yml,
map.png and manifest.json.

Conventions of the sample layout:
  * Camera index i uses official camera C(i+1). Frames are the undistorted 10 fps views that start
    at official frame 1800 (the 40 test annotations 1800, 1805, ..., 1995).
  * Only the annotated frames (every 5th) are exact Image_subsets frames. The four frames in
    between are decoded from the raw 59.94 fps GoPro video, taken on the raw frame grid that
    starts at the raw frame matched to each annotated frame and advances 6 raw frames per output
    frame. They are undistorted with intrinsic_original, so all frames share the geometry of
    intrinsic_zero.
"""

import sys
import cv2
import json
import shutil
import hashlib
import argparse
import subprocess

import numpy as np
import xml.etree.ElementTree as ET

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

CAMERAS = ('CVLab1', 'CVLab2', 'CVLab3', 'CVLab4', 'IDIAP1', 'IDIAP2', 'IDIAP3')
FIRST_FRAME = 1800
ANNOTATION_STEP = 5
RAW_PER_ANNOTATION = 30
RAW_PER_FRAME = RAW_PER_ANNOTATION // ANNOTATION_STEP
RAW_FPS = 60000 / 1001
DEFAULT_FRAMES = 196
MAX_FRAMES = 201
OUTPUT_FPS = 25
GRID_COLUMNS = 480
GRID_ORIGIN = (2.4, 7.2)
WORLD_TO_METERS = ((2.5, 0.0, 0.0, 3.0), (0.0, 2.5, 0.0, 9.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0))
MODEL_HEIGHT = 1.7
MODEL_RADIUS = 0.15
SEARCH_RADIUS = 300
SMALL_SIZE = (480, 270)
MAP_SIZE = 860
MAP_MARGIN = 30
MAP_X_RANGE = (-13.0, 21.0)
MAP_Y_RANGE = (-18.0, 16.0)
MAP_X_TICKS = (-10, -5, 0, 5, 10)
MAP_Y_TICKS = (-15, -10, -5, 0, 5, 10, 15)
ZIP_URL = 'https://documents.epfl.ch/groups/c/cv/cvlab-unit/www/data/Wildtrack/Wildtrack_dataset_full.zip'
PAGE_URL = 'https://www.epfl.ch/labs/cvlab/data/data-wildtrack/'
DRIVE_IDS = ('1sGUnExmJM2_tFuBd9LNlexf0LN2m0_c-', '1OnaRN2qYhZ2n4rSaNZQJzQPd1Cl2fluk',
             '1I7ARLVVfZqdZTbYG_TPEfDt1Bc8N35Lb', '1sXn70X-bV_YGPv43r4-iMtK_Js09eUVB',
             '1ExTyezMWqmLefi2kJTQwYxZivJuag7G5', '1NM4kNjdyiC6JioOT90s9vFmMUkeSnV0z',
             '1pZ3pWBuaLgPWGfOcY-tAtsBAUyQZW_YX')


class SourceError(ValueError):
    """Raised for a missing or malformed official input. The message says how to fix it."""


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def frame_name(frame):
    return f'{frame:08d}'


def annotated_frames(frames):
    """Official frame numbers of the annotated frames among the first N output frames."""
    return [FIRST_FRAME + j for j in range(0, frames, ANNOTATION_STEP)]


def required_files(source, frames, with_videos=True):
    """Return a list of (path, hint) for every official file the conversion reads."""
    source = Path(source)
    zip_hint = f'Download Wildtrack_dataset_full.zip from {ZIP_URL} (see {PAGE_URL}) and unzip it'
    items = []
    for index, name in enumerate(CAMERAS):
        for kind in ('intrinsic_zero', 'intrinsic_original'):
            items.append((source / 'calibrations' / kind / f'intr_{name}.xml', zip_hint))
        items.append((source / 'calibrations' / 'extrinsic' / f'extr_{name}.xml', zip_hint))
        for frame in annotated_frames(frames):
            items.append((source / 'Image_subsets' / f'C{index + 1}' / f'{frame_name(frame)}.png', zip_hint))
        if with_videos:
            drive = f'https://drive.google.com/open?id={DRIVE_IDS[index]}' if index < len(DRIVE_IDS) else PAGE_URL
            items.append((source / f'cam{index + 1}.mp4',
                          f'Download the camera {index + 1} video from {drive} (listed on {PAGE_URL}) and save it as cam{index + 1}.mp4'))
    for frame in annotated_frames(frames):
        items.append((source / 'annotations_positions' / f'{frame_name(frame)}.json', zip_hint))
    return items


def validate_source(source, frames, with_videos=True):
    source = Path(source)
    if not source.is_dir():
        raise SourceError(f'Source directory not found: {source}. Pass the folder that contains the official '
                          f'WILDTRACK files (calibrations, annotations_positions, Image_subsets, cam1.mp4 ...). '
                          f'Download Wildtrack_dataset_full.zip from {ZIP_URL}.')
    missing = {}
    for path, hint in required_files(source, frames, with_videos):
        if not path.is_file():
            missing.setdefault(hint, []).append(path)
    if missing:
        lines = [f'Official WILDTRACK files are missing under {source}:']
        for hint, paths in missing.items():
            shown = ', '.join(str(path.relative_to(source)) for path in paths[:4])
            more = f' and {len(paths) - 4} more' if len(paths) > 4 else ''
            lines.append(f'  {len(paths)} missing ({shown}{more}). {hint}.')
        raise SourceError('\n'.join(lines))
    if with_videos and shutil.which('ffmpeg') is None:
        raise SourceError('ffmpeg was not found on PATH. Install it, for example with: sudo apt install ffmpeg')


def read_xml_numbers(path, tag):
    """Read the numbers of an OpenCV XML node, either plain text or an opencv-matrix with a data child."""
    try:
        node = ET.parse(path).getroot().find(tag)
    except ET.ParseError as error:
        raise SourceError(f'{path} is not valid XML: {error}') from error
    if node is None:
        raise SourceError(f'{path} has no {tag} node. Re-extract it from Wildtrack_dataset_full.zip.')
    data = node.find('data')
    text = (data if data is not None else node).text or ''
    try:
        return np.array(text.split(), dtype=float)
    except ValueError as error:
        raise SourceError(f'{path} has non numeric {tag} values') from error


def load_calibration(source, name):
    source = Path(source)
    zero = source / 'calibrations' / 'intrinsic_zero' / f'intr_{name}.xml'
    original = source / 'calibrations' / 'intrinsic_original' / f'intr_{name}.xml'
    extrinsic = source / 'calibrations' / 'extrinsic' / f'extr_{name}.xml'
    matrix = read_xml_numbers(zero, 'camera_matrix').reshape(3, 3)
    original_matrix = read_xml_numbers(original, 'camera_matrix').reshape(3, 3)
    if not np.allclose(matrix, original_matrix):
        raise SourceError(f'{zero} and {original} have different camera matrices. Undistortion would not match.')
    return {'K': matrix, 'dist': read_xml_numbers(original, 'distortion_coefficients'),
            'rvec': read_xml_numbers(extrinsic, 'rvec'), 'tvec': read_xml_numbers(extrinsic, 'tvec')}


def projection_matrix(K, rvec, tvec):
    """3x4 world to pixel matrix. Official tvec is in centimeters, the output world is the grid frame."""
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=float).reshape(3, 1))
    pose = np.hstack([rotation, np.asarray(tvec, dtype=float).reshape(3, 1) / 100.0])
    return np.asarray(K, dtype=float) @ pose @ np.asarray(WORLD_TO_METERS, dtype=float)


def camera_info_text(matrix):
    lines = ['modelInfo:', f'  height: {MODEL_HEIGHT}', f'  radius: {MODEL_RADIUS}', 'projectionMatrix_3x4_w2p:']
    lines += ['-' + ' ' * 15 + repr(float(value)) for value in np.asarray(matrix).reshape(-1)]
    return '\n'.join(lines) + '\n'


def grid_to_world(position_id):
    """Convert an official positionID (480 column grid) to the sample world coordinates."""
    column, row = position_id % GRID_COLUMNS, position_id // GRID_COLUMNS
    return column / 100 - GRID_ORIGIN[0], row / 100 - GRID_ORIGIN[1]


def is_visible(view):
    return (view['xmin'], view['ymin'], view['xmax'], view['ymax']) != (-1, -1, -1, -1)


def ground_truth_lines(source, frames):
    """Rows 'camera id frame x y w h X Y'. Frame counts annotated frames from 1, camera from 0."""
    lines = []
    for number, frame in enumerate(annotated_frames(frames), 1):
        path = Path(source) / 'annotations_positions' / f'{frame_name(frame)}.json'
        try:
            people = json.loads(path.read_text())
        except json.JSONDecodeError as error:
            raise SourceError(f'{path} is not valid JSON: {error}') from error
        for person in people:
            x, y = grid_to_world(int(person['positionID']))
            for view in person['views']:
                if view['viewNum'] < len(CAMERAS) and is_visible(view):
                    width, height = view['xmax'] - view['xmin'], view['ymax'] - view['ymin']
                    lines.append(f"{view['viewNum']} {person['personID']} {number} {view['xmin']} {view['ymin']} {width} {height} {x!r} {y!r}")
    return lines


def map_transform():
    scale = (MAP_SIZE - 2 * MAP_MARGIN) / (MAP_X_RANGE[1] - MAP_X_RANGE[0])
    return np.array([[scale, 0.0, MAP_MARGIN - MAP_X_RANGE[0] * scale],
                     [0.0, -scale, MAP_MARGIN + MAP_Y_RANGE[1] * scale],
                     [0.0, 0.0, 1.0]])


def transforms_text(transform):
    return 'T_ov2px:\n' + ''.join(f'- {float(value)!r}\n' for value in transform.reshape(-1))


def draw_text(image, text, origin):
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1, cv2.LINE_8)


def render_map():
    transform = map_transform()
    image = np.full((MAP_SIZE, MAP_SIZE, 3), 245, np.uint8)
    top, bottom = MAP_MARGIN, MAP_SIZE - MAP_MARGIN
    for value in MAP_X_TICKS:
        column = int(transform[0, 2] + transform[0, 0] * value)
        cv2.line(image, (column, top), (column, bottom), (210, 210, 210), 1)
    for value in MAP_Y_TICKS:
        row = int(transform[1, 2] + transform[1, 1] * value)
        cv2.line(image, (top, row), (bottom, row), (210, 210, 210), 1)
    for value in MAP_X_TICKS:
        draw_text(image, str(value), (int(transform[0, 2] + transform[0, 0] * value), MAP_SIZE - 10))
    for value in MAP_Y_TICKS:
        draw_text(image, str(value), (0, int(transform[1, 2] + transform[1, 1] * value)))
    return image


def ffmpeg_decode(video, first_index, count, size=None, gray=False):
    """Yield decoded frames first_index ... first_index + count - 1 of a raw video as numpy arrays.

    Seeking is by timestamp with a half frame bias, so the first yielded frame is frame first_index
    of the constant frame rate stream. Frames are never dropped or duplicated.
    """
    start = max(first_index - 0.5, 0.0) / RAW_FPS
    command = ['ffmpeg', '-v', 'error', '-nostdin', '-ss', f'{start:.6f}', '-i', str(video), '-frames:v', str(count)]
    if size:
        command += ['-vf', f'scale={size[0]}:{size[1]}:flags=area']
    command += ['-fps_mode', 'passthrough', '-f', 'rawvideo', '-pix_fmt', 'gray' if gray else 'bgr24', '-']
    probe = cv2.VideoCapture(str(video))
    width, height = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH)), int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    probe.release()
    if size:
        width, height = size
    channels = 1 if gray else 3
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        while True:
            data = process.stdout.read(width * height * channels)
            if len(data) < width * height * channels:
                break
            yield np.frombuffer(data, np.uint8).reshape((height, width) if gray else (height, width, 3))
    finally:
        process.stdout.close()
        process.kill()
        process.wait()


def scaled_camera(K, full_size, small_size):
    sx, sy = small_size[0] / full_size[0], small_size[1] / full_size[1]
    scaled = np.array(K, dtype=float)
    scaled[0] = scaled[0] * sx
    scaled[1] = scaled[1] * sy
    scaled[0, 2] = (K[0, 2] + 0.5) * sx - 0.5
    scaled[1, 2] = (K[1, 2] + 0.5) * sy - 0.5
    return scaled


def small_gray(image):
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.resize(image, SMALL_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)


def score_frames(frames, K, dist, target):
    """Mean absolute error of each undistorted gray frame against target, without keeping the frames."""
    return np.array([float(np.abs(cv2.undistort(frame, K, dist).astype(np.float32) - target).mean()) for frame in frames])


def combine_scores(scores, radius=SEARCH_RADIUS):
    """Pick the raw frame of annotated frame 0 that best explains every probe at once.

    scores maps probe number k to an array of length 2 * radius + 1 holding the error of raw frame
    (center + 30 * k + shift) for shift in -radius ... radius. Returns (shift, per probe best shift, total error).
    """
    total = sum(scores.values())
    best = int(np.argmin(total)) - radius
    individual = {k: int(np.argmin(values)) - radius for k, values in scores.items()}
    return best, individual, float(total[best + radius] / len(scores))


def find_alignment(source, index, calibration, frames, log=print):
    """Find the raw frame number a such that raw frame a + 30 k shows annotated frame k of camera index."""
    video = Path(source) / f'cam{index + 1}.mp4'
    count = len(annotated_frames(frames))
    chosen = sorted({0, (count - 1) // 2, count - 1})
    expected = int(round(FIRST_FRAME * RAW_PER_FRAME * 0.999))
    small_K = None
    scores = {}
    for k in chosen:
        png = cv2.imread(str(Path(source) / 'Image_subsets' / f'C{index + 1}' / f'{frame_name(FIRST_FRAME + ANNOTATION_STEP * k)}.png'))
        if png is None:
            raise SourceError(f'Cannot read the Image_subsets frame for camera {index + 1}, annotated frame {k}')
        size = (png.shape[1], png.shape[0])
        if small_K is None:
            small_K = scaled_camera(calibration['K'], size, SMALL_SIZE)
        target = small_gray(png)
        first = expected + RAW_PER_ANNOTATION * k - SEARCH_RADIUS
        values = score_frames(ffmpeg_decode(video, first, 2 * SEARCH_RADIUS + 1, SMALL_SIZE, gray=True),
                              small_K, calibration['dist'], target)
        if len(values) < 2 * SEARCH_RADIUS + 1:
            raise SourceError(f'{video} is too short or truncated. Download cam{index + 1}.mp4 again.')
        scores[k] = values
    best, individual, error = combine_scores(scores)
    spread = max(abs(value - best) for value in individual.values())
    typical = float(np.median(sum(scores.values())) / len(scores))
    if spread > 1 or error > 0.5 * typical:
        raise SourceError(f'Could not align cam{index + 1}.mp4 with Image_subsets/C{index + 1} '
                          f'(per frame matches {individual}, error {error:.2f}). Use the official files of the same camera.')
    log(f'  camera {index}: raw frame {expected + best} is annotated frame 0 (match error {error:.2f})')
    return expected + best


def frame_plan(alignment, frames):
    """For output frame j return ('png', official frame) or ('raw', raw frame number)."""
    plan = []
    for j in range(frames):
        block, step = divmod(j, ANNOTATION_STEP)
        if step == 0:
            plan.append(('png', FIRST_FRAME + j))
        else:
            plan.append(('raw', alignment + RAW_PER_ANNOTATION * block + RAW_PER_FRAME * step))
    return plan


def encoder_command(output, width, height):
    return ['ffmpeg', '-v', 'error', '-nostdin', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{width}x{height}',
            '-r', str(OUTPUT_FPS), '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
            '-pix_fmt', 'yuv420p', '-threads', '1', '-fflags', '+bitexact', '-flags:v', '+bitexact', str(output)]


def write_video(source, index, calibration, alignment, frames, output):
    plan = frame_plan(alignment, frames)
    first_png = cv2.imread(str(Path(source) / 'Image_subsets' / f'C{index + 1}' / f'{frame_name(FIRST_FRAME)}.png'))
    height, width = first_png.shape[:2]
    raw_numbers = [number for kind, number in plan if kind == 'raw']
    maps = cv2.initUndistortRectifyMap(calibration['K'], calibration['dist'], None, calibration['K'],
                                       (width, height), cv2.CV_32FC1)
    stream = position = None
    if raw_numbers:
        # Raw frame numbers increase with the output frame, so the raw video is decoded once, in order.
        stream = ffmpeg_decode(Path(source) / f'cam{index + 1}.mp4', raw_numbers[0], raw_numbers[-1] - raw_numbers[0] + 1)
        position = raw_numbers[0] - 1
    temporary = Path(str(output) + '.part.mp4')
    encoder = subprocess.Popen(encoder_command(temporary, width, height), stdin=subprocess.PIPE)
    try:
        for kind, number in plan:
            if kind == 'png':
                image = cv2.imread(str(Path(source) / 'Image_subsets' / f'C{index + 1}' / f'{frame_name(number)}.png'))
            else:
                frame = None
                while position < number:
                    frame = next(stream, None)
                    position += 1
                    if frame is None:
                        raise SourceError(f'cam{index + 1}.mp4 ended early. Download it again.')
                image = cv2.remap(frame, maps[0], maps[1], cv2.INTER_LINEAR)
            if image is None or image.shape[:2] != (height, width):
                raise SourceError(f'Unexpected frame size for camera {index + 1}, {kind} {number}')
            encoder.stdin.write(np.ascontiguousarray(image).tobytes())
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise SourceError(f'ffmpeg failed while encoding {output}')
    except BaseException:
        encoder.kill()
        raise
    finally:
        if stream is not None:
            stream.close()
    temporary.replace(output)


def convert_camera(source, index, output, frames, log=print):
    calibration = load_calibration(source, CAMERAS[index])
    alignment = find_alignment(source, index, calibration, frames, log=log)
    write_video(source, index, calibration, alignment, frames, Path(output) / 'videos' / f'{index:03d}.mp4')
    return {'camera': index, 'raw_start_frame': alignment}


def check_overlap(source, output):
    """Refuse an output folder that is, contains or lies inside the source folder."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or output.is_relative_to(source) or source.is_relative_to(output):
        raise SourceError(f'--output {output} and --source {source} overlap. Choose an output folder that is '
                          f'separate from the official files, otherwise generated data could mix with them.')


def has_content(folder):
    """True if the folder holds anything other than stale partial videos from an interrupted run."""
    return folder.exists() and any(path.is_file() and not path.name.endswith('.part.mp4') for path in folder.rglob('*'))


def protocol_description(frames):
    return {'first_official_frame': FIRST_FRAME, 'frames': frames, 'fps': OUTPUT_FPS,
            'annotated_frames': len(annotated_frames(frames)), 'cameras': len(CAMERAS)}


def source_description(source, frames, with_videos=True):
    """Describe the official inputs. text_inputs_sha256 hashes the XML and JSON content. fingerprint
    covers the path, size and mtime of every input used (including videos and PNGs), so it is cheap but not
    a content hash."""
    digest = hashlib.sha256()
    lines = []
    for path, _ in required_files(source, frames, with_videos):
        status = path.stat()
        lines.append(f'{path.relative_to(source)} {status.st_size} {status.st_mtime_ns}')
        if path.suffix in ('.xml', '.json'):
            digest.update(f'{path.relative_to(source)} {sha256(path)}\n'.encode())
    text_hash = digest.hexdigest()
    lines.append(f'text_inputs_sha256 {text_hash}')
    fingerprint = hashlib.sha256('\n'.join(sorted(lines)).encode()).hexdigest()
    return {'cameras': list(CAMERAS), 'official_frames': [FIRST_FRAME, FIRST_FRAME + frames - 1],
            'text_inputs_sha256': text_hash, 'fingerprint': fingerprint}


def manifest_differences(old, protocol, source):
    """List how an existing manifest disagrees with the requested protocol and source subset."""
    differences = []
    if old.get('dataset') != 'wildtrack':
        differences.append(f"dataset: {old.get('dataset')!r} != 'wildtrack'")
    for name, wanted, found in (('protocol', protocol, old.get('protocol') or {}), ('source', source, old.get('source') or {})):
        for key in sorted(set(wanted) | set(found)):
            if wanted.get(key) != found.get(key):
                differences.append(f'{name}.{key}: existing {found.get(key)!r}, requested {wanted.get(key)!r}')
    return differences


def read_manifest(output):
    path = Path(output) / 'manifest.json'
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def probe_video(path):
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
               'stream=codec_name,pix_fmt,width,height,avg_frame_rate,nb_frames', '-of', 'json', str(path)]
    try:
        stream = json.loads(subprocess.run(command, capture_output=True, check=True, text=True).stdout)['streams'][0]
    except (subprocess.CalledProcessError, OSError, KeyError, IndexError, json.JSONDecodeError):
        return None
    return {'codec': stream.get('codec_name'), 'pix_fmt': stream.get('pix_fmt'), 'width': stream.get('width'),
            'height': stream.get('height'), 'fps': stream.get('avg_frame_rate'), 'frames': stream.get('nb_frames')}


def expected_video(source, index, frames):
    image = cv2.imread(str(Path(source) / 'Image_subsets' / f'C{index + 1}' / f'{frame_name(FIRST_FRAME)}.png'))
    if image is None:
        return None
    return {'codec': 'h264', 'pix_fmt': 'yuv420p', 'width': image.shape[1], 'height': image.shape[0],
            'fps': f'{OUTPUT_FPS}/1', 'frames': str(frames)}


def reusable_video(path, recorded, expected):
    """True only if the file matches the recorded bytes and sha256 and the expected stream properties."""
    path = Path(path)
    if not path.is_file() or not recorded or expected is None:
        return False
    if path.stat().st_size != recorded.get('bytes') or sha256(path) != recorded.get('sha256'):
        return False
    return probe_video(path) == expected


def expected_outputs(cameras=None):
    """Relative paths of every generated file. Nothing else belongs in the manifest."""
    count = len(CAMERAS) if cameras is None else cameras
    return sorted([f'videos/{index:03d}.mp4' for index in range(count)] + [f'camInfo/{index:03d}.yml' for index in range(count)] +
                  ['ground_truth.txt', 'transforms.yml', 'map.png'])


def remove_stale_cameras(output):
    """Delete numbered camera files outside the selected cameras. Other files and folders stay untouched."""
    removed = []
    for folder, suffix in (('videos', '.mp4'), ('camInfo', '.yml')):
        for path in sorted((Path(output) / folder).glob(f'[0-9][0-9][0-9]{suffix}')):
            if int(path.stem) >= len(CAMERAS):
                path.unlink()
                removed.append(f'{folder}/{path.name}')
    return removed


def build_manifest(output, frames, source, alignments):
    files = [Path(output) / name for name in expected_outputs() if (Path(output) / name).is_file()]
    protocol = protocol_description(frames)
    return {'dataset': 'wildtrack', 'cameras': len(CAMERAS), 'frames': frames,
            'annotated_frames': protocol['annotated_frames'], 'first_official_frame': FIRST_FRAME,
            'protocol': protocol, 'source': source,
            'ground_truth_rows': len((Path(output) / 'ground_truth.txt').read_text().splitlines()),
            'alignment': alignments,
            'files': {str(path.relative_to(output)): {'sha256': sha256(path), 'bytes': path.stat().st_size} for path in files}}


def prepare(source, output, frames=DEFAULT_FRAMES, force=False, jobs=None, videos=True, log=print):
    source, output = Path(source), Path(output)
    if not 1 <= frames <= MAX_FRAMES:
        raise SourceError(f'--frames must be between 1 and {MAX_FRAMES}')
    check_overlap(source, output)
    validate_source(source, frames, videos)
    protocol = protocol_description(frames)
    description = source_description(source, frames, videos)
    old = None
    if has_content(output) and not force:
        old = read_manifest(output)
        if old is None:
            raise SourceError(f'{output} is not empty and has no valid manifest.json. Use --force to overwrite the generated files.')
        differences = manifest_differences(old, protocol, description)
        if differences:
            raise SourceError(f'{output} was built with a different protocol or source subset:\n  ' + '\n  '.join(differences) +
                              '\nUse --force to rebuild everything or choose another --output.')
    for folder in ('videos', 'camInfo'):
        (output / folder).mkdir(parents=True, exist_ok=True)
    for leftover in (output / 'videos').glob('*.part.mp4'):
        leftover.unlink()
    for name in remove_stale_cameras(output):
        log(f'  removed stale generated file {name}')
    for index, name in enumerate(CAMERAS):
        calibration = load_calibration(source, name)
        matrix = projection_matrix(calibration['K'], calibration['rvec'], calibration['tvec'])
        (output / 'camInfo' / f'{index:03d}.yml').write_text(camera_info_text(matrix))
    (output / 'ground_truth.txt').write_text('\n'.join(ground_truth_lines(source, frames)) + '\n')
    (output / 'transforms.yml').write_text(transforms_text(map_transform()))
    cv2.imwrite(str(output / 'map.png'), render_map())
    results = {}
    if videos:
        recorded_alignment = {item['camera']: item for item in (old or {}).get('alignment', [])}
        for index in range(len(CAMERAS)):
            name = f'videos/{index:03d}.mp4'
            if index in recorded_alignment and reusable_video(output / name, old.get('files', {}).get(name),
                                                              expected_video(source, index, frames)):
                results[index] = recorded_alignment[index]
                log(f'  camera {index}: reusing verified {name}')
        todo = [index for index in range(len(CAMERAS)) if index not in results]
        with ThreadPoolExecutor(jobs or len(CAMERAS)) as pool:
            futures = {index: pool.submit(convert_camera, source, index, output, frames, log) for index in todo}
            results.update({index: future.result() for index, future in futures.items()})
    alignments = [results[index] for index in sorted(results)]
    manifest = build_manifest(output, frames, description, alignments)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    log(f'Done. Output: {output}')
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', type=Path, default=Path('datasets/raw/wildtrack'), help='Folder with the official WILDTRACK files')
    parser.add_argument('--output', type=Path, default=Path('datasets/wildtrack'))
    parser.add_argument('--frames', type=int, default=DEFAULT_FRAMES, help='Frames per video (default 196)')
    parser.add_argument('--force', action='store_true', help='Rebuild everything, ignoring any existing manifest and videos')
    parser.add_argument('--jobs', type=int, help='Cameras converted in parallel (default 7)')
    args = parser.parse_args(argv)
    try:
        prepare(args.source, args.output, args.frames, args.force, args.jobs)
    except (SourceError, OSError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
