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
"""Check the SCOUT preparation on tiny synthetic fixtures, no real SCOUT data is used."""

import io
import os
import sys
import gzip
import json
import shutil
import unittest
import tempfile
import contextlib
import subprocess

import cv2
import yaml
import numpy as np

from pathlib import Path
from unittest import mock
from utils import evaluate_tracking
from utils import prepare_scout as scout
from utils import transform_scout as transform

SIZE = (64, 48)
DIST = [-0.12, 0.03, 0.001, -0.001, 0.0] + [0.0]*9
MATRIX = [[50.0, 0.0, 32.0], [0.0, 50.0, 24.0], [0.0, 0.0, 1.0]]
TRANSFORMED_MATRIX = [[52.0, 0.0, 31.5], [0.0, 52.5, 24.5], [0.0, 0.0, 1.0]]
TARGET = np.array([[5.0, 10.0, 0.0]])
OFFICIAL_PERSON = np.array([5.1, 10.2, -0.1])
CENTERS = {0: (5.0, -3.0, 3.0), 1: (14.0, 12.0, 3.0)}
REGION = scout.REGION
RECORDED_NAMES = (scout.RECORDED_CALIBRATION, scout.RECORDED_POSITIONS)
APP = Path(__file__).resolve().parents[1]
FFMPEG = shutil.which('ffmpeg') and shutil.which('ffprobe')
needs_ffmpeg = unittest.skipUnless(FFMPEG, 'ffmpeg is not installed')
SHARED = {}


def camera(center, target=TARGET[0]):
    center = np.asarray(center, dtype=float)
    forward = (target - center)/np.linalg.norm(target - center)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward])
    return rotation, -rotation @ center


def write_calibration_file(path, rotation, translation, suffix='json'):
    values = {'K': MATRIX, 'R': rotation.tolist(), 'T': np.asarray(translation).reshape(3, 1).tolist(), 'dist': DIST}
    text = json.dumps(values) if suffix == 'json' else yaml.safe_dump(values)
    Path(path).write_text(text)


def encode_positions(entries):
    """Columnar positions as stored in assets/scout_transformed, entries map (person, frame) to (x, y, z) in cm."""
    order = sorted(entries, key=lambda pair: (pair[1], pair[0]))
    frames, counts, persons, x, y, z_index, z = [], [], [], [], [], [], []
    for index, (person, frame) in enumerate(order):
        if not frames or frames[-1] != frame:
            frames.append(frame)
            counts.append(0)
            previous = 0
        counts[-1] += 1
        persons.append(person - previous)
        previous = person
        x.append(entries[(person, frame)][0])
        y.append(entries[(person, frame)][1])
        if entries[(person, frame)][2]:
            z_index.append(index)
            z.append(entries[(person, frame)][2])
    return {'format': 'scout-transformed-positions-1', 'count': len(order), 'frames': frames, 'per_frame': counts,
            'person': persons, 'x_cm': x, 'y_cm': y, 'z_index': z_index, 'z_cm': z}


def make_recorded(root, frames=range(0, 6), gzip_positions=False, drop=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    cameras = {}
    for index in range(8):
        rotation, translation = camera(CENTERS[index % 2])
        cameras[str(index)] = {'K': TRANSFORMED_MATRIX, 'R': rotation.tolist(), 'T': translation.tolist()}
    calibration = {'format': 'scout-transformed-calibration-1', 'cameras': cameras,
                   'map_frame': {'scale': 1.0, 'yaw_deg': 0.0, 'translation': [0.0, 0.0]}}
    (root / 'transformed_calibration.json').write_text(json.dumps(calibration))
    entries = {(7, frame): (500, 1000, 0) for frame in frames}
    entries[(9, 4)] = (600, 1100, 30)
    entries.pop(drop, None)
    text = json.dumps(encode_positions(entries))
    if gzip_positions:
        (root / 'transformed_positions.json.gz').write_bytes(gzip.compress(text.encode(), mtime=0))
    else:
        (root / 'transformed_positions.json').write_text(text)
    return root


def make_source(root, frames=6, cameras=(0, 1), suffix='json', images=True):
    root = Path(root)
    (root / 'calibrations').mkdir(parents=True)
    (root / 'timestamps').mkdir()
    (root / 'timestamps' / 'timestamps_sequence_1.json').write_text(json.dumps({str(i): {} for i in range(frames)}))
    for index in cameras:
        rotation, translation = camera(CENTERS[index % 2], OFFICIAL_PERSON)
        write_calibration_file(root / 'calibrations' / f'cam_{index}.{suffix}', rotation, translation, suffix)
        images_dir = root / 'images' / 'sequence_1' / f'cam_{index}'
        notes_dir = root / 'annotations' / 'individual_format' / f'cam_{index}'
        images_dir.mkdir(parents=True)
        notes_dir.mkdir(parents=True)
        for frame in range(frames):
            if images:
                cv2.imwrite(str(images_dir / f'image_{frame}.jpg'), np.full((SIZE[1], SIZE[0], 3), 10*frame, np.uint8))
            line = f'7, 20.00, 10.00, 10.00, 20.00, -1, {OFFICIAL_PERSON[0]:.3f}, {OFFICIAL_PERSON[1]:.3f}, {OFFICIAL_PERSON[2]:.3f}\n'
            (notes_dir / f'image_{frame}.txt').write_text(line)
    return root


@contextlib.contextmanager
def quiet():
    """Silence stdout and stderr, yields the captured stderr."""
    error = io.StringIO()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(error):
        yield error


def setUpModule():
    """One prepared output (cameras 0 and 1, frames 3 to 5) shared by the end-to-end tests, copied when a test mutates it."""
    if not FFMPEG:
        return
    SHARED['temporary'] = tempfile.TemporaryDirectory()
    root = Path(SHARED['temporary'].name)
    recorded, source = make_recorded(root / 'recorded'), make_source(root / 'source', frames=6)
    with mock.patch.object(scout, 'FRAME_SIZE', SIZE), mock.patch.object(scout, 'RECORDED_DIR', recorded), quiet():
        SHARED['code'] = scout.main(['--source', str(source), '--output', str(root / 'out'), '--cameras', '0', '1', '--no-map',
                                     '--workers', '2'])
    SHARED['root'] = root


def tearDownModule():
    if 'temporary' in SHARED:
        SHARED['temporary'].cleanup()


class ScoutCase(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.recorded = make_recorded(self.root / 'recorded')
        self.use_recorded(self.recorded)
        patcher = mock.patch.object(scout, 'FRAME_SIZE', SIZE)
        patcher.start()
        self.addCleanup(patcher.stop)

    def use_recorded(self, directory):
        patcher = mock.patch.object(scout, 'RECORDED_DIR', directory)
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_main(self, source, output, *extra, cameras=('0', '1')):
        """Run the converter without a map, returns the exit code and the captured stderr."""
        argv = ['--source', str(source), '--output', str(output), '--no-map', '--workers', '2']
        if cameras:
            argv += ['--cameras', *cameras]
        with quiet() as error:
            code = scout.main([*argv, *extra])
        return code, error.getvalue()


class TinyTests(ScoutCase):
    def test_calibration_parsing(self):
        rotation, translation = camera(CENTERS[0])
        for suffix in ('json', 'yaml'):
            path = self.root / f'cam.{suffix}'
            write_calibration_file(path, rotation, translation, suffix)
            calibration = scout.load_calibration(path)
            self.assertEqual((calibration['K'].shape, calibration['T'].shape, calibration['dist'].size), ((3, 3), (3, 1), 14))
            np.testing.assert_allclose(calibration['R'], rotation)
        bad = {'missing calibration keys': {'K': MATRIX},
               'distortion length': {'K': MATRIX, 'R': np.eye(3).tolist(), 'T': [0, 0, 1], 'dist': [0, 0, 0]}}
        for message, values in bad.items():
            with self.subTest(message):
                (self.root / 'bad.json').write_text(json.dumps(values))
                with self.assertRaisesRegex(scout.SourceError, message):
                    scout.load_calibration(self.root / 'bad.json')

    def test_projection_matrix_matches_opencv_projection(self):
        rotation, translation = camera(CENTERS[0])
        matrix = np.array(TRANSFORMED_MATRIX)
        projection = scout.projection_matrix(matrix, rotation, translation)
        points = np.array([[5.0, 10.0, 0.0], [6.0, 12.0, 1.3], [3.0, 8.0, 0.4]])
        homogeneous = projection @ np.c_[points, np.ones(3)].T
        np.testing.assert_allclose((homogeneous[:2]/homogeneous[2]).T, scout.project_points(points, matrix, rotation, translation), atol=1e-6)

    def test_recorded_json_loading_and_errors(self):
        recorded = scout.load_recorded()
        self.assertEqual(sorted(recorded.cameras), list(range(8)))
        np.testing.assert_allclose(recorded.lookup([7, 9, 7], [2, 4, 5]), [[5.0, 10.0, 0.0], [6.0, 11.0, 0.3], [5.0, 10.0, 0.0]])
        with self.assertRaisesRegex(scout.SourceError, 'no position in the transformed files.*person 8 frame 3'):
            recorded.lookup([7, 8], [3, 3])
        zipped = make_recorded(self.root / 'zipped', gzip_positions=True)
        np.testing.assert_allclose(scout.load_recorded(zipped).lookup([9], [4]), [[6.0, 11.0, 0.3]])
        broken = make_recorded(self.root / 'broken')
        path = broken / 'transformed_positions.json'
        values = json.loads(path.read_text())
        values['count'] += 1
        path.write_text(json.dumps(values))
        with self.assertRaisesRegex(scout.SourceError, 'inconsistent'):
            scout.load_recorded(broken)
        values['count'] -= 1
        values['format'] = 'other'
        path.write_text(json.dumps(values))
        with self.assertRaisesRegex(scout.SourceError, 'unexpected format'):
            scout.load_recorded(broken)
        path.unlink()
        with self.assertRaisesRegex(scout.SourceError, 'Missing transformed files'):
            scout.load_recorded(broken)

    def test_shift_rule_and_row_filters(self):
        rotation, translation = camera(CENTERS[0], OFFICIAL_PERSON)
        write_calibration_file(self.root / 'cam.json', rotation, translation)
        calibration = scout.load_calibration(self.root / 'cam.json')
        transformed = scout.load_recorded().cameras[0]
        outside = np.array([45.0, 10.0, 0.0])
        official = np.array([[3, 7, 20, 10, 10, 20, *OFFICIAL_PERSON], [3, 8, 20, 10, 10, 20, *OFFICIAL_PERSON],
                             [3, 9, 20, 10, 0.5, 20, *OFFICIAL_PERSON], [4, 7, 20, 10, 10, 20, *OFFICIAL_PERSON]])
        positions = np.array([TARGET[0], outside, TARGET[0], TARGET[0]])

        def kept(annotations, positions):
            return scout.camera_ground_truth(annotations, calibration, transformed['K'], transformed['R'], transformed['T'], positions, REGION)
        rows = kept(official, positions)
        self.assertEqual(rows[:, 1].tolist(), [7, 7])
        np.testing.assert_allclose(rows[0, 6:8], [5.0, 10.0], atol=1e-9)
        distorted = scout.project_points(OFFICIAL_PERSON[None], calibration['K'], calibration['R'], calibration['T'], calibration['dist'])
        moved = scout.project_points(TARGET, transformed['K'], transformed['R'], transformed['T'])
        np.testing.assert_allclose(rows[0, 2:4], np.array([20, 10]) + moved[0] - distorted[0], atol=1e-9)
        np.testing.assert_allclose(rows[0, 4:6], [10, 20])
        with mock.patch.object(scout, 'MAX_DIST', 1.0):
            self.assertEqual(kept(official, positions).shape[0], 0)
        far = np.array([[3, 7, 5000, 10, 10, 20, *OFFICIAL_PERSON]])
        self.assertEqual(kept(far, positions[:1]).shape[0], 0)

    def test_default_protocol_matches_the_evaluator(self):
        args = scout.parse_args(['--source', 'a', '--output', 'b'])
        expected = evaluate_tracking.DATASETS['scout']['protocol']
        protocol = scout.protocol_of(args, scout.split_frames(11960, 'test'))
        self.assertEqual(protocol, {**expected, 'calibration': 'recorded', 'region': [float(v) for v in expected['region']]})

    def test_source_and_output_default_to_the_datasets_folders(self):
        args = scout.parse_args([])
        self.assertEqual((args.source, args.output), ('datasets/raw/scout', 'datasets/scout'))

    def test_frame_split_and_id_mapping(self):
        self.assertEqual(scout.split_frames(7, 'train'), [0, 1, 2])
        args = scout.parse_args(['--source', 'a', '--output', 'b'])
        frames, annotation_frames, ids = scout.plan_frames(args, 11960)
        self.assertEqual((frames[0], len(frames), annotation_frames[-1], ids[5980]), (5980, 5980, 11959, 1))
        self.assertEqual(scout.plan_frames(args, 6), ([3, 4, 5], [3, 4, 5], {3: 1, 4: 2, 5: 3}))

    def test_missing_inputs_are_reported(self):
        source = make_source(self.root / 'source', frames=6)
        frames = [3, 4, 5]
        self.assertEqual(scout.validate_source(source, 'sequence_1', (0, 1), frames, [2, 3, 4, 5], need_map=False), [])
        (source / 'images' / 'sequence_1' / 'cam_1' / 'image_4.jpg').unlink()
        shutil.rmtree(source / 'annotations' / 'individual_format' / 'cam_0')
        (source / 'calibrations' / 'cam_1.json').unlink()
        text = '\n'.join(scout.validate_source(source, 'sequence_1', (0, 1), frames, [2, 3, 4, 5], need_map=True))
        for expected in ('Calibration files', 'Images of camera 1', 'image_4.jpg', 'Annotations of camera 0', 'meshes/mesh.ply'):
            self.assertIn(expected, text)
        self.assertIn('Source directory', scout.validate_source(self.root / 'absent', 'sequence_1', (0,), frames, frames, False)[0])
        with self.assertRaisesRegex(scout.SourceError, 'timestamps_sequence_1.json'):
            scout.total_frames(self.root)
        code, message = self.run_main(source, self.root / 'out')
        self.assertEqual(code, scout.EXIT_SOURCE)
        self.assertIn('official download page', message)

    def test_missing_transformed_files_stop_the_run(self):
        source = make_source(self.root / 'source', frames=6)
        with mock.patch.object(scout, 'RECORDED_DIR', self.root / 'nowhere'):
            code, message = self.run_main(source, self.root / 'out', cameras=('0',))
        self.assertEqual(code, scout.EXIT_SOURCE)
        self.assertIn('Restore assets/scout_transformed', message)

    def test_invalid_arguments_are_rejected(self):
        for flag in (['--region', '5', '1', '0', '2'], ['--cameras', '0', '9'], ['--workers', '0'], ['--sequence', 'sequence_2']):
            with self.subTest(flag), self.assertRaises(SystemExit), quiet():
                scout.parse_args(['--source', 'a', '--output', 'b', *flag])

    def test_output_folder_safety(self):
        source = make_source(self.root / 'source', frames=6)
        for output in (source, source / 'inside', self.root):
            with self.subTest(output):
                self.assertTrue(scout.output_problems(source, output, True))
        occupied = self.root / 'occupied'
        occupied.mkdir()
        (occupied / 'notes.txt').write_text('x')
        message = scout.output_problems(source, occupied, False)[0]
        self.assertIn('not empty', message)
        self.assertIn('--force', message)
        self.assertEqual(scout.output_problems(source, occupied, True), [])
        for output in (occupied, source / 'out'):
            with self.subTest(output):
                self.assertEqual(self.run_main(source, output, cameras=('0',))[0], scout.EXIT_OUTPUT)
        self.assertFalse((source / 'out').exists())

    def test_fingerprint_covers_every_selected_input(self):
        source = make_source(self.root / 'source', frames=6)

        def fingerprint():
            return scout.source_fingerprint(source, 'sequence_1', (0, 1), [3, 4, 5], [3, 4, 5], False)
        base = fingerprint()
        self.assertEqual((base[1], fingerprint()), (15, base))
        for relative in ('annotations/individual_format/cam_1/image_5.txt', 'calibrations/cam_0.json',
                         'timestamps/timestamps_sequence_1.json'):
            os.utime(source / relative, ns=(7, 7))
            changed = fingerprint()
            self.assertNotEqual(changed[0], base[0], relative)
            base = changed
        (source / 'images' / 'sequence_1' / 'cam_0' / 'image_3.jpg').unlink()
        with self.assertRaisesRegex(scout.SourceError, 'Cannot stat'):
            fingerprint()


class EncodeCameraTests(ScoutCase):
    """The ffmpeg writer, with a stand-in process instead of ffmpeg."""

    def encode(self, body, source):
        script = f'import sys, time; open(sys.argv[1], "w").close(); {body}'
        with mock.patch.object(scout, 'ffmpeg_command', lambda path: [sys.executable, '-c', script, str(path)]):
            scout.encode_camera((str(source), 'sequence_1', 0, [3, 4, 5], self.root / 'cam_0.mp4'))

    def test_failures_leave_no_partial_file(self):
        source = make_source(self.root / 'source', frames=6)
        with self.assertRaisesRegex(RuntimeError, 'camera 0 with exit status 3'):
            self.encode('sys.exit(3)', source)
        self.assertEqual(list(self.root.glob('cam_0*')), [])
        (source / 'images' / 'sequence_1' / 'cam_0' / 'image_4.jpg').unlink()
        with self.assertRaisesRegex(scout.SourceError, 'Cannot read image'):
            self.encode('time.sleep(60)', source)
        self.assertEqual(list(self.root.glob('cam_0*')), [])


@needs_ffmpeg
class PreparedOutputTests(unittest.TestCase):
    """Read-only checks of the shared prepared output (frames 3 to 5 of six, cameras 0 and 1)."""

    def test_output_matches_the_manifest_and_the_frame_alignment(self):
        self.assertEqual(SHARED['code'], 0, 'the shared preparation failed')
        output = SHARED['root'] / 'out'
        manifest = json.loads((output / 'manifest.json').read_text())
        self.assertEqual(manifest['protocol'], {
            'sequence': 'sequence_1', 'split': 'test', 'calibration': 'recorded', 'gt_alignment': 'exact',
            'cameras': [0, 1], 'region': [0.0, 26.0, 0.0, 140.0], 'max_dist': 50.0, 'fps': 10,
            'frames_per_video': 3, 'first_frame': 3, 'last_frame': 5})
        hashes = {name: scout.sha256_file(SHARED['root'] / 'recorded' / name) for name in RECORDED_NAMES}
        self.assertEqual(manifest['source']['transformed_sha256'], hashes)
        lines = (output / 'ground_truth.txt').read_text().splitlines()
        self.assertEqual(sorted({int(line.split()[2]) for line in lines}), [1, 2, 3])
        self.assertEqual(lines[0].split()[:3], ['1', '7', '1'])
        self.assertEqual((manifest['annotated_frames'], manifest['counts']['ground_truth_lines']), (3, 6))
        for name in ('cam_0', 'cam_1'):
            video = manifest['videos'][name]
            self.assertEqual((video['frames'], video['pix_fmt'], video['codec'], video['fps']), (3, 'yuv420p', 'h264', '10/1'))
            self.assertEqual((video['width'], video['height']), SIZE)
        self.assertEqual(sorted(manifest['files']), sorted(scout.expected_outputs((0, 1), False)))
        for name, entry in manifest['files'].items():
            self.assertEqual(entry, {'sha256': scout.sha256_file(output / name), 'bytes': (output / name).stat().st_size}, name)


@needs_ffmpeg
class ReuseTests(ScoutCase):
    """A copy of the shared prepared output that each test may change."""

    def setUp(self):
        super().setUp()
        shutil.copytree(SHARED['root'], self.root / 'work')
        self.source, self.output, self.recorded = (self.root / 'work' / name for name in ('source', 'out', 'recorded'))
        self.use_recorded(self.recorded)

    def snapshot(self):
        return (self.output / 'manifest.json').read_bytes(), (self.output / 'ground_truth.txt').read_bytes()

    def test_matching_rerun_reuses_a_verified_video_and_rebuilds_a_tampered_one(self):
        untouched = (self.output / 'videos' / 'cam_0.mp4').stat().st_mtime_ns
        with (self.output / 'videos' / 'cam_1.mp4').open('ab') as stream:
            stream.write(b'0')
        self.assertEqual(self.run_main(self.source, self.output)[0], 0)
        self.assertEqual((self.output / 'videos' / 'cam_0.mp4').stat().st_mtime_ns, untouched)
        rebuilt = json.loads((self.output / 'manifest.json').read_text())['files']['videos/cam_1.mp4']
        self.assertEqual(scout.sha256_file(self.output / 'videos' / 'cam_1.mp4'), rebuilt['sha256'])

    def test_a_different_protocol_is_refused_and_leaves_the_outputs_untouched(self):
        before = self.snapshot()
        for flags, expected in ((['--split', 'train'], 'protocol.split: test -> train'),
                                (['--region', '0', '10', '0', '10'], 'protocol.region')):
            with self.subTest(flags):
                code, message = self.run_main(self.source, self.output, *flags)
                self.assertEqual(code, scout.EXIT_MISMATCH)
                self.assertIn(expected, message)
                self.assertEqual(self.snapshot(), before)

    def test_a_changed_transformed_file_is_refused_until_it_is_restored(self):
        path = self.recorded / 'transformed_positions.json'
        values = json.loads(path.read_text())
        values['x_cm'][0] += 1
        path.write_text(json.dumps(values))
        code, message = self.run_main(self.source, self.output)
        self.assertEqual(code, scout.EXIT_MISMATCH)
        self.assertIn('source.transformed_sha256.transformed_positions.json', message)
        values['x_cm'][0] -= 1
        path.write_text(json.dumps(values))
        self.assertEqual(self.run_main(self.source, self.output)[0], 0)

    def test_a_changed_source_is_refused_until_forced(self):
        calibration = self.source / 'calibrations' / 'cam_0.json'
        calibration.write_text(calibration.read_text() + ' ')
        image = self.source / 'images' / 'sequence_1' / 'cam_0' / 'image_4.jpg'
        cv2.imwrite(str(image), np.full((SIZE[1], SIZE[0], 3), 200, np.uint8))
        os.utime(image, ns=(1, 1))
        video = (self.output / 'videos' / 'cam_0.mp4').read_bytes()
        code, message = self.run_main(self.source, self.output)
        self.assertEqual(code, scout.EXIT_MISMATCH)
        self.assertIn('source.subset.calibration_sha256', message)
        self.assertIn('source.fingerprint', message)
        self.assertEqual((self.output / 'videos' / 'cam_0.mp4').read_bytes(), video)
        (self.output / 'ground_truth.txt').write_text('stale')
        self.assertEqual(self.run_main(self.source, self.output, '--force')[0], 0)
        self.assertEqual(len((self.output / 'ground_truth.txt').read_text().splitlines()), 6)

    def test_an_interrupted_run_keeps_the_manifest_and_reruns_without_force(self):
        before = (self.output / 'manifest.json').read_bytes()
        (self.output / 'ground_truth.txt').unlink()
        with mock.patch.object(scout, 'write_calibration', side_effect=RuntimeError('interrupted')), \
                self.assertRaisesRegex(RuntimeError, 'interrupted'):
            self.run_main(self.source, self.output)
        self.assertEqual((self.output / 'manifest.json').read_bytes(), before)
        self.assertEqual(self.run_main(self.source, self.output)[0], 0)
        self.assertEqual(len((self.output / 'ground_truth.txt').read_text().splitlines()), 6)

    def test_fewer_cameras_leave_no_stale_files_and_foreign_folders_alone(self):
        detections = self.output / 'detections' / '001' / 'gt'
        detections.mkdir(parents=True)
        (detections / 'gt.txt').write_text('keep me')
        (self.output / 'videos' / 'cam_6.mp4').write_bytes(b'old')
        (self.output / 'camInfo' / 'cam_6.yml').write_text('old')
        (self.output / 'camInfo' / 'notes.txt').write_text('mine')
        self.assertEqual(self.run_main(self.source, self.output, cameras=('0',))[0], scout.EXIT_MISMATCH)
        self.assertTrue((self.output / 'videos' / 'cam_1.mp4').exists())
        self.assertEqual(self.run_main(self.source, self.output, '--force', cameras=('0',))[0], 0)
        self.assertEqual(sorted(path.name for path in (self.output / 'videos').iterdir()), ['cam_0.mp4'])
        self.assertEqual(sorted(path.name for path in (self.output / 'camInfo').iterdir()), ['cam_0.yml', 'notes.txt'])
        self.assertEqual((detections / 'gt.txt').read_text(), 'keep me')
        manifest = json.loads((self.output / 'manifest.json').read_text())
        self.assertEqual(sorted(manifest['files']), ['camInfo/cam_0.yml', 'ground_truth.txt', 'transforms.yml', 'videos/cam_0.mp4'])


@needs_ffmpeg
class FreshRunTests(ScoutCase):
    def test_unrecorded_annotation_rows_stop_the_run(self):
        source = make_source(self.root / 'source', frames=6)
        (self.recorded / 'transformed_positions.json').write_text(
            json.dumps(encode_positions({(7, frame): (500, 1000, 0) for frame in (2, 3, 4)})))
        code, message = self.run_main(source, self.root / 'out')
        self.assertEqual(code, scout.EXIT_SOURCE)
        self.assertIn('person 7 frame 5', message)

    def test_annotated_frames_count_ground_truth_ids_not_annotation_files(self):
        scenarios = {'last frame empty': ((5,), 2, 4, 2), 'all frames empty': ((2, 3, 4, 5), 0, 0, 0)}
        for name, (empty, annotated, lines, last_id) in scenarios.items():
            with self.subTest(name):
                source = make_source(self.root / name.replace(' ', '_') / 'source', frames=6)
                for index in (0, 1):
                    for frame in empty:
                        (source / 'annotations' / 'individual_format' / f'cam_{index}' / f'image_{frame}.txt').write_text('')
                output = self.root / name.replace(' ', '_') / 'out'
                self.assertEqual(self.run_main(source, output)[0], 0)
                manifest = json.loads((output / 'manifest.json').read_text())
                self.assertEqual(manifest['annotated_frames'], annotated)
                self.assertEqual(manifest['counts'], {'ground_truth_lines': lines, 'ground_truth_last_frame_id': last_id,
                                                      'annotation_files_in_span': 3})


class TransformCalibTests(ScoutCase):
    def run_all_cameras(self, source, output, *extra):
        return self.run_main(source, output, *extra, cameras=())

    def flags(self):
        return ['--transform-calib', '--transformed-out', str(self.root / 'new')]

    def test_default_run_does_not_import_torch(self):
        source = make_source(self.root / 'source', frames=6, cameras=tuple(range(8)))
        program = (
            'import sys\n'
            f'sys.path.insert(0, {str(APP)!r})\n'
            'from utils import prepare_scout, transform_scout\n'
            f'code = prepare_scout.main(["--source", {str(source)!r}, "--output", {str(self.root / "out")!r}, "--check-only", "--no-map"])\n'
            'print(code, "torch" in sys.modules, "kornia" in sys.modules)\n')
        result = subprocess.run([sys.executable, '-c', program], capture_output=True, text=True)
        self.assertEqual(result.stdout.split()[-3:], ['0', 'False', 'False'], result.stderr)

    def test_missing_packages_and_correspondences_fail_early(self):
        for missing, expected in ((['torch', 'kornia'], 'needs torch and kornia, which are not installed'),
                                  (['kornia'], 'needs kornia, which is not installed')):
            with self.subTest(missing), mock.patch.object(transform, 'missing_packages', return_value=missing):
                code, message = self.run_all_cameras(self.root / 'absent', self.root / 'out', *self.flags())
            self.assertEqual(code, scout.EXIT_MISSING_PACKAGES)
            self.assertIn(expected, message)
            self.assertFalse((self.root / 'out').exists() or (self.root / 'new').exists())
        with mock.patch.object(transform, 'missing_packages', return_value=[]):
            code, message = self.run_all_cameras(self.root / 'absent', self.root / 'out', *self.flags(),
                                                 '--correspondences', str(self.root / 'none.json'))
        self.assertEqual(code, scout.EXIT_SOURCE)
        self.assertIn('does not exist', message)

    def test_transform_calib_arguments_are_validated(self):
        base = ['--source', 'a', '--output', 'b']
        for flags in (['--transform-calib'], ['--transform-calib', '--transformed-out', 'n', '--cameras', '0', '1'],
                      ['--transform-calib', '--transformed-out', 'n', '--max-iterations', '0']):
            with self.subTest(flags), self.assertRaises(SystemExit), quiet():
                scout.parse_args(base + flags)

    @needs_ffmpeg
    def test_recomputed_manifest(self):
        source = make_source(self.root / 'source', frames=6, cameras=tuple(range(8)))
        output, new = self.root / 'out', self.root / 'new'

        def fake(args, source_dir, cameras, count):
            return scout.load_recorded(), {'best_loss': 1.0, 'first_loss': 2.0, 'final_loss': 1.5, 'iterations': 3, 'transformed_out': str(new)}
        with mock.patch.object(transform, 'missing_packages', return_value=[]), mock.patch.object(scout, 'transform_cameras', fake):
            self.assertEqual(self.run_all_cameras(source, output, *self.flags())[0], 0)
            manifest = json.loads((output / 'manifest.json').read_text())
            self.assertEqual(manifest['protocol']['calibration'], 'recomputed')
            self.assertEqual(manifest['source']['correspondences_sha256'], scout.sha256_file(scout.CORRESPONDENCES))
            self.assertEqual(manifest['extra']['transform_calib']['best_loss'], 1.0)
            self.assertEqual(self.run_all_cameras(source, output, *self.flags())[0], 0)
            self.assertEqual(self.run_all_cameras(source, output, '--force')[0], 0)
        self.assertEqual(json.loads((output / 'manifest.json').read_text())['protocol']['calibration'], 'recorded')
        with mock.patch.object(transform, 'missing_packages', return_value=[]), mock.patch.object(scout, 'transform_cameras', fake):
            code, message = self.run_all_cameras(source, output, *self.flags())
        self.assertEqual(code, scout.EXIT_MISMATCH)
        self.assertIn('protocol.calibration: recorded -> recomputed', message)

    def test_transform_calib_folder_must_not_be_inside_the_official_data(self):
        source = make_source(self.root / 'source', frames=6, cameras=tuple(range(8)), images=False)
        with mock.patch.object(transform, 'missing_packages', return_value=[]), \
                mock.patch.object(scout, 'validate_source', return_value=[]):
            code, message = self.run_all_cameras(source, self.root / 'out', '--transform-calib', '--transformed-out', str(source / 'new'))
        self.assertEqual(code, scout.EXIT_OUTPUT)
        self.assertIn('inside the official data', message)

    def test_unusable_correspondences_and_failed_adjustment_exit_cleanly(self):
        source = make_source(self.root / 'source', frames=6, cameras=tuple(range(8)))
        for text, expected in (('{}', 'no correspondences for cameras'), ('{', 'Expecting property name')):
            with self.subTest(text):
                (self.root / 'corr.json').write_text(text)
                with mock.patch.object(transform, 'missing_packages', return_value=[]):
                    code, message = self.run_all_cameras(source, self.root / 'out', *self.flags(),
                                                         '--correspondences', str(self.root / 'corr.json'))
                self.assertEqual(code, scout.EXIT_SOURCE)
                self.assertIn(expected, message)

        def failing(*arguments):
            raise ValueError('the bundle adjustment never reached a finite loss')
        with mock.patch.object(transform, 'missing_packages', return_value=[]), mock.patch.object(scout, 'transform_cameras', failing):
            code, message = self.run_all_cameras(source, self.root / 'out', *self.flags())
        self.assertEqual(code, scout.EXIT_SOURCE)
        self.assertIn('never reached a finite loss', message)

    def test_written_transformed_files_match_the_shipped_format(self):
        text = (APP / 'assets' / 'scout_transformed' / 'transformed_positions.json').read_text()
        keys, meters = scout.decode_positions(json.loads(text))
        persons, frames = keys//scout.KEY_SCALE, keys % scout.KEY_SCALE
        centimeters = np.rint(meters*100).astype(int)
        again = transform.encode_positions(persons, frames, centimeters[:, 0], centimeters[:, 1], centimeters[:, 2])
        self.assertEqual(json.dumps(again, separators=(',', ':')), text)
        cameras = {str(index): {'K': MATRIX, 'R': np.eye(3).tolist(), 'T': [0.0, 0.0, 1.0]} for index in range(8)}
        xy = np.array([[1.234, 2.0], [5.0, 7.0], [3.0, 9.0], [2.0, 4.0]])
        losses = {'total_loss': 1.0, 'feet_rmse_pixels': 1.0, 'head_rmse_pixels': 1.0, 'correspondences_rmse_pixels': 1.0}
        transform.write_recorded_files(self.root / 'written', cameras, [1, 2, 3, 4], [5, 5, 6, 6], xy,
                                           np.array([0.0, 0.3, 0.0, 0.0]), xy + 0.5, losses, REGION, RECORDED_NAMES)
        np.testing.assert_allclose(scout.load_recorded(self.root / 'written').lookup([2], [5]), [[5.0, 7.0, 0.3]])

    @unittest.skipIf(transform.missing_packages(), 'torch and kornia are not installed')
    def test_bundle_adjustment_decreases_the_loss_and_writes_loadable_json(self):
        rng = np.random.default_rng(0)
        official, matrices, annotations, correspondences = {}, {}, {}, {}
        base = {(frame, person): (20 + 2*person + rng.normal(0, 0.05), 60 + frame + rng.normal(0, 0.05))
                for frame in range(5) for person in (1, 2, 3)}
        for camera_index in range(8):
            matrix = np.array([[800.0, 0, 960], [0, 800.0, 540], [0, 0, 1.0]])
            rotation, translation = camera(np.array([20.0 + 3*camera_index, 20.0, 4.0]), np.array([20.0, 60.0, 0.0]))
            translation = translation.reshape(3, 1)
            official[camera_index] = {'K': matrix, 'R': rotation, 'T': translation, 'dist': np.zeros(5)}
            matrices[camera_index] = matrix
            annotations[camera_index] = np.array([[frame, person, 100, 100, 20, 40, *base[(frame, person)], 0.0]
                                                  for frame in range(5) for person in (1, 2, 3)])
            pairs = []
            for gx in range(6):
                for gy in range(2):
                    point = np.array([5.0 + gx*2, 55.0 + gy*8, 0.0])
                    pixel = scout.project_points(point[None], matrix, rotation, translation)[0] + rng.normal(0, 2.0, 2)
                    pairs.append({'camera_point': pixel.tolist(), 'mesh_point': [(14 - point[0])*100 + 800, point[1]*100 + 800]})
            correspondences[camera_index] = pairs
        result = transform.transform_calibration(official, matrices, annotations, correspondences, self.root / 'ba', REGION, RECORDED_NAMES,
                                           1.34, (1920, 1080), 120, save_interval=50, device='cpu')
        self.assertLess(result['best_loss'], result['first_loss'])
        loaded = scout.load_recorded(self.root / 'ba')
        self.assertEqual((sorted(loaded.cameras), len(loaded.keys)), (list(range(8)), 15))


if __name__ == '__main__':
    unittest.main()
