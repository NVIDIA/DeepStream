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
"""Check the WILDTRACK converter output, manifest protocol, reuse safety and input errors on tiny fixtures."""

import io
import os
import cv2
import yaml
import json
import shutil
import unittest
import tempfile
import subprocess
import contextlib

import numpy as np

from pathlib import Path
from unittest import mock
from utils import prepare_wildtrack as prep

CAMERAS = ('CamA', 'CamB')


def write_xml(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'<?xml version="1.0"?>\n<opencv_storage>\n{body}\n</opencv_storage>\n')


def matrix_xml(values):
    return ('<camera_matrix type_id="opencv-matrix"><rows>3</rows><cols>3</cols><dt>d</dt>'
            f'<data>{" ".join(map(str, values))}</data></camera_matrix>')


def write_calibration(root, name, rvec=(0, 0, 0), tvec=(100, 200, 300), dist=(0, 0, 0, 0, 0)):
    K = (50, 0, 32, 0, 50, 24, 0, 0, 1)
    for kind, distortion in (('intrinsic_zero', (0, 0, 0, 0, 0)), ('intrinsic_original', dist)):
        write_xml(root / 'calibrations' / kind / f'intr_{name}.xml', matrix_xml(K) +
                  '<distortion_coefficients type_id="opencv-matrix"><rows>5</rows><cols>1</cols><dt>d</dt>'
                  f'<data>\n {" ".join(map(str, distortion))}</data></distortion_coefficients>')
    write_xml(root / 'calibrations' / 'extrinsic' / f'extr_{name}.xml',
              f'<rvec>\n {" ".join(map(str, rvec))}\n</rvec>\n<tvec>\n {" ".join(map(str, tvec))}\n</tvec>')


def person(person_id, position_id, boxes):
    views = []
    for number, box in enumerate(boxes):
        xmin, ymin, xmax, ymax = box if box else (-1, -1, -1, -1)
        views.append({'viewNum': number, 'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax})
    return {'personID': person_id, 'positionID': position_id, 'views': views}


def build_source(root, frames=6, videos=False):
    for index, name in enumerate(CAMERAS):
        write_calibration(root, name)
        for frame in prep.annotated_frames(frames):
            path = root / 'Image_subsets' / f'C{index + 1}' / f'{frame:08d}.png'
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), np.full((48, 64, 3), frame % 251, np.uint8))
        if videos:
            (root / f'cam{index + 1}.mp4').write_bytes(b'')
    folder = root / 'annotations_positions'
    folder.mkdir(parents=True, exist_ok=True)
    for number, frame in enumerate(prep.annotated_frames(frames)):
        people = [person(7, 399 + 480 * 833, [(10, 20, 30, 60), None]),
                  person(9, 0, [(-5, 1, 12, 40), (1, 2, 3, 4)])]
        (folder / f'{frame:08d}.json').write_text(json.dumps(people))


def level(index):
    return 3 * (index % 80) + 5


def synthetic_video(path, frames, size=(64, 48)):
    """Constant gray frames whose level encodes the frame index (period 80), lossless at 59.94 fps."""
    command = ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'gray', '-s', f'{size[0]}x{size[1]}',
               '-r', '60000/1001', '-i', '-', '-c:v', 'libx264', '-qp', '0', '-pix_fmt', 'yuv444p', str(path)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    for index in range(frames):
        process.stdin.write(np.full((size[1], size[0]), level(index), np.uint8).tobytes())
    process.stdin.close()
    process.wait()


def evaluator():
    """Import the evaluator lazily so this module works without trackeval."""
    try:
        from utils import evaluate_tracking
    except ImportError as error:
        raise unittest.SkipTest(f'evaluate_tracking is unavailable: {error}')
    return evaluate_tracking


def use_cameras(add_cleanup, cameras=CAMERAS):
    """Replace the seven official cameras by the fixture cameras until the cleanup runs."""
    patcher = mock.patch.object(prep, 'CAMERAS', cameras)
    patcher.start()
    add_cleanup(patcher.stop)


def prepare_fixture(source, output, force=False, videos=False):
    return prep.prepare(source, output, frames=6, force=force, videos=videos, log=lambda *_: None)


class TempCase(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)


class ParsingTests(TempCase):

    def test_calibration_is_read(self):
        write_calibration(self.root, 'CamA', rvec=(0.1, -0.2, 0.3), tvec=(1, 2, 3), dist=(-0.4, 0.5, 0.01, 0.002, -0.6))
        calibration = prep.load_calibration(self.root, 'CamA')
        np.testing.assert_allclose(calibration['rvec'], (0.1, -0.2, 0.3))
        np.testing.assert_allclose(calibration['tvec'], (1, 2, 3))
        np.testing.assert_allclose(calibration['dist'], (-0.4, 0.5, 0.01, 0.002, -0.6))
        np.testing.assert_allclose(calibration['K'], [[50, 0, 32], [0, 50, 24], [0, 0, 1]])

    def test_bad_calibration_is_reported(self):
        path = self.root / 'bad.xml'
        write_xml(path, '<other>1 2 3</other>')
        with self.assertRaisesRegex(prep.SourceError, 'no rvec node'):
            prep.read_xml_numbers(path, 'rvec')
        path.write_text('<opencv_storage>')
        with self.assertRaisesRegex(prep.SourceError, 'not valid XML'):
            prep.read_xml_numbers(path, 'rvec')
        write_calibration(self.root, 'CamA')
        write_xml(self.root / 'calibrations' / 'intrinsic_original' / 'intr_CamA.xml',
                  matrix_xml((60, 0, 32, 0, 50, 24, 0, 0, 1)) + '<distortion_coefficients><data>0 0 0 0 0</data></distortion_coefficients>')
        with self.assertRaisesRegex(prep.SourceError, 'different camera matrices'):
            prep.load_calibration(self.root, 'CamA')


class ProjectionTests(unittest.TestCase):
    def test_camera_info_is_readable_by_evaluator_geometry(self):
        K = np.array([[50.0, 0, 32], [0, 50, 24], [0, 0, 1]])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / '000.yml'
            path.write_text(prep.camera_info_text(prep.projection_matrix(K, (0, 0, 0), (100, 200, 300))))
            center, height = evaluator().camera_geometry(path)
        # Camera center in sample coordinates: meters (-1, -2, -3) mapped back through 2.5 x + 3 and z.
        np.testing.assert_allclose(center, [(-1 - 3) / 2.5, (-2 - 9) / 2.5, -3.0])
        self.assertEqual(height, 1.7)

    def test_full_protocol_satisfies_evaluator_settings(self):
        protocol = prep.protocol_description(prep.DEFAULT_FRAMES)
        settings = evaluator().DATASETS['wildtrack']
        for key, value in settings['protocol'].items():
            self.assertEqual(protocol[key], value, key)
        self.assertEqual(protocol['annotated_frames'], settings['annotated_frames'])


class GroundTruthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        build_source(cls.root)

    def setUp(self):
        use_cameras(self.addCleanup)

    def test_rows_skip_invisible_views_and_number_frames_from_one(self):
        rows = [line.split() for line in prep.ground_truth_lines(self.root, 6)]
        self.assertEqual([row[:7] for row in rows[:3]],
                         [['0', '7', '1', '10', '20', '20', '40'], ['0', '9', '1', '-5', '1', '17', '39'],
                          ['1', '9', '1', '1', '2', '2', '2']])
        self.assertEqual(len(rows), 2 * 3)
        self.assertEqual(rows[3][:3], ['0', '7', '2'])
        self.assertAlmostEqual(float(rows[0][7]), 1.59)
        self.assertAlmostEqual(float(rows[0][8]), 1.13)
        use_cameras(self.addCleanup, ('CamA',))
        self.assertEqual({line.split()[0] for line in prep.ground_truth_lines(self.root, 1)}, {'0'})

    def test_bad_json_is_reported(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / 'annotations_positions').mkdir()
            (Path(folder) / 'annotations_positions' / '00001800.json').write_text('[')
            with self.assertRaisesRegex(prep.SourceError, 'not valid JSON'):
                prep.ground_truth_lines(folder, 1)

    def test_rows_work_with_evaluator_reader(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'ground_truth.txt'
            path.write_text('\n'.join(prep.ground_truth_lines(self.root, 6)) + '\n')
            geometry = {0: (np.zeros(3), 1.7), 1: (np.zeros(3), 1.7)}
            objects = evaluator().read_ground_truth(path, 2, 0, geometry, None)
        self.assertEqual(sorted({key[0] for key in objects}), [1, 2])
        self.assertEqual(sorted({key[1] for key in objects}), [7, 9])


class PreparedOutputTests(unittest.TestCase):
    """Read-only checks on one conversion shared by all tests."""

    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        patcher = mock.patch.object(prep, 'CAMERAS', CAMERAS)
        patcher.start()
        cls.addClassCleanup(patcher.stop)
        base = Path(cls.temporary.name)
        cls.root, cls.first, cls.second = base / 'source', base / 'one', base / 'two'
        build_source(cls.root)
        cls.manifest = prepare_fixture(cls.root, cls.first)
        prepare_fixture(cls.root, cls.second)
        cls.rerun = prepare_fixture(cls.root, cls.first)

    def test_output_is_deterministic_and_rerun_without_force_matches(self):
        self.assertEqual((self.first / 'manifest.json').read_bytes(), (self.second / 'manifest.json').read_bytes())
        self.assertEqual(self.manifest, self.rerun)
        self.assertEqual(self.manifest['ground_truth_rows'], 6)
        self.assertEqual(self.manifest['source']['official_frames'], [1800, 1805])
        for name in ('ground_truth.txt', 'transforms.yml', 'map.png', 'camInfo/000.yml', 'camInfo/001.yml'):
            self.assertEqual(self.manifest['files'][name]['bytes'], (self.first / name).stat().st_size)

    def test_manifest_passes_evaluator_verification(self):
        settings = {'annotated_frames': 2, 'protocol': {'first_official_frame': 1800, 'frames': 6, 'cameras': 2}}
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'out'
            shutil.copytree(self.first, output)
            self.assertEqual(evaluator().verify_dataset_manifest(output, 'wildtrack', settings)['annotated_frames'], 2)
            with self.assertRaisesRegex(ValueError, 'needs frames'):
                evaluator().verify_dataset_manifest(output, 'wildtrack', {**settings, 'protocol': {'frames': 196}})
            with (output / 'ground_truth.txt').open('a') as stream:
                stream.write('0 1 1 0 0 1 1 0.0 0.0\n')
            with self.assertRaisesRegex(ValueError, 'GT hash'):
                evaluator().verify_dataset_manifest(output, 'wildtrack', settings)


class LayoutTests(TempCase):
    def setUp(self):
        super().setUp()
        use_cameras(self.addCleanup)
        self.root, self.out = self.root / 'source', self.root / 'out'
        build_source(self.root)

    def run_prepare(self, output=None, **keywords):
        return prepare_fixture(self.root, output or self.out, **keywords)

    def test_missing_files_name_the_file_and_the_download(self):
        (self.root / 'calibrations' / 'extrinsic' / 'extr_CamB.xml').unlink()
        shutil.rmtree(self.root / 'Image_subsets' / 'C1')
        with self.assertRaises(prep.SourceError) as caught:
            prep.validate_source(self.root, 6, with_videos=False)
        message = str(caught.exception)
        for text in ('calibrations/extrinsic/extr_CamB.xml', 'Image_subsets/C1/00001800.png', 'Wildtrack_dataset_full.zip'):
            self.assertIn(text, message)
        with self.assertRaises(prep.SourceError) as caught:
            prep.validate_source(self.root, 6, with_videos=True)
        self.assertIn('cam1.mp4', str(caught.exception))
        self.assertIn(prep.DRIVE_IDS[0], str(caught.exception))

    def test_source_and_output_default_to_the_datasets_folders(self):
        with mock.patch.object(prep, 'prepare') as prepare:
            self.assertEqual(prep.main([]), 0)
        self.assertEqual(prepare.call_args.args[:2], (Path('datasets/raw/wildtrack'), Path('datasets/wildtrack')))

    def test_main_reports_errors_with_exit_code_1(self):
        errors = io.StringIO()
        with contextlib.redirect_stderr(errors):
            self.assertEqual(prep.main(['--source', str(self.root / 'none'), '--output', str(self.out)]), 1)
        self.assertIn('Source directory not found', errors.getvalue())

    def test_foreign_or_overlapping_output_is_refused(self):
        for output in (self.root, self.root / 'inside', self.root.parent):
            with self.assertRaisesRegex(prep.SourceError, 'overlap'):
                self.run_prepare(output, force=True)
        self.assertFalse((self.root / 'inside').exists())
        self.out.mkdir()
        (self.out / 'stale.txt').write_text('x')
        with self.assertRaisesRegex(prep.SourceError, 'no valid manifest.json.*--force'):
            self.run_prepare()
        self.run_prepare(force=True)

    def test_reuse_is_refused_when_protocol_or_source_differs(self):
        self.run_prepare()
        before = (self.out / 'manifest.json').read_bytes()
        with self.assertRaisesRegex(prep.SourceError, 'protocol.frames'):
            prep.prepare(self.root, self.out, frames=1, videos=False, log=lambda *_: None)
        self.assertEqual((self.out / 'manifest.json').read_bytes(), before)
        png = self.root / 'Image_subsets' / 'C2' / '00001805.png'
        stamp = png.stat()
        os.utime(png, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 10 ** 9))
        with self.assertRaisesRegex(prep.SourceError, 'source.fingerprint'):
            self.run_prepare()
        self.run_prepare(force=True)
        (self.root / 'annotations_positions' / '00001800.json').write_text(json.dumps([person(7, 0, [(1, 2, 3, 4), None])]))
        with self.assertRaisesRegex(prep.SourceError, 'source.text_inputs_sha256'):
            self.run_prepare()

    def test_manifest_lists_only_generated_files_and_stale_cameras_are_removed(self):
        self.run_prepare()
        (self.out / 'videos' / '000.mp4.part.mp4').write_bytes(b'junk')
        kept = ('detections/000.txt', 'videos/notes.txt', 'camInfo/000.yml.bak', 'readme.txt')
        stale = ('videos/007.mp4', 'camInfo/007.yml', 'videos/002.mp4', 'camInfo/002.yml')
        for name in kept:
            (self.out / name).parent.mkdir(exist_ok=True)
            (self.out / name).write_text('keep')
        for force in (False, True):
            for name in stale:
                (self.out / name).write_text('stale')
            manifest = self.run_prepare(force=force)
            self.assertEqual(sorted(manifest['files']), [name for name in prep.expected_outputs(2) if not name.startswith('videos/')])
            for name in stale:
                self.assertFalse((self.out / name).exists(), name)
            for name in kept:
                self.assertEqual((self.out / name).read_text(), 'keep')
            self.assertFalse((self.out / 'videos' / '000.mp4.part.mp4').exists())


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'ffmpeg and ffprobe are required')
class ReuseTests(TempCase):
    def setUp(self):
        super().setUp()
        self.root, self.output = self.root / 'source', self.root / 'output'
        build_source(self.root, videos=True)
        for name in ('cam1.mp4', 'cam2.mp4'):
            synthetic_video(self.root / name, 200)

    def run_prepare(self, **keywords):
        with mock.patch.object(prep, 'CAMERAS', CAMERAS), mock.patch.object(prep, 'find_alignment', lambda *a, **k: 100):
            return prep.prepare(self.root, self.output, frames=6, log=lambda *_: None, **keywords)

    def test_only_verified_videos_are_reused(self):
        with mock.patch.object(prep, 'write_video', wraps=prep.write_video) as writer:
            first = self.run_prepare()
            self.assertEqual(writer.call_count, 2)
            second = self.run_prepare()
            self.assertEqual(writer.call_count, 2)
            self.assertEqual(first, second)
            with (self.output / 'videos' / '001.mp4').open('ab') as stream:
                stream.write(b'x')
            third = self.run_prepare()
            self.assertEqual(writer.call_count, 3)
            self.assertEqual(first, third)
            self.run_prepare(force=True)
            self.assertEqual(writer.call_count, 5)

    def test_replaced_raw_video_is_refused_not_reused(self):
        self.run_prepare()
        synthetic_video(self.root / 'cam2.mp4', 210)
        with self.assertRaisesRegex(prep.SourceError, 'source.fingerprint'):
            self.run_prepare()
        self.run_prepare(force=True)

    def test_wrong_stream_properties_are_not_reused(self):
        self.run_prepare()
        recorded = json.loads((self.output / 'manifest.json').read_text())['files']['videos/000.mp4']
        expected = prep.expected_video(self.root, 0, 6)
        path = self.output / 'videos' / '000.mp4'
        self.assertTrue(prep.reusable_video(path, recorded, expected))
        self.assertFalse(prep.reusable_video(path, recorded, {**expected, 'frames': '7'}))
        self.assertFalse(prep.reusable_video(path, {**recorded, 'bytes': recorded['bytes'] + 1}, expected))
        self.assertFalse(prep.reusable_video(path, {**recorded, 'sha256': '0' * 64}, expected))
        path.write_bytes(b'not a video')
        self.assertIsNone(prep.probe_video(path))


class MapTests(unittest.TestCase):
    def test_map_has_grid_at_world_origin_and_labels(self):
        image = prep.render_map()
        self.assertEqual(image.shape, (860, 860, 3))
        x, y, _ = prep.map_transform() @ np.array([0.0, 0.0, 1.0])
        self.assertTrue((image[int(y), 300] != image[0, 0]).all())
        self.assertTrue((image[300, int(x)] != image[0, 0]).all())
        self.assertTrue((image[-30:] == 50).any())


@unittest.skipUnless(shutil.which('ffmpeg'), 'ffmpeg is required')
class VideoTests(TempCase):
    def test_decode_returns_requested_frame_numbers(self):
        synthetic_video(self.root / 'cam1.mp4', 200)
        frames = list(prep.ffmpeg_decode(self.root / 'cam1.mp4', 37, 3))
        self.assertEqual([round(float(frame.mean())) for frame in frames], [level(37), level(38), level(39)])
        small = list(prep.ffmpeg_decode(self.root / 'cam1.mp4', 120, 2, (32, 24), gray=True))
        self.assertEqual([frame.shape for frame in small], [(24, 32)] * 2)
        self.assertEqual([round(float(frame.mean())) for frame in small], [level(120), level(121)])

    def test_write_video_mixes_png_and_raw_frames(self):
        build_source(self.root, frames=6)
        synthetic_video(self.root / 'cam1.mp4', 200)
        calibration = prep.load_calibration(self.root, 'CamA')
        output = self.root / 'out.mp4'
        prep.write_video(self.root, 0, calibration, 100, 6, output)
        capture = cv2.VideoCapture(str(output))
        means = [float(frame.mean()) for ok, frame in iter(capture.read, (False, None)) if ok]
        self.assertEqual(len(means), 6)
        expected = [1800 % 251, level(106), level(112), level(118), level(124), 1805 % 251]
        for value, wanted in zip(means, expected):
            self.assertAlmostEqual(value, wanted, delta=8)  # lossy encode of flat frames
        self.assertEqual(capture.get(cv2.CAP_PROP_FPS), 25.0)


if __name__ == '__main__':
    unittest.main()
