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
"""Check public metric scoring, aggregation, frame mapping, and provenance."""

import io
import csv
import yaml
import json
import unittest
import tempfile
import contextlib

import numpy as np

from pathlib import Path
from unittest import mock
from types import SimpleNamespace
from utils import evaluate_tracking as evaluation


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_metric_scoring(self):
        truth = {(1, 7): (0, 0, 1, 2), (2, 7): (0, 0, 2, 2)}
        origin = {(1, 7): (0, 0, 0, 0)}
        switched = {(1, 100): truth[(1, 7)], (2, 101): truth[(2, 7)]}
        extras = {(1, 100): (0, 0, 0, 0), (1, 101): (0, 0, 10, 0), (1, 102): (0, 0, 20, 0)}
        cases = [
            ('perfect with empty frame', truth, truth, 3,
             {**dict.fromkeys(evaluation.METRICS[:5], 100), 'CLR_TP': 2, 'CLR_FP': 0, 'CLR_FN': 0, 'IDSW': 0}),
            ('misses', truth, {}, 2, {'HOTA': 0, 'DetA': 0, 'AssA': 0, 'CLR_TP': 0, 'CLR_FN': 2}),
            ('ID switch', truth, switched, 2, {'DetA': 100, 'AssA': 50, 'IDSW': 1, 'IDF1': 50, 'MOTA': 50}),
            ('negative MOTA', origin, extras, 1, {'DetA': 100/3, 'AssA': 100, 'CLR_FP': 2, 'MOTA': -100}),
            ('CLEAR distance threshold', origin, {(1, 8): (0, 0, 1.1, 0)}, 1, {'CLR_TP': 0, 'CLR_FP': 1, 'CLR_FN': 1}),
        ]
        for name, gt, prediction, frames, expected in cases:
            with self.subTest(case=name):
                result = evaluation.score(gt, prediction, frames, zero_distance=2)
                self.assertEqual(set(result), set(evaluation.METRICS))
                self.assertEqual(result['GT_Dets'], len(gt))
                self.assertEqual(result['GT_Dets'], result['CLR_TP'] + result['CLR_FN'])
                for key, value in expected.items():
                    self.assertAlmostEqual(result[key], value, msg=key)

    def test_only_requested_aggregation_changes_positions(self):
        views = {(1, 7): [(1, 0, 1, 1), (2, 1, 3, 3), (3, 2, 100, 100)]}
        self.assertEqual(evaluation.fuse(views, 'closest')[(1, 7)][2:], (1, 1))
        self.assertEqual(evaluation.fuse(views, 'median')[(1, 7)][2:], (3, 3))
        self.assertAlmostEqual(evaluation.fuse(views, 'mean')[(1, 7)][2], 104/3)
        self.assertEqual(set(evaluation.fuse(views, 'mean')), {(1, 7)})
        geometry = {0: (np.array([0, 0, -10]), 1.7)}
        objects = {}
        for x in (1, 3):
            evaluation.add_observation(objects, 1, 7, 0, x, 1, geometry, None)
        self.assertEqual(len(objects[(1, 7)]), 2)
        self.assertEqual(evaluation.fuse(objects, 'mean'), {(1, 7): (0, -1, 2, 1)})

    def test_roi_and_nonfinite_positions(self):
        objects = {}
        geometry = {0: (np.array([0, 0, -10]), 1.7)}
        evaluation.add_observation(objects, 1, 7, 0, -1, 1, geometry, [0, 2, 0, 2])
        self.assertEqual(objects, {})
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            evaluation.add_observation(objects, 1, 7, 0, float('nan'), 1, geometry, None)

    def test_per_view_distance_cutoff_uses_height_and_includes_boundary(self):
        geometry = {0: (np.array([0, 0, -3]), 2.0), 1: (np.array([100, 0, -3]), 2.0)}
        objects = {}
        evaluation.add_observation(objects, 1, 7, 0, 3, 0, geometry, None, max_dist=5)
        evaluation.add_observation(objects, 1, 8, 0, 4, 0, geometry, None, max_dist=5)
        evaluation.add_observation(objects, 1, 7, 1, 4, 0, geometry, None, max_dist=5)
        self.assertEqual(set(objects), {(1, 7)})
        self.assertEqual(objects[(1, 7)][0][0], 25)
        self.assertEqual(len(objects[(1, 7)]), 1)
        self.assertEqual(evaluation.fuse(objects, 'mean')[(1, 7)][2:], (3, 0))

    def fixture(self):
        dataset = self.root / 'dataset'
        experiment = self.root / 'experiment'
        (dataset / 'camInfo').mkdir(parents=True)
        (experiment / 'trajDumps').mkdir(parents=True)
        camera_path = dataset / 'camInfo/000.yml'
        camera_path.write_text(yaml.safe_dump({'projectionMatrix_3x4_w2p': [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 10],
                                               'modelInfo': {'height': 1.7}}))
        truth = dataset / 'ground_truth.txt'
        truth.write_text('0 7 1 0 0 1 1 0 0\n0 7 2 0 0 1 1 1 0\n')
        (experiment / 'trajDumps/trajDump_Stream_0.txt').write_text(
            '1,900,0,0,1,1,1,0,0\n2,900,0,0,1,1,1,100,100\n6,900,0,0,1,1,1,1,0\n')
        (experiment / 'config_tracker.yml').write_text(yaml.safe_dump({
            'ObjectModelProjection': {'cameraModelFilepath': ['/workspace/inputs/camInfo/000.yml']},
            'TrajectoryManagement': {'trajectoryDumpFileName': '/workspace/experiments/trajDumps/trajDump_Stream_'},
        }))
        (experiment / 'config_deepstream.txt').write_text('[application]\nkitti-track-output-dir=tracker-kitti-dump\n')
        (experiment / 'tracker-kitti-dump').mkdir()
        for frame in range(6):
            (experiment / 'tracker-kitti-dump' / f'00_000_{frame:06d}.txt').touch()
        (experiment / 'run_deepstream_app.log').write_text('App run successful\n')
        settings = {'ground_truth_sha256': evaluation.sha256(truth),
                    'ground_truth_source': 'fixture', 'camera_id_base': 0, 'annotated_frames': 2,
                    'tracker_frame_start': 1, 'tracker_frame_step': 5, 'region': None,
                    'max_dist': 30.0}
        args = SimpleNamespace(dataset='wildtrack', dataset_dir=dataset, experiment=experiment,
                               output_dir=None, zero_distance=2, max_dist=None, region=None, fusion_method='closest', frames=None)
        return settings, args

    def test_cli_workflow_uses_annotation_frame_map_and_records_inputs(self):
        settings, args = self.fixture()
        with mock.patch.dict(evaluation.DATASETS, {'wildtrack': settings}), contextlib.redirect_stdout(io.StringIO()) as output:
            result = evaluation.evaluate(args)
        self.assertEqual(result['tracker_frames_evaluated'], [1, 6])
        self.assertEqual(result['metrics']['MOTA'], 100)
        self.assertEqual(result['metrics']['IDF1'], 100)
        self.assertFalse(result['partial_sequence'])
        text = (args.experiment / 'evaluation/predictions_mot.txt').read_text()
        self.assertIn('900', text)
        self.assertEqual(len(text.splitlines()), 2)
        self.assertTrue(result['trajectory_sha256'])
        self.assertFalse((args.experiment / 'run_manifest.json').exists())
        self.assertTrue(result['completion_verified_from_log'])
        self.assertEqual(result['max_dist'], 30)
        directory = args.experiment / 'evaluation'
        self.assertEqual(json.loads((directory / 'metrics.json').read_text()), result)
        rows = list(csv.reader(io.StringIO((directory / 'metrics.csv').read_text())))
        self.assertEqual(rows, [list(evaluation.METRICS), ['100.00']*5 + ['2', '2', '0', '0', '0']])
        lines = output.getvalue().splitlines()
        header = next(index for index, line in enumerate(lines) if line.split() == list(evaluation.METRICS))
        self.assertEqual(lines[header+2].split(), rows[1])

    def test_reevaluation_replaces_outputs_and_preserves_other_files(self):
        settings, args = self.fixture()
        for output in (None, self.root / 'custom_evaluation'):
            with self.subTest(output=output), mock.patch.dict(evaluation.DATASETS, {'wildtrack': settings}):
                args.output_dir = output
                evaluation.evaluate(args)
                directory = output or args.experiment / 'evaluation'
                expected = {path.name: path.read_bytes() for path in directory.iterdir()}
                (directory / 'notes.txt').write_text('Keep this file\n')
                for name in expected:
                    (directory / name).write_text('stale output\n')
                evaluation.evaluate(args)
                self.assertEqual({name: (directory / name).read_bytes() for name in expected}, expected)
                self.assertEqual((directory / 'notes.txt').read_text(), 'Keep this file\n')

    def test_distance_cutoff_is_symmetric_for_gt_and_predictions(self):
        _, args = self.fixture()
        truth = args.dataset_dir / 'ground_truth.txt'
        truth.write_text(truth.read_text() + '0 8 2 0 0 1 1 100 100\n')
        prediction = args.experiment / 'trajDumps/trajDump_Stream_0.txt'
        prediction.write_text(prediction.read_text() + '6,901,0,0,1,1,1,100,100\n')
        geometry = {0: evaluation.camera_geometry(args.dataset_dir / 'camInfo/000.yml')}
        for limit, count in ((30, 2), (float('inf'), 3)):
            with self.subTest(max_dist=limit):
                gt = evaluation.read_ground_truth(truth, 2, 0, geometry, None, max_dist=limit)
                pred = evaluation.read_predictions([prediction], {1: 1, 6: 2}, geometry, None, max_dist=limit)
                self.assertEqual((len(gt), len(pred)), (count, count))

    def test_cli_distance_defaults_and_overrides(self):
        self.assertEqual(evaluation.DATASETS['scout']['max_dist'], 30)
        self.assertEqual(evaluation.DATASETS['wildtrack']['max_dist'], 30)
        settings, args = self.fixture()
        for value, expected in ((None, 30), ('50', 50), ('inf', 'inf')):
            directory = self.root / f'evaluation_{value}'
            argv = ['evaluate_tracking', '--dataset', 'wildtrack', '--experiment', str(args.experiment),
                    '--dataset-dir', str(args.dataset_dir), '--output-dir', str(directory)]
            if value is not None:
                argv += ['--max-dist', value]
            with self.subTest(max_dist=value), mock.patch('sys.argv', argv), mock.patch.dict(evaluation.DATASETS, {'wildtrack': settings}):
                evaluation.main()
                self.assertEqual(json.loads((directory / 'metrics.json').read_text())['max_dist'], expected)

    def test_invalid_distance_cutoffs_are_rejected(self):
        settings, args = self.fixture()
        for limit in (0, -1, float('nan'), float('-inf')):
            with self.subTest(max_dist=limit), mock.patch.dict(evaluation.DATASETS, {'wildtrack': settings}):
                args.max_dist = limit
                with self.assertRaisesRegex(ValueError, '--max-dist must be positive or inf'):
                    evaluation.evaluate(args)

    def test_missing_frame_coverage_requires_explicit_count(self):
        settings, args = self.fixture()
        for path in (args.experiment / 'tracker-kitti-dump').glob('*.txt'):
            path.unlink()
        with mock.patch.dict(evaluation.DATASETS, {'wildtrack': settings}):
            with self.assertRaisesRegex(ValueError, 'Pass --frames'):
                evaluation.evaluate(args)
            args.frames = 6
            result = evaluation.evaluate(args)
        self.assertEqual(result['metrics']['MOTA'], 100)

    def test_incomplete_frame_coverage_is_rejected(self):
        settings, args = self.fixture()
        (args.experiment / 'tracker-kitti-dump/00_000_000002.txt').unlink()
        with mock.patch.dict(evaluation.DATASETS, {'wildtrack': settings}), self.assertRaisesRegex(ValueError, 'missing frames'):
            evaluation.evaluate(args)

    def test_failed_run_or_wrong_gt_is_rejected(self):
        settings, args = self.fixture()
        cases = [(args.experiment / 'run_deepstream_app.log', 'App run failed\n', 'App run successful'),
                 (args.dataset_dir / 'ground_truth.txt', '0 7 1 0 0 1 1 99 99\n', 'GT hash')]
        for path, content, error in cases:
            original = path.read_text()
            path.write_text(content)
            with self.subTest(path=path.name), mock.patch.dict(evaluation.DATASETS, {'wildtrack': settings}), self.assertRaisesRegex(ValueError, error):
                evaluation.evaluate(args)
            path.write_text(original)

    def test_missing_experiment_has_actionable_error(self):
        settings, args = self.fixture()
        args.experiment = self.root / 'missing_run'
        with mock.patch.dict(evaluation.DATASETS, {'wildtrack': settings}), self.assertRaisesRegex(ValueError, 'Experiment not found:.*completed run directory'):
            evaluation.evaluate(args)


if __name__ == '__main__':
    unittest.main()
