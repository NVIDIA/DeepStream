# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise budget, isolation, adapters and scoring without GPU workloads."""

import io
import os
import sys
import json
import time
import signal
import unittest
import tempfile
import contextlib
import subprocess
import importlib.util

from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

SKILL = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, SKILL / 'scripts' / f'{name}.py')
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


campaign = module('campaign')
evaluation = module('evaluate')


class HarnessTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
        return path

    def ready(self, approved=True):
        self.write('state.json', json.dumps({'status': 'ready', 'deadline': time.time() + 60 if approved else None}))
        self.write('inputs.json', '{}')
        self.write('dataset.json', '{}')
        self.write('config.yml', 'ReID:\n  reidType: 0\n')
        self.write('job.json', json.dumps({'cwd': str(self.root), 'run': [sys.executable, '-c',
                   'import sys; print("App run successful")', '{config}'], 'timeout_seconds': 5}))
        return SimpleNamespace(campaign=str(self.root), job=str(self.root / 'job.json'),
                               config=str(self.root / 'config.yml'), name='basic-r1',
                               hypothesis='CPU harness test', gpu_lock=str(self.root / 'gpu.lock'))

    def test_camera_minutes_and_approval_are_not_runtime_estimates(self):
        self.assertEqual(campaign.proposed_seconds([{'duration_seconds': 120}] * 12), 86400)
        self.assertEqual(campaign.proposed_seconds([{'duration_seconds': 60}, {'duration_seconds': 180}]), 14400)
        self.write('state.json', '{"deadline": null}')
        args = SimpleNamespace(campaign=str(self.root), hours=24, note='User accepted 24 hours')
        with contextlib.redirect_stdout(io.StringIO()):
            campaign.approve(args)
        state = campaign.read_json(self.root / 'state.json')
        self.assertEqual(state['approved_seconds'], 86400)
        self.assertAlmostEqual(state['deadline'] - state['approved_at'], 86400, places=2)
        with self.assertRaisesRegex(ValueError, 'already approved'):
            campaign.approve(args)
        self.assertEqual(state, campaign.read_json(self.root / 'state.json'))
        self.write('child/state.json', '{"deadline": null}')
        deadline = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(time.time() + 120))
        args = SimpleNamespace(campaign=self.root / 'child', hours=None, deadline=deadline,
                               note='Inherit the user-approved parent deadline')
        with contextlib.redirect_stdout(io.StringIO()):
            campaign.approve(args)
        child = campaign.read_json(self.root / 'child/state.json')
        self.assertEqual(child['deadline'], campaign.datetime.fromisoformat(deadline).timestamp())

    def test_media_duration_uses_presented_video_frames(self):
        cases = [
            ({'duration': '1', 'nb_frames': '13', 'avg_frame_rate': '10/1'}, 10, 0, 1),
            ({'avg_frame_rate': '10/1', 'tags': {'DURATION': '00:00:12'}}, 20, 10, 2),
            ({'avg_frame_rate': '0/0'}, 3, 0, .3),
        ]
        for stream, count, start, duration in cases:
            def ffprobe(command, stream=stream, count=count, start=start, **kwargs):
                if '-show_frames' in command:
                    for index in range(count):
                        kwargs['stdout'].write(f'frame|best_effort_timestamp_time={start + index / 10}|duration_time=0.1\n')
                    return SimpleNamespace(returncode=0, stderr='')
                return SimpleNamespace(stdout=json.dumps({'streams': [stream], 'format': {'duration': '12'}}))
            with self.subTest(stream=stream), patch.object(campaign.subprocess, 'run', side_effect=ffprobe):
                result = campaign.probe('camera with spaces.mkv')
            self.assertEqual(result['frame_count'], count)
            self.assertEqual(result['frame_count_basis'], 'decoded presented frames')
            self.assertAlmostEqual(result['duration_seconds'], duration)
            self.assertAlmostEqual(result['average_fps'], 10)
            self.assertFalse(result['pts_variable'])
        self.assertIsNone(campaign.probe_number('0/0'))
        self.assertIsNone(campaign.probe_number('N/A'))
        with patch.object(campaign.subprocess, 'run', return_value=SimpleNamespace(stderr='decode failed')), \
                self.assertRaisesRegex(ValueError, 'decode errors'):
            campaign.scan_video('corrupt.mp4', timeout=2)

    def test_no_run_without_approval_and_no_duplicate_or_changed_input(self):
        args = self.ready(approved=False)
        with self.assertRaisesRegex(ValueError, 'approval'):
            campaign.run(args)
        self.assertFalse((self.root / 'runs').exists())
        self.write('state.json', json.dumps({'deadline': time.time() + 30}))
        with contextlib.redirect_stdout(io.StringIO()):
            campaign.run(args)
        record = campaign.read_json(self.root / 'runs/basic-r1/run.json')
        self.assertEqual(record['status'], 'completed')
        with self.assertRaises(FileExistsError):
            campaign.run(args)
        self.write('inputs.json', json.dumps({str(self.root / 'dataset.json'): 'wrong'}))
        args.name = 'next'
        with self.assertRaisesRegex(ValueError, 'Frozen input changed'):
            campaign.run(args)
        self.assertFalse((self.root / 'runs/next').exists())

    def test_busy_lock_and_timeout(self):
        args = self.ready()
        with campaign.lock(self.root / 'gpu.lock'), self.assertRaisesRegex(RuntimeError, 'in use'):
            campaign.run(args)
        self.write('job.json', json.dumps({'cwd': str(self.root), 'run': [sys.executable, '-c',
                   'import time; time.sleep(30)', '{config}'], 'timeout_seconds': .2}))
        started = time.monotonic()
        with self.assertRaises(TimeoutError):
            campaign.run(args)
        self.assertLess(time.monotonic() - started, 5)
        self.assertEqual(campaign.read_json(self.root / 'runs/basic-r1/run.json')['status'], 'failed')

    def test_resume_refuses_orphan_after_controller_death(self):
        args = self.ready()
        self.write('job.json', json.dumps({'cwd': str(self.root), 'run': [sys.executable, '-c',
                   'import os,time; from pathlib import Path; Path("child.pid").write_text(str(os.getpid())); time.sleep(20)',
                   '{config}'], 'timeout_seconds': 25}))
        command = [sys.executable, str(SKILL / 'scripts/campaign.py'), 'run']
        for key, value in vars(args).items():
            command += ['--' + key.replace('_', '-'), value]
        child_pid = None
        controller = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            until = time.monotonic() + 5
            while not (self.root / 'child.pid').exists() and time.monotonic() < until:
                time.sleep(.02)
            self.assertTrue((self.root / 'child.pid').exists())
            child_pid = int((self.root / 'child.pid').read_text())
            controller.kill()
            controller.wait(timeout=2)
            args.name = 'resume'
            with self.assertRaisesRegex(RuntimeError, 'worker|running|active'):
                campaign.run(args)
        finally:
            if controller.poll() is None:
                controller.kill()
            controller.wait(timeout=2)
            if child_pid and campaign.group_alive(child_pid):
                os.killpg(child_pid, signal.SIGKILL)

    def test_deadlines_and_owned_descendant_cleanup(self):
        with patch.object(campaign.subprocess, 'Popen') as launch:
            with self.assertRaises(TimeoutError):
                campaign.execute(['unused'], self.root, self.root / 'expired.log', time.time() - 1,
                                 self.root / 'worker.json')
            launch.assert_not_called()
        args = self.ready()
        child = 'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)'
        leader = f'import subprocess,sys,time; subprocess.Popen([sys.executable,"-c",{child!r}]); time.sleep(30)'
        self.write('job.json', json.dumps({'cwd': str(self.root), 'run': [sys.executable, '-c', leader, '{config}'],
                                        'timeout_seconds': .6}))
        try:
            with self.assertRaises(TimeoutError):
                campaign.run(args)
            self.assertFalse(campaign.group_alive(campaign.read_json(self.root / 'worker.json')['pid']))
        finally:
            worker = campaign.read_json(self.root / 'worker.json')
            if campaign.group_alive(worker['pid']):
                os.killpg(worker['pid'], signal.SIGKILL)

    def test_prepare_cannot_mutate_or_malform_candidate(self):
        scripts = [
            'from pathlib import Path; Path("{run_dir}/config_tracker.yml").write_text("ReID: 0\\n")',
            'from pathlib import Path; p=Path("{config}"); p.write_text("ReID:\\n  reidType: 3\\n"); Path("{run_dir}/config_tracker.yml").write_text(p.read_text())',
        ]
        for i, script in enumerate(scripts):
            args = self.ready()
            args.name = f'case-{i}'
            self.write('job.json', json.dumps({'cwd': str(self.root), 'prepare': [sys.executable, '-c', script, '{config}'],
                                             'run': [sys.executable, '-c', 'print("App run successful")']}))
            with self.subTest(i=i), self.assertRaises((ValueError, TypeError)):
                campaign.run(args)
            self.assertEqual(campaign.read_json(self.root / f'runs/{args.name}/run.json')['status'], 'failed')

    def test_run_normalizes_valid_yaml_for_the_sample_configurator(self):
        args = self.ready()
        original = 'ObjectModelProjection:\n  cameraModelFilepath:\n  - /inputs/cam0.yml\n'
        self.write('config.yml', original)
        script = 'from pathlib import Path; text=Path("{config}").read_text(); assert "    - /inputs/cam0.yml" in text; Path("{run_dir}/config_tracker.yml").write_text(text)'
        self.write('job.json', json.dumps({'cwd': str(self.root), 'prepare': [sys.executable, '-c', script],
                   'run': [sys.executable, '-c', 'print("App run successful")']}))
        campaign.run(args)
        self.assertEqual((self.root / 'config.yml').read_text(), original)
        self.assertEqual((self.root / 'runs/basic-r1/source-config.yml').read_text(), original)
        record = campaign.read_json(self.root / 'runs/basic-r1/run.json')
        self.assertNotEqual(record['source_config_sha256'], record['candidate_sha256'])

    def test_completed_command_cannot_change_frozen_inputs_or_prepared_config(self):
        for target in ('dataset.json', '{run_dir}/config_tracker.yml'):
            args = self.ready()
            args.name = 'input-change' if target == 'dataset.json' else 'config-change'
            self.write('inputs.json', json.dumps({str(self.root / 'dataset.json'): campaign.sha256(self.root / 'dataset.json')}))
            prepare = 'import shutil; shutil.copyfile("{config}", "{run_dir}/config_tracker.yml")'
            change = f'from pathlib import Path; Path("{target}").write_text("changed"); print("App run successful")'
            self.write('job.json', json.dumps({'cwd': str(self.root),
                       'prepare': [sys.executable, '-c', prepare], 'run': [sys.executable, '-c', change]}))
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, 'Frozen input changed'):
                campaign.run(args)
            self.assertEqual(campaign.read_json(self.root / f'runs/{args.name}/run.json')['status'], 'failed')

    def test_legacy_selector_is_not_an_allowed_path_rewrite(self):
        args = self.ready()
        self.write('config.yml', 'TrajectoryManagement:\n  enableArchiveReAssoc: 0\n')
        script = 'from pathlib import Path; Path("{run_dir}/config_tracker.yml").write_text("TrajectoryManagement:\\n  enableArchiveReAssoc: 1\\n")'
        self.write('job.json', json.dumps({'cwd': str(self.root), 'prepare': [sys.executable, '-c', script, '{config}'],
                   'run': [sys.executable, '-c', 'print("App run successful")'],
                   'config_rewrites': {'TrajectoryManagement.enableArchiveReAssoc': 1}}))
        with self.assertRaisesRegex(ValueError, 'Preparation changed'):
            campaign.run(args)
        expected = {'ReID': {'reidType': 2, 'model0': {'reidType': 2, 'modelEngineFile': '/old.engine'}}}
        actual = {'ReID': {'reidType': 2, 'model0': {'reidType': 1, 'modelEngineFile': '/old.engine'}}}
        with self.assertRaisesRegex(ValueError, 'ReID.model0.reidType'):
            campaign.verify_config(expected, actual, {'ReID.model0': actual['ReID']['model0']})
        actual['ReID']['model0'].update(reidType=2, modelEngineFile='/new.engine')
        campaign.verify_config(expected, actual, {'ReID.model0.modelEngineFile': '/new.engine'})

    def test_basic_and_final_config_keep_required_production_outputs(self):
        base = self.write('base.yml', 'ObjectModelProjection:\n  outputVisibility: 1\n  cameraModelFilepath:\n    - /inputs/old.yml\nStateEstimator:\n  stateEstimatorType: 4\nCommunicator:\n  communicatorType: 1\nReID:\n  reidType: 3\n  outputReidTensor: 1\n')
        output = self.root / 'basic.yml'
        campaign.candidate(SimpleNamespace(base=base, overlay=[], basic=True, output=output))
        config = campaign.yaml_load(output)
        self.assertIn('    - /inputs/old.yml', output.read_text())
        for group, field in [('ReID', 'reidType'), ('VisualTracker', 'visualTrackerType'),
                             ('PoseEstimator', 'poseEstimatorType'), ('ReIDService', 'reidServiceType')]:
            self.assertEqual(config[group][field], 0)
        self.assertEqual(config['MultiViewAssociator']['multiViewAssociatorType'], 1)
        self.assertEqual(config['DataAssociator']['usePrediction4Assoc'], 0)
        self.assertEqual(config['TargetManagement']['outputVisibility'], 1)
        self.assertEqual(config['ObjectModelProjection']['outputFootLocation'], 1)
        self.assertNotIn('outputVisibility', config['ObjectModelProjection'])
        self.assertLess(config['StateEstimator']['noiseWeightVar4Loc'], 0)
        self.assertLess(config['StateEstimator']['noiseWeightVar4Vel'], 0)
        debug = self.write('debug.yml', 'ReID:\n  reidType: 2\n  outputReidTensor: 1\n  debugVerboseLevel: 4\nVisualTracker:\n  debugVerboseLevel_cufft: 3\n  debugVerboseLevel_gpumemory: 2\n  showGpuTiming: 1\nPoseEstimator:\n  saveFrameViz: 1\nTrajectoryManagement:\n  dumpTrajectoryIntoFile: 1\nObjectModelProjection:\n  outputFootLocation: 1\n')
        clean = self.root / 'clean.yml'
        campaign.export(SimpleNamespace(config=debug, output=clean))
        actual = campaign.yaml_load(clean)
        self.assertEqual(actual['ReID']['debugVerboseLevel'], 0)
        self.assertEqual(actual['ReID']['reidType'], 2)
        self.assertEqual(actual['ReID']['outputReidTensor'], 1)
        self.assertEqual(actual['ObjectModelProjection']['outputFootLocation'], 1)
        self.assertTrue(all(value == 0 for value in actual['VisualTracker'].values()))
        self.assertEqual(actual['PoseEstimator']['saveFrameViz'], 0)

    def test_pipeline_checks_detector_mode_and_crop_alignment(self):
        pipeline = self.write('pipeline.txt', '[primary-gie]\nenable=0\n[replay]\nenable=1\n[tracker]\ntracker-width=1920\ntracker-height=1080\n')
        tracker = {'PoseEstimator': {'poseEstimatorType': 1, 'useVPICropScaler': 0}}
        with self.assertRaisesRegex(ValueError, 'detector enablement'):
            campaign.check_pipeline(pipeline, tracker, 'live')
        with self.assertRaisesRegex(ValueError, 'divisible by 32'):
            campaign.check_pipeline(pipeline, tracker, 'replay')
        tracker['PoseEstimator']['useVPICropScaler'] = 1
        campaign.check_pipeline(pipeline, tracker, 'replay')
        pipeline.write_text('[primary-gie]\nenable=1\n[tracker]\ntracker-width=1920\ntracker-height=1080\n')
        campaign.check_pipeline(pipeline, tracker, 'live')
        ensemble = {'ReID': {'reidType': 2, 'enableModelId': 0,
                    'model0': {'reidType': 2, 'useVPICropScaler': 1},
                    'model1': {'reidType': 2, 'useVPICropScaler': 0}}}
        campaign.check_pipeline(pipeline, ensemble, 'live')
        ensemble['ReID']['enableModelId'] = 1
        with self.assertRaisesRegex(ValueError, 'divisible by 32'):
            campaign.check_pipeline(pipeline, ensemble, 'live')
        ensemble['ReID']['reidType'] = 0
        campaign.check_pipeline(pipeline, ensemble, 'live')
        ensemble['ReID'].update(reidType=2, enableModelId=0, useVPICropScaler=1)
        ensemble['ReID']['model0'].pop('useVPICropScaler')
        campaign.check_pipeline(pipeline, ensemble, 'live')
        ensemble['ReID']['model0'].update(reidType=0, useVPICropScaler=0)
        with self.assertRaisesRegex(ValueError, 'divisible by 32'):
            campaign.check_pipeline(pipeline, ensemble, 'live')

    def test_conversion_preserves_large_ids_and_explicit_annotation_mapping(self):
        source = self.write('gt.txt', 'cam0 18446744073709551615 1 10 20 30 40 3.5 4.5\n')
        mapping = self.write('map.json', '{"cam0":{"1":6}}')
        output = self.root / 'gt.csv'
        evaluation.convert(SimpleNamespace(input=source, output=output, frame_map=mapping,
                           camera=None, format='nine-field-gt', class_id='0', frame_base=1))
        row = evaluation.load_rows(output)[0]
        self.assertEqual(row['frame'], 6)
        self.assertEqual(row['id'], '18446744073709551615')
        self.assertEqual(row['width'], 30)
        self.assertEqual(row['x'], 3.5)

    def test_native_classes_are_preserved_or_explicitly_filtered(self):
        source = self.write('native.txt', '1,0,10,20,30,40,0.9,3.5,4.5,-1,0\n1,1,10,20,30,40,0.9,5,6,-1,1\n')
        args = SimpleNamespace(input=source, output=self.root / 'native.csv', frame_map=None,
                               camera='cam0', format='deepstream-traj', class_id=None, frame_base=1)
        evaluation.convert(args)
        self.assertEqual([r['class'] for r in evaluation.load_rows(args.output)], ['0', '1'])
        args.output, args.class_id = self.root / 'person.csv', '0'
        evaluation.convert(args)
        self.assertEqual(len(evaluation.load_rows(args.output)), 1)

    def test_generic_scoring_has_no_inherited_thirty_unit_filter(self):
        camera = self.write('camera.yml', 'projectionMatrix_3x4_w2p: [1,0,0,0,0,1,0,0,0,0,1,10]\nmodelInfo:\n  height: 1.7\n')
        truth = self.write('gt.csv', 'camera,frame,id,class,x,y\ncam0,1,0,0,100,1\ncam0,6,0,0,101,1\n')
        pred = self.write('runs/basic/predictions.csv', 'camera,frame,id,class,x,y\ncam0,1,99,0,100,1\ncam0,6,99,0,101,1\n')
        manifest = self.write('dataset.json', json.dumps({'cameras': [{'id': 'cam0', 'calibration': camera.name}],
            'ground_truth': truth.name, 'world_units': 'm', 'evaluation': {'frames': [1, 6, 11],
            'class_id': '0', 'fusion': 'closest', 'zero_distance': 2, 'region': None, 'max_distance': None,
            'output_semantics': 'synthetic frame-local world observations'}}))
        out = self.root / 'runs/basic/metrics.json'
        args = SimpleNamespace(manifest=manifest, predictions=pred, output=out,
                               evaluator=evaluation.APP / 'utils/evaluate_tracking.py')
        with contextlib.redirect_stdout(io.StringIO()):
            evaluation.score(args)
        result = json.loads(out.read_text())
        self.assertEqual(result['metrics']['GT_Dets'], 2)
        self.assertEqual(result['metrics']['HOTA'], 100)
        self.assertEqual(result['annotation_steps'], 3)
        self.assertEqual(result['gt_counts']['admitted_observations'], 2)
        self.write('runs/basic/run.json', '{"name":"basic","status":"completed"}')
        campaign.summarize(SimpleNamespace(campaign=self.root))
        self.assertEqual(campaign.read_json(self.root / 'summary.json')['hota_order'], ['basic'])
        pred.write_text(pred.read_text() + 'cam0,1,99,0,100,1\n')
        with self.assertRaisesRegex(ValueError, 'hash'):
            campaign.summarize(SimpleNamespace(campaign=self.root))
        args.output = self.root / 'bad.json'
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            evaluation.score(args)

    def test_adapter_rejects_ambiguous_columns_and_frames(self):
        for frames in ([True], {'start': 1.9, 'stop': 3.9, 'step': 1}):
            with self.subTest(frames=frames), self.assertRaises(ValueError):
                evaluation.frame_list({'frames': frames})
        for text in ('camera,frame,id,class,x,x,y\na,1,0,0,3,4,5\n', 'camera,frame,id,class,x,y\na,1,0,0,3\n'):
            source = self.write('bad.csv', text)
            with self.assertRaises(ValueError):
                evaluation.load_rows(source)

    def test_diagnostics_do_not_invent_accuracy(self):
        pred = self.write('pred.csv', 'camera,frame,id,class,x,y\na,1,0,0,1,1\na,2,0,0,11,1\nb,2,0,0,1,1\n')
        output = self.root / 'proxy.json'
        evaluation.diagnostics(SimpleNamespace(predictions=pred, fps=10, speed_limit=3,
                    disagreement=2, short_seconds=.3, limit=50, output=output))
        result = json.loads(output.read_text())
        self.assertNotIn('metrics', result)
        self.assertEqual(result['flagged_events'], 2)
        self.assertEqual(result['speed_comparisons'], 1)
        self.assertEqual(result['shared_id_cross_view_pairs'], 1)
        self.assertEqual(result['speed_flag_rate'], 1)
        self.assertEqual(result['shared_id_disagreement_rate'], 1)

    def test_repeat_summary_does_not_select_the_best_lucky_run(self):
        rows = []
        for config, scores in [('variable', [90, 40, 40]), ('stable', [60, 60, 60])]:
            for index, score in enumerate(scores):
                rows.append({'name': f'{config}-{index}', 'candidate_sha256': config,
                             'effective_config_sha256': config, 'evaluation': {'metrics': {
                             'HOTA': score, 'MOTA': -10, 'IDF1': 50}}})
        groups = campaign.configuration_summary(rows)
        self.assertEqual(groups[0]['representative_run'], 'stable-0')
        self.assertEqual(groups[1]['range']['HOTA'], 50)
        self.assertTrue(all(group['confirmed'] for group in groups))
        self.assertFalse(campaign.configuration_summary(rows[:1])[0]['confirmed'])

    def test_contact_sheets_use_raw_frames_and_boxes(self):
        import cv2

        import numpy as np
        video = self.root / 'cam.mp4'
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 10, (320, 180))
        for i in range(10):
            picture = np.zeros((180, 320, 3), dtype=np.uint8)
            cv2.rectangle(picture, (40 + i, 30), (80 + i, 140), (50, 200, 100), -1)
            writer.write(picture)
        writer.release()
        manifest = self.write('dataset.json', json.dumps({'cameras': [{'id': 'cam0', 'video': str(video)}], 'physical_fps': 10}))
        campaign_root = self.root / 'campaign'
        with contextlib.redirect_stdout(io.StringIO()):
            campaign.inspect(SimpleNamespace(manifest=manifest, output=campaign_root))
        proposal = campaign.read_json(campaign_root / 'state.json')
        self.assertAlmostEqual(proposal['proposed_hours'], 1 / 60)
        self.assertEqual(proposal['proposed_seconds'], 60)
        self.assertIsNone(proposal['deadline'])
        slower = json.loads(manifest.read_text())
        slower['physical_fps'] = 5
        slow_manifest = self.write('slower.json', json.dumps(slower))
        with contextlib.redirect_stdout(io.StringIO()):
            campaign.inspect(SimpleNamespace(manifest=slow_manifest, output=self.root / 'slower'))
        self.assertAlmostEqual(campaign.read_json(self.root / 'slower/state.json')['proposed_hours'], 2 / 60)
        predictions = self.write('pred.csv', 'camera,frame,id,class,x,y,left,top,width,height\ncam0,1,5,0,1,1,40,30,40,110\n')
        output = self.root / 'review'
        subprocess.run([sys.executable, str(SKILL / 'scripts/review_frames.py'), '--manifest', str(manifest),
                        '--predictions', str(predictions), '--frames', '1,5', '--output', str(output)], check=True, capture_output=True)
        self.assertTrue((output / 'frame-000001.jpg').is_file())
        self.assertEqual(len(json.loads((output / 'frames.json').read_text())), 2)


if __name__ == '__main__':
    unittest.main()
