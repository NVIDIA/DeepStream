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
"""Check native sample generation without Docker, models, or services."""

import os
import sys
import cv2
import yaml
import json
import shutil
import zipfile
import tarfile
import unittest
import tempfile
import subprocess
import configparser

import numpy as np

from pathlib import Path
from utils.deepstream_auto_configurator import DeepStreamAutoConfigurator

APP = Path(__file__).resolve().parents[1]


class SampleGenerationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / 'SDK files' / APP.relative_to(APP.parents[3])
        self.dataset = self.root / 'dataset'
        (self.dataset / 'videos').mkdir(parents=True)
        (self.dataset / 'camInfo').mkdir()
        self.labels = self.dataset / 'detections/PeopleNetTransformer'
        self.labels.mkdir(parents=True)
        for index in range(2):
            path = self.dataset / f'videos/{index:03d}.mp4'
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 25, (32, 32))
            self.assertTrue(writer.isOpened())
            for _ in range(4):
                writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
            writer.release()
            (self.dataset / f'camInfo/{index:03d}.yml').write_text(yaml.safe_dump({
                'projectionMatrix_3x4_w2p': [1, 0, 0, index, 0, 1, 0, 0, 0, 0, 1, 4],
                'modelInfo': {'height': 1.7, 'radius': .3},
            }))
            (self.labels / f'{index:03d}.txt').write_text('1,-1,1.25,2.5,3.75,4.5,0.9,-1,-1\n')

    def generator(self, profile='scout_pn2'):
        return DeepStreamAutoConfigurator(str(self.dataset), str(self.root / 'output'), profile)

    def pipeline(self, content):
        config = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=('#',))
        config.read_string(content)
        return config

    def test_cli_generates_live_and_replay_configs(self):
        for replay in (False, True):
            with self.subTest(replay=replay):
                output = self.root / f'cli_{replay}'
                command = [sys.executable, '-m', 'utils.deepstream_auto_configurator',
                           '--profile', 'scout_pn2', '--dataset-dir', str(self.dataset), '--output-dir', str(output)]
                if replay:
                    command += ['--replay', str(self.labels)]
                subprocess.run(command, cwd=APP, check=True, capture_output=True, text=True)
                main = self.pipeline((output / 'config_deepstream.txt').read_text())
                self.assertEqual(main['primary-gie']['enable'], str(int(not replay)))
                self.assertEqual('replay' in main, replay)
                if replay:
                    self.assertEqual(dict(main['replay']), {
                        'enable': '1', 'label-dir': '/workspace/inputs/detections/PeopleNetTransformer/',
                        'file-names': '000.txt;001.txt', 'max-frame-nums': '4;4', 'label-width': '32',
                        'label-height': '32', 'interval': '0',
                    })
                tracker = yaml.safe_load((output / 'config_tracker.yml').read_text())
                self.assertEqual(tracker['Communicator']['communicatorType'], 1)
                peers = yaml.safe_load((output / 'pub_sub_info_config_0.yml').read_text())
                self.assertEqual(peers, {'camIds': ['1', '2'], 'subToCamIds': [['2'], ['1']]})
                for folder in ('outVideos', 'trajDumps', 'tracker-kitti-dump', 'infer-kitti-dump'):
                    self.assertTrue((output / folder).is_dir())

    def test_missing_and_external_replay_inputs_fail(self):
        (self.labels / '000.txt').unlink()
        with self.assertRaises(FileNotFoundError):
            self.generator().generate_configs(replay_dir=self.labels)
        external = self.root / 'external'
        external.mkdir()
        with self.assertRaises(ValueError):
            self.generator().generate_configs(replay_dir=external)

    def test_wildtrack_recipe_preserves_dataset_and_uses_video_timestamps(self):
        camera = self.dataset / 'camInfo/000.yml'
        original = camera.read_bytes()
        configs = self.generator('wildtrack').generate_configs(replay_dir=self.labels)
        tracker = yaml.safe_load(configs['config_tracker.yml'])
        self.assertEqual(tracker['StateEstimator']['stateEstimatorType'], 3)
        for section, key in [('VisualTracker', 'visualTrackerType'), ('PoseEstimator', 'poseEstimatorType'),
                             ('ReID', 'reidType'), ('ReIDService', 'reidServiceType')]:
            self.assertEqual(tracker[section][key], 0)
        self.assertEqual(tracker['TargetManagement']['maxShadowTrackingAge'], 24)
        self.assertEqual(tracker['TrajectoryManagement']['minVisibility4TrajDump'], 0.15)
        self.assertEqual(tracker['MultiViewAssociator']['recentlyActiveAge'], 40)
        self.assertEqual(tracker['MultiViewAssociator']['enableMsgSync'], 1)
        self.assertEqual(tracker['ObjectModelProjection']['useDefaultModelHeight'], 1.8)
        self.assertEqual(tracker['ObjectModelProjection']['useDefaultModelRadius'], 0.15)
        pipeline = self.pipeline(configs['config_deepstream.txt'])
        self.assertEqual(pipeline['streammux']['attach-sys-ts-as-ntp'], '0')
        self.assertEqual(pipeline['tracker']['tracker-width'], '1920')
        self.assertEqual(pipeline['tracker']['tracker-height'], '1088')
        self.assertEqual(camera.read_bytes(), original)

    def test_optional_reid_service_preserves_tracker_recipe(self):
        for profile in ('scout_pnt', 'scout_pn2'):
            with self.subTest(profile=profile):
                generator = self.generator(profile)
                configs = generator.generate_configs(detector_config='config_pgie.txt', replay_dir=self.labels)
                original = yaml.safe_load(configs['config_tracker.yml'])
                self.assertEqual(original['ReIDService']['reidServiceType'], 0)
                enabled = generator.configure_reid_service(configs)
                tracker = yaml.safe_load(enabled['config_tracker.yml'])
                for section in original.keys() - {'ReID', 'ReIDService', 'TrajectoryManagement'}:
                    self.assertEqual(tracker[section], original[section])
                self.assertEqual(tracker['TrajectoryManagement'], {**original['TrajectoryManagement'], 'reidExtractionInterval': 5})
                self.assertEqual(tracker['ReID']['reidType'], 2)
                self.assertEqual(tracker['ReID']['reidFeatureSize'], 1280)
                self.assertEqual(tracker['ReID']['outputReidTensor'], 1)
                self.assertEqual(tracker['ReIDService']['reidServiceType'], 1)
                self.assertEqual(tracker['ReIDService']['operateOnClassIds'], [])
                pipeline = self.pipeline(enabled['config_deepstream.txt'])
                self.assertEqual(pipeline['sink3']['enable'], '1')
                self.assertEqual(pipeline['sink3']['msg-broker-conn-str'], 'localhost;9092;mv3dt')
                self.assertEqual(pipeline['sink3']['topic'], 'mv3dt')
                self.assertEqual(pipeline['sink3']['msg-conv-payload-type'], '2')
                self.assertEqual(pipeline['primary-gie']['enable'], '0')
                self.assertEqual(self.pipeline(configs['config_deepstream.txt'])['sink3']['enable'], '0')
        with self.assertRaisesRegex(ValueError, 'only SCOUT'):
            self.generator('wildtrack').configure_reid_service({})

    def test_mqtt_deployment_creates_its_output_directory(self):
        generator = self.generator('mtmc_4cam_pnt')
        path = Path(generator.generate_deployment_config([1, 2]))
        self.assertEqual(yaml.safe_load(path.read_text())['ds_instance_cam_assignment'], [[1, 2]])

    def launcher(self, sample, backend='ds', **settings):
        commands = self.root / 'commands'
        commands.mkdir(exist_ok=True)
        venv = self.root / 'mv3dt_venv/bin'
        venv.mkdir(parents=True, exist_ok=True)
        (venv / 'activate').write_text('')
        models = self.root / 'model files'
        for directory in (models, self.root / 'models'):
            for model in ('PeopleNetTransformer', 'PeopleNet2.6.3', 'RTDETR'):
                (directory / model).mkdir(parents=True, exist_ok=True)
        builder = self.root.parents[3] / 'tools/inference_builder'
        (builder / '.venv/bin').mkdir(parents=True, exist_ok=True)
        (builder / '.venv/bin/activate').touch()
        log = self.root / 'commands.jsonl'
        log.write_text('')
        stub = f'''#!{sys.executable}
import os
import sys
import json
from pathlib import Path
command = Path(sys.argv[0]).name
with open(os.environ['COMMAND_LOG'], 'a') as output:
    output.write(json.dumps([command, *sys.argv[1:]]) + '\\n')
if command == 'git':
    print(os.environ['DS_DIR_FALLBACK'])
elif command == 'python':
    if os.environ.get('CONFIG_STATUS'):
        sys.exit(int(os.environ['CONFIG_STATUS']))
    output = next((arg.split('=', 1)[1] for arg in sys.argv if arg.startswith('--output-dir=')), None)
    if '--output-dir' in sys.argv:
        output = sys.argv[sys.argv.index('--output-dir') + 1]
    if output:
        Path(output).mkdir(parents=True, exist_ok=True)
        (Path(output) / 'nvdsinfer_config.yaml').touch()
elif command == 'docker' and sys.argv[1:] == ['info']:
    print('Runtimes: nvidia')
elif command == 'curl':
    print('{{"status":"ready"}}')
    sys.exit(int(os.environ.get('CURL_STATUS', '0')))
else:
    print(os.environ.get('APP_MESSAGE', 'App run successful'))
    sys.exit(int(os.environ.get('DOCKER_STATUS', '0')))
'''
        for name in ('python', 'docker', 'git', 'curl'):
            path = commands / name
            path.write_text(stub)
            path.chmod(0o755)
        env = {key: value for key, value in os.environ.items() if key not in {
            'DATASET_DIR', 'MODEL_REPO', 'EXPERIMENT_DIR', 'DETECTOR_MODEL', 'REPLAY', 'DEEPSTREAM_IMAGE', 'INFERENCE_BUILDER_DIR', 'REID_SERVICE',
        }}
        env.update(PATH=f'{commands}:{env["PATH"]}', COMMAND_LOG=str(log),
                   DS_DIR=str(self.root.parents[3]), DS_DIR_FALLBACK=str(self.root.parents[3]),
                   DATASET_DIR=str(self.dataset), MODEL_REPO=str(models),
                   EXPERIMENT_DIR=str(self.root / 'run output'), DEEPSTREAM_IMAGE='test-image:local')
        env.update(settings)
        result = subprocess.run(['bash', str(APP / f'scripts/test_{sample}_{backend}.sh')],
                                cwd=self.root.parent, env=env, check=False, capture_output=True, text=True)
        calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        return result, calls

    def test_launchers_select_profiles_and_optional_replay(self):
        for sample, detector, profile in [('wildtrack', 'PeopleNetTransformer', 'wildtrack'),
                                          ('scout', 'PeopleNetTransformer', 'scout_pnt'),
                                          ('scout', 'PeopleNet2.6.3', 'scout_pn2')]:
            modes = [('0', '0'), ('1', '0')]
            if sample == 'scout':
                modes += [('0', '1'), ('1', '1')]
            for replay, reid in modes:
                with self.subTest(sample=sample, detector=detector, replay=replay, reid=reid):
                    output = self.root / f'{profile}-{replay}-{reid}'
                    result, calls = self.launcher(sample, DETECTOR_MODEL=detector, REPLAY=replay,
                                                  REID_SERVICE=reid, EXPERIMENT_DIR=str(output))
                    self.assertEqual(result.returncode, 0, result.stderr)
                    generator, docker = calls[-2:]
                    self.assertEqual(generator[:5], ['python', '-m', 'utils.deepstream_auto_configurator', '--profile', profile])
                    self.assertIn('--enable-file-output', generator)
                    self.assertNotIn('--enable-msg-broker', generator)
                    self.assertNotIn('--enable-osd', generator)
                    self.assertEqual('--enable-reid-service' in generator, reid == '1')
                    self.assertEqual(any(call[0] == 'curl' for call in calls), reid == '1')
                    self.assertEqual('--replay' in generator, replay == '1')
                    if replay == '1':
                        self.assertEqual(generator[-1], str(self.dataset / 'detections' / detector))
                    network = 'host' if reid == '1' else 'bridge'
                    self.assertEqual(docker[:8], ['docker', 'run', '-t', '--rm', '--gpus', 'all', '--network', network])
                    self.assertIn(f'{self.dataset}:/workspace/inputs:ro', docker)
                    self.assertIn(f'{self.root / "model files"}:/workspace/models', docker)
                    self.assertIn(f'{output}:/workspace/experiments', docker)
                    self.assertIn('test-image:local', docker)
                    self.assertEqual(docker[-3:], ['deepstream-test5-app', '-c', 'config_deepstream.txt'])
                    self.assertEqual((output / 'run_deepstream_app.log').read_text(), 'App run successful\n')

    def test_launcher_defaults_and_sdk_root_discovery(self):
        for sample in ('wildtrack', 'scout'):
            for sdk_dir in (str(self.root.parents[3]), ''):
                with self.subTest(sample=sample, sdk_dir=sdk_dir):
                    output = self.root / f'{sample}_{bool(sdk_dir)}'
                    result, calls = self.launcher(sample, DS_DIR=sdk_dir, EXPERIMENT_DIR=str(output))
                    self.assertEqual(result.returncode, 0, result.stderr)
                    generator = next(call for call in calls if call[0] == 'python')
                    self.assertIn('--replay', generator)
                    self.assertEqual(any(call[0] == 'git' for call in calls), not sdk_dir)

    def test_launchers_stop_on_errors_and_existing_output(self):
        for sample in ('scout', 'wildtrack'):
            for settings in ({'CONFIG_STATUS': '3'}, {'DOCKER_STATUS': '7'}, {'APP_MESSAGE': 'App run failed'}):
                with self.subTest(sample=sample, settings=settings):
                    output = self.root / f'{sample}-{next(iter(settings))}'
                    result, calls = self.launcher(sample, EXPERIMENT_DIR=str(output), **settings)
                    self.assertNotEqual(result.returncode, 0)
                    if 'CONFIG_STATUS' in settings:
                        self.assertEqual([call[0] for call in calls], ['python'])
            output = self.root / f'{sample}-existing'
            output.mkdir()
            result, calls = self.launcher(sample, EXPERIMENT_DIR=str(output))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('already exists', result.stderr)
            self.assertEqual(calls, [])
        result, calls = self.launcher('scout', DETECTOR_MODEL='RTDETR')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('DETECTOR_MODEL must be', result.stderr)
        self.assertEqual(calls, [])
        for settings in ({'REID_SERVICE': 'invalid'}, {'REID_SERVICE': '1', 'CURL_STATUS': '7'}):
            result, calls = self.launcher('scout', **settings)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(any(call[0] in ('python', 'docker') for call in calls))

    def test_legacy_launchers_route_profiles_and_paths(self):
        for sample, cameras in (('4cam', 4), ('12cam', 12), ('custom_2d_tracker', 6)):
            variants = [('PeopleNetTransformer', 'pnt'), ('PeopleNet2.6.3', 'pn2'), ('RTDETR', 'detr')]
            if cameras == 6:
                variants = variants[:1]
            for backend in ('ds', 'ib'):
                for detector, suffix in variants:
                    with self.subTest(cameras=cameras, backend=backend, detector=detector):
                        directory = 'deepstream' if backend == 'ds' else 'inference_builder'
                        output = self.root / 'experiments' / directory / f'{cameras}cam'
                        result, calls = self.launcher(sample, backend, DETECTOR_MODEL=detector,
                                                      DATASET_DIR='', MODEL_REPO='', EXPERIMENT_DIR='')
                        self.assertEqual(result.returncode, 0, result.stderr)
                        generator = next(call for call in calls if len(call) > 2 and call[1] == '-m')
                        profile = f'mtmc_{12 if cameras == 6 else cameras}cam_{suffix}'
                        self.assertIn(f'--profile={profile}', generator)
                        self.assertIn(f'--dataset-dir={self.root}/datasets/mtmc_{cameras}cam', generator)
                        self.assertIn(f'--output-dir={output}', generator)
                        if cameras == 6:
                            self.assertIn('--tracker-config=config_tracker_2d.yml', generator)
                        if cameras == 12 and backend == 'ib' and detector == 'RTDETR':
                            self.assertIn('--tracker-config=config_tracker_ib.yml', generator)
                        docker = next(call for call in calls if call[:2] == ['docker', 'run'])
                        self.assertIn(f'{self.root}/models:/workspace/models', docker)
                        self.assertIn(f'{output}:/workspace/experiments', docker)

    @unittest.skipUnless(shutil.which('unzip'), 'Dataset setup requires unzip')
    def test_setup_extracts_new_samples_without_overwriting_existing_data(self):
        assets = self.root / 'assets'
        assets.mkdir()
        archive = assets / 'datasets.zip'
        with zipfile.ZipFile(archive, 'w') as bundle:
            for dataset in ('mtmc_4cam', 'mtmc_12cam', 'scout', 'wildtrack'):
                for folder, filename in [('videos', '000.mp4'), ('camInfo', '000.yml')]:
                    bundle.writestr(f'datasets/{dataset}/{folder}/{filename}', 'archive content')
                    if dataset.startswith('mtmc_'):
                        directory = self.root / 'datasets' / dataset / folder
                        directory.mkdir(parents=True)
                        (directory / filename).write_text('existing local data')
        setup = (APP / 'scripts/setup_prerequisites.sh').read_text()
        function = setup[setup.index('setup_git_lfs() {'):setup.index('\nsetup_mosquitto() {')]
        command = 'set -e\nlog_info() { :; }\nlog_success() { :; }\nlog_error() { :; }\n' + function + '\nsetup_git_lfs'
        env = dict(os.environ, MV3DT_DIR=str(self.root))
        subprocess.run(['bash', '-c', command], env=env, check=True, capture_output=True, text=True)
        self.assertEqual((self.root / 'datasets/scout/videos/000.mp4').read_text(), 'archive content')
        self.assertEqual((self.root / 'datasets/wildtrack/camInfo/000.yml').read_text(), 'archive content')
        self.assertEqual((self.root / 'datasets/mtmc_4cam/videos/000.mp4').read_text(), 'existing local data')
        archive.unlink()
        subprocess.run(['bash', '-c', command], env=env, check=True, capture_output=True, text=True)

    def test_kafka_download_uses_current_mirror_then_archive_without_starting_services(self):
        fixture = self.root / 'broker.txt'
        fixture.write_text('Kafka fixture')
        archive = self.root / 'kafka.tgz'
        with tarfile.open(archive, 'w:gz') as bundle:
            bundle.add(fixture, arcname='kafka_2.13-4.2.0/bin/kafka-server-start.sh')
        commands = self.root / 'commands'
        commands.mkdir()
        wget = commands / 'wget'
        wget.write_text(f'''#!{sys.executable}
import os
import sys
import json
import shutil
from pathlib import Path
with open(os.environ['DOWNLOAD_LOG'], 'a') as output:
    output.write(json.dumps(sys.argv[1:]) + '\\n')
url = next(value for value in sys.argv if value.startswith('https://'))
destination = Path(sys.argv[sys.argv.index('-O') + 1])
destination.touch()
if os.environ['DOWNLOAD_SCENARIO'] == 'failed' or (os.environ['DOWNLOAD_SCENARIO'] == 'fallback' and 'dlcdn.apache.org' in url):
    sys.exit(8)
shutil.copyfile(os.environ['DOWNLOAD_FIXTURE'], destination)
''')
        wget.chmod(0o755)
        setup = (APP / 'scripts/setup_prerequisites.sh').read_text()
        block = setup[setup.index('    # Download and extract Kafka if not exists'):setup.index('    cd "$KAFKA_DIR"')]
        command = 'log_info() { :; }\nlog_success() { :; }\nlog_warning() { :; }\nlog_error() { :; }\nprepare_archive() {\n' + block + '\n}\nprepare_archive'
        for scenario, attempts in [('current', 1), ('fallback', 2), ('failed', 2)]:
            with self.subTest(scenario=scenario):
                base = self.root / scenario
                base.mkdir()
                log = base / 'downloads.jsonl'
                env = dict(os.environ, PATH=f'{commands}:{os.environ["PATH"]}', BASE_DIR=str(base),
                           KAFKA_VERSION='4.2.0', SCALA_VERSION='2.13', KAFKA_DIR=str(base / 'kafka_2.13-4.2.0'),
                           DOWNLOAD_FIXTURE=str(archive), DOWNLOAD_LOG=str(log), DOWNLOAD_SCENARIO=scenario)
                result = subprocess.run(['bash', '-c', command], cwd=self.root, env=env,
                                        check=False, capture_output=True, text=True)
                calls = [json.loads(line) for line in log.read_text().splitlines()]
                self.assertEqual(len(calls), attempts)
                self.assertIn('https://dlcdn.apache.org/kafka/4.2.0/kafka_2.13-4.2.0.tgz', calls[0])
                if attempts == 2:
                    self.assertIn('https://archive.apache.org/dist/kafka/4.2.0/kafka_2.13-4.2.0.tgz', calls[1])
                if scenario == 'failed':
                    self.assertNotEqual(result.returncode, 0)
                    self.assertFalse(Path(env['KAFKA_DIR']).exists())
                else:
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual((Path(env['KAFKA_DIR']) / 'bin/kafka-server-start.sh').read_text(), 'Kafka fixture')


if __name__ == '__main__':
    unittest.main()
