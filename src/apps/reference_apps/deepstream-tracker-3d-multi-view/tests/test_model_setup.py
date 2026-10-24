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
"""Check model setup and recovery without downloads, GPUs, or services."""

import os
import sys
import json
import shutil
import unittest
import tempfile
import subprocess

from pathlib import Path

APP = Path(__file__).resolve().parents[1]
MODELS = ('PeopleNetTransformer/peoplenet_transformer_model_op17.onnx',
          'RTDETR/rtdetr_warehouse_v1.0.fp16.onnx',
          'PeopleNet2.6.3/resnet34_peoplenet.onnx',
          'BodyPose3DNet/bodypose3dnet_accuracy.onnx', 'ClipReID/reid_model.onnx')


class ModelSetupTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / 'app files'
        (self.root / 'scripts').mkdir(parents=True)
        shutil.copy2(APP / 'scripts/setup_models.sh', self.root / 'scripts/setup_models.sh')
        self.commands = self.root / 'commands'
        self.commands.mkdir()
        self.log = self.root / 'commands.jsonl'
        stub = f'''#!{sys.executable}
import os
import sys
import json
from pathlib import Path
command = Path(sys.argv[0]).name
with open(os.environ['COMMAND_LOG'], 'a') as output:
    output.write(json.dumps([command, *sys.argv[1:]]) + '\\n')
if os.environ.get('FAIL_COMMAND') == command:
    sys.exit(7)
if command == 'wget':
    Path(sys.argv[sys.argv.index('-O') + 1]).write_text('downloaded model')
elif command == 'tar':
    source = Path(sys.argv[sys.argv.index('-C') + 1]) / 'model/make_model_clipreid.py'
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text('if get_image == True: pass\\n')
elif command == 'docker':
    export = Path(sys.argv[sys.argv.index('-v') + 1].removesuffix(':/work'))
    (export / 'reid_model.onnx').write_text('exported CLIP ReID')
'''
        for name in ('wget', 'sha256sum', 'tar', 'docker'):
            command = self.commands / name
            command.write_text(stub)
            command.chmod(0o755)

    def run_setup(self, **settings):
        self.log.write_text('')
        env = dict(os.environ, PATH=f'{self.commands}:{os.environ["PATH"]}',
                   COMMAND_LOG=str(self.log), MODEL_REPO=str(self.root / 'models'),
                   CLIP_REID_IMAGE='test-export-image:local')
        env.update(settings)
        result = subprocess.run(['bash', str(self.root / 'scripts/setup_models.sh')],
                                env=env, check=False, capture_output=True, text=True)
        return result, [json.loads(line) for line in self.log.read_text().splitlines()]

    def test_model_download_export_cache_and_recovery(self):
        result, calls = self.run_setup()
        self.assertEqual(result.returncode, 0, result.stderr)
        for filename in MODELS:
            self.assertGreater((self.root / 'models' / filename).stat().st_size, 0)
        docker = next(call for call in calls if call[0] == 'docker')
        self.assertEqual(docker[:7], ['docker', 'run', '--rm', '--gpus', 'all', '--network', 'bridge'])
        self.assertIn('test-export-image:local', docker)
        self.assertIn('python3 convert_clipreid_to_onnx.py', docker[-1])
        self.assertNotIn('--privileged', docker)
        result, calls = self.run_setup()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(all(call[0] == 'sha256sum' for call in calls))
        (self.root / 'models' / MODELS[0]).write_text('')
        result, calls = self.run_setup()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertGreater((self.root / 'models' / MODELS[0]).stat().st_size, 0)
        self.assertEqual(sum(call[0] == 'wget' for call in calls), 1)
        self.assertFalse(any(call[0] == 'docker' for call in calls))

    def test_download_checksum_and_export_errors_do_not_publish_models(self):
        for command, missing in [('wget', MODELS[0]), ('sha256sum', MODELS[2]), ('docker', MODELS[4])]:
            with self.subTest(command=command):
                destination = self.root / command
                result, _ = self.run_setup(FAIL_COMMAND=command, MODEL_REPO=str(destination))
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((destination / missing).exists())

    def test_readiness_rejects_missing_or_empty_onnx_not_just_missing_folders(self):
        for filename in (*MODELS, 'PeopleNetTransformer/detector_labels.txt', 'RTDETR/detector_labels.txt',
                         'PeopleNet2.6.3/detector_labels.txt', 'PeopleNetTransformer/custom_parser/libnvds_infercustomparser_tao.so',
                         'RTDETR/custom_parser/libnvds_infercustomparser_tao.so'):
            path = self.root / 'models' / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('asset')
        checker = (APP / 'scripts/check_prerequisites.sh').read_text()
        block = checker[checker.index('if [[ ! -d "$MV3DT_DIR/models" ]]'):checker.index('\n# Summary')]
        command = 'MV3DT_DIR="$PWD"\nDATASETS_MODELS_STATUS="✓"\n' + block + '\ntest "$DATASETS_MODELS_STATUS" = "✓"'
        result = subprocess.run(['bash', '-c', command], cwd=self.root, check=False, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        for filename in (MODELS[0], MODELS[4]):
            with self.subTest(filename=filename):
                path = self.root / 'models' / filename
                path.write_text('')
                result = subprocess.run(['bash', '-c', command], cwd=self.root, check=False, capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(filename, result.stdout)
                path.write_text('asset')


if __name__ == '__main__':
    unittest.main()
