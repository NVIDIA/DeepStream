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
"""Check native profile templates and both configuration generators."""

import io
import yaml
import unittest
import tempfile
import contextlib
import configparser

from pathlib import Path
from unittest import mock
from utils.deepstream_auto_configurator import PROFILES, DeepStreamAutoConfigurator
from utils.inference_builder_auto_configurator import InferenceBuilderAutoConfigurator

APP = Path(__file__).resolve().parents[1]


class ConfigProfileTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / 'dataset/videos').mkdir(parents=True)
        (self.root / 'dataset/camInfo').mkdir()
        for index in range(2):
            (self.root / f'dataset/videos/{index:03d}.mp4').touch()
            (self.root / f'dataset/camInfo/{index:03d}.yml').touch()

    def generator(self, profile='mtmc_4cam_pnt', cls=DeepStreamAutoConfigurator):
        return cls(str(self.root / 'dataset'), str(self.root / profile / cls.__name__), profile)

    def test_template_resolution(self):
        generator = self.generator()
        generator.template_dir = self.root / 'templates'
        generator.profile_dir = generator.template_dir / 'selected'
        generator.profile_dir.mkdir(parents=True)
        (generator.template_dir / 'common').mkdir()
        (generator.template_dir / 'common/config_deepstream.txt').write_text('shared')
        path = generator.profile_dir / 'config_deepstream.txt'
        path.write_text('selected')
        self.assertEqual(generator.template_path(path.name), path)
        custom = self.root / 'custom.yml'
        custom.write_text('BaseConfig: {}\n')
        self.assertEqual(generator.template_path(str(custom)), custom)

    def test_unknown_profile_and_missing_template_fail(self):
        with self.assertRaises(ValueError):
            self.generator('unknown')
        with self.assertRaises(FileNotFoundError):
            self.generator().template_path('missing.yml')
        with self.assertRaises(FileNotFoundError):
            self.generator()._load_section_overrides('missing.yml')

    def test_all_profiles_generate_native_configs_for_both_backends(self):
        for cls in (DeepStreamAutoConfigurator, InferenceBuilderAutoConfigurator):
            for profile in PROFILES:
                with self.subTest(profile=profile, backend=cls.__name__):
                    generator = self.generator(profile, cls)
                    for filename in ('config_tracker.yml', 'config_pgie.txt'):
                        self.assertEqual(generator.template_path(filename).parent.name, profile)
                    for filename in ('config_deepstream.txt', 'config_tracker_2d.yml'):
                        expected = profile if (generator.profile_dir / filename).is_file() else 'common'
                        self.assertEqual(generator.template_path(filename).parent.name, expected)
                    with mock.patch.object(generator, 'get_video_resolution', return_value=(1920, 1080)), contextlib.redirect_stdout(io.StringIO()):
                        configs = generator.generate_configs(detector_config='config_pgie.txt')
                        generator.save_configs(configs)
                        generator.copy_static_configs('config_pgie.txt')
                        model = {'pn2': 'PeopleNet2.6.3', 'detr': 'RTDETR'}.get(profile.rsplit('_', 1)[-1], 'PeopleNetTransformer')
                        self.assertIn(model, generator.get_engine_file_from_detector_config('config_pgie.txt', 2))
                    detector = configparser.ConfigParser()
                    detector.read(generator.output_dir / 'config_pgie.txt')
                    self.assertIn(f'/models/{model}/', detector['property']['labelfile-path'])
                    for filename in ('config_tracker.yml', 'config_msgconv.txt', 'config_pgie.txt', 'config_mqtt.txt'):
                        self.assertTrue((generator.output_dir / filename).is_file())
                    if cls is DeepStreamAutoConfigurator:
                        self.assertIn('config-file=config_pgie.txt', configs['config_deepstream.txt'])
                        if profile == 'wildtrack':
                            self.assertIn('tracker-height=1088', configs['config_deepstream.txt'])
                    else:
                        for filename in ('source_list_static.yaml', 'nvdsinfer_config.yaml'):
                            self.assertIn(filename, configs)
                        self.assertEqual(yaml.safe_load(configs['ds_mv3dt.yaml'])['models'][0]['max_batch_size'], 2)
        self.assertTrue(self.generator('mtmc_12cam_detr').template_path('config_tracker_ib.yml').is_file())
        self.assertTrue(self.generator('mtmc_12cam_pnt').template_path('override_tracker_12cam.yml').is_file())
        self.assertTrue(self.generator('mtmc_4cam_pnt').template_path('override_tracker_4cam.yml').is_file())

    def test_all_templates_are_standard_yaml(self):
        for path in (APP / 'config_templates').rglob('*.yml'):
            text = path.read_text()
            self.assertNotIn('%YAML:1.0', text)
            self.assertIsInstance(yaml.safe_load(text), dict)


if __name__ == '__main__':
    unittest.main()
