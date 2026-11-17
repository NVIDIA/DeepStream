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
"""Check dataset setup guidance and the launcher checks without downloads or real data."""

import os
import shutil
import zipfile
import unittest
import tempfile
import subprocess

from pathlib import Path
from utils import prepare_wildtrack as prep

APP = Path(__file__).resolve().parents[1]
SCRIPT = APP / 'scripts/setup_datasets.sh'
BASH = shutil.which('bash')
LAUNCHERS = ('test_scout_ds.sh', 'test_wildtrack_ds.sh')
# Git LFS object id of the archive that once carried SCOUT and WILDTRACK, it must never be referenced again.
SCOUT_WILDTRACK_LFS_OID = '8b1e27c4541e6878953fa1d752b19c1caaa9bf98a0cffcf506503a9fa56ba25c'


class SetupDatasetsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def run_script(self, *args, path=None, datasets=None):
        """Run the setup script. path replaces PATH, datasets replaces the output root."""
        environment = dict(os.environ, MV3DT_DATASETS_DIR=str(datasets or self.root / 'datasets'))
        if path is not None:
            environment['PATH'] = str(path)
        return subprocess.run([BASH, str(SCRIPT), *args], cwd=APP, capture_output=True, text=True, env=environment)

    def fake_tools(self, **scripts):
        """A folder of executable stand-ins named after the keywords, used as the only PATH entry."""
        folder = self.root / 'bin'
        folder.mkdir(exist_ok=True)
        for name, text in scripts.items():
            (folder / name).write_text('#!/bin/bash\n' + text)
            (folder / name).chmod(0o755)
        return folder

    def link_tools(self, *names):
        folder = self.fake_tools()
        for name in names:
            (folder / name).symlink_to(shutil.which(name))
        return folder

    def test_help_names_official_sources_and_terms_and_bad_options_are_rejected(self):
        result = self.run_script('--help')
        self.assertEqual(result.returncode, 0)
        for text in ('https://www.epfl.ch/labs/cvlab/data/data-wildtrack/', 'https://scout.epfl.ch/download', 'do not redistribute'):
            self.assertIn(text, result.stdout)
        self.assertEqual(self.run_script('--dataset', 'other').returncode, 2)
        self.assertEqual(self.run_script('--bogus').returncode, 2)

    def test_missing_downloads_get_actionable_messages(self):
        cases = {'scout': ('https://scout.epfl.ch/download', 'sequence_1', 'arXiv 2507.08494'),
                 'wildtrack': ('Wildtrack_dataset_full.zip',)}
        for name, texts in cases.items():
            with self.subTest(dataset=name):
                if name == 'wildtrack' and not shutil.which('ffmpeg'):
                    self.skipTest('the wildtrack run stops at the ffmpeg check without ffmpeg')
                result = self.run_script('--dataset', name)
                self.assertNotEqual(result.returncode, 0)
                for text in texts:
                    self.assertIn(text, result.stdout + result.stderr)

    def test_download_uses_the_same_https_zip_url_as_the_converter(self):
        self.assertIn(f'WILDTRACK_ZIP_URL="{prep.ZIP_URL}"', SCRIPT.read_text())

    def test_failed_download_or_extraction_aborts_with_an_error(self):
        cases = {'curl fails': ('exit 22\n', 'Downloading'),
                 'zip is not a zip': ('while [[ $# -gt 0 ]]; do [[ "$1" = -o ]] && { echo junk > "$2"; exit 0; }; shift; done\n', 'Extracting')}
        for name, (curl, message) in cases.items():
            with self.subTest(name):
                tools = self.fake_tools(curl=curl)
                path = f'{tools}:{os.environ["PATH"]}'
                result = self.run_script('--dataset', 'wildtrack', '--download', path=path,
                                         datasets=self.root / name.replace(' ', '_'))
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)
                self.assertNotIn('Generating datasets/wildtrack', result.stdout)

    def test_ffmpeg_is_needed_only_when_a_conversion_runs(self):
        source = self.root / 'datasets/raw/scout'
        (source / 'annotations').mkdir(parents=True)
        (source / 'images').mkdir()
        tools = self.link_tools('dirname', 'mkdir', 'cp')
        convert = self.run_script('--dataset', 'scout', path=tools)
        self.assertEqual(convert.returncode, 1)
        self.assertIn('ffmpeg is required', convert.stderr)
        (self.root / 'datasets/wildtrack/videos').mkdir(parents=True)
        copy = self.run_script('--dataset', 'wildtrack', '--detections-only', path=tools)
        self.assertEqual(copy.returncode, 0, copy.stderr)

    def test_no_third_party_dataset_in_archive_or_source(self):
        listing = subprocess.run(['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z', '--', '.'],
                                 cwd=APP, capture_output=True)
        if listing.returncode != 0:
            self.skipTest('not a git checkout')
        archive = APP / 'assets/datasets.zip'
        if archive.is_file() and zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as bundle:
                self.assertFalse([name for name in bundle.namelist() if name.startswith(('datasets/scout/', 'datasets/wildtrack/'))])
        files = [APP / name for name in listing.stdout.decode().split('\0') if name and (APP / name).is_file()]
        self.assertEqual([str(path) for path in files if path.suffix in ('.mp4', '.mkv', '.avi')], [])
        self.assertEqual([str(path) for path in files if path != Path(__file__).resolve() and path.suffix in ('.py', '.sh', '.md', '.yml')
                          and SCOUT_WILDTRACK_LFS_OID in path.read_text(errors='ignore')], [])

    def test_recorded_detections_match_the_generated_datasets(self):
        expected = {'scout': (8, 5980, ('PeopleNet2.6.3', 'PeopleNetTransformer')), 'wildtrack': (7, 196, ('PeopleNetTransformer',))}
        for name, (cameras, frames, models) in expected.items():
            for model in models:
                with self.subTest(dataset=name, model=model):
                    folder = APP / 'assets/detections' / name / model
                    files = sorted(folder.glob('*.txt'))
                    self.assertEqual([f.name for f in files], [f'{index:03d}.txt' for index in range(cameras)])
                    for path in files:
                        numbers = [int(line.split(',')[0]) for line in path.read_text().splitlines() if line.strip()]
                        self.assertTrue(numbers)
                        self.assertGreaterEqual(min(numbers), 1)
                        self.assertLessEqual(max(numbers), frames)

    def test_detections_only_copies_the_assets_and_never_overwrites(self):
        cases = {'wildtrack': ['PeopleNetTransformer/000.txt'],
                 'scout': ['PeopleNet2.6.3/000.txt', 'PeopleNetTransformer/007.txt']}
        for dataset, kept in cases.items():
            with self.subTest(dataset):
                datasets = self.root / dataset
                (datasets / dataset / 'videos').mkdir(parents=True)
                for relative in kept:
                    path = datasets / dataset / 'detections' / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text('mine\n')
                    self.assertNotEqual((APP / 'assets/detections' / dataset / relative).read_text(), 'mine\n')
                result = self.run_script('--dataset', dataset, '--detections-only', datasets=datasets)
                self.assertEqual(result.returncode, 0, result.stderr)
                source = APP / 'assets/detections' / dataset
                copied = datasets / dataset / 'detections'
                for original in source.glob('*/*.txt'):
                    relative = original.relative_to(source)
                    expected = b'mine\n' if str(relative) in kept else original.read_bytes()
                    self.assertEqual((copied / relative).read_bytes(), expected, str(relative))

    def test_detections_only_needs_a_generated_dataset(self):
        result = self.run_script('--dataset', 'scout', '--detections-only')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('has not been generated yet', result.stderr)

    def test_launchers_reject_an_invalid_replay_value_before_running(self):
        for launcher in LAUNCHERS:
            with self.subTest(launcher=launcher):
                result = subprocess.run([BASH, str(APP / 'scripts' / launcher)], cwd=APP, capture_output=True, text=True,
                                        env=dict(os.environ, REPLAY='2', DS_DIR=str(APP.parents[3])))
                self.assertEqual(result.returncode, 1)
                self.assertIn('REPLAY must be auto, 0, or 1', result.stderr)


if __name__ == '__main__':
    unittest.main()
