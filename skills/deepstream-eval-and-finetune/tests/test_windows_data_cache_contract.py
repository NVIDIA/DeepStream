# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parents[1]


class TestWindowsDataCacheContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.launch_sh = (SKILL_ROOT / "app" / "launch.sh").read_text()
        cls.launch_ps1 = (SKILL_ROOT / "app" / "launch.ps1").read_text()
        cls.pipeline = (SKILL_ROOT / "app" / "pipeline.py").read_text()
        cls.train = (
            SKILL_ROOT / "examples" / "rtdetr-aerial-sheep" / "assets" / "train.py"
        ).read_text()
        cls.index = (SKILL_ROOT / "app" / "static" / "index.html").read_text()
        cls.app_js = (SKILL_ROOT / "app" / "static" / "app.js").read_text()

    def test_both_launchers_mount_project_scoped_linux_cache(self):
        for text in (self.launch_sh, self.launch_ps1):
            self.assertIn("DS_DATA_CACHE", text)
            self.assertIn("DS_DATA_VOLUME", text)
            self.assertIn("DS_DATA_CACHE_ROOT=/work/data", text)
            self.assertIn(":/work/data", text)
            self.assertIn("docker volume create", text)
        self.assertIn("microsoft /proc/version", self.launch_sh)
        self.assertIn("SHA256", self.launch_ps1)
        self.assertIn("DS_WINDOWS_DATASET_MAP", self.launch_ps1)
        self.assertIn("/datasets/windows-", self.launch_ps1)

    def test_arrow_data_uses_configured_linux_path(self):
        self.assertIn('os.path.join(cache_root, ".arrow", key)', self.pipeline)
        self.assertIn("local_data_dir:", self.pipeline)
        self.assertIn('cfg.get("local_data_dir", "data")', self.train)

    def test_ui_exposes_cache_status_and_explicit_refresh(self):
        self.assertIn('id="cacheBox"', self.index)
        self.assertIn('id="f_refresh_cache"', self.index)
        self.assertIn("refresh_dataset_cache", self.app_js)
        self.assertIn("Reused across UI restarts", self.app_js)
        self.assertIn("rebuild this run", self.index)

    def test_skill_documents_persistent_results_and_cache_controls(self):
        skill = (SKILL_ROOT / "SKILL.md").read_text()
        windows = (SKILL_ROOT / "references" / "windows.md").read_text()
        for token in ("DS_DATA_CACHE=off", "DS_DATA_VOLUME", "DS_DATA_CACHE_REFRESH=1"):
            self.assertIn(token, skill)
            self.assertIn(token, windows)
        self.assertIn("Checkpoints, engines, metrics, logs, reports", skill)
        self.assertIn("persist across UI/container restarts", windows)


if __name__ == "__main__":
    unittest.main()
