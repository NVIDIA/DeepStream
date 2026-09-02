#!/usr/bin/env python3

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

"""
Installer safety tests for deepstream-eval-and-finetune.

Covers:
  - install.sh target validation, dry-run behavior, and safe in-place reinstall

Run:
    python3 -m unittest discover -s tests -v
"""
from __future__ import annotations

import base64
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parent.parent
SCRIPTS = SKILL_DIR / "scripts"
INSTALL_SH = SKILL_DIR / "install.sh"


def run_script(cmd, cwd=None, timeout=30):
    """Run a command and return (returncode, stdout, stderr)."""
    r = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=False,
        cwd=str(cwd) if cwd else None,
        timeout=timeout,
    )
    return r.returncode, r.stdout, r.stderr


class TestInstallScript(unittest.TestCase):
    """install.sh TARGET validation (B6)."""

    def test_rejects_missing_target_arg(self):
        rc, out, err = run_script(["bash", str(INSTALL_SH)])
        self.assertNotEqual(rc, 0)
        self.assertIn("--target is required", out + err)

    def test_rejects_filesystem_root(self):
        rc, out, err = run_script(["bash", str(INSTALL_SH), "--target", "/", "--dry-run"])
        self.assertNotEqual(rc, 0)
        self.assertIn("invalid --target", out + err)

    def test_rejects_path_traversal(self):
        rc, out, err = run_script(
            ["bash", str(INSTALL_SH), "--target", "/tmp/../etc", "--dry-run"]
        )
        self.assertNotEqual(rc, 0)
        self.assertIn("invalid --target", out + err)

    def test_rejects_nonexistent_target(self):
        rc, out, err = run_script(
            ["bash", str(INSTALL_SH), "--target", "/does/not/exist", "--dry-run"]
        )
        self.assertNotEqual(rc, 0)
        self.assertIn("target directory not found", out + err)

    def test_dry_run_previews_all_skill_locations(self):
        with tempfile.TemporaryDirectory() as td:
            rc, out, _ = run_script(
                ["bash", str(INSTALL_SH), "--target", td, "--dry-run", "--no-plugin"]
            )
            self.assertEqual(rc, 0, f"install.sh --dry-run failed: {out}")
            self.assertIn("[dry-run] cp -r", out)
            # The self-contained skill is previewed for every supported skill directory.
            self.assertIn(".claude/skills/deepstream-eval-and-finetune", out)
            self.assertIn(".codex/skills/deepstream-eval-and-finetune", out)
            # Cursor skill also installed
            self.assertIn(".cursor/skills/deepstream-eval-and-finetune", out)
            # Nothing must be written in dry-run mode
            self.assertEqual(os.listdir(td), [])


    def test_installer_never_touches_agent_settings(self):
        """Plugin provisioning goes through the `claude` CLI, which owns its own config.

        The installer must not read or rewrite settings.json — an existing file (even a
        malformed one) is left byte-for-byte untouched, and no new one is created. This is
        the contract the NVSkills-Eval AS1 'agent config directory access' check enforces.
        """
        with tempfile.TemporaryDirectory() as td:
            settings = Path(td) / ".claude" / "settings.json"
            settings.parent.mkdir(parents=True)
            settings.write_text("{ invalid json")
            # PATH without `claude` exercises the no-CLI branch.
            rc, out, err = run_script([
                "env", "PATH=/usr/bin:/bin", "bash", str(INSTALL_SH),
                "--target", td, "--no-cursor",
            ])
            self.assertEqual(rc, 0, f"stdout={out} stderr={err}")
            self.assertEqual(settings.read_text(), "{ invalid json")
            # The skill still installs, and the plugin commands are printed rather than applied.
            self.assertTrue((Path(td) / ".claude" / "skills" / SKILL_DIR.name / "SKILL.md").exists())
            self.assertIn("claude plugin install", out)

    def test_installer_creates_no_settings_file(self):
        """With no pre-existing settings.json, the installer must not create one."""
        with tempfile.TemporaryDirectory() as td:
            rc, out, err = run_script([
                "env", "PATH=/usr/bin:/bin", "bash", str(INSTALL_SH),
                "--target", td, "--no-cursor",
            ])
            self.assertEqual(rc, 0, f"stdout={out} stderr={err}")
            self.assertFalse((Path(td) / ".claude" / "settings.json").exists())

    def test_reinstall_from_installed_path_is_safe(self):
        with tempfile.TemporaryDirectory() as td:
            installed = Path(td) / ".claude" / "skills" / "deepstream-eval-and-finetune"
            shutil.copytree(SKILL_DIR, installed)
            marker = installed / "SKILL.md"
            rc, out, err = run_script([
                "bash", str(installed / "install.sh"),
                "--target", td, "--no-cursor", "--no-plugin",
            ])
            self.assertEqual(rc, 0, f"stdout={out} stderr={err}")
            self.assertTrue(marker.exists(), "in-place reinstall must not delete its source")
            self.assertIn("source and destination are identical", out)
