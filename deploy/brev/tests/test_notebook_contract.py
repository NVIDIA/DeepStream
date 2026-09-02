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
"""Regression contracts for notebook source, prompt availability, and served paths."""

import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import nbformat


BREV_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BREV_ROOT.parents[1]
SCRIPTS = BREV_ROOT / "scripts"
os.environ.setdefault("OUTPUT_ROOT", tempfile.mkdtemp(prefix="brev-tests-output-"))
sys.path.insert(0, str(SCRIPTS))
lab = importlib.import_module("ds_agent_lab")
build_notebook = importlib.import_module("build_notebook")


class NotebookSourceTests(unittest.TestCase):
    def test_checked_in_notebook_matches_build_source(self):
        notebook = nbformat.read(BREV_ROOT / "deepstream_code_agent_launchable.ipynb", as_version=4)
        expected = build_notebook.build_cells()

        self.assertEqual(len(notebook.cells), len(expected))
        for cell, (kind, source) in zip(notebook.cells, expected):
            self.assertEqual(cell.cell_type, "markdown" if kind == "md" else "code")
            self.assertEqual(cell.source, source.rstrip("\n"))

    def test_config_docs_explain_edit_and_checkpoint(self):
        docs = build_notebook.MD_CONFIG

        self.assertIn("only cell you normally edit", docs)
        self.assertIn("`OUTPUT_ROOT`", docs)
        self.assertIn("run **Install**", docs)
        self.assertIn("`Configuration set`", docs)

    def test_tutorial_explains_sequences_and_run_copy(self):
        docs = build_notebook.MD_STEP3 + build_notebook.MD_STEP4

        self.assertIn("two-stage workflow", docs)
        self.assertIn("both prompts in order", docs)
        self.assertIn("isolated", docs)
        self.assertIn("applied only to that copy", docs)

    def test_prompt_callbacks_invalidate_and_lock_cross_step_controls(self):
        source = build_notebook.CODE_STEP3 + build_notebook.CODE_STEP4

        self.assertIn("lab.set_selection(lab.AGENT, prompt_dd.value)", source)
        self.assertIn('lab.set_selection(lab.AGENT, change["new"])', source)
        self.assertIn('lab.state["generated"] = False', source)
        self.assertIn('prompt_tx.observe(_on_prompt_edit, names="value")', source)
        self.assertIn('globals().get("run_btn")', source)
        self.assertIn('globals().get("gen_btn")', source)
        self.assertEqual(source.count("controls=tuple("), 2)

    def test_run_control_tracks_generate_completion(self):
        source = build_notebook.CODE_STEP3 + build_notebook.CODE_STEP4

        self.assertIn("generated_ok = lab.run_step", source)
        self.assertIn("if run_control is not None:", source)
        self.assertIn("run_control.disabled = not generated_ok", source)
        self.assertIn('run_control.disabled = True', source)
        self.assertIn('disabled=not lab.state["generated"]', source)

    def test_default_prompt_has_public_subset_fallback(self):
        self.assertIn(
            '"video_infer_app" if "video_infer_app" in lab.MENU_PROMPT_IDS',
            build_notebook.CODE_STEP3,
        )


class PromptCatalogTests(unittest.TestCase):
    def test_full_checkout_catalog_matches_present_markdown_prompts(self):
        prompt_files = {path.stem for path in (REPO_ROOT / "example_prompts").glob("*.md")}
        self.assertEqual(set(lab.PROMPT_IDS), prompt_files)

    def test_public_subset_hides_missing_prompts_and_degrades_partial_sequence(self):
        with tempfile.TemporaryDirectory(prefix="brev-catalog-") as tmp:
            root = Path(tmp)
            scripts = root / "deploy/brev/scripts"
            prompts = root / "example_prompts"
            scripts.mkdir(parents=True)
            prompts.mkdir()
            shutil.copy2(SCRIPTS / "ds_lab_config.py", scripts)
            for name in ("rtvi_vlm_core_app.md", "video_infer_app.md"):
                shutil.copy2(REPO_ROOT / "example_prompts" / name, prompts)
            code = (
                "import json,sys; "
                f"sys.path.insert(0, {str(scripts)!r}); "
                "import ds_lab_config as c; "
                "print(json.dumps([c.PROMPT_IDS,c.MENU_PROMPT_IDS,c.PROMPT_SEQUENCES]))"
            )
            result = subprocess.run(
                [sys.executable, "-c", code],
                text=True,
                capture_output=True,
                check=True,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            prompt_ids, menu_ids, sequences = json.loads(result.stdout)

        self.assertEqual(prompt_ids, ["rtvi_vlm_core_app", "video_infer_app"])
        self.assertEqual(menu_ids, ["rtvi_vlm_core_app", "video_infer_app"])
        self.assertEqual(sequences, {})


class DeploymentScriptTests(unittest.TestCase):
    def test_brev_post_setup_replaces_legacy_bootstrap(self):
        post_setup = SCRIPTS / "brev_post_setup.sh"

        self.assertTrue(post_setup.is_file())
        self.assertFalse((SCRIPTS / "bootstrap_host_ubuntu2404.sh").exists())
        self.assertTrue(post_setup.read_text().startswith("#!/bin/bash\n"))
        self.assertIn("poppler-utils", post_setup.read_text())
        subprocess.run(["bash", "-n", str(post_setup)], check=True)

    def test_jupyter_readiness_avoids_redundant_notebook_url_probe(self):
        source = (SCRIPTS / "brev_post_setup.sh").read_text()

        self.assertIn("http://127.0.0.1:8888/api/status", source)
        self.assertNotIn("/api/contents/", source)
        self.assertNotIn("notebook_url", source)
        self.assertNotIn("from urllib.parse import quote", source)

    def test_jupyter_preparation_has_one_platform_independent_readiness_gate(self):
        source = (SCRIPTS / "brev_post_setup.sh").read_text()

        self.assertIn("prepare_jupyter_environment()", source)
        self.assertEqual(source.count("  prepare_jupyter_environment\n"), 1)
        self.assertNotIn("install_jupyter_widgets()", source)
        self.assertNotIn("restart_and_wait_for_jupyter()", source)
        self.assertNotIn("systemctl restart jupyter", source)
        self.assertNotIn("systemctl is-active --quiet jupyter", source)

    def test_post_setup_is_publishable_without_temporary_overlay(self):
        source = (SCRIPTS / "brev_post_setup.sh").read_text()

        self.assertNotIn("ds_launchable", source)
        self.assertNotIn("post-checkout", source)
        self.assertNotIn("init.templateDir", source)
        self.assertNotIn("DEEPSTREAM_REPO_URL", source)
        self.assertNotIn("LAUNCHABLE_REPO_URL", source)
        self.assertNotIn("git clone", source)

    def test_vlm_launcher_is_present_and_referenced_by_runtime(self):
        launcher = SCRIPTS / "serve_vlm.sh"
        runtime_source = (SCRIPTS / "ds_agent_lab.py").read_text()

        self.assertTrue(launcher.is_file())
        self.assertIn('"deploy/brev/scripts/serve_vlm.sh"', runtime_source)
        subprocess.run(["bash", "-n", str(launcher)], check=True)


class JupyterPathTests(unittest.TestCase):
    def test_repo_root_server_url(self):
        with tempfile.TemporaryDirectory(prefix="brev-repo-root-") as tmp:
            repo = Path(tmp) / "deepstream"
            artifact = repo / "deploy/brev/outputs/workspace/app/out.mp4"
            with mock.patch(
                "jupyter_server.serverapp.list_running_servers",
                return_value=[{"root_dir": str(repo)}],
            ):
                url = lab._jupyter_file_url(artifact)
        self.assertEqual(url, "/files/deploy/brev/outputs/workspace/app/out.mp4")

    def test_home_root_server_url_includes_checkout_directory(self):
        with tempfile.TemporaryDirectory(prefix="brev-home-root-") as tmp:
            home = Path(tmp)
            artifact = home / "deepstream/deploy/brev/outputs/workspace/app/out.mp4"
            with mock.patch(
                "jupyter_server.serverapp.list_running_servers",
                return_value=[{"root_dir": str(home)}],
            ):
                url = lab._jupyter_file_url(artifact)
        self.assertEqual(url, "/files/deepstream/deploy/brev/outputs/workspace/app/out.mp4")

    def test_file_url_escapes_path_and_can_download(self):
        with tempfile.TemporaryDirectory(prefix="brev-url-root-") as tmp:
            root = Path(tmp)
            artifact = root / "result files/report #1.pdf"
            with mock.patch(
                "jupyter_server.serverapp.list_running_servers",
                return_value=[{"root_dir": str(root)}],
            ):
                url = lab._jupyter_file_url(artifact, download=True)
        self.assertEqual(url, "/files/result%20files/report%20%231.pdf?download=1")


if __name__ == "__main__":
    unittest.main()
