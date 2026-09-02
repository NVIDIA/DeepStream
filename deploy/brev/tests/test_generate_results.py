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
"""Regression contracts for Generate execution and result rendering."""

import ast
import importlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
os.environ.setdefault("OUTPUT_ROOT", tempfile.mkdtemp(prefix="brev-tests-output-"))
sys.path.insert(0, str(SCRIPTS))
lab = importlib.import_module("ds_agent_lab")


def require_catalog_item(test_case, prompt_id):
    item = lab.catalog_by_id.get(prompt_id)
    if item is None:
        test_case.skipTest(f"prompt is not present in this partial checkout: {prompt_id}")
    return item


class GeneratePolicyTests(unittest.TestCase):
    def test_ordinary_generate_keeps_agent_autonomy(self):
        item = lab.catalog_by_id["video_infer_app"]
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.object(lab, "WORKSPACE", Path(tmp)),
            mock.patch.object(lab, "selected", item),
            mock.patch.object(lab, "SELECTED_PROMPT_ID", item["id"]),
            mock.patch.object(lab, "klog"),
        ):
            prompt = lab.build_agent_prompt()

        self.assertNotIn("Keep long-running commands in the foreground", prompt)
        self.assertNotIn("Do not schedule a wakeup", prompt)
        self.assertNotIn(lab.RUN_CONTRACT, prompt)

    def test_profile_generate_pins_the_measured_report_path(self):
        item = require_catalog_item(self, "ds_profiling_efficient_pipeline")
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.object(lab, "WORKSPACE", Path(tmp)),
            mock.patch.object(lab, "selected", item),
            mock.patch.object(lab, "SELECTED_PROMPT_ID", item["id"]),
            mock.patch.object(lab, "klog"),
        ):
            prompt = lab.build_agent_prompt()

        self.assertIn("profiling_report.txt", prompt)
        self.assertNotIn("foreground", prompt)

    def test_synchronous_claude_preserves_bash_input_and_forces_foreground(self):
        captured = {}

        def capture(cmd, timeout):
            captured["cmd"] = cmd
            captured["timeout"] = timeout
            return 0

        with (
            mock.patch.object(lab, "AGENT", "claude"),
            mock.patch.object(lab, "AGENT_CONTAINER", "test-agent"),
            mock.patch.object(lab, "ctr_run_out", "/workspace/agent_outputs/test"),
            mock.patch.object(lab, "build_cred_env", return_value=[]),
            mock.patch.object(lab, "_stream_agent_json", side_effect=capture),
            mock.patch.object(lab, "klog"),
        ):
            lab.run_agent(
                prompt_path="/workspace/agent_prompt.md",
                timeout=5400,
                synchronous=True,
            )

        command = captured["cmd"][-1]
        self.assertIn("BASH_MAX_TIMEOUT_MS=5400000", command)
        self.assertNotIn("--disallowedTools", command)
        argv = shlex.split(command)
        settings = json.loads(argv[argv.index("--settings") + 1])
        hook = settings["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
        event = {
            "tool_name": "Bash",
            "tool_input": {
                "command": "trtexec --saveEngine=model.engine",
                "timeout": 1800000,
                "run_in_background": True,
            },
        }
        result = subprocess.run(
            hook,
            shell=True,
            input=json.dumps(event),
            text=True,
            capture_output=True,
            check=True,
        )
        updated = json.loads(result.stdout)["hookSpecificOutput"]["updatedInput"]
        self.assertEqual(updated["command"], event["tool_input"]["command"])
        self.assertEqual(updated["timeout"], 1800000)
        self.assertIs(updated["run_in_background"], False)

    def test_combined_generate_displays_only_the_final_code_once(self):
        stages = ["rtvi_vlm_core_app", "rtvi_vlm_openapi_spec"]
        saved_selected = lab.selected
        saved_prompt_id = lab.SELECTED_PROMPT_ID
        saved_sequence = lab.selected_sequence

        try:
            lab.selected_sequence = stages
            lab.selected = lab.catalog_by_id[stages[0]]
            lab.SELECTED_PROMPT_ID = stages[0]
            with tempfile.TemporaryDirectory(prefix="brev-sequence-") as tmp:
                app = Path(tmp) / "rtvi_app"
                app.mkdir()
                (app / "main.py").write_text("print('combined app')\n")
                with (
                    mock.patch.object(lab, "WORKSPACE", Path(tmp)),
                    mock.patch.object(lab, "sync_workspace_to_container"),
                    mock.patch.object(
                        lab, "dexec_root", return_value=mock.Mock(returncode=0)
                    ),
                    mock.patch.object(lab, "build_agent_prompt"),
                    mock.patch.object(lab, "run_agent", return_value=0),
                    mock.patch.object(lab, "show_generated_code", return_value=True) as show,
                    mock.patch.object(lab, "klog"),
                    mock.patch.object(lab, "print_plan"),
                ):
                    lab.generate()
        finally:
            lab.selected = saved_selected
            lab.SELECTED_PROMPT_ID = saved_prompt_id
            lab.selected_sequence = saved_sequence

        show.assert_called_once_with()


class GenerateResultValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="brev-result-")
        self.workspace_patch = mock.patch.object(lab, "WORKSPACE", Path(self.tmp.name))
        self.model_patch = mock.patch.object(lab, "MODEL_NAME", "rtdetr_2d_warehouse")
        self.workspace_patch.start()
        self.model_patch.start()

    def tearDown(self):
        self.model_patch.stop()
        self.workspace_patch.stop()
        self.tmp.cleanup()

    def test_unrelated_model_pdf_does_not_complete_current_import(self):
        report = lab.WORKSPACE / "models/other_model/reports/benchmark_report_other.pdf"
        report.parent.mkdir(parents=True)
        report.write_bytes(b"%PDF-unrelated")
        item = lab.catalog_by_id["import_vision_model_detection_pipeline"]

        complete, _ = lab._generate_result_status(item)

        self.assertFalse(complete)

    def test_expected_benchmark_pdf_completes_current_import(self):
        report = (
            lab.WORKSPACE
            / "models/rtdetr_2d_warehouse/reports/benchmark_report_rtdetr_2d_warehouse.pdf"
        )
        report.parent.mkdir(parents=True)
        report.write_bytes(b"%PDF-current")
        item = lab.catalog_by_id["import_vision_model_detection_pipeline"]

        complete, _ = lab._generate_result_status(item)

        self.assertTrue(complete)

    def test_arbitrary_pdf_is_not_accepted_as_the_benchmark_report(self):
        report = lab.WORKSPACE / "models/rtdetr_2d_warehouse/reports/design_notes.pdf"
        report.parent.mkdir(parents=True)
        report.write_bytes(b"%PDF-not-a-benchmark")
        item = lab.catalog_by_id["import_vision_model_detection_pipeline"]

        complete, _ = lab._generate_result_status(item)

        self.assertFalse(complete)

    def test_incomplete_profiling_text_is_rejected(self):
        item = require_catalog_item(self, "ds_profiling_efficient_pipeline")
        report = lab.WORKSPACE / "agent_outputs" / item["id"] / "profiling_report.txt"
        report.parent.mkdir(parents=True)
        report.write_text("Profiling finished.\n")

        complete, _ = lab._generate_result_status(item)

        self.assertFalse(complete)

    def test_profiling_report_requires_requested_conclusions(self):
        item = require_catalog_item(self, "ds_profiling_efficient_pipeline")
        report = lab.WORKSPACE / "agent_outputs" / item["id"] / "profiling_report.txt"
        report.parent.mkdir(parents=True)
        report.write_text(
            "Bottleneck: decode\n"
            "Maximum streams at 30 FPS: 20\n"
            "Measured per-stream FPS: 30.4\n"
            "Hardware recommendation: upgrade NVDEC capacity.\n"
        )

        complete, _ = lab._generate_result_status(item)

        self.assertTrue(complete)


class RunCopyBoundaryTests(unittest.TestCase):
    def test_copy_keeps_application_artifacts_and_excludes_runtime_directories(self):
        retained = {
            "main.py": b"print('run')\n",
            "config/app.yaml": b"batch-size: 1\n",
            "models/detector.onnx": b"onnx",
            "models/detector.engine": b"engine",
            "parser/libnvdsparse.so": b"shared-object",
            "assets/prompt-specific.resource": b"resource",
        }
        excluded = (
            "venv/marker",
            ".venv/marker",
            "node_modules/marker",
            "__pycache__/marker",
            ".cache/marker",
            ".pytest_cache/marker",
            ".mypy_cache/marker",
            ".ruff_cache/marker",
            ".git/marker",
            "nested/venv/marker",
            "nested/.cache/marker",
        )

        with tempfile.TemporaryDirectory(prefix="brev-run-copy-") as tmp:
            root = Path(tmp)
            source = root / "generated"
            destination = root / "run-copy"
            for relative, content in retained.items():
                path = source / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            for relative in excluded:
                path = source / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("exclude me")
            destination.mkdir()
            (destination / "stale.txt").write_text("old run")

            subprocess.run(
                ["bash", "-lc", lab._run_copy_command(source, destination)],
                check=True,
                text=True,
                capture_output=True,
            )

            for relative, content in retained.items():
                self.assertEqual((destination / relative).read_bytes(), content)
            for relative in excluded:
                self.assertFalse((destination / relative).exists(), relative)
                self.assertTrue((source / relative).exists(), relative)
            self.assertFalse((destination / "stale.txt").exists())

    def test_run_and_fix_no_longer_uses_unbounded_cp(self):
        source = (SCRIPTS / "ds_agent_lab.py").read_text()
        run_and_fix = source[source.index("def run_and_fix():") : source.index("def run_and_view():")]

        self.assertNotIn("cp -a", run_and_fix)


class ResultRenderingTests(unittest.TestCase):
    def test_mp4_uses_simple_html_video_with_streamed_url(self):
        item = lab.catalog_by_id["yolov26s_detection"]
        with tempfile.TemporaryDirectory(prefix="brev-video-url-") as tmp:
            workspace = Path(tmp)
            video = workspace / "agent_outputs" / item["id"] / "out.mp4"
            video.parent.mkdir(parents=True)
            video.write_bytes(b"small-video")
            rendered = []

            class Marker:
                def __init__(self, kind, args, kwargs):
                    self.kind = kind
                    self.args = args
                    self.kwargs = kwargs

            with (
                mock.patch.object(lab, "WORKSPACE", workspace),
                mock.patch.object(lab, "selected", item),
                mock.patch.object(lab, "SELECTED_PROMPT_ID", item["id"]),
                mock.patch.object(lab, "sync_workspace_to_container"),
                mock.patch.object(lab, "_jupyter_relative_path", return_value=None),
                mock.patch.object(lab, "_serve_file", return_value="/files/out.mp4") as serve,
                mock.patch.object(lab, "emit_display", side_effect=rendered.append),
                mock.patch(
                    "IPython.display.Video",
                    side_effect=lambda *a, **k: Marker("video", a, k),
                ),
                mock.patch(
                    "IPython.display.HTML",
                    side_effect=lambda *a, **k: Marker("html", a, k),
                ),
            ):
                lab.show_results()

        players = [marker for marker in rendered if marker.kind == "html"]
        self.assertEqual(len(players), 1)
        markup = players[0].args[0]
        self.assertEqual(
            markup,
            '<video controls width="900" src="/files/out.mp4"></video>',
        )
        serve.assert_called_once_with(video)

    def test_mp4_never_falls_back_to_base64_when_url_is_unavailable(self):
        item = lab.catalog_by_id["yolov26s_detection"]
        with tempfile.TemporaryDirectory(prefix="brev-video-no-url-") as tmp:
            workspace = Path(tmp)
            video = workspace / "agent_outputs" / item["id"] / "out.mp4"
            video.parent.mkdir(parents=True)
            video.write_bytes(b"video")
            rendered = []

            with (
                mock.patch.object(lab, "WORKSPACE", workspace),
                mock.patch.object(lab, "selected", item),
                mock.patch.object(lab, "SELECTED_PROMPT_ID", item["id"]),
                mock.patch.object(lab, "sync_workspace_to_container"),
                mock.patch.object(lab, "_jupyter_relative_path", return_value=None),
                mock.patch.object(lab, "_serve_file", return_value=None),
                mock.patch.object(lab, "emit_display", side_effect=rendered.append),
                mock.patch("IPython.display.Video") as video_display,
            ):
                lab.show_results()

        video_display.assert_not_called()
        self.assertEqual(rendered, [])

    def test_show_results_has_no_call_to_removed_serve_helper(self):
        tree = ast.parse((SCRIPTS / "ds_agent_lab.py").read_text())
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_serve"
        ]
        self.assertEqual(calls, [])

    def test_pdf_is_rasterized_before_markdown_fallback(self):
        item = lab.catalog_by_id["import_vision_model_detection_pipeline"]
        with tempfile.TemporaryDirectory(prefix="brev-pdf-") as tmp:
            workspace = Path(tmp)
            report = (
                workspace
                / "models/rtdetr_2d_warehouse/reports/benchmark_report_rtdetr_2d_warehouse.pdf"
            )
            report.parent.mkdir(parents=True)
            report.write_bytes(b"%PDF-current")
            (report.parent / "benchmark_report.md").write_text("markdown fallback")
            rendered = []

            class Marker:
                def __init__(self, kind, *args, **kwargs):
                    self.kind = kind

            def fake_run(argv, **_kwargs):
                if argv and argv[0] == "pdftoppm":
                    Path(str(argv[-1]) + "-1.png").write_bytes(b"png")
                return SimpleCompleted()

            class SimpleCompleted:
                returncode = 0

            with (
                mock.patch.object(lab, "WORKSPACE", workspace),
                mock.patch.object(lab, "MODEL_NAME", "rtdetr_2d_warehouse"),
                mock.patch.object(lab, "selected", item),
                mock.patch.object(lab, "SELECTED_PROMPT_ID", item["id"]),
                mock.patch.object(lab, "sync_workspace_to_container"),
                mock.patch.object(lab, "_jupyter_relative_path", return_value=None),
                mock.patch.object(lab, "_serve_file", return_value="/files/report.pdf"),
                mock.patch.object(lab, "emit_display", side_effect=rendered.append),
                mock.patch("shutil.which", return_value="/usr/bin/pdftoppm"),
                mock.patch("subprocess.run", side_effect=fake_run),
                mock.patch("IPython.display.HTML", side_effect=lambda *a, **k: Marker("html")),
                mock.patch("IPython.display.Image", side_effect=lambda *a, **k: Marker("image")),
                mock.patch("IPython.display.Markdown", side_effect=lambda *a, **k: Marker("markdown")),
            ):
                lab.show_results()

        self.assertIn("image", [item.kind for item in rendered])
        self.assertNotIn("markdown", [item.kind for item in rendered])


if __name__ == "__main__":
    unittest.main()
