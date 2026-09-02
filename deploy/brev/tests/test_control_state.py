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
"""Regression contracts for prompt selection, step gates, and widget locking."""

import contextlib
import importlib
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
os.environ.setdefault("OUTPUT_ROOT", tempfile.mkdtemp(prefix="brev-tests-output-"))
sys.path.insert(0, str(SCRIPTS))
lab = importlib.import_module("ds_agent_lab")


class FakeOutput:
    def __init__(self):
        self.clear_count = 0

    def clear_output(self):
        self.clear_count += 1

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeControl:
    def __init__(self, disabled=False):
        self.disabled = disabled


@contextlib.contextmanager
def successful_status(_label):
    status = SimpleNamespace(ok=False)
    yield status
    status.ok = True


@contextlib.contextmanager
def swallowing_status(_label):
    status = SimpleNamespace(ok=False)
    try:
        yield status
        status.ok = True
    except Exception:
        pass


class RunStepStateTests(unittest.TestCase):
    def setUp(self):
        self.saved_state = dict(lab.state)

    def tearDown(self):
        lab.state.clear()
        lab.state.update(self.saved_state)

    def test_missing_prerequisite_does_not_run_work(self):
        lab.state["generated"] = False
        called = []

        result = lab.run_step(
            FakeOutput(),
            FakeControl(),
            "Run",
            lambda: called.append(True),
            requires="generated",
        )

        self.assertFalse(result)
        self.assertEqual(called, [])

    def test_existing_container_agent_restores_installed_gate(self):
        lab.state["installed"] = False
        saved_agent = lab.AGENT
        lab.AGENT = ""
        called = []

        def probe(command, **_kwargs):
            installed = "command -v claude" in command[-1]
            return SimpleNamespace(returncode=0 if installed else 1)

        try:
            with (
                mock.patch.object(lab.subprocess, "run", side_effect=probe),
                mock.patch.object(lab, "step_status", successful_status),
            ):
                result = lab.run_step(
                    FakeOutput(),
                    FakeControl(),
                    "Generate",
                    lambda: called.append(True),
                    requires="installed",
                )
        finally:
            restored_agent = lab.AGENT
            lab.AGENT = saved_agent

        self.assertTrue(result)
        self.assertEqual(called, [True])
        self.assertTrue(lab.state["installed"])
        self.assertEqual(restored_agent, "claude")

    def test_stale_installed_state_does_not_bypass_container_probe(self):
        lab.state["installed"] = True
        called = []

        with mock.patch.object(lab, "_installed_agent_available", return_value=False):
            result = lab.run_step(
                FakeOutput(),
                FakeControl(),
                "Generate",
                lambda: called.append(True),
                requires="installed",
            )

        self.assertFalse(result)
        self.assertEqual(called, [])
        self.assertFalse(lab.state["installed"])

    def test_related_controls_stay_locked_for_the_full_operation(self):
        button = FakeControl()
        prompt = FakeControl()
        editor = FakeControl()

        def work():
            self.assertTrue(button.disabled)
            self.assertTrue(prompt.disabled)
            self.assertTrue(editor.disabled)

        with mock.patch.object(lab, "step_status", successful_status):
            result = lab.run_step(
                FakeOutput(),
                button,
                "Generate",
                work,
                controls=(prompt, editor),
            )

        self.assertTrue(result)
        self.assertFalse(button.disabled)
        self.assertFalse(prompt.disabled)
        self.assertFalse(editor.disabled)

    def test_control_states_are_restored_after_failure(self):
        button = FakeControl(disabled=False)
        prompt = FakeControl(disabled=False)
        already_disabled = FakeControl(disabled=True)

        with mock.patch.object(lab, "step_status", swallowing_status):
            result = lab.run_step(
                FakeOutput(),
                button,
                "Run",
                lambda: (_ for _ in ()).throw(RuntimeError("boom")),
                controls=(prompt, already_disabled),
            )

        self.assertFalse(result)
        self.assertFalse(button.disabled)
        self.assertFalse(prompt.disabled)
        self.assertTrue(already_disabled.disabled)

    def test_success_flag_is_set_only_after_successful_work(self):
        lab.state["generated"] = False
        with mock.patch.object(lab, "step_status", swallowing_status):
            lab.run_step(
                FakeOutput(),
                FakeControl(),
                "Generate",
                lambda: (_ for _ in ()).throw(RuntimeError("boom")),
                success_flag="generated",
            )
        self.assertFalse(lab.state["generated"])


class PromptSelectionStateTests(unittest.TestCase):
    def setUp(self):
        self.saved_state = dict(lab.state)
        self.saved_agent = lab.AGENT
        self.saved_selected = lab.selected
        self.saved_prompt_id = lab.SELECTED_PROMPT_ID
        self.saved_sequence = lab.selected_sequence

    def tearDown(self):
        lab.state.clear()
        lab.state.update(self.saved_state)
        lab.AGENT = self.saved_agent
        lab.selected = self.saved_selected
        lab.SELECTED_PROMPT_ID = self.saved_prompt_id
        lab.selected_sequence = self.saved_sequence

    def test_selecting_prompt_invalidates_previous_generate_success(self):
        lab.state["generated"] = True

        lab.set_selection("claude", "yolov26s_detection")

        self.assertEqual(lab.SELECTED_PROMPT_ID, "yolov26s_detection")
        self.assertFalse(lab.state["generated"])

    def test_agent_only_selection_does_not_invalidate_generated_result(self):
        lab.state["generated"] = True

        lab.set_selection("claude", None)

        self.assertTrue(lab.state["generated"])


if __name__ == "__main__":
    unittest.main()
