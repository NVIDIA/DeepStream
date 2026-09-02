# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import re
import unittest
from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parents[1]


class TestSkillIntakeOptions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.skill = (SKILL_ROOT / "SKILL.md").read_text()
        cls.run_flow = (SKILL_ROOT / "references" / "run-flow.md").read_text()
        cls.intake = cls.skill.split(
            "## Workload choice — always offer three options", 1
        )[1].split("## One path", 1)[0]
        cls.ui_choice = cls.skill.split(
            "## UI launch choice — always ask", 1
        )[1].split("## One path", 1)[0]
        cls.epoch_choice = cls.skill.split(
            "## Epoch choice — always ask", 1
        )[1].split("## UI launch choice", 1)[0]

    def test_skill_requires_exactly_three_ordered_options(self):
        self.assertEqual(
            re.findall(r"^### ([123])\. ", self.intake, flags=re.MULTILINE),
            ["1", "2", "3"],
        )
        self.assertRegex(self.intake, r"(?m)^### 1\. PCB Defects")
        self.assertRegex(self.intake, r"(?m)^### 2\. Aerial Sheep")

    def test_skill_stays_below_ci_token_budget_with_headroom(self):
        # SkillEvaluator enforces a 5,000-token ceiling. A conservative byte ceiling
        # prevents the top-level instructions from regressing close to that boundary;
        # detailed procedures belong in direct references.
        self.assertLessEqual(len(self.skill.encode("utf-8")), 18_000)

    def test_both_validated_demo_configs_are_present(self):
        self.assertIn("keremberke/aerial-sheep-object-detection", self.intake)
        self.assertIn("itsyoboieltr/pcb", self.intake)
        self.assertRegex(self.intake, r"Aerial Sheep[\s\S]+epochs: 30")
        self.assertRegex(self.intake, r"PCB Defects[\s\S]+epochs: 30")

    def test_custom_option_collects_required_inputs(self):
        custom = self.intake.split("### 3. Custom model and dataset", 1)[1]
        for field in ("model_id", "dataset source", "split", "eval-image count", "canvas", "precision"):
            self.assertIn(field, custom)

    def test_runbook_repeats_the_three_choice_gate(self):
        section = self.run_flow.split("## Intake — choose the workload", 1)[1].split(
            "## Sequence", 1
        )[0]
        workload_section = section.split("Then always ask the **epoch choice**", 1)[0]
        self.assertEqual(
            re.findall(r"^([123])\. \*\*", workload_section, flags=re.MULTILINE),
            ["1", "2", "3"],
        )
        self.assertRegex(workload_section, r"(?m)^1\. \*\*PCB Defects")
        self.assertRegex(workload_section, r"(?m)^2\. \*\*Aerial Sheep")
        # Assert the *instruction* survives, not one exact sentence: this file is prose and a
        # reworded-but-equivalent gate should not fail the build.
        self.assertRegex(section, r"(?i)do not collapse|do not (?:show|offer) only|present all three")

    def test_both_demo_presets_the_ui_ships_are_offered(self):
        """SKILL.md must not offer a workload the bundled UI has no preset for."""
        presets = json.loads(
            (SKILL_ROOT / "app" / "presets.json").read_text()
        )["presets"]
        for name, preset in presets.items():
            self.assertIn(
                preset["dataset_id"], self.intake,
                f"UI ships preset '{name}' but SKILL.md's intake never offers {preset['dataset_id']}",
            )

    def test_offered_epochs_match_the_ui_presets(self):
        """A user picking an option from SKILL.md should get what the UI would actually run."""
        presets = json.loads(
            (SKILL_ROOT / "app" / "presets.json").read_text()
        )["presets"]
        for name, preset in presets.items():
            block = self.intake.split(preset["dataset_id"], 1)[1]
            self.assertRegex(
                block, rf"epochs:\s*{preset['epochs']}\b",
                f"SKILL.md's epochs for '{name}' disagree with app/presets.json ({preset['epochs']})",
            )

    def test_skill_requires_explicit_start_or_skip_ui_choice(self):
        normalized = " ".join(self.ui_choice.split())
        self.assertIn("### A. Start the UI", self.ui_choice)
        self.assertIn("### B. Skip the UI", self.ui_choice)
        self.assertIn("Do not infer the answer", normalized)

    def test_ui_choice_preserves_other_sessions_and_dry_runs(self):
        normalized = " ".join(self.ui_choice.split())
        for requirement in (
            "never stop, replace, or rebind",
            "loopback-only",
            "semantic readiness",
            "For a dry run",
            "never launch or stop anything",
        ):
            self.assertIn(requirement, normalized)

    def test_runbook_repeats_ui_launch_gate(self):
        intake = self.run_flow.split("## Intake — choose the workload", 1)[1].split(
            "## Sequence", 1
        )[0]
        self.assertIn("Start UI", intake)
        self.assertIn("Skip UI", intake)
        self.assertIn("perform neither launch nor stop actions", intake)

    def test_skill_requires_default_or_custom_epoch_choice(self):
        self.assertEqual(
            re.findall(r"^([12])\. \*\*", self.epoch_choice, flags=re.MULTILINE),
            ["1", "2"],
        )
        normalized = " ".join(self.epoch_choice.split())
        self.assertIn("Default — 30 epochs", normalized)
        self.assertIn("Custom epoch count", normalized)
        self.assertIn("positive integer", normalized)
        self.assertIn("num_train_epochs", normalized)
        self.assertIn("FINAL_EPOCHS", normalized)

    def test_runbook_requires_epoch_choice(self):
        intake = self.run_flow.split("## Intake — choose the workload", 1)[1].split(
            "## Sequence", 1
        )[0]
        normalized = " ".join(intake.split())
        self.assertIn("Default — 30 epochs", normalized)
        self.assertIn("Custom epoch count", normalized)
        self.assertIn("FINAL_EPOCHS", normalized)

    def test_baseline_is_mandatory_for_both_training_lanes(self):
        normalized_skill = " ".join(self.skill.split())
        normalized_flow = " ".join(self.run_flow.split())
        for text in (normalized_skill, normalized_flow):
            self.assertIn("engine_metrics.json", text)
            self.assertIn("perf.json", text)
            self.assertRegex(text, r"(?i)AutoML.{0,160}(?:never|not).{0,80}(?:skip|bypass)")

    def test_ui_and_backend_default_to_30_epochs(self):
        index = (SKILL_ROOT / "app" / "static" / "index.html").read_text()
        app_js = (SKILL_ROOT / "app" / "static" / "app.js").read_text()
        server = (SKILL_ROOT / "app" / "server.py").read_text()
        pipeline = (SKILL_ROOT / "app" / "pipeline.py").read_text()
        self.assertRegex(index, r'id="f_epochs"[^>]+value="30"')
        self.assertNotRegex(app_js, r"epochs\s*\|\|\s*12")
        self.assertNotIn("epochs or 12", server)
        self.assertIn('preset.get("epochs", 30)', pipeline)

    def test_gates_have_a_non_interactive_fallback(self):
        """An unanswerable gate must fall back to a default, not stall.

        The intake gates assume someone can answer. Automated callers — an evaluation
        harness, a batch run — cannot, and blocking on input that never arrives fails the
        task outright. NVSkills-Eval Tier 3 scored 0.0 goal accuracy on exactly the trials
        that reach these gates, so this fallback is graded behaviour, not a nicety.
        """
        normalized = " ".join(self.skill.split())
        self.assertRegex(normalized, r"(?i)no answer can arrive|no interactive")
        self.assertRegex(normalized, r"(?i)do not stall")
        fallback = self.skill.split("### When no answer can arrive", 1)[1].split("## Inputs", 1)[0]
        for gate in ("Workload", "Epochs", "Fine-tune lane", "UI launch"):
            self.assertIn(gate, fallback, f"no documented default for the {gate!r} gate")
        self.assertIn("Skip the UI", fallback)   # never start a server nobody asked for
        self.assertIn("30", fallback)            # matches the offered epoch default

    def test_runbook_repeats_the_non_interactive_fallback(self):
        """The runbook is what the agent reads at intake time, so it must say this too."""
        normalized = " ".join(self.run_flow.split())
        self.assertRegex(normalized, r"(?i)no interactive caller|cannot be answered")
        self.assertRegex(normalized, r"(?i)recommended default")


if __name__ == "__main__":
    unittest.main()
