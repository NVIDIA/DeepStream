# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "ui_artifacts", SKILL_ROOT / "app" / "artifacts.py"
)
artifacts = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(artifacts)


class TestUIArtifactResolution(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.preset = {
            "orig_dir": "models/ui_baseline",
            "orig_dir_aliases": ["models/cli_baseline"],
        }

    def tearDown(self):
        self.tmp.cleanup()

    def _complete(self, model_dir, n_target_classes=None):
        metrics = os.path.join(self.root, model_dir, "eval", "engine_metrics.json")
        os.makedirs(os.path.dirname(metrics), exist_ok=True)
        Path(metrics).write_text("{}")
        if n_target_classes is None:
            return
        gt = os.path.join(self.root, model_dir, "eval", "eval_set", "ground_truth.json")
        os.makedirs(os.path.dirname(gt), exist_ok=True)
        Path(gt).write_text(json.dumps(
            {"category_map": {str(i): f"c{i}" for i in range(n_target_classes)}}))

    def test_falls_back_to_canonical_write_path_when_no_complete_artifact_exists(self):
        self.assertEqual(
            artifacts.resolve_orig_dir(self.root, self.preset),
            "models/ui_baseline",
        )

    def test_adopts_complete_cli_alias_when_canonical_is_missing(self):
        self._complete("models/cli_baseline")
        self.assertEqual(
            artifacts.resolve_orig_dir(self.root, self.preset),
            "models/cli_baseline",
        )

    def test_partial_canonical_directory_does_not_hide_complete_cli_alias(self):
        os.makedirs(os.path.join(self.root, "models", "ui_baseline", "model"))
        self._complete("models/cli_baseline")
        self.assertEqual(
            artifacts.resolve_orig_dir(self.root, self.preset),
            "models/cli_baseline",
        )

    def test_complete_canonical_artifact_has_precedence(self):
        self._complete("models/cli_baseline")
        self._complete("models/ui_baseline")
        self.assertEqual(
            artifacts.resolve_orig_dir(self.root, self.preset),
            "models/ui_baseline",
        )

    # --- cross-workload safety -------------------------------------------------------------
    # Built-in demos share a base model, so a model-derived alias is ambiguous between them:
    # whichever workload ran last owns the directory. Adopting a sibling's eval would report one
    # demo's baseline under the other (e.g. aerial-sheep's non-zero mAP shown as PCB's, whose
    # honest baseline is 0).

    def test_rejects_alias_whose_eval_targets_a_different_label_space(self):
        preset = dict(self.preset, ft_nclasses=6)
        self._complete("models/cli_baseline", n_target_classes=1)   # a sheep run left this behind
        self.assertEqual(
            artifacts.resolve_orig_dir(self.root, preset),
            "models/ui_baseline",                                   # refuse it, stay canonical
        )

    def test_adopts_alias_whose_eval_targets_the_expected_label_space(self):
        preset = dict(self.preset, ft_nclasses=6)
        self._complete("models/cli_baseline", n_target_classes=6)
        self.assertEqual(
            artifacts.resolve_orig_dir(self.root, preset),
            "models/cli_baseline",
        )

    def test_label_space_check_fails_open_when_ground_truth_is_absent(self):
        # Older artifacts predate the eval_set layout; never regress adoption on missing data.
        preset = dict(self.preset, ft_nclasses=6)
        self._complete("models/cli_baseline")
        self.assertEqual(
            artifacts.resolve_orig_dir(self.root, preset),
            "models/cli_baseline",
        )

    # --- malformed preset data ------------------------------------------------------------
    # Presets are user-editable, so the aliases value is not guaranteed to be a list.

    def test_string_alias_is_treated_as_one_path_not_split_into_characters(self):
        preset = {"orig_dir": "models/ui_baseline", "orig_dir_aliases": "models/cli_baseline"}
        self.assertEqual(
            artifacts.artifact_dir_candidates(preset, "orig_dir"),
            ["models/ui_baseline", "models/cli_baseline"],
        )

    def test_malformed_aliases_are_ignored_rather_than_raising(self):
        for bad in (None, 3, {"a": 1}, ["ok", 7, None]):
            with self.subTest(aliases=bad):
                preset = {"orig_dir": "models/ui_baseline", "orig_dir_aliases": bad}
                got = artifacts.artifact_dir_candidates(preset, "orig_dir")
                self.assertEqual(got[0], "models/ui_baseline")
                self.assertTrue(all(isinstance(p, str) for p in got))
                self.assertEqual(
                    artifacts.resolve_orig_dir(self.root, preset), "models/ui_baseline")

    def test_canonical_is_trusted_even_if_its_label_space_is_unreadable(self):
        preset = dict(self.preset, ft_nclasses=6)
        self._complete("models/ui_baseline", n_target_classes=1)
        self.assertEqual(
            artifacts.resolve_orig_dir(self.root, preset),
            "models/ui_baseline",
        )


if __name__ == "__main__":
    unittest.main()
