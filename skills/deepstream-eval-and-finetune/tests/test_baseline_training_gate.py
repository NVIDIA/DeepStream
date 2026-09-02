# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parents[1]
REGISTER = SKILL_ROOT / "scripts" / "ngc" / "ui_register.py"


class TestBaselineTrainingGate(unittest.TestCase):
    def _config(self, root):
        config = root / "config.yaml"
        config.write_text(
            "model_id: PekingU/rtdetr_r50vd\nnum_train_epochs: 30\n",
            encoding="utf-8",
        )
        return config

    def test_automl_registration_refuses_missing_baseline(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            result = subprocess.run(
                [
                    sys.executable, str(REGISTER), "--run", "demo",
                    "--config", str(self._config(root)), "--require-baseline",
                    "--index", "build/ui_runs/index.json",
                ],
                cwd=root, capture_output=True, text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("baseline is required before training", result.stderr)
            self.assertFalse((root / "build/ui_runs/index.json").exists())

    def test_registration_preserves_existing_baseline_path(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            orig = root / "models" / "exact_workload_baseline" / "eval"
            orig.mkdir(parents=True)
            (orig / "engine_metrics.json").write_text('{"map": 0.1}', encoding="utf-8")
            (orig / "perf.json").write_text('{"trtexec_qps": 100}', encoding="utf-8")
            index = root / "build" / "ui_runs" / "index.json"
            index.parent.mkdir(parents=True)
            index.write_text(json.dumps([{
                "key": "custom_demo",
                "orig_dir": "models/exact_workload_baseline",
                "comparison": "reports/existing_comparison.json",
            }]), encoding="utf-8")

            subprocess.run(
                [
                    sys.executable, str(REGISTER), "--run", "demo",
                    "--config", str(self._config(root)), "--require-baseline",
                    "--index", str(index),
                ],
                cwd=root, check=True, capture_output=True, text=True,
            )
            saved = json.loads(index.read_text(encoding="utf-8"))[0]
            self.assertEqual(saved["orig_dir"], "models/exact_workload_baseline")
            self.assertEqual(saved["comparison"], "reports/existing_comparison.json")

    def test_ngc_runners_apply_gate_before_creating_logs(self):
        for name in ("run_ngc_automl.sh", "run_ngc_finetune.sh"):
            script = (SKILL_ROOT / "scripts" / "ngc" / name).read_text()
            gate = script.index("--require-baseline")
            log_creation = script.index("mkdir -p", gate)
            self.assertLess(gate, log_creation, name)
            self.assertNotIn("--require-baseline 2>/dev/null || true", script)

    def test_backend_and_stepper_require_baseline_for_training(self):
        server = (SKILL_ROOT / "app" / "server.py").read_text()
        app_js = (SKILL_ROOT / "app" / "static" / "app.js").read_text()
        baseline_fn = server.split("def _has_baseline", 1)[1].split(
            "def _has_finetuned", 1
        )[0]
        self.assertIn("engine_metrics.json", baseline_fn)
        self.assertIn("perf.json", baseline_fn)
        self.assertIn("!!r.has_baseline && (", app_js)


if __name__ == "__main__":
    unittest.main()
