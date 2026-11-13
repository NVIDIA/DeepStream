# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Packaging contract tests: synthetic files only, no GPU/download/install."""
import argparse
from contextlib import ExitStack
import json
import os
import re
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import prepare_samples as samples


OLD = "resnet18_trafficcamnet_pruned.onnx"
NEW = "resnet18_trafficcamnet_pruned_modelopt_fp16.onnx"
PACKAGED_TOOL_FILES = ("prepare_samples.py", "prepare_model.py", "requirements.txt", "README.md")


class InstalledRewriteTests(unittest.TestCase):
    """Installed canonical naming is distinct from legacy producer rewriting."""

    def test_ini_precision_sections_are_independent_and_idempotent(self):
        text = ("[half]\nonnx-file=" + OLD + "\nnetwork-mode=2\n"
                "[float]\nonnx-file=" + OLD + "\nnetwork-mode=0\n"
                "[int8]\nmodel-file=" + OLD + "\nnetwork-mode=1\n")
        expected = ("[half]\nonnx-file=" + OLD + "\nnetwork-mode=2\n"
                    "[float]\nonnx-file=fp32_" + OLD + "\nnetwork-mode=0\n"
                    "[int8]\nmodel-file=fp32_" + OLD + "\nnetwork-mode=1\n")
        self.assertEqual(samples.installed_rewrite_text(text), expected)
        self.assertEqual(samples.installed_rewrite_text(expected), expected)

    def test_yaml_precision_groups_are_independent(self):
        text = ("half:\n  onnx-file: " + OLD + "\n  network-mode: 2\n"
                "float:\n  onnx-file: " + OLD + "\n  network-mode: 0\n")
        result = samples.installed_rewrite_text(text)
        self.assertIn("half:\n  onnx-file: " + OLD + "\n", result)
        self.assertIn("float:\n  onnx-file: fp32_" + OLD + "\n", result)

    def test_unrelated_prefixed_and_suffixed_model_names_stay_untouched(self):
        for name in ["customer_" + OLD, "fp32_" + OLD, OLD + "_custom", "other.onnx"]:
            with self.subTest(name=name):
                text = "[property]\nonnx-file=" + name + "\nnetwork-mode=0\n"
                self.assertEqual(samples.installed_rewrite_text(text), text)

    def test_unsupported_or_conflicting_precision_fails_instead_of_silent_model_change(self):
        for declaration in ["network-mode=3\n", "network-mode=0\nnetwork-mode=2\n"]:
            with self.subTest(declaration=declaration):
                with self.assertRaisesRegex(RuntimeError, "Unsupported or conflicting network-mode"):
                    samples.installed_rewrite_text("[property]\nonnx-file=" + OLD + "\n" + declaration)

    def test_engine_only_gpu_dla_and_no_device_overrides_keep_precision_identity(self):
        for name in [OLD, NEW]:
            for device in ["", "_gpu2", "_dla1"]:
                for precision in ["fp16", "fp32", "int8"]:
                    with self.subTest(name=name, device=device, precision=precision):
                        tail = "_b16" + device + "_" + precision + ".engine"
                        text = 'ENGINE = "/samples/' + name + tail + '"\n'
                        selected = OLD if precision == "fp16" else "fp32_" + OLD
                        expected = 'ENGINE = "/samples/' + selected + tail + '"\n'
                        self.assertEqual(samples.installed_rewrite_text(text), expected)
                        self.assertEqual(samples.installed_rewrite_text(expected), expected)


class InstalledWorkflowTests(unittest.TestCase):
    """Synthetic post-install sequencing and publication, not GPU qualification."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.sdk = self.root / "sdk"
        self.cache = self.root / "cache"
        self.builder = self.sdk / "bin/nvdsinfer-engine-builder"
        self.builder.parent.mkdir(parents=True)
        self.builder.write_text("#!/bin/sh\nexit 0\n")
        self.builder.chmod(0o755)
        self.image = self.sdk / "samples/streams/sample_720p.jpg"
        self.image.parent.mkdir(parents=True)
        self.image.write_bytes(b"synthetic JPEG identity")
        self.originals = {}
        self.prepared = []
        for directory, _, original, size, *_ in samples.MODELS:
            source = self.sdk / "samples/models" / directory / original
            source.parent.mkdir(parents=True)
            source.write_bytes(b"Original FP32 fixture " + directory.encode())
            self.originals[source] = source.read_bytes()
            output = self.root / "converted" / directory
            output.mkdir(parents=True)
            model = output / samples.mixed_name(original)
            model.write_bytes(b"Prepared mixed fixture " + directory.encode())
            metadata = output / (model.stem + ".provenance.json")
            metadata.write_text(json.dumps({"conversion": {
                "request": {"source": {"model": samples.identity(source), "external_data": {}}},
                "external_io": {"inputs": [{"name": "input", "dtype": 1,
                    "shape": ["N", 3, size[1], size[0]]}], "outputs": []},
                "outputs": {model.name: samples.identity(model)},
            }}))
            self.prepared.append((directory, [model, metadata]))
        self.config = self.sdk / "samples/configs/example.txt"
        self.config.parent.mkdir(parents=True)
        self.config.write_text("[half]\nonnx-file=" + OLD + "\nnetwork-mode=2\n"
                               "[float]\nonnx-file=" + OLD + "\nnetwork-mode=0\n")
        self.config_bytes = self.config.read_bytes()

    def call(self, *flags):
        return samples.main(["--sdk-root", str(self.sdk), "--cache-dir", str(self.cache), *flags])

    def assert_originals_unchanged(self):
        for path, content in self.originals.items():
            self.assertEqual(path.read_bytes(), content)

    def assert_not_published(self):
        self.assert_originals_unchanged()
        self.assertEqual(self.config.read_bytes(), self.config_bytes)
        for directory, _, original, *_ in samples.MODELS:
            self.assertFalse((self.sdk / "samples/models" / directory / samples.mixed_name(original)).exists())
            self.assertFalse((self.sdk / "samples/models" / directory / ("fp32_" + original)).exists())

    def assert_canonical_models_published(self):
        for directory, files in self.prepared:
            original = next(model[2] for model in samples.MODELS if model[0] == directory)
            destination = self.sdk / "samples/models" / directory / original
            self.assertEqual(destination.read_bytes(), files[0].read_bytes())
            backup = destination.with_name("fp32_" + original)
            self.assertEqual(backup.read_bytes(), self.originals[destination])
            self.assertFalse(destination.with_name(samples.mixed_name(original)).exists(),
                             "The installed mixed graph must use its canonical sample filename")

    def installation_snapshot(self):
        # Locks are implementation artifacts rather than user model/config bytes.
        return {path.relative_to(self.sdk): path.read_bytes() for path in self.sdk.rglob("*")
                if path.is_file() and path.name != ".modelopt-setup.lock"}

    def test_check_validates_assets_without_dependencies_or_cache_writes(self):
        installation_before = {path.relative_to(self.sdk): path.read_bytes()
                               for path in self.sdk.rglob("*") if path.is_file()}
        with patch.object(samples, "tool_python", side_effect=AssertionError("check must not create tools")), \
                patch.object(samples, "prepare_models", side_effect=AssertionError("check must not convert")), \
                patch.object(samples, "build_installed_engines", side_effect=AssertionError("check must not build")):
            self.assertEqual(self.call("--check"), 0)
        self.assertFalse(self.cache.exists())
        self.assert_not_published()
        self.assertEqual({path.relative_to(self.sdk): path.read_bytes()
                          for path in self.sdk.rglob("*") if path.is_file()}, installation_before)

    def test_installed_sdk_root_is_inferred_from_packaged_script_location(self):
        installed_script = self.sdk / "samples/modelopt/prepare_samples.py"
        with patch.object(samples, "__file__", str(installed_script)), \
                patch.object(samples, "prepare_models", side_effect=AssertionError("check must not convert")):
            self.assertEqual(samples.main(["--check"]), 0)
        self.assert_not_published()

    def test_missing_original_fails_before_dependency_installation(self):
        next(iter(self.originals)).unlink()
        with patch.object(samples, "tool_python", side_effect=AssertionError("inputs must be checked first")), \
                patch.object(samples, "prepare_models", side_effect=AssertionError("must not convert")):
            self.assertEqual(self.call(), 1)
        self.assertEqual(self.config.read_bytes(), self.config_bytes)
        self.assertFalse(self.cache.exists())

    def test_missing_representative_image_fails_before_tools(self):
        self.image.unlink()
        with patch.object(samples, "tool_python", side_effect=AssertionError("inputs must be checked first")), \
                patch.object(samples, "prepare_models", side_effect=AssertionError("must not convert")):
            self.assertEqual(self.call(), 1)
        self.assert_not_published()
        self.assertFalse(self.cache.exists())

    def test_failed_conversion_does_not_rewrite_configs_or_publish_partial_models(self):
        with patch.object(samples, "prepare_models", side_effect=RuntimeError("synthetic conversion failure")), \
                patch.object(samples, "build_installed_engines") as builder, \
                patch.object(samples, "publish_installed") as publish:
            self.assertEqual(self.call(), 1)
        builder.assert_not_called()
        publish.assert_not_called()
        self.assert_not_published()

    def test_failed_engine_build_does_not_publish_successful_earlier_conversion(self):
        with patch.object(samples, "prepare_models", return_value=self.prepared), \
                patch.object(samples, "build_installed_engines", side_effect=RuntimeError("synthetic engine failure")), \
                patch.object(samples, "publish_installed") as publish:
            self.assertEqual(self.call(), 1)
        publish.assert_not_called()
        self.assert_not_published()

    def test_prepare_only_backs_up_fp32_and_publishes_mixed_at_canonical_name(self):
        self.builder.unlink()
        with patch.object(samples, "prepare_models", return_value=self.prepared), \
                patch.object(samples, "build_installed_engines", side_effect=AssertionError("prepare-only must not build")):
            self.assertEqual(self.call("--prepare-only"), 0)
        self.assert_canonical_models_published()
        self.assertIn("[half]\nonnx-file=" + OLD, self.config.read_text())
        self.assertIn("[float]\nonnx-file=fp32_" + OLD, self.config.read_text())
        for directory, files in self.prepared:
            for source in files[1:]:
                original = next(model[2] for model in samples.MODELS if model[0] == directory)
                destination = self.sdk / "samples/models" / directory / (original[:-5] + ".provenance.json")
                self.assertEqual(destination.read_bytes(), source.read_bytes())

    def test_fp16_configs_and_service_maker_engine_overrides_keep_canonical_names(self):
        configs = {}
        for relative in ["samples/configs/fp16.txt", "sources/apps/test/fp16.yml",
                         "service-maker/sources/apps/python/deepstream_test3.py"]:
            path = self.sdk / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("[property]\nonnx-file=" + OLD + "\nmodel-engine-file=" +
                            OLD + "_b16_gpu0_fp16.engine\nnetwork-mode=2\n")
            configs[path] = path.read_bytes()
        with patch.object(samples, "prepare_models", return_value=self.prepared):
            self.assertEqual(self.call("--prepare-only"), 0)
        for path, content in configs.items():
            self.assertEqual(path.read_bytes(), content, str(path))
        self.assert_canonical_models_published()

    def test_fp32_int8_and_unspecified_onnx_configs_select_original_backups(self):
        paths = []
        for base in ["samples/configs", "sources/apps", "service-maker/sources/apps"]:
            for label, mode in [("float", "network-mode=0\n"), ("int8", "network-mode=1\n"),
                                ("unspecified", "")]:
                path = self.sdk / base / (label + ".txt")
                path.parent.mkdir(parents=True, exist_ok=True)
                engine = OLD + "_b16_gpu0_" + ("int8" if label == "int8" else "fp32") + ".engine"
                path.write_text("[property]\nonnx-file=" + OLD + "\nmodel-engine-file=" + engine + "\n" + mode)
                paths.append((path, engine))
        with patch.object(samples, "prepare_models", return_value=self.prepared):
            self.assertEqual(self.call("--prepare-only"), 0)
        for path, engine in paths:
            self.assertIn("onnx-file=fp32_" + OLD + "\n", path.read_text())
            self.assertIn("model-engine-file=fp32_" + engine + "\n", path.read_text())
        self.assert_canonical_models_published()

    def test_previous_derived_fp16_config_names_return_to_canonical(self):
        path = self.sdk / "service-maker/sources/apps/previous.yml"
        path.parent.mkdir(parents=True)
        path.write_text("property:\n  onnx-file: " + NEW + "\n  model-engine-file: " +
                        NEW + "_b4_gpu1_fp16.engine\n  network-mode: 2\n")
        with patch.object(samples, "prepare_models", return_value=self.prepared):
            self.assertEqual(self.call("--prepare-only"), 0)
        self.assertIn("onnx-file: " + OLD + "\n", path.read_text())
        self.assertIn("model-engine-file: " + OLD + "_b4_gpu1_fp16.engine\n", path.read_text())
        self.assertNotIn("_modelopt_fp16", path.read_text())

    def test_unknown_existing_fp32_backup_is_never_overwritten(self):
        source = next(iter(self.originals))
        backup = source.with_name("fp32_" + source.name)
        backup.write_bytes(b"Unrelated user backup; not the original sample model")
        before = self.installation_snapshot()
        with patch.object(samples, "prepare_models", side_effect=AssertionError("conflict must fail before conversion")):
            self.assertEqual(self.call("--prepare-only"), 1)
        self.assertEqual(self.installation_snapshot(), before)

    def test_pristine_identical_preexisting_backup_can_be_preserved(self):
        source = next(iter(self.originals))
        backup = source.with_name("fp32_" + source.name)
        backup.write_bytes(source.read_bytes())
        with patch.object(samples, "prepare_models", return_value=self.prepared):
            self.assertEqual(self.call("--prepare-only"), 0)
        self.assert_canonical_models_published()

    def test_symlink_backup_is_rejected_even_when_target_has_original_bytes(self):
        source = next(iter(self.originals))
        external = self.root / "unrelated-original.onnx"
        external.write_bytes(source.read_bytes())
        source.with_name("fp32_" + source.name).symlink_to(external)
        before = self.installation_snapshot()
        with patch.object(samples, "prepare_models", side_effect=AssertionError("unsafe backup must fail first")):
            self.assertEqual(self.call("--prepare-only"), 1)
        self.assertEqual(self.installation_snapshot(), before)

    def test_repeated_conversion_reads_only_fp32_backup_never_canonical_mixed_graph(self):
        with patch.object(samples, "prepare_models", return_value=self.prepared):
            self.assertEqual(self.call("--prepare-only"), 0)
        calibration = self.root / "calibration"
        for directory, *_ in samples.MODELS:
            target = calibration / directory
            target.mkdir(parents=True)
            (target / "sample.npz").write_bytes(b"Synthetic NPZ; no model computation executed")
        selected = []
        def source_identity(_python, _tool_dir, source):
            selected.append(source)
            canonical = source.with_name(source.name.removeprefix("fp32_"))
            self.assertTrue(source.name.startswith("fp32_"))
            self.assertEqual(source.read_bytes(), self.originals[canonical])
            return {"model": samples.identity(source), "external_data": {}}
        def prepare(command):
            command = list(map(str, command))
            source = Path(command[command.index("--input") + 1])
            output_dir = Path(command[command.index("--output-dir") + 1])
            output_dir.mkdir(parents=True, exist_ok=True)
            output = output_dir / command[command.index("--output-name") + 1]
            output.write_bytes(b"Repeated mixed fixture from FP32 " + source.parent.name.encode())
            (output_dir / "provenance.json").write_text(json.dumps({
                "request": {"source": {"model": samples.identity(source), "external_data": {}}},
                "outputs": {output.name: samples.identity(output)},
                "external_io": {"inputs": [], "outputs": []}}))
        with patch.object(samples, "tool_python", return_value=Path("/synthetic/python")), \
                patch.object(samples, "source_identity", side_effect=source_identity), \
                patch.object(samples, "run", side_effect=prepare):
            self.assertEqual(self.call("--prepare-only", "--calibration-root", str(calibration)), 0)
        self.assertEqual(len(selected), 3)
        for canonical, content in self.originals.items():
            self.assertEqual(canonical.with_name("fp32_" + canonical.name).read_bytes(), content)
            self.assertTrue(canonical.read_bytes().startswith(b"Repeated mixed fixture from FP32 "))

    def test_missing_or_changed_backup_after_setup_fails_before_reconversion(self):
        with patch.object(samples, "prepare_models", return_value=self.prepared):
            self.assertEqual(self.call("--prepare-only"), 0)
        source = next(iter(self.originals))
        backup = source.with_name("fp32_" + source.name)
        for mutation in [lambda: backup.unlink(), lambda: backup.write_bytes(b"Changed FP32 backup")]:
            with self.subTest(mutation=mutation):
                backup.write_bytes(self.originals[source])
                mutation()
                before = self.installation_snapshot()
                with patch.object(samples, "prepare_models", side_effect=AssertionError("must not reconvert mixed model")):
                    self.assertEqual(self.call("--prepare-only"), 1)
                self.assertEqual(self.installation_snapshot(), before)

    def test_missing_canonical_with_valid_backup_and_provenance_can_be_regenerated(self):
        with patch.object(samples, "prepare_models", return_value=self.prepared):
            self.assertEqual(self.call("--prepare-only"), 0)
        source = next(iter(self.originals))
        source.unlink()
        with patch.object(samples, "prepare_models", return_value=self.prepared):
            self.assertEqual(self.call("--prepare-only"), 0)
        self.assert_canonical_models_published()

    def test_source_external_weights_are_retained_and_validated_on_repeated_setup(self):
        canonical = next(iter(self.originals))
        weights = canonical.parent / "weights.bin"
        weights.write_bytes(b"Original FP32 external tensor payload")
        metadata = self.prepared[0][1][-1]
        record = json.loads(metadata.read_text())
        record["conversion"]["request"]["source"]["external_data"] = {"weights.bin": samples.identity(weights)}
        metadata.write_text(json.dumps(record))
        with patch.object(samples, "prepare_models", return_value=self.prepared):
            self.assertEqual(self.call("--prepare-only"), 0)
            self.assertEqual(self.call("--prepare-only"), 0)
        self.assertEqual(weights.read_bytes(), b"Original FP32 external tensor payload")
        weights.write_bytes(b"Changed external tensor payload")
        before = self.installation_snapshot()
        with patch.object(samples, "prepare_models", side_effect=AssertionError("changed source closure must fail first")):
            self.assertEqual(self.call("--prepare-only"), 1)
        self.assertEqual(self.installation_snapshot(), before)

    def valid_external_original(self, payload_name, index=0):
        """Use a checker-valid source, not an engine's filename as its type."""
        try:
            import onnx
        except ImportError:
            self.skipTest("ONNX is required for real external-data publication fixtures")
        directory, files = self.prepared[index]
        original = next(model[2] for model in samples.MODELS if model[0] == directory)
        source = self.sdk / "samples/models" / directory / original
        payload = source.parent / payload_name
        payload.parent.mkdir(parents=True, exist_ok=True)
        payload.write_bytes(b"\0\0\x80?")
        weight = onnx.TensorProto(name="weight", data_type=onnx.TensorProto.FLOAT,
                                  dims=[1], data_location=onnx.TensorProto.EXTERNAL)
        for key, value in [("location", payload_name), ("offset", "0"), ("length", "4")]:
            weight.external_data.add(key=key, value=value)
        graph = onnx.helper.make_graph(
            [onnx.helper.make_node("Add", ["input", "weight"], ["output"])], "original",
            [onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1])],
            [onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1])], [weight])
        source.write_bytes(onnx.helper.make_model(
            graph, opset_imports=[onnx.helper.make_opsetid("", 13)]).SerializeToString())
        onnx.checker.check_model(str(source))
        self.originals[source] = source.read_bytes()
        record = json.loads(files[-1].read_text())
        record["conversion"]["request"]["source"] = {
            "model": samples.identity(source), "external_data": {payload_name: samples.identity(payload)}}
        files[-1].write_text(json.dumps(record))
        return source, payload

    def assert_source_collision_rejected_before_staging(self, source, payload, engines):
        import onnx
        before = self.installation_snapshot()
        with patch.object(samples.tempfile, "mkdtemp", wraps=samples.tempfile.mkdtemp) as stage:
            with self.assertRaisesRegex(RuntimeError, "External model data collides"):
                samples.publish_installed(self.sdk, self.prepared, engines)
        stage.assert_not_called()
        self.assertEqual(self.installation_snapshot(), before)
        self.assertEqual(list(self.sdk.rglob(".modelopt-*")), [])
        selected = samples.installed_source_model(self.sdk, source.parent.name, source.name)
        self.assertEqual(selected, source)
        self.assertEqual(onnx.load(str(selected)).graph.initializer[0].raw_data, payload.read_bytes())

    def test_engine_retirement_cannot_delete_valid_original_external_weights(self):
        original = samples.MODELS[0][2]
        source, payload = self.valid_external_original(original + "_b4_gpu0_fp16.engine")
        self.assert_source_collision_rejected_before_staging(source, payload, [])

    def test_new_engine_cannot_replace_valid_original_external_weights(self):
        original = samples.MODELS[0][2]
        source, payload = self.valid_external_original(original + "_b1_gpu0_fp16.engine")
        engine = self.root / "new.engine"
        engine.write_bytes(b"Synthetic new plan, not TensorRT execution")
        self.assert_source_collision_rejected_before_staging(
            source, payload, [(payload.relative_to(self.sdk), engine)])

    def test_engine_output_aliases_cannot_replace_nested_original_payload(self):
        for spelling in ("literal", "dotdot", "symlink-parent"):
            with self.subTest(spelling=spelling):
                case = InstalledWorkflowTests()
                case.setUp()
                self.addCleanup(case.doCleanups)
                source, payload = case.valid_external_original("weights/" + spelling + ".bin")
                destination = payload
                if spelling == "dotdot":
                    child = payload.parent / "child"
                    child.mkdir()
                    destination = child / ".." / payload.name
                elif spelling == "symlink-parent":
                    alias = source.parent / "payload-alias"
                    alias.symlink_to(payload.parent, target_is_directory=True)
                    destination = alias / payload.name
                engine = case.root / (spelling + ".engine")
                engine.write_bytes(b"Synthetic new plan, not TensorRT execution")
                case.assert_source_collision_rejected_before_staging(
                    source, payload, [(destination.relative_to(case.sdk), engine)])

    def test_publication_conservatively_rejects_hardlinked_payload_alias(self):
        source, payload = self.valid_external_original("weights/original.bin")
        destination = source.parent / "payload-hardlink.bin"
        os.link(payload, destination)
        engine = self.root / "new.engine"
        engine.write_bytes(b"Synthetic new plan, not TensorRT execution")
        before = self.installation_snapshot()
        with patch.object(samples.tempfile, "mkdtemp", wraps=samples.tempfile.mkdtemp) as stage:
            with self.assertRaisesRegex(RuntimeError, "External model data collides"):
                samples.publish_installed(self.sdk, self.prepared, [(destination.relative_to(self.sdk), engine)])
        stage.assert_not_called()
        self.assertEqual(self.installation_snapshot(), before)
        # ONNX itself rejects multi-link external data. Remove only this test's
        # alias after proving byte preservation, then verify the original loads.
        destination.unlink()
        import onnx
        self.assertEqual(onnx.load(str(source)).graph.initializer[0].raw_data, payload.read_bytes())

    def test_engine_output_checks_all_prepared_source_closures_before_staging(self):
        source, payload = self.valid_external_original("weights/nested.bin", index=-1)
        engine = self.root / "first-request.engine"
        engine.write_bytes(b"Synthetic first requested plan with last-model payload destination")
        self.assert_source_collision_rejected_before_staging(
            source, payload, [(payload.relative_to(self.sdk), engine)])

    def test_normal_nested_original_payload_survives_unrelated_engine_retirement(self):
        import onnx
        source, payload = self.valid_external_original("weights/original.bin")
        stale, retained = self.old_engine_fixtures()
        payload_bytes, source_bytes = payload.read_bytes(), source.read_bytes()
        samples.publish_installed(self.sdk, self.prepared, [])
        self.assertTrue(all(not path.exists() for path in stale))
        for path, content in retained.items():
            self.assertEqual(path.read_bytes(), content)
        selected = samples.installed_source_model(self.sdk, source.parent.name, source.name)
        self.assertEqual(selected, source.with_name(samples.fp32_name(source.name)))
        self.assertEqual(selected.read_bytes(), source_bytes)
        self.assertEqual(payload.read_bytes(), payload_bytes)
        self.assertEqual(onnx.load(str(selected)).graph.initializer[0].raw_data, payload_bytes)
        before = self.installation_snapshot()
        samples.publish_installed(self.sdk, self.prepared, [])
        self.assertEqual(self.installation_snapshot(), before)

    def test_publication_verifies_source_and_prepared_bytes_before_any_mutation(self):
        for label, target in [("original", next(iter(self.originals))), ("prepared", self.prepared[0][1][0])]:
            with self.subTest(target=label):
                content = target.read_bytes()
                target.write_bytes(content + b" changed since preparation")
                before = self.installation_snapshot()
                with self.assertRaises(RuntimeError):
                    samples.publish_installed(self.sdk, self.prepared, [])
                self.assertEqual(self.installation_snapshot(), before)
                target.write_bytes(content)

    def test_source_change_during_backup_staging_aborts_before_publishing_any_model(self):
        source = next(iter(self.originals))
        before = self.installation_snapshot()
        copyfile = samples.shutil.copyfile
        changed = False
        def copy_then_change(original, destination, *args, **kwargs):
            nonlocal changed
            result = copyfile(original, destination, *args, **kwargs)
            if Path(original) == source and not changed:
                changed = True
                source.write_bytes(b"Concurrent external source edit; not a model setup write")
            return result
        with patch.object(samples.shutil, "copyfile", side_effect=copy_then_change):
            with self.assertRaises(RuntimeError):
                samples.publish_installed(self.sdk, self.prepared, [])
        self.assertTrue(changed)
        expected = dict(before)
        expected[source.relative_to(self.sdk)] = source.read_bytes()
        self.assertEqual(self.installation_snapshot(), expected)
        self.assertEqual(list(self.sdk.rglob(".modelopt-*")), [])

    def test_corrupt_staged_backup_is_rejected_without_replacing_original(self):
        source = next(iter(self.originals))
        before = self.installation_snapshot()
        copyfile = samples.shutil.copyfile
        changed = False
        def copy_then_corrupt(original, destination, *args, **kwargs):
            nonlocal changed
            result = copyfile(original, destination, *args, **kwargs)
            if Path(original) == source and not changed:
                changed = True
                Path(destination).write_bytes(b"Synthetic copy corruption")
            return result
        with patch.object(samples.shutil, "copyfile", side_effect=copy_then_corrupt):
            with self.assertRaises(RuntimeError):
                samples.publish_installed(self.sdk, self.prepared, [])
        self.assertTrue(changed)
        self.assertEqual(self.installation_snapshot(), before)
        self.assertEqual(list(self.sdk.rglob(".modelopt-*")), [])

    def test_invalid_batch_and_gpu_selection_do_not_prepare(self):
        for flags in [("--batch-size", "0"), ("--batch-size", "-1"), ("--gpu-id", "-1")]:
            with self.subTest(flags=flags), patch.object(samples, "prepare_models") as prepare:
                try:
                    result = self.call(*flags)
                except SystemExit as error:
                    result = error.code
                self.assertNotEqual(result, 0)
                prepare.assert_not_called()
        self.assert_not_published()

    def test_builder_uses_sdk_fp16_contract_and_full_colon_named_shapes(self):
        """--fp16 is the SDK builder assertion, not a removed TRT builder flag."""
        for _, files in self.prepared:
            path = files[-1]
            metadata = json.loads(path.read_text())
            metadata["conversion"]["external_io"]["inputs"][0]["name"] = "input_1:0"
            path.write_text(json.dumps(metadata))
        self.cache.mkdir()
        args = argparse.Namespace(builder=str(self.builder), batch_size=4, gpu_id=2)
        commands = []
        def produce_engine(command):
            commands.append(list(map(str, command)))
            output = next(str(value).split("=", 1)[1] for value in command if str(value).startswith("--saveEngine="))
            Path(output).write_bytes(b"Synthetic serialized engine fixture")
        with patch.object(samples.subprocess, "check_output", return_value="TensorRT 11.4.0.106\n"), \
                patch.object(samples, "run", side_effect=produce_engine):
            engines = samples.build_installed_engines(args, self.prepared, self.cache)
        self.assertEqual(len(engines), len(samples.MODELS))
        self.assertEqual(len(commands), len(samples.MODELS))
        for command, (_, _, original, size, *_), (relative, engine) in zip(commands, samples.MODELS, engines):
            self.assertEqual(command[0], str(self.builder))
            self.assertIn("--fp16", command)
            self.assertIn("--device=2", command)
            for selector, batch in [("minShapes", 1), ("optShapes", 4), ("maxShapes", 4)]:
                self.assertIn("--%s=input_1:0:%dx3x%dx%d" % (selector, batch, size[1], size[0]), command)
            self.assertTrue(str(relative).endswith(original + "_b4_gpu2_fp16.engine"))
            self.assertTrue(engine.is_file())
            self.assertFalse((self.sdk / relative).exists(), "Building must not publish into the installation")
        self.assert_not_published()

    def test_non_batch_dynamic_shape_is_rejected_before_engine_build(self):
        metadata = self.prepared[0][1][-1]
        value = json.loads(metadata.read_text())
        value["conversion"]["external_io"]["inputs"][0]["shape"][2] = "H"
        metadata.write_text(json.dumps(value))
        self.cache.mkdir()
        args = argparse.Namespace(builder=str(self.builder), batch_size=1, gpu_id=0)
        with patch.object(samples.subprocess, "check_output", return_value="TensorRT 11.4.1.31\n"), \
                patch.object(samples, "run") as run:
            with self.assertRaisesRegex(RuntimeError, "dynamic batch"):
                samples.build_installed_engines(args, self.prepared, self.cache)
        run.assert_not_called()
        self.assert_not_published()

    def test_successful_default_setup_builds_then_publishes_all_three_engines(self):
        def produce_engine(command):
            output = next(str(value).split("=", 1)[1] for value in command if str(value).startswith("--saveEngine="))
            Path(output).write_bytes(b"Synthetic engine, not TensorRT execution")
            self.assert_not_published()
        with patch.object(samples, "prepare_models", return_value=self.prepared), \
                patch.object(samples.subprocess, "check_output", return_value="TensorRT 11.4.1.31\n"), \
                patch.object(samples, "run", side_effect=produce_engine):
            self.assertEqual(self.call("--batch-size", "2", "--gpu-id", "1"), 0)
        self.assert_canonical_models_published()
        for directory, _, original, *_ in samples.MODELS:
            engine = self.sdk / "samples/models" / directory / (original + "_b2_gpu1_fp16.engine")
            self.assertEqual(engine.read_bytes(), b"Synthetic engine, not TensorRT execution")
        self.assertIn("[half]\nonnx-file=" + OLD, self.config.read_text())
        self.assertIn("[float]\nonnx-file=fp32_" + OLD, self.config.read_text())
        self.assertEqual(list(self.cache.glob("engine-build-*")), [], "Published engine build scratch must be removed")

    def test_second_engine_build_failure_leaves_all_original_models_and_configs_untouched(self):
        before = self.installation_snapshot()
        built = []
        def produce_engine(command):
            self.assertEqual(self.installation_snapshot(), before,
                             "No models/backups/configs may be published until every engine builds")
            built.append(command)
            if len(built) == 2:
                raise RuntimeError("Synthetic second-model build failure")
            output = next(str(value).split("=", 1)[1] for value in command if str(value).startswith("--saveEngine="))
            Path(output).write_bytes(b"One successful private engine before later failure")
        with patch.object(samples, "prepare_models", return_value=self.prepared), \
                patch.object(samples.subprocess, "check_output", return_value="TensorRT 11.4.1.31\n"), \
                patch.object(samples, "run", side_effect=produce_engine):
            self.assertEqual(self.call(), 1)
        self.assertEqual(len(built), 2)
        self.assertEqual(self.installation_snapshot(), before)
        self.assertEqual(list(self.cache.glob("engine-build-*")), [])

    def test_wrong_builder_major_is_rejected_before_any_engine_command(self):
        self.cache.mkdir()
        args = argparse.Namespace(builder=str(self.builder), batch_size=1, gpu_id=0)
        with patch.object(samples.subprocess, "check_output", return_value="TensorRT 10.16.0.72\n"), \
                patch.object(samples, "run") as run:
            with self.assertRaisesRegex(RuntimeError, "TensorRT 11"):
                samples.build_installed_engines(args, self.prepared, self.cache)
        run.assert_not_called()
        self.assert_not_published()

    def test_publication_error_rolls_back_previous_outputs_and_configs(self):
        directory, files = self.prepared[0]
        original = next(model[2] for model in samples.MODELS if model[0] == directory)
        old_model = self.sdk / "samples/models" / directory / original
        before = self.installation_snapshot()
        replace = samples.os.replace
        calls = []
        def fail_once(source, destination):
            calls.append((source, destination))
            if len(calls) == 3:
                raise OSError("Synthetic rename failure")
            return replace(source, destination)
        with patch.object(samples.os, "replace", side_effect=fail_once):
            with self.assertRaisesRegex(OSError, "Synthetic rename failure"):
                samples.publish_installed(self.sdk, self.prepared, [])
        self.assertEqual(self.installation_snapshot(), before)
        self.assertEqual(list(self.sdk.rglob(".modelopt-*")), [])

    def test_keyboard_interrupt_after_commit_rolls_back_models_configs_and_engines(self):
        directory, files = self.prepared[0]
        original = next(model[2] for model in samples.MODELS if model[0] == directory)
        old_model = self.sdk / "samples/models" / directory / original
        stale, original_engines = self.old_engine_fixtures()
        stale_contents = {path: path.read_bytes() for path in stale}
        before = self.installation_snapshot()
        replace = samples.os.replace
        interrupted = False
        committed = []
        def interrupt_config_once(source, destination):
            nonlocal interrupted
            if Path(destination) == self.config and Path(source).name == "new" and not interrupted:
                interrupted = True
                raise KeyboardInterrupt("Synthetic interrupt after model commit")
            result = replace(source, destination)
            if Path(source).name == "new":
                committed.append(Path(destination))
            return result
        with patch.object(samples.os, "replace", side_effect=interrupt_config_once):
            with self.assertRaisesRegex(KeyboardInterrupt, "Synthetic interrupt"):
                samples.publish_installed(self.sdk, self.prepared, [])
        self.assertTrue(interrupted)
        self.assertIn(old_model, committed, "The interrupt must occur after replacing an existing model")
        self.assertEqual(self.installation_snapshot(), before)
        self.assertEqual(list(self.sdk.rglob(".modelopt-*")), [])

    def test_interrupt_after_successful_backup_rename_restores_unrecorded_original(self):
        directory, files = self.prepared[0]
        original = next(model[2] for model in samples.MODELS if model[0] == directory)
        old_model = self.sdk / "samples/models" / directory / original
        before = self.installation_snapshot()
        replace = samples.os.replace
        interrupted = False
        backup_was_real = False
        def interrupt_after_backup(source, destination):
            nonlocal interrupted, backup_was_real
            result = replace(source, destination)
            if Path(source) == old_model and Path(destination).name == "old" and not interrupted:
                backup_was_real = Path(destination).is_file() and not old_model.exists()
                interrupted = True
                raise KeyboardInterrupt("Synthetic interrupt after successful backup rename")
            return result
        with patch.object(samples.os, "replace", side_effect=interrupt_after_backup):
            with self.assertRaisesRegex(KeyboardInterrupt, "successful backup rename"):
                samples.publish_installed(self.sdk, self.prepared, [])
        self.assertTrue(interrupted)
        self.assertTrue(backup_was_real, "The real rename must precede the injected bookkeeping-window interrupt")
        self.assertEqual(self.installation_snapshot(), before)
        self.assertEqual(list(self.sdk.rglob(".modelopt-*")), [])

    def test_rollback_restore_failure_retains_only_recoverable_original_and_reports_path(self):
        directory, files = self.prepared[0]
        original = next(model[2] for model in samples.MODELS if model[0] == directory)
        old_model = self.sdk / "samples/models" / directory / original
        before = self.installation_snapshot()
        replace = samples.os.replace
        failed_publication = False
        backups = []
        def fail_publication_then_restore(source, destination):
            nonlocal failed_publication
            if Path(destination) == self.config and Path(source).name == "new" and not failed_publication:
                failed_publication = True
                raise OSError("Synthetic publication failure")
            if Path(destination) == old_model and Path(source).name == "old":
                backups.append(Path(source).parent)
                raise OSError("Synthetic filesystem failure during rollback")
            return replace(source, destination)
        with patch.object(samples.os, "replace", side_effect=fail_publication_then_restore):
            with self.assertRaisesRegex(RuntimeError, "manual recovery; retained backups") as raised:
                samples.publish_installed(self.sdk, self.prepared, [])
        self.assertTrue(failed_publication)
        self.assertEqual(len(backups), 1)
        private = backups[0]
        self.assertIn(str(private), str(raised.exception))
        self.assertEqual((private / "old").read_bytes(), before[old_model.relative_to(self.sdk)])
        self.assertTrue(old_model.is_file(), "Failed restore must not silently remove the replacement either")
        self.assertEqual(old_model.read_bytes(), files[0].read_bytes())
        self.assertEqual(self.config.read_bytes(), self.config_bytes)
        for path, content in self.originals.items():
            if path != old_model:
                self.assertEqual(path.read_bytes(), content)
        self.assertEqual(list(self.sdk.rglob(".modelopt-*")), [private])
        for directory, prepared_files in self.prepared:
            for source in prepared_files:
                destination = self.sdk / "samples/models" / directory / source.name
                if destination != old_model:
                    self.assertFalse(destination.exists())

    def old_engine_fixtures(self):
        stale, originals = [], {}
        for directory, _, original, *_ in samples.MODELS:
            destination = self.sdk / "samples/models" / directory
            for name in [original, samples.mixed_name(original)]:
                for suffix in ["_b16_gpu0_fp16.engine", "_b4_fp16.engine", "_b2_dla0_fp16.engine"]:
                    path = destination / (name + suffix)
                    path.write_bytes(b"Stale engine from previous sample model bytes")
                    stale.append(path)
            for precision in ["fp32", "int8"]:
                path = destination / (original + "_b16_gpu0_" + precision + ".engine")
                path.write_bytes(b"Unrelated original " + precision.encode() + b" engine")
                originals[path] = path.read_bytes()
            unrelated = destination / "customer_modelopt_fp16.onnx_b4_gpu0_fp16.engine"
            unrelated.write_bytes(b"Customer engine outside the three sample contracts")
            originals[unrelated] = unrelated.read_bytes()
        return stale, originals

    def test_prepare_only_retires_canonical_and_derived_fp16_engines_not_other_precision_or_customer(self):
        stale, engines = self.old_engine_fixtures()
        with patch.object(samples, "prepare_models", return_value=self.prepared):
            self.assertEqual(self.call("--prepare-only"), 0)
        for path in stale:
            self.assertFalse(path.exists(), "Prepared selectors must not reuse an engine from old model bytes")
        for path, content in engines.items():
            self.assertEqual(path.read_bytes(), content)
        self.assert_canonical_models_published()

    def test_unchanged_repeat_preserves_valid_other_batch_engine_and_original_backups(self):
        with patch.object(samples, "prepare_models", return_value=self.prepared):
            self.assertEqual(self.call("--prepare-only"), 0)
        source = next(iter(self.originals))
        plan = source.with_name(source.name + "_b16_gpu2_fp16.engine")
        plan.write_bytes(b"Valid matching-graph engine created by nvinfer after first setup")
        before = self.installation_snapshot()
        with patch.object(samples, "prepare_models", return_value=self.prepared):
            self.assertEqual(self.call("--prepare-only"), 0)
        self.assertEqual(self.installation_snapshot(), before)

    def test_changed_conversion_retires_stale_plans_without_overwriting_fp32_backups(self):
        with patch.object(samples, "prepare_models", return_value=self.prepared):
            self.assertEqual(self.call("--prepare-only"), 0)
        stale, retained = self.old_engine_fixtures()
        for _, files in self.prepared:
            files[0].write_bytes(files[0].read_bytes() + b" new conversion")
            record = json.loads(files[1].read_text())
            record["conversion"]["outputs"][files[0].name] = samples.identity(files[0])
            files[1].write_text(json.dumps(record))
        with patch.object(samples, "prepare_models", return_value=self.prepared):
            self.assertEqual(self.call("--prepare-only"), 0)
        self.assert_canonical_models_published()
        for path in stale:
            self.assertFalse(path.exists())
        for path, content in retained.items():
            self.assertEqual(path.read_bytes(), content)

    def test_second_setup_publication_failure_preserves_backups_config_cache_and_previous_mixed_models(self):
        with patch.object(samples, "prepare_models", return_value=self.prepared):
            self.assertEqual(self.call("--prepare-only"), 0)
        self.old_engine_fixtures()
        sentinel = self.cache / "unrelated-cached-model.bin"
        sentinel.write_bytes(b"Existing cache must never be changed by publication rollback")
        before = self.installation_snapshot()
        cache_before = sentinel.read_bytes()
        for _, files in self.prepared:
            files[0].write_bytes(files[0].read_bytes() + b" next conversion")
            record = json.loads(files[1].read_text())
            record["conversion"]["outputs"][files[0].name] = samples.identity(files[0])
            files[1].write_text(json.dumps(record))
        last_directory, *_ = samples.MODELS[-1]
        last_original = samples.MODELS[-1][2]
        last_model = self.sdk / "samples/models" / last_directory / last_original
        replace = samples.os.replace
        failed = False
        def fail_once(source, destination):
            nonlocal failed
            if Path(destination) == last_model and Path(source).name == "new" and not failed:
                failed = True
                raise OSError("Synthetic final-model publication failure")
            return replace(source, destination)
        with patch.object(samples.os, "replace", side_effect=fail_once):
            with self.assertRaisesRegex(OSError, "final-model publication failure"):
                samples.publish_installed(self.sdk, self.prepared, [])
        self.assertTrue(failed)
        self.assertEqual(self.installation_snapshot(), before)
        self.assertEqual(sentinel.read_bytes(), cache_before)

    def test_failed_publication_restores_retired_derived_engines(self):
        stale, original_engines = self.old_engine_fixtures()
        stale_contents = {path: path.read_bytes() for path in stale}
        replace = samples.os.replace
        failed = False
        retired = []
        def fail_config_once(source, destination):
            nonlocal failed
            if Path(source) in stale_contents:
                retired.append(Path(source))
            if Path(destination) == self.config and Path(source).name == "new" and not failed:
                failed = True
                raise OSError("Synthetic config publication failure")
            return replace(source, destination)
        with patch.object(samples.os, "replace", side_effect=fail_config_once):
            with self.assertRaisesRegex(OSError, "Synthetic config publication failure"):
                samples.publish_installed(self.sdk, self.prepared, [])
        self.assertTrue(failed)
        self.assertEqual(set(retired), set(stale), "The injected failure must occur after real engine retirement")
        for path, content in {**stale_contents, **original_engines}.items():
            self.assertEqual(path.read_bytes(), content)
        self.assert_not_published()
        self.assertEqual(list(self.sdk.rglob(".modelopt-*")), [])

    def test_failed_builder_removes_private_engine_scratch(self):
        self.cache.mkdir()
        args = argparse.Namespace(builder=str(self.builder), batch_size=1, gpu_id=0)
        with patch.object(samples.subprocess, "check_output", return_value="TensorRT 11.4.1.31\n"), \
                patch.object(samples, "run", side_effect=RuntimeError("Synthetic builder failure")):
            with self.assertRaisesRegex(RuntimeError, "Synthetic builder failure"):
                samples.build_installed_engines(args, self.prepared, self.cache)
        self.assertEqual(list(self.cache.glob("engine-build-*")), [])
        self.assert_not_published()

    def test_cached_conversion_bytes_are_verified_before_publication(self):
        calibration_root = self.root / "calibration"
        for directory, *_ in samples.MODELS:
            calibration = calibration_root / directory
            calibration.mkdir(parents=True)
            (calibration / "sample.npz").write_bytes(b"Synthetic NPZ identity; no conversion executed")
        self.cache.mkdir()
        args = argparse.Namespace(sdk_root=self.sdk, assets_root=self.sdk / "samples",
                                  bootstrap_python=None, calibration_root=calibration_root)
        def corrupt_producer_output(command):
            output_dir = Path(command[command.index("--output-dir") + 1])
            output_dir.mkdir(parents=True)
            output = output_dir / command[command.index("--output-name") + 1]
            output.write_bytes(b"Verified conversion bytes")
            (output_dir / "provenance.json").write_text(json.dumps({"outputs": {output.name: samples.identity(output)}}))
            output.write_bytes(b"Corrupted cached conversion bytes")
        tools = Path(__file__).resolve().parent
        with patch.object(samples, "tool_python", return_value=Path("/synthetic/python")), \
                patch.object(samples, "source_identity", return_value={"model": "Synthetic model closure"}), \
                patch.object(samples, "run", side_effect=corrupt_producer_output):
            with self.assertRaisesRegex(RuntimeError, "Prepared model output changed"):
                samples.prepare_models(args, tools, self.cache)
        self.assert_not_published()



if __name__ == "__main__":
    unittest.main()
