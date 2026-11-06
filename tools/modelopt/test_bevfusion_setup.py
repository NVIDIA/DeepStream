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


def shell_function(text, name):
    """Extract only a helper, never source the download/setup top level."""
    match = re.search(r"^function\s+" + re.escape(name) + r"(?:\(\))?\s*\{\n.*?^\}",
                      text, re.MULTILINE | re.DOTALL)
    if match is None:
        raise AssertionError("Required shell function missing: " + name)
    return match.group(0)


class BevFusionEngineIdentityTests(unittest.TestCase):
    """Real cache/functions with genuine tiny ONNX and fake CUDA/builder outputs."""

    @classmethod
    def setUpClass(cls):
        try:
            import onnx
        except ImportError:
            raise unittest.SkipTest("ONNX is required for CPU external-data cache fixtures")
        cls.onnx_module = onnx
        cls.script = (Path(__file__).resolve().parents[2] /
                      "src/apps/sample_apps/deepstream-3d-lidar-sensor-fusion/bevfusion/build_bevfusion_model.sh").read_text()

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.models = self.root / "models"
        self.models.mkdir()
        self.model = self.models / "chunk.onnx"
        self.weights = self.models / "weights.bin"
        self.weights.write_bytes(b"\0\0\x80?")
        self.model.write_bytes(self.model_proto().SerializeToString())
        self.uuid = self.root / "cuda-uuid.txt"
        self.uuid.write_text("cuda-device-uuid:logical-zero-A\n")
        self.calls = self.root / "builder-calls.txt"
        self.arguments = self.root / "builder-arguments.txt"
        self.plan = self.models / "build/chunk.plan"
        self.marker = self.models / "build/chunk.plan.mixed.identity"

    def external_tensor(self, location="weights.bin", **metadata):
        onnx = self.onnx_module
        tensor = onnx.TensorProto(name="weight", data_type=onnx.TensorProto.FLOAT,
                                  dims=[1], data_location=onnx.TensorProto.EXTERNAL)
        for key, value in dict(location=location, offset="0", length="4", **metadata).items():
            tensor.external_data.add(key=key, value=str(value))
        return tensor

    def model_proto(self, tensor=None):
        onnx = self.onnx_module
        graph = onnx.helper.make_graph(
            [onnx.helper.make_node("Add", ["input", "weight"], ["output"])], "external-fixture",
            [onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1])],
            [onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1])],
            initializer=[tensor if tensor is not None else self.external_tensor()])
        return onnx.helper.make_model(graph)

    def bash(self, commands, *, major=11, uuid_override=True, extra_environment=None):
        definitions = []
        for name in ("model_dependency_identity", "resolve_cuda_device_identity", "compile_trt_model"):
            if re.search(r"^function\s+" + name + r"(?:\(\))?\s*\{", self.script, re.MULTILINE):
                definitions.append(shell_function(self.script, name))
        fixture = r'''
nvidia-smi() { printf 'physical-A\nphysical-B\n'; }
fixture_builder() {
    if [[ "${1:-}" == --version ]]; then printf 'TensorRT 11.4.0.106\n'; return; fi
    printf 'build\n' >> "$FIXTURE_CALLS"
    printf '%s\n' "$@" >> "$FIXTURE_ARGUMENTS"
    if [[ "${FIXTURE_FAIL:-0}" == 1 ]]; then return 1; fi
    local argument
    for argument in "$@"; do
        case "$argument" in
            --saveEngine=*) printf 'plan-%s\n' "$(wc -l < "$FIXTURE_CALLS")" > "${argument#--saveEngine=}" ;;
            --exportLayerInfo=*) printf '{}\n' > "${argument#--exportLayerInfo=}" ;;
        esac
    done
    if [[ -n "${FIXTURE_MUTATE:-}" ]]; then printf changed >> "$FIXTURE_MUTATE"; fi
}
'''
        if uuid_override:
            fixture += '\nresolve_cuda_device_identity() { [[ -s "$FIXTURE_UUID" ]] && cat "$FIXTURE_UUID"; }\n'
        environment = os.environ.copy()
        environment.update(TRT_MAJOR=str(major), FIXTURE_MODELS=str(self.models),
                           FIXTURE_UUID=str(self.uuid), FIXTURE_CALLS=str(self.calls),
                           FIXTURE_ARGUMENTS=str(self.arguments), PYTHONDONTWRITEBYTECODE="1",
                           PATH=str(Path(sys.executable).parent) + os.pathsep + environment.get("PATH", ""))
        if extra_environment:
            environment.update(extra_environment)
        code = ("set -e -o pipefail\n" + "\n".join(definitions) + fixture +
                '\nmodel_dir="$FIXTURE_MODELS"\nTensorRT_Bin=fixture_builder\n' + commands)
        return subprocess.run(["/bin/bash", "-c", code], env=environment, cwd=self.root,
                              capture_output=True, text=True, timeout=15)

    def compile(self, **kwargs):
        return self.bash('compile_trt_model chunk "--fp16" 1 1\n', **kwargs)

    def assert_success(self, process):
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)

    def build_count(self):
        return len(self.calls.read_text().splitlines()) if self.calls.exists() else 0

    def shared_sidecar_model(self, locations=None):
        """A valid Add chain using every external tensor in one shared file."""
        onnx = self.onnx_module
        locations = locations or [self.weights.name] * 16
        self.weights.write_bytes(b"\0\0\x80?" * len(locations))
        initializers, nodes = [], []
        for index, location in enumerate(locations):
            tensor = self.external_tensor(location)
            tensor.name = "weight" + str(index)
            next(entry for entry in tensor.external_data if entry.key == "offset").value = str(index * 4)
            initializers.append(tensor)
            incoming = "input" if index == 0 else "sum" + str(index - 1)
            outgoing = "output" if index == len(locations) - 1 else "sum" + str(index)
            nodes.append(onnx.helper.make_node("Add", [incoming, tensor.name], [outgoing]))
        graph = onnx.helper.make_graph(
            nodes, "shared-external-file",
            [onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1])],
            [onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1])],
            initializer=initializers)
        return onnx.helper.make_model(graph)

    def dependency_identity_probe(self, bootstrap=""):
        """Run the unchanged helper's Python, counting real shared-file reads."""
        definition = shell_function(self.script, "model_dependency_identity")
        program = definition.split("<<'PY'\n", 1)[1].split("\nPY", 1)[0]
        instrumentation = '''
import atexit, os, sys
from pathlib import Path
fixture_payload = Path(os.environ['FIXTURE_PAYLOAD'])
fixture_opens = 0
fixture_original_open = Path.open
def fixture_counted_open(self, *arguments, **keywords):
    global fixture_opens
    if self == fixture_payload:
        fixture_opens += 1
    return fixture_original_open(self, *arguments, **keywords)
Path.open = fixture_counted_open
atexit.register(lambda: print('FIXTURE_PAYLOAD_READS=' + str(fixture_opens), file=sys.stderr))
'''
        return subprocess.run(
            [sys.executable, "-B", "-c", instrumentation + bootstrap + program, str(self.model)],
            env=dict(os.environ, FIXTURE_PAYLOAD=str(self.weights.resolve())), cwd=self.root,
            capture_output=True, text=True, timeout=10)

    def assert_single_payload_read(self, process):
        self.assert_success(process)
        self.assertRegex(process.stdout.strip(), r"^[0-9a-f]{64}$")
        counts = [line for line in process.stderr.splitlines() if line.startswith("FIXTURE_PAYLOAD_READS=")]
        self.assertEqual(counts, ["FIXTURE_PAYLOAD_READS=1"], process.stderr)

    def test_dependency_walker_accepts_legacy_descriptor_without_is_repeated(self):
        # Mirror the real Protobuf 4.x/5.x descriptor API without pinning or
        # installing an old runtime in this otherwise dependency-neutral suite.
        bootstrap = '''
import onnx
fixture_original_load = onnx.load
class LegacyField:
    LABEL_REPEATED = 3
    def __init__(self, field):
        self.message_type = field.message_type
        repeated = getattr(field, 'is_repeated', None)
        self.label = 3 if (repeated if repeated is not None else field.label == 3) else 1
class LegacyMessage:
    def __init__(self, message): self.message = message
    def __getattr__(self, name): return getattr(self.message, name)
    def ListFields(self):
        fields = []
        for field, value in self.message.ListFields():
            legacy = LegacyField(field)
            if field.message_type is not None:
                value = [LegacyMessage(child) for child in value] if legacy.label == 3 else LegacyMessage(value)
            fields.append((legacy, value))
        return fields
onnx.load = lambda *arguments, **keywords: LegacyMessage(fixture_original_load(*arguments, **keywords))
'''
        for external in (False, True):
            with self.subTest(external=external):
                tensor = (self.external_tensor() if external else self.onnx_module.helper.make_tensor(
                    "weight", self.onnx_module.TensorProto.FLOAT, [1], [1.0]))
                self.model.write_bytes(self.model_proto(tensor).SerializeToString())
                ordinary = self.dependency_identity_probe()
                self.assert_success(ordinary)
                legacy = self.dependency_identity_probe(bootstrap)
                self.assert_success(legacy)
                self.assertEqual(legacy.stdout, ordinary.stdout)

    def test_shared_external_file_is_hashed_once_for_all_used_tensors(self):
        self.model.write_bytes(self.shared_sidecar_model().SerializeToString())
        self.onnx_module.checker.check_model(str(self.model))
        self.assert_single_payload_read(self.dependency_identity_probe())

    def test_shared_external_file_symlink_aliases_are_hashed_once(self):
        alias = self.models / "weights-alias.bin"
        alias.symlink_to(self.weights)
        locations = [self.weights.name, alias.name] * 8
        self.model.write_bytes(self.shared_sidecar_model().SerializeToString())
        self.onnx_module.checker.check_model(str(self.model))
        # The identity helper permits aliases resolving inside the model root.
        # ONNX's checker separately forbids symlink sidecars; this checks only
        # the helper's accepted alias contract, not parser/engine support.
        self.model.write_bytes(self.shared_sidecar_model(locations).SerializeToString())
        self.assert_single_payload_read(self.dependency_identity_probe())

    def test_shared_external_file_still_validates_each_later_tensor_range(self):
        proto = self.shared_sidecar_model()
        self.model.write_bytes(proto.SerializeToString())
        self.onnx_module.checker.check_model(str(self.model))
        self.assert_success(self.dependency_identity_probe())
        last = proto.graph.initializer[-1]
        next(entry for entry in last.external_data if entry.key == "length").value = "5"
        self.model.write_bytes(proto.SerializeToString())
        invalid = self.dependency_identity_probe()
        self.assertNotEqual(invalid.returncode, 0, invalid.stdout + invalid.stderr)
        self.assertIn("External tensor payload exceeds file", invalid.stderr)
        self.assertEqual(invalid.stdout, "")

    def test_shared_external_file_digest_is_fresh_on_each_helper_invocation(self):
        self.model.write_bytes(self.shared_sidecar_model().SerializeToString())
        self.onnx_module.checker.check_model(str(self.model))
        first = self.dependency_identity_probe()
        self.assert_single_payload_read(first)
        original = self.model.read_bytes()
        self.weights.write_bytes(b"\0\0\0@" * 16)
        second = self.dependency_identity_probe()
        self.assert_single_payload_read(second)
        self.assertEqual(self.model.read_bytes(), original)
        self.assertNotEqual(first.stdout, second.stdout)

    def test_external_weights_change_invalidates_engine_without_protobuf_change(self):
        original = self.model.read_bytes()
        self.assert_success(self.compile())
        self.weights.write_bytes(b"\0\0\0@")
        self.assert_success(self.compile())
        self.assertEqual(self.model.read_bytes(), original)
        self.assertEqual(self.build_count(), 2)

    def test_effective_cuda_uuid_change_invalidates_cache_with_same_physical_gpu_list(self):
        self.assert_success(self.compile())
        self.uuid.write_text("cuda-device-uuid:logical-zero-B\n")
        self.assert_success(self.compile())
        self.assertEqual(self.build_count(), 2)

    def test_unknown_device_identity_never_counts_as_verified_hit(self):
        self.uuid.write_text("")
        self.assert_success(self.compile())
        self.assert_success(self.compile())
        self.assertEqual(self.build_count(), 2)

    def test_unchanged_model_device_and_plan_reuse_without_building(self):
        self.assert_success(self.compile())
        self.assert_success(self.compile())
        self.assertEqual(self.build_count(), 1)

    def test_cuda_order_change_and_corrupt_plan_each_require_rebuild(self):
        self.assert_success(self.compile(extra_environment={"CUDA_DEVICE_ORDER": "FASTEST_FIRST"}))
        self.assert_success(self.compile(extra_environment={"CUDA_DEVICE_ORDER": "PCI_BUS_ID"}))
        self.plan.write_bytes(b"Corrupt plan bytes")
        self.assert_success(self.compile(extra_environment={"CUDA_DEVICE_ORDER": "PCI_BUS_ID"}))
        self.assertEqual(self.build_count(), 3)

    def test_failed_builder_preserves_previous_plan_identity_and_mode(self):
        self.assert_success(self.compile())
        self.plan.chmod(0o640)
        before = (self.plan.read_bytes(), self.marker.read_bytes(), self.plan.stat().st_mode)
        self.uuid.write_text("cuda-device-uuid:different\n")
        process = self.compile(extra_environment={"FIXTURE_FAIL": "1"})
        self.assertNotEqual(process.returncode, 0, process.stdout + process.stderr)
        self.assertEqual((self.plan.read_bytes(), self.marker.read_bytes(), self.plan.stat().st_mode), before)
        self.assertEqual(list(self.plan.parent.glob("chunk.build.*")), [])

    def test_weights_change_during_builder_preserves_previous_engine(self):
        self.assert_success(self.compile())
        before = (self.plan.read_bytes(), self.marker.read_bytes())
        self.uuid.write_text("cuda-device-uuid:force-cold-build\n")
        process = self.compile(extra_environment={"FIXTURE_MUTATE": str(self.weights)})
        self.assertNotEqual(process.returncode, 0, process.stdout + process.stderr)
        self.assertEqual((self.plan.read_bytes(), self.marker.read_bytes()), before)
        self.assertEqual(list(self.plan.parent.glob("chunk.build.*")), [])

    def test_symlink_model_is_canonicalized_for_both_identity_and_builder(self):
        source = self.root / "real-model"
        source.mkdir()
        actual = source / "model.onnx"
        actual.write_bytes(self.model.read_bytes())
        (source / "weights.bin").write_bytes(self.weights.read_bytes())
        self.model.unlink()
        self.model.symlink_to(actual)
        self.weights.write_bytes(b"Different lexical-directory sidecar")
        self.assert_success(self.compile())
        self.assertIn("--onnx=" + str(actual), self.arguments.read_text().splitlines())

    def test_trt10_existing_plan_still_skips_new_identity_providers(self):
        self.assert_success(self.compile(major=10))
        self.weights.unlink()
        self.model.unlink()
        self.uuid.write_text("")
        self.assert_success(self.compile(major=10))
        self.assertEqual(self.build_count(), 1)

    def test_invalid_external_metadata_fails_before_build_and_preserves_old_outputs(self):
        self.assert_success(self.compile())
        before = (self.plan.read_bytes(), self.marker.read_bytes())
        cases = []
        for location in ("missing.bin", "../escape.bin", "/absolute.bin", "bad\\name.bin"):
            cases.append(self.external_tensor(location))
        tensor = self.external_tensor()
        tensor.external_data.add(key="location", value="weights.bin")
        cases.append(tensor)
        for key, value in (("offset", "-1"), ("length", "9"), ("offset", "invalid"),
                           ("length", "-1"), ("length", "invalid")):
            tensor = self.external_tensor()
            next(entry for entry in tensor.external_data if entry.key == key).value = value
            cases.append(tensor)
        cases.append(self.onnx_module.TensorProto(name="weight", data_type=1, dims=[1], data_location=1))
        for index, tensor in enumerate(cases):
            with self.subTest(index=index):
                self.model.write_bytes(self.model_proto(tensor).SerializeToString())
                process = self.compile()
                self.assertNotEqual(process.returncode, 0, process.stdout + process.stderr)
                self.assertEqual(self.build_count(), 1)
                self.assertEqual((self.plan.read_bytes(), self.marker.read_bytes()), before)

    def test_external_sidecar_symlink_escape_is_rejected_before_build(self):
        outside = self.root / "outside-weights.bin"
        outside.write_bytes(self.weights.read_bytes())
        self.weights.unlink()
        self.weights.symlink_to(outside)
        process = self.compile()
        self.assertNotEqual(process.returncode, 0, process.stdout + process.stderr)
        self.assertEqual(self.build_count(), 0)
        self.assertFalse(self.plan.exists())

    def test_dependency_walker_hashes_nested_sparse_and_function_tensors(self):
        onnx = self.onnx_module
        for kind in ("nested", "sparse", "function"):
            with self.subTest(kind=kind):
                proto = self.model_proto()
                extra = self.external_tensor("extra.bin")
                if kind == "nested":
                    graph = proto.graph.node.add(op_type="If", input=["condition"], output=["branch"])
                    graph.attribute.add(name="then_branch", type=onnx.AttributeProto.GRAPH).g.CopyFrom(
                        onnx.helper.make_graph([], "nested", [], [], initializer=[extra]))
                elif kind == "sparse":
                    sparse = proto.graph.sparse_initializer.add(dims=[1])
                    sparse.values.CopyFrom(extra)
                    sparse.indices.CopyFrom(onnx.helper.make_tensor("indices", onnx.TensorProto.INT64, [1], [0]))
                else:
                    function = proto.functions.add(name="fixture_function", domain="fixture")
                    attribute = function.node.add(op_type="Constant", output=["constant"]).attribute.add(
                        name="value", type=onnx.AttributeProto.TENSOR)
                    attribute.t.CopyFrom(extra)
                self.model.write_bytes(proto.SerializeToString())
                payload = self.models / "extra.bin"
                payload.write_bytes(b"aaaa")
                first = self.bash('model_dependency_identity "$FIXTURE_MODELS/chunk.onnx"\n')
                self.assert_success(first)
                payload.write_bytes(b"bbbb")
                second = self.bash('model_dependency_identity "$FIXTURE_MODELS/chunk.onnx"\n')
                self.assert_success(second)
                self.assertRegex(first.stdout.strip(), r"^[0-9a-f]{64}$")
                self.assertNotEqual(first.stdout, second.stdout)

    def test_real_cuda_uuid_probe_uses_logical_zero_and_rejects_unprovable_identity(self):
        definition = shell_function(self.script, "resolve_cuda_device_identity")
        program = definition.split("<<'PY'\n", 1)[1].split("\nPY", 1)[0]
        bootstrap = '''
import ctypes, os
class Call:
    def __init__(self, implementation): self.implementation = implementation
    def __call__(self, *arguments): return self.implementation(*arguments)
class Driver:
    def __init__(self):
        self.cuInit = Call(lambda flags: 1 if os.environ['CASE'] == 'init-fails' else 0)
        self.cuDeviceGet = Call(self.device)
        if os.environ['CASE'] != 'missing-v2': self.cuDeviceGetUuid_v2 = Call(self.uuid)
    def device(self, output, ordinal):
        if ordinal != 0: raise RuntimeError('Wrong logical CUDA ordinal')
        output._obj.value = 17
        return 1 if os.environ['CASE'] == 'device-fails' else 0
    def uuid(self, output, device):
        if device.value != 17: raise RuntimeError('Wrong effective CUDA device')
        if os.environ['CASE'] != 'zero-uuid':
            for index in range(16): output._obj[index] = index + 1
        return 1 if os.environ['CASE'] == 'uuid-fails' else 0
ctypes.CDLL = lambda name: Driver()
'''
        for case in ("valid", "init-fails", "device-fails", "missing-v2", "zero-uuid", "uuid-fails"):
            with self.subTest(case=case):
                process = subprocess.run([sys.executable, "-I", "-B", "-c", bootstrap + program],
                                         env=dict(os.environ, CASE=case), capture_output=True,
                                         text=True, timeout=10)
                if case == "valid":
                    self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
                    self.assertEqual(process.stdout.strip(), "cuda-device-uuid:01020304-0506-0708-090a-0b0c0d0e0f10")
                else:
                    self.assertNotEqual(process.returncode, 0)
                    self.assertEqual(process.stdout, "")
                    self.assertIn("cache reuse disabled", process.stderr)



if __name__ == "__main__":
    unittest.main()
