# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public adapter contracts using ONNX fixtures, without installs or GPUs."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import onnx
from onnx import TensorProto as T, helper as h

import prepare_model as tool


def model(half=False, cast_only=False):
    nodes = ([h.make_node("Cast", ["input"], ["half"], to=T.FLOAT16)] if half else [])
    if not cast_only:
        nodes.append(h.make_node("Add", ["half" if half else "input"] * 2,
                                 ["value" if half else "output"]))
    if half:
        nodes.append(h.make_node("Cast", ["half" if cast_only else "value"], ["output"], to=T.FLOAT))
    graph = h.make_graph(nodes, "fixture", [h.make_tensor_value_info("input", T.FLOAT, [1, 2])],
                         [h.make_tensor_value_info("output", T.FLOAT, [1, 2])])
    return h.make_model(graph, opset_imports=[h.make_opsetid("", 17)])


class PublicAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "original.onnx"
        onnx.save(model(), self.source)
        self.original = self.source.read_bytes()
        self.data = self.root / "calibration"
        self.data.mkdir()
        np.savez(self.data / "batch.npz", input=np.array([[1.0, 2.0]], dtype=np.float32))
        self.args = argparse.Namespace(input=str(self.source), output_dir=str(self.root / "artifact"),
                                       output_name="mixed.onnx", calibration_data=str(self.data))

    def prepare(self, converted=None, action=None):
        def convert(**kwargs):
            self.assertEqual(kwargs["providers"], ["cpu"])
            self.assertTrue(kwargs["keep_io_types"])
            self.assertEqual(kwargs["low_precision_type"], "fp16")
            self.assertNotEqual(Path(kwargs["onnx_path"]), self.source)
            if action:
                action(kwargs)
            return model(half=True) if converted is None else converted
        with patch.object(tool, "environment_identity", return_value={"fixture": "environment"}), \
                patch.dict(sys.modules, {"modelopt.onnx.autocast": SimpleNamespace(convert_to_mixed_precision=convert)}):
            return tool.prepare(self.args)

    def test_documented_cpu_api_preserves_io_and_source_and_reuses_verified_output(self):
        self.assertEqual(self.prepare()["status"], "PREPARED")
        self.assertEqual(self.source.read_bytes(), self.original)
        metadata = json.loads((self.root / "artifact/provenance.json").read_text())
        self.assertEqual(metadata["external_io"], tool.io_contract(model()))
        self.assertEqual(metadata["fp16_arithmetic"][0]["op_type"], "Add")
        self.assertEqual(self.prepare(action=lambda _: self.fail("cached conversion called vendor"))["status"], "REUSED")

    def test_cast_only_is_not_fp16_computation(self):
        with self.assertRaisesRegex(tool.PreparationError, "FP16 arithmetic"):
            self.prepare(model(half=True, cast_only=True))
        self.assertFalse((self.root / "artifact").exists())
        self.assertEqual(self.source.read_bytes(), self.original)

    def test_changed_io_is_rejected_without_publication(self):
        converted = model(half=True)
        converted.graph.output[0].name = "different"
        with self.assertRaisesRegex(tool.PreparationError, "input/output"):
            self.prepare(converted)
        self.assertFalse((self.root / "artifact").exists())

    def test_changed_output_and_calibration_invalidate_cache(self):
        self.prepare()
        output = self.root / "artifact/mixed.onnx"
        output.write_bytes(b"corrupt")
        with self.assertRaisesRegex(tool.PreparationError, "bytes changed"):
            self.prepare()
        np.savez(self.data / "batch.npz", input=np.array([[3.0, 4.0]], dtype=np.float32))
        with self.assertRaisesRegex(tool.PreparationError, "different inputs"):
            self.prepare()

    def test_calibration_names_shape_dtype_and_finite_values(self):
        for arrays in ({"wrong": np.ones((1, 2), np.float32)},
                       {"input": np.ones((2, 2), np.float32)},
                       {"input": np.ones((1, 2), np.float16)},
                       {"input": np.array([[float("nan"), 0]], np.float32)}):
            with self.subTest(arrays=arrays):
                np.savez(self.data / "batch.npz", **arrays)
                with self.assertRaises(tool.PreparationError):
                    self.prepare(action=lambda _: self.fail("invalid calibration reached vendor"))
                self.assertFalse((self.root / "artifact").exists())

    def test_source_mutation_during_conversion_is_not_published(self):
        with self.assertRaisesRegex(tool.PreparationError, "Source changed"):
            self.prepare(action=lambda _: self.source.write_bytes(b"changed source"))
        self.assertFalse((self.root / "artifact").exists())

    def test_output_basename_and_symlink_destination_rejected(self):
        self.args.output_name = "../escape.onnx"
        with self.assertRaises(tool.PreparationError):
            self.prepare()
        self.args.output_name = "mixed.onnx"
        (self.root / "artifact").symlink_to(self.data, target_is_directory=True)
        with self.assertRaises(tool.PreparationError):
            self.prepare()

    def test_original_weight_closure_validates_every_range_and_memoizes_files(self):
        weight = self.root / "weights.bin"
        weight.write_bytes(b"01234567")
        proto = model()
        for index in range(2):
            tensor = proto.graph.initializer.add(name="weights" + str(index), data_type=T.FLOAT)
            tensor.dims.append(1)
            tensor.data_location = T.EXTERNAL
            for key, value in (("location", "weights.bin"), ("offset", str(index * 4)), ("length", "4")):
                tensor.external_data.add(key=key, value=value)
        with patch.object(tool, "identity", wraps=tool.identity) as digest:
            closure = tool.external_files(proto, self.source)
            self.assertEqual(list(closure), ["weights.bin"])
            self.assertEqual(digest.call_count, 1)
        proto.graph.initializer[1].external_data[-1].value = "100"
        with self.assertRaisesRegex(tool.PreparationError, "range"):
            tool.external_files(proto, self.source)

    def test_qdq_and_custom_operators_require_separate_workflow(self):
        for domain, operation in (("private", "Custom"), ("", "DequantizeLinear")):
            proto = model()
            proto.graph.node.add(domain=domain, op_type=operation)
            with self.assertRaises(tool.PreparationError):
                tool.check_graph(proto)


if __name__ == "__main__":
    unittest.main()
