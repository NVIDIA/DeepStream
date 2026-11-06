#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Create a private, reproducible mixed-FP16 artifact using ModelOpt AutoCast.

This public adapter uses NVIDIA's documented Python API. It does not quantize
models, change the source files, install dependencies or qualify task accuracy.
"""

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile


PINNED = {"nvidia-modelopt": "0.48.0", "onnx": "1.21.0",
          "onnxruntime": "1.30.0", "numpy": "2.5.4", "torch": "2.15.0+cpu"}
VALUE_OPERATIONS = {
    "Add", "Sub", "Mul", "Div", "Pow", "Conv", "ConvTranspose", "Gemm",
    "MatMul", "Einsum", "Relu", "LeakyRelu", "PRelu", "Sigmoid", "Tanh",
    "Softmax", "LogSoftmax", "Exp", "Log", "Sqrt", "Reciprocal", "Abs",
    "Neg", "Clip", "Max", "Min", "ReduceMean", "ReduceSum", "ReduceMax",
    "ReduceMin", "AveragePool", "GlobalAveragePool", "MaxPool", "Erf",
    "BatchNormalization", "InstanceNormalization", "LayerNormalization",
    "GroupNormalization", "Sin", "Cos", "Attention",
}


class PreparationError(RuntimeError):
    pass


def identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def protobuf_messages(message):
    yield message
    for field, value in message.ListFields():
        if field.message_type:
            repeated = (field.is_repeated if hasattr(field, "is_repeated")
                        else field.label == field.LABEL_REPEATED)
            for child in value if repeated else [value]:
                yield from protobuf_messages(child)


def external_files(model, source):
    """Validate every tensor range; hash shared external payloads only once."""
    source = Path(source).resolve(strict=True)
    parent, payloads, hashes = source.parent, {}, {}
    for tensor in protobuf_messages(model):
        if tensor.DESCRIPTOR.full_name != "onnx.TensorProto":
            continue
        if not tensor.external_data:
            if tensor.data_location == 1:
                raise PreparationError("External tensor has no location")
            continue
        metadata = {entry.key: entry.value for entry in tensor.external_data}
        if len(metadata) != len(tensor.external_data):
            raise PreparationError("Duplicate external tensor metadata")
        location = metadata.get("location", "")
        relative = Path(location)
        if not location or relative.is_absolute() or ".." in relative.parts or "\\" in location:
            raise PreparationError("Unsafe external tensor path: " + location)
        payload = (parent / relative).resolve(strict=True)
        if not payload.is_relative_to(parent) or not payload.is_file() or payload == source:
            raise PreparationError("External tensor must be a file inside the model directory")
        try:
            offset = int(metadata.get("offset", "0"))
            length = int(metadata.get("length", str(payload.stat().st_size - offset)))
        except ValueError as error:
            raise PreparationError("Invalid external tensor range") from error
        if min(offset, length) < 0 or offset + length > payload.stat().st_size:
            raise PreparationError("External tensor range exceeds its file")
        if payload not in hashes:
            hashes[payload] = identity(payload)
        payloads[relative.as_posix()] = hashes[payload]
    return dict(sorted(payloads.items()))


def io_contract(model):
    initializers = {value.name for value in model.graph.initializer}
    def describe(value):
        if not value.type.HasField("tensor_type"):
            raise PreparationError("Only tensor input/output contracts are supported")
        tensor = value.type.tensor_type
        dimensions = [dimension.dim_value if dimension.HasField("dim_value") else
                      dimension.dim_param if dimension.HasField("dim_param") else None
                      for dimension in tensor.shape.dim]
        return {"name": value.name, "dtype": tensor.elem_type, "shape": dimensions}
    return {"inputs": [describe(value) for value in model.graph.input if value.name not in initializers],
            "outputs": [describe(value) for value in model.graph.output]}


def check_graph(model):
    if model.functions:
        raise PreparationError("Local functions require a separately qualified conversion workflow")
    for message in protobuf_messages(model):
        if message.DESCRIPTOR.full_name == "onnx.NodeProto":
            if message.domain not in {"", "ai.onnx"} or message.op_type in {"QuantizeLinear", "DequantizeLinear"}:
                raise PreparationError("Custom operations and Q/DQ graphs are not supported by this CPU route")


def reduced_arithmetic(model, onnx):
    """A cast/layout/constant/shape-only path is not reduced computation."""
    graph = onnx.shape_inference.infer_shapes(model).graph
    types = {value.name: value.type.tensor_type.elem_type for value in
             list(graph.input) + list(graph.output) + list(graph.value_info)}
    types.update({value.name: value.data_type for value in graph.initializer})
    live = {value.name for value in graph.output}
    for node in reversed(graph.node):
        if live.intersection(node.output):
            live.update(node.input)
    values = {value["name"] for value in io_contract(model)["inputs"]}
    matches = []
    for node in graph.node:
        depends_on_input = node.op_type not in {"Shape", "Size"} and bool(values.intersection(node.input))
        if depends_on_input:
            values.update(node.output)
        if (depends_on_input and live.intersection(node.output) and node.op_type in VALUE_OPERATIONS
                and any(types.get(name) == onnx.TensorProto.FLOAT16 for name in node.input)):
            matches.append({"name": node.name, "op_type": node.op_type})
    if not matches:
        raise PreparationError("No provable input-dependent FP16 arithmetic in the converted model")
    return matches


def environment_identity():
    if sys.version_info < (3, 12):
        raise PreparationError("The pinned preparation environment requires Python >=3.12")
    for name, expected in PINNED.items():
        if importlib.metadata.version(name) != expected:
            raise PreparationError("Use the qualified version of " + name + ": " + expected)
    return {"python": list(sys.version_info[:3]), "packages": dict(sorted(
        (distribution.metadata["Name"].lower().replace("_", "-"), distribution.version)
        for distribution in importlib.metadata.distributions() if distribution.metadata["Name"]))}


def calibration_snapshot(directory, contract, output, numpy, onnx):
    directory = Path(directory).resolve(strict=True)
    files = sorted(directory.glob("*.npz")) if directory.is_dir() else [directory]
    if not files:
        raise PreparationError("Representative preprocessed NPZ inputs are required")
    output.mkdir()
    records = {}
    expected = {value["name"]: value for value in contract["inputs"]}
    for index, path in enumerate(files):
        with numpy.load(path, allow_pickle=False) as arrays:
            if set(arrays.files) != set(expected):
                raise PreparationError("Calibration input names do not match the model")
            for name, value in expected.items():
                array = arrays[name]
                shape = value["shape"]
                if (array.dtype != numpy.dtype(onnx.helper.tensor_dtype_to_np_dtype(value["dtype"])) or
                        len(array.shape) != len(shape) or any(
                            type(size) is int and size > 0 and actual != size
                            for actual, size in zip(array.shape, shape))):
                    raise PreparationError("Calibration shape/dtype mismatch: " + name)
                if array.dtype.kind in "fc" and not numpy.isfinite(array).all():
                    raise PreparationError("Non-finite calibration values: " + name)
        before = identity(path)
        snapshot = output / (str(index) + ".npz")
        shutil.copyfile(path, snapshot)
        if identity(snapshot) != before:
            raise PreparationError("Calibration data changed during snapshot")
        records[path.name] = before
    return records


@contextmanager
def destination_lock(destination):
    lock = destination.parent / ("." + destination.name + ".modelopt.lock")
    descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(str(os.getpid()) + "\n")
        yield
    finally:
        lock.unlink()


def prepare(args):
    versions = environment_identity()
    import numpy
    import onnx

    source = Path(args.input).resolve(strict=True)
    destination = Path(args.output_dir).absolute()
    if not source.is_file() or source.suffix != ".onnx" or destination.name in {"", ".", ".."}:
        raise PreparationError("Use an ONNX input and a distinct private output directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination = destination.parent.resolve() / destination.name
    if source.is_relative_to(destination):
        raise PreparationError("Output directory cannot contain the source model")
    output_name = args.output_name or source.stem + ".modelopt-fp16.onnx"
    if Path(output_name).name != output_name or "\\" in output_name or not output_name.endswith(".onnx"):
        raise PreparationError("Output model name must be an ONNX basename")
    model = onnx.load(str(source), load_external_data=False)
    check_graph(model)
    contract = io_contract(model)
    closure = {"model": identity(source), "external_data": external_files(model, source)}
    with destination_lock(destination), tempfile.TemporaryDirectory(prefix=".modelopt-", dir=destination.parent) as directory:
        work = Path(directory)
        calibration = calibration_snapshot(args.calibration_data, contract, work / "calibration", numpy, onnx)
        request = {"source": closure, "calibration": calibration, "versions": versions,
                   "preparation_tool": identity(Path(__file__)), "output_name": output_name,
                   "options": {"low_precision_type": "fp16", "keep_io_types": True, "providers": ["cpu"]}}
        if destination.exists() or destination.is_symlink():
            if destination.is_symlink() or not destination.is_dir():
                raise PreparationError("Output is not a private artifact directory")
            previous = json.loads((destination / "provenance.json").read_text())
            if previous["request"] != request or output_name not in previous["outputs"]:
                raise PreparationError("Cached output has different inputs; select a new output directory")
            for name, expected in previous["outputs"].items():
                if Path(name).name != name or identity(destination / name) != expected:
                    raise PreparationError("Cached output bytes changed")
            return {"status": "REUSED", "model": str(destination / output_name)}
        snapshot = work / "source"
        snapshot.mkdir()
        for name, expected in {source.name: closure["model"], **closure["external_data"]}.items():
            target = snapshot / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source.parent / name, target)
            if identity(target) != expected:
                raise PreparationError("Source changed during snapshot")
        artifact = work / "artifact"
        artifact.mkdir()
        with (artifact / "conversion.log").open("w") as log, redirect_stdout(log), redirect_stderr(log):
            from modelopt.onnx.autocast import convert_to_mixed_precision
            converted = convert_to_mixed_precision(onnx_path=str(snapshot / source.name),
                low_precision_type="fp16", keep_io_types=True, providers=["cpu"],
                calibration_data=str(work / "calibration"))
            if io_contract(converted) != contract:
                raise PreparationError("AutoCast changed the public input/output contract")
            arithmetic = reduced_arithmetic(converted, onnx)
            onnx.external_data_helper.convert_model_from_external_data(converted)
            onnx.save_model(converted, str(artifact / output_name), save_as_external_data=False)
            onnx.checker.check_model(str(artifact / output_name), full_check=True)
        if identity(source) != closure["model"] or external_files(model, source) != closure["external_data"]:
            raise PreparationError("Source changed during conversion")
        provenance = {"request": request, "outputs": {output_name: identity(artifact / output_name)},
                      "external_io": contract, "fp16_arithmetic": arithmetic,
                      "qualification": "Preparation checks only; target accuracy/runtime qualification is required"}
        (artifact / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
        if destination.exists() or destination.is_symlink():
            raise PreparationError("Output appeared while preparing; refusing replacement")
        os.rename(artifact, destination)
    return {"status": "PREPARED", "model": str(destination / output_name)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-name")
    parser.add_argument("--calibration-data", required=True,
                        help="Representative preprocessed NPZ file/directory; no implicit random inputs")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(prepare(args), sort_keys=True))
        return 0
    except Exception as error:
        print("ModelOpt preparation failed: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
