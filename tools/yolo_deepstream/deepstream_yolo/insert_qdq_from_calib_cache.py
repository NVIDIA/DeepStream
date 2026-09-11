#!/usr/bin/env python3
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

import argparse
import struct
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = SCRIPT_DIR.parent

try:
    import onnx
    from onnx import helper, numpy_helper
except ModuleNotFoundError:
    # The workspace used for this investigation keeps Python deps here.
    sys.path.insert(0, str(WORKSPACE_DIR / ".pydeps"))
    import onnx
    from onnx import helper, numpy_helper


MAGIC_COMMENT_PREFIX = "#"
DEFAULT_OUTPUT_SUFFIX = "_cache_qdq"
QDQ_SUFFIX = "_cache_qdq"
QUANT_MAX = 127.0
WEIGHT_TARGET_OPS = {"Conv", "ConvTranspose", "Gemm", "MatMul"}


def parse_cache(cache_path: Path) -> dict[str, float]:
    """Read TensorRT calibration-cache entries as tensor-name -> ONNX Q scale."""
    scales: dict[str, float] = {}
    with cache_path.open("r", encoding="utf-8") as cache_file:
        for line_no, raw_line in enumerate(cache_file, start=1):
            line = raw_line.strip()
            if not line or line.startswith(MAGIC_COMMENT_PREFIX) or ":" not in line:
                continue

            # Tensor names may contain ONNX scope separators like "onnx::Mul_254".
            if ": " in line:
                tensor_name, hex_value = [part.strip() for part in line.rsplit(": ", 1)]
            else:
                tensor_name, hex_value = [part.strip() for part in line.rsplit(":", 1)]
            if not tensor_name:
                continue

            try:
                value = struct.unpack(">f", bytes.fromhex(hex_value))[0]
            except (ValueError, struct.error) as exc:
                raise ValueError(f"Invalid cache hex value on line {line_no}: {raw_line.rstrip()}") from exc

            scale = value
            if scale <= 0.0 or not np.isfinite(scale):
                raise ValueError(f"Invalid non-positive scale for {tensor_name!r}: {scale}")
            scales[tensor_name] = float(scale)

    return scales


def should_quantize_activation_edge(node: onnx.NodeProto) -> bool:
    return node.op_type in WEIGHT_TARGET_OPS


def select_activation_tensors(
    scales: dict[str, float],
    graph_tensors: set[str],
    initializer_names: set[str],
    consumers: dict[str, list[onnx.NodeProto]],
) -> tuple[list[str], list[str]]:
    """Select cache entries consumed by supported weighted ops."""
    selected: list[str] = []
    skipped: list[str] = []
    for tensor_name in scales:
        if tensor_name not in graph_tensors:
            skipped.append(tensor_name)
            continue
        if tensor_name in initializer_names:
            skipped.append(tensor_name)
            continue
        if not any(should_quantize_activation_edge(node) for node in consumers.get(tensor_name, [])):
            skipped.append(tensor_name)
            continue
        selected.append(tensor_name)
    return selected, skipped


def collect_graph_names(
    graph: onnx.GraphProto,
) -> tuple[set[str], set[str], dict[str, list[onnx.NodeProto]]]:
    graph_tensors: set[str] = set()
    initializer_names = {initializer.name for initializer in graph.initializer}
    consumers: dict[str, list[onnx.NodeProto]] = defaultdict(list)

    for value in list(graph.input) + list(graph.output) + list(graph.value_info):
        graph_tensors.add(value.name)

    for node in graph.node:
        for input_name in node.input:
            if input_name:
                graph_tensors.add(input_name)
                consumers[input_name].append(node)
        for output_name in node.output:
            if output_name:
                graph_tensors.add(output_name)

    graph_tensors.update(initializer_names)
    return graph_tensors, initializer_names, consumers


def make_unique_name(base: str, used_names: set[str]) -> str:
    if base not in used_names:
        used_names.add(base)
        return base

    index = 1
    while f"{base}_{index}" in used_names:
        index += 1
    name = f"{base}_{index}"
    used_names.add(name)
    return name


def make_qdq_nodes(
    tensor_name: str,
    scale: float,
    used_names: set[str],
) -> tuple[list[onnx.NodeProto], list[onnx.TensorProto], str]:
    prefix = tensor_name.replace("/", "_")
    scale_name = make_unique_name(f"{prefix}{QDQ_SUFFIX}_scale", used_names)
    zero_point_name = make_unique_name(f"{prefix}{QDQ_SUFFIX}_zero_point", used_names)
    q_output_name = make_unique_name(f"{prefix}{QDQ_SUFFIX}_quantized", used_names)
    dq_output_name = make_unique_name(f"{prefix}{QDQ_SUFFIX}_dequantized", used_names)
    q_node_name = make_unique_name(f"{prefix}{QDQ_SUFFIX}_QuantizeLinear", used_names)
    dq_node_name = make_unique_name(f"{prefix}{QDQ_SUFFIX}_DequantizeLinear", used_names)

    initializers = [
        numpy_helper.from_array(np.array([scale], dtype=np.float32), scale_name),
        numpy_helper.from_array(np.array([0], dtype=np.int8), zero_point_name),
    ]
    nodes = [
        helper.make_node(
            "QuantizeLinear",
            [tensor_name, scale_name, zero_point_name],
            [q_output_name],
            name=q_node_name,
        ),
        helper.make_node(
            "DequantizeLinear",
            [q_output_name, scale_name, zero_point_name],
            [dq_output_name],
            name=dq_node_name,
        ),
    ]
    return nodes, initializers, dq_output_name


def make_weight_qdq_nodes(
    tensor_name: str,
    scale: np.ndarray,
    axis: int | None,
    used_names: set[str],
) -> tuple[list[onnx.NodeProto], list[onnx.TensorProto], str]:
    prefix = tensor_name.replace("/", "_")
    scale_name = make_unique_name(f"{prefix}{QDQ_SUFFIX}_weight_scale", used_names)
    zero_point_name = make_unique_name(f"{prefix}{QDQ_SUFFIX}_weight_zero_point", used_names)
    q_output_name = make_unique_name(f"{prefix}{QDQ_SUFFIX}_weight_quantized", used_names)
    dq_output_name = make_unique_name(f"{prefix}{QDQ_SUFFIX}_weight_dequantized", used_names)
    q_node_name = make_unique_name(f"{prefix}{QDQ_SUFFIX}_WeightQuantizeLinear", used_names)
    dq_node_name = make_unique_name(f"{prefix}{QDQ_SUFFIX}_WeightDequantizeLinear", used_names)

    scale = np.asarray(scale, dtype=np.float32)
    zero_point = np.zeros(scale.shape, dtype=np.int8)
    initializers = [
        numpy_helper.from_array(scale, scale_name),
        numpy_helper.from_array(zero_point, zero_point_name),
    ]

    q_attrs = {"axis": axis} if axis is not None else {}
    dq_attrs = {"axis": axis} if axis is not None else {}
    nodes = [
        helper.make_node(
            "QuantizeLinear",
            [tensor_name, scale_name, zero_point_name],
            [q_output_name],
            name=q_node_name,
            **q_attrs,
        ),
        helper.make_node(
            "DequantizeLinear",
            [q_output_name, scale_name, zero_point_name],
            [dq_output_name],
            name=dq_node_name,
            **dq_attrs,
        ),
    ]
    return nodes, initializers, dq_output_name


def get_attribute_int(node: onnx.NodeProto, name: str, default: int) -> int:
    for attr in node.attribute:
        if attr.name == name:
            return int(attr.i)
    return default


def get_weight_input_index_and_axis(node: onnx.NodeProto, weight_shape: tuple[int, ...]) -> tuple[int, int | None] | None:
    """Return the weight input index and per-channel axis for supported weighted ops."""
    if node.op_type == "Conv":
        return 1, 0
    if node.op_type == "ConvTranspose":
        # ONNX ConvTranspose weight layout is [C, M/group, kH, kW...].
        return 1, 1 if len(weight_shape) > 1 else 0
    if node.op_type == "Gemm":
        trans_b = get_attribute_int(node, "transB", 0)
        return 1, 0 if trans_b else 1
    if node.op_type == "MatMul":
        return 1, len(weight_shape) - 1
    return None


def compute_weight_scale(weight: np.ndarray, axis: int | None) -> tuple[np.ndarray, int | None]:
    """Compute per-channel symmetric INT8 weight scales from an initializer."""
    weight = np.asarray(weight)
    if not np.issubdtype(weight.dtype, np.floating):
        raise ValueError(f"Only floating-point weights can be quantized, got {weight.dtype}")

    if weight.ndim == 0:
        max_abs = float(np.max(np.abs(weight))) if weight.size else 0.0
        scale = max_abs / QUANT_MAX
        if scale == 0.0 or not np.isfinite(scale):
            scale = 1.0
        return np.array([scale], dtype=np.float32), None

    if axis is None:
        axis = 0
    if axis < 0:
        axis += weight.ndim
    if axis < 0 or axis >= weight.ndim:
        raise ValueError(f"Invalid per-channel axis {axis} for weight shape {weight.shape}")

    reduce_axes = tuple(dim for dim in range(weight.ndim) if dim != axis)
    max_abs = np.max(np.abs(weight), axis=reduce_axes)
    scale = max_abs.astype(np.float32) / np.float32(QUANT_MAX)
    scale = np.where(np.isfinite(scale) & (scale > 0.0), scale, np.float32(1.0)).astype(np.float32)
    return scale, axis


def get_activation_input_index(node: onnx.NodeProto, weight_input_index: int) -> int:
    """Return the input index that must have an activation scale for weight Q/DQ."""
    if node.op_type == "MatMul" and weight_input_index == 0:
        return 1
    return 0


def insert_qdq_from_cache(args: argparse.Namespace) -> tuple[int, int]:
    input_path = Path(args.input)
    cache_path = Path(args.cache)
    output_path = (
        Path(args.output)
        if args.output
        else SCRIPT_DIR / f"{input_path.stem}{DEFAULT_OUTPUT_SUFFIX}{input_path.suffix}"
    )

    model = onnx.load(str(input_path))
    graph = model.graph
    scales = parse_cache(cache_path)
    graph_tensors, initializer_names, consumers = collect_graph_names(graph)
    initializer_by_name = {initializer.name: numpy_helper.to_array(initializer) for initializer in graph.initializer}
    selected, skipped = select_activation_tensors(
        scales=scales,
        graph_tensors=graph_tensors,
        initializer_names=initializer_names,
        consumers=consumers,
    )
    selected_set = set(selected)

    print(f"Cache entries: {len(scales)}")
    print(f"Selected activation tensors: {len(selected)}")
    print(f"Skipped cache entries: {len(skipped)}")
    weight_edges: dict[tuple[int, int], str] = {}
    weight_qdq_nodes_by_key: dict[tuple[str, int | None], list[onnx.NodeProto]] = {}
    weight_qdq_initializers: list[onnx.TensorProto] = []
    weight_dq_by_key: dict[tuple[str, int | None], str] = {}
    skipped_weight_edges = 0

    # Weight Q/DQ is edge-scoped. A weight is quantized only when its matching
    # activation input also has a cache scale and will be rewired through Q/DQ.
    tmp_used_names = set(graph_tensors)
    tmp_used_names.update(node.name for node in graph.node if node.name)
    for node_index, node in enumerate(graph.node):
        if node.op_type not in WEIGHT_TARGET_OPS:
            continue

        if node.op_type == "MatMul":
            candidate_indices = [1, 0]
        else:
            candidate_indices = [1]

        for input_index in candidate_indices:
            if input_index >= len(node.input):
                continue
            weight_name = node.input[input_index]
            if weight_name not in initializer_by_name:
                continue

            activation_input_index = get_activation_input_index(node, input_index)
            if activation_input_index >= len(node.input) or node.input[activation_input_index] not in selected_set:
                skipped_weight_edges += 1
                continue

            weight = initializer_by_name[weight_name]
            weight_info = get_weight_input_index_and_axis(node, tuple(weight.shape))
            if weight_info is None:
                continue
            expected_input_index, axis = weight_info
            if node.op_type != "MatMul" and input_index != expected_input_index:
                continue
            if node.op_type == "MatMul" and input_index == 0:
                # Constant left input is uncommon; use the second-to-last dimension as the reduction-output axis.
                axis = max(weight.ndim - 2, 0)

            try:
                scale, normalized_axis = compute_weight_scale(weight, axis)
            except ValueError:
                skipped_weight_edges += 1
                continue

            key = (weight_name, normalized_axis)
            if key not in weight_dq_by_key:
                qdq_nodes, qdq_initializers, dq_output_name = make_weight_qdq_nodes(
                    weight_name, scale, normalized_axis, tmp_used_names
                )
                weight_qdq_nodes_by_key[key] = qdq_nodes
                weight_qdq_initializers.extend(qdq_initializers)
                weight_dq_by_key[key] = dq_output_name
            weight_edges[(node_index, input_index)] = weight_dq_by_key[key]

    print("Weight Q/DQ mode: per-channel")
    print(f"Weight target ops: {','.join(sorted(WEIGHT_TARGET_OPS))}")
    print(f"Selected weight edges: {len(weight_edges)}")
    print(f"Skipped weight edges: {skipped_weight_edges}")

    used_names = set(graph_tensors)
    used_names.update(node.name for node in graph.node if node.name)

    qdq_nodes_by_tensor: dict[str, list[onnx.NodeProto]] = {}
    replacement_by_tensor: dict[str, str] = {}
    for tensor_name in selected:
        qdq_nodes, qdq_initializers, dq_output_name = make_qdq_nodes(tensor_name, scales[tensor_name], used_names)
        qdq_nodes_by_tensor[tensor_name] = qdq_nodes
        replacement_by_tensor[tensor_name] = dq_output_name
        graph.initializer.extend(qdq_initializers)

    graph.initializer.extend(weight_qdq_initializers)

    emitted: set[str] = set()
    emitted_weight_keys: set[tuple[str, int | None]] = set()
    new_nodes: list[onnx.NodeProto] = []
    original_nodes = list(graph.node)

    for node_index, node in enumerate(original_nodes):
        for input_name in node.input:
            if input_name in selected_set and input_name not in emitted:
                new_nodes.extend(qdq_nodes_by_tensor[input_name])
                emitted.add(input_name)

        for input_index, input_name in enumerate(node.input):
            edge_key = (node_index, input_index)
            if edge_key not in weight_edges:
                continue
            weight = initializer_by_name[input_name]
            weight_info = get_weight_input_index_and_axis(node, tuple(weight.shape))
            if weight_info is None:
                continue
            _, axis = weight_info
            if node.op_type == "MatMul" and input_index == 0:
                axis = max(weight.ndim - 2, 0)
            _, normalized_axis = compute_weight_scale(weight, axis)
            weight_key = (input_name, normalized_axis)
            if weight_key not in emitted_weight_keys:
                new_nodes.extend(weight_qdq_nodes_by_key[weight_key])
                emitted_weight_keys.add(weight_key)

        for index, input_name in enumerate(node.input):
            if input_name in replacement_by_tensor and should_quantize_activation_edge(node):
                node.input[index] = replacement_by_tensor[input_name]
            if (node_index, index) in weight_edges:
                node.input[index] = weight_edges[(node_index, index)]
        new_nodes.append(node)

    del graph.node[:]
    graph.node.extend(new_nodes)

    onnx.checker.check_model(model)

    onnx.save(model, str(output_path))
    print(f"Wrote: {output_path}")
    print(f"Inserted Q/DQ pairs: {len(selected)}")
    print(f"Inserted weight Q/DQ pairs: {len(weight_dq_by_key)}")
    return len(selected), len(skipped)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(WORKSPACE_DIR / "yolov8s_640_dynamic.onnx"), help="Input ONNX model path.")
    parser.add_argument(
        "--cache",
        default=str(WORKSPACE_DIR / "yolov8s_gpu_precision_config_calib.cache"),
        help="TensorRT calibration cache path.",
    )
    parser.add_argument("--output", help="Output ONNX model path. Defaults to <input>_cache_qdq.onnx.")
    args = parser.parse_args()

    insert_qdq_from_cache(args)


if __name__ == "__main__":
    main()
