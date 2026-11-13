<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Post-install sample model preparation

DeepStream packages these scripts under `samples/modelopt`:

```text
/opt/nvidia/deepstream/deepstream/samples/modelopt/
```

The canonical `deepstream` directory is the normal alias of the versioned SDK.
Packaging copies these scripts only. Packaging does not invoke ModelOpt,
install conversion dependencies, convert ONNX or build target engines. Original sample models are packaged normally, unchanged. ModelOpt is
not an inference-plugin or Debian runtime dependency.

## Run after installation

Stop sample pipelines before setup. Sample model/configuration and source
directories must be writable by the account running setup; do not change system
Python or broadly relax permissions as a shortcut.

```sh
cd /opt/nvidia/deepstream/deepstream/samples/modelopt

# Read-only preflight: missing original files or SDK engine builder.
python3.12 prepare_samples.py --check

# Prepare three models and prebuild batch-1 GPU engines.
python3.12 prepare_samples.py
```

`--sdk-root /path/to/deepstream-X.Y` selects another installation, including a
private writable SDK copy. Without it, setup infers the root from its installed
`samples/modelopt` location. The source checkout helper requires an explicit
installed `--sdk-root`; the source checkout is not an installed SDK.

The script:

1. Checks the original TrafficCamNet, VehicleMakeNet and VehicleTypeNet ONNX
   files (or their verified `fp32_` backups on a repeat run) and
   `samples/streams/sample_720p.jpg`. Missing or conflicting inputs fail
   **before tool installation**; the script does not download models.
2. Uses an isolated pinned CPU conversion environment. First use creates a
   private virtual environment and installs `requirements.txt`; no system
   packages, CUDA wheels or Sparse4D environment are modified.
3. Calls unmodified NVIDIA ModelOpt AutoCast privately for mixed FP16/FP32 graphs.
   Cached conversion reuse verifies model/external weights, calibration, complete
   environment/tool identity and output bytes, not just file existence.
4. Uses installed `bin/nvdsinfer-engine-builder` with target TensorRT/CUDA to
   build fresh GPU engines for the requested batch size. Existing plans alone
   are not trusted; no engine from a different platform is shipped/downloaded.
5. Only after all requested conversions/builds succeed, preserves each original
   ONNX as `fp32_<original-name>.onnx` and publishes its mixed graph at the
   **original filename**. Existing FP16 model/engine references, including
   Service Maker overrides, keep their normal names. A handled publication
   failure rolls back already changed files.

The backup is not overwritten on repeat runs. Setup verifies its relationship
to the installed model/provenance and always converts from that FP32 backup,
never from an already mixed graph. A conflicting backup fails instead of being
silently replaced. Do not manually change the model/backup/provenance while
setup is running.

For known models in `samples/configs`, `sources` and `service-maker/sources`,
setup preserves direct FP32/INT8 ONNX consumers by pointing their model and
matching cached-engine references at the `fp32_` name. An omitted `network-mode`
uses the normal FP32 default; explicit mode `2` keeps the canonical mixed model.
Unknown or conflicting modes for a direct model reference fail before publishing.
This does not turn a legacy INT8 model/calibration cache into a TRT11 Q/DQ model.
References produced by the previous `_modelopt_fp16.onnx` setup are migrated back
to the canonical names.

When canonical mixed model/provenance changes, setup invalidates those models'
old canonical and previous `_modelopt_fp16` FP16 engines (including other
batches and GPU/DLA suffixes) so changed weights cannot reuse old plans. Newly
built requested engines are published in their place. FP32/INT8 and unrelated
engines are retained; preserved FP32/INT8 consumers use their `fp32_` engine
names and may rebuild on first use.

Fresh engine-build scratch is removed after success/failure; reusable conversion
and environment caches remain.

Default engine batch is **1**, GPU **0**. Other batches are built by `nvinfer`
on first use; a batch-1 plan does not satisfy a batch-30 app.

```sh
python3.12 prepare_samples.py --batch-size 4 --gpu-id 0
python3.12 prepare_samples.py --prepare-only
```

`--prepare-only` converts and publishes the canonical models/backups without
requiring a GPU builder; the application builds its engine on first use. The SDK
builder's `--fp16` validates the prepared graph; it is not the removed TensorRT
precision flag. This setup is for TensorRT 11 GPU engines, not DLA qualification.

## Environment and outputs

Conversion needs Python **3.12 or newer** with `venv`, Linux glibc **>=2.28**,
and download access for first-time provisioning. ARM64 wheel availability does not establish actual installation or
runtime qualification; those remain target-specific checks.

Private cache: `$XDG_CACHE_HOME/deepstream/modelopt` or
`~/.cache/deepstream/modelopt`. Allow room for the environment (roughly 1.5 GiB
in the tested x86 setup), conversion intermediates and target engines; sizes vary.
A compiler may be needed for pinned small source-distribution dependencies.
No environment, wheel, generated graph, engine or conversion cache is added to
the package by these packaging rules.

Options/environment:

- `--cache-dir PATH`: private preparation/build cache.
- `DS_MODELOPT_PYTHON=/path/to/qualified-venv/bin/python`: reuse an already
  provisioned pinned environment instead of installing a second environment.
- `--bootstrap-python PATH` / `DS_MODELOPT_BOOTSTRAP_PYTHON`: interpreter used
  to create the isolated environment.
- `--builder PATH`: matching target SDK engine builder.
- `--calibration-root PATH` / `DS_MODELOPT_CALIBRATION_ROOT`: representative
  NPZ directories `Primary_Detector`, `Secondary_VehicleMake` and
  `Secondary_VehicleTypes`. Default: shipped JPEG and fixed vehicle crops.

Files are published under `samples/models/<model-directory>`:

| Directory | Mixed model at existing filename | Preserved original |
| --- | --- | --- |
| `Primary_Detector` | `resnet18_trafficcamnet_pruned.onnx` | `fp32_resnet18_trafficcamnet_pruned.onnx` |
| `Secondary_VehicleMake` | `resnet18_vehiclemakenet_pruned.onnx` | `fp32_resnet18_vehiclemakenet_pruned.onnx` |
| `Secondary_VehicleTypes` | `resnet18_vehicletypenet_pruned.onnx` | `fp32_resnet18_vehicletypenet_pruned.onnx` |

Each canonical graph has a matching `<model-stem>.provenance.json`. Engine names
retain the canonical ONNX filename, batch and GPU ordinal, for example
`resnet18_trafficcamnet_pruned.onnx_b1_gpu0_fp16.engine`.

**The original filename now selects mixed computation.** `network-mode=0` does
not turn that graph back into FP32. Custom configs outside the scanned SDK trees,
application-generated model paths and arbitrary source-code strings must explicitly
select the `fp32_` backup and a separate engine path when FP32 is required. The
script cannot infer every application's configuration or precision intent.

`--check` checks input/builder availability only. It does not install/import
conversion dependencies, create caches, validate ONNX arithmetic, prove engine
freshness or run inference. Passing preflight is not accuracy/release approval.

## Prepare another model

Supply representative inputs **after the application's preprocessing**, not raw
JPEG bytes. NPZ keys, shapes and dtypes must exactly match ONNX input names.

```sh
/path/to/qualified-venv/bin/python prepare_model.py \
  --input /path/to/original/model.onnx \
  --output-dir /path/to/private/model-v1 \
  --output-name model.modelopt-fp16.onnx \
  --calibration-data /path/to/reference-npz-directory
```

The tool uses CPU execution and `keep_io_types=True`. It rejects custom
operations, local ONNX functions and Q/DQ graphs in this route; these need their
producer's typed model or a qualified plugin-aware workflow. Original ONNX and
external weights remain unchanged. Inspect a stale output lock only after
confirming the owning conversion process has stopped.

Provenance binds the source closure, reference inputs, complete tool environment
and outputs. Cached mismatches fail rather than selecting old bytes. Preparation
requires output-reachable, input-dependent FP16 arithmetic, not HALF constants,
a Cast or a shape-only branch. Public I/O types stay unchanged; this does not prove
every fused kernel's accumulator precision or task accuracy.

The JPEG/crops are reproducible sample references, **not a labeled dataset**.
Qualify numeric/parsed outputs, representative dataset accuracy, cold build,
latency and memory on each target. CPU conversion or x86 engine success cannot
certify Jetson, SBSA, DLA or a complete pipeline.

There is no producer-bundle or build-system input selection mode in this public
helper. Source checkout use requires an explicit installed `--sdk-root`.

Run source tests:

```sh
python3.12 -B -m unittest discover -s tools/modelopt -p 'test_*.py' -v
```

Official reference: [NVIDIA ModelOpt AutoCast](https://github.com/NVIDIA/Model-Optimizer/blob/main/docs/source/guides/8_autocast.rst).
