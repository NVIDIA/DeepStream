#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare installed TRT11 sample models offline, never inside nvinfer.

Run this script after installing DeepStream. Packaging copies scripts only.
Original FP32 ONNX bytes are preserved with an fp32_ prefix. Mixed models take
the canonical filenames so existing FP16 consumers need no filename changes.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


MODELS = (
    ("Primary_Detector", "TrafficCamNet/pruned_onnx_v1.0.3/resnet18_trafficcamnet_pruned.onnx",
     "resnet18_trafficcamnet_pruned.onnx", (960, 544), None, "RGB", 1.0 / 255.0),
    ("Secondary_VehicleMake", "VehicleMakeNet/pruned_v1.1.0/resnet18_pruned.tf2onnx.onnx",
     "resnet18_vehiclemakenet_pruned.onnx", (224, 224),
     ((405, 325, 480, 390), (360, 310, 407, 354), (470, 160, 1200, 650)), "BGR", 1.0),
    ("Secondary_VehicleTypes", "VehicleTypeNet/pruned_v1.1.0/resnet18_pruned.tf2onnx.onnx",
     "resnet18_vehicletypenet_pruned.onnx", (224, 224),
     ((405, 325, 480, 390), (360, 310, 407, 354), (470, 160, 1200, 650)), "BGR", 1.0),
)
TOOL_FILES = ("prepare_model.py", "requirements.txt", "README.md")
TEXT_EXTENSIONS = {".txt", ".yaml", ".yml", ".json", ".pbtxt", ".py", ".cpp", ".c", ".h", ".sh", ".md"}
MAX_INPUT_BYTES = 1024 * 1024 * 1024


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mixed_name(original):
    return original[:-5] + "_modelopt_fp16.onnx"


def fp32_name(original):
    return "fp32_" + original


def installed_source_model(root, directory, original):
    """Never infer an original from an already-replaced canonical model."""
    parent = Path(root) / "samples/models" / directory
    canonical, backup = parent / original, parent / fp32_name(original)
    state = parent / (original[:-5] + ".provenance.json")
    for path in (canonical, backup, state):
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise RuntimeError("Unsafe installed model/state: " + str(path))
    if backup.exists():
        if not backup.stat().st_size:
            raise RuntimeError("Empty FP32 backup: " + str(backup))
        if state.exists():
            try:
                provenance = load_json(state)["conversion"]
                source = provenance["request"]["source"]
                check_identity(backup, source["model"])
                for name, expected in source["external_data"].items():
                    check_identity(safe_input(parent, name), expected)
                if canonical.exists():
                    check_identity(canonical, provenance["outputs"][mixed_name(original)])
            except (KeyError, TypeError, ValueError, OSError, RuntimeError) as error:
                raise RuntimeError("FP32 backup/canonical provenance mismatch: " + str(parent)) from error
            return backup
        # Accept a pristine pre-copy, but never choose between differing files.
        if canonical.is_file() and identity(canonical) == identity(backup):
            return backup
        raise RuntimeError("Existing FP32 backup has no matching provenance; refusing to overwrite or guess: " + str(backup))
    if state.exists():
        raise RuntimeError("Installed mixed model is missing its FP32 backup: " + str(backup))
    if not canonical.is_file() or not canonical.stat().st_size:
        raise RuntimeError("Missing original installed sample input: " + str(canonical))
    return canonical


def identity(path):
    return {"sha256": sha256(path), "bytes": Path(path).stat().st_size}


def safe_input(root, relative):
    """Do not let a manifest or changed provider redirect an input outside root."""
    if (not isinstance(relative, str) or "\\" in relative or
            any(part in {"", ".", ".."} for part in relative.split("/"))):
        raise RuntimeError("Unsafe prepared-input path: " + repr(relative))
    path = Path(root)
    for part in relative.split("/"):
        path = path / part
        if path.is_symlink():
            raise RuntimeError("Symlink in prepared input: " + str(path))
    if not path.is_file():
        raise RuntimeError("Missing prepared input: " + str(path))
    return path


def check_identity(path, expected):
    if (not isinstance(expected, dict) or set(expected) != {"sha256", "bytes"} or
            not isinstance(expected["sha256"], str) or
            not re.fullmatch(r"[0-9a-f]{64}", expected["sha256"]) or
            type(expected["bytes"]) is not int or not 0 < expected["bytes"] <= MAX_INPUT_BYTES):
        raise RuntimeError("Invalid prepared-input identity: " + str(path))
    if Path(path).stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
        raise RuntimeError("Prepared-input identity mismatch: " + str(path))


def load_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeError("Duplicate prepared-input JSON key: " + key)
            result[key] = value
        return result
    if Path(path).stat().st_size > 2 * 1024 * 1024:
        raise RuntimeError("Prepared-input JSON is too large: " + str(path))
    return json.loads(Path(path).read_text(), object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Invalid JSON constant " + value)))


def child_environment():
    env = os.environ.copy()
    # Do not import packages from a caller's Sparse4D/system Python environment.
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env["PYTHONNOUSERSITE"] = "1"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    # Isolate CPU Torch from any inherited CUDA Torch loader directories while
    # preserving unrelated shared-library providers needed by the interpreter.
    if "LD_LIBRARY_PATH" in env:
        markers = ("libtorch_python.so", "libtorch.so", "libtorch_cpu.so", "libtorch_cuda.so", "libc10.so")
        paths = [entry for entry in env["LD_LIBRARY_PATH"].split(os.pathsep)
                 if not any((Path(entry or os.curdir) / name).is_file() for name in markers)]
        if paths:
            env["LD_LIBRARY_PATH"] = os.pathsep.join(paths)
        else:
            env.pop("LD_LIBRARY_PATH")
    return env


def run(command):
    print("ModelOpt build input: " + " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), env=child_environment(), check=True)


def source_identity(python, tool_dir, source):
    program = ("import json,sys; from pathlib import Path; sys.path.insert(0,sys.argv[1]); "
               "import prepare_model as p; import onnx; source=Path(sys.argv[2]).resolve(); "
               "print(json.dumps({'model':p.identity(source),'external':p.external_files(onnx.load(str(source), "
               "load_external_data=False),source)},sort_keys=True))")
    return json.loads(subprocess.check_output([str(python), "-I", "-B", "-c", program, str(tool_dir), str(source)],
                                             env=child_environment(), text=True))


def tool_python(tool_dir, cache, bootstrap_python=None):
    override = os.environ.get("DS_MODELOPT_PYTHON")
    probe = ("import sys; assert sys.version_info >= (3,12), 'ModelOpt host Python must be >=3.12'; "
             "from modelopt.onnx.autocast import convert_to_mixed_precision; import onnx, onnxruntime, numpy, PIL")
    if override:
        # Resolving a venv/bin/python symlink loses the virtual environment and
        # silently invokes its base interpreter. Keep the lexical absolute path.
        executable = Path(os.path.abspath(os.path.expanduser(override)))
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise RuntimeError("DS_MODELOPT_PYTHON is not an executable Python path: " + str(executable))
        run([executable, "-I", "-c", probe])
        return executable
    requirement_file = tool_dir / "requirements.txt"
    fingerprint = sha256(requirement_file)[:20]
    venv = cache / ("cpu-tools-" + fingerprint)
    python = venv / "bin/python"
    marker = venv / ".requirements-ready"
    if not marker.is_file():
        interpreter = bootstrap_python or os.environ.get("DS_MODELOPT_BOOTSTRAP_PYTHON")
        if not interpreter:
            for candidate in (shutil.which("python3.12"), sys.executable):
                if candidate and subprocess.run([candidate, "-c", "import sys; sys.exit(sys.version_info < (3, 12))"],
                                                env=child_environment()).returncode == 0:
                    interpreter = candidate
                    break
        if not interpreter:
            raise RuntimeError("Pinned ModelOpt preparation needs host Python >=3.12 with venv; set DS_MODELOPT_BOOTSTRAP_PYTHON.")
        if subprocess.run([interpreter, "-c", "import sys; sys.exit(sys.version_info < (3, 12))"],
                          env=child_environment()).returncode:
            raise RuntimeError("Selected ModelOpt bootstrap Python is older than 3.12")
        run([interpreter, "-m", "venv", venv])
        # Requirements pin CPU Torch explicitly. Never install into the existing
        # Sparse4D CUDA Torch venv or the system Python, including on ARM builds.
        # cppimport is an officially published small sdist in this pinned set;
        # allow pip to build its wheel rather than imposing an impossible
        # --only-binary requirement. Versions remain pinned in requirements.
        run([python, "-I", "-m", "pip", "install", "-r", requirement_file])
        run([python, "-I", "-m", "pip", "check"])
        run([python, "-I", "-c", probe])
        marker.write_text(sha256(requirement_file) + "\n")
    else:
        run([python, "-I", "-c", probe])
    return python


# Decode the already-shipped JPEG on the host; no CUDA, ffmpeg, display or GPU.
# The three car/bus crops are sample calibration, NOT labeled accuracy evidence.
CALIBRATION_PROGRAM = r'''
import json, sys
from pathlib import Path
import numpy as np
import onnx
from PIL import Image
model, image, output, width, height, boxes, color, scale = sys.argv[1:]
output = Path(output); output.mkdir(parents=True, exist_ok=True)
proto = onnx.load(model, load_external_data=False)
initializers = {x.name for x in proto.graph.initializer}
inputs = [x for x in proto.graph.input if x.name not in initializers]
if len(inputs) != 1 or inputs[0].type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
    raise RuntimeError("Shipped sample calibration requires one FLOAT input")
source = Image.open(image).convert("RGB")
if source.size != (1280, 720):
    raise RuntimeError("Sample crop coordinates require the known 1280x720 sample JPEG")
boxes = json.loads(boxes) or [None]
for index, box in enumerate(boxes):
    crop = source.crop(box) if box else source
    tensor = np.asarray(crop.resize((int(width), int(height)), Image.Resampling.BILINEAR), dtype=np.float32)
    if color == "BGR": tensor = tensor[:, :, ::-1]
    tensor = np.ascontiguousarray(tensor.transpose(2, 0, 1)[None] * np.float32(scale))
    np.savez(output / ("sample-%02d.npz" % index), **{inputs[0].name: tensor})
'''


def calibration_info(model, image):
    _, _, _, size, boxes, color, scale = model
    return {"kind": "shipped-sample-jpeg", "image_sha256": sha256(image),
            "image_size": [1280, 720], "crop_boxes": [list(box) for box in boxes] if boxes else None,
            "resize": list(size), "resize_filter": "Pillow BILINEAR", "color": color, "scale": scale,
            "layout": "NCHW", "dtype": "float32", "task_accuracy_qualified": False}

def host_supports_tools():
    if sys.platform != "linux":
        return True
    try:
        match = re.fullmatch(r"glibc ([0-9]+)\.([0-9]+)", os.confstr("CS_GNU_LIBC_VERSION") or "")
    except (OSError, ValueError):
        return False
    return bool(match and tuple(map(int, match.groups())) >= (2, 28))

def prepare_models(args, tool_dir, cache):
    assets = args.assets_root.resolve(strict=True)
    bootstrap = args.bootstrap_python
    installed = args.sdk_root
    python = tool_python(tool_dir, cache, bootstrap)
    image = assets / "streams/sample_720p.jpg"
    prepared = []
    for directory, relative, original, size, boxes, color, scale in MODELS:
        source = installed_source_model(installed, directory, original)
        if not source.is_file():
            raise RuntimeError("Missing original shipped model: " + str(source))
        calibration = args.calibration_root / directory if args.calibration_root else None
        if args.calibration_root:
            if not calibration.is_dir() or not list(calibration.glob("*.npz")):
                raise RuntimeError("Calibration override needs NPZ batches in " + str(calibration))
            calibration_details = {"kind": "explicit-npz-override"}
        else:
            if not image.is_file():
                raise RuntimeError("Missing representative sample JPEG: " + str(image))
            calibration_details = calibration_info((directory, relative, original, size, boxes, color, scale), image)
            calibration_key = hashlib.sha256(json.dumps(calibration_details, sort_keys=True).encode()).hexdigest()[:24]
            calibration = cache / "calibration" / directory / calibration_key
            run([python, "-I", "-c", CALIBRATION_PROGRAM, source, image, calibration,
                 size[0], size[1], json.dumps(boxes), color, scale])
        # Include full model closure, calibration, policy and tool identity.
        # Never invalidate maps from earlier invocations by overwriting an alias.
        request = {"source": source_identity(python, tool_dir, source),
                   "calibration": {path.name: sha256(path) for path in sorted(calibration.glob("*.npz"))},
                   "policy": sha256(Path(__file__)), "tool": sha256(tool_dir / "prepare_model.py"),
                   "requirements": sha256(tool_dir / "requirements.txt")}
        key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()[:24]
        output_dir = cache / "models" / directory / key
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        output_name = mixed_name(original)
        run([python, "-I", tool_dir / "prepare_model.py", "--input", source,
             "--output-dir", output_dir, "--output-name", output_name, "--calibration-data", calibration])
        output = output_dir / output_name
        provenance = json.loads((output_dir / "provenance.json").read_text())
        package_provenance = output_dir / (output_name[:-5] + ".provenance.json")
        package_provenance.write_text(json.dumps({"sample_calibration": calibration_details,
                                                  "conversion": provenance}, indent=2, sort_keys=True) + "\n")
        files = []
        for filename, expected in provenance["outputs"].items():
            if Path(filename).name != filename:
                raise RuntimeError("Unsafe prepared model output: " + filename)
            path = output_dir / filename
            if sha256(path) != expected["sha256"] or path.stat().st_size != expected["bytes"]:
                raise RuntimeError("Prepared model output changed: " + str(path))
            files.append(path)
        if output not in files:
            raise RuntimeError("Prepared ONNX is missing from its provenance inventory")
        prepared.append((directory, files + [package_provenance]))
    return prepared

def section_modes(lines, index):
    """Collect precision declarations only from the current INI/YAML group."""
    if any(re.match(r"\s*\[", line) for line in lines):
        start = index
        while start > 0 and not re.match(r"\s*\[", lines[start]):
            start -= 1
        end = index + 1
        while end < len(lines) and not re.match(r"\s*\[", lines[end]):
            end += 1
    else:
        indent = len(lines[index]) - len(lines[index].lstrip())
        start, end = index, index + 1
        while start > 0:
            candidate = lines[start - 1]
            if candidate.strip() and not candidate.lstrip().startswith("#") and len(candidate) - len(candidate.lstrip()) < indent:
                break
            start -= 1
        while end < len(lines):
            candidate = lines[end]
            if candidate.strip() and not candidate.lstrip().startswith("#") and len(candidate) - len(candidate.lstrip()) < indent:
                break
            end += 1
    modes = [int(match.group(1)) for line in lines[start:end]
             if (match := re.match(r"\s*network-mode\s*[:=]\s*([0-9]+)\s*(?:#.*)?$", line))]
    return modes

def installed_rewrite_text(text):
    """Canonical names remain FP16; only original-precision consumers move.

    Also migrate the previous separate-name setup, without rewriting unrelated
    model names or repeatedly adding fp32_ on a second invocation.
    """
    lines = text.splitlines(keepends=True)
    boundary = r"(?<![A-Za-z0-9_])"
    for index, line in enumerate(lines):
        direct = bool(re.match(r"\s*(?:onnx-file|model-file)\s*[:=]", line))
        for _, _, original, *_ in MODELS:
            names = r"(?:" + re.escape(original) + "|" + re.escape(mixed_name(original)) + ")"
            engine = boundary + names + r"(_b[0-9]+(?:_(?:gpu|dla)[0-9]+)?_(fp16|fp32|int8)\.engine)"
            line = re.sub(engine, lambda match: (original if match.group(2) == "fp16" else fp32_name(original)) + match.group(1), line)
            if direct and re.search(boundary + names + r"(?![_\w])", line):
                modes = section_modes(lines, index)
                if modes not in ([], [0], [1], [2]):
                    raise RuntimeError("Unsupported or conflicting network-mode for canonical sample model")
                selected = original if modes == [2] else fp32_name(original)
                line = re.sub(boundary + names + r"(?![_\w])", selected, line)
        lines[index] = line
    return "".join(lines)

def installed_preflight(args):
    """Check installation inputs without downloads, imports or output changes."""
    root = args.sdk_root.expanduser().resolve(strict=True)
    required = [installed_source_model(root, directory, original)
                for directory, _, original, *_ in MODELS]
    required.append(root / "samples/streams/sample_720p.jpg")
    missing = [str(path) for path in required if not path.is_file() or not path.stat().st_size]
    if missing:
        raise RuntimeError("Missing original installed sample inputs (no download attempted):\n  " + "\n  ".join(missing))
    if args.batch_size < 1 or args.gpu_id < 0:
        raise RuntimeError("--batch-size must be positive and --gpu-id must be nonnegative")
    if not args.prepare_only:
        builder = args.builder or root / "bin/nvdsinfer-engine-builder"
        resolved = shutil.which(str(builder))
        if not resolved:
            raise RuntimeError("Missing executable SDK engine builder: " + str(builder))
        args.builder = str(Path(resolved).absolute())
    args.sdk_root = root
    args.assets_root = root / "samples"
    return root

def build_installed_engines(args, prepared, cache):
    """Build fresh GPU engines; existence alone is never a cache-validity test."""
    version = subprocess.check_output([str(args.builder), "--version"], env=child_environment(), text=True).strip()
    if not re.fullmatch(r"TensorRT 11\.[0-9]+\.[0-9]+\.[0-9]+", version):
        raise RuntimeError("This workflow requires the matching TensorRT 11 SDK builder, got: " + version)
    work = Path(tempfile.mkdtemp(prefix="engine-build-", dir=cache))
    args._engine_work = work
    try:
        return _build_installed_engines(args, prepared, work)
    except BaseException:
        shutil.rmtree(work)  # Only this invocation's newly owned engine scratch.
        args._engine_work = None
        raise

def _build_installed_engines(args, prepared, work):
    result = []
    for directory, files in prepared:
        original = next(model[2] for model in MODELS if model[0] == directory)
        name = mixed_name(original)
        model = next(path for path in files if path.name == name)
        provenance = load_json(next(path for path in files if path.name == name[:-5] + ".provenance.json"))
        inputs = provenance["conversion"]["external_io"]["inputs"]
        shapes = []
        for value in inputs:
            dims = value["shape"]
            if not dims or any(type(dimension) is not int or dimension <= 0 for dimension in dims[1:]):
                raise RuntimeError("Only a dynamic batch dimension is supported by sample setup: " + value["name"])
            dynamic = type(dims[0]) is not int or dims[0] <= 0
            if not dynamic and dims[0] != args.batch_size:
                raise RuntimeError("Requested batch does not match static ONNX batch for " + value["name"])
            if dynamic:
                # The SDK builder splits at the LAST colon; shipped names
                # themselves contain a colon (input_1:0).
                if any(char in value["name"] for char in ",\n\r"):
                    raise RuntimeError("Unsupported shape-selector input name")
                shapes.append((value["name"], dims[1:]))
        target = original + "_b%d_gpu%d_fp16.engine" % (args.batch_size, args.gpu_id)
        destination = work / directory / target
        destination.parent.mkdir()
        command = [args.builder, "--onnx=" + str(model), "--saveEngine=" + str(destination),
                   "--fp16", "--device=" + str(args.gpu_id)]
        if shapes:
            for selector, batch in (("minShapes", 1), ("optShapes", args.batch_size), ("maxShapes", args.batch_size)):
                command.append("--" + selector + "=" + ",".join(
                    name + ":" + "x".join(map(str, [batch] + tail)) for name, tail in shapes))
        run(command)
        if not destination.is_file() or not destination.stat().st_size:
            raise RuntimeError("Engine builder did not produce a nonempty engine: " + str(destination))
        result.append((Path("samples/models") / directory / target, destination))
    return result

def publish_installed(root, prepared, engines):
    """Stage outputs first; rollback publication errors without changing originals.

    Call only while sample pipelines are stopped and the installation lock is held.
    Engines for other batch sizes remain nvinfer's normal on-demand responsibility.
    """
    outputs, expected_copies, source_checks, external_inputs = [], {}, [], []
    changed_directories = []
    for directory, files in prepared:
        original = next(model[2] for model in MODELS if model[0] == directory)
        parent = root / "samples/models" / directory
        source = installed_source_model(root, directory, original)
        converted = next(path for path in files if path.name == mixed_name(original))
        provenance = next(path for path in files if path.name == mixed_name(original)[:-5] + ".provenance.json")
        conversion = load_json(provenance)["conversion"]
        check_identity(source, conversion["request"]["source"]["model"])
        external = conversion["request"]["source"]["external_data"]
        for name, expected in external.items():
            payload = safe_input(parent, name)
            check_identity(payload, expected)
            external_inputs.append(payload)
        check_identity(converted, conversion["outputs"][mixed_name(original)])
        source_checks.append((source, conversion["request"]["source"]["model"]))
        source_checks.extend((safe_input(parent, name), expected) for name, expected in external.items())
        canonical, backup = parent / original, parent / fp32_name(original)
        state = parent / (original[:-5] + ".provenance.json")
        if not backup.exists():
            outputs.append((backup, source))
            expected_copies[backup] = conversion["request"]["source"]["model"]
        outputs.extend(((canonical, converted), (state, provenance)))
        expected_copies[canonical] = conversion["outputs"][mixed_name(original)]
        expected_copies[state] = identity(provenance)
        if (not canonical.is_file() or sha256(canonical) != sha256(converted) or
                not state.is_file() or sha256(state) != sha256(provenance)):
            changed_directories.append((parent, original))
    outputs.extend((root / relative, source) for relative, source in engines)
    expected_copies.update((root / relative, identity(source)) for relative, source in engines)
    replaced = {destination for destination, _ in outputs}
    # nvinfer can otherwise accept an older, shape-compatible engine without
    # checking its ONNX identity. Retire only our derived FP16 plan names when
    # the prepared model/provenance changes, including other batches/devices.
    for parent, original in changed_directories:
        names = r"(?:" + re.escape(original) + "|" + re.escape(mixed_name(original)) + ")"
        pattern = re.compile(names + r"_b[0-9]+(?:_(?:gpu|dla)[0-9]+)?_fp16\.engine")
        for path in sorted(parent.iterdir()):
            if pattern.fullmatch(path.name) and path not in replaced:
                outputs.append((path, None))
    for base in (root / "samples/configs", root / "sources", root / "service-maker/sources"):
        if base.is_dir():
            for path in sorted(base.rglob("*")):
                if path.is_symlink() or not path.is_file() or (path.suffix not in TEXT_EXTENSIONS and not path.name.startswith("README")):
                    continue
                try:
                    before = path.read_text()
                except UnicodeError:
                    continue
                text = installed_rewrite_text(before)
                if text != before:
                    outputs.append((path, text))
    # Protect every original model's validated external payload from ALL output
    # actions, including engine replacement/retirement and rewritten configs.
    # A legal ONNX weight filename can look exactly like one of our old engines.
    # Check the complete plan before making even private staging directories;
    # canonical ONNX replacement remains intentional because its backup is kept.
    payload_paths, payload_inodes = {}, {}
    for payload in external_inputs:
        payload_paths[payload.resolve(strict=True)] = payload
        info = payload.stat()
        payload_inodes[(info.st_dev, info.st_ino)] = payload
    for destination, _ in outputs:
        if destination.is_symlink() or (destination.exists() and not destination.is_file()):
            raise RuntimeError("Refusing unsafe installed output: " + str(destination))
        payload = payload_paths.get(destination.resolve())
        if payload is None and destination.exists():
            info = destination.stat()
            payload = payload_inodes.get((info.st_dev, info.st_ino))
        if payload is not None:
            raise RuntimeError("External model data collides with installed publication: %s -> %s" %
                               (destination, payload))
    staged, committed, preserve = [], [], set()
    try:
        for destination, source in outputs:
            # Per-parent directories keep renames on the destination filesystem.
            private = Path(tempfile.mkdtemp(prefix=".modelopt-", dir=destination.parent))
            staged.append((destination, private))
            temporary = private / "new"
            if source is None:
                continue  # Retired plan: restore on error, discard only on success.
            if isinstance(source, str):
                temporary.write_text(source)
                shutil.copymode(destination, temporary)
            else:
                shutil.copyfile(source, temporary)
                check_identity(temporary, expected_copies[destination])
                temporary.chmod(destination.stat().st_mode & 0o777 if destination.exists() else 0o644)
        for source, expected in source_checks:
            check_identity(source, expected)
        for destination, private in staged:
            had_original = destination.exists()
            if had_original:
                os.replace(destination, private / "old")
            committed.append((destination, private, had_original))
            if (private / "new").exists():
                os.replace(private / "new", destination)
    except BaseException as failure:
        recovery = []
        created = {private for _, private, had_original in committed if not had_original}
        # Inspect all actual backups, not just completed bookkeeping: an
        # interruption can arrive after rename succeeds but before append().
        for destination, private in reversed(staged):
            try:
                if (private / "old").exists():
                    os.replace(private / "old", destination)
                elif private in created and destination.exists():
                    destination.unlink()
            except OSError:
                # Never discard the only recoverable original if restoration
                # itself fails (for example, a filesystem becomes read-only).
                preserve.add(private)
                recovery.append(str(private))
        if recovery:
            raise RuntimeError("Publication rollback needs manual recovery; retained backups: " + ", ".join(recovery)) from failure
        raise
    finally:
        for _, private in staged:
            if private not in preserve:
                try:
                    shutil.rmtree(private)  # Only newly owned staging, after commit/rollback.
                except OSError as error:
                    print("WARNING: retained setup staging %s: %s" % (private, error), file=sys.stderr)
    print("ModelOpt: installed mixed models at canonical names and %d fresh engines; originals preserved as fp32_ backups." % len(engines))

def installed_main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-root", type=Path, default=Path(__file__).resolve().parents[2],
                        help="Installed DeepStream root (default: inferred from samples/modelopt)")
    parser.add_argument("--cache-dir", type=Path, default=Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "deepstream/modelopt",
                        help="Private conversion environment/model cache; never included in the package")
    parser.add_argument("--check", action="store_true", help="Check required original inputs/builder only; no downloads or writes")
    parser.add_argument("--prepare-only", action="store_true", help="Prepare models/update FP16 samples without GPU engine building")
    parser.add_argument("--batch-size", type=int, default=1, help="Prebuild this batch size (default 1); nvinfer builds other sizes on demand")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--builder", help="Matching installed SDK nvdsinfer-engine-builder executable")
    parser.add_argument("--bootstrap-python", default=os.environ.get("DS_MODELOPT_BOOTSTRAP_PYTHON"))
    parser.add_argument("--calibration-root", type=Path, default=os.environ.get("DS_MODELOPT_CALIBRATION_ROOT"))
    args = parser.parse_args(argv)
    try:
        root = installed_preflight(args)
        if args.check:
            print("ModelOpt: original sample files and selected builder are present. No conversion/inference validated; no changes made.")
            return 0
        if not host_supports_tools():
            raise RuntimeError("Run post-install preparation on a modern Linux target (glibc >=2.28), with the pinned CPU preparation tools")
        # One lock per installation, even if callers select different caches.
        with (root / "samples/models/.modelopt-setup.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            installed_preflight(args)  # State may have changed while waiting for the installation lock.
            cache = args.cache_dir.expanduser().resolve()
            cache.mkdir(parents=True, exist_ok=True)
            with (cache / ".prepare.lock").open("a") as cache_lock:
                fcntl.flock(cache_lock, fcntl.LOCK_EX)
                prepared = prepare_models(args, Path(__file__).resolve().parent, cache)
                engines = [] if args.prepare_only else build_installed_engines(args, prepared, cache)
                publish_installed(root, prepared, engines)
        print("Setup complete. Batch-%d GPU engines were%s prebuilt; other sample batches build on demand." %
              (args.batch_size, " not" if args.prepare_only else ""))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, StopIteration, subprocess.CalledProcessError) as error:
        print("ERROR: ModelOpt post-install setup: " + str(error), file=sys.stderr)
        return 1
    finally:
        work = getattr(args, "_engine_work", None)
        if work:
            shutil.rmtree(work)  # Already published, newly owned scratch; not a reusable model cache.

def main(argv=None):
    return installed_main(argv)


if __name__ == "__main__":
    sys.exit(main())
