# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Collect the system + tool environment a run executed on, for report provenance.

Used by make_report.py to render the "System & environment" section so every
report is reproducible (what GPU/driver/DeepStream/TensorRT and which package
versions produced these numbers). Everything is best-effort: a missing tool
yields "unknown" rather than an error. Run inside the DeepStream container (where
nvidia-smi, the DeepStream version file, and TensorRT headers are visible).

CLI:  python collect_env.py --out env_info.json [--image <container image>]
API:  collect_env.collect(image="") -> dict
"""
import argparse
import json
import os
import platform
import re
import subprocess
from pathlib import Path


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return ""


def _cpu():
    model, cores = "unknown", os.cpu_count()
    try:
        txt = Path("/proc/cpuinfo").read_text()
        m = re.search(r"model name\s*:\s*(.+)", txt)
        if m:
            model = m.group(1).strip()
        phys = len(set(re.findall(r"physical id\s*:\s*(\d+)", txt))) or 1
        return f"{model}  ({cores} logical cores, {phys} socket{'s' if phys != 1 else ''})"
    except Exception:
        return f"{model} ({cores} logical cores)"


def _ram():
    try:
        m = re.search(r"MemTotal:\s*(\d+)\s*kB", Path("/proc/meminfo").read_text())
        if m:
            return f"{int(m.group(1)) / 1024 / 1024:.0f} GB"
    except Exception:
        pass
    return "unknown"


def _gpu():
    out = _run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader"])
    if not out:
        return "unknown", "unknown", "unknown"
    lines = [l.strip() for l in out.splitlines() if l.strip()]
    parts = [p.strip() for p in lines[0].split(",")]
    name = parts[0] if parts else "unknown"
    mem = parts[1] if len(parts) > 1 else ""
    driver = parts[2] if len(parts) > 2 else "unknown"
    count = len(lines)
    gpu = f"{count}× {name}" + (f"  ({mem})" if mem else "")
    return gpu, driver, _cuda_driver()


def _cuda_driver():
    txt = _run(["nvidia-smi"])
    m = re.search(r"CUDA Version:\s*([0-9.]+)", txt)
    return m.group(1) if m else "unknown"


def _tensorrt():
    # The DS container ships TensorRT via apt — dpkg is the reliable source
    # (the NvInferVersion.h macros are indirection, not literals).
    for pkg in ("libnvinfer-bin", "libnvinfer-dev", "libnvinfer10"):
        v = _run(["dpkg-query", "-W", "-f=${Version}", pkg])
        if v:
            m = re.match(r"([0-9]+(?:\.[0-9]+)+)(?:.*?(cuda[0-9.]+))?", v)
            if m:
                return m.group(1) + (f" ({m.group(2)})" if m.group(2) else "")
    try:
        import tensorrt  # noqa
        return tensorrt.__version__
    except Exception:
        return "unknown"


def _deepstream():
    for p in ("/opt/nvidia/deepstream/deepstream/version",
              "/opt/nvidia/deepstream/deepstream-9.1/version"):
        try:
            t = Path(p).read_text()
            m = re.search(r"Version:\s*([0-9.]+)", t)
            if m:
                return m.group(1)
        except Exception:
            pass
    return "unknown"


def _pkgs():
    from importlib import metadata
    want = ["torch", "torchvision", "transformers", "datasets", "torchmetrics",
            "onnx", "onnxscript", "pycocotools", "numpy", "albumentations", "reportlab"]
    out = {}
    for p in want:
        try:
            out[p] = metadata.version(p)
        except Exception:
            pass
    try:
        import torch
        out["torch"] = f"{torch.__version__} (CUDA {torch.version.cuda})"
    except Exception:
        pass
    return out


def collect(image=""):
    gpu, driver, cuda = _gpu()
    return {
        "os": platform.platform(),
        "python": platform.python_version(),
        "cpu": _cpu(),
        "ram": _ram(),
        "gpu": gpu,
        "gpu_driver": driver,
        "cuda_driver": cuda,
        "deepstream": _deepstream(),
        "tensorrt": _tensorrt(),
        "container_image": image or os.environ.get("DS_IMAGE", ""),
        "packages": _pkgs(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--image", default="", help="DeepStream container image (for provenance)")
    args = ap.parse_args()
    info = collect(image=args.image)
    Path(args.out).write_text(json.dumps(info, indent=2))
    print(f"[collect_env] wrote {args.out}: GPU={info['gpu']} · DS={info['deepstream']} · TRT={info['tensorrt']}")


if __name__ == "__main__":
    main()
