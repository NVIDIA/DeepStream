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

"""Authoritative engine perf via trtexec — qps (FPS) + GPU compute latency.

The end-to-end `pipeline_fps` from eval_engine.py includes nvinfer's one-time
engine build, so it is NOT real throughput. This benchmarks the serialized
engine directly (`trtexec --loadEngine`, GPU-only timing) and writes a perf.json
with the true single-stream FPS and latency (mean + p99). Run in the DeepStream
container.
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--engine", help="serialized .engine to load + benchmark (fast, no rebuild)")
    g.add_argument("--onnx", help="ONNX to build + benchmark (slower; use if no engine)")
    ap.add_argument("--out", required=True, help="perf.json output")
    ap.add_argument("--shapes", default="pixel_values:1x3x640x640")
    ap.add_argument("--trtexec", default="/usr/src/tensorrt/bin/trtexec")
    ap.add_argument("--batch", type=int, default=1)
    args = ap.parse_args()

    if args.engine:
        cmd = [args.trtexec, f"--loadEngine={args.engine}", f"--shapes={args.shapes}"]
        src = args.engine
    else:
        cmd = [args.trtexec, f"--onnx={args.onnx}", "--fp16",
               f"--minShapes={args.shapes}", f"--optShapes={args.shapes}", f"--maxShapes={args.shapes}"]
        src = args.onnx
    print(f"[bench_trtexec] {' '.join(cmd)}")
    out = subprocess.run(cmd, capture_output=True, text=True)
    log = out.stdout + out.stderr

    qps = re.search(r"Throughput:\s*([\d.]+)\s*qps", log)
    gpu = re.search(r"GPU Compute Time:.*?mean\s*=\s*([\d.]+)\s*ms.*?percentile\(99%\)\s*=\s*([\d.]+)\s*ms", log)
    if not qps:
        print(log[-2000:]); sys.exit("ERROR: could not parse trtexec Throughput")

    perf = {
        "trtexec_qps": round(float(qps.group(1)), 2),       # single-stream FPS (batch=1)
        "gpu_compute_ms_mean": round(float(gpu.group(1)), 4) if gpu else None,
        "gpu_compute_ms_p99": round(float(gpu.group(2)), 4) if gpu else None,
        "batch": args.batch, "precision": "fp16", "source": src,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    # Merge into an existing perf.json (keeps eval_engine's pipeline_fps if present).
    if Path(args.out).exists():
        try:
            perf = {**json.loads(Path(args.out).read_text()), **perf}
        except Exception:
            pass
    Path(args.out).write_text(json.dumps(perf, indent=2))
    print(f"[bench_trtexec] qps={perf['trtexec_qps']} gpu_mean={perf['gpu_compute_ms_mean']}ms "
          f"p99={perf['gpu_compute_ms_p99']}ms -> {args.out}")


if __name__ == "__main__":
    main()
