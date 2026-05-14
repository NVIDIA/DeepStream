#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import os
import subprocess
import shutil
from typing import Optional, List
from pathlib import Path
from dataclasses import dataclass

@dataclass
class Target:
    name: str
    models: List[str]
    needs_processor: bool = False

TARGETS = {
    "tao": Target("tao", ["rtdetr", "cls", "seg"]),
    "gdino": Target("gdino", ["gdino", "mgdino"], needs_processor=True),
    "changenet": Target("changenet", [])
}

def clean_tmp(target: Optional[str] = None, model: Optional[str] = None) -> None:
    """Clean temporary directories."""
    validation_path = Path("builder/samples/tao/validation")

    if not target:
        print("Cleaning all temporary directories...")
        for path in validation_path.glob("*/.tmp"):
            print(f"Removing {path}")
            shutil.rmtree(path, ignore_errors=True)
        return

    target_info = TARGETS[target]
    if not model:
        print(f"Cleaning temporary directories for {target} models...")
        for model_name in target_info.models:
            tmp_path = validation_path / model_name / ".tmp"
            print(f"Removing {tmp_path}")
            shutil.rmtree(tmp_path, ignore_errors=True)
    else:
        print(f"Cleaning temporary directory for {model}...")
        tmp_path = validation_path / model / ".tmp"
        print(f"Removing {tmp_path}")
        shutil.rmtree(tmp_path, ignore_errors=True)

def build_target(target: str, model: Optional[str] = None) -> None:
    """Build a specific target with optional model validation."""
    print("\n" + "="*80)  # Add a clear separator line
    target_info = TARGETS[target]
    ds_prefix = f"ds_{target}"

    cmd = [
        "python", "builder/main.py",
        f"builder/samples/tao/{ds_prefix}.yaml",
        "--server-type", "fastapi",
        "-a", "builder/samples/tao/openapi.yaml",
        "-o", "builder/samples/tao",
        "-t"
    ]

    if model:
        cmd.extend(["--validation-dir", f"builder/samples/tao/validation/{model}"])

    if target_info.needs_processor:
        cmd.extend(["-c", "builder/samples/tao/processors.py"])

    # If running in container, don't launch openapi generator container to build client
    # eg, in CI, export NO_DOCKER=true
    no_docker = os.environ.get("NO_DOCKER", "false").lower() == "true"
    if no_docker:
        cmd.extend(["--no-docker"])

    print(f"Building {target}" + (f" with {model} validation" if model else ""))
    print("="*80 + "\n")
    subprocess.run(cmd, check=True)

def build_all_models(target: str) -> None:
    """Build all models for a specific target."""
    target_info = TARGETS[target]
    if target_info.models:
        for model in target_info.models:
            build_target(target, model)
    else:
        build_target(target)

def main() -> None:
    parser = argparse.ArgumentParser(description="Build pipeline targets with validation")
    parser.add_argument("-t", "--target", choices=TARGETS.keys(),
                      help="Select target pipeline (tao, changenet, gdino)")
    parser.add_argument("-m", "--model", help="Select AI model type for validation")
    args = parser.parse_args()

    # Validate model selection
    if args.target and args.model:
        target_info = TARGETS[args.target]
        if args.model not in target_info.models:
            valid_models = ", ".join(target_info.models) if target_info.models else "none"
            parser.error(f"Invalid model for {args.target}. Valid options: {valid_models}")

    # Change to the inference_builder root directory
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[3]
    os.chdir(repo_root)

    # Clean directories
    clean_tmp(args.target, args.model)

    try:
        if not args.target:
            # Build all targets
            for target in TARGETS:
                if TARGETS[target].models:
                    build_all_models(target)
                else:
                    build_target(target)
        elif not args.model:
            # Build all models for target
            build_all_models(args.target)
        else:
            # Build specific target with model
            build_target(args.target, args.model)
    except subprocess.CalledProcessError as e:
        print(f"Build failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()