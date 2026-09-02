#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved. Apache-2.0.
#
# pip helper for the NGC PyTorch image entrypoints (run_ngc_finetune.sh, run_ngc_automl.sh).
#
# WHY: NGC images ship a global pip constraint file (`PIP_CONSTRAINT=/etc/pip/constraint.txt`)
# holding NVIDIA's tested version pins — 309 of them in nvcr.io/nvidia/pytorch:25.03-py3.
# Exactly three conflict with this skill's dependency set, and each makes `pip install` fail
# outright with "ResolutionImpossible":
#
#   regex               NGC pins 2024.11.6  but transformers 5.14.1 needs >=2025.10.22
#   safetensors         NGC pins 0.5.3      but transformers 5.14.1 needs >=0.8.0
#   lightning-utilities NGC pins 0.14.0     but torchmetrics 1.9.0  needs >=0.15.3
#
# Clearing PIP_CONSTRAINT entirely would work but discards all 309 of NVIDIA's tested pins.
# This drops only the conflicting lines and leaves the rest of the constraint file in force.
# Re-derive the list after any dependency bump — see the conflict-detection snippet in
# references/finetune-and-report.md.
#
# USAGE:
#     . "$(dirname "${BASH_SOURCE[0]}")/pip_env.sh"
#     ngc_pip_install transformers==5.14.1 torchmetrics==1.9.0 ...

# Packages whose NGC pin must be relaxed for this skill's dependency set.
NGC_PIP_RELAX="${NGC_PIP_RELAX:-regex safetensors lightning-utilities}"

ngc_pip_install() {
    local constraint="${PIP_CONSTRAINT:-}"

    if [ -n "$constraint" ] && [ -r "$constraint" ]; then
        local relaxed pattern
        relaxed="$(mktemp)" || return 1
        pattern="^($(echo "$NGC_PIP_RELAX" | tr ' ' '|'))[=<>!~[:space:]]"
        grep -Ev "$pattern" "$constraint" > "$relaxed" || true
        echo "[ngc-pip] relaxed NGC constraints for: $NGC_PIP_RELAX"
        PIP_CONSTRAINT="$relaxed" pip install --no-cache-dir -q "$@"
        local rc=$?
        rm -f "$relaxed"
        return $rc
    fi

    pip install --no-cache-dir -q "$@"
}
