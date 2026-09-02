#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved. Apache-2.0.
#
# Shared self-snapshot + re-exec guard for the long-running NGC entrypoints
# (run_ngc_finetune.sh, run_ngc_automl.sh).
#
# WHY: bash reads a script incrementally from disk rather than loading it whole. Editing a script
# while a container is still executing it shifts the byte offsets of the not-yet-read lines and
# corrupts the in-flight run — this bit us once, an AutoML run died with `line NN: s: command not
# found`. Snapshotting the entrypoint to an immutable temp copy and re-exec'ing from that copy
# means later edits to the canonical file can never affect a job already running.
#
# USAGE — source this as the first thing after `set -euo pipefail`, passing a short unique prefix
# and the caller's own path plus arguments:
#
#     . "${NGC_FT_HERE:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}/snapshot_reexec.sh"
#     ngc_snapshot_reexec NGC_FT "${BASH_SOURCE[0]}" "$@"
#     HERE="${NGC_FT_HERE}"
#
# The <PREFIX>_HERE guard on the source line matters: after the re-exec ${BASH_SOURCE[0]} is the
# temp snapshot in /tmp, which would no longer resolve this helper.
#
# The function exports <PREFIX>_HERE (the real script directory, preserved across the re-exec so
# siblings like train_ngc.py / ui_register.py still resolve) and <PREFIX>_SNAPSHOT (the re-entry
# guard). It returns normally when already running from the snapshot, so callers can source it
# unconditionally.

ngc_snapshot_reexec() {
    local prefix="${1:?prefix}" self="${2:?script path}"
    shift 2

    local guard_var="${prefix}_SNAPSHOT" here_var="${prefix}_HERE"

    # Already re-exec'd from the immutable copy — just make sure the dir is exported and return.
    if [ -n "${!guard_var:-}" ]; then
        return 0
    fi

    export "${here_var}=$(cd "$(dirname "$self")" && pwd)"
    local snap
    snap="$(mktemp)" || return 0
    cp "$self" "$snap" || return 0
    export "${guard_var}=1"
    exec bash "$snap" "$@"
}
