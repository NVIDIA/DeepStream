#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# server_manager.sh — start, stop, restart, and check the Inference Builder
#                     MCP server (HTTP/SSE transport).
#
# Usage:
#   ./mcp/server_manager.sh <command> [options]
#
# Commands:
#   start    Start the MCP server in the background
#   stop     Stop a running server
#   restart  Stop (if running) then start
#   status   Print whether the server is running and its PID
#   logs     Tail the server log (Ctrl-C to exit)
#
# Options (start / restart):
#   --host HOST          Network interface to bind  (default: 0.0.0.0)
#   --port PORT          TCP port to listen on       (default: 8000)
#   --api-key TOKEN      Bearer token for auth       (sets MCP_API_KEY in the server env)
#   --workspace-root DIR Per-client workspace base   (overrides MCP_WORKSPACE_ROOT)
#   --model-root DIR     Shared model download root  (overrides MCP_MODEL_ROOT)
#
# Environment variables (used when corresponding flag is not set):
#   MCP_API_KEY          Bearer token for authentication
#   MCP_WORKSPACE_ROOT   Base directory for per-client workspaces
#                        Default: /tmp/inference-builder-workspaces
#   MCP_MODEL_ROOT       Shared directory for downloaded models
#                        Default: /tmp/inference-builder-models
#
# Examples:
#   ./mcp/server_manager.sh start
#   ./mcp/server_manager.sh start --port 8888 --api-key MY_SECRET
#   MCP_API_KEY=MY_SECRET ./mcp/server_manager.sh start --port 8888
#   ./mcp/server_manager.sh status
#   ./mcp/server_manager.sh logs
#   ./mcp/server_manager.sh stop
#   ./mcp/server_manager.sh restart --port 8888

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths — all relative to the repository root so the script works regardless
# of where it is invoked from.
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_PYTHON="${REPO_ROOT}/.venv/bin/python"
MCP_SERVER="${SCRIPT_DIR}/mcp_server.py"
PID_FILE="${SCRIPT_DIR}/.mcp_server.pid"
LOG_FILE="${SCRIPT_DIR}/.mcp_server.log"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log()  { echo "[server_manager] $*"; }
die()  { echo "[server_manager] ERROR: $*" >&2; exit 1; }

python_bin() {
    if [[ -x "${VENV_PYTHON}" ]]; then
        echo "${VENV_PYTHON}"
    elif command -v python3 &>/dev/null; then
        echo "python3"
    else
        die "No Python interpreter found. Expected ${VENV_PYTHON} or python3 in PATH."
    fi
}

is_running() {
    [[ -f "${PID_FILE}" ]] || return 1
    local pid
    pid=$(<"${PID_FILE}")
    kill -0 "${pid}" 2>/dev/null
}

# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
cmd_start() {
    # ---- Parse start options -----------------------------------------------
    local host="0.0.0.0"
    local port="8000"
    local api_key="${MCP_API_KEY:-}"
    local workspace_root="${MCP_WORKSPACE_ROOT:-}"
    local model_root="${MCP_MODEL_ROOT:-}"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --host)           host="$2";           shift 2 ;;
            --port)           port="$2";           shift 2 ;;
            --api-key)        api_key="$2";        shift 2 ;;
            --workspace-root) workspace_root="$2"; shift 2 ;;
            --model-root)     model_root="$2";     shift 2 ;;
            *) die "Unknown option for start: $1" ;;
        esac
    done

    if is_running; then
        local pid
        pid=$(<"${PID_FILE}")
        log "Server is already running (PID ${pid}). Use 'restart' to apply new options."
        exit 0
    fi

    # ---- Build command -----------------------------------------------------
    local py
    py=$(python_bin)
    local cmd=("${py}" "${MCP_SERVER}" --transport sse --host "${host}" --port "${port}")
    [[ -n "${workspace_root}" ]] && cmd+=(--workspace-root "${workspace_root}")
    [[ -n "${model_root}"     ]] && cmd+=(--model-root     "${model_root}")

    # ---- Start in background -----------------------------------------------
    log "Starting MCP server..."
    log "Command : ${cmd[*]}"
    log "Log file: ${LOG_FILE}"

    # Pass the API key via the environment so it does not appear in the
    # process list (ps aux / /proc/<pid>/cmdline).
    MCP_API_KEY="${api_key}" nohup "${cmd[@]}" >> "${LOG_FILE}" 2>&1 &
    local pid=$!
    echo "${pid}" > "${PID_FILE}"

    # Give it a moment and confirm it is still alive
    sleep 2
    if ! is_running; then
        rm -f "${PID_FILE}"
        die "Server failed to start. Check the log: ${LOG_FILE}"
    fi

    log "Server started (PID ${pid})"
    log "Endpoints:"
    log "  Streamable HTTP : http://${host}:${port}/mcp   (Claude Code)"
    log "  Legacy SSE      : http://${host}:${port}/sse   (older clients)"
    [[ -n "${api_key}" ]] && log "  Auth            : Bearer token required" \
                          || log "  Auth            : none (open access)"
}

cmd_stop() {
    if ! is_running; then
        log "Server is not running."
        return 0
    fi

    local pid
    pid=$(<"${PID_FILE}")
    log "Stopping server (PID ${pid})..."
    kill "${pid}"

    # Wait up to 10 s for graceful shutdown
    local waited=0
    while kill -0 "${pid}" 2>/dev/null; do
        sleep 1
        (( ++waited ))
        if (( waited >= 10 )); then
            log "Server did not stop gracefully; sending SIGKILL..."
            kill -9 "${pid}" 2>/dev/null || true
            break
        fi
    done

    rm -f "${PID_FILE}"
    log "Server stopped."
}

cmd_restart() {
    cmd_stop
    # Pass remaining arguments straight to start
    cmd_start "$@"
}

cmd_status() {
    if is_running; then
        local pid
        pid=$(<"${PID_FILE}")
        log "Server is running (PID ${pid})"
        log "Log file: ${LOG_FILE}"
    else
        log "Server is not running."
        [[ -f "${PID_FILE}" ]] && { rm -f "${PID_FILE}"; log "(Removed stale PID file)"; }
        return 1
    fi
}

cmd_logs() {
    [[ -f "${LOG_FILE}" ]] || die "No log file found at ${LOG_FILE}"
    tail -f "${LOG_FILE}"
}

usage() {
    sed -n '2,/^set -/p' "${BASH_SOURCE[0]}" | grep -E '^#' | sed 's/^# \{0,1\}//'
    exit 0
}

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
[[ $# -eq 0 ]] && usage

COMMAND="$1"; shift

case "${COMMAND}" in
    start)   cmd_start   "$@" ;;
    stop)    cmd_stop         ;;
    restart) cmd_restart "$@" ;;
    status)  cmd_status       ;;
    logs)    cmd_logs         ;;
    -h|--help|help) usage     ;;
    *) die "Unknown command: '${COMMAND}'. Use: start | stop | restart | status | logs" ;;
esac
