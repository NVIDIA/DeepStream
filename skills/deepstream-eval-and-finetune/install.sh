#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied.

# deepstream-eval-and-finetune — Install script
# Installs the skill into a target project AND provisions its tao-skill-bank dependency
# (the fine-tune/automl lanes), so the target has EVERYTHING it needs. Mirrors the
# deepstream-import-vision-model install.sh convention.
#
# It does two things:
#   1. Copies the skill into <target>/.claude/skills/deepstream-eval-and-finetune/
#      (plus <target>/.codex/skills/ and, unless --no-cursor, <target>/.cursor/skills/).
#   2. Installs the tao-skill-bank plugin (same sequence as the Windows install.ps1 twin):
#        marketplace add -> plugin install [--scope <scope>] -> list
#      Provisioning runs entirely through the `claude` CLI, which owns its own configuration.
#      This installer never reads or writes the agent's config files directly. If the CLI is
#      not on PATH, the skill is still installed and the exact plugin commands are printed for
#      the user to run.
#
# Always installs the plugin from THIS link (never a mirror):
#     https://github.com/NVIDIA-TAO/tao-skill-bank
#
# Usage (mirror of the Windows install.ps1):
#   bash install.sh --target <project-path> [--scope project|user] [--dry-run] [--no-cursor] [--no-plugin]
set -euo pipefail

SKILL_DIR="$(cd "$(dirname "$0")" && pwd)"
SKILL_NAME="deepstream-eval-and-finetune"
TAO_SKILL_BANK_URL="https://github.com/NVIDIA-TAO/tao-skill-bank"
PLUGIN_REF="tao-skills@tao-skill-bank"

TARGET=""
SCOPE="project"
DRY_RUN=false
NO_CURSOR=false
NO_PLUGIN=false

usage() {
    local rc="${1:-0}"
    cat <<EOF
Usage: $0 --target <project-path> [--scope project|user] [--dry-run] [--no-cursor] [--no-plugin]

  --target <path>          Project directory to install into (required)
  --scope <project|user>   Plugin scope passed through to 'claude plugin install'
                           (default: project). 'user' = enabled globally for every project.
  --dry-run                Show what would be done without making any changes
  --no-cursor              Skip Cursor (.cursor/skills/) installation
  --no-plugin              Skip installing/declaring the tao-skill-bank plugin
  -h, --help               Show this help

Example:
  bash install.sh --target ~/work/my-deepstream-project
EOF
    exit "$rc"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --target)
            if [[ $# -lt 2 || -z "${2:-}" || "$2" == -* ]]; then
                echo "Error: --target requires a path argument" >&2; usage 1
            fi
            TARGET="$2"; shift 2 ;;
        --scope)
            if [[ "${2:-}" != "project" && "${2:-}" != "user" ]]; then
                echo "Error: --scope must be 'project' or 'user'" >&2; usage 1
            fi
            SCOPE="$2"; shift 2 ;;
        --dry-run)   DRY_RUN=true; shift ;;
        --no-cursor) NO_CURSOR=true; shift ;;
        --no-plugin) NO_PLUGIN=true; shift ;;
        -h|--help)   usage 0 ;;
        *)           echo "Unknown option: $1" >&2; usage 1 ;;
    esac
done

# --- TARGET validation (hardened) ----------------------------------------------
[[ -z "$TARGET" ]] && { echo "Error: --target is required" >&2; usage 1; }
case "$TARGET" in
    ""|"/"|*..*) echo "Error: invalid --target value: $TARGET" >&2; exit 1 ;;
esac
[[ -d "$TARGET" ]] || { echo "Error: target directory not found: $TARGET" >&2; exit 1; }
TARGET="$(cd "$TARGET" && pwd -P)"
if [[ "$TARGET" == "/" ]] || [[ ${#TARGET} -lt 3 ]]; then
    echo "Error: refusing to install into '$TARGET' (path too short / is root)" >&2; exit 1
fi
# -------------------------------------------------------------------------------

safe_rm_under_target() {
    local path="$1" resolved
    resolved="$(cd "$(dirname "$path")" 2>/dev/null && pwd -P)/$(basename "$path")" || return 1
    case "$resolved" in
        "$TARGET"/*) ;;
        *) echo "  Refusing to remove $resolved (not under $TARGET)" >&2; return 1 ;;
    esac
    if $DRY_RUN; then
        echo "  [dry-run] would remove $resolved"
        return 0
    fi
    echo "  Removing: $resolved"
    # Depth-first delete of contents, then the directory itself. Avoids a recursive force
    # delete so a mis-resolved path cannot widen the blast radius past the guard above.
    find "$resolved" -mindepth 1 -delete 2>/dev/null || true
    rmdir "$resolved" 2>/dev/null || rm -f "$resolved" 2>/dev/null || true
}

# Copy the whole self-contained skill (SKILL.md, references, scripts, examples, app,
# setup.sh) into a skills dir, minus machine-local build artifacts.
install_skill_to_dir() {
    local skills_dir="$1" dest="$1/$SKILL_NAME"
    if [[ -d "$dest" ]]; then
        local dest_real
        dest_real="$(cd "$dest" && pwd -P)"
        if [[ "$dest_real" == "$SKILL_DIR" ]]; then
            echo "  Already installed at $dest; source and destination are identical"
            return
        fi
        safe_rm_under_target "$dest"
    fi
    if $DRY_RUN; then
        echo "  [dry-run] cp -r $SKILL_DIR -> $dest  (minus __pycache__, ds_image_eval binary)"
        return
    fi
    mkdir -p "$skills_dir"
    cp -r "$SKILL_DIR" "$dest"
    # strip machine-local artifacts so the target rebuilds them fresh
    find "$dest" -type d -name '__pycache__' -prune -exec rm -r {} + 2>/dev/null || true
    find "$dest/scripts" -maxdepth 1 -name ds_image_eval -delete 2>/dev/null || true
    echo "  Installed: $dest"
}

# Provision the tao-skill-bank plugin into the target.
#
# Provisioning goes exclusively through the `claude` CLI, which owns its own configuration.
# This installer deliberately does NOT read or write the agent's config files itself: doing so
# means parsing and rewriting another tool's private state, and it is what the NVSkills-Eval
# AS1 "agent config directory access" check exists to prevent. When the CLI is unavailable we
# print the exact commands instead of editing anything behind the user's back.
install_tao_plugin() {
    if ! command -v claude >/dev/null 2>&1; then
        echo "  'claude' CLI not found — skipping tao-skill-bank provisioning."
        echo "  The skill itself is installed. To enable the plugin, run these once:"
        echo "      cd $TARGET"
        echo "      claude plugin marketplace add $TAO_SKILL_BANK_URL"
        echo "      claude plugin install $PLUGIN_REF --scope $SCOPE"
        return 0
    fi

    if $DRY_RUN; then
        echo "  [dry-run] (cd $TARGET && claude plugin marketplace add $TAO_SKILL_BANK_URL"
        echo "  [dry-run]              && claude plugin install $PLUGIN_REF --scope $SCOPE && claude plugin list)"
        return 0
    fi

    ( cd "$TARGET" \
      && claude plugin marketplace add "$TAO_SKILL_BANK_URL" 2>&1 | sed 's/^/    /' \
      && claude plugin install "$PLUGIN_REF" --scope "$SCOPE" 2>&1 | sed 's/^/    /' \
      && claude plugin list 2>&1 | sed 's/^/    /' ) \
      || {
        echo "    (claude plugin step returned non-zero — it may already be installed)"
        echo "    Verify with: claude plugin list"
      }
}

echo "=== deepstream-eval-and-finetune Install ==="
echo "Skill dir: $SKILL_DIR"
echo "Target:    $TARGET"
echo "Scope:     $SCOPE"
echo "Cursor:    $($NO_CURSOR && echo 'disabled (--no-cursor)' || echo enabled)"
echo "Plugin:    $($NO_PLUGIN && echo 'disabled (--no-plugin)' || echo "tao-skill-bank ($PLUGIN_REF)")"
echo ""

echo "Claude Code skill -> $TARGET/.claude/skills/$SKILL_NAME/"
install_skill_to_dir "$TARGET/.claude/skills"

echo ""
echo "Codex skill -> $TARGET/.codex/skills/$SKILL_NAME/"
install_skill_to_dir "$TARGET/.codex/skills"

if ! $NO_CURSOR; then
    echo ""
    echo "Cursor skill -> $TARGET/.cursor/skills/$SKILL_NAME/"
    install_skill_to_dir "$TARGET/.cursor/skills"
fi

if ! $NO_PLUGIN; then
    echo ""
    echo "tao-skill-bank plugin (fine-tune/automl lanes) -> scope=$SCOPE"
    install_tao_plugin
fi

echo ""
echo "=== Done ==="
# Step 1 depends on whether the plugin was actually provisioned: with --no-plugin there is no
# tao-skill-bank to auto-load, so promising it would be wrong post-install guidance.
if $NO_PLUGIN; then
    PLUGIN_STEP="  1. Start a fresh Claude Code session there and trust the folder.
     (tao-skill-bank was NOT installed: --no-plugin. Fine-tune/automl lanes are unavailable
      until you rerun without --no-plugin, or run scripts/install_tao_skills.sh.)"
else
    PLUGIN_STEP="  1. Start a fresh Claude Code session there and trust the folder — the tao-skill-bank
     plugin loads automatically (enabledPlugins)."
fi
cat <<EOF

Next steps in the target project ($TARGET):
$PLUGIN_STEP
  2. One-time bootstrap (build venv + C app), from the target working root:
       docker run --rm -it --gpus all --shm-size=16g -v "\$PWD":/work -w /work \\
         --entrypoint bash nvcr.io/nvidia/deepstream:9.1-triton-multiarch \\
         .claude/skills/$SKILL_NAME/setup.sh
  3. Verify:  bash .claude/skills/$SKILL_NAME/scripts/preflight.sh   (expect PASS)
              bash .claude/skills/$SKILL_NAME/scripts/check_tao_skills.sh   (expect PASS)
  4. Test:    bash .claude/skills/$SKILL_NAME/examples/rtdetr-aerial-sheep/run.sh
              (or invoke the skill in Claude, or launch app/launch.sh)
  See .claude/skills/$SKILL_NAME/NEW_MACHINE_SETUP.md for the full runbook.
EOF
