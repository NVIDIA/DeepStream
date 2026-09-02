#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
#
# Stage 0b bootstrap check for deepstream-eval-and-finetune: verify the tao-skill-bank
# plugin is installed BEFORE the loop reaches Stage 3, so a missing dependency skill surfaces
# up front (not mid-run). Once the plugin is installed, EVERY tao skill is invocable by
# name; this checks three representative skills the orchestrator actually uses.
#
# This is a DETECTOR ONLY — it never installs. Installation is done by install.sh (or the
# manual install_tao_skills.sh); this script tells you whether that step is still needed.
# Idempotent, no side effects.
#
#   PASS          — the plugin is installed; the launch/finetune/automl skills resolve.
#   INSTALL-NEEDED — run install.sh / scripts/install_tao_skills.sh, then re-run this check.
#
# Usage: bash scripts/check_tao_skills.sh
set -u

# Always install the bank from THIS link (do not substitute a mirror).
TAO_SKILL_BANK_URL="https://github.com/NVIDIA-TAO/tao-skill-bank"
MARKETPLACE="tao-skill-bank"             # marketplace name (from the bank README)
PLUGIN_REF="tao-skills@tao-skill-bank"   # plugin@marketplace to install

# Representative bank skills the orchestrator uses. If the plugin is installed all of
# these resolve (and so does every other tao skill). Stage 3 routes through the launch
# skill, which targets the finetune skill; automl is the optional HPO lane.
LAUNCH_SKILL="tao-launch-workflow"
FINETUNE_SKILL="tao-finetune-huggingface-model"
AUTOML_SKILL="tao-run-automl"

fail=0
ok(){   echo "  [OK]      $1"; }
bad(){  echo "  [MISSING] $1"; fail=1; }

# Resolve the project root (the dir holding .claude/), so we find project-scoped
# skills + settings regardless of where the script is invoked from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
while [ "$PROJECT_ROOT" != "/" ] && [ ! -d "$PROJECT_ROOT/.claude" ]; do
  PROJECT_ROOT="$(dirname "$PROJECT_ROOT")"
done

# A skill is "available" if a SKILL.md for it exists in any supported agent skill location:
# project .claude/skills, user ~/.claude/skills (shallow), or an INSTALLED plugin's payload
# under ~/.claude/plugins/cache (deeper, versioned layout). We scan `cache/` (installed
# plugins), NOT `marketplaces/` (a clone of the whole repo), so PASS means resolvable.
skill_available(){
  local name="$1"
  local specs=(
    "$PROJECT_ROOT/.claude/skills:4"
    "$PROJECT_ROOT/.codex/skills:4"
    "$PROJECT_ROOT/.agents/skills:4"
    "$HOME/.claude/skills:4"
    "$HOME/.codex/skills:4"
    "$HOME/.claude/plugins/cache:9"
    "$HOME/.codex/plugins/cache:10"
  )
  local spec root depth
  for spec in "${specs[@]}"; do
    root="${spec%:*}"; depth="${spec##*:}"
    [ -d "$root" ] || continue
    if find "$root" -maxdepth "$depth" -type d -name "$name" 2>/dev/null \
         | while read -r d; do [ -f "$d/SKILL.md" ] && echo hit; done \
         | grep -q hit; then
      return 0
    fi
  done
  return 1
}

echo "[check-tao] source of truth: $TAO_SKILL_BANK_URL"

echo "[check-tao] 1/3 launch skill ($LAUNCH_SKILL, Stage-3 entry point)"
skill_available "$LAUNCH_SKILL" && ok "launch skill available" || bad "launch skill not installed"

echo "[check-tao] 2/3 finetune skill ($FINETUNE_SKILL)"
skill_available "$FINETUNE_SKILL" && ok "finetune skill available" || bad "finetune skill not installed"

echo "[check-tao] 3/3 automl skill ($AUTOML_SKILL)"
skill_available "$AUTOML_SKILL" && ok "automl skill available" || bad "automl skill not installed"

echo
if [ $fail -eq 0 ]; then
  echo "[check-tao] RESULT: PASS — tao-skill-bank plugin installed; all tao skills resolvable. Proceed to Stage 1"
else
  cat <<EOF
[check-tao] RESULT: INSTALL-NEEDED — install the tao-skill-bank plugin, then re-run this check.

  Auto-install (hands-free, via the claude CLI — installs the full bank):

    bash scripts/install_tao_skills.sh

  Manual fallback (run these Claude Code slash commands yourself):

    /plugin marketplace add $TAO_SKILL_BANK_URL
    /plugin install $PLUGIN_REF

  The marketplace + plugin are recorded in this project's .claude/settings.json so the
  install is persistent and project-scoped. If a just-installed skill isn't yet visible
  to the active agent runtime, start a new session so the enabled plugin loads, then re-run this check.
EOF
fi
exit $fail
