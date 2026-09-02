#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
#
# Manual fallback installer for the tao-skill-bank plugin.
#
# NORMALLY YOU DON'T NEED THIS: the skill's install.sh already installs the tao-skill-bank
# plugin and records it in the project's .claude/settings.json, so it loads at session start.
# Run this only if `check_tao_skills.sh` reports INSTALL-NEEDED (e.g. the skill was copied in
# without running install.sh).
#
# Installs at PROJECT scope, then re-runs the detector. Idempotent.
# Always installs from THIS link (never a mirror): https://github.com/NVIDIA-TAO/tao-skill-bank
#
# Usage: bash scripts/install_tao_skills.sh
set -u

TAO_SKILL_BANK_URL="https://github.com/NVIDIA-TAO/tao-skill-bank"
PLUGIN_REF="tao-skills@tao-skill-bank"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[install-tao] source of truth: $TAO_SKILL_BANK_URL"

# The install runs in the HOST Claude session (not the DeepStream container). If the CLI is
# missing, print the manual /plugin slash commands and stop.
if ! command -v claude >/dev/null 2>&1; then
  cat <<EOF
[install-tao] The 'claude' CLI isn't available here. Run these Claude Code slash commands,
[install-tao] then re-run the check:

    /plugin marketplace add $TAO_SKILL_BANK_URL
    /plugin install $PLUGIN_REF

EOF
  exit 1
fi

echo "[install-tao] 1/2 add marketplace (project scope)"
claude plugin marketplace add "$TAO_SKILL_BANK_URL" --scope project 2>&1 \
  | sed 's/^/  /' || echo "  (marketplace add returned non-zero — likely already added; continuing)"

echo "[install-tao] 2/2 install plugin (project scope): $PLUGIN_REF"
claude plugin install "$PLUGIN_REF" --scope project 2>&1 \
  | sed 's/^/  /' || echo "  (install returned non-zero — likely already installed; continuing)"

echo
echo "[install-tao] re-running detector…"
bash "$SCRIPT_DIR/check_tao_skills.sh"
rc=$?

if [ $rc -ne 0 ]; then
  cat <<EOF

[install-tao] The plugin is installed on disk but its skills aren't registered in THIS
[install-tao] session yet (plugin skills register at session start). Start a new session —
[install-tao] the enabled plugin loads automatically — then re-run scripts/check_tao_skills.sh.
EOF
fi
exit $rc
