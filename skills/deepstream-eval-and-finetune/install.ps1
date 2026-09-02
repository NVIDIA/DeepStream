# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# deepstream-eval-and-finetune - Windows (PowerShell) installer.
# The native-Windows twin of install.sh, with the IDENTICAL sequence of operations:
#   1. Copy the skill into Claude, Codex, and (unless -NoCursor) Cursor skill directories
#   2. Install the tao-skill-bank plugin (fine-tune/automl lanes) via the cross-platform `claude` CLI:
#        marketplace add -> plugin install [--scope <scope>] -> list
#      The CLI owns its own configuration; this installer never reads or writes agent config files.
#   3. Print the next steps (bootstrap + preflight, run THROUGH Docker)
# Neither the copy nor the plugin install can run in a container (they write host skill directories);
# everything ELSE runs through Docker (see references/windows.md). PowerShell 5.1+ compatible.
#
# Usage (mirror of: bash install.sh --target <p> [--scope s] [--no-cursor] [--no-plugin] [--dry-run]):
#   .\install.ps1 -Target C:\path\to\project [-Scope project|user] [-NoCursor] [-NoPlugin] [-DryRun]
#   # default -Target = current dir; default -Scope = project (same as install.sh).
param(
    [string]$Target = (Get-Location).Path,
    [ValidateSet('project','user')][string]$Scope = 'project',
    [switch]$NoCursor,
    [switch]$NoPlugin,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'

$SkillDir       = $PSScriptRoot
$SkillName      = 'deepstream-eval-and-finetune'
$MarketplaceUrl = 'https://github.com/NVIDIA-TAO/tao-skill-bank'
$PluginRef      = 'tao-skills@tao-skill-bank'

if (-not (Test-Path -LiteralPath $Target -PathType Container)) { throw "Target directory not found: $Target" }
$Target = (Resolve-Path -LiteralPath $Target).Path

# Copy the whole self-contained skill into a skills dir, minus machine-local build artifacts.
function Install-SkillTo([string]$SkillsDir) {
    $dest = Join-Path $SkillsDir $SkillName
    if ($DryRun) { Write-Host "  [dry-run] copy $SkillDir -> $dest  (minus __pycache__, ds_image_eval binary)"; return }
    if (Test-Path -LiteralPath $dest) {
        $sourceResolved = (Resolve-Path -LiteralPath $SkillDir).Path
        $destResolved = (Resolve-Path -LiteralPath $dest).Path
        if ([StringComparer]::OrdinalIgnoreCase.Equals($sourceResolved, $destResolved)) {
            Write-Host "  Already installed at $dest; source and destination are identical"
            return
        }
        Remove-Item -LiteralPath $dest -Recurse -Force
    }
    # Create the destination, then copy the source's CHILDREN into it one by one.
    # `Copy-Item -LiteralPath <dir> -Destination <dir> -Recurse` is not portable: whether it copies
    # the folder's contents or the folder itself depends on whether the destination already exists,
    # and on Windows PowerShell 5.1 it has been observed to create the directory tree without
    # copying any leaf files -- producing an empty skill dir and an unloadable skill. Enumerating
    # children and copying each explicitly is well-defined on every version.
    New-Item -ItemType Directory -Force -Path $dest | Out-Null
    Get-ChildItem -LiteralPath $SkillDir -Force | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination $dest -Recurse -Force
    }
    Get-ChildItem -LiteralPath $dest -Recurse -Directory -Filter '__pycache__' -ErrorAction SilentlyContinue |
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath (Join-Path $dest 'scripts\ds_image_eval') -Force -ErrorAction SilentlyContinue
    # Fail loudly rather than leaving a skill dir the agent runtime will silently refuse to load.
    if (-not (Test-Path -LiteralPath (Join-Path $dest 'SKILL.md'))) {
        throw "Install failed: SKILL.md is missing from $dest -- the skill would not load. Please report this with your `$PSVersionTable.PSVersion."
    }
    Write-Host "  Installed: $dest"
}

# Provision the tao-skill-bank plugin (same steps as install.sh's install_tao_plugin).
#
# Provisioning goes exclusively through the `claude` CLI, which owns its own configuration.
# This installer deliberately does NOT read or write the agent's config files itself: doing so
# means parsing and rewriting another tool's private state, and it is what the NVSkills-Eval
# AS1 "agent config directory access" check exists to prevent. When the CLI is unavailable we
# print the exact commands instead of editing anything behind the user's back.
function Install-TaoPlugin {
    $claude = Get-Command claude -ErrorAction SilentlyContinue
    if (-not $claude) {
        Write-Host "  'claude' CLI not found on PATH - skipping tao-skill-bank provisioning."
        Write-Host "  The skill itself is installed. To enable the plugin, run these once:"
        Write-Host "      cd $Target"
        Write-Host "      claude plugin marketplace add $MarketplaceUrl"
        Write-Host "      claude plugin install $PluginRef --scope $Scope"
        return
    }
    if ($DryRun) {
        Write-Host "  [dry-run] claude plugin marketplace add $MarketplaceUrl"
        Write-Host "  [dry-run] claude plugin install $PluginRef --scope $Scope"
        return
    }
    try {
        & claude plugin marketplace add $MarketplaceUrl 2>&1 | ForEach-Object { "    $_" }
        & claude plugin install $PluginRef --scope $Scope 2>&1 | ForEach-Object { "    $_" }
        & claude plugin list                               2>&1 | ForEach-Object { "    $_" }
    } catch {
        Write-Host "    (claude plugin step returned non-zero - it may already be installed)"
        Write-Host "    Verify with: claude plugin list"
    }
}

Write-Host "=== deepstream-eval-and-finetune Install (Windows) ==="
Write-Host "Skill dir: $SkillDir"
Write-Host "Target:    $Target"
Write-Host "Scope:     $Scope"
Write-Host "Cursor:    $(if ($NoCursor) { 'disabled (-NoCursor)' } else { 'enabled' })"
Write-Host "Plugin:    $(if ($NoPlugin) { 'disabled (-NoPlugin)' } else { "tao-skill-bank ($PluginRef)" })"
Write-Host ""

Write-Host "Claude Code skill -> $Target\.claude\skills\$SkillName\"
Install-SkillTo (Join-Path $Target '.claude\skills')

Write-Host ""
Write-Host "Codex skill -> $Target\.codex\skills\$SkillName\"
Install-SkillTo (Join-Path $Target '.codex\skills')

if (-not $NoCursor) {
    Write-Host ""
    Write-Host "Cursor skill -> $Target\.cursor\skills\$SkillName\"
    Install-SkillTo (Join-Path $Target '.cursor\skills')
}

if (-not $NoPlugin) {
    Write-Host ""
    Write-Host "tao-skill-bank plugin (fine-tune/automl lanes) -> scope=$Scope"
    Install-TaoPlugin
}

Write-Host ""
Write-Host "=== Done ==="
# Step 1 depends on whether the plugin was actually provisioned: with -NoPlugin there is no
# tao-skill-bank to auto-load, so promising it would be wrong post-install guidance.
$pluginStep = if ($NoPlugin) {
    "  1. Start Claude Code there and trust the folder.`n     (tao-skill-bank was NOT installed: -NoPlugin. Fine-tune/automl lanes are unavailable`n      until you rerun without -NoPlugin, or run scripts/install_tao_skills.sh.)"
} else {
    "  1. Start Claude Code there and trust the folder - the tao-skill-bank plugin loads automatically.`n     Verify:  claude plugin list"
}
Write-Host @"

Next steps in the target project ($Target):
$pluginStep
  2. One-time bootstrap (build venv + C app) IN Docker, from the project root:
       docker run --rm -it --gpus all --shm-size=16g -v "`${PWD}:/work" -w /work ``
         --entrypoint bash nvcr.io/nvidia/deepstream:9.1-triton-multiarch ``
         .claude/skills/$SkillName/setup.sh
  3. Verify (through Docker):
       docker run --rm --gpus all -v "`${PWD}:/work" -w /work ``
         --entrypoint bash nvcr.io/nvidia/deepstream:9.1-triton-multiarch ``
         .claude/skills/$SkillName/scripts/preflight.sh          # expect PASS
  See .claude/skills/$SkillName/references/windows.md for the full Windows runbook.
"@
