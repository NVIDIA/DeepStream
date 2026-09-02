# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# launch.ps1 - Windows (PowerShell) twin of launch.sh. Boots the DeepStream Eval & Fine-tune Web UI
# INSIDE the DeepStream container (uvicorn on 0.0.0.0:PORT), publishes the port to Windows localhost,
# and bind-mounts the working root. Durable results (engines, metrics, reports, history) land under
# the Windows root; high-frequency dataset/Arrow inputs use a persistent Linux Docker volume.
#
# Run from the working root (where models/, runs/, reports/, build/ live):
#   .\.claude\skills\deepstream-eval-and-finetune\app\launch.ps1
# Then open http://localhost:8078.
#
# Env knobs (mirror launch.sh):
#   $env:PORT           published port (default 8078)
#   $env:IMAGE          DeepStream image (default nvcr.io/nvidia/deepstream:9.1-triton-multiarch)
#   $env:DATASETS_ROOT  one or more host roots (semicolon-separated when paths contain spaces);
#                       mapped read-only under /datasets/windows-N and translated by the UI
#   $env:DATASET_MOUNT  a single explicit mount, "src" or "src:dst"
#   $env:DS_DATA_CACHE  automatic Linux dataset cache: auto (default) or off
#   $env:DS_DATA_VOLUME override the project-scoped Docker volume name
#   $env:DS_DATA_CACHE_REFRESH=1 refresh a staged local source on its next ingestion
# GPU needs Docker Desktop's WSL2 backend + an NVIDIA driver. If PowerShell blocks the script:
#   powershell -ExecutionPolicy Bypass -File .\.claude\skills\deepstream-eval-and-finetune\app\launch.ps1
$ErrorActionPreference = 'Stop'

$SK    = ".claude/skills/deepstream-eval-and-finetune"
$Port  = if ($env:PORT)  { $env:PORT }  else { "8078" }
$Image = if ($env:IMAGE) { $env:IMAGE } else { "nvcr.io/nvidia/deepstream:9.1-triton-multiarch" }
$CacheMode = if ($env:DS_DATA_CACHE) { $env:DS_DATA_CACHE.ToLowerInvariant() } else { "auto" }

if (-not (Test-Path -LiteralPath $SK -PathType Container)) {
    throw "Run from the working root - '$SK' not found under the current directory ($($PWD.Path))."
}

# Datasets outside the project must be bind-mounted, or the container (which only sees /work) can't
# read them. DATASETS_ROOT entries map to Linux paths and the UI translates Explorer paths.
$mounts = @()
$datasetMap = [ordered]@{}
if ($env:DATASETS_ROOT) {
    $roots = if ($env:DATASETS_ROOT.Contains(';')) {
        $env:DATASETS_ROOT -split ';'
    } else {
        $env:DATASETS_ROOT -split '\s+'
    }
    $mountIndex = 0
    foreach ($root in $roots) {
        $root = $root.Trim()
        if ([string]::IsNullOrWhiteSpace($root)) { continue }
        if (Test-Path -LiteralPath $root -PathType Container) {
            $hostRoot = [IO.Path]::GetFullPath($root).TrimEnd('\')
            $containerRoot = "/datasets/windows-$mountIndex"
            $mounts += @('-v', "${hostRoot}:${containerRoot}:ro")
            $datasetMap[$hostRoot] = $containerRoot
            Write-Host "Dataset root: $hostRoot -> $containerRoot"
            $mountIndex++
        } else { Write-Warning "DATASETS_ROOT entry '$root' is not a directory on the host - skipping" }
    }
}
if ($env:DATASET_MOUNT) {
    if ($env:DATASET_MOUNT -like '*:*') { $mounts += @('-v', "$($env:DATASET_MOUNT):ro") }
    else { $mounts += @('-v', "$($env:DATASET_MOUNT):$($env:DATASET_MOUNT):ro") }
}

# Docker Desktop stores named volumes in its Linux VM. Mount only /work/data from the volume:
# checkpoints, engines, logs, reports, and UI history continue to persist on the Windows project.
$cacheArgs = @()
if ($CacheMode -ne 'off') {
    if ($env:DS_DATA_VOLUME) {
        $dataVolume = $env:DS_DATA_VOLUME
    } else {
        $canonical = [IO.Path]::GetFullPath($PWD.Path).ToLowerInvariant()
        $sha = [Security.Cryptography.SHA256]::Create()
        try {
            $digest = $sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($canonical))
        } finally {
            $sha.Dispose()
        }
        $projectHash = -join ($digest | Select-Object -First 6 | ForEach-Object { $_.ToString('x2') })
        $dataVolume = "ds-eval-data-$projectHash"
    }
    & docker volume create `
        --label 'com.nvidia.deepstream.eval-cache=true' `
        --label "com.nvidia.deepstream.project=$($PWD.Path)" `
        $dataVolume | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Could not create dataset cache volume '$dataVolume'." }
    $env:DS_DATA_VOLUME = $dataVolume
    $cacheArgs += @('-v', "${dataVolume}:/work/data",
                    '-e', 'DS_DATA_CACHE_ROOT=/work/data',
                    '-e', "DS_DATA_VOLUME=$dataVolume")
    if ($env:DS_DATA_CACHE_REFRESH -match '^(1|true|yes|on)$') {
        $cacheArgs += @('-e', 'DS_DATA_CACHE_REFRESH=1')
    }
    Write-Host "Windows dataset cache: $dataVolume -> /work/data"
}
if ($datasetMap.Count -gt 0) {
    $mounts += @('-e', "DS_WINDOWS_DATASET_MAP=$($datasetMap | ConvertTo-Json -Compress)")
}

# in-container command: bootstrap the env on first run (venv + deps + ds_image_eval), then start
# uvicorn bound to all interfaces. --app-dir points at app/ so the module is the bare `server:app`.
$inner = @"
set -e
if ! build/.venv_train/bin/python -c 'import torch' 2>/dev/null; then
  bash $SK/setup.sh
fi
if ! build/.venv_train/bin/python -c 'import fastapi,uvicorn' 2>/dev/null; then
  bash $SK/app/setup.sh
fi
echo '== DeepStream Eval & Fine-tune UI -> http://localhost:$Port =='
DS_EVAL_ROOT=/work build/.venv_train/bin/python -m uvicorn server:app \
  --app-dir /work/$SK/app --host 0.0.0.0 --port $Port
"@

$dockerArgs = @(
    'run', '--rm', '-it', '--gpus', 'all', '--shm-size=16g',
    '-v', "${PWD}:/work", '-w', '/work'
) + $cacheArgs + $mounts + @(
    '-p', "127.0.0.1:${Port}:${Port}",
    '-e', 'HF_HOME=/work/build/hf_cache',
    '-e', 'PYTHONDONTWRITEBYTECODE=1',
    '--entrypoint', 'bash', $Image, '-c', $inner
)

Write-Host "Booting UI in the DeepStream container -> http://localhost:$Port   (Ctrl+C to stop)"
& docker @dockerArgs
