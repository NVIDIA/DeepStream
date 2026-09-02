<!--
Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License").
-->

# Running on Windows (and cross-platform)

The skill runs on **Windows with Docker Desktop**: every compute script runs **inside the Linux
container** via `docker run`, exactly as on Linux. The host only needs `docker` + an NVIDIA driver.

`install.ps1` installs the skill itself for **Claude Code, Codex, and Cursor** (`.claude\`, `.codex\`,
`.cursor\skills\`) — no agent CLI required for that step. The **tao-skill-bank plugin** (fine-tune /
automl lanes) is the part that needs an agent host, and support differs:

| Agent | Skill install | tao-skill-bank plugin |
|-------|---------------|-----------------------|
| **Claude Code** | `install.ps1` | installed by `install.ps1` via the cross-platform `claude` CLI |
| **Codex** | `install.ps1` | run the TAO Skill Bank **Codex** installer separately (different installer; also provides its `AGENTS.md` identity) |
| **Cursor** | `install.ps1` | **not supported** — Cursor can drive the bundled CLI/UI lanes but cannot invoke TAO plugin skills |

> **macOS:** every compute stage requires an NVIDIA GPU via `--gpus all`, which Docker Desktop on macOS
> cannot provide. macOS can run the host-side helpers and the **read-only UI viewer** against artifacts
> produced elsewhere, but cannot deploy, evaluate, or fine-tune. Use Linux or Windows+WSL2 for real runs.

Two host-side actions cannot run in a container — installing the skill + tao plugin, and launching the
UI container — so the skill ships exactly **two** PowerShell scripts for them: **`install.ps1`** (twin of
`install.sh`) and **`app/launch.ps1`** (twin of `app/launch.sh`). Every other script stays bash-only and
runs in the container; there are no `.ps1` duplicates of the compute scripts.

## Why this works
Every compute script (`setup.sh`, `examples/*/run.sh`, `scripts/*.py`, `scripts/ngc/*.sh`,
`scripts/Makefile`, and `preflight.sh` in container-mode) is executed **inside a Linux container**.
The only per-shell difference is the `docker run` **bind-mount token** for the working directory.

## Prerequisites (Windows)
1. **Docker Desktop** with the **WSL2 backend enabled** (Settings → General → *Use the WSL 2 based
   engine*). This is **required for GPU** — `--gpus all` on Windows works only through the WSL2 backend.
2. A recent **NVIDIA driver** with WSL/CUDA support (the standard Game-Ready/Studio or datacenter
   driver ≥ the CUDA-on-WSL minimum). No CUDA toolkit needed on the host — the container ships it.
3. In Docker Desktop → Settings → Resources → **File Sharing**, ensure the drive holding your working
   directory is shared, so `-v` bind mounts resolve.
4. Pull the images once:
   `docker pull nvcr.io/nvidia/deepstream:9.1-triton-multiarch` and (for fine-tune) `nvcr.io/nvidia/pytorch:25.03-py3`.
5. **Claude Code for Windows** (`irm https://claude.ai/install.ps1 | iex` in PowerShell) and **Git for
   Windows** (Claude Code uses Git Bash for its Bash tool; the marketplace add clones a git repo).

## Install on Windows (native PowerShell)
Run the bundled **`install.ps1`** — the exact twin of Linux's `install.sh`, with the **same sequence of
operations** (copy the skill → install the tao-skill-bank plugin via the `claude` CLI → list). From the
skill folder:
```powershell
# one command — installs into <project> at project scope (default), identical to `bash install.sh`
.\install.ps1 -Target C:\path\to\project

# ...or a global/user-scope plugin (enabled in EVERY project — lands in %USERPROFILE%\.claude\plugins):
.\install.ps1 -Target C:\path\to\project -Scope user
```
The flags mirror `install.sh` 1:1: `-Target`=`--target`, `-Scope project|user`=`--scope`,
`-NoCursor`=`--no-cursor`, `-NoPlugin`=`--no-plugin`, `-DryRun`=`--dry-run`.

It does the two **host-side** things (neither can run in a container): (1) copies the skill into
`<project>\.claude\skills\deepstream-eval-and-finetune\`, and (2) installs the tao-skill-bank plugin via
`claude plugin marketplace add` + `install --scope <scope>` — which lands the plugin in
**`%USERPROFILE%\.claude\plugins\cache\`** and enables it at the requested scope. The CLI owns that
configuration; the installer does not write it. If `claude` is not on PATH, the skill still installs
and the two plugin commands are printed for you to run once.

If PowerShell's execution policy blocks the script, run it once as:
`powershell -ExecutionPolicy Bypass -File .\install.ps1 -Target C:\path\to\project`.

**Manual equivalent** (same two steps by hand, if you'd rather not run the script):
```powershell
Copy-Item -Recurse -Force "<src>\...\deepstream-eval-and-finetune" "<project>\.claude\skills\deepstream-eval-and-finetune"
claude plugin marketplace add https://github.com/NVIDIA-TAO/tao-skill-bank
claude plugin install tao-skills@tao-skill-bank --scope user     # or --scope project
claude plugin list
```
Then **bootstrap + verify through Docker** (next sections), start Claude Code in `<project>`, trust the
folder, and invoke the skill.

## The one thing that differs per shell: the mount token
Use the token for your shell in every `docker run` (Claude Code fills this in automatically based on
the host OS):

| Shell | working-dir mount |
|-------|-------------------|
| **PowerShell** | `-v "${PWD}:/work"` |
| **cmd** | `-v "%cd%:/work"` |
| **WSL2 / Linux / macOS bash** | `-v "$PWD":/work` |

## Bootstrap + preflight (PowerShell example)
```powershell
# one-time bootstrap: venv + deps + ds_image_eval (idempotent), from the working root
docker run --rm -it --gpus all --shm-size=16g -v "${PWD}:/work" -w /work `
  --entrypoint bash nvcr.io/nvidia/deepstream:9.1-triton-multiarch `
  .claude/skills/deepstream-eval-and-finetune/setup.sh

# preflight — run it THROUGH the container (container-mode auto-detects /.dockerenv and
# checks GPU + venv directly; no host bash needed)
docker run --rm --gpus all -v "${PWD}:/work" -w /work `
  --entrypoint bash nvcr.io/nvidia/deepstream:9.1-triton-multiarch `
  .claude/skills/deepstream-eval-and-finetune/scripts/preflight.sh
```
Then run a validated example the same way (swap `setup.sh` for `examples/rtdetr-aerial-sheep/run.sh`).

### Web UI on Windows
The UI runs **through Docker**, same as everything else: it's a `uvicorn` server started **inside** the
DeepStream container, with the port published to Windows `localhost`. Use the bundled **`app/launch.ps1`**
(the PowerShell twin of `launch.sh`) — one command from the working root:
```powershell
.\.claude\skills\deepstream-eval-and-finetune\app\launch.ps1
# optional external roots; use semicolons when paths contain spaces:
$env:DATASETS_ROOT="D:\data;E:\Team Datasets"
```
It runs `docker run --gpus all -v "${PWD}:/work" -p 127.0.0.1:8078:8078 … uvicorn server:app --host 0.0.0.0`, so
the WSL2 backend forwards the port — open **`http://localhost:8078`** in your Windows browser. Run
outputs persist on the Windows disk under the mounted working root. (Ctrl+C stops the container.)

The launcher also creates a persistent, project-scoped Linux Docker volume named
`ds-eval-data-<project-path-hash>` and mounts it at `/work/data`. This is automatic on native
PowerShell and WSL; the same Windows project path hashes to the same name in either shell.

| Variable | Behavior |
|----------|----------|
| `DS_DATA_CACHE=auto` | Default: enable on Windows/WSL; native Linux stays unchanged |
| `DS_DATA_CACHE=off` | Disable the Linux dataset cache and use the legacy bind-mounted `data/` |
| `DS_DATA_VOLUME=name` | Override the generated project-scoped volume name |
| `DS_DATA_CACHE_REFRESH=1` | Refresh the selected local source and rebuild that run |

The UI Setup page shows the active volume, staged-source size, and a **Refresh source cache and
rebuild this run** checkbox. Reuse is the default. Refresh is explicit because recursively checking
a large NTFS directory on every run would recreate the performance problem.

Each `DATASETS_ROOT` entry is mounted read-only under `/datasets/windows-N`. The launcher passes a
translation table to the UI, so users may enter the familiar Explorer path such as
`E:\Team Datasets\pcb`; ingestion resolves it to the mounted Linux path and stages it into the
fast cache. Paths containing spaces are supported.

## The tao-skill-bank plugin (host-native, NOT through Docker)
The fine-tune/automl lanes come from the `tao-skill-bank` **Claude Code plugin**, installed via the
cross-platform `claude` plugin CLI on the host — never in a container. The installers handle it:
`install.ps1` (Windows) and `install.sh` (Linux) run the **same sequence** and take the same
`-Scope`/`--scope project|user` flag (see **Install on Windows** above). Regardless of scope the plugin
cache lands in `%USERPROFILE%\.claude\plugins`; scope only sets whether it is enabled for the one
project or globally. The `claude` CLI records that itself — the installers do not write it.

## Git Bash pitfalls (Claude Code uses Git Bash on Windows)

Claude Code's Bash tool runs through **Git Bash** (`MINGW64`). Three issues surface when driving
`docker run` from it — all confirmed on a Windows 11 + Docker Desktop 29.x session:

### 1. Always set `MSYS_NO_PATHCONV=1`
Git Bash path-converts arguments that look like Unix paths before passing them to Docker. `/work`
becomes `C:/Program Files/Git/work`, breaking `-w /work` and any `/`-prefixed flags inside the
container. Fix: prefix every `docker run` call:

```bash
MSYS_NO_PATHCONV=1 docker run --rm --gpus all --shm-size=16g \
  -v "$(pwd)":/work -w /work \
  nvcr.io/nvidia/deepstream:9.1-triton-multiarch \
  bash /work/.claude/skills/deepstream-eval-and-finetune/scripts/preflight.sh
```

### 2. Never pass inline scripts via `bash -lc '...'`
Git Bash word-splits the single-quoted string when the total command length is large — the script
body arrives as 100+ individual positional arguments (`$0`, `$1`, …) instead of a single `-c`
string. `BASH_EXECUTION_STRING` will show `set` (the first word of your script) and nothing runs.

**Fix:** write the script to a file first, then pass the filename:

```bash
# BAD  — Git Bash word-splits the inline string
MSYS_NO_PATHCONV=1 docker run ... bash -lc 'set -e; python foo.py; ...'

# GOOD — script is a file; no quoting issues
cat > /tmp/myscript.sh <<'EOF'
#!/usr/bin/env bash
set -e
python foo.py
EOF
MSYS_NO_PATHCONV=1 docker run ... bash /work/myscript.sh
```

The validated per-stage scripts for the rtdetr-aerial-sheep example are already split this way:
`examples/rtdetr-aerial-sheep/run_stages_02.sh`, `run_stage3_finetune.sh`, `run_stages_45.sh`,
`run_ui.sh`. Run them with:

```bash
MSYS_NO_PATHCONV=1 docker run --rm --gpus all --shm-size=16g \
  -v "$(pwd)":/work -w /work \
  nvcr.io/nvidia/deepstream:9.1-triton-multiarch \
  bash /work/.claude/skills/deepstream-eval-and-finetune/examples/rtdetr-aerial-sheep/run_stages_02.sh
```

### 3. No `-it` for background/detached containers
`docker run -it` requires an interactive terminal. For containers started in the background (e.g.
the UI), use `-d` (detached) without `-it`:

```bash
# UI — detached, no -it, script file, port published
MSYS_NO_PATHCONV=1 docker run --rm -d \
  -v "$(pwd)":/work -w /work \
  -p 127.0.0.1:8078:8078 \
  nvcr.io/nvidia/deepstream:9.1-triton-multiarch \
  bash /work/.claude/skills/deepstream-eval-and-finetune/examples/rtdetr-aerial-sheep/run_ui.sh
# → open http://localhost:8078
```

### 4. Windows dataset reads are staged once into Linux storage

Without caching, each epoch opens thousands of images through the Windows NTFS → WSL2 → Docker
bridge. The launcher now mounts its Linux volume over `/work/data`; `ingest_dataset.py` stages a
local directory/zip once under `/work/data/.sources/<hash>`, and its manifest points to that cached
copy. HF downloads and the PCB converter write there directly. The generated Hugging Face Arrow
datasets live under `/work/data/.arrow/<run-key>`.

Only high-frequency inputs move:

| Linux cache volume | Windows project bind mount |
|--------------------|----------------------------|
| source image copy, extracted archives, COCO data, Arrow train/eval data | checkpoints, TensorRT engines, predictions, metrics, logs, samples, PDFs, UI history |

Both sides persist across UI/container restarts. Deleting the cache volume removes only reusable
input copies; completed results remain in the Windows project and the source can be staged again.

For a separate NGC fine-tune/AutoML `docker run`, mount the same printed volume:

```powershell
docker run --rm --gpus all --shm-size=16g `
  -v "${PWD}:/work" -v "${env:DS_DATA_VOLUME}:/work/data" -w /work `
  -e DS_DATA_CACHE_ROOT=/work/data `
  nvcr.io/nvidia/pytorch:25.03-py3 <command>
```

Inspect or explicitly remove project caches:

```powershell
docker volume ls --filter label=com.nvidia.deepstream.eval-cache=true
docker volume inspect <volume-name>
# Explicit/destructive cache cleanup only; completed Windows-side results are untouched:
docker volume rm <volume-name>
```

### 5. First-run venv write is slow (~20-30 min)
`setup.sh` writes ~100k small Python files (torch, CUDA libs, etc.) into `build/.venv_train`
via the WSL2 → Windows NTFS bind mount. Small-file I/O across that bridge is inherently slow.
This is a **one-time cost** — subsequent `setup.sh` runs detect the venv and skip the install
entirely. To avoid the wait on future machines, use a named Docker volume for the venv (stays
in the WSL2 Linux filesystem, ~10× faster) — trade-off: venv is not visible from Windows Explorer.

## Notes
- Line endings: the skill ships a `.gitattributes` forcing **LF** on all scripts, so a Windows
  checkout won't CRLF-corrupt them (CRLF breaks bash-in-container).
- `--shm-size=16g` is still required (default `/dev/shm` deadlocks DataLoader workers) and works on
  the WSL2 backend.
- Prefer running from a **WSL2 Ubuntu terminal** if you want the exact Linux experience — inside WSL2
  everything (including the host-side `install.sh`/`preflight.sh`) runs unchanged.
