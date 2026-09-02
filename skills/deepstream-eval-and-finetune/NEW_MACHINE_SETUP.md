<!--
Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License").
-->

# New-machine setup & testing — deepstream-eval-and-finetune

Follow these steps in order to install and test the skill on a fresh machine. The skill's
`install.sh` installs the `tao-skill-bank` fine-tune/automl plugin **with** the skill by calling
the `claude` CLI, which records it in its own configuration so it loads at session start. The
installer never edits agent config files itself; if the `claude` CLI is not on PATH it prints the
two plugin commands for you to run once.

> Verified on: Ubuntu + Docker 29.x, NVIDIA RTX A6000 (48 GB), DeepStream
> `9.1-triton-multiarch`. Baseline→fine-tuned on the aerial-sheep demo: mAP@50 ~0.19 → ~0.87.

> **Windows / macOS:** the steps below use `bash`; on **Windows with Docker Desktop** the same scripts
> run **through the container** (every bash script runs inside Linux) — see
> [references/windows.md](references/windows.md) for the Docker Desktop + WSL2 setup and the PowerShell
> `${PWD}` / cmd `%cd%` mount tokens. GPU requires the WSL2 backend. The one-time `install.sh` +
> `tao-skill-bank` plugin steps use the cross-platform `claude` CLI (or run `install.sh` from WSL2/Git
> Bash).

---

## 0. Machine prerequisites (one-time)

You need an **NVIDIA GPU**, **Docker** with GPU access, the **Claude Code CLI**, and network
access to GitHub + HuggingFace.

```bash
# NVIDIA driver + Docker + NVIDIA Container Toolkit installed and working, then:
docker pull nvcr.io/nvidia/deepstream:9.1-triton-multiarch

# sanity checks — GPU is verified THROUGH the container (no native nvidia-smi needed;
# only the host NVIDIA driver + Docker GPU access must be present)
docker --version
docker run --rm --gpus all --entrypoint nvidia-smi \
  nvcr.io/nvidia/deepstream:9.1-triton-multiarch -L     # GPU visible in-container
claude --version                                         # Claude Code CLI present (host orchestrator)
```

---

## 1. Install the skill into your project

**One command (installs the skill AND the tao-skill-bank plugin):** from wherever the skill
lives (your skills marketplace checkout or this repo), run its `install.sh` against your project:

```bash
bash <path-to>/deepstream-eval-and-finetune/install.sh --target <your-project-path>
#   flags: --dry-run (preview) · --no-cursor · --no-plugin
```

This copies the skill into `<target>/.claude/skills/deepstream-eval-and-finetune/` and installs
`tao-skills@tao-skill-bank` (project scope) through the `claude` CLI, so a fresh session auto-loads
the tao fine-tune/automl lanes. If the CLI is not on PATH the skill still installs and the two
plugin commands are printed for you to run once.

```bash
cd <your-project-path>
```

**Alternative:** clone a project that already has the skill and the plugin enabled:
`git clone <repo> && cd <project>`.

---

## 2. Start Claude Code — the skill + tao plugin load automatically

```bash
claude          # start a session IN the project directory
```

- On first start, **trust the folder** when prompted.
- `install.sh` already registered the tao-skill-bank marketplace and enabled
  `tao-skills@tao-skill-bank` via `claude plugin install`, so Claude **loads the tao skills at
  session start** — automatically. Confirm with `claude plugin list`.
- The `deepstream-eval-and-finetune` skill is present as a project skill.

---

## 3. One-time skill bootstrap (venv + C app)

From the project working root, run the skill's setup **inside the DeepStream container** so
torch/CUDA and the C app match the runtime:

```bash
docker run --rm -it --gpus all --shm-size=16g -v "$PWD":/work -w /work \
  --entrypoint bash nvcr.io/nvidia/deepstream:9.1-triton-multiarch \
  .claude/skills/deepstream-eval-and-finetune/setup.sh
```

Creates `build/.venv_train` (torch, transformers, datasets, …) and builds `ds_image_eval`.
Idempotent, ~a few minutes.

---

## 4. Verify readiness (proceed only on PASS × 2)

```bash
bash .claude/skills/deepstream-eval-and-finetune/scripts/preflight.sh
#   → PASS: docker · DeepStream image · GPU · venv packages

bash .claude/skills/deepstream-eval-and-finetune/scripts/check_tao_skills.sh
#   → PASS: tao-launch-workflow + tao-finetune-huggingface-model + tao-run-automl resolve
```

If `check_tao_skills.sh` says **INSTALL-NEEDED** (skill copied in without the plugin), run
`bash .claude/skills/deepstream-eval-and-finetune/scripts/install_tao_skills.sh`, then start a
new session so the enabled plugin loads, and re-run the check.

---

## 5. Run a test — pick one

**A) Validated end-to-end demo** (turnkey; exercises the whole pipeline incl. the tao
fine-tune lane):

```bash
docker run --rm -it --gpus all --shm-size=16g -v "$PWD":/work -w /work \
  --entrypoint bash nvcr.io/nvidia/deepstream:9.1-triton-multiarch \
  .claude/skills/deepstream-eval-and-finetune/examples/rtdetr-aerial-sheep/run.sh
```
Expected: baseline mAP@50 ~0.19 → fine-tuned ~0.87, ~605 FPS (FP16), before/after PDF under
`reports/`.

**B) Web UI** (guided, visual — baseline → fine-tune → compare with live charts):

```bash
bash .claude/skills/deepstream-eval-and-finetune/app/launch.sh   # then open http://localhost:8078
```

**C) Interactive via the skill** — in the Claude session say, e.g.:
> run deepstream-eval-and-finetune on the default RT-DETR + aerial-sheep example

It orchestrates deploy → eval → fine-tune → redeploy → eval → report, pausing at the single
`tao-launch-workflow` launch-review gate.

**Quick smoke test** (fast, no training): drive the demo but stop after the baseline eval —
deploy the original model + measure baseline mAP/FPS to confirm the deploy+eval path, before
committing to the full fine-tune.

---

## Flow recap

```
prereqs → install.sh --target <proj> → `claude` (trust; tao plugin loads) → setup.sh
        → preflight PASS + check_tao_skills PASS → run demo / UI / skill
```

The only interactive point during a run is the `tao-launch-workflow` launch review at the
fine-tune stage. See `references/run-flow.md` for the full flow and `SKILL.md` for stage
details.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| preflight: `image not pulled` | `docker pull nvcr.io/nvidia/deepstream:9.1-triton-multiarch` |
| preflight: `no GPU through --gpus all` | install/repair NVIDIA Container Toolkit; verify `docker run --rm --gpus all … nvidia-smi -L` |
| preflight: `venv not found` | run step 3 (`setup.sh`) |
| check_tao_skills: `INSTALL-NEEDED` | run `scripts/install_tao_skills.sh`, then start a new session |
| tao skill "not found" when invoked | plugin installed mid-session — start a new session so enabled plugins load |
| DataLoader hang / shm errors | ensure `--shm-size=16g` on every `docker run` |
| Git Bash: `-w /work` becomes `C:/Program Files/Git/work` | prefix every `docker run` with `MSYS_NO_PATHCONV=1` |
| Git Bash: inline `bash -lc '...'` runs nothing / `BASH_EXECUTION_STRING=set` | write script to a file and pass the filename (`bash /work/script.sh`) — see `references/windows.md` |
| UI `docker run -it` blocks in background | use `-d` (detached) without `-it`; use `run_ui.sh` from `examples/rtdetr-aerial-sheep/` |
| First-time `setup.sh` takes 20-30 min on Windows | expected — WSL2→NTFS small-file I/O is slow; one-time cost; subsequent runs skip the install |
