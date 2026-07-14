# AGENTS.md — Working in the DeepStream Repository

Guidance for AI coding agents (Claude Code, Cursor, Codex, …) working in this
repository. For product/user documentation, read `README.md`; this file is about
*how to work here*.

## Project overview

NVIDIA DeepStream SDK 9.1 repository — a GStreamer-based framework for multi-stream,
multi-model video-analytics pipelines on NVIDIA GPUs (x86 dGPU, Jetson, SBSA/DGX
Spark). This repo holds the full DeepStream 9.1 source plus prebuilt proprietary
runtime artifacts (downloaded from GitHub Release assets), build tooling, agent skills, and example prompts.

## Repository layout

- `src/` — all buildable source: `gst-plugins/`, `gst-utils/`, `utils/`,
  `apps/{sample_apps,reference_apps,tao_apps}/`, `service-maker/`. See `src/README.md`.
- `build/` — `build.sh` (top-level build driver) and `BUILD.md` (authoritative build guide).
- `artifacts/` — prebuilt proprietary libs + sample data, downloaded automatically from GitHub Release assets by `build/build.sh` (not committed to the repo).
- `includes/` — shared public headers.
- `scripts/` — install/uninstall, dependency and artifact install, Triton setup, rtpmanager fix.
- `skills/`, `example_prompts/` — agent assets (see "Skills & prompts" below).
- `tools/`, `deepstream_libraries/` (submodule), `deepstream_dockers/`.

## Build & run

Full build (auto-detects platform; prompts for sudo only when writing to system paths):

```bash
bash build/build.sh
```

- Stages: `artifacts` → `deps` → source (`gst-utils`, `utils`, `gst-plugins`,
  `sample_apps`, `yolo`, `tao_apps`, `reference_apps`, `service-maker`) → `install.sh`.
- Common flags: `--only=STAGE[,STAGE]`, `--resume`, `--skip-deps`, `--skip-artifacts`,
  `--install-method=deb|tar`, `--verbose`, `-j N`.
- Defaults: `CUDA_VER=13.2`, `NVDS_VERSION=9.1`. Outputs install to
  `/opt/nvidia/deepstream/deepstream-9.1/`.
- Each run writes a transcript to `build/build.log`; stage state lives in
  `build/.stage-state*` (enables `--resume`).
- Full details, per-platform prerequisites, and CLI reference: **`build/BUILD.md`**.

After the first install, clear the GStreamer plugin cache: `rm -rf ~/.cache/gstreamer-1.0/`.

## Conventions — DOs and DON'Ts

**DO**
- Clone with `--recurse-submodules`; run `git lfs install` before building (required for LFS-tracked sample media).
- Run each sample/reference app **from its own source directory** — configs use
  paths relative to the config file, not the cwd.
- Follow each component's own `README` for component-specific build/run steps
  (they are authoritative over any summary).
- Keep the SPDX license header (`SPDX-FileCopyrightText` + `SPDX-License-Identifier:
  Apache-2.0`) on every new/edited source file.
- Prefer `bash build/build.sh --only=<stage> --skip-deps --skip-artifacts` for
  fast iteration once deps/artifacts are installed.

**DON'T**
- **Don't commit build artifacts** — `.o`, `bin/`, `libs/`, `*.so.*`, `*.deb`,
  `*.tar.gz`/`*.zip`, `build/build.log`,
  `build/.stage-state*`. See `.gitignore`.
- Don't assume `build/build.sh` builds everything — `gst-nvdsudp` and
  `gst-dsexample-cuda` must be built separately per their READMEs.
- Don't attempt bare-metal install on SBSA / DGX Spark — build inside the NVIDIA
  SBSA Docker container; the `artifacts` stage is skipped there automatically.
- This project is **not accepting external contributions**; do not add
  contribution/PR scaffolding.

## Skills & prompts (for agent discovery)

- **`skills/`** — curated Agent Skills (each a `SKILL.md` + `references/` + `scripts/`).
  Claude Code auto-discovers them via `.claude/skills/`. Other agents: load a skill
  by reading its `SKILL.md`. Index and per-skill summaries: **`skills/README.md`**.
- **`example_prompts/`** — copy-pasteable prompts for common and specialized
  pipelines; use as starting points or verification tasks.

> Generated code is a starting point — run it through full SDLC (review, testing,
> security validation) before production use.

## DeepStream documentation (external)

Authoritative DeepStream product/API documentation is hosted online (HTML, many
sub-pages). When a task involves SDK usage, plugin/config semantics, Python APIs,
Service Maker, performance, or troubleshooting, **consult the public docs first
instead of guessing from memory.**

- Main docs (source of truth): https://docs.nvidia.com/metropolis/deepstream/dev-guide/index.html

High-value entry points (prefer these targeted links when the area is known —
don't recursively crawl the whole site):

- Quickstart / setup: https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Quickstart.html
- Plugins reference (intro): https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_Intro.html
- Gst-nvinfer plugin: https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvinfer.html
- Gst-nvinferserver (Triton) plugin: https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvinferserver.html
- Service Maker: https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_service_maker_intro.html
- Python API: https://docs.nvidia.com/metropolis/deepstream/dev-guide/python-api/index.html
- Performance: https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Performance.html
- Troubleshooting: https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_troubleshooting.html

Use the main URL as the canonical reference, but prefer the targeted sub-page
links above whenever the task maps to a known area.

## Key references

- Build: `build/BUILD.md` · Source layout: `src/README.md` · Product overview: `README.md`
- Env diagnostics for bug reports: `scripts/print_env.sh`
