# AGENTS.md — Working in the DeepStream Repository

Guidance for AI coding agents (Claude Code, Cursor, Codex, …) working in this
repository. For product/user documentation, read `README.md`; this file is about
*how to work here*.

## Project overview

NVIDIA DeepStream SDK 9.1 repository — a GStreamer-based framework for multi-stream,
multi-model video-analytics pipelines on NVIDIA GPUs (x86 dGPU, Jetson, SBSA/DGX
Spark). This repo holds the full DeepStream 9.1 source plus prebuilt proprietary
runtime assets (downloaded from GitHub Releases), build tooling, agent skills, and example prompts.

## Repository layout

- `src/` — all buildable source: `gst-plugins/`, `gst-utils/`, `utils/`,
  `apps/{sample_apps,reference_apps,tao_apps}/`, `service-maker/`. See `src/README.md`.
  `src/service-maker/` is the source for the `pyservicemaker` Python flow API — it is
  built and installed as part of `bash build/build.sh` and is also included in the
  all-in-one SDK deb/tarball. Do **not** use `deepstream_libraries-*.whl` as a substitute.
- `build/` — `build.sh` (top-level build driver) and `BUILD.md` (authoritative build guide).
  During the build, `build/build.sh` downloads GitHub Release assets into a temporary
  `artifacts/` directory; this is created automatically and removed after a successful install.
  See **GitHub Release Assets** section below for asset descriptions and manual install rules.
- `includes/` — shared public headers.
- `scripts/` — install/uninstall, Triton setup, rtpmanager fix, and build-flow helpers.
  `install_artifacts.sh` is invoked internally by `build/build.sh` to install component
  packages (`deepstream-binaries-*`, `deepstream-sample-data-*`) as part of the source
  build flow — do not run it directly for a manual bare-metal install.
- `skills/`, `example_prompts/` — agent assets (see "Skills & prompts" below).
- `tools/`, `deepstream_libraries/` (submodule), `deepstream_dockers/`.

## GitHub Release Assets

Packages are available at https://github.com/NVIDIA/DeepStream/releases/tag/v9.1.0.
`build/build.sh` downloads them automatically; for manual installs, pick from below.

- **Bare-metal DeepStream install** — use the all-in-one package for your platform:
  `deepstream-9.1_9.1.0-1_amd64.deb` (x86, ~970 MB) or `deepstream-9.1_9.1.0-1_arm64.deb` (Jetson, ~615 MB).
  Tarball equivalents: `deepstream_sdk_v9.1.0_x86_64.tbz2` / `deepstream_sdk_v9.1.0_jetson.tbz2`.
  These are the only packages needed for a full bare-metal install.
- **`deepstream-sample-data_*`** — contains sample models, configs, and video streams.
  Already bundled inside the full SDK packages above; no need to install separately.
- **`deepstream_libraries-*.whl`** — standalone Python package providing APIs for reference
  applications built on CVCUDA, NvImageCodec, and PyNvVideoCodec modules. Not required for
  a bare-metal DS mono-repo build or when the full SDK deb/tarball is installed. Install
  only when your use case explicitly requires these Python bindings. For `pyservicemaker`,
  install the full SDK deb/tarball or build from `src/service-maker/` — this whl is not a substitute.
- **`deepstream-binaries-*`** — prebuilt proprietary runtime libraries subset. Already bundled
  in the full SDK packages. Use only when updating a specific binary without replacing the
  full install. **Never use for a fresh bare-metal install.**

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

## Quick Reference

| Task | Command |
|------|---------|
| Full build | `bash build/build.sh` |
| Build one stage (after deps/artifacts installed) | `bash build/build.sh --only=<stage> --skip-deps --skip-artifacts` |
| Resume interrupted build | `bash build/build.sh --resume` |
| Install system deps only | `bash build/build.sh --only=deps` |
| Download + install artifacts only | `bash build/build.sh --only=artifacts` |
| Clear GStreamer plugin cache | `rm -rf ~/.cache/gstreamer-1.0/` |
| Run DeepStream reference app | `deepstream-app -c <config-file>` |
| Print environment (for diagnostics) | `bash scripts/print_env.sh` |
| Install full SDK (bare-metal, x86) | deb: `sudo apt-get install -y ./deepstream-9.1_9.1.0-1_amd64.deb` \| tar: `tar -xjvf deepstream_sdk_v9.1.0_x86_64.tbz2 -C / && sudo ldconfig` |
| Install full SDK (bare-metal, Jetson) | deb: `sudo apt-get install -y ./deepstream-9.1_9.1.0-1_arm64.deb` \| tar: `tar -xjvf deepstream_sdk_v9.1.0_jetson.tbz2 -C / && sudo ldconfig` |

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
- Don't install `deepstream-binaries-*` or `deepstream-sample-data-*` for a bare-metal
  DeepStream install — use `deepstream-9.1_*.deb` (the all-in-one package). Component
  packages are subsets bundled in the full SDK; `build/build.sh` installs them
  automatically. Install them separately only when updating a specific component.
- Don't install `deepstream_libraries-*.whl` for a bare-metal DS install or when the full
  SDK deb/tarball is already installed — it is a standalone Python package for CVCUDA,
  NvImageCodec, and PyNvVideoCodec reference app bindings only.
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
