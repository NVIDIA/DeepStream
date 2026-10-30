<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Agent Guidance

Use the DeepStream repository's combined AutoMagicCalib skill for setup and calibration workflows.

- Use `../../skills/amc-run-auto-calibration/SKILL.md` for service setup, sample dataset validation, pre-recorded MP4 calibration, RTSP/VIOS capture, troubleshooting, and output verification.
- Load the relevant reference file from that skill instead of duplicating setup or REST API logic.
- Use `README.md` for product setup, UI workflow, dataset requirements, and REST API details.

Full calibration runs require a real runtime host with an NVIDIA GPU with NVENC, NVIDIA driver, Docker without sudo, NVIDIA Container Toolkit, and NGC image access. Optional VGGT refinement also requires HuggingFace access and the VGGT model. In restricted or no-GPU environments, inspect docs and prepare commands, but do not report runtime validation as complete.

Typical runtime depends on image pulls, model downloads, video length, and detector choice. The bundled sample can take several minutes up to about 30 minutes; custom video calibration can take 10-60+ minutes; VGGT refinement usually adds a few minutes when enabled.
