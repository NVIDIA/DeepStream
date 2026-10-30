<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# AMC Run Auto-Calibration Evals

This directory contains combined eval cases for the `amc-run-auto-calibration` skill. The cases cover the workflows that were previously split across setup, sample dataset, local video, and RTSP calibration skills. Existing baseline cases are retained; appended cases cover custom-dataset tuning.

The evals should check that an agent:

- routes the request to the correct reference file
- uses `<DEEPSTREAM_REPO>/...` or `$DEEPSTREAM_REPO/...` paths instead of user-home paths
- avoids references to other repositories for compose files or runtime assets
- protects NGC and RTSP credentials
- reports service health, project/session status, and output paths
- asks for missing release paths or dataset source only when required

The evals are workflow-level and release-facing. They should not depend on private machine paths, obsolete external helpers, or benchmark-only artifacts.

Tuning cases check bounded attempts, evidence-driven adjustments, per-project VGGT/AMC preservation, stable naming, verified baseline reuse, configuration handoff, and approved cleanup. Keep assertions focused on these decisions rather than exact wording.
