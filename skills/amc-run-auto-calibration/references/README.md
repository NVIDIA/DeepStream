<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# AMC Reference Routing

Load only the primary reference for the requested workflow. When `video-files.md`, `rtsp.md`, or `troubleshooting.md` explicitly hands a custom dataset to tuning, load `tuning.md` next. Do not preload every reference.

Primary routes:

- `setup.md` — service lifecycle and readiness
- `sample-dataset.md` — SDG sample workflow
- `video-files.md` — local MP4 workflow
- `rtsp.md` — VIOS RTSP capture workflow
- `tuning.md` — bounded custom-dataset settings search
- `calibration-parameters.md` — core AMC parameter meanings, polarity, and tuning guardrails
- `troubleshooting.md` — setup, API, capture, and calibration failures
