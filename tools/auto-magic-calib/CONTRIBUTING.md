<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0 -->

# Contribution Rules

AutoMagicCalib is part of the DeepStream repository and follows the project-wide
contribution process: see the root [CONTRIBUTING.md](../../CONTRIBUTING.md) for
contribution paths, the pull request process, the Developer Certificate of Origin
(DCO) sign-off (`git commit -s`), and licensing requirements.

For changes in this directory:

- Open an issue first for new features, changes to the REST API, or changes to the
  deployment scripts and container configuration, and wait for maintainer feedback
  before writing code.
- Test with the sample calibration dataset or your own `cam_*.mp4` set, and state
  the DeepStream version, GPU, OS, and container tag you used in the pull request.
- Keep the SPDX header on every new or edited file.

## Reporting issues

We welcome bug reports and feature requests.

**File an [issue](https://github.com/NVIDIA/DeepStream/issues)** - Include reproduction steps, expected vs. actual behavior, and environment details
  (DeepStream version, GPU, OS, container tag).

## Asking questions

For usage questions, deployment help, or general discussion, please use the
[NVIDIA DeepStream SDK Developer Forum](https://forums.developer.nvidia.com/c/accelerated-computing/intelligent-video-analytics/deepstream-sdk/15).
