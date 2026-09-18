# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# DeepStream Fern docs

Author MDX under `docs/` (this repository). Fern configuration lives in this directory.

```text
docs/                 authored MDX, images, and table CSVs
docs/fern/            fern.config.json, docs.yml, versions/
```

Preview locally from this directory:

```bash
npm install fern-api
npx fern docs dev
```

Validate:

```bash
npx fern check
```

The site is configured for `docs.nvidia.com/metropolis/deepstream` with the NVIDIA global theme. CI workflows are intentionally not included in this change.
