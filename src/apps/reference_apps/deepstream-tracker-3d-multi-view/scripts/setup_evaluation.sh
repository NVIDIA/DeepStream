#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ ! -x mv3dt_venv/bin/python ]]; then
    python3 -m venv mv3dt_venv
fi
mv3dt_venv/bin/python -m pip install -r requirements.txt scipy pycocotools==2.0.10 tabulate==0.9.0 \
    'trackeval @ git+https://github.com/JonathonLuiten/TrackEval.git@12c8791b303e0a0b50f753af204249e622d0281a'
mv3dt_venv/bin/python - <<'PY'
from pathlib import Path
from importlib.metadata import distribution

package = Path(distribution('trackeval').locate_file('trackeval/metrics'))
for filename, old, new in [('hota.py', 'dtype=np.float)', 'dtype=float)'),
                           ('identity.py', '.astype(np.int)', '.astype(int)')]:
    path = package / filename
    path.write_text(path.read_text().replace(old, new))
print('Evaluation environment ready in mv3dt_venv')
PY
