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

"""
deepstream_python_probe.py (flow_api) — demonstrates how to attach a Python
probe module using the Flow API.

The Flow API abstracts pipeline element wiring; probes are attached with
flow.attach(what=<name>, name=<instance>, properties={...}).

Usage:
  python3 deepstream_python_probe.py <h264_file>
"""

from pyservicemaker import Pipeline, Flow, RenderMode
from multiprocessing import Process
import sys
import os

_PROBES_DIR = "/opt/nvidia/deepstream/deepstream/service-maker/modules/python"
sys.path.insert(0, _PROBES_DIR)

import sample_video_probe  # noqa: F401  registers "sample_video_probe"

PIPELINE_NAME = "deepstream-python-probe-flow"
PGIE_CONFIG   = "/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-test1/dstest1_pgie_config.yml"

RENDER_MODE = RenderMode.DISPLAY if os.environ.get("DISPLAY") else RenderMode.DISCARD


def main(stream_path):
    (Flow(Pipeline(PIPELINE_NAME))
        .batch_capture([stream_path])
        .infer(PGIE_CONFIG)
        .attach(what="sample_video_probe", name="svp", properties={"font-size": 18})
        .render(RENDER_MODE)())


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.stderr.write("usage: %s <h264_file>\n" % sys.argv[0])
        sys.exit(1)

    process = Process(target=main, args=(sys.argv[1],))
    try:
        process.start()
        process.join()
    except KeyboardInterrupt:
        print("\nCtrl+C — terminating.")
        process.terminate()
