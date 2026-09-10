# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
deepstream_python_probe.py — demonstrates how to use an installed Python probe module.

Python probes live in:
  /opt/nvidia/deepstream/deepstream/service-maker/modules/python/

Importing the module registers it in CommonFactory under its probe name so it
can be attached identically to a C++ .so probe — by name string in YAML or API.

Two modes:
  Config (YAML): python3 deepstream_python_probe.py deepstream_python_probe_config.yaml
  Direct (API):  python3 deepstream_python_probe.py <h264_file>
"""

from pyservicemaker import Pipeline
from multiprocessing import Process
import sys
import platform
import os

# Add the installed Python probe modules directory to the path and import.
# Importing the module is all that is needed — the @probe decorator inside
# registers "sample_video_probe" in CommonFactory at import time.
_PROBES_DIR = "/opt/nvidia/deepstream/deepstream/service-maker/modules/python"
sys.path.insert(0, _PROBES_DIR)

import sample_video_probe  # noqa: F401  registers "sample_video_probe"

PIPELINE_NAME = "deepstream-python-probe"
PGIE_CONFIG   = "/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-test1/dstest1_pgie_config.yml"
STREAM_WIDTH  = 1280
STREAM_HEIGHT = 720


def main(arg):
    ext = os.path.splitext(arg)[1]

    if ext in (".yaml", ".yml"):
        # ---------------------------------------------------------------------
        # Config mode — probe is declared in the YAML as:
        #   type: python.sample_video_probe
        # Properties (e.g. font-size) are set directly in the YAML node.
        # ---------------------------------------------------------------------
        Pipeline(PIPELINE_NAME, arg).start().wait()

    else:
        # ---------------------------------------------------------------------
        # Direct (API) mode — attach the probe by its registered name.
        # properties= sets the declared probe params programmatically,
        # identical to setting them in a YAML node.
        # ---------------------------------------------------------------------
        sink = "nv3dsink" if platform.processor() == "aarch64" else "nveglglessink"

        (Pipeline(PIPELINE_NAME)
            .add("filesrc",       "src",   {"location": arg})
            .add("h264parse",     "parse")
            .add("nvv4l2decoder", "dec")
            .add("nvstreammux",   "mux",   {"batch-size": 1,
                                            "width": STREAM_WIDTH,
                                            "height": STREAM_HEIGHT})
            .add("nvinfer",       "infer", {"config-file-path": PGIE_CONFIG})
            .add("nvosdbin",      "osd")
            .add(sink,            "sink")
            .link("src", "parse", "dec")
            .link(("dec", "mux"), ("", "sink_%u"))
            .link("mux", "infer", "osd", "sink")
            .attach("infer", "sample_video_probe", "svp",
                    properties={"font-size": 18})
            .start()
            .wait())


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.stderr.write(
            "usage: %s <h264_file>  |  <pipeline_config.yaml>\n" % sys.argv[0]
        )
        sys.exit(1)

    process = Process(target=main, args=(sys.argv[1],))
    try:
        process.start()
        process.join()
    except KeyboardInterrupt:
        print("\nCtrl+C — terminating.")
        process.terminate()
