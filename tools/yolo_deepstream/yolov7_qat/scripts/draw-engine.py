################################################################################
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
################################################################################


"""
This script generates an SVG diagram of the input engine graph SVG file.
Note:
    THIS SCRIPT DEPENDS ON LIB: https://github.com/NVIDIA/TensorRT/tree/main/tools/experimental/trt-engine-explorer
    this script requires graphviz which can be installed manually:
    $ sudo apt-get --yes install graphviz
    $ python3 -m pip install graphviz networkx
"""

import graphviz
from trex import *
import argparse
import shutil


def draw_engine(engine_json_fname: str, engine_profile_fname: str):
    graphviz_is_installed =  shutil.which("dot") is not None
    if not graphviz_is_installed:
        print("graphviz is required but it is not installed.\n")
        print("To install on Ubuntu:")
        print("sudo apt --yes install graphviz")
        exit()

    plan = EnginePlan(engine_json_fname, engine_profile_fname)
    formatter = layer_type_formatter
    display_regions = True
    expand_layer_details = False

    graph = to_dot(plan, formatter,
                display_regions=display_regions,
                expand_layer_details=expand_layer_details)
    render_dot(graph, engine_json_fname, 'svg')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', help="name of engine JSON file to draw")
    parser.add_argument('--profile', help="name of profile JSON file to draw")
    args = parser.parse_args()
    draw_engine(engine_json_fname=args.layer,engine_profile_fname=args.profile)
