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
Python equivalent of the C++ measure_latency_probe plugin.

Importing this module registers "measure_latency_probe" in CommonFactory:

    pipeline.attach("sink", "measure_latency_probe", "latency")
"""

from pyservicemaker import probe, BufferObserver


@probe(name="measure_latency_probe")
class NvDsMeasureLatency(BufferObserver):

    def handle_buffer(self, buffer):
        for entry in buffer.measure_latency():
            print(
                f"Source id = {entry['source_id']}"
                f" Frame_num = {entry['frame_num']}"
                f" Frame latency = {entry['latency']:.4f} (ms)"
            )
