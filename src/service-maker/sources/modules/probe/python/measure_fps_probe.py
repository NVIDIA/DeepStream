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
Python equivalent of the C++ measure_fps_probe plugin.

Importing this module registers "measure_fps_probe" in CommonFactory so it
can be attached by name string or declared in a YAML config:

    pipeline.attach("pgie", "measure_fps_probe", "fps")
    # or in dstest2_config.yaml:
    #   - type: python.measure_fps_probe
    #     name: fps
"""

import time
import threading
import weakref
from pyservicemaker import probe, BatchMetadataOperator

DEFAULT_INTERVAL = 5


@probe(
    name="measure_fps_probe",
    params={"interval": ("integer", DEFAULT_INTERVAL, "FPS printing interval in seconds")}
)
class FPSCounter(BatchMetadataOperator):

    def __init__(self):
        super().__init__()
        self._lock = threading.Lock()
        self._buf_count = {}           # current interval: pad_idx → frame count
        self._buf_count_lifetime = {}  # all-time:         pad_idx → frame count
        self._first_frame_time = {}    # pad_idx → monotonic timestamp of first frame
        self._last_measurement = time.monotonic()
        self._interval = DEFAULT_INTERVAL
        self._interval_loaded = False
        # An Event rather than a bool so stop() wakes the scheduler at once
        # instead of waiting out the remaining interval, matching the C++
        # probe which notifies its condition variable before joining.
        self._stop = threading.Event()
        # The thread must not hold a strong reference to the instance: a bound
        # method target forms a cycle (thread -> self -> thread) which keeps the
        # instance, and its FPS output, alive for the life of the process. Pass
        # a weakref so the instance stays collectable and __del__ can run.
        # daemon=True so the thread doesn't block process exit
        self._thread = threading.Thread(
            target=self._scheduler, args=(weakref.ref(self),), daemon=True)
        self._thread.start()

    def __del__(self):
        self.stop()

    def stop(self):
        """Signal the scheduler thread and wait for it to exit."""
        # getattr guards: __del__ may run after a partially failed __init__.
        stop = getattr(self, "_stop", None)
        if stop is not None:
            stop.set()
        thread = getattr(self, "_thread", None)
        if thread is not None and thread.is_alive():
            thread.join(timeout=5)

    @staticmethod
    def _scheduler(self_ref):
        while True:
            this = self_ref()
            if this is None:
                return
            stop, interval = this._stop, this._interval
            this = None  # release the strong reference before sleeping
            if stop.wait(interval):
                return
            this = self_ref()
            if this is None:
                return
            with this._lock:
                this._print_fps()
            this = None

    def _print_fps(self):
        now = time.monotonic()
        counts = dict(self._buf_count)
        self._buf_count.clear()
        elapsed = max(now - self._last_measurement, 1e-6)
        self._last_measurement = now

        parts = []
        for pad_idx in sorted(counts):
            instant_fps  = counts[pad_idx] / elapsed
            first        = self._first_frame_time.get(pad_idx, now)
            lifetime_fps = self._buf_count_lifetime.get(pad_idx, 0) / max(now - first, 1e-6)
            parts.append(f"{instant_fps:.2f} ({lifetime_fps:.2f})")

        if parts:
            print("**FPS:  " + "\t".join(parts) + "\t", flush=True)

    def handle_metadata(self, batch_meta):
        if not self._interval_loaded:
            val = self.get_property("interval")
            if val:
                self._interval = int(val)
            self._interval_loaded = True
        now = time.monotonic()
        with self._lock:
            for frame_meta in batch_meta.frame_items:
                pad_idx = frame_meta.pad_index
                self._first_frame_time.setdefault(pad_idx, now)
                self._buf_count[pad_idx]          = self._buf_count.get(pad_idx, 0) + 1
                self._buf_count_lifetime[pad_idx] = self._buf_count_lifetime.get(pad_idx, 0) + 1
