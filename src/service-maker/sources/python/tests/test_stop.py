# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise stop while a streaming callback needs to reacquire the GIL."""

import faulthandler
import subprocess
import sys
import threading
import time

import pytest


@pytest.mark.parametrize("with_probe", [False, True])
def test_stop_releases_gil(with_probe: bool) -> None:
    result = subprocess.run(
        [sys.executable, __file__, str(int(with_probe))],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "all runs stopped" in result.stdout


def run_shutdown_cycles(with_probe: bool) -> None:
    from pyservicemaker import BufferOperator, Pipeline, Probe

    class BlockingProbe(BufferOperator):
        def __init__(self) -> None:
            super().__init__()
            self.armed = threading.Event()
            self.entered = threading.Event()
            self.release = threading.Event()
            self.buffers = 0

        def handle_buffer(self, buffer) -> bool:
            self.buffers += 1
            if self.armed.is_set():
                self.entered.set()
                # Wait releases the GIL while this thread holds the streaming lock.
                self.release.wait(5)
            return True

    for cycle in range(10):
        faulthandler.dump_traceback_later(10, exit=True)
        operator = BlockingProbe()
        pipeline = Pipeline(f"stop-regression-{cycle}")
        pipeline.add("videotestsrc", "source", {"pattern": 2})
        pipeline.add("fakesink", "sink", {"sync": False})
        pipeline.link("source", "sink")
        if with_probe:
            pipeline.attach("source", Probe("blocking", operator))
        pipeline.start()
        # Let initial preroll complete before holding a streaming callback.
        time.sleep(0.1)
        operator.armed.set()
        if with_probe:
            assert operator.entered.wait(5), "Probe never received a buffer"
        else:
            time.sleep(0.05)
        waiter = threading.Thread(target=pipeline.wait, daemon=True)
        waiter.start()
        # This Python timer must acquire the GIL while stop waits for the probe.
        timer = threading.Timer(0.1, operator.release.set)
        timer.start()
        assert pipeline.stop() is pipeline
        waiter.join(5)
        assert not waiter.is_alive(), "wait() did not complete after EOS"
        timer.join(5)
        assert not timer.is_alive()
        if with_probe:
            assert operator.buffers > 0
        faulthandler.cancel_dump_traceback_later()
    print("all runs stopped", flush=True)


if __name__ == "__main__":
    run_shutdown_cycles(bool(int(sys.argv[1])))
