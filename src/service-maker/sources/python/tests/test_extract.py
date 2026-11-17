# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU extraction regression tests; isolate native crashes and hangs in subprocesses."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize("format_name", ["RGB", "RGBA", "NV12"])
@pytest.mark.parametrize("batched", [False, True])
def test_extract(format_name: str, batched: bool) -> None:
    result = subprocess.run(
        [sys.executable, __file__, format_name, str(int(batched))],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def run_case(format_name: str, batched: bool) -> None:
    from pyservicemaker import Buffer, BufferOperator, Pipeline, Probe

    pipeline = Pipeline("extract-regression")
    assert Buffer().extract(0) is None
    assert Buffer([1, 2, 3]).extract(0) is None

    class Inspect(BufferOperator):
        def __init__(self) -> None:
            super().__init__()
            self.results = []
            self.errors = []
            self.tensors = []

        def handle_buffer(self, buffer: Buffer) -> bool:
            # Never let test assertions escape a streaming-thread callback (#100).
            try:
                tensor = buffer.extract(0)
                if format_name == "RGB":
                    assert tensor is not None
                    assert tensor.shape == (48, 64, 3)
                    assert tensor.__dlpack_device__()[0] == 2
                    assert tensor.__dlpack__() is not None
                    self.tensors.append(tensor)
                else:
                    assert tensor is None
                # In a capacity-four batch only slot zero holds a frame.
                for index in (1, 3, 4, 2**32 - 1):
                    assert buffer.extract(index) is None, f"Unexpected tensor for slot {index}"
                self.results.append(True)
            except Exception as error:
                self.errors.append(repr(error))
            return True

    inspector = Inspect()
    pipeline.add("videotestsrc", "source", {"num-buffers": 1, "is-live": True})
    pipeline.add("capsfilter", "input_caps", {"caps": "video/x-raw,format=RGBA,width=64,height=48"})
    pipeline.add("nvvideoconvert", "upload", {})
    pipeline.add("capsfilter", "nvmm", {"caps": "video/x-raw(memory:NVMM),format=RGBA"})
    pipeline.link("source", "input_caps", "upload", "nvmm")
    upstream = "nvmm"
    if batched:
        pipeline.add("nvstreammux", "mux", {"batch-size": 4, "width": 64, "height": 48, "batched-push-timeout": 10000})
        pipeline.link(("nvmm", "mux"), ("", "sink_0"))
        upstream = "mux"
    pipeline.add("nvvideoconvert", "convert", {})
    pipeline.add("capsfilter", "output_caps", {"caps": f"video/x-raw(memory:NVMM),format={format_name}"})
    pipeline.add("fakesink", "sink", {"sync": False, "async": False})
    pipeline.link(upstream, "convert", "output_caps", "sink")
    pipeline.attach("output_caps", Probe("inspect", inspector))
    pipeline.start().wait()
    assert not inspector.errors, inspector.errors
    assert inspector.results, "No frames reached extraction probe"
    # Extracted tensors keep their GstBuffer alive after pipeline teardown.
    for tensor in inspector.tensors:
        copied = tensor.clone()
        assert copied is not None
        assert copied.shape == (48, 64, 3)


if __name__ == "__main__":
    run_case(sys.argv[1], bool(int(sys.argv[2])))
