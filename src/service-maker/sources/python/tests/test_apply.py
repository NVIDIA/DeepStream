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
Test suite for Flow.apply() method (nvdsvideotemplate plugin) integration with pyservicemaker.

Note: Set NVDS_VIDEOTEMPLATE_CUSTOMLIB environment variable to enable
functional testing with a custom library. Otherwise, tests validate API only.
"""

from pyservicemaker import Pipeline, Flow, RenderMode, BufferOperator, Probe
import threading
import time
import os
import tempfile

CUSTOM_LIB_PATH = os.environ.get("NVDS_VIDEOTEMPLATE_CUSTOMLIB", None)

RENDER_MODE = RenderMode.DISPLAY if os.environ.get("DISPLAY") else RenderMode.DISCARD

def stop_pipeline(pipeline, timeout=1):
    time.sleep(timeout)
    print("Stop")
    pipeline.stop()

class VideoTemplateValidator(BufferOperator):
    """Validates that apply() method is processing buffers"""
    def __init__(self):
        super().__init__()
        self.frame_count = 0

    def handle_buffer(self, buffer):
        self.frame_count += buffer.batch_size
        return True

def test_apply_basic():
    """Test basic apply() method functionality"""
    pipeline = Pipeline("test")
    validator = VideoTemplateValidator()
    thread = threading.Thread(target=stop_pipeline, args=(pipeline, 5))
    thread.start()

    flow = Flow(pipeline).batch_capture(["/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4"], file_loop=False)
    flow = flow.apply(template_lib=CUSTOM_LIB_PATH, gpu_id=0)
    flow = flow.attach(what=Probe("validator", validator))
    flow.render(RENDER_MODE)()

    thread.join()

    if CUSTOM_LIB_PATH:
        assert validator.frame_count > 0, f"Expected frames to be processed with custom lib, got {validator.frame_count}"

def test_apply_with_config():
    """Test apply() method with configuration file"""
    if not CUSTOM_LIB_PATH:
        return  # Skip if no custom library

    pipeline = Pipeline("test")
    validator = VideoTemplateValidator()
    thread = threading.Thread(target=stop_pipeline, args=(pipeline, 5))
    thread.start()

    # Create config with customlib-name included
    config_content = f"""# nvdsvideotemplate config
    gpu-id=0
    customlib-name={CUSTOM_LIB_PATH}
    """

    with tempfile.NamedTemporaryFile(mode='w', suffix='.conf', delete=False) as f:
        f.write(config_content)
        config_file = f.name

    try:
        flow = Flow(pipeline).batch_capture(["/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4"], file_loop=False)
        flow = flow.apply(template_config=config_file)
        flow = flow.attach(what=Probe("validator", validator))
        flow.render(RENDER_MODE)()

        thread.join()

        assert validator.frame_count > 0, f"Expected frames to be processed with config, got {validator.frame_count}"
    finally:
        if os.path.exists(config_file):
            os.remove(config_file)

def test_apply_with_customlib():
    """Test apply() method with custom library"""
    custom_lib = "libnvdscustomlib.so"

    # Check if custom library exists in standard paths
    import subprocess
    lib_exists = False
    try:
        result = subprocess.run(["ldconfig", "-p"], capture_output=True, text=True, timeout=5)
        lib_exists = custom_lib in result.stdout
    except:
        pass

    if not lib_exists:
        print(f"⚠ Skipping test_apply_with_customlib: {custom_lib} not found")
        return

    pipeline = Pipeline("test")
    validator = VideoTemplateValidator()
    thread = threading.Thread(target=stop_pipeline, args=(pipeline, 10))
    thread.start()

    # Test with custom library name
    flow = Flow(pipeline).batch_capture(["/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4"], file_loop=False)
    flow = flow.apply(template_lib=custom_lib, gpu_id=0)
    flow = flow.attach(what=Probe("validator", validator))
    flow.render(RENDER_MODE)()

    thread.join()

    # Verify that frames were processed with custom lib
    assert validator.frame_count > 0, f"Expected frames to be processed with custom lib, got {validator.frame_count}"

def test_apply_properties():
    """Test apply() method with various properties"""
    pipeline = Pipeline("test")
    validator = VideoTemplateValidator()
    thread = threading.Thread(target=stop_pipeline, args=(pipeline, 5))
    thread.start()

    flow = Flow(pipeline).batch_capture(["/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4"], file_loop=False)
    flow = flow.apply(template_lib=CUSTOM_LIB_PATH, gpu_id=0)
    flow = flow.attach(what=Probe("validator", validator))
    flow.render(RENDER_MODE)()

    thread.join()

    if CUSTOM_LIB_PATH:
        assert validator.frame_count > 0, f"Expected frames to be processed with custom lib, got {validator.frame_count}"

if __name__ == '__main__':
    test_apply_basic()
    test_apply_with_config()
    test_apply_with_customlib()
    test_apply_properties()
