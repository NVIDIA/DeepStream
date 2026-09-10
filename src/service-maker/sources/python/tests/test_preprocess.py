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

from pyservicemaker import Pipeline, Flow, BufferProvider, BufferOperator, Buffer, Probe, BatchMetadataOperator, as_tensor, RenderMode
import numpy as np
import os

RENDER_MODE = RenderMode.DISPLAY if os.environ.get("DISPLAY") else RenderMode.DISCARD
PREPROCESS_CONFIG = "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_preprocess.yml"
PREPROCESS_WIDTH = 960
PREPROCESS_HEIGHT = 544
N_ROIS = 8

class MyBufferProvider(BufferProvider):
    def __init__(self, width, height, device='cpu', framerate=30, format="RGB"):
        super().__init__()
        self.width = width
        self.height = height
        self.format = format
        self.framerate = framerate
        self.device = device
        self.count = 0
        self.expected = 255

    def generate(self, size):
        data = [self.count]*(self.width*self.height*3)
        if self.count < self.expected:
            self.count += 1
        return Buffer() if self.count == self.expected else Buffer(data)

class MyMetadataObserver(BatchMetadataOperator):
    def handle_metadata(self, batch_meta):
        for u_mata in batch_meta.preprocess_batch_items:
            preprocess_batch = u_mata.as_preprocess_batch()
            assert len(preprocess_batch.rois) == N_ROIS
            if not preprocess_batch:
                continue
            preprocess_tensor = preprocess_batch.preprocess_tensor_meta
            dims = preprocess_tensor.tensor.shape
            assert dims[0] == N_ROIS
            assert dims[1] == 3
            assert dims[2] == PREPROCESS_HEIGHT
            assert dims[3] == PREPROCESS_WIDTH

def tensor_generator(n_frames: int):
    return {
        "input_1": as_tensor(np.random.rand(3, 224, 224).astype(np.float32), "CHW").to_gpu(0),
        "input_2": as_tensor(np.random.rand(3, 224, 224).astype(np.float32), "CHW").to_gpu(0),
    }

class TensorInspector(BatchMetadataOperator):
    def handle_metadata(self, batch_meta):
        collected = []
        for u_mata in batch_meta.preprocess_batch_items:
            preprocess_batch = u_mata.as_preprocess_batch()
            if not preprocess_batch:
                continue
            preprocess_tensor = preprocess_batch.preprocess_tensor_meta
            if preprocess_tensor.name == "input_1" or preprocess_tensor.name == "input_2":
                collected.append(preprocess_tensor)
        assert len(collected) == 2
        assert "input_1" in [collected[0].name, collected[1].name]
        assert "input_2" in [collected[0].name, collected[1].name]
        assert collected[0].tensor.shape == (3, 224, 224)
        assert collected[1].tensor.shape == (3, 224, 224)

def test_preprocess_tensor():
    pipeline = Pipeline("test")
    providers = [MyBufferProvider(320, 240) for _ in range(4)]
    probe = Probe("probe", MyMetadataObserver())
    Flow(pipeline).inject(providers).batch().preprocess(PREPROCESS_CONFIG).attach(what=probe).render(RENDER_MODE, sync=False)()

def test_preprocess_tensor_set():
    pipeline = Pipeline("test")
    providers = [MyBufferProvider(320, 240)]
    probe = Probe("probe", TensorInspector())
    Flow(pipeline).inject(providers).batch().preprocess(PREPROCESS_CONFIG, tensor_generator).attach(what=probe).render(RENDER_MODE, sync=False)()

if __name__ == '__main__':
    test_preprocess_tensor()
    test_preprocess_tensor_set()
