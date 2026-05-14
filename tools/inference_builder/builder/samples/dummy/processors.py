# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


import numpy as np
import torch

class DummyPreprocessor:
    name = "dummy-preprocessor"
    def __init__(self, config):
        self.network_size = config['network_size']

    def __call__(self, *args):
        if len(args) != 1:
            raise ValueError("DummyPreprocessor expects exactly one argument")
        input = args[0]
        if len(input.shape) != 3:
            raise ValueError("DummyPreprocessor expects a 3D tensor")
        if isinstance(input, np.ndarray) and input.dtype != np.uint8:
            raise ValueError("DummyPreprocessor expects a uint8 tensor")
        if isinstance(input, torch.Tensor) and input.dtype != torch.uint8:
            raise ValueError("DummyPreprocessor expects a uint8 tensor")
        return np.random.randn(*self.network_size)


class DummyTokenizer:
    name = "dummy-tokenizer"
    def __init__(self, config):
        pass

    def __call__(self, *args):
        return np.random.randint(0, 100, size=10), np.array([10])
