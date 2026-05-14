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
import cvcuda
import torch
from PIL import Image
from io import BytesIO
import base64

class PreprocessorCvcuda:
    """This module converts the input RGB frame into the format expected by the model."""
    name = "changenet-normalizer"
    def __init__(self, config):
        self.device_id = config["device_id"]
        self.network_size = config['network_size']
        self.mean_tensor = torch.Tensor([0.5, 0.5, 0.5])
        self.mean_tensor = self.mean_tensor.reshape(1, 1, 1, 3).cuda(self.device_id)
        self.mean_tensor = cvcuda.as_tensor(self.mean_tensor, "NHWC")
        self.stddev_tensor = torch.Tensor([0.5, 0.5, 0.5])
        self.stddev_tensor = self.stddev_tensor.reshape(1, 1, 1, 3).cuda(self.device_id)
        self.stddev_tensor = cvcuda.as_tensor(self.stddev_tensor, "NHWC")

    def __call__(self, *args):
        images = [arg for arg in args]
        frame_nhwc = torch.stack(images)
        resized = cvcuda.resize(
            cvcuda.as_tensor(frame_nhwc, "NHWC"),
            (
                frame_nhwc.shape[0],
                self.network_size[1],
                self.network_size[0],
                frame_nhwc.shape[3],
            ),
            cvcuda.Interp.LINEAR,
        )

        # Convert to floating point range 0-1.
        normalized = cvcuda.convertto(resized, np.float32, scale=1 / 255)

        # Normalize with mean and std-dev.
        normalized = cvcuda.normalize(
            normalized,
            base=self.mean_tensor,
            scale=self.stddev_tensor,
            flags=cvcuda.NormalizeFlags.SCALE_IS_STDDEV,
        )

        # Convert it to NCHW layout and return it.
        normalized = cvcuda.reformat(normalized, "NCHW")
        result = torch.as_tensor(normalized.cuda(), device=f"cuda:{self.device_id}")
        result = list(torch.unbind(result, dim=0))

        return tuple(result)

class PostprocessorCvcuda:
    name = "changenet-masking"
    def __init__(self, config):
        self.color_map = {
            "0": torch.ByteTensor([255, 255, 255]),
            "1": torch.ByteTensor([255, 165, 0]),
            "2": torch.ByteTensor([230, 30, 100]),
            "3": torch.ByteTensor([70, 140, 0]),
            "4": torch.ByteTensor([218, 112, 214]),
            "5": torch.ByteTensor([0, 170, 240]),
            "6": torch.ByteTensor([127, 235, 170]),
            "7": torch.ByteTensor([230, 80, 0]),
            "8": torch.ByteTensor([205, 220, 57]),
            "9": torch.ByteTensor([218, 165, 32]),
        }
        self.network_size = config['network_size']
        self.n_class = config['n_class']

    def get_color_coded_img(self, vis, num_class=10):
        color_coded = torch.ones(
            (self.network_size[1], self.network_size[0], 3),
            dtype=torch.uint8,
        )
        for idx in range(num_class):
            filtered_indices = vis == idx
            color_coded[filtered_indices] = self.color_map[str(idx)]
        return color_coded

    def __call__(self, *args):
        if not (isinstance(args[0], torch.Tensor) or isinstance(args[0], np.ndarray)):
            raise Exception(f"Unsupported type {type(args[0])}")
        output = args[0] if isinstance(args[0], torch.Tensor) else torch.from_numpy(args[0])
        pred = torch.argmax(output, dim=0, keepdim=True)
        pred = pred.squeeze(0)
        image_tensor = self.get_color_coded_img(pred, self.n_class)
        buffer = BytesIO()
        image = Image.fromarray(image_tensor.numpy())
        image.save(buffer, format="JPEG")
        return np.array([base64.b64encode(buffer.getvalue())])
