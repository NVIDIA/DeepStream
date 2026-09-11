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

import torch
import logging

logger = logging.getLogger(__name__)

class FramePickerProcessor:
    name = "frame-picker"

    def __init__(self, config):
        self.num_frames = config['num_frames']
        # Eagerly initialize CUDA to avoid GIL deadlock with GStreamer threads
        torch.cuda.init()

    def __call__(self, *args, **kwargs):
        if len(args) != 1:
            raise ValueError(
                "FramePickerProcessor expects exactly one argument"
            )
        frame_list = args[0]
        if len(frame_list) != self.num_frames:
            raise ValueError(
                f"FramePickerProcessor expects a list of "
                f"{self.num_frames} frames, while got {len(frame_list)}"
            )
        torch_tensors = [
            torch.utils.dlpack.from_dlpack(frame.tensor)
            for frame in frame_list
        ]

        # Log frame dimensions for verification
        if torch_tensors:
            # Tensor shape is typically (H, W, C) from MediaExtractor
            h, w = torch_tensors[0].shape[0], torch_tensors[0].shape[1]
            logger.info(f"FramePickerProcessor: received {len(torch_tensors)} frames with dimensions {w}x{h}")

        return torch.stack(torch_tensors)
