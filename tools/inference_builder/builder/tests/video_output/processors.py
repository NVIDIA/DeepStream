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

import logging

import torch

logger = logging.getLogger(__name__)


class VideoOutputFrameGeneratorProcessor:
    name = "video-output-frame-generator"

    def __init__(self, config):
        self.num_frames = config["num_frames"]
        self.height = config["height"]
        self.width = config["width"]

    def __call__(self, *args, **kwargs):
        if len(args) != 1:
            raise ValueError("VideoOutputFrameGeneratorProcessor expects exactly one argument")

        seed = int(torch.as_tensor(args[0]).flatten()[0].item())
        y = torch.arange(self.height, dtype=torch.int16).view(self.height, 1)
        x = torch.arange(self.width, dtype=torch.int16).view(1, self.width)

        frames = []
        for index in range(self.num_frames):
            frame = torch.empty((self.height, self.width, 3), dtype=torch.int16)
            frame[..., 0] = (x + seed + index * 17) % 256
            frame[..., 1] = (y + seed + index * 29) % 256
            frame[..., 2] = (x // 2 + y // 2 + seed + index * 11) % 256
            frames.append(frame.to(torch.uint8))

        logger.info(
            "VideoOutputFrameGeneratorProcessor: generated %d frames at %dx%d",
            len(frames),
            self.width,
            self.height,
        )

        return torch.tensor([float(seed)], dtype=torch.float32), frames, frames
