# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import io
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode

class OpenclipTokenizer:
    name = "openclip-tokenizer"
    def __init__(self, config):
        import open_clip
        open_clip.add_model_config(config["model_home"])
        self._tokenizer = open_clip.get_tokenizer("NVCLIP_224_700M_ViTH14")

    def __call__(self, *args):
        return self._tokenizer(args[0])

class VisionPreprocessor:
    name = "nvclip-vision-preprocessor"
    def __init__(self, config):
        self._transform = Compose(
            [
                Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
                CenterCrop(224),
                lambda image: image.convert('RGB'),
                ToTensor(),
                Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    def __call__(self, *args):
        return self._transform(Image.open(io.BytesIO(args[0])))

class NvClipPostProcessor:
    name = "nvclip-postprocessor"
    def __init__(self, config):
        self._config = config

    def __call__(self, *args):
        text = args[0].tolist()
        images = args[1].tolist()
        indices = args[2].tolist()
        total_tokens = sum(len(s) for s in text)
        num_images = len(images)
        embeddings = []
        for index in indices:
            if index == "text":
                embeddings.append(text.pop(0))
            elif index == "image":
                embeddings.append(images.pop(0))
        return np.array(embeddings), np.array(total_tokens), np.array(num_images)


