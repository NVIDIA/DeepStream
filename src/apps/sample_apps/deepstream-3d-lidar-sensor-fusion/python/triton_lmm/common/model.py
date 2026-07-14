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
from typing import List

# triton model repo extra
logger = logging.getLogger(__name__)

INPUT_IMAGE = "input_image"
INPUT_LIDAR = "input_lidar"
INPUT_PROMPT = "input_prompt"

OUTPUT_IMAGE = "output_image"
OUTPUT_3D_BBOX = "output_3d_bbox"


class IModel:
    """ Model interface """

    def __init__(self, name_: str, model=None) -> None:
        self.name: str = name_ if name_ else ""
        # assert model
        self._model = model

    def start(self, **kwargs):
        pass

    def stop(self):
        pass

    def _infer(self, **inputs):
        return self._model(inputs)

    def bind_model(self, triton):
        pass


class ModeList:
    """ Model List Management """

    def __init__(self) -> None:
        self._models: List[IModel] = []

    def append(self, model) -> bool:
        if not isinstance(model, IModel):
            return False
        self._models.append(model)
        return True

    def start_models(self, triton, **kwargs) -> bool:
        for model in self._models:
            model.start(**kwargs)
            if triton:
                model.bind_model(triton)
        return True

    def stop_models(self) -> bool:
        for model in self._models:
            model.stop()
        return True
