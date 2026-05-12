################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
################################################################################

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
