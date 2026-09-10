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
from abc import ABC, abstractmethod
from typing import Any, List, Dict

class PostProcessing(ABC):
    """Base tensor output post-processing class"""

    @abstractmethod
    def __call__(self, output_layers: Dict) -> Any:
        """
        Convert output tensors to real world representation.

        Args:
            output_layers: direct output layers from the model, iterables in format of (name, tensor)
        """

class ObjectDetectorOutputConverter(PostProcessing):
    """Base tensor output post-processing for object detection model"""
    @abstractmethod
    def __call__(self, output_layers: Dict) -> List:
        """
        Convert output tensors to object detection results

        Args:
            output_layers: direct output layers from the model, dictionary of (name, tensor)
        return:
            List of BBox tensor ``(class_id, confidence, x1, y1, x2, y2)``
        """
