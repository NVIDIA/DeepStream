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
nvcv compatibility module for CVCUDA
This module provides compatibility for nvcv API using cvcuda
"""
import cvcuda
import numpy as np

# Tensor class - map to cvcuda.Tensor for isinstance checks
# Use cvcuda.Tensor as the actual type
Tensor = cvcuda.Tensor if hasattr(cvcuda, 'Tensor') else type(None)

# Type constants - map to cvcuda.Type
class Type:
    U8 = getattr(cvcuda.Type, 'U8', 'uint8') if hasattr(cvcuda, 'Type') else "uint8"
    U16 = getattr(cvcuda.Type, 'U16', 'uint16') if hasattr(cvcuda, 'Type') else "uint16"
    F32 = getattr(cvcuda.Type, 'F32', 'float32') if hasattr(cvcuda, 'Type') else "float32"
    F64 = getattr(cvcuda.Type, 'F64', 'float64') if hasattr(cvcuda, 'Type') else "float64"

# TensorLayout constants - map to cvcuda.TensorLayout
class TensorLayout:
    NHWC = getattr(cvcuda.TensorLayout, 'NHWC', 'NHWC') if hasattr(cvcuda, 'TensorLayout') else "NHWC"
    NCHW = getattr(cvcuda.TensorLayout, 'NCHW', 'NCHW') if hasattr(cvcuda, 'TensorLayout') else "NCHW"
    HWC = "HWC"
    CHW = "CHW"

# Format constants - map to cvcuda.Format
class Format:
    U8 = getattr(cvcuda.Format, 'U8', 'U8') if hasattr(cvcuda, 'Format') else "U8"
    RGB8 = getattr(cvcuda.Format, 'RGB8', 'RGB8') if hasattr(cvcuda, 'Format') else "RGB8"
    BGR8 = getattr(cvcuda.Format, 'BGR8', 'BGR8') if hasattr(cvcuda, 'Format') else "BGR8"

def as_tensor(data, layout=None):
    """Convert data to tensor, compatible with nvcv.as_tensor"""
    if hasattr(cvcuda, 'as_tensor'):
        if layout:
            return cvcuda.as_tensor(data, layout=layout)
        return cvcuda.as_tensor(data)
    # Fallback: return data as numpy array or tensor
    if isinstance(data, np.ndarray):
        return data
    return np.asarray(data)

def as_image(data, format=None):
    """Convert data to image, compatible with nvcv.as_image"""
    # For compatibility, return data as-is or convert using cvcuda
    if hasattr(cvcuda, 'as_image'):
        return cvcuda.as_image(data)
    # Fallback: return data
    if isinstance(data, np.ndarray):
        return data
    return np.asarray(data)
