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

from .model_config_pb2 import DataType, ModelConfig, ModelParameter, ModelTensorReshape
from google.protobuf import text_format
from typing import Dict

datatype_mapping = {
    "TYPE_INVALID": DataType.TYPE_INVALID,
    "TYPE_BOOL": DataType.TYPE_BOOL,
    "TYPE_UINT8": DataType.TYPE_UINT8,
    "TYPE_UINT16": DataType.TYPE_UINT16,
    "TYPE_UINT32": DataType.TYPE_UINT32,
    "TYPE_UINT64": DataType.TYPE_UINT64,
    "TYPE_INT8": DataType.TYPE_INT8,
    "TYPE_INT16": DataType.TYPE_INT16,
    "TYPE_INT32": DataType.TYPE_INT32,
    "TYPE_INT64": DataType.TYPE_INT64,
    "TYPE_FP16": DataType.TYPE_FP16,
    "TYPE_FP32": DataType.TYPE_FP32,
    "TYPE_FP64": DataType.TYPE_FP64,
    "TYPE_STRING": DataType.TYPE_STRING,
    "TYPE_BF16": DataType.TYPE_BF16
}

def generate_pbtxt(model_config: Dict, backend:str):
    triton_model_config = ModelConfig()
    for key, value in model_config.items():
        if hasattr(triton_model_config, key):
            # only pick the standard triton model configuration items to the pbtxt
            if key == "input" or key == "output":
                for i in value:
                    entry = triton_model_config.input.add() if key == "input" else triton_model_config.output.add()
                    for k, v in i.items():
                        if hasattr(entry, k):
                            if k == "data_type":
                                setattr(entry, k, datatype_mapping[v])
                            elif k == "dims":
                                for dim in v:
                                    entry.dims.append(dim)
                            elif k == "reshape":
                                if len(v['shape']) == 0:
                                    # WAR: force generating reshape
                                    entry.reshape.shape.append(-1)
                                else:
                                    for s in v['shape']:
                                        entry.reshape.shape.append(s)
                            else:
                                setattr(entry, k, v)
            elif key == "parameters":
                for k, p in value.items():
                    triton_model_config.parameters[k].string_value = p
            elif key == "instance_group":
                for i in value:
                    entry = triton_model_config.instance_group.add()
                    for k, v in i.items():
                        if k == "gpus":
                            entry.gpus.extend(v)
                        else:
                            setattr(entry, k, v)
            elif key == "model_transaction_policy":
                for k, v in value.items():
                    setattr(triton_model_config.model_transaction_policy, k, v)
            else:
                setattr(triton_model_config, key, value)
            # set the triton backend
            setattr(triton_model_config, "backend", backend)
    return text_format.MessageToString(triton_model_config, use_short_repeated_primitives=True).replace('shape: [-1]', 'shape: [ ]')