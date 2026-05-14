#!/usr/bin/env bash

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


set -xe

python3 export_to_onnx.py

mkdir -p /workspace/checkpoints/optimized/nvclip_clipa_vit_h14_700M_vision
/usr/src/tensorrt/bin/trtexec --onnx=/workspace/checkpoints/baseline/nvclip_clipa_vit_h14_700M_vision.onnx --saveEngine=/workspace/checkpoints/optimized/nvclip_clipa_vit_h14_700M_vision/model.plan --optShapes=IMAGE:${MAX_BATCH_SIZE}x3x${INPUT_HEIGHT}x${INPUT_WIDTH} \
 --minShapes=IMAGE:1x3x${INPUT_HEIGHT}x${INPUT_WIDTH} --maxShapes=IMAGE:${MAX_BATCH_SIZE}x3x${INPUT_HEIGHT}x${INPUT_WIDTH} --fp16

mkdir -p /workspace/checkpoints/optimized/nvclip_clipa_vit_h14_700M_text
/usr/src/tensorrt/bin/trtexec --onnx=/workspace/checkpoints/baseline/nvclip_clipa_vit_h14_700M_text.onnx --saveEngine=/workspace/checkpoints/optimized/nvclip_clipa_vit_h14_700M_text/model.plan --optShapes=TEXT:${MAX_BATCH_SIZE}x${TEXT_LENGTH} \
 --minShapes=TEXT:1x${TEXT_LENGTH} --maxShapes=TEXT:${MAX_BATCH_SIZE}x${TEXT_LENGTH} --fp16


