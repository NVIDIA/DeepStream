#!/bin/bash
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

# This script only worked on the Quantization model, otherwise, errors would throw up (Origin model, etc.)
weight=$1
prefix=${weight%.*}
onnx=${prefix}.onnx
graph=${prefix}.graph
engine=${prefix}.engine

# onnx must be 672x672 of input
python scripts/qat.py export $weight --dynamic --save=$onnx --size=672

# To obtain more QPS can add --fp16 flag for detect layer
trtexec --onnx=$onnx \
    --saveEngine=${engine} --int8 --buildOnly --memPoolSize=workspace:1024MiB \
    --dumpLayerInfo --exportLayerInfo=${graph} --profilingVerbosity=detailed

python scripts/draw-engine.py ${graph}
python scripts/eval-trt.py --engine=${engine}
