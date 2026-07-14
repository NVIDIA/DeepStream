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

set -e

export PATH=$PATH:/usr/src/tensorrt/bin

MODEL_NAME=pointpillars_deployable

mkdir -p models/pointpillars/1/

wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/pointpillarnet/deployable_v1.1/files?redirect=true&path=pointpillars_deployable.onnx' -O ./models/pointpillars/1/pointpillars_deployable.onnx

trtexec --onnx=./models/pointpillars/1/${MODEL_NAME}.onnx \
 --saveEngine=./models/pointpillars/1/${MODEL_NAME}.engine \
 --minShapes=points:1x204800x4,num_points:1 \
 --optShapes=points:1x204800x4,num_points:1 \
 --maxShapes=points:1x204800x4,num_points:1 \
 --dumpLayerInfo --exportLayerInfo=./models/pointpillars/1/${MODEL_NAME}.layer.json \
 --verbose > ./models/pointpillars/1/${MODEL_NAME}.log 2>&1
