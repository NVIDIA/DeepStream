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

#!/bin/bash

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
