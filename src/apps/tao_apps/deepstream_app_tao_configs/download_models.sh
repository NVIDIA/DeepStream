#!/bin/sh
################################################################################
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
################################################################################

# Check following part for how to download the TAO models:
# https://docs.nvidia.com/tao/tao-toolkit/text/ds_tao/deepstream_tao_integration.html

echo "==================================================================="
echo "begin download models for vehiclemakenet / vehicletypenet"
echo " / trafficcamnet"
echo "==================================================================="
mkdir -p ../../models/tao_pretrained_models/vehiclemakenet && \
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/vehiclemakenet/pruned_onnx_v1.1.0/files?redirect=true&path=resnet18_pruned.onnx' \
-O ../../models/tao_pretrained_models/vehiclemakenet/resnet18_pruned.onnx
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/vehiclemakenet/pruned_onnx_v1.1.0/files?redirect=true&path=resnet18_pruned_int8.txt' \
-O ../../models/tao_pretrained_models/vehiclemakenet/resnet18_pruned_int8.txt
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/vehiclemakenet/pruned_onnx_v1.1.0/files?redirect=true&path=labels.txt' \
-O ../../models/tao_pretrained_models/vehiclemakenet/labels.txt
mkdir -p ../../models/tao_pretrained_models/vehicletypenet && \
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/vehicletypenet/pruned_onnx_v1.1.0/files?redirect=true&path=resnet18_pruned.onnx' \
-O ../../models/tao_pretrained_models/vehicletypenet/resnet18_pruned.onnx
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/vehicletypenet/pruned_onnx_v1.1.0/files?redirect=true&path=labels.txt' \
-O ../../models/tao_pretrained_models/vehicletypenet/labels.txt
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/vehicletypenet/pruned_onnx_v1.1.0/files?redirect=true&path=resnet18_pruned_int8.txt' \
-O ../../models/tao_pretrained_models/vehicletypenet/resnet18_pruned_int8.txt
mkdir -p ../../models/tao_pretrained_models/trafficcamnet && \
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/trafficcamnet/pruned_onnx_v1.0.4/files?redirect=true&path=labels.txt' \
-O ../../models/tao_pretrained_models/trafficcamnet/labels.txt
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/trafficcamnet/pruned_onnx_v1.0.4/files?redirect=true&path=resnet18_trafficcamnet_pruned.onnx' \
-O ../../models/tao_pretrained_models/trafficcamnet/resnet18_trafficcamnet_pruned.onnx
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/trafficcamnet/pruned_onnx_v1.0.4/files?redirect=true&path=resnet18_trafficcamnet_pruned_int8.txt' \
-O ../../models/tao_pretrained_models/trafficcamnet/resnet18_trafficcamnet_pruned_int8.txt

echo "==================================================================="
echo "begin download models for peopleNet "
echo "==================================================================="
mkdir -p ../../models/tao_pretrained_models/peopleNet/
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/peoplenet/pruned_quantized_decrypted_v2.3.4/files?redirect=true&path=labels.txt' \
-O ../../models/tao_pretrained_models/peopleNet/labels.txt
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/peoplenet/pruned_quantized_decrypted_v2.3.4/files?redirect=true&path=resnet34_peoplenet_int8.onnx' \
-O ../../models/tao_pretrained_models/peopleNet/resnet34_peoplenet_int8.onnx
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/peoplenet/pruned_quantized_decrypted_v2.3.4/files?redirect=true&path=resnet34_peoplenet_int8.txt' \
-O ../../models/tao_pretrained_models/peopleNet/resnet34_peoplenet_int8.txt

echo "==================================================================="
echo "begin download models for peoplenet_transformer_v2 "
echo "==================================================================="
mkdir -p ../../models/tao_pretrained_models/peoplenet_transformer_v2/
wget https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplenet_transformer/versions/deployable_v1.0/files/labels.txt -O ../../models/tao_pretrained_models/peoplenet_transformer_v2/labels.txt
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/peoplenet_transformer_v2/deployable_v1.0/files?redirect=true&path=dino_fan_small_astro_delta.onnx' -O ../../models/tao_pretrained_models/peoplenet_transformer_v2/dino_fan_small_astro_delta.onnx

echo "==================================================================="
echo "Download models successfully "
echo "==================================================================="
