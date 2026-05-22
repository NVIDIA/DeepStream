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

#This script is to download peoplenet pretrained model and run peoplenet DeepStream application

echo "==================================================================="
echo "begin download models for peopleNet "
echo "==================================================================="
mkdir -p ../../models/tao_pretrained_models/peopleNet/
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/peoplenet/pruned_quantized_decrypted_v2.3.4/files?redirect=true&path=labels.txt' \
-O ../../models/tao_pretrained_models/peopleNet/labels.txt
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/peoplenet/pruned_quantized_decrypted_v2.3.4/files?redirect=true&path=resnet34_peoplenet_int8.onnx' \
-O ../../models/tao_pretrained_models/peopleNet/resnet34_peoplenet_int8.onnx

echo "==================================================================="
echo "Run deepstream-app"
echo "==================================================================="
deepstream-app -c deepstream_app_source1_peoplenet.txt
