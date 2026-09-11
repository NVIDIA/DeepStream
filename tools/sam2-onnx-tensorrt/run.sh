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

pip3 install -e .
cd checkpoints; bash download_ckpts.sh; cd ..

# --model can be chosen from tiny, small, base_plus, large
MODEL_TYPE="large"
mkdir -p checkpoints/${MODEL_TYPE}
python3 export_sam2_onnx.py --model ${MODEL_TYPE}
echo "SAM2 ONNX models are exported to checkpoints/${MODEL_TYPE}"

# Copy models to DeepStream tracker path
mkdir -p /opt/nvidia/deepstream/deepstream/samples/models/Tracker/
cp checkpoints/${MODEL_TYPE}/*.onnx /opt/nvidia/deepstream/deepstream/samples/models/Tracker/
echo "SAM2 ONNX models are copied to /opt/nvidia/deepstream/deepstream/samples/models/Tracker/"
