# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#################################################################

# Example script to build x86 docker (currently building the x86 triton docker).

cd x86_dockerfiles

# build x86 triton

sudo docker build --network host --progress=plain --build-arg DS_DIR=/opt/nvidia/deepstream/deepstream-9.1 -t deepstream:9.1.0-triton-local -f Dockerfile_triton_x86 ..

# x86-samples

# sudo docker build --network host --progress=plain -t deepstream:9.1.0-samples-local -f Dockerfile_samples_x86 ..

cd ..
