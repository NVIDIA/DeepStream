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

# Example script to build Jetson triton docker on x86 Linux PC (cross compile)

cd jetson_dockerfiles

# Jetson triton

sudo docker build --platform linux/arm64 --network host --progress=plain -t deepstream-l4t:9.1.0-triton-local -f Dockerfile_Jetson_Devel ..

# Jetson samples

# sudo docker build --platform linux/arm64 --network host --progress=plain -t deepstream-l4t:9.1.0-samples-local -f Dockerfile_Jetson_Run ..

cd ..
