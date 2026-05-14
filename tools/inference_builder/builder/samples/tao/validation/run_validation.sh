#!/bin/bash

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

set -e  # Exit on any error
# Validate NIM_MODEL_NAME
valid_models=("rtdetr" "cls" "seg" "gdino" "mgdino")
if [[ ! " ${valid_models[@]} " =~ " ${NIM_MODEL_NAME} " ]]; then
    echo "Error: NIM_MODEL_NAME must be one of: ${valid_models[*]}"
    exit 1
fi

# Change to the appropriate test directory
cd "/app/validation/${NIM_MODEL_NAME}/.tmp" || exit 1
echo $(pwd)
# Run the test
python test_runner.py
