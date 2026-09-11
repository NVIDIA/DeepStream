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

# Get the directory of the script
script_dir=$(dirname "$(readlink -f "$0")")

# Specify the folder containing the test files
test_folder="$script_dir/tests"

# Iterate over all Python files prefixed with "test" in the test folder
for test_file in "$test_folder"/test_*.py; do
    echo "Running tests in $test_file"
    pytest "$test_file"
    echo "--------------------------------------------------"
done