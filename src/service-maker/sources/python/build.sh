#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2017-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Function to display usage
usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --minor-ver <version>  Python minor version (e.g., 10, 12) - optional, defaults to using system's python3"
    echo "  --output-dir <path>     Output directory for wheel file - optional, defaults to ./"
    echo "  --arch <architecture>   Target architecture (x86_64 or aarch64) - optional, defaults to native"
    echo "  --help                  Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Use python3, output to ./output, native arch"
    echo "  $0 --minor-ver 9                      # Use python3.9, output to ./output"
    echo "  $0 --output-dir /tmp/build            # Use python3, output to /tmp/build"
    echo "  $0 --arch aarch64                     # Cross-compile for aarch64"
    echo "  $0 --minor-ver 9 --output-dir /tmp/build --arch aarch64  # Cross-compile for aarch64 with python3.9"
    exit 1
}

# Default values
PYTHON_VER="python3"
OUTPUT_DIR="./"
TARGET_ARCH=""

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --minor-ver)
            if [[ -z "$2" || "$2" =~ ^-- ]]; then
                echo "Error: --minor-ver requires a value"
                usage
            fi
            PYTHON_VER="python3.$2"
            shift 2
            ;;
        --output-dir)
            if [[ -z "$2" || "$2" =~ ^-- ]]; then
                echo "Error: --output-dir requires a value"
                usage
            fi
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --arch)
            if [[ -z "$2" || "$2" =~ ^-- ]]; then
                echo "Error: --arch requires a value"
                usage
            fi
            if [[ "$2" != "x86_64" && "$2" != "aarch64" ]]; then
                echo "Error: --arch must be either x86_64 or aarch64"
                usage
            fi
            TARGET_ARCH="$2"
            shift 2
            ;;
        --help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

OUTPUT_DIR=$(realpath $OUTPUT_DIR)

SCRIPT_DIR=$(dirname $(realpath $0))

# pybind11 and dlpack are installed as headers by
# scripts/install_opensource_deps.sh; override to build against copies
# installed elsewhere.
PYBIND11_ROOT=${PYBIND11_ROOT:-/opt/pybind11}
DLPACK_ROOT=${DLPACK_ROOT:-/opt/dlpack}

# Absolute paths: python -m build copies the tree into a temporary sdist
# directory and builds the wheel from there, so anything resolved relative to
# CMAKE_CURRENT_SOURCE_DIR would no longer reach the repository.
SM_INCLUDE_DIR=$(realpath $SCRIPT_DIR/../../includes)
DS_INCLUDE_DIR=$(realpath $SCRIPT_DIR/../../../../includes)

export CMAKE_ARGS="-DPYBIND11_ROOT=$PYBIND11_ROOT -DDLPACK_ROOT=$DLPACK_ROOT \
-DSM_INCLUDE_DIR=$SM_INCLUDE_DIR -DDS_INCLUDE_DIR=$DS_INCLUDE_DIR"

# Set up cross-compilation for aarch64 if specified
if [[ "$TARGET_ARCH" == "aarch64" ]]; then
    echo "Setting up cross-compilation for aarch64..."

    # Create CMake toolchain file for aarch64
    TOOLCHAIN_FILE="$SCRIPT_DIR/aarch64-toolchain.cmake"
    cat > "$TOOLCHAIN_FILE" << 'EOF'
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# Specify the cross compiler
set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

# Where to look for the target environment
set(CMAKE_FIND_ROOT_PATH /usr/aarch64-linux-gnu)

# Search for programs in the build host directories
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)

# Search for libraries and headers in the target directories
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
EOF

    export CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_TOOLCHAIN_FILE=$TOOLCHAIN_FILE"

    # Set platform tag for Python wheel
    export _PYTHON_HOST_PLATFORM="linux-aarch64"
    export ARCHFLAGS="-arch aarch64"

    echo "Cross-compilation toolchain configured for aarch64"
    echo "Note: Make sure you have aarch64-linux-gnu-gcc and aarch64-linux-gnu-g++ installed"
    echo "      Install with: sudo apt-get install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu"
fi

echo "PYTHON_VER: $PYTHON_VER"
echo "PYBIND11_ROOT: $PYBIND11_ROOT"
echo "DLPACK_ROOT: $DLPACK_ROOT"
echo "TARGET_ARCH: ${TARGET_ARCH:-native}"
echo "CMAKE_ARGS: $CMAKE_ARGS"
cd $SCRIPT_DIR/src

$PYTHON_VER -m build

cp dist/pyservicemaker*.whl "$OUTPUT_DIR/"

rm -rf dist/
rm -rf *.egg-info
cd $OUTPUT_DIR

# Clean up toolchain file if it was created
if [[ -f "$SCRIPT_DIR/aarch64-toolchain.cmake" ]]; then
    rm -f "$SCRIPT_DIR/aarch64-toolchain.cmake"
fi

WHL_FILE=$(ls pyservicemaker*.whl)

echo "Wheel file generated at $OUTPUT_DIR/$WHL_FILE"
