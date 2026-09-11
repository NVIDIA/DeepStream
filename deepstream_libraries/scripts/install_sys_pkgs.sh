#!/bin/bash -e

# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# This script installs all the dependencies required to run the CVCUDA samples.
# It uses the /tmp folder to download temporary data and libraries.

# SCRIPT_DIR is the directory where this script is located.
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

set -e  # Exit script if any command fails

# Move to parent directory
if [ ! -d "assets" ]; then
    echo "Moving to parent directory."
    cd ..
fi

# Install basic packages first.
cd /tmp
apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    yasm \
    unzip \
    cmake \
    git \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Add repositories and install g++
add-apt-repository -y ppa:ubuntu-toolchain-r/test
apt-get update && apt-get install -y --no-install-recommends \
    gcc-12 g++-12 \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 12
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 12
update-alternatives --set gcc /usr/bin/gcc-12
update-alternatives --set g++ /usr/bin/g++-12

# Install Python and gtest
apt-get update && apt-get install -y --no-install-recommends \
    libgtest-dev \
    libgmock-dev \
    python3-pip \
    ninja-build ccache \
    mlocate && updatedb \
    && rm -rf /var/lib/apt/lists/*

# Install ffmpeg and other libraries needed for VPF and av package.
# Note: We are not installing either libnv-encode or decode libraries here.
apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libavfilter-dev \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    pkg-config \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install libssl 1.1.1
cd /tmp
wget http://archive.ubuntu.com/ubuntu/pool/main/o/openssl/libssl1.1_1.1.0g-2ubuntu4_amd64.deb
dpkg -i libssl1.1_1.1.0g-2ubuntu4_amd64.deb

# Install NVIDIA NSIGHT 2024.6.1
cd /tmp
wget https://developer.download.nvidia.com/devtools/nsight-systems/nsight-systems-2024.6.1_2024.6.1.90-1_amd64.deb
apt-get update && apt-get install -y \
    libsm6 \
    libxrender1 \
    libfontconfig1 \
    libxext6 \
    libx11-dev \
    libxkbfile-dev \
    ./nsight-systems-2024.6.1_2024.6.1.90-1_amd64.deb \
    && rm -rf /var/lib/apt/lists/*

# Install venv and python3.12-venv
apt-get update && apt-get install -y --no-install-recommends \
    python3-venv \
    python3.12-venv \
    && rm -rf /var/lib/apt/lists/*