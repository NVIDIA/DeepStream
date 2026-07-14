#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# User script to download and install PyDS from https://github.com/NVIDIA-AI-IOT/deepstream_python_apps
# -v option can be used to specify which version of PyDS to download and install
# -b option can be used to indicate that latest available bindings should be downloaded and installed

function usage() {
  echo "Usage: $0 -v PyDS-version -b (build-latest-bindings)"
  echo "  -v, --version            The version of PyDS to download and install"
  echo "  OR  "
  echo "  -b, --build-bindings     Compile and install latest PyDS instead of downloading wheels"
  echo "  -r, --remote-branch      Specify which branch of deepstream_python_apps to clone and build from"
  echo ""
  echo "Example: $0 --version 1.1.4"
  echo "Example: $0 --build-bindings"
  echo "Example: $0 --build-bindings -r master"
  exit 1
}

while [[ "$#" > 0 ]]; do
    case $1 in
        -v|--version) version="$2"; shift 2;;
        -r|--remote-branch) remote_branch="$2"; shift 2;;
        -b|--build-bindings) build_bindings=1; shift 1;;
        -h|--help) usage; shift 1;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
done

if [ -z "$version" ] && [ -z "$build_bindings" ]; then usage "The version of PyDS to download and install"; fi;

cd /opt/nvidia/deepstream/deepstream
echo "####################################"
echo "Downloading necessary pre-requisites"
echo "####################################"
apt-get update

apt install -y python3-gi python3-dev python3-gst-1.0 python-gi-dev git meson \
    python3 python3-pip python3-venv cmake g++ build-essential libglib2.0-dev \
    libglib2.0-dev-bin libgstreamer1.0-dev libtool m4 autoconf automake libgirepository-2.0-dev libcairo2-dev

cd /opt/nvidia/deepstream/deepstream/sources
if [ -z "$remote_branch" ]
then
    remote_branch="master"
    echo "#################################"
    echo "Default sync branch set to master"
    echo "#################################"
fi

git clone -b "$remote_branch" https://github.com/NVIDIA-AI-IOT/deepstream_python_apps.git
if [ $? -eq 0 ]; then
   echo "deepstream_python_apps cloned successfully from branch $remote_branch"
else
   echo "deepstream_python_apps clone from branch $remote_branch FAILED! Exiting..."
   exit 1
fi

cd /opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps/
python3 -m venv pyds
source ./pyds/bin/activate

pip3 install build
pip3 install PyGObject
pip3 install cuda-python

if [ -z "$version" ] && [ $build_bindings == 1 ]
then
    echo "############################"
    echo "Building downloaded bindings"
    echo "############################"
    cd /opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps/bindings
    git submodule update --init
    python3 3rdparty/git-partial-submodule/git-partial-submodule.py restore-sparse
    cd 3rdparty/gstreamer/subprojects/gst-python/
    meson setup build
    cd build
    ninja
    ninja install
    cd /opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps/bindings
    export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc)
    python3 -m build
    cd dist/
    echo "###########################"
    echo "Installing built PyDS wheel"
    echo "###########################"
    pip3 install ./pyds-1*_aarch64.whl
elif [ -z "$build_bindings" ] && [[ ! -z $version ]]
then
    echo "##############################"
    echo "Pulling PyDS version: $version"
    echo "##############################"
    cd /opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps
    URL="https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/download/v$version/pyds-$version-cp312-cp312-linux_aarch64.whl"
    echo "url"
    echo $URL
    wget "$URL"
    if [ -f "pyds-$version-cp312-cp312-linux_aarch64.whl" ]
    then
        echo "########################################################"
        echo "Downloaded wheel pyds-$version-cp312-cp312-linux_aarch64.whl"
        echo "########################################################"
    else
        echo "#########################################"
        echo "PyDS wheel was not downloaded. Exiting..."
        echo "#########################################"
        exit 1
    fi
    echo "#####################"
    echo "Installing PyDS wheel"
    echo "#####################"
    pip3 install pyds-$version-cp312-cp312-linux_aarch64.whl
fi
