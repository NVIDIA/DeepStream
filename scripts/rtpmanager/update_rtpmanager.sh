#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

export DEBIAN_FRONTEND=noninteractive

# Directory containing this script (and the patch files alongside it)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -d "/tmp/gst-1.24.2/" ]]; then
    rm -rf /tmp/gst-1.24.2/
fi
mkdir -p /tmp/gst-1.24.2
pushd /tmp/gst-1.24.2

#Install prerequisites
apt-get -y install python3-pip python3-venv
python3 -m venv rtppython
source ./rtppython/bin/activate
pip3 install meson
pip3 install ninja
pip3 install setuptools==73.0.0
apt-get -y install libmount-dev
apt-get -y install flex
apt-get -y install flex bison
apt-get -y install libharfbuzz-dev
apt-get -y install libpango1.0-dev


#Build gstreamer and place the rtpmanager library at the required path.
git clone https://gitlab.freedesktop.org/gstreamer/gstreamer.git -b 1.24.2
pushd gstreamer
#Place rtpjitterbuffer_eos_handling.patch at the appropriate location
git apply "$SCRIPT_DIR/rtpjitterbuffer_eos_handling.patch"
git apply "$SCRIPT_DIR/0001-rtspsrc-Fixes-related-to-slow-rtsp-inputs.patch"
git apply "$SCRIPT_DIR/0002-rtspsrc-Resolve-hang-delete-rtsp-when-stream-ends-bug-5939802.patch"
git apply "$SCRIPT_DIR/h264parser-Avoid-meta-parsing-conditionally.patch"
#Build gstreamer 1.24.2 with the applied fix patch
meson build --buildtype=release -Dbad=disabled -Dugly=disabled -Dexamples=disabled -Dlibav=disabled -Ddevtools=disabled -Dintrospection=disabled
ninja -C build/
#Place the rtpmanager library at the gstreamer installation path
if [[ $(uname -m) == "aarch64" ]]; then
    cp build/subprojects/gst-plugins-good/gst/rtpmanager/libgstrtpmanager.so /usr/lib/aarch64-linux-gnu/gstreamer-1.0
    cp build/subprojects/gst-plugins-good/gst/rtsp/libgstrtsp.so /usr/lib/aarch64-linux-gnu/gstreamer-1.0
else
    cp build/subprojects/gst-plugins-good/gst/rtpmanager/libgstrtpmanager.so /usr/lib/x86_64-linux-gnu/gstreamer-1.0
    cp build/subprojects/gst-plugins-good/gst/rtsp/libgstrtsp.so /usr/lib/x86_64-linux-gnu/gstreamer-1.0/
fi

################################################################################
# Build gst-plugins-bad videoparsers and place the library at the required path
################################################################################
echo "Building gst-plugins-bad videoparsers (libgstvideoparsersbad.so)..."
pushd subprojects/gst-plugins-bad

# Setup build directory for gst-plugins-bad
meson setup builddir --prefix=/tmp/gst-install

# Build only the videoparsers plugin (includes h264parse with custom modifications)
ninja -j4 -C builddir gst/videoparsers/libgstvideoparsersbad.so

# Copy the videoparsersbad library to the gstreamer installation path
if [[ $(uname -m) == "aarch64" ]]; then
    echo "Copying libgstvideoparsersbad.so to /usr/lib/aarch64-linux-gnu/gstreamer-1.0/"
    cp builddir/gst/videoparsers/libgstvideoparsersbad.so /usr/lib/aarch64-linux-gnu/gstreamer-1.0/
else
    echo "Copying libgstvideoparsersbad.so to /usr/lib/x86_64-linux-gnu/gstreamer-1.0/"
    cp builddir/gst/videoparsers/libgstvideoparsersbad.so /usr/lib/x86_64-linux-gnu/gstreamer-1.0/
fi

echo "libgstvideoparsersbad.so build and installation completed successfully!"

popd
################################################################################

popd
popd
# Deactivate virtual environment if the function exists
type deactivate &>/dev/null && deactivate
rm -rf /tmp/gst-1.24.2
