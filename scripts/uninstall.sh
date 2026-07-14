#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2020-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

TARGET_DEVICE=$(uname -m)
OS=$(cat /etc/os-release | awk -F= '$1=="ID"{print $2}' | sed 's/"//g')

if [ "${TARGET_DEVICE}" = "x86_64" ]; then
    if [ "${OS}" = "rhel" ]; then
        BASE_LIB_DIR="/usr/lib64/"
    elif [ "${OS}" = "ubuntu" ]; then
        BASE_LIB_DIR="/usr/lib/x86_64-linux-gnu/"
    else
        echo "Unsupported OS" 2>&1
        exit 1
    fi
fi

# DeepStream version to remove. Override via the first argument or the
# PREV_DS_VER environment variable; defaults to 9.1.
#   sudo ./uninstall.sh 9.0
#   sudo PREV_DS_VER=9.0 ./uninstall.sh
PREV_DS_VER="${1:-${PREV_DS_VER:-9.1}}"
if [ -z "${PREV_DS_VER}" ]; then
  echo "PREV_DS_VER not set (usage: $0 [version], e.g. $0 9.1)"
  exit 1
fi

if [ ! -d /opt/nvidia/deepstream/deepstream-${PREV_DS_VER} ]; then
    echo "This version of DeepStream is not present in the system."
    exit 1
fi

if [ -n "$TARGET_DEVICE" ]; then
  if [ "${TARGET_DEVICE}" = "x86_64" ]; then
    update-alternatives --remove deepstream-v4l2plugin /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/lib/libv4l/plugins/libcuvidv4l2_plugin.so
    update-alternatives --remove deepstream-v4l2library /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/lib/libnvv4l2.so
    update-alternatives --remove deepstream-v4lconvert /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/lib/libnvv4lconvert.so
    update-alternatives --remove deepstream-appsrc-cuda-test /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-appsrc-cuda-test
    update-alternatives --remove deepstream-multigpu-nvlink-test /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-multigpu-nvlink-test
    update-alternatives --remove deepstream-ucx-test-app /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-ucx-test-app
    rm -rf $BASE_LIB_DIR/libv4lconvert.so.0.0.99999
    rm -rf $BASE_LIB_DIR/libv4l2.so.0.0.99999
    rm -f /etc/ld.so.conf.d/deepstream.conf
  elif [ "${TARGET_DEVICE}" = "aarch64" ]; then
    # deepstream-ipc-test is Jetson-only; not installed on SBSA (both report uname -m as aarch64)
    if [ -f /etc/nv_tegra_release ]; then
      update-alternatives --remove deepstream-ipc-test-app /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-ipc-test-app
    fi
  fi
fi
update-alternatives --remove deepstream-plugins /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/lib/gst-plugins
update-alternatives --remove deepstream-app /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-app
update-alternatives --remove deepstream-test1-app /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-test1-app
update-alternatives --remove deepstream-test2-app /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-test2-app
update-alternatives --remove deepstream-test3-app /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-test3-app
update-alternatives --remove deepstream-test4-app /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-test4-app
update-alternatives --remove deepstream-test5-app /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-test5-app
update-alternatives --remove deepstream-testsr-app /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-testsr-app
update-alternatives --remove deepstream-transfer-learning-app /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-transfer-learning-app
update-alternatives --remove deepstream-dynamicsrcbin-test /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-dynamicsrcbin-test
update-alternatives --remove deepstream-user-metadata-app /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-user-metadata-app
update-alternatives --remove deepstream-dewarper-app /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-dewarper-app
update-alternatives --remove deepstream-nvof-app /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-nvof-app
update-alternatives --remove deepstream-image-decode-app /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-image-decode-app
update-alternatives --remove deepstream-gst-metadata-app /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-gst-metadata-app
update-alternatives --remove deepstream-opencv-test /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-opencv-test
update-alternatives --remove deepstream-preprocess-test /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-preprocess-test
update-alternatives --remove deepstream-image-meta-test /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-image-meta-test
update-alternatives --remove deepstream-appsrc-test /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-appsrc-test
update-alternatives --remove deepstream-can-orientation-app /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-can-orientation-app
update-alternatives --remove deepstream-nvdsanalytics-test /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-nvdsanalytics-test
update-alternatives --remove deepstream-3d-action-recognition /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-3d-action-recognition
update-alternatives --remove deepstream-3d-depth-camera /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-3d-depth-camera
update-alternatives --remove deepstream-lidar-inference-app /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-lidar-inference-app
update-alternatives --remove deepstream-3d-lidar-sensor-fusion /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-3d-lidar-sensor-fusion
update-alternatives --remove deepstream-nmos-app /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-nmos-app
update-alternatives --remove deepstream-server-app /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-server-app
update-alternatives --remove deepstream-demuxer-static /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-demuxer-static
update-alternatives --remove deepstream-demuxer-dynamic /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-demuxer-dynamic
update-alternatives --remove service-maker-3d-action-recognition-app /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/service-maker-3d-action-recognition-app
update-alternatives --remove service-maker-appsrc-test /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/service-maker-appsrc-test
update-alternatives --remove service-maker-sr-test-app /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/service-maker-sr-test-app
update-alternatives --remove service-maker-test1-app /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/service-maker-test1-app
update-alternatives --remove service-maker-test2-app /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/service-maker-test2-app
update-alternatives --remove service-maker-test3-app /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/service-maker-test3-app
update-alternatives --remove service-maker-test4-app /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/service-maker-test4-app
update-alternatives --remove service-maker-test5-app /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/service-maker-test5-app
pip uninstall pyservicemaker -y --break-system-packages

# Delete the installed files first, then deregister the Debian packages. Doing
# the rm -rf up front leaves dpkg nothing to complain about (e.g. "directory not
# empty so not removed"); the dpkg -r step afterward just clears the package
# registration from the database. Tarball installs are not dpkg-registered, so
# for them only the rm -rf below applies.
rm -rf /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/
rm -rf /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/lib/
rm -rf /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/samples/
rm -rf /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/sources/
rm -rf /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/LicenseAgreement.pdf
rm -rf /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/service-maker/

# Deregister artifacts that were installed via Debian packages
# (INSTALL_METHOD=deb). Only deb installs are recorded in the dpkg database.
if [ "${TARGET_DEVICE}" = "x86_64" ]; then
  PLATFORM=x86
elif [ "${TARGET_DEVICE}" = "aarch64" ]; then
  if [ -f /etc/nv_tegra_release ]; then
    PLATFORM=aarch64
  else
    PLATFORM=sbsa
  fi
fi

remove_if_deb() {
  pkg=$1
  if dpkg-query -W -f='${Status}' "$pkg" 2>/dev/null | grep -q "install ok installed"; then
    echo "Removing Debian package: $pkg"
    dpkg -r "$pkg" || apt-get remove -y "$pkg" || true
  fi
}

remove_if_deb "deepstream-sample-data"
if [ -n "${PLATFORM}" ]; then
  remove_if_deb "deepstream-binaries-${PLATFORM}"
fi

# Remove the base version symlink created during install
# (/opt/nvidia/deepstream/deepstream -> deepstream-${PREV_DS_VER}), but only if
# it still points at the version we are uninstalling, so other installed
# versions are left untouched. The sources/includes and samples/trtis_model_repo
# symlinks are already removed by the rm -rf of sources/ and samples/ above.
DS_BASE_LINK=/opt/nvidia/deepstream/deepstream
if [ -L "${DS_BASE_LINK}" ] && [ "$(readlink "${DS_BASE_LINK}")" = "deepstream-${PREV_DS_VER}" ]; then
  rm -f "${DS_BASE_LINK}"
fi

ldconfig
rm -rf /home/*/.cache/gstreamer-1.0/
rm -rf /root/.cache/gstreamer-1.0/

# Remove the DeepStream install directory entirely, then the base
# /opt/nvidia/deepstream directory if no other DeepStream versions remain, so the
# system returns to its pre-install state. rmdir only deletes the base directory
# when it is empty, protecting any other installed versions.
rm -rf /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}
rmdir /opt/nvidia/deepstream 2>/dev/null || true
