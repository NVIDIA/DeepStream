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

set -e


# Remove only this SDK's registered target. Older releases used a different
# group for the same UCX executable; do not delete that group or its other targets.
remove_alternative() {
    local name=$1 target=$2 group info
    local groups=("$name")
    if [[ "$name" == deepstream-ucx-test-app ]]; then
        groups+=(deepstream-ucx-test-app-client)
    fi
    for group in "${groups[@]}"; do
        info=$(LC_ALL=C update-alternatives --query "$group" 2>/dev/null || true)
        if printf '%s\n' "$info" | grep -Fxq "Alternative: $target"; then
            update-alternatives --remove "$group" "$target"
        fi
    done
}

TARGET_DEVICE=$(uname -m)
OS=$(cat /etc/os-release | awk -F= '$1=="ID"{print $2}' | sed 's/"//g')

if [[ "${TARGET_DEVICE}" = "x86_64" ]]; then
    if [[ "${OS}" = "rhel" ]]; then
        BASE_LIB_DIR="/usr/lib64/"
    elif [[ "${OS}" = "ubuntu" ]]; then
        BASE_LIB_DIR="/usr/lib/x86_64-linux-gnu/"
    else
        echo "Unsupported OS" 2>&1
        exit 1
    fi
fi

# DeepStream version to remove. Override via the first argument, PREV_DS_VER,
# or NVDS_VERSION; defaults to 9.2.1. Must be MAJOR.MINOR.PATCH.
#   sudo ./uninstall.sh 9.2.1
#   sudo PREV_DS_VER=9.2.1 ./uninstall.sh
#   sudo NVDS_VERSION=9.2.1 ./uninstall.sh
PREV_DS_VER="${1:-${PREV_DS_VER:-${NVDS_VERSION:-9.2.1}}}"
if [[ -z "${PREV_DS_VER}" ]]; then
  echo "PREV_DS_VER not set (usage: $0 [version], e.g. $0 9.2.1)"
  exit 1
fi
if [[ ! "$PREV_DS_VER" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "error: invalid version '$PREV_DS_VER' (expected MAJOR.MINOR.PATCH, e.g. 9.2.1)" >&2
  exit 1
fi
# Install tree still uses MAJOR.MINOR for now; strip .PATCH so 9.2.1 removes
# deepstream-9.2/.
PREV_DS_VER="${PREV_DS_VER%.*}"

if [[ ! -d /opt/nvidia/deepstream/deepstream-${PREV_DS_VER} ]]; then
    echo "This version of DeepStream is not present in the system."
    exit 1
fi

if [[ -n "$TARGET_DEVICE" ]]; then
  if [[ "${TARGET_DEVICE}" = "x86_64" ]]; then
    remove_alternative deepstream-v4l2plugin /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/lib/libv4l/plugins/libcuvidv4l2_plugin.so
    remove_alternative deepstream-v4l2library /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/lib/libnvv4l2.so
    remove_alternative deepstream-v4lconvert /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/lib/libnvv4lconvert.so
    remove_alternative deepstream-appsrc-cuda-test /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-appsrc-cuda-test
    remove_alternative deepstream-multigpu-nvlink-test /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-multigpu-nvlink-test
    remove_alternative deepstream-ucx-test-app /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-ucx-test-app
  elif [[ "${TARGET_DEVICE}" = "aarch64" ]]; then
    # deepstream-ipc-test is Jetson-only; not installed on SBSA (both report uname -m as aarch64)
    if [[ -f /etc/nv_tegra_release ]]; then
      remove_alternative deepstream-ipc-test-app /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-ipc-test-app
    fi
  fi
fi
remove_alternative deepstream-plugins /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/lib/gst-plugins
remove_alternative deepstream-app /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-app
remove_alternative deepstream-test1-app /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-test1-app
remove_alternative deepstream-test2-app /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-test2-app
remove_alternative deepstream-test3-app /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-test3-app
remove_alternative deepstream-test4-app /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-test4-app
remove_alternative deepstream-test5-app /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-test5-app
remove_alternative deepstream-testsr-app /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-testsr-app
remove_alternative deepstream-transfer-learning-app /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-transfer-learning-app
remove_alternative deepstream-dynamicsrcbin-test /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-dynamicsrcbin-test
remove_alternative deepstream-user-metadata-app /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-user-metadata-app
remove_alternative deepstream-dewarper-app /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-dewarper-app
remove_alternative deepstream-nvof-app /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-nvof-app
remove_alternative deepstream-image-decode-app /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-image-decode-app
remove_alternative deepstream-gst-metadata-app /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-gst-metadata-app
remove_alternative deepstream-opencv-test /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-opencv-test
remove_alternative deepstream-preprocess-test /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-preprocess-test
remove_alternative deepstream-image-meta-test /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-image-meta-test
remove_alternative deepstream-appsrc-test /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-appsrc-test
remove_alternative deepstream-can-orientation-app /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-can-orientation-app
remove_alternative deepstream-nvdsanalytics-test /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-nvdsanalytics-test
remove_alternative deepstream-3d-action-recognition /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-3d-action-recognition
remove_alternative deepstream-3d-depth-camera /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-3d-depth-camera
remove_alternative deepstream-lidar-inference-app /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-lidar-inference-app
remove_alternative deepstream-3d-lidar-sensor-fusion /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-3d-lidar-sensor-fusion
remove_alternative deepstream-nmos-app /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-nmos-app
remove_alternative deepstream-server-app /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-server-app
remove_alternative deepstream-demuxer-static /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-demuxer-static
remove_alternative deepstream-demuxer-dynamic /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-demuxer-dynamic
remove_alternative deepstream-kitti-overlay /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/deepstream-kitti-overlay
remove_alternative service-maker-3d-action-recognition-app /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/service-maker-3d-action-recognition-app
remove_alternative service-maker-appsrc-test /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/service-maker-appsrc-test
remove_alternative service-maker-sr-test-app /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/service-maker-sr-test-app
remove_alternative service-maker-test1-app /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/service-maker-test1-app
remove_alternative service-maker-test2-app /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/service-maker-test2-app
remove_alternative service-maker-test3-app /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/service-maker-test3-app
remove_alternative service-maker-test4-app /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/service-maker-test4-app
remove_alternative service-maker-test5-app /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/service-maker-test5-app
remove_alternative ds-launch /opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/bin/ds-launch
# Uninstall with python3 -m pip for the same reason install.sh installs with it:
# a 'pip' resolved from PATH may belong to a different interpreter than the one
# the wheel was installed for, in which case the uninstall removes nothing and
# reports the package as absent.
# Do not remove a loader configuration belonging to another SDK installation.
if [[ -f /etc/ld.so.conf.d/deepstream.conf ]] &&
   grep -Fxq "/opt/nvidia/deepstream/deepstream-${PREV_DS_VER}/lib" /etc/ld.so.conf.d/deepstream.conf; then
    rm -f /etc/ld.so.conf.d/deepstream.conf
fi
python3 -m pip uninstall pyservicemaker -y --break-system-packages

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
if [[ "${TARGET_DEVICE}" = "x86_64" ]]; then
  PLATFORM=x86
elif [[ "${TARGET_DEVICE}" = "aarch64" ]]; then
  if [[ -f /etc/nv_tegra_release ]]; then
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
  return 0
}

remove_if_deb "deepstream-sample-data"
if [[ -n "${PLATFORM}" ]]; then
  remove_if_deb "deepstream-binaries-${PLATFORM}"
fi

# Remove the base version symlink created during install
# (/opt/nvidia/deepstream/deepstream -> deepstream-${PREV_DS_VER}), but only if
# it still points at the version we are uninstalling, so other installed
# versions are left untouched. The sources/includes and samples/trtis_model_repo
# symlinks are already removed by the rm -rf of sources/ and samples/ above.
DS_BASE_LINK=/opt/nvidia/deepstream/deepstream
if [[ -L "${DS_BASE_LINK}" ]] && [[ "$(readlink "${DS_BASE_LINK}")" = "deepstream-${PREV_DS_VER}" ]]; then
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
