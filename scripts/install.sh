#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

NVDS_VERSION=${NVDS_VERSION:-9.1}
# update-alternatives priority derived from DeepStream major+minor version (e.g. 9.1 -> 91)
NVDS_MAJOR=${NVDS_VERSION%%.*}
NVDS_VERSION_REST=${NVDS_VERSION#*.}
NVDS_MINOR=${NVDS_VERSION_REST%%.*}
NVDS_PRIORITY="${NVDS_MAJOR}${NVDS_MINOR}"
TARGET_DEVICE=$(uname -m)
OS=$(cat /etc/os-release | awk -F= '$1=="ID"{print $2}' | sed 's/"//g')

if [ "${TARGET_DEVICE}" = "x86_64" ]; then
    if [ "${OS}" = "rhel" ]; then
        mkdir -p /usr/lib/x86_64-linux-gnu/libv4l/plugins/
        ln -sf /opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/lib/libv4l/plugins/libcuvidv4l2_plugin.so /usr/lib/x86_64-linux-gnu/libv4l/plugins/libcuvidv4l2_plugin.so
        BASE_LIB_DIR="/usr/lib64/"
    elif [ "${OS}" = "ubuntu" ]; then
        BASE_LIB_DIR="/usr/lib/x86_64-linux-gnu/"
    else
        echo "Unsupported OS" 2>&1
        exit 1
    fi
elif [ "${TARGET_DEVICE}" = "aarch64" ]; then
    BASE_LIB_DIR="/usr/lib/aarch64-linux-gnu"
    if [ -f "$BASE_LIB_DIR/nvidia/libnvbufsurface.so" ]; then
        NVIDIA_LIB_DIR="$BASE_LIB_DIR/nvidia"
    else
        NVIDIA_LIB_DIR="$BASE_LIB_DIR/tegra"
    fi
fi

while getopts "d:" option; do
    case "${option}" in
    d)
        BASE_LIB_DIR="${OPTARG}"
        ;;
    *)
        echo "ERROR! Unsupported option!"
        exit 1
        ;;
    esac
done

if [ -n "$TARGET_DEVICE" ]; then
    if [ "${TARGET_DEVICE}" = "x86_64" ]; then
        update-alternatives --install $BASE_LIB_DIR/gstreamer-1.0/deepstream deepstream-plugins /opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/lib/gst-plugins ${NVDS_PRIORITY}
        update-alternatives --install $BASE_LIB_DIR/libv4l/plugins/libcuvidv4l2_plugin.so deepstream-v4l2plugin /opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/lib/libv4l/plugins/libcuvidv4l2_plugin.so ${NVDS_PRIORITY}
        update-alternatives --install /usr/bin/deepstream-appsrc-cuda-test deepstream-appsrc-cuda-test /opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/bin/deepstream-appsrc-cuda-test ${NVDS_PRIORITY}
        update-alternatives --install /usr/bin/deepstream-multigpu-nvlink-test deepstream-multigpu-nvlink-test /opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/bin/deepstream-multigpu-nvlink-test ${NVDS_PRIORITY}
        update-alternatives --install $BASE_LIB_DIR/libv4l2.so.0.0.99999 deepstream-v4l2library /opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/lib/libnvv4l2.so ${NVDS_PRIORITY}
        update-alternatives --install $BASE_LIB_DIR/libv4lconvert.so.0.0.99999 deepstream-v4lconvert /opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/lib/libnvv4lconvert.so ${NVDS_PRIORITY}
        update-alternatives --install /usr/bin/deepstream-ucx-test-app deepstream-ucx-test-app /opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/bin/deepstream-ucx-test-app ${NVDS_PRIORITY}
        echo "/opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/lib" > /etc/ld.so.conf.d/deepstream.conf
        echo "/opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/lib/gst-plugins" >> /etc/ld.so.conf.d/deepstream.conf
    elif [ "${TARGET_DEVICE}" = "aarch64" ]; then
        update-alternatives --install $BASE_LIB_DIR/gstreamer-1.0/deepstream deepstream-plugins /opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/lib/gst-plugins ${NVDS_PRIORITY}
        for lib in libnvbufsurface.so libnvbufsurftransform.so libnvdsbufferpool.so libgstnvcustomhelper.so; do
            if [ ! -f "$NVIDIA_LIB_DIR/$lib" ]; then
                echo "ERROR: required library $NVIDIA_LIB_DIR/$lib not found; cannot create symlink" >&2
                exit 1
            fi
            ln -sf "$NVIDIA_LIB_DIR/$lib" "/opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/lib/$lib"
        done
        # deepstream-ipc-test is Jetson-only; not built on SBSA (both report uname -m as aarch64)
        if [ -f /etc/nv_tegra_release ]; then
            update-alternatives --install /usr/bin/deepstream-ipc-test-app deepstream-ipc-test-app /opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/bin/deepstream-ipc-test-app ${NVDS_PRIORITY}
        fi
        echo "/opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/lib" > /etc/ld.so.conf.d/deepstream.conf
        echo "/opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/lib/gst-plugins" >> /etc/ld.so.conf.d/deepstream.conf
        echo "/opt/tritonclient/lib" >> /etc/ld.so.conf.d/deepstream.conf
    fi
fi
update-alternatives --install /usr/bin/deepstream-app deepstream-app /opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/bin/deepstream-app ${NVDS_PRIORITY}
update-alternatives --install /usr/bin/deepstream-test1-app deepstream-test1-app /opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/bin/deepstream-test1-app ${NVDS_PRIORITY}
update-alternatives --install /usr/bin/deepstream-test2-app deepstream-test2-app /opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/bin/deepstream-test2-app ${NVDS_PRIORITY}
update-alternatives --install /usr/bin/deepstream-test3-app deepstream-test3-app /opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/bin/deepstream-test3-app ${NVDS_PRIORITY}
update-alternatives --install /usr/bin/deepstream-test4-app deepstream-test4-app /opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/bin/deepstream-test4-app ${NVDS_PRIORITY}
update-alternatives --install /usr/bin/deepstream-test5-app deepstream-test5-app /opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/bin/deepstream-test5-app ${NVDS_PRIORITY}
update-alternatives --install /usr/bin/deepstream-testsr-app deepstream-testsr-app /opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/bin/deepstream-testsr-app ${NVDS_PRIORITY}
update-alternatives --install /usr/bin/deepstream-transfer-learning-app deepstream-transfer-learning-app /opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/bin/deepstream-transfer-learning-app ${NVDS_PRIORITY}
update-alternatives --install /usr/bin/deepstream-dynamicsrcbin-test deepstream-dynamicsrcbin-test /opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/bin/deepstream-dynamicsrcbin-test ${NVDS_PRIORITY}
update-alternatives --install /usr/bin/deepstream-user-metadata-app deepstream-user-metadata-app /opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/bin/deepstream-user-metadata-app ${NVDS_PRIORITY}
update-alternatives --install /usr/bin/deepstream-dewarper-app deepstream-dewarper-app /opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/bin/deepstream-dewarper-app ${NVDS_PRIORITY}
update-alternatives --install /usr/bin/deepstream-nvof-app deepstream-nvof-app /opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/bin/deepstream-nvof-app ${NVDS_PRIORITY}
update-alternatives --install /usr/bin/deepstream-image-decode-app deepstream-image-decode-app /opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/bin/deepstream-image-decode-app ${NVDS_PRIORITY}
update-alternatives --install /usr/bin/deepstream-gst-metadata-app deepstream-gst-metadata-app /opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/bin/deepstream-gst-metadata-app ${NVDS_PRIORITY}
update-alternatives --install /usr/bin/deepstream-opencv-test deepstream-opencv-test /opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/bin/deepstream-opencv-test ${NVDS_PRIORITY}
update-alternatives --install /usr/bin/deepstream-preprocess-test deepstream-preprocess-test /opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/bin/deepstream-preprocess-test ${NVDS_PRIORITY}
update-alternatives --install /usr/bin/deepstream-image-meta-test deepstream-image-meta-test /opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/bin/deepstream-image-meta-test ${NVDS_PRIORITY}
update-alternatives --install /usr/bin/deepstream-appsrc-test deepstream-appsrc-test /opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/bin/deepstream-appsrc-test ${NVDS_PRIORITY}
update-alternatives --install /usr/bin/deepstream-can-orientation-app deepstream-can-orientation-app /opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/bin/deepstream-can-orientation-app ${NVDS_PRIORITY}
update-alternatives --install /usr/bin/deepstream-nvdsanalytics-test deepstream-nvdsanalytics-test /opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/bin/deepstream-nvdsanalytics-test ${NVDS_PRIORITY}
update-alternatives --install /usr/bin/deepstream-3d-action-recognition deepstream-3d-action-recognition /opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/bin/deepstream-3d-action-recognition ${NVDS_PRIORITY}
update-alternatives --install /usr/bin/deepstream-3d-depth-camera deepstream-3d-depth-camera /opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/bin/deepstream-3d-depth-camera ${NVDS_PRIORITY}
update-alternatives --install /usr/bin/deepstream-lidar-inference-app deepstream-lidar-inference-app /opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/bin/deepstream-lidar-inference-app ${NVDS_PRIORITY}
update-alternatives --install /usr/bin/deepstream-3d-lidar-sensor-fusion deepstream-3d-lidar-sensor-fusion /opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/bin/deepstream-3d-lidar-sensor-fusion ${NVDS_PRIORITY}
update-alternatives --install /usr/bin/deepstream-nmos-app deepstream-nmos-app /opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/bin/deepstream-nmos-app ${NVDS_PRIORITY}
update-alternatives --install /usr/bin/deepstream-server-app deepstream-server-app /opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/bin/deepstream-server-app ${NVDS_PRIORITY}
update-alternatives --install /usr/bin/deepstream-demuxer-static deepstream-demuxer-static /opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/bin/deepstream-demuxer-static ${NVDS_PRIORITY}
update-alternatives --install /usr/bin/deepstream-demuxer-dynamic deepstream-demuxer-dynamic /opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/bin/deepstream-demuxer-dynamic ${NVDS_PRIORITY}
update-alternatives --install /usr/bin/service-maker-3d-action-recognition-app service-maker-3d-action-recognition-app /opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/bin/service-maker-3d-action-recognition-app ${NVDS_PRIORITY}
update-alternatives --install /usr/bin/service-maker-appsrc-test service-maker-appsrc-test /opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/bin/service-maker-appsrc-test ${NVDS_PRIORITY}
update-alternatives --install /usr/bin/service-maker-sr-test-app service-maker-sr-test-app /opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/bin/service-maker-sr-test-app ${NVDS_PRIORITY}
update-alternatives --install /usr/bin/service-maker-test1-app service-maker-test1-app /opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/bin/service-maker-test1-app ${NVDS_PRIORITY}
update-alternatives --install /usr/bin/service-maker-test2-app service-maker-test2-app /opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/bin/service-maker-test2-app ${NVDS_PRIORITY}
update-alternatives --install /usr/bin/service-maker-test3-app service-maker-test3-app /opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/bin/service-maker-test3-app ${NVDS_PRIORITY}
update-alternatives --install /usr/bin/service-maker-test4-app service-maker-test4-app /opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/bin/service-maker-test4-app ${NVDS_PRIORITY}
update-alternatives --install /usr/bin/service-maker-test5-app service-maker-test5-app /opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/bin/service-maker-test5-app ${NVDS_PRIORITY}
pip install /opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/service-maker/python/pyservicemaker*.whl --force-reinstall --break-system-packages
ldconfig
rm -rf /home/*/.cache/gstreamer-1.0/
rm -rf /root/.cache/gstreamer-1.0/
