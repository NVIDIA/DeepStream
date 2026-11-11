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

set -e

NVDS_VERSION=${NVDS_VERSION:-9.2.1}
# NVDS_VERSION must be MAJOR.MINOR.PATCH (default: 9.2.1). Install tree still
# uses MAJOR.MINOR for now (e.g. deepstream-9.2/); strip the patch for paths.
if [[ ! "$NVDS_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "error: invalid NVDS_VERSION='$NVDS_VERSION' (expected MAJOR.MINOR.PATCH, e.g. 9.2.1)" >&2
  exit 1
fi
# update-alternatives priority: MAJOR*10000 + MINOR*100 + PATCH
# (zero-pads minor/patch to 2 digits so 9.2.10 != 9.21.0 and 10.0.0 > 9.10.0)
NVDS_MAJOR=${NVDS_VERSION%%.*}
NVDS_VERSION_REST=${NVDS_VERSION#*.}
NVDS_MINOR=${NVDS_VERSION_REST%%.*}
NVDS_PATCH=${NVDS_VERSION##*.}
NVDS_PRIORITY=$((NVDS_MAJOR * 10000 + NVDS_MINOR * 100 + NVDS_PATCH))
NVDS_VERSION="${NVDS_VERSION%.*}"
DS_ROOT="/opt/nvidia/deepstream/deepstream-${NVDS_VERSION}"
alt_rank=$NVDS_PRIORITY

# Inspect every registration before changing system paths. update-alternatives can
# return success while refusing to replace a regular executable, so verify links too.
die() { echo "ERROR: $*" >&2; exit 1; }
alt_links=()
alt_names=()
alt_targets=()
alt_manual_values=()
provider_links=()
provider_targets=()

query_alternative() {
    LC_ALL=C update-alternatives --query "$1" 2>/dev/null || true
}

queue_alternative() {
    local link=$1 name=$2 target=$3 required=${4:-optional}
    local info legacy_info registered_link current_link owner status value
    if [[ ! -e "$target" ]]; then
        [[ "$required" != required ]] || die "Required SDK target is missing: $target"
        echo "Skipping optional target not built for this platform: $target"
        return
    fi
    if [[ "$target" == "$DS_ROOT/bin/"* && ! -x "$target" ]]; then
        die "SDK executable is not executable: $target"
    fi

    info=$(query_alternative "$name")
    if [[ "$name" == deepstream-ucx-test-app ]]; then
        legacy_info=$(query_alternative deepstream-ucx-test-app-client)
        registered_link=$(printf '%s\n' "$legacy_info" | sed -n 's/^Link: //p')
        if [[ "$registered_link" == "$link" ]]; then
            [[ -z "$info" ]] || die "Both UCX alternative groups exist; review their ownership before installation."
            name=deepstream-ucx-test-app-client
            info=$legacy_info
            echo "Using existing UCX alternative group: $name"
        fi
    fi
    if [[ -e "$link" && ! -L "$link" ]]; then
        owner=$(dpkg-query -S "$link" 2>/dev/null || true)
        die "$link is a regular file, not a managed alternative${owner:+ ($owner)}. Review its owner and migrate it explicitly; no overwrite was attempted. Inspect the new SDK with $DS_ROOT/bin/deepstream-app --version-all."
    fi
    registered_link=$(printf '%s\n' "$info" | sed -n 's/^Link: //p')
    [[ -z "$info" || "$registered_link" == "$link" ]] ||
        die "Alternative $name is registered for another link: $registered_link"
    if [[ -L "$link" ]]; then
        current_link=$(readlink "$link")
        [[ "$current_link" == "/etc/alternatives/$name" ]] ||
            die "$link is an unmanaged/conflicting symlink to $current_link; review its ownership before installation."
    fi
    status=$(printf '%s\n' "$info" | sed -n 's/^Status: //p')
    value=$(printf '%s\n' "$info" | sed -n 's/^Value: //p')
    if [[ "$status" == manual && "$value" != "$target" ]]; then
        [[ -e "$value" ]] || die "Manual alternative $name points to a missing target: $value"
        echo "Preserving manual selection for $name: $value (new SDK target: $target)"
    fi
    alt_links+=("$link")
    alt_names+=("$name")
    alt_targets+=("$target")
    if [[ "$status" == manual ]]; then
        alt_manual_values+=("$value")
    else
        alt_manual_values+=("")
    fi
}

queue_app() {
    queue_alternative "/usr/bin/$1" "$1" "$DS_ROOT/bin/$1" "${2:-optional}"
}

queue_media_provider() {
    local lib=$1 required=${2:-required} source=
    # SBSA packages may contain real, BSP-qualified media libraries. Never
    # replace those with a guessed nvidia/tegra link (or with a dangling link).
    [[ ! -f "$DS_ROOT/lib/$lib" ]] || return 0
    if [[ -n "$NVIDIA_LIB_DIR" && -f "$NVIDIA_LIB_DIR/$lib" ]]; then
        source="$NVIDIA_LIB_DIR/$lib"
    else
        [[ "$required" != required ]] || die "Missing ARM media provider for $lib: neither $DS_ROOT/lib nor the platform provider contains it."
        echo "Skipping unavailable optional ARM media provider: $lib"
        return
    fi
    [[ ! -e "$DS_ROOT/lib/$lib" || -L "$DS_ROOT/lib/$lib" ]] ||
        die "Refusing to replace a non-library entry: $DS_ROOT/lib/$lib"
    provider_links+=("$DS_ROOT/lib/$lib")
    provider_targets+=("$source")
}

apply_alternatives() {
    local i info value best status actual
    for ((i=0; i<${#provider_links[@]}; i++)); do
        ln -sfn "${provider_targets[i]}" "${provider_links[i]}"
    done
    for ((i=0; i<${#alt_links[@]}; i++)); do
        mkdir -p "$(dirname "${alt_links[i]}")"
        update-alternatives --install "${alt_links[i]}" "${alt_names[i]}" "${alt_targets[i]}" "$alt_rank"
        info=$(query_alternative "${alt_names[i]}")
        value=$(printf '%s\n' "$info" | sed -n 's/^Value: //p')
        best=$(printf '%s\n' "$info" | sed -n 's/^Best: //p')
        status=$(printf '%s\n' "$info" | sed -n 's/^Status: //p')
        [[ -n "$value" && -e "$value" && -L "${alt_links[i]}" ]] ||
            die "Alternative ${alt_names[i]} did not create a usable link."
        [[ "$(readlink "${alt_links[i]}")" == "/etc/alternatives/${alt_names[i]}" ]] ||
            die "Alternative ${alt_names[i]} did not manage ${alt_links[i]}."
        actual=$(readlink -f "${alt_links[i]}")
        [[ "$actual" == "$(readlink -f "$value")" ]] ||
            die "Alternative ${alt_names[i]} link does not resolve to its selected target."
        if [[ -n "${alt_manual_values[i]}" ]]; then
            [[ "$status" == manual && "$value" == "${alt_manual_values[i]}" ]] ||
                die "Alternative ${alt_names[i]} unexpectedly changed a manual selection."
        elif [[ "$status" != auto || -z "$best" || "$value" != "$best" ]]; then
            die "Alternative ${alt_names[i]} does not select its highest-priority target."
        fi
        if [[ "$value" != "${alt_targets[i]}" ]]; then
            echo "NOTICE: ${alt_links[i]} selects $value; use ${alt_targets[i]} to run this SDK explicitly."
        fi
    done
}

write_loader_config() {
    local root=${TRITON_CLIENT_SDK_ROOT:-/opt/tritonclient} dir lib provider=
    # Match the source linker's complete client SDK selection. A partial lib/
    # must not shadow the complete provider selected from lib64/.
    for dir in "$root/lib" "$root/lib64"; do
        if [[ -r "$dir/libprotobuf.a" && -r "$dir/libcares.a" &&
              ( -r "$dir/libgrpcclient.so" ||
                ( -r "$dir/libgrpcclient.a" && -r "$dir/libgrpcclient_static.a" &&
                  -r "$dir/libgrpc++.a" ) ) ]]; then
            provider=$dir
            break
        fi
    done
    # Compact Docker runtime providers contain shared clients only. Keep that
    # supported, but select a single directory rather than mixing providers.
    if [[ -z "$provider" ]]; then
        for dir in "$root/lib" "$root/lib64"; do
            for lib in "$dir"/libgrpcclient.so* "$dir"/libhttpclient.so*; do
                if [[ -f "$lib" ]]; then
                    provider=$dir
                    break 2
                fi
            done
        done
    fi
    {
        printf '%s\n' "$DS_ROOT/lib" "$DS_ROOT/lib/gst-plugins"
        # Static-client GVS packages and staged Docker installs need no shared
        # client provider here. Register actual shared-provider directories only.
        if [[ -n "$provider" ]]; then
            for lib in "$provider"/libgrpcclient.so* "$provider"/libhttpclient.so*; do
                if [[ -f "$lib" ]]; then
                    printf '%s\n' "$provider"
                    break
                fi
            done
        fi
    } > /etc/ld.so.conf.d/deepstream.conf
}

command -v update-alternatives >/dev/null || die "update-alternatives is required."
command -v ldconfig >/dev/null || die "ldconfig is required."
NVIDIA_LIB_DIR=
TARGET_DEVICE=$(uname -m)
OS=$(awk -F= '$1=="ID" {gsub(/"/, "", $2); print $2}' /etc/os-release)
case "$TARGET_DEVICE" in
    x86_64)
        case "$OS" in
            rhel) BASE_LIB_DIR=/usr/lib64 ;;
            ubuntu) BASE_LIB_DIR=/usr/lib/x86_64-linux-gnu ;;
            *) die "Unsupported OS: $OS" ;;
        esac
        ;;
    aarch64)
        BASE_LIB_DIR=/usr/lib/aarch64-linux-gnu
        ;;
    *) die "Unsupported architecture: $TARGET_DEVICE" ;;
esac
while getopts "d:" option; do
    case "$option" in
        d) BASE_LIB_DIR=$OPTARG ;;
        *) die "Unsupported option" ;;
    esac
done
if [[ "$TARGET_DEVICE" == aarch64 ]]; then
    for dir in "$BASE_LIB_DIR/nvidia" "$BASE_LIB_DIR/tegra"; do
        if [[ -f "$dir/libnvbufsurface.so" ]]; then
            NVIDIA_LIB_DIR=$dir
            break
        fi
    done
    for lib in libnvbufsurface.so libnvbufsurftransform.so libnvdsbufferpool.so; do
        queue_media_provider "$lib"
    done
    queue_media_provider libgstnvcustomhelper.so optional
fi

queue_alternative "$BASE_LIB_DIR/gstreamer-1.0/deepstream" deepstream-plugins "$DS_ROOT/lib/gst-plugins" required
queue_app deepstream-app required
if [[ "$TARGET_DEVICE" == x86_64 ]]; then
    queue_alternative "$BASE_LIB_DIR/libv4l/plugins/libcuvidv4l2_plugin.so" deepstream-v4l2plugin "$DS_ROOT/lib/libv4l/plugins/libcuvidv4l2_plugin.so" required
    queue_alternative "$BASE_LIB_DIR/libv4l2.so.0.0.99999" deepstream-v4l2library "$DS_ROOT/lib/libnvv4l2.so" required
    queue_alternative "$BASE_LIB_DIR/libv4lconvert.so.0.0.99999" deepstream-v4lconvert "$DS_ROOT/lib/libnvv4lconvert.so" required
    queue_app deepstream-appsrc-cuda-test
    queue_app deepstream-multigpu-nvlink-test
    queue_app deepstream-ucx-test-app
elif [[ -f /etc/nv_tegra_release ]]; then
    queue_app deepstream-ipc-test-app
fi

apps=(deepstream-test1-app deepstream-test2-app deepstream-test3-app
    deepstream-test4-app deepstream-test5-app deepstream-testsr-app
    deepstream-transfer-learning-app deepstream-dynamicsrcbin-test
    deepstream-user-metadata-app deepstream-dewarper-app deepstream-nvof-app
    deepstream-image-decode-app deepstream-gst-metadata-app deepstream-opencv-test
    deepstream-preprocess-test deepstream-image-meta-test deepstream-appsrc-test
    deepstream-can-orientation-app deepstream-nvdsanalytics-test
    deepstream-3d-action-recognition deepstream-3d-depth-camera
    deepstream-lidar-inference-app deepstream-3d-lidar-sensor-fusion
    deepstream-nmos-app deepstream-server-app deepstream-demuxer-static
    deepstream-demuxer-dynamic service-maker-3d-action-recognition-app
    service-maker-appsrc-test service-maker-sr-test-app
    service-maker-test1-app service-maker-test2-app service-maker-test3-app
    service-maker-test4-app service-maker-test5-app deepstream-kitti-overlay ds-launch)
for app in "${apps[@]}"; do
    queue_app "$app"
done

# Preserve the RHEL compatibility link without overwriting unmanaged files.
rhel_compat=
if [[ "$TARGET_DEVICE" == x86_64 && "$OS" == rhel ]]; then
    rhel_compat=/usr/lib/x86_64-linux-gnu/libv4l/plugins/libcuvidv4l2_plugin.so
    if [[ -e "$rhel_compat" || -L "$rhel_compat" ]]; then
        [[ -L "$rhel_compat" && "$(readlink -f "$rhel_compat")" == "$(readlink -f "$DS_ROOT/lib/libv4l/plugins/libcuvidv4l2_plugin.so")" ]] ||
            die "RHEL compatibility link is occupied: $rhel_compat; review its ownership."
        rhel_compat=
    fi
fi

# All collision and target checks precede modifications.
if [[ -n "${TRITON_CLIENT_SDK_ROOT:-}" && "$TRITON_CLIENT_SDK_ROOT" != /opt/tritonclient ]]; then
    [[ "$TRITON_CLIENT_SDK_ROOT" == /* && -d "$TRITON_CLIENT_SDK_ROOT" ]] ||
        die "Selected TRITON_CLIENT_SDK_ROOT must be an existing absolute provider directory: $TRITON_CLIENT_SDK_ROOT"
fi
wheels=("$DS_ROOT"/service-maker/python/pyservicemaker*.whl)
[[ ${#wheels[@]} == 1 && -f "${wheels[0]}" ]] ||
    die "Expected exactly one matching Service Maker wheel in $DS_ROOT/service-maker/python"
command -v python3 >/dev/null || die "python3 is required to install Service Maker."
apply_alternatives
if [[ -n "$rhel_compat" ]]; then
    mkdir -p "$(dirname "$rhel_compat")"
    ln -s "$DS_ROOT/lib/libv4l/plugins/libcuvidv4l2_plugin.so" "$rhel_compat"
fi
write_loader_config
python3 -m pip install "${wheels[0]}" --force-reinstall --break-system-packages
ldconfig
rm -rf /home/*/.cache/gstreamer-1.0/
rm -rf /root/.cache/gstreamer-1.0/
