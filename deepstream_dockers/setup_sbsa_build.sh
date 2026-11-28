#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# =================================================

# SBSA build setup
# triton (devel) -- the same image runs on GB200 / Grace and on DGX Spark

set -o errexit
set -o nounset

cp ./sbsa_specific_files/entrypoint.sh ./sbsa_dockerfiles/
cp ./sbsa_specific_files/extract_packages.sh ./sbsa_dockerfiles/
cp ./sbsa_specific_files/user_additional_install_devel.sh ./sbsa_dockerfiles/
cp ./sbsa_specific_files/user_deepstream_python_apps_install.sh ./sbsa_dockerfiles/
cp ./sbsa_specific_files/deps/gRPC_installation.sh ./sbsa_dockerfiles/
cp ./sbsa_specific_files/deps/rsyslog ./sbsa_dockerfiles/
cp ./common/files/* ./sbsa_dockerfiles/

# The published SBSA SDK tarball does not ship these three libraries. Stage the
# SBSA builds from sbsa_specific_files and reject a Git LFS pointer.
sbsa_media_libs=(libnvbufsurface.so.1.0.0 libnvbufsurftransform.so.1.0.0 libnvcuvidv4l2.so)
for lib in "${sbsa_media_libs[@]}"; do
  src="./sbsa_specific_files/$lib"
  if [[ ! -f "$src" ]]; then
    echo "Error: missing $src" >&2
    exit 1
  fi
  magic=$(od -An -t x1 -N 4 -- "$src" | tr -d ' \n')
  if [[ "$magic" != 7f454c46 ]]; then
    echo "Error: $lib is not an ELF library." >&2
    exit 1
  fi
  cp -- "$src" "./sbsa_dockerfiles/$lib"
done

# copy /tmp99 components to a specific directory and then point the environment

if [[ -z "${ADDVAR99:-}" ]]; then
  echo "Error: ADDVAR99 is not set" >&2
  exit 1
fi

# echo 'copying gst files ...'

cp "$ADDVAR99/sbsa/gst/libgstrtpmanager.so" ./sbsa_dockerfiles/
cp "$ADDVAR99/sbsa/gst/libgstrtsp.so" ./sbsa_dockerfiles/
cp "$ADDVAR99/sbsa/gst/libgstvideoparsersbad.so" ./sbsa_dockerfiles/

trt_repo=nv-tensorrt-local-repo-ubuntu2404-11.4.1-cuda-13.4_1.0-1_arm64.deb
if [[ ! -f "./sbsa_dockerfiles/$trt_repo" ]]; then
  cp "$ADDVAR99/sbsa/$trt_repo" ./sbsa_dockerfiles/
fi

if [[ ! -d ./sbsa_dockerfiles/vpi ]]; then
  cp -a "$ADDVAR99/sbsa/vpi" ./sbsa_dockerfiles/
fi

# The Tegra multimedia stack is baked into the SBSA image, so the Jetson Linux
# Driver Package (BSP) has to be in the build context. Stage it from
# $ADDVAR99/sbsa/bsp/ if it is not already in ./sbsa_dockerfiles/.
shopt -s nullglob
vpi_debs=(./sbsa_dockerfiles/vpi/*.deb)
if [[ ${#vpi_debs[@]} -eq 0 ]]; then
  echo "Error: ./sbsa_dockerfiles/vpi has no .deb files." >&2
  echo "       Place the SBSA VPI 4.1.6 runtime and development debs in $ADDVAR99/sbsa/vpi/." >&2
  echo "       Remove an empty ./sbsa_dockerfiles/vpi directory, then rerun setup." >&2
  exit 1
fi
staged_bsp=(./sbsa_dockerfiles/Jetson_Linux_R*_aarch64.tbz2)
if [[ ${#staged_bsp[@]} -gt 1 ]]; then
  echo "Error: multiple BSP tarballs in ./sbsa_dockerfiles/; keep only one." >&2
  printf '       %s\n' "${staged_bsp[@]}" >&2
  exit 1
fi
if [[ ${#staged_bsp[@]} -eq 0 ]]; then
  source_bsp=("$ADDVAR99"/sbsa/bsp/Jetson_Linux_R*_aarch64.tbz2)
  if [[ ${#source_bsp[@]} -eq 0 ]]; then
    echo "Error: no Jetson_Linux_R*_aarch64.tbz2 found in $ADDVAR99/sbsa/bsp/" >&2
    echo "       Download the Driver Package (BSP) for the matching Jetson Linux" >&2
    echo "       release from https://developer.nvidia.com/embedded/jetson-linux" >&2
    echo "       and place it there (see README section 5)." >&2
    exit 1
  fi
  if [[ ${#source_bsp[@]} -gt 1 ]]; then
    echo "Error: multiple BSP tarballs in $ADDVAR99/sbsa/bsp/; keep only one." >&2
    printf '       %s\n' "${source_bsp[@]}" >&2
    exit 1
  fi
  echo "copying BSP $(basename "${source_bsp[0]}") into the build context ..."
  cp "${source_bsp[0]}" ./sbsa_dockerfiles/
fi
staged_sdk=(./sbsa_dockerfiles/deepstream_sdk_v*_sbsa.tbz2)
if [[ ${#staged_sdk[@]} -ne 1 || "$(basename "${staged_sdk[0]}")" != deepstream_sdk_v9.2.1_sbsa.tbz2 ]]; then
  echo "Error: place exactly deepstream_sdk_v9.2.1_sbsa.tbz2 in ./sbsa_dockerfiles/." >&2
  exit 1
fi
shopt -u nullglob

echo "SBSA build context staged in ./sbsa_dockerfiles/"
