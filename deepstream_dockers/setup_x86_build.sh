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

# x86 build setup
# samples and triton

cp ./x86_64_specific_files/10_nvidia.json ./x86_dockerfiles/
cp ./x86_64_specific_files/deps/gRPC_installation.sh ./x86_dockerfiles/
cp ./x86_64_specific_files/user_additional_install_runtime.sh ./x86_dockerfiles/
cp ./x86_64_specific_files/user_additional_install_devel.sh ./x86_dockerfiles/
cp ./x86_64_specific_files/user_deepstream_python_apps_install.sh ./x86_dockerfiles/
cp ./common/files/* ./x86_dockerfiles/

cp ./x86_64_specific_files/deps/rsyslog ./x86_dockerfiles/

cp ./x86_64_specific_files/10_nvidia.json ./x86_dockerfiles/
cp ./x86_64_specific_files/entrypoint.sh ./x86_dockerfiles/

# copy /tmp99 components to a specific directory and then point the environment

# ADDVAR99x86="${ADDVAR99x86:-$(pwd -P)}"

if [ -z "${ADDVAR99:-}" ]; then
  echo "Error: ADDVAR99 is not set" >&2
  exit 1
fi

# echo 'copying gst files ...'

cp $ADDVAR99/x86/gst/libgstrtpmanager.so ./x86_dockerfiles/
cp $ADDVAR99/x86/gst/libgstrtsp.so ./x86_dockerfiles/
cp $ADDVAR99/x86/gst/libgstvideoparsersbad.so ./x86_dockerfiles/

mkdir -p ./x86_dockerfiles/optel

# echo 'copying open tel *.deb files'

cp $ADDVAR99/x86/optel/* ./x86_dockerfiles/optel/


