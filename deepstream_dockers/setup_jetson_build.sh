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

# Jetson build setup

cp ./jetson_specific_files/deps/rsyslog ./jetson_dockerfiles/
cp ./jetson_specific_files/entrypoint.sh  ./jetson_dockerfiles/
cp ./jetson_specific_files/user_additional_install_runtime.sh ./jetson_dockerfiles/
cp ./jetson_specific_files/user_additional_install_devel.sh ./jetson_dockerfiles/
cp ./jetson_specific_files/user_deepstream_python_apps_install.sh ./jetson_dockerfiles/
cp ./common/files/* ./jetson_dockerfiles/

# copy /tmp99 components to a specific directory and then point the environment

# ADDVAR99jetson="${ADDVAR99jetson:-$(pwd -P)}"

if [ -z "${ADDVAR99:-}" ]; then
  echo "Error: ADDVAR99 is not set" >&2
  exit 1
fi

# echo 'copying gst files'

cp $ADDVAR99/jetson/gst/libgstrtpmanager.so ./jetson_dockerfiles/
cp $ADDVAR99/jetson/gst/libgstrtsp.so ./jetson_dockerfiles/
cp $ADDVAR99/jetson/gst/libgstvideoparsersbad.so ./jetson_dockerfiles/

mkdir -p ./jetson_dockerfiles/optel

# echo 'copying open tel *.deb files'


cp $ADDVAR99/jetson/optel/* ./jetson_dockerfiles/optel/
