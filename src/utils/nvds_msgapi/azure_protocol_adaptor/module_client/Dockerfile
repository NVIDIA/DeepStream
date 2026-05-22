################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
################################################################################
#
#######################################################################################################

# This is the dockerfile to create a container to run the test application to send messages to azure iot edge

# x86 deepstream-<ds-version> image
FROM nvcr.io/nvidia/deepstream:<ds-version>-<tag>

# Jetson deepstream-<ds-version> image
#FROM nvcr.io/nvidia/deepstream-l4t:<ds-version>-<tag>

# copy the compiled sample test applications
COPY test_azure_iotedge_sync /root
COPY test_azure_iotedge_async /root

# copy the azure cfg file
COPY cfg_azure.txt /root
