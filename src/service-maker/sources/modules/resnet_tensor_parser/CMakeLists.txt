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

cmake_minimum_required(VERSION 3.16)

project(ResnetTensorParser)
find_package(nvds_service_maker REQUIRED PATHS "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake")

# Find the GStreamer package
find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0>=1.0 gstreamer-base-1.0>=1.0)

find_package(CUDA REQUIRED)

# Include the GStreamer headers
include_directories(${GSTREAMER_INCLUDE_DIRS})

# DBScan clustering calls (NvDsInferDBScanCreate/Cluster/Destroy) are implemented
# in libnvds_inferutils.so; the resnet bbox parser (NvDsInferParseCustomResnet) is
# in libnvds_infercustomparser.so. Neither is pulled in by nvds_service_maker.
find_library(NVDS_INFERUTILS_LIBRARY NAMES nvds_inferutils HINTS /opt/nvidia/deepstream/deepstream/lib)
find_library(NVDS_INFERCUSTOMPARSER_LIBRARY NAMES nvds_infercustomparser HINTS /opt/nvidia/deepstream/deepstream/lib)

add_library(resnet_tensor_parser SHARED resnet_tensor_parser.cpp)
target_include_directories(resnet_tensor_parser PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(resnet_tensor_parser PRIVATE  nvds_service_maker nvds_service_maker_utils ${NVDS_INFERUTILS_LIBRARY} ${NVDS_INFERCUSTOMPARSER_LIBRARY})