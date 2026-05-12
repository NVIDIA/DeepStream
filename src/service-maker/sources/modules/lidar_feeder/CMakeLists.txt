####################################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
####################################################################################################

cmake_minimum_required(VERSION 3.16)

project(Sample)
if(NOT DEFINED ENV{NVDS_VERSION})
  set(NVDS_VERSION "9.0")
else()
  set(NVDS_VERSION $ENV{NVDS_VERSION})
endif()
set(DS_PATH "/opt/nvidia/deepstream/deepstream-${NVDS_VERSION}")

find_package(nvds_service_maker REQUIRED PATHS "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake")
find_package(CUDA)

add_library(lidar_feeder SHARED lidar_feeder.cpp)

target_include_directories(lidar_feeder PRIVATE ${CUDA_INCLUDE_DIRS})
target_include_directories(lidar_feeder PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../includes")
target_include_directories(lidar_feeder PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../includes/ds3d/commmon")
target_include_directories(lidar_feeder PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../src/utils")

include_directories(
    /usr/include/gstreamer-1.0
    /usr/include/glib-2.0
    /usr/lib/aarch64-linux-gnu/glib-2.0/include/
    /usr/lib/x86_64-linux-gnu/glib-2.0/include
    /usr/local/cuda/include
)
target_link_libraries(lidar_feeder nvds_service_maker nvds_service_maker_utils)
target_link_libraries(lidar_feeder ${DS_PATH}/lib/libnvds_3d_common.so)
target_link_libraries(lidar_feeder ${DS_PATH}/lib/libnvdsgst_3d_gst.so)

