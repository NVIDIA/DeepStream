####################################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
find_package(nvds_service_maker REQUIRED PATHS "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake")
find_package(CUDA)

add_library(sample_video_feeder SHARED sample_video_feeder.cpp)
# Add include directories for cuda
target_include_directories(sample_video_feeder PRIVATE ${CUDA_INCLUDE_DIRS})
include_directories(
    /usr/local/cuda/include
)
target_link_libraries(sample_video_feeder PRIVATE  nvds_service_maker nvds_service_maker_utils
    /usr/local/cuda/lib64/libcudart.so)
