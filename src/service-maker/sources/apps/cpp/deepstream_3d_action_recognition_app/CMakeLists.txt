####################################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
find_package(nvds_service_maker REQUIRED PATHS "${CMAKE_CURRENT_SOURCE_DIR}/../../../../cmake")

add_executable(deepstream-3d-action-recognition-app deepstream_3d_action_recognition.cpp)
target_link_libraries(deepstream-3d-action-recognition-app PRIVATE nvds_service_maker)
