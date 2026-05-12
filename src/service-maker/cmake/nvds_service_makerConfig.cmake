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

find_path(nvds_service_maker_INCLUDE_DIR includes HINTS "${CMAKE_CURRENT_LIST_DIR}/..")
find_library(nvds_service_maker_LIBRARY NAMES nvds_service_maker HINTS /opt/nvidia/deepstream/deepstream/lib)
find_library(nvds_plugin_LIBRARY NAMES nvds_service_maker_utils HINTS /opt/nvidia/deepstream/deepstream/lib)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GLIB REQUIRED glib-2.0)

if (nvds_service_maker_INCLUDE_DIR AND nvds_service_maker_LIBRARY AND nvds_plugin_LIBRARY)
    set(nvds_service_maker_FOUND TRUE)
    set(DEEPSTREAM_SDK_INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../includes")
    add_library(nvds_service_maker SHARED IMPORTED)
    target_include_directories(nvds_service_maker INTERFACE "${nvds_service_maker_INCLUDE_DIR}/includes" ${DEEPSTREAM_SDK_INC_DIR})
    set_target_properties(nvds_service_maker PROPERTIES IMPORTED_LOCATION "${nvds_service_maker_LIBRARY}")
    target_compile_features(nvds_service_maker INTERFACE cxx_std_17)
    add_library(nvds_service_maker_utils STATIC IMPORTED)
    set_target_properties(nvds_service_maker_utils PROPERTIES IMPORTED_LOCATION "${nvds_plugin_LIBRARY}")
else ()
    set(nvds_service_maker_FOUND FALSE)
endif ()