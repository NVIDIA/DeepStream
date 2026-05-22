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