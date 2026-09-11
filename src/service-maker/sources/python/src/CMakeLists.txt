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

function(check_variable_set variable_name default_value)
        if (NOT DEFINED ${variable_name})
                set(${variable_name} ${default_value} PARENT_SCOPE)
        endif()
endfunction()

project(pyservicemaker)

check_variable_set(PYTHON_MAJOR_VERSION 3)
check_variable_set(PYTHON_MINOR_VERSION 12)

# pybind11 and dlpack are installed as headers by
# scripts/install_opensource_deps.sh; override if they live elsewhere.
check_variable_set(PYBIND11_ROOT /opt/pybind11)
check_variable_set(DLPACK_ROOT /opt/dlpack)

# Service Maker public headers and the shared DeepStream headers, both in-repo.
# build.sh passes these as absolute paths because the wheel is built from a
# relocated sdist copy; the defaults below suit configuring this directory
# directly with cmake.
check_variable_set(SM_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../includes)
check_variable_set(DS_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../includes)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_SHARED_LINKER_FLAGS "-shared")

set(PYTHON_VERSION ${PYTHON_MAJOR_VERSION}.${PYTHON_MINOR_VERSION})

project(_pydeepstream)

add_compile_options(-fPIC -DHAVE_CONFIG_H -DHAVE_X11 -std=c++17 -Wall -Wno-deprecated-declarations)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED gstreamer-1.0)
pkg_check_modules(GLIB REQUIRED glib-2.0)

include_directories(
    ./bindings/
    ./bindings/docstrings
    ${PYBIND11_ROOT}/include
    ${DLPACK_ROOT}/include
    /usr/include/python${PYTHON_VERSION}
    ${GST_INCLUDE_DIRS}
    ${GLIB_INCLUDE_DIRS}
    ${SM_INCLUDE_DIR}
    ${DS_INCLUDE_DIR}
)

add_library(_pydeepstream SHARED ./bindings/main.cpp)

set(PYTHON_LIB python${PYTHON_VERSION})

set_target_properties(_pydeepstream PROPERTIES
    PREFIX ""
    OUTPUT_NAME "_pydeepstream"
    SUFFIX ".so"
    INSTALL_RPATH "/opt/nvidia/deepstream/deepstream/lib"
    BUILD_WITH_INSTALL_RPATH TRUE
)

# The core library and its static helper archive are installed by the
# service-maker build stage before this runs.
check_variable_set(DS_LIB_DIR /opt/nvidia/deepstream/deepstream/lib)

target_link_directories(_pydeepstream PRIVATE ${DS_LIB_DIR})

# Link libraries - only link the necessary DeepStream libraries
target_link_libraries(_pydeepstream PRIVATE 
    ${PYTHON_LIB}
    nvds_service_maker
    nvds_service_maker_utils
    gstreamer-1.0
    glib-2.0
)