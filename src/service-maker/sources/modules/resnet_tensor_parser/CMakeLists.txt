cmake_minimum_required(VERSION 3.16)

project(ResnetTensorParser)
find_package(nvds_service_maker REQUIRED PATHS "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake")

# Find the GStreamer package
find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0>=1.0 gstreamer-base-1.0>=1.0)

find_package(CUDA REQUIRED)

# Include the GStreamer headers
include_directories(${GSTREAMER_INCLUDE_DIRS})

add_library(resnet_tensor_parser SHARED resnet_tensor_parser.cpp)
target_include_directories(resnet_tensor_parser PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(resnet_tensor_parser PRIVATE  nvds_service_maker nvds_service_maker_utils)
