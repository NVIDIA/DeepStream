################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
################################################################################

CUDA_VER?=
ifeq ($(CUDA_VER),)
  $(error "CUDA_VER is not set")
endif


LIB_INSTALL_DIR ?= /opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/
CC:= g++

NVCC:=/usr/local/cuda-$(CUDA_VER)/bin/nvcc

# Change to your deepstream SDK includes
CFLAGS+= -I../../../../../includes \
         -I/usr/local/cuda/include \
         -I../../../../../includes/nvdsinferserver \
         -I../../../nvdsinferserver \
         -I../../../../../includes \
         -I../../../../../includes/ds3d \
         -I../../../../../includes/ds3d/common \
         -I../../../../../includes/ds3d/common/helper

CFLAGS+= -Wall -std=c++17 -shared -fPIC

LIBS+= -L/usr/local/cuda/lib64 -lcudart -lcublas -lyaml-cpp

TARGET_LIB:= libnvds_3d_lidar_preprocess_datafilter.so
TARGET_LIB_SRCS:= lidar_preprocess_filter_impl.cpp    \
                  voxelization.cu                     \
                  ../../../nvdsinferserver/infer_utils.cpp \
                  ../../../nvdsinferserver/infer_cuda_utils.cpp

TARGET_LIB_INCS:= $(wildcard ./*.h)
TARGET_LIB_OBJS:= $(TARGET_LIB_SRCS:.cpp=.o)
TARGET_LIB_OBJS:= $(TARGET_LIB_OBJS:.cu=.o)

LDFLAGS:= -shared -Wl,--no-undefined -Wl,--start-group $(LIBS) -Wl,--end-group

all: $(TARGET_LIB)

$(TARGET_LIB): $(TARGET_LIB_OBJS) Makefile
	$(CC) -o $(TARGET_LIB) $(TARGET_LIB_OBJS) $(LDFLAGS)

%.o: %.cpp $(TARGET_LIB_INCS) Makefile
	$(CC) -c -o $@ $(CFLAGS) $<

%.o: %.cu $(INCS) Makefile
	@echo $(CFLAGS)
	$(NVCC) -c -o $@ --compiler-options '-fPIC -I../../../../../includes/ds3d/common/helper' $<

install: $(TARGET_LIB)
	ldconfig

clean:
	rm -rf $(TARGET_LIB_OBJS) $(TARGET_LIB)

