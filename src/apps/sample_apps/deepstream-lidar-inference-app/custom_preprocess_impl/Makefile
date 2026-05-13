################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
################################################################################

CC:= g++

LIB_INSTALL_DIR ?= /opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/

CFLAGS:= -Wall -std=c++11 -shared -fPIC -Wno-error=deprecated-declarations
CFLAGS+= -I../../../../../includes/ \
	 -I../../../../../includes/nvdsinferserver  \
	 -I/usr/local/cuda/include  \
	 -std=c++14

LIBS:= -lnvinfer_plugin -lnvinfer -L/usr/local/cuda/lib64 -lcudart -lcublas -lstdc++fs
LFLAGS:= -shared -Wl,--start-group $(LIBS) -Wl,--end-group
NVCC:=/usr/local/cuda/bin/nvcc

INCS:= $(wildcard *.h)
SRCFILES:=  $(wildcard ./*.cpp ./*.cu)
TARGET_LIB:= libnvds_lidar_custom_preprocess_impl.so

OBJ_DIR := .build
_OBJS := $(SRCFILES:.cpp=.o)
_OBJS := $(_OBJS:.cu=.o)
TARGET_OBJS := $(addprefix $(OBJ_DIR)/,$(_OBJS))

all: $(TARGET_LIB)

$(OBJ_DIR)/%.o: %.cpp $(INCS) Makefile
	@mkdir -p $(OBJ_DIR)
	$(CC) -c -o $@ $(CFLAGS) $<
$(OBJ_DIR)/%.o: %.cu $(INCS) Makefile
	@mkdir -p $(OBJ_DIR)
	$(NVCC) -c -o $@ --compiler-options '-fPIC' $<

$(TARGET_LIB) : $(TARGET_OBJS)
	$(CC) -o $@  $(TARGET_OBJS) $(LFLAGS)

install: $(TARGET_LIB)
	ldconfig

clean:
	rm -rf $(OBJ_DIR) $(TARGET_LIB)
