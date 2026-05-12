################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

CXX:= g++
SRCS:= $(wildcard *.cpp)
LIB:=libnvds_streammux.a
LIB_DIR = /opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib
OBJ_DIR := .build
include ../../Rules.mk
PKGS:= glib-2.0 gobject-2.0 json-glib-1.0 uuid

NVDS_VERSION?=9.0

LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/

CFLAGS+= -fPIC -Wno-deprecated-declarations -std=c++11 \
	 -I /usr/local/cuda-$(CUDA_VER)/include \
	 -I ../../../includes -DNDEBUG \
	 -I ./include

CFLAGS+= $(shell pkg-config --cflags $(PKGS))

OBJS:= $(SRCS:.cpp=.o)
OBJS := $(addprefix $(OBJ_DIR)/,$(OBJS))

all: $(LIB_DIR)/$(LIB)

$(OBJ_DIR)/%.o: %.cpp $(INCS) Makefile
	@mkdir -p $(OBJ_DIR)
	$(CXX) -c -o $@ $(CFLAGS) $<

$(LIB_DIR)/$(LIB): $(OBJS) $(DEP) Makefile
	@mkdir -p $(LIB_DIR)
	@echo $(CFLAGS)
	ar rcs $@ $(OBJS)

clean:
	rm -rf $(OBJ_DIR) $(LIB_DIR)/$(LIB)
