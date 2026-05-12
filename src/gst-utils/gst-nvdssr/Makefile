################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

CXX:= gcc
SRCS:= gst-nvdssr.c

INCS:= $(wildcard *.h) \

LIB:=libnvdsgst_smartrecord.so
NVDS_VERSION?=9.0
UTILS_LIB_DIR ?= /opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib

LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/

CFLAGS+= -fPIC -DDS_VERSION=\"9.0.0\" \
	  -I../../../includes \
      -shared

OBJ_DIR := .build
OBJS := $(addprefix $(OBJ_DIR)/,$(SRCS:.c=.o))

PKGS:= gstreamer-pbutils-1.0

CFLAGS+= $(shell pkg-config --cflags $(PKGS))
LIBS+= $(shell pkg-config --libs $(PKGS))
all: $(UTILS_LIB_DIR)/$(LIB)

$(OBJ_DIR)/%.o: %.c $(INCS) Makefile
	@mkdir -p $(OBJ_DIR)
	$(CXX) -c -o $@ $(CFLAGS) $<

$(UTILS_LIB_DIR)/$(LIB): $(OBJS) $(DEP) Makefile
	@mkdir -p $(UTILS_LIB_DIR)
	$(CXX) -o $@ $(OBJS) $(LIBS) $(CFLAGS)

install: $(UTILS_LIB_DIR)/$(LIB)
	ldconfig

clean:
	rm -rf $(OBJ_DIR) $(UTILS_LIB_DIR)/$(LIB)
