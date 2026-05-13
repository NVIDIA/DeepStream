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

NVDS_VERSION?=9.0

LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/
APP_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/bin/

APP:= deepstream-3d-depth-camera
include ../Rules.mk

CC = g++

CFLAGS:= -fvisibility=hidden -Wall -Werror
debug: CFLAGS+= -g -O0


APP_SRCS:= $(wildcard *.cpp)
APP_INCS:= $(wildcard *.h)

PKGS:= gstreamer-1.0 yaml-cpp

CFLAGS+= -I../../../../includes \
         -I../../../libs \
         -I /usr/local/cuda-$(CUDA_VER)/include \
         -fPIC -std=c++14

CFLAGS+= $(shell pkg-config --cflags $(PKGS))

LIBS:= $(shell pkg-config --libs $(PKGS))

LIBS+= -L/usr/local/cuda-$(CUDA_VER)/lib64/ -lnvdsgst_helper -lm \
-L/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib -Wl,-rpath-link,/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib -lnvdsgst_3d_gst \
-L/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib -Wl,-rpath-link,/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib -lnvdsgst_meta -lnvds_meta \
		-lcuda -Wl,-rpath,$(LIB_INSTALL_DIR) \
		-lgstapp-1.0 -ldl
APP_OBJS:= $(APP_SRCS:.cpp=.o)
APP_OBJS := $(addprefix $(OBJ_DIR)/,$(APP_OBJS))

debug: $(BIN_DIR)/$(APP)
all: $(BIN_DIR)/$(APP)

$(OBJ_DIR)/%.o: %.cpp $(APP_INCS) Makefile
	@mkdir -p $(OBJ_DIR)
	$(CC) -c -o $@ $(CFLAGS) $<

$(BIN_DIR)/$(APP): $(APP_OBJS) Makefile
	@mkdir -p $(BIN_DIR)
	$(CC) -o $(BIN_DIR)/$(APP) $(APP_OBJS) $(LIBS)

install: $(BIN_DIR)/$(APP)

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)/$(APP)

.PHONY: all
