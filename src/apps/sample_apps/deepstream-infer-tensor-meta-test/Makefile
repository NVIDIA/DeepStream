################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

WITH_OPENCV:=0

APP:= deepstream-infer-tensor-meta-app
include ../Rules.mk
CC = g++

NVDS_VERSION?=9.0

LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/
APP_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/bin/

SRCS:= $(wildcard *.cpp)
INCS:= $(wildcard *.h)

PKGS:= gstreamer-1.0

CFLAGS:= -fPIC -std=c++11    \
         -I ../../../../includes \
         -I /usr/local/cuda-$(CUDA_VER)/include \
	 -I /usr/include/opencv4

CFLAGS+= $(shell pkg-config --cflags $(PKGS))
LIBS:= $(shell pkg-config --libs $(PKGS))

ifeq ($(WITH_OPENCV),1)
 CFLAGS+= -DWITH_OPENCV
endif

LIBS+= -L/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib -Wl,-rpath-link,/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib -lnvdsgst_meta -lnvds_meta -lnvds_inferutils -lnvds_utils -lm \
       -L/usr/local/cuda-$(CUDA_VER)/lib64/ -lcudart     \
       -lcuda -Wl,-rpath,$(LIB_INSTALL_DIR) \
       -lnvds_infercustomparser

ifeq ($(WITH_OPENCV),1)
LIBS += -lopencv_objdetect -lopencv_imgproc -lopencv_core
endif


OBJS:= $(SRCS:.cpp=.o)
OBJS := $(addprefix $(OBJ_DIR)/,$(OBJS))

all: $(BIN_DIR)/$(APP)

$(OBJ_DIR)/%.o: %.cpp $(INCS) Makefile
	@mkdir -p $(OBJ_DIR)
	$(CC) -c -o $@ $(CFLAGS) $<

$(BIN_DIR)/$(APP): $(OBJS) Makefile
	@mkdir -p $(BIN_DIR)
	$(CC) -o $(BIN_DIR)/$(APP) $(OBJS) $(LIBS)

install: $(BIN_DIR)/$(APP)

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)/$(APP)
