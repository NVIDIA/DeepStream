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

CUDA_VER?=
ifeq ($(CUDA_VER),)
  $(error "CUDA_VER is not set")
endif

NVDS_VERSION?=9.0

LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/
APP_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/bin/
SOURCE_DIR?=$(realpath ../../../../)

DS_CAN_CONTEXT_LIB:= libnvds_can_context.a
APP:= deepstream-can-orientation-app
include ../Rules.mk

CC = g++

CFLAGS:= -g -O0 -fvisibility=hidden -Wall -Werror
debug: CFLAGS+= -g -O0


APP_SRCS:= deepstream_can_main.cpp
DS_CAN_CONTEXT_SRCS:= deepstream_can_context.cpp
APP_INCS:= $(wildcard *.h)

PKGS:= gstreamer-1.0 yaml-cpp

CFLAGS+= -I../../../../includes \
         -I../../../../src/utils \
         -I /usr/local/cuda-$(CUDA_VER)/include \
         -fPIC -std=c++14

CFLAGS+= $(shell pkg-config --cflags $(PKGS))

LIBS:= $(shell pkg-config --libs $(PKGS))

LIBS+= -L/usr/local/cuda-$(CUDA_VER)/lib64/ -lnvdsgst_helper -lm \
-L/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib -Wl,-rpath-link,/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib -lnvdsgst_3d_gst \
-L/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib -Wl,-rpath-link,/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib -lnvdsgst_meta -lnvds_meta \
		-lcuda -Wl,-rpath,$(LIB_INSTALL_DIR) \
		-lgstapp-1.0 -ldl -lcuda -lcudart -pthread

APP_OBJS:= $(APP_SRCS:.cpp=.o)
APP_OBJS := $(addprefix $(OBJ_DIR)/,$(APP_OBJS))
DS_CAN_CONTEXT_OBJS:= $(addprefix $(OBJ_DIR)/,$(DS_CAN_CONTEXT_SRCS:.cpp=.o))

debug: $(BIN_DIR)/$(APP)
all: $(BIN_DIR)/$(APP)

$(OBJ_DIR)/%.o: %.cpp $(APP_INCS)
	@mkdir -p $(OBJ_DIR)
	$(CC) -c -o $@ $(CFLAGS) $<

$(DS_CAN_CONTEXT_LIB): $(DS_CAN_CONTEXT_OBJS)
	$(AR) rcs $@ $(DS_CAN_CONTEXT_OBJS)
	ranlib $@

$(BIN_DIR)/$(APP): $(APP_OBJS) $(DS_CAN_CONTEXT_LIB)
	@mkdir -p $(BIN_DIR)
	$(CC) -o $(BIN_DIR)/$(APP) $(APP_OBJS) -L. -l:$(DS_CAN_CONTEXT_LIB) $(LIBS)

install: $(BIN_DIR)/$(APP)

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)/$(APP) $(DS_CAN_CONTEXT_LIB)

.PHONY: all
