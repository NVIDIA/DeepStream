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

APP:= deepstream-3d-action-recognition
include ../Rules.mk

CC = g++

CFLAGS:= -fvisibility=hidden -Wall -Werror


APP_SRCS:= $(wildcard *.cpp)
APP_INCS:= $(wildcard *.h)

PKGS:= gstreamer-1.0

CFLAGS+= -I../../../../includes \
         -I ../../../gst-plugins/gst-nvdspreprocess/include \
	 -I /usr/local/cuda-$(CUDA_VER)/include \
         -fPIC -std=c++14

CFLAGS+= $(shell pkg-config --cflags $(PKGS))

LIBS:= $(shell pkg-config --libs $(PKGS))

LIBS+= -L/usr/local/cuda-$(CUDA_VER)/lib64/ -lcudart -lnvdsgst_helper -lm \
-L/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib -Wl,-rpath-link,/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib -lnvdsgst_meta -lnvds_meta \
		-lcuda -Wl,-rpath,$(LIB_INSTALL_DIR)
APP_OBJS:= $(APP_SRCS:.cpp=.o)
APP_OBJS := $(addprefix $(OBJ_DIR)/,$(APP_OBJS))

SUBFOLDER:=custom_sequence_preprocess

all: $(BIN_DIR)/$(APP) $(SUBFOLDER)

$(OBJ_DIR)/%.o: %.cpp $(APP_INCS) Makefile
	@mkdir -p $(OBJ_DIR)
	$(CC) -c -o $@ $(CFLAGS) $<

$(BIN_DIR)/$(APP): $(APP_OBJS) Makefile
	@mkdir -p $(BIN_DIR)
	$(CC) -o $(BIN_DIR)/$(APP) $(APP_OBJS) $(LIBS)

$(SUBFOLDER):
	$(MAKE) -C $@ $(MAKECMDGOALS)

install: $(BIN_DIR)/$(APP) $(SUBFOLDER)

clean: $(SUBFOLDER)
	rm -rf $(OBJ_DIR) $(BIN_DIR)/$(APP)

.PHONY: all $(SUBFOLDER)
