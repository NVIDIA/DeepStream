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
