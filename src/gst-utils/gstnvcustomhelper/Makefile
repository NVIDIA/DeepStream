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

CXX:= gcc
SRCS:= gst-nvcustomevent.c

INCS:= $(wildcard *.h)
NVDS_VERSION ?= 9.0
UTILS_LIB_DIR ?= /opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib
SO_NAME := libgstnvcustomhelper.so
LIB:= $(UTILS_LIB_DIR)/$(SO_NAME)
OBJ_DIR := .build
OBJS := $(addprefix $(OBJ_DIR)/,$(SRCS:%.c=%.o))
LIB_INSTALL_DIR ?= /opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/
CFLAGS:= -DDS_VERSION=\"9.0.0\"

CFLAGS+= -fPIC

LIBS := -shared -Wl,-no-undefined

LIBS+= -L$(LIB_INSTALL_DIR) -ldl\
        -Wl,-rpath,$(LIB_INSTALL_DIR)

PKGS:= gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0
CFLAGS+= $(shell pkg-config --cflags $(PKGS))
LIBS+= $(shell pkg-config --libs $(PKGS))

all: $(LIB)

$(OBJ_DIR)/%.o: %.c $(INCS) Makefile
	@mkdir -p $(OBJ_DIR)
	$(CXX) -c -o $@ $(CFLAGS) $<

$(LIB): $(OBJS) Makefile
	@mkdir -p $(UTILS_LIB_DIR)
	$(CXX) -o $@ $(OBJS) $(LIBS)

install: $(LIB)
	ldconfig

clean:
	rm -rf $(OBJ_DIR) $(UTILS_LIB_DIR)/$(SO_NAME)
