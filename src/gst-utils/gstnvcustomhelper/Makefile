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

CXX:= gcc
SRCS:= gst-nvcustomevent.c

INCS:= $(wildcard *.h)
OUT_DIR ?= .
SO_NAME := libgstnvcustomhelper.so
LIB:= $(OUT_DIR)/$(SO_NAME)
OBJS :=  $(SRCS:%.c=$(OUT_DIR)/%.o)

TARGET_DEVICE = $(shell gcc -dumpmachine | cut -f1 -d -)
ifeq ($(TARGET_DEVICE),aarch64)
  LIB_INSTALL_DIR?=/usr/lib/aarch64-linux-gnu/tegra/
  CFLAGS:=
else
  NVDS_VERSION ?=9.1
  LIB_INSTALL_DIR ?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/
  CFLAGS:= -DDS_VERSION=\"9.1.0\"
endif

CFLAGS+= -fPIC

LIBS := -shared -Wl,-no-undefined

LIBS+= -L$(LIB_INSTALL_DIR) -ldl\
        -Wl,-rpath,$(LIB_INSTALL_DIR)

PKGS:= gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0
CFLAGS+= $(shell pkg-config --cflags $(PKGS))
LIBS+= $(shell pkg-config --libs $(PKGS))

all: $(LIB)

$(OUT_DIR)/%.o: %.c $(INCS) Makefile
	@echo $(CFLAGS)
	$(CXX) -c -o $@ $(CFLAGS) $<

$(LIB): $(OBJS) Makefile
	$(CXX) -o $@ $(OBJS) $(LIBS)

install: $(LIB)
	cp -rv $(LIB) $(LIB_INSTALL_DIR)

clean:
	rm -rf $(OBJS) $(LIB)
