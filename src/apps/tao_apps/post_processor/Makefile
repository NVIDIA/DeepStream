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

LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream/lib/

CC:= g++

# Base CFLAGS
CFLAGS+= -Wall -std=c++11 -shared -fPIC

# Deepstream SDK includes and CUDA includes
# Change to your path if needed
CFLAGS+= -I../../../../includes \
	     -I/usr/local/cuda-$(CUDA_VER)/include

# Debug logger header
CFLAGS+= -I.

# Libraries
LIBS+= -lnvinfer -L/usr/local/cuda-$(CUDA_VER)/lib64 -lcudart -lcublas
LFLAGS:= -Wl,--start-group $(LIBS) -Wl,--end-group

# Source files and targets
SRCFILES:= nvdsinfer_custombboxparser_tao.cpp
SRCFILES+= debug_logger_raii.cpp debug_logger_tensor.hpp
TARGET_LIB:= libnvds_infercustomparser_tao.so
TARGET_LIB_DEBUG:= libnvds_infercustomparser_tao_debug.so

# Release build flags (default)
# = $(CFLAGS) -O2
CFLAGS_RELEASE = $(CFLAGS)

# Debug build flags
CFLAGS_DEBUG = $(CFLAGS) -g -DDEBUG -O0

# Default target
all: $(TARGET_LIB)

# Debug target
debug: $(TARGET_LIB_DEBUG)

# Release build
$(TARGET_LIB) : $(SRCFILES)
	$(CC) -o $@ $^ $(CFLAGS_RELEASE) $(LFLAGS)

# Debug build
$(TARGET_LIB_DEBUG): $(SRCFILES)
	$(CC) -o $@ $^ $(CFLAGS_DEBUG) $(LFLAGS)

install: $(TARGET_LIB)
	@mkdir -p $(LIB_INSTALL_DIR)
	cp -rv $(TARGET_LIB) $(LIB_INSTALL_DIR)

clean:
	rm -rf $(TARGET_LIB) $(TARGET_LIB_DEBUG)

.PHONY: all debug install clean
