################################################################################
# Copyright (c) 2021-2026, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
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
