# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

CXX ?= g++
DS ?= /opt/nvidia/deepstream/deepstream
CUDA_INC ?= /usr/local/cuda/include
TARGET := libnvdsinfer_rtdetr.so
CFLAGS := -fPIC -shared -std=c++14 -O2 -I$(DS)/sources/includes -I$(CUDA_INC)
$(TARGET): rtdetr_parser.cpp
	$(CXX) $(CFLAGS) -o $@ rtdetr_parser.cpp
clean:
	rm -f $(TARGET)
.PHONY: clean
