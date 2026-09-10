# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

CUDA_VER?=
ifeq ($(CUDA_VER),)
  $(error "CUDA_VER is not set")
endif

CC:= g++

NVDS_VERSION:= 9.1

TARGET_DEVICE = $(shell gcc -dumpmachine | cut -f1 -d -)

LIB_INSTALL_DIR?= /opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/

CFLAGS:= -fPIC -std=c++11 -Wall \
	-I ../../../includes \
	-I /usr/local/cuda-$(CUDA_VER)/include \
	$(shell pkg-config --cflags glib-2.0 gobject-2.0)

SRCFILES:= nvds_stats.cpp

TARGET_LIB:= libnvds_stats.so

NVML_LIB:= -lnvidia-ml

# Jetson needs its architecture-specific NVML location. SBSA provides the
# standard linker name, so preserve the default when build.sh sets this flag.
ifeq ($(TARGET_DEVICE),aarch64)
ifneq ($(AARCH64_IS_SBSA),1)
	NVML_LIB:= /usr/lib/aarch64-linux-gnu/nvidia/libnvidia-ml.so.1
endif
endif

LIBS:= -shared -Wl,-no-undefined \
	$(NVML_LIB) \
	$(shell pkg-config --libs glib-2.0 gobject-2.0)

all: $(TARGET_LIB)

$(TARGET_LIB): $(SRCFILES)
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)

install: $(TARGET_LIB)
	cp -rv $(TARGET_LIB) $(LIB_INSTALL_DIR)

clean:
	rm -rf $(TARGET_LIB)
