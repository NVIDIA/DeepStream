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

CUDA_VER?=
ifeq ($(CUDA_VER),)
  $(error "CUDA_VER is not set")
endif

CC:= g++

PKGS:= gstreamer-1.0

NVDS_VERSION:=9.1

LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/

CFLAGS:= -fPIC -std=c++11

CFLAGS+= -I../../../includes \
	 -I/usr/local/cuda-$(CUDA_VER)/include
CFLAGS+= $(shell pkg-config --cflags $(PKGS))

SRCFILES:= nvds_version.cpp nvds_mask_utils.cpp

TARGET_LIB:= libnvds_utils.so

LIBS:= -shared -Wl,--no-undefined -ldl \
	-L/usr/local/cuda-$(CUDA_VER)/lib64/ -lcudart \
	-lnppc -lnppig -lnppitc -lnppial \
	-lnvinfer
LIBS+= $(shell pkg-config --libs $(PKGS))

all: $(TARGET_LIB)

$(TARGET_LIB) : $(SRCFILES)
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)

install: $(TARGET_LIB)
	cp -rv $(TARGET_LIB) $(LIB_INSTALL_DIR)

clean:
	rm -rf $(TARGET_LIB)
