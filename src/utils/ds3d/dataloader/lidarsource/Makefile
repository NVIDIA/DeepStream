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
CUDA_HOME:=/usr/local/cuda
ifeq ($(CUDA_VER),)
  $(warning CUDA_VER is not set, use default CUDA_HOME)
else
  CUDA_HOME:=/usr/local/cuda-$(CUDA_VER)
endif
$(info CUDA_HOME=$(CUDA_HOME))

NVDS_VERSION?=
DS_PATH=../../../
#DS_PATH=../../../-adfasdf
ifeq ($(NVDS_VERSION),)
  $(info NVDS_VERSION is not set, trying to figure out)
  FULL_PATH:=$(realpath $(DS_PATH))
  ifeq ($(FULL_PATH),)
    $(error $(DS_PATH) is not exist)
  endif
  NVDS_VERSION:=$(shell basename $(FULL_PATH) | cut -f2 -d -)
endif

DS_HOME=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)
LIB_INSTALL_DIR=$(DS_HOME)/lib

$(info NVDS_VERSION=$(NVDS_VERSION))
$(info DS_HOME=$(DS_HOME))

TARGET_LIB:=libnvds_lidarfileread.so

CC = g++

CFLAGS:= -fvisibility=hidden -Wall -Werror


PKGS:= yaml-cpp

CFLAGS+= -I $(DS_HOME)/sources/includes \
	 -I $(DS_HOME)/sources/libs \
	 -I $(CUDA_HOME)/include \
	 -fPIC -std=c++14

CFLAGS+= $(shell pkg-config --cflags $(PKGS))

LIBS:= $(shell pkg-config --libs $(PKGS))

LIBS+= -L$(CUDA_HOME)/lib64/ -lcudart -lm \
		-L$(LIB_INSTALL_DIR) -lnvds_3d_common \
		-lcuda -Wl,-rpath,$(LIB_INSTALL_DIR)


TARGET_LIB_SRCS:= $(wildcard ./*.cpp)
TARGET_LIB_INCS:= $(wildcard ./*.h)
TARGET_LIB_OBJS:= $(TARGET_LIB_SRCS:.cpp=.o)

LFLAGS:= -shared -Wl,--start-group $(LIBS) -Wl,--end-group


all: $(TARGET_LIB)

$(TARGET_LIB): $(TARGET_LIB_OBJS) Makefile
	$(CC) -o $(TARGET_LIB) $(TARGET_LIB_OBJS) $(LFLAGS)


%.o: %.cpp $(TARGET_LIB_INCS) Makefile
	$(CC) -c -o $@ $(CFLAGS) $<

install: $(TARGET_LIB)
	ldconfig

clean:
	rm -rf $(TARGET_LIB_OBJS) $(TARGET_LIB)

