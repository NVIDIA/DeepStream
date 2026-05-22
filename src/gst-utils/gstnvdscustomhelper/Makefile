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
SRCS:= gst-nvdscustommessage.c gst-nvdscustomevent.c \
        gst-nvmultiurisrcbincreator.cpp

INCS:= $(wildcard *.h) \

LIB:=libnvdsgst_customhelper.so
NVDS_VERSION?=9.0
LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/
UTILS_LIB_DIR ?= /opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib

CFLAGS+= -fPIC -DDS_VERSION=\"9.0.0\" \
 	 -I../../../includes \
	 -I../../apps/common/includes/ \
 	 -I/usr/local/cuda-$(CUDA_VER)/include \
	 -I/usr/include/jsoncpp

LIBS := -shared -Wl,-no-undefined \
 	-L/usr/local/cuda-$(CUDA_VER)/lib64/ -lcudart \
 	-pthread

LIBS+= -L$(LIB_INSTALL_DIR) -Wl,-rpath-link,$(LIB_INSTALL_DIR) \
       -lnvdsgst_helper -lnvds_rest_server -ldl \
       -Wl,-rpath,$(LIB_INSTALL_DIR) \
       -ljsoncpp

OBJ_DIR := .build
_OBJS := $(SRCS:.c=.o)
_OBJS := $(_OBJS:.cpp=.o)
OBJS  := $(addprefix $(OBJ_DIR)/,$(_OBJS))

PKGS:= gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0
CFLAGS+= $(shell pkg-config --cflags $(PKGS))
LIBS+= $(shell pkg-config --libs $(PKGS))
LIBS+= -lnvdsgst_meta -lnvds_meta


all: $(UTILS_LIB_DIR)/$(LIB)

$(OBJ_DIR)/%.o: %.c $(INCS) Makefile
	@mkdir -p $(OBJ_DIR)
	$(CXX) -c -o $@ $(CFLAGS) $<

$(OBJ_DIR)/%.o: %.cpp $(INCS) Makefile
	@mkdir -p $(OBJ_DIR)
	$(CXX) -c -o $@ $(CFLAGS) $<

$(UTILS_LIB_DIR)/$(LIB): $(OBJS) $(DEP) Makefile
	@mkdir -p $(UTILS_LIB_DIR)
	$(CXX) -o $@ $(OBJS) $(LIBS)

install: $(UTILS_LIB_DIR)/$(LIB)
	ldconfig

clean:
	rm -rf $(OBJ_DIR) $(UTILS_LIB_DIR)/$(LIB)
