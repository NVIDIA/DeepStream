# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Override UCX_PATH if UCX is installed elsewhere.
UCX_PATH ?= /usr

CXX:= g++

NVDS_VERSION:=9.1

GST_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/gst-plugins/
LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/

CFLAGS:= -fPIC -std=c++17 -DDS_VERSION=\"9.1.1\"

CFLAGS+= -I../../../includes \
	 -I../../utils/nvds_mem_allocator \
	 -I$(UCX_PATH)/include \
	 -I/usr/local/cuda-$(CUDA_VER)/include

PKGS:= gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0
CFLAGS+= $(shell pkg-config --cflags $(PKGS))

SRCS:= gstucx.c gstucxclientsrc.c gstucxplugin.c gstucxserversink.c \
	gstucxclientsink.c gstucxserversrc.c gstucxsharedtaskpool.c

LIB:= libnvdsgst_ucx.so

LIBS:= -shared -Wl,-no-undefined \
	-L$(UCX_PATH)/lib -lucs -lucp \
	-L/usr/local/cuda-$(CUDA_VER)/lib64/ -lcudart \
	-L$(LIB_INSTALL_DIR) -lnvdsbufferpool -lnvdsgst_helper \
	-lnvdsgst_audio -lnvds_meta -lnvdsgst_meta -lnvds_nvtxhelper \
	-Wl,-rpath,$(LIB_INSTALL_DIR) -ldl -lpthread
LIBS+= $(shell pkg-config --libs $(PKGS))

OBJS:= $(SRCS:.c=.o)

all: $(LIB)

%.o: %.c Makefile
	$(CXX) -c -o $@ $(CFLAGS) $<

$(LIB): $(OBJS) Makefile
	$(CXX) -o $@ $(OBJS) $(LIBS)

install: $(LIB)
	cp -rv $(LIB) $(GST_INSTALL_DIR)

clean:
	rm -rf $(OBJS) $(LIB)
