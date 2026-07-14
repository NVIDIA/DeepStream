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

PKGS:= glib-2.0 gobject-2.0 json-glib-1.0 uuid
PKGS += gstreamer-1.0 gstreamer-base-1.0 gstreamer-audio-1.0 gstreamer-video-1.0
TARGET_DEVICE = $(shell gcc -dumpmachine | cut -f1 -d -)

CXX:= g++

SRCS:= $(wildcard *.c)
SRCS += $(wildcard *.cpp)

INCS:= $(wildcard *.h)
LIB:=libnvdsgst_multistream.so

NVDS_VERSION:=9.1

CFLAGS+= -fPIC -DDS_VERSION=\"9.1.0\" \
	 -I /usr/local/cuda-$(CUDA_VER)/include \
	 -I ../../../includes \
	 -I ../../utils/nvstreammux/include \
	 -std=c++17 -Wall -Werror

GST_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/gst-plugins/
LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/

ifeq ($(TARGET_DEVICE),aarch64)
	LIBS := -L/usr/lib/aarch64-linux-gnu/tegra
endif

LIBS+= -shared -Wl,-no-undefined \
	-L$(GST_INSTALL_DIR) -lnvdsgst_multistream_legacy \
	-L/usr/local/cuda-$(CUDA_VER)/lib64/ -lcudart -ldl -lpthread \
	-L../../utils/nvstreammux/ -lnvds_streammux \
	-L$(LIB_INSTALL_DIR) -lnvdsgst_helper -lnvdsgst_meta -lnvds_meta \
	-lnvbufsurface -lnvbufsurftransform -lgstnvdsseimeta \
	-lyaml-cpp -lm \
	-lnvds_nvtxhelper -lnvdsbufferpool\
	-Wl,-rpath,$(LIB_INSTALL_DIR)

OBJS:= $(SRCS:.cpp=.o)

CFLAGS+=$(shell pkg-config --cflags $(PKGS))
LIBS+=$(shell pkg-config --libs $(PKGS))

all: $(LIB)

%.o: %.cpp $(INCS) Makefile
	@echo $(CFLAGS)
	$(CXX) -c -o $@ $(CFLAGS) $<

$(LIB): $(OBJS) $(DEP) Makefile
	@echo $(CFLAGS)
	$(CXX) -o $@ $(OBJS) $(LIBS)

install: $(LIB)
	cp -rv $(LIB) $(GST_INSTALL_DIR)

clean:
	rm -rf $(OBJS) $(LIB)
