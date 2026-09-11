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

CXX:= g++

NVDS_VERSION:=9.1

GST_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/gst-plugins/
LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/

CFLAGS:= -fPIC -DIS_DESKTOP=1 -DNEW_METADATA -O3 -std=c++17 -Wall \
	 -DDS_VERSION=\"9.1.1\"

CFLAGS+= -I../../../includes \
	 -I../../utils/nvdsmeta \
	 -I../../utils/nvstreammux/include \
	 -I/usr/local/cuda-$(CUDA_VER)/include

PKGS:= gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0 glib-2.0 gobject-2.0 json-glib-1.0
CFLAGS+= $(shell pkg-config --cflags $(PKGS))

SRCS:= gstnvmultistream.cpp gstnvstreamdemux.cpp gstnvstreammux.cpp \
	gstnvstreammuxpool.cpp gstnvstreampad.cpp

LIB:= libnvdsgst_multistream_legacy.a

OBJS:= $(SRCS:.cpp=.o)

all: $(LIB)

%.o: %.cpp Makefile
	$(CXX) -c -o $@ $(CFLAGS) $<

$(LIB): $(OBJS) Makefile
	ar rcs $@ $(OBJS)

install: $(LIB)
	cp -rv $(LIB) $(GST_INSTALL_DIR)

clean:
	rm -rf $(OBJS) $(LIB)
