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

CXX:= gcc

# Set LEGACY_API=1 to build with the legacy Rivermax API (rmax_* functions).
# By default, the new Rivermax API (rmx_* functions) is used.
# Example: export LEGACY_API=1 && make
ifdef LEGACY_API
  SRC_DIR := legacy_api/
else
  SRC_DIR :=
endif

SRCS:= gstnvdsudp.c $(SRC_DIR)gstnvdsudpsrc.c $(SRC_DIR)gstnvdsudpsink.c gstnvdsudpcommon.c
INCS:= gstnvdsudpcommon.h $(wildcard $(SRC_DIR)*.h)
LIB:=libnvdsgst_udp.so

TARGET_DEVICE = $(shell gcc -dumpmachine | cut -f1 -d -)

NVDS_VERSION:=9.1

CFLAGS+= -fPIC -DDS_VERSION=\"9.1.0\" \
	 -I /usr/local/cuda-$(CUDA_VER)/include \
	 -I ../../../includes \
	 -I /usr/include/mellanox/

GST_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/gst-plugins/
LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/

LIBS := -shared -Wl,-no-undefined \
	-L/usr/local/cuda-$(CUDA_VER)/lib64/ -lcudart

ifeq ($(TARGET_DEVICE),aarch64)
    LIBS+= -L/usr/lib/aarch64-linux-gnu/ -lrivermax
else
    LIBS+= -L/usr/lib/x86_64-linux-gnu/ -lrivermax
endif

LIBS+= -L$(LIB_INSTALL_DIR) -lnvbufsurface -lnvdsbufferpool -ldl -lm -lpthread -lcuda -Wl,-rpath,$(LIB_INSTALL_DIR)

OBJS:= $(SRCS:.c=.o)

PKGS:= gstreamer-1.0 gstreamer-base-1.0 gstreamer-sdp-1.0 gio-2.0
CFLAGS+=$(shell pkg-config --cflags $(PKGS))
LIBS+=$(shell pkg-config --libs $(PKGS))

all: $(LIB)

%.o: %.c $(INCS) Makefile
	$(CXX) -c -o $@ $(CFLAGS) $<

$(LIB): $(OBJS) $(DEP) Makefile
	$(CXX) -o $@ $(OBJS) $(LIBS)

install: $(LIB)
	cp -rv $(LIB) $(GST_INSTALL_DIR)

clean:
	rm -rf $(OBJS) $(LIB)