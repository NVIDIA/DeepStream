# SPDX-FileCopyrightText: Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

NVCC:=/usr/local/cuda-$(CUDA_VER)/bin/nvcc
CXX:= g++

NVDS_VERSION:=9.1

GST_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/gst-plugins/
LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/

CFLAGS:= -fPIC -std=c++11 -DDS_VERSION=\"9.1.1\"

CFLAGS+= -I. -I../../../includes \
	 -I/usr/local/cuda-$(CUDA_VER)/include

NVCC_CFLAGS+= -I. -I../../../includes \
	 -I/usr/local/cuda-$(CUDA_VER)/include

PKGS:= gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0
CFLAGS+= $(shell pkg-config --cflags $(PKGS))

SRCS:= gstnvofvisual.cpp nvofvisual_draw_cuda.cu
LIB:= libnvdsgst_ofvisual.so

LIBS:= -shared -Wl,-no-undefined \
	-L/usr/local/cuda-$(CUDA_VER)/lib64/ -lcudart -lcuda \
	-L$(LIB_INSTALL_DIR) -lnvdsgst_helper -lnvbufsurface \
	-lnvdsbufferpool -lnvds_meta -lnvdsgst_meta -lnvds_nvtxhelper \
	-Wl,-rpath,$(LIB_INSTALL_DIR) -ldl
LIBS+= $(shell pkg-config --libs $(PKGS))

OBJS:= gstnvofvisual.o nvofvisual_draw_cuda.o

all: $(LIB)

%.o: %.cpp Makefile
	$(CXX) -c -o $@ $(CFLAGS) $<

%.o: %.cu Makefile
	$(NVCC) -c -o $@ $(NVCC_CFLAGS) --compiler-options '-fPIC' $<

$(LIB): $(OBJS) Makefile
	$(CXX) -o $@ $(OBJS) $(LIBS)

install: $(LIB)
	cp -rv $(LIB) $(GST_INSTALL_DIR)

clean:
	rm -rf $(OBJS) $(LIB)
