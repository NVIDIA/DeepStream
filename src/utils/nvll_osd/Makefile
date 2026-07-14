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

NVCC:=/usr/local/cuda-$(CUDA_VER)/bin/nvcc
CXX:= g++
SRCS:= nvll_osd.cpp nvll_osd_int.cpp nvll_obb.cpp cuosd.cpp \
	textbackend/backend.cpp textbackend/pango-cairo.cpp cuda/drawBbox.cu \
	cuda/cuosd_kernel.cu cuda/obb_scanline.cu cuda/drawOBB.cu

INCS:= $(wildcard *.h)
LIB:= libnvds_osd.so
PKGS:= glib-2.0 gobject-2.0
NVDS_VERSION:=9.1

LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/

CFLAGS:= -fPIC -Wno-deprecated-declarations -std=c++11 -DENABLE_TEXT_BACKEND_PANGO \
	 -I/usr/include/harfbuzz \
	 -I/usr/include/cairo -I/usr/include/pango-1.0 -I. \
	 -Icuda \
	 -I /usr/local/cuda-$(CUDA_VER)/include \
	 -I ../../../includes -DNDEBUG

CFLAGS+= $(shell pkg-config --cflags $(PKGS))

LIBS := -shared -Wl,-no-undefined \
	-lpthread \
	-L/usr/local/cuda-$(CUDA_VER)/lib64/ -lcudart

LIBS+= -L$(LIB_INSTALL_DIR) -lpango-1.0 -lcairo -lpangocairo-1.0 -lgobject-2.0 \
       -lnvds_utils -lnvds_batch_jpegenc \
       -Wl,-rpath,$(LIB_INSTALL_DIR) \
       -lnvbufsurface -lnvbufsurftransform -lcuda -ljpeg

LIBS+= $(shell pkg-config --libs $(PKGS))

OBJS:= $(SRCS:.cpp=.o)
OBJS:= $(OBJS:.cu=.o)

all: $(LIB)

%.o: %.cpp $(INCS) Makefile
	$(CXX) -c -o $@ $(CFLAGS) $<

%.o: %.cu $(INCS) Makefile
	@echo $(CFLAGS)
	$(NVCC) -c -o $@ --compiler-options '-fPIC' $<

$(LIB): $(OBJS) $(DEP) Makefile
	@echo $(CFLAGS)
	$(CXX) -o $@ $(OBJS) $(LIBS)

install: $(LIB)
	cp -rv $(LIB) $(LIB_INSTALL_DIR)

clean:
	rm -rf $(OBJS) $(LIB)
