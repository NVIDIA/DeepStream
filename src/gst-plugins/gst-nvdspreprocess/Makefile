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

WITH_OPENCV:=0

CUDA_VER?=
# Require CUDA_VER for every goal except a clean-only invocation.
# A bare `make` has empty MAKECMDGOALS and still checks.
ifneq ($(filter-out clean,$(MAKECMDGOALS)),)
  _DS_NEED_CUDA_VER := 1
else ifeq ($(MAKECMDGOALS),)
  _DS_NEED_CUDA_VER := 1
endif
ifeq ($(_DS_NEED_CUDA_VER),1)
ifeq ($(CUDA_VER),)
  $(error "CUDA_VER is not set")
endif
endif

TARGET_DEVICE = $(shell gcc -dumpmachine | cut -f1 -d -)
WSL:=$(shell uname -a | grep -io "microsoft" | tr '[:upper:]' '[:lower:]')

CXX:= g++

SRCS:= gstnvdspreprocess.cpp gstnvdspreprocess_allocator.cpp \
       nvdspreprocess_property_parser.cpp nvdspreprocess_yaml_parser.cpp

INCS:= $(wildcard *.h)
LIB:=libnvdsgst_preprocess.so

NVDS_VERSION:=9.2

CFLAGS+= -fPIC -DHAVE_CONFIG_H -std=c++17 -Wall -Werror -DDS_VERSION=\"9.2.1\" \
	 -I /usr/local/cuda-$(CUDA_VER)/include \
	 -I include \
	 -I ../../../includes

GST_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/gst-plugins/
LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/

LIBS := -shared -Wl,-no-undefined \
	-L/usr/local/cuda-$(CUDA_VER)/lib64/ -lcudart -ldl \
	-L$(LIB_INSTALL_DIR) -lnvdsgst_helper -lnvdsgst_customhelper -lnvdsgst_meta -lnvds_meta -lnvbufsurface -lnvbufsurftransform\
	-lcuda -lyaml-cpp -Wl,-rpath,$(LIB_INSTALL_DIR)

ifeq ("microsoft", "$(WSL)")
  LIBS+= -L/usr/lib/wsl/lib/
endif

OBJS:= $(SRCS:.cpp=.o)

PKGS:= gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0

ifeq ($(WITH_OPENCV),1)
  CFLAGS+= -DWITH_OPENCV \
    -I /usr/include/opencv4
  LIBS+= -lopencv_imgproc -lopencv_core -lopencv_imgcodecs
  PKGS+= opencv4
endif

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
