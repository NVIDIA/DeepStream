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
NVCC:=/usr/local/cuda-$(CUDA_VER)/bin/nvcc

# CUDA code generation flags
GPU_ARCHS	= 75 80 86 89 90 100 120
SMS = $(GPU_ARCHS)
ifeq ($(GENCODES),)
  $(foreach sm,$(SMS),$(eval GENCODES += -gencode arch=compute_$(sm),code=sm_$(sm)))
  GENCODES += -gencode arch=compute_90,code=compute_90
endif

GENCODE_FLAGS ?= $(GENCODES)

SRCS:= nvdspreprocess_lib.cpp \
       roi_conversion.cu

INCS:= $(wildcard *.h)
LIB:=libcustom2d_preprocess.so

CFLAGS += -fPIC -DHAVE_CONFIG_H -std=c++17 -Wall -Werror \
         -I/usr/local/cuda-$(CUDA_VER)/include \
         -I/opt/nvidia/deepstream/deepstream/sources/includes \
         -I/opt/nvidia/deepstream/deepstream/sources/gst-plugins/gst-nvdspreprocess/include

GST_INSTALL_DIR ?= /opt/nvidia/deepstream/deepstream/lib/gst-plugins/
LIB_INSTALL_DIR ?= /opt/nvidia/deepstream/deepstream/lib/

LIBS := -shared -Wl,-no-undefined \
        -L/usr/local/cuda-$(CUDA_VER)/lib64/ -lcudart -ldl \
        -L$(LIB_INSTALL_DIR) -lnvdsgst_helper -lnvdsgst_meta -lnvds_meta \
        -lnvbufsurface -lnvbufsurftransform -lcuda \
        -Wl,-rpath,$(LIB_INSTALL_DIR)

OBJS:= $(SRCS:.cpp=.o)
OBJS:= $(notdir $(OBJS:.cu=.o))

PKGS:= gstreamer-1.0

CFLAGS+=$(shell pkg-config --cflags $(PKGS))
LIBS+=$(shell pkg-config --libs $(PKGS))

all: $(LIB)

%.o: %.cpp $(INCS)
	$(CXX) -c -o $@ $(CFLAGS) $<

%.o: %.cu roi_conversion.hpp
	$(NVCC) -c -O3 -o $@ $(GENCODE_FLAGS) --compiler-options '-fPIC' $<

$(LIB): $(OBJS) $(DEP)
	$(CXX) -o $@ $(OBJS) $(LIBS)

install: $(LIB)
	cp -rv $(LIB) $(GST_INSTALL_DIR)

clean:
	rm -rf $(OBJS) $(LIB)
