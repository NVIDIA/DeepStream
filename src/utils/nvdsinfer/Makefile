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

WITH_OPENCV?=0

NVCC:=/usr/local/cuda-$(CUDA_VER)/bin/nvcc
CXX:= g++
SRCS:= nvdsinfer_context_impl.cpp  nvdsinfer_context_impl_capi.cpp \
       nvdsinfer_context_impl_output_parsing.cpp \
       nvdsinfer_func_utils.cpp nvdsinfer_model_builder.cpp \
       nvdsinfer_backend.cpp nvdsinfer_conversion.cu \
       nvdsinfer_dynamic_properties.cpp

INCS:= $(wildcard *.h)
LIB:=libnvds_infer.so
LIB_DIR = /opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib
OBJ_DIR := .build
include ../../Rules.mk

NVDS_VERSION?=9.0

LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/

CFLAGS+= -fPIC -Wno-deprecated-declarations -std=c++14 \
	 -I /usr/local/cuda-$(CUDA_VER)/include \
	 -I ../../../includes -DNDEBUG

ifeq ($(WITH_OPENCV),1)
CFLAGS+= -DWITH_OPENCV \
         -I/usr/include/opencv4
endif

LIBS := -shared -Wl,-no-undefined \
	 -lnvinfer -lnvinfer_plugin -lnvonnxparser -lpthread \
	-L/usr/local/cuda-$(CUDA_VER)/lib64/ -lcudart

ifeq ($(WITH_OPENCV),1)
LIBS += -lopencv_objdetect -lopencv_imgproc -lopencv_core
endif

LIBS+= -L/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib -L/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib -L/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib -Wl,-rpath-link,/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib -lnvdsgst_helper -lnvdsgst_meta -lnvds_meta \
       -lnvds_inferlogger -lnvds_inferutils -ldl \
       -Wl,-rpath,$(LIB_INSTALL_DIR)


OBJS:= $(SRCS:.cpp=.o)
OBJS:= $(OBJS:.cu=.o)
OBJS := $(addprefix $(OBJ_DIR)/,$(OBJS))

all: $(LIB_DIR)/$(LIB)

$(OBJ_DIR)/%.o: %.cpp $(INCS) Makefile
	@mkdir -p $(OBJ_DIR)
	$(CXX) -c -o $@ $(CFLAGS) $<

$(OBJ_DIR)/%.o: %.cu $(INCS) Makefile
	@mkdir -p $(OBJ_DIR)
	@echo $(CFLAGS)
	$(NVCC) -c -o $@ --compiler-options '-fPIC' $<

$(LIB_DIR)/$(LIB): $(OBJS) $(DEP) Makefile
	@mkdir -p $(LIB_DIR)
	@echo $(CFLAGS)
	$(CXX) -o $@ $(OBJS) $(LIBS)

install: $(LIB_DIR)/$(LIB)
	ldconfig

clean:
	rm -rf $(OBJ_DIR) $(LIB_DIR)/$(LIB)
