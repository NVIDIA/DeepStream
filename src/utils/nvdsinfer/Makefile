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
ENGINE_BUILDER:=nvdsinfer-engine-builder

NVDS_VERSION:=9.2

LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/
BIN_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/bin/

CFLAGS+= -fPIC -Wno-deprecated-declarations -std=c++14 \
	 -I /usr/local/cuda-$(CUDA_VER)/include \
	 -I ../../../includes -I ../nvdsinferutils/include -DNDEBUG

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

LIBS+= -L$(LIB_INSTALL_DIR) -lnvdsgst_helper -lnvdsgst_meta -lnvds_meta \
       -lnvds_inferlogger -lnvds_inferutils -ldl \
       -Wl,-rpath,$(LIB_INSTALL_DIR)


OBJS:= $(SRCS:.cpp=.o)
OBJS:= $(OBJS:.cu=.o)

all: $(LIB) $(ENGINE_BUILDER)

%.o: %.cpp $(INCS) Makefile
	$(CXX) -c -o $@ $(CFLAGS) $<

%.o: %.cu $(INCS) Makefile
	@echo $(CFLAGS)
	$(NVCC) -c -o $@ --compiler-options '-fPIC' $<

$(LIB): $(OBJS) $(DEP) Makefile
	@echo $(CFLAGS)
	$(CXX) -o $@ $(OBJS) $(LIBS)

$(ENGINE_BUILDER): nvdsinfer_engine_builder.cpp $(INCS) $(LIB) Makefile
	$(CXX) $(CFLAGS) -o $@ $< -L. -lnvds_infer $(filter-out -shared,$(LIBS)) \
		-Wl,-rpath,'$$ORIGIN:$$ORIGIN/../lib'

install: $(LIB) $(ENGINE_BUILDER)
	cp -rv $(LIB) $(LIB_INSTALL_DIR)
	mkdir -p $(BIN_INSTALL_DIR)
	cp -v $(ENGINE_BUILDER) $(BIN_INSTALL_DIR)

clean:
	rm -rf $(OBJS) $(LIB) $(ENGINE_BUILDER)
