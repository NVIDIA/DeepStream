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

NVDS_VERSION?=9.0

LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/

CUSTOM_LIB:=libnvds_custom_sequence_preprocess.so

CC = g++

CFLAGS:= -fvisibility=hidden -Wall -Werror


PKGS:= gstreamer-1.0

CFLAGS+= -I../../../../../includes \
         -I ../../../../gst-plugins/gst-nvdspreprocess/include \
	 -I /usr/local/cuda-$(CUDA_VER)/include \
         -fPIC -std=c++14

CFLAGS+= $(shell pkg-config --cflags $(PKGS))

LIBS:= $(shell pkg-config --libs $(PKGS))

LIBS+= -L/usr/local/cuda-$(CUDA_VER)/lib64/ -lcudart -lnvdsgst_helper -lm \
-L/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib -Wl,-rpath-link,/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib -lnvdsgst_meta -lnvds_meta \
		-lcuda -Wl,-rpath,$(LIB_INSTALL_DIR)


CUSTOM_LIB_SRCS:= $(wildcard ./*.cpp ./*.cu)
CUSTOM_LIB_INCS:= $(wildcard ./*.h)
CUSTOM_LIB_OBJS:= $(CUSTOM_LIB_SRCS:.cpp=.o)
CUSTOM_LIB_OBJS:= $(CUSTOM_LIB_OBJS:.cu=.o)

NVCC:=/usr/local/cuda-$(CUDA_VER)/bin/nvcc
LFLAGS:= -shared -Wl,--start-group $(LIBS) -Wl,--end-group


all: $(CUSTOM_LIB)

$(CUSTOM_LIB): $(CUSTOM_LIB_OBJS) Makefile
	$(CC) -o $(CUSTOM_LIB) $(CUSTOM_LIB_OBJS) $(LFLAGS)


%.o: %.cpp $(CUSTOM_LIB_INCS) Makefile
	$(CC) -c -o $@ $(CFLAGS) $<
%.o: %.cu $(CUSTOM_LIB_INCS) Makefile
	@echo $(CFLAGS)
	$(NVCC) -c -o $@ --compiler-options '-fPIC' $<

install: $(CUSTOM_LIB)
	ldconfig

clean:
	rm -rf $(CUSTOM_LIB_OBJS) $(CUSTOM_LIB)

