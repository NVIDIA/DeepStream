################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
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

