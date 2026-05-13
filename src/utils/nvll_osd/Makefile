#######################################################################################################
#
# SPDX-FileCopyrightText: Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
######################################################################################################

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
NVDS_VERSION?=9.0

LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/
UTILS_LIB_DIR?= /opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib

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

LIBS+= -L/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib -L/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib -L/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib -Wl,-rpath-link,/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib -lpango-1.0 -lcairo -lpangocairo-1.0 -lgobject-2.0 \
       -lnvds_utils -lnvds_batch_jpegenc \
       -Wl,-rpath,$(LIB_INSTALL_DIR) \
       -lnvbufsurface -lnvbufsurftransform -lcuda -ljpeg

LIBS+= $(shell pkg-config --libs $(PKGS))

OBJ_DIR := .build
_OBJS := $(SRCS:.cpp=.o)
_OBJS := $(_OBJS:.cu=.o)
OBJS  := $(addprefix $(OBJ_DIR)/,$(_OBJS))

all: $(UTILS_LIB_DIR)/$(LIB)

$(OBJ_DIR)/%.o: %.cpp $(INCS) Makefile
	@mkdir -p $(dir $@)
	$(CXX) -c -o $@ $(CFLAGS) $<

$(OBJ_DIR)/%.o: %.cu $(INCS) Makefile
	@mkdir -p $(dir $@)
	$(NVCC) -c -o $@ --compiler-options '-fPIC' $<

$(UTILS_LIB_DIR)/$(LIB): $(OBJS) $(DEP) Makefile
	@mkdir -p $(UTILS_LIB_DIR)
	$(CXX) -o $@ $(OBJS) $(LIBS)

install: $(UTILS_LIB_DIR)/$(LIB)
	ldconfig

clean:
	rm -rf $(OBJ_DIR) $(UTILS_LIB_DIR)/$(LIB)
