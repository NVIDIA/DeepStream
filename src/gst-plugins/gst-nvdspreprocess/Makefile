# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# compile with opencv to dump ROIs
WITH_OPENCV:=0

CUDA_VER?=
ifeq ($(CUDA_VER),)
  $(error "CUDA_VER is not set")
endif

WSL:=$(shell uname -a | grep -io "microsoft" | tr '[:upper:]' '[:lower:]')

CXX:= g++

SRCS:= gstnvdspreprocess.cpp gstnvdspreprocess_allocator.cpp \
       nvdspreprocess_property_parser.cpp nvdspreprocess_yaml_parser.cpp

INCS:= $(wildcard *.h)
LIB:=libnvdsgst_preprocess.so
include ../Rules.mk

NVDS_VERSION?=9.0

CFLAGS+= -fPIC -DHAVE_CONFIG_H -std=c++17 -Wall -Werror -DDS_VERSION=\"9.0.0\" \
	 -I /usr/local/cuda-$(CUDA_VER)/include \
	 -I include \
	 -I ../../../includes

GST_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/gst-plugins/
LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/

LIBS := -shared -Wl,-no-undefined \
	-L/usr/local/cuda-$(CUDA_VER)/lib64/ -lcudart -ldl \
	-L/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib -L/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib -L/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib -Wl,-rpath-link,/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib -lnvdsgst_helper -lnvdsgst_customhelper -lnvdsgst_meta -lnvds_meta -lnvbufsurface -lnvbufsurftransform\
	-lcuda -lyaml-cpp -Wl,-rpath,$(LIB_INSTALL_DIR)

ifeq ("microsoft", "$(WSL)")
  LIBS+= -L/usr/lib/wsl/lib/
endif

OBJS:= $(SRCS:.cpp=.o)
OBJS := $(addprefix $(OBJ_DIR)/,$(OBJS))

PKGS:= gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0

ifeq ($(WITH_OPENCV),1)
  CFLAGS+= -DWITH_OPENCV \
    -I /usr/include/opencv4
  LIBS+= -lopencv_imgproc -lopencv_core -lopencv_imgcodecs
  PKGS+= opencv4
endif

CFLAGS+=$(shell pkg-config --cflags $(PKGS))
LIBS+=$(shell pkg-config --libs $(PKGS))

all: $(LIB_DIR)/$(LIB)

$(OBJ_DIR)/%.o: %.cpp $(INCS) Makefile
	@mkdir -p $(OBJ_DIR)
	@echo $(CFLAGS)
	$(CXX) -c -o $@ $(CFLAGS) $<

$(LIB_DIR)/$(LIB): $(OBJS) $(DEP) Makefile
	@mkdir -p $(LIB_DIR)
	@echo $(CFLAGS)
	$(CXX) -o $(OBJ_DIR)/$(LIB) $(OBJS) $(LIBS)
	cp $(OBJ_DIR)/$(LIB) $@

install: $(LIB_DIR)/$(LIB)
	ldconfig

clean:
	rm -rf $(OBJ_DIR) $(LIB_DIR)/$(LIB)
