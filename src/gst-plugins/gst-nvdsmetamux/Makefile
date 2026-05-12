# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

CUDA_VER?=
ifeq ($(CUDA_VER),)
  $(error "CUDA_VER is not set")
endif


CXX:= g++

SRCS:= gstnvdsmetamux.cpp \
       nvdsmetamux_property_parser.cpp

INCS:= $(wildcard *.h)
LIB:=libnvdsgst_metamux.so
include ../Rules.mk

NVDS_VERSION?=9.0

CFLAGS+= -fPIC -DHAVE_CONFIG_H -std=c++17 -Wall -Werror -DDS_VERSION=\"9.0.0\" \
	 -I include \
	 -I ../../../includes

GST_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/gst-plugins/
LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/

LIBS := -shared -Wl,-no-undefined \
	-L/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib -L/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib -L/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib -Wl,-rpath-link,/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib -lnvdsgst_helper -lnvdsgst_meta -lnvds_meta \
	-Wl,-rpath,$(LIB_INSTALL_DIR)

OBJS:= $(SRCS:.cpp=.o)
OBJS := $(addprefix $(OBJ_DIR)/,$(OBJS))

PKGS:= gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0

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
