# SPDX-FileCopyrightText: Copyright (c) 2020-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

CXX:= gcc

# Set LEGACY_API=1 to build with the legacy Rivermax API (rmax_* functions).
# By default, the new Rivermax API (rmx_* functions) is used.
# Example: export LEGACY_API=1 && make
ifdef LEGACY_API
  SRC_DIR := legacy_api/
else
  SRC_DIR :=
endif

SRCS:= gstnvdsudp.c $(SRC_DIR)gstnvdsudpsrc.c $(SRC_DIR)gstnvdsudpsink.c gstnvdsudpcommon.c
INCS:= gstnvdsudpcommon.h $(wildcard $(SRC_DIR)*.h)
LIB:=libnvdsgst_udp.so
include ../Rules.mk


NVDS_VERSION?=9.0

CFLAGS+= -fPIC -DDS_VERSION=\"9.0.0\" \
	 -I /usr/local/cuda-$(CUDA_VER)/include \
	 -I ../../../includes \
	 -I /usr/include/mellanox/

GST_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/gst-plugins/
LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/

LIBS := -shared -Wl,-no-undefined \
	-L/usr/local/cuda-$(CUDA_VER)/lib64/ -lcudart

LIBS+= -L/usr/lib/aarch64-linux-gnu/ -L/usr/lib/x86_64-linux-gnu/ -lrivermax

LIBS+= -L/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib -Wl,-rpath-link,/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib -lnvbufsurface -lnvdsbufferpool -ldl -lm -lpthread -lcuda -Wl,-rpath,$(LIB_INSTALL_DIR)

OBJS:= $(SRCS:.c=.o)
OBJS := $(addprefix $(OBJ_DIR)/,$(OBJS))

PKGS:= gstreamer-1.0 gstreamer-base-1.0 gstreamer-sdp-1.0 gio-2.0
CFLAGS+=$(shell pkg-config --cflags $(PKGS))
LIBS+=$(shell pkg-config --libs $(PKGS))

all: $(LIB_DIR)/$(LIB)

$(OBJ_DIR)/%.o: %.c $(INCS) Makefile
	@mkdir -p $(OBJ_DIR)
	$(CXX) -c -o $@ $(CFLAGS) $<

$(LIB_DIR)/$(LIB): $(OBJS) $(DEP) Makefile
	@mkdir -p $(LIB_DIR)
	$(CXX) -o $(OBJ_DIR)/$(LIB) $(OBJS) $(LIBS)
	cp $(OBJ_DIR)/$(LIB) $@

install: $(LIB_DIR)/$(LIB)
	ldconfig

clean:
	rm -rf $(OBJ_DIR) $(LIB_DIR)/$(LIB)