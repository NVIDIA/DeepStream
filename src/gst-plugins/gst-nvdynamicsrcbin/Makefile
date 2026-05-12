# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


CC:= gcc
SRCS:= gstnvdsdynamicsrcbin.c
 

LIB:=libnvdsgst_dynamicsrcbin.so
include ../Rules.mk
LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/
GST_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/gst-plugins/

CFLAGS+= -fPIC -DDS_VERSION=\"9.0.0\" \
     -I../../../includes \
     -I../../apps/common/includes/ \
     -I/usr/local/cuda-$(CUDA_VER)/include \
     -DHAVE_CONFIG_H -DNEW_METADATA

LIBS := -shared -Wl,-no-undefined \
     -L/usr/local/cuda-$(CUDA_VER)/lib64/ -lcudart

LIBS+= -L/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib -Wl,-rpath-link,/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib \
       -lnvdsgst_helper -lnvdsgst_meta -lnvds_meta -ldl \
       -Wl,-rpath,$(LIB_INSTALL_DIR)


OBJS:= $(SRCS:.c=.o)
OBJS := $(addprefix $(OBJ_DIR)/,$(OBJS))

PKGS:= gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0 gstreamer-audio-1.0
CFLAGS+= $(shell pkg-config --cflags $(PKGS))
LIBS+= $(shell pkg-config --libs $(PKGS))

all: $(LIB_DIR)/$(LIB)

$(OBJ_DIR)/%.o: %.c $(INCS) Makefile
	@mkdir -p $(OBJ_DIR)
	@echo $(CFLAGS)
	$(CC) -c -o $@ $(CFLAGS) $<

$(LIB_DIR)/$(LIB): $(OBJS) $(DEP) Makefile
	@mkdir -p $(LIB_DIR)
	$(CC) -o $@ $(OBJS) $(LIBS)

install: $(LIB_DIR)/$(LIB)
	ldconfig

clean:
	rm -rf $(OBJ_DIR) $(LIB_DIR)/$(LIB)