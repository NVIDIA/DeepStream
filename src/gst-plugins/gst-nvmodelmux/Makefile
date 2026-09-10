# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

CUDA_VER?=
ifeq ($(CUDA_VER),)
  $(error "CUDA_VER is not set")
endif

CC:=gcc
NVDS_VERSION:=9.1
GST_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/gst-plugins/
LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/
PKGS:=gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0 json-glib-1.0
SRCS:=gstnvmodelmuxinit.c gstnvmodelmux.c gstnvmodelmux_bin.c gstnvmodelmux_config.c \
      gstnvmodelmux_control.c gstnvmodelmux_control_metadata.c gstnvmodelmux_debug.c \
      gstnvmodelmux_model_helpers.c gstnvmodelmux_model_status.c gstnvmodelmux_perf.c \
      gstnvmodelmux_routing_table.c
OBJS:=$(SRCS:.c=.o)
LIB:=libnvdsgst_modelmux.so
CFLAGS:=-fPIC -Wall -Werror -Wno-deprecated-declarations -Wmissing-prototypes -Wsign-compare \
        -DDS_VERSION=\"9.1.1\" -I. -I../../gst-utils/gstnvdscustomhelper \
        -I../../gst-utils/gstnvdshelper -I../../utils/nvds_rest_server \
        -I../../apps/common/includes -I../../../includes -I/usr/local/cuda-$(CUDA_VER)/include
CFLAGS+=$(shell pkg-config --cflags $(PKGS))
LIBS:=-shared -Wl,-no-undefined -L/usr/local/cuda-$(CUDA_VER)/lib64/ -lcudart \
      -L../../gst-utils/gstnvdscustomhelper -L$(LIB_INSTALL_DIR) -lnvdsgst_customhelper -lnvdsgst_helper -lnvdsgst_meta -lnvds_meta \
      -lnvds_rest_server -lnvbufsurface -ldl -lm -Wl,-rpath,$(LIB_INSTALL_DIR)
LIBS+=$(shell pkg-config --libs $(PKGS))
all: $(LIB)
%.o: %.c Makefile
	$(CC) -c -o $@ $(CFLAGS) $<
$(LIB): $(OBJS) Makefile
	$(CC) -o $@ $(OBJS) $(LIBS)
install: $(LIB)
	cp -rv $(LIB) $(GST_INSTALL_DIR)
clean:
	rm -rf $(OBJS) $(LIB)
