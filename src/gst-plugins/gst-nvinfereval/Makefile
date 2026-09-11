# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

CUDA_VER?=
ifeq ($(CUDA_VER),)
  $(error "CUDA_VER is not set")
endif

CXX:=g++
NVDS_VERSION:=9.1
GST_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/gst-plugins/
LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/
PKGS:=gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0 json-glib-1.0
SRCS:=gstnvinfereval.c gstnvinferevalinit.c nvds_eval_engine.cpp nvds_eval_config.cpp \
      nvds_eval_gt_loader.cpp nvds_eval_ap.cpp nvds_eval_accumulator.cpp nvds_eval_report.cpp
OBJS:=$(SRCS:.c=.o)
OBJS:=$(OBJS:.cpp=.o)
LIB:=libnvdsgst_infereval.so
CFLAGS:=-fPIC -std=c++11 -fno-exceptions -Wall -Werror -Wno-deprecated-declarations -Wsign-compare \
        -DDS_VERSION=\"9.1.1\" -I. -I../../../includes -I../../gst-utils/gstnvdscustomhelper \
        -I../../gst-utils/gstnvdshelper -I../gst-nvmodelmux -I/usr/local/cuda-$(CUDA_VER)/include
CFLAGS+=$(shell pkg-config --cflags $(PKGS))
LIBS:=-shared -Wl,-no-undefined -L/usr/local/cuda-$(CUDA_VER)/lib64/ -lcudart \
      -L$(LIB_INSTALL_DIR) -lnvdsgst_customhelper -lnvdsgst_helper -lnvdsgst_meta -lnvds_meta \
      -lnvbufsurface -ldl -lm -Wl,-rpath,$(LIB_INSTALL_DIR)
LIBS+=$(shell pkg-config --libs $(PKGS))
all: $(LIB)
%.o: %.c Makefile
	$(CXX) -c -o $@ $(CFLAGS) $<
%.o: %.cpp Makefile
	$(CXX) -c -o $@ $(CFLAGS) $<
$(LIB): $(OBJS) Makefile
	$(CXX) -o $@ $(OBJS) $(LIBS)
install: $(LIB)
	cp -rv $(LIB) $(GST_INSTALL_DIR)
clean:
	rm -rf $(OBJS) $(LIB)
