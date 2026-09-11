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
ifeq ($(CUDA_VER),)
  $(error "CUDA_VER is not set")
endif

TARGET_DEVICE = $(shell g++ -dumpmachine | cut -f1 -d -)

CXX:= g++
CC:= gcc

APP_COMMON_DIR:= ../../../apps/common

SRCS:= src/element.cpp src/pipeline.cpp src/ds_yaml_parser.cpp src/object.cpp \
       src/custom_object.cpp src/common_factory.cpp src/custom_factory.cpp \
       src/buffer_probe.cpp src/signal_handler.cpp src/signal_emitter.cpp \
       src/metadata.cpp src/event_meta.cpp src/buffer.cpp src/pad.cpp \
       src/source_config.cpp src/mediainfo.cpp src/msgbroker_c2d_receiver.cpp \
       src/model_engine_watch_otf_trigger.cpp src/tiler_event_handler.cpp \
       src/data_feeder.cpp src/data_receiver.cpp src/perf_monitor.cpp \
       src/tensor.cpp src/tensor_meta.cpp src/source_manager.cpp

CSRCS:= src/gst/buffer_probe.c src/gst/signal_handler.c src/gst/signal_emitter.c \
        $(APP_COMMON_DIR)/src/deepstream_perf.c

INCS:= $(wildcard ../../includes/*.hpp) $(wildcard src/gst/*.h)
LIB:=libnvds_service_maker.so
LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/
NVDS_VERSION:=9.1

CFLAGS+= -fPIC -DHAVE_CONFIG_H -DHAVE_X11 -DDS_VERSION=\"9.1.1\" \
	 -Wall -Wno-deprecated-declarations \
	 -I../../includes \
	 -I../../includes/lib \
	 -Isrc \
	 -I$(APP_COMMON_DIR)/includes \
	 -I../../../../includes \
	 -I../../../gst-plugins/gst-nvdspreprocess/include \
	 -I../../../gst-utils/gst-nvdssr \
	 -I../../../gst-utils/gstnvdscustomhelper \
	 -I../../../utils/nvmsgbroker \
	 -I/usr/local/cuda-$(CUDA_VER)/include

CXXFLAGS+= -std=c++17

LIBS := -shared -Wl,-no-undefined \
	-L/usr/local/cuda-$(CUDA_VER)/lib64/ -lcudart

LIBS+= -L$(LIB_INSTALL_DIR) -lnvds_meta -lnvdsgst_meta -lnvdsgst_helper \
	-lnvdsgst_customhelper -lnvds_msgbroker \
	-lnvbufsurftransform -lnvbufsurface \
	-Wl,-rpath,$(LIB_INSTALL_DIR) \
	-lyaml-cpp -ldl -lpthread -lX11

# buffer.cpp uses the CUDA driver API (cuGraphicsEGLRegisterImage and friends)
# only under __aarch64__, so libcuda is needed on Tegra but not on x86.
ifeq ($(TARGET_DEVICE),aarch64)
  LIBS+= -L/usr/lib/aarch64-linux-gnu/nvidia -lcuda
endif

OBJS:= $(SRCS:.cpp=.o) $(CSRCS:.c=.o)

PKGS:= gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0 gstreamer-app-1.0 \
       gstreamer-pbutils-1.0 gstreamer-rtsp-server-1.0 glib-2.0 gobject-2.0 \
       json-glib-1.0
CFLAGS+= $(shell pkg-config --cflags $(PKGS))
LIBS+= $(shell pkg-config --libs $(PKGS))

all: $(LIB)

%.o: %.cpp $(INCS) Makefile
	$(CXX) -c -o $@ $(CFLAGS) $(CXXFLAGS) $<

%.o: %.c $(INCS) Makefile
	$(CC) -c -o $@ $(CFLAGS) $<

$(LIB): $(OBJS) $(DEP) Makefile
	$(CXX) -o $@ $(OBJS) $(LIBS)

install: $(LIB)
	cp -rv $(LIB) $(LIB_INSTALL_DIR)

clean:
	rm -rf $(OBJS) $(LIB)
