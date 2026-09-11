# SPDX-FileCopyrightText: Copyright (c) 2020-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

CXX:= g++

NVDS_VERSION:=9.1

GST_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/gst-plugins/
LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/

CFLAGS:= -fPIC -std=c++11 -Wall -Wno-deprecated-declarations -DDS_VERSION=\"9.1.1\"

CFLAGS+= -I../../../includes \
	 -I../../utils/nvmsgbroker \
	 -I../gst-nvtiler \
	 -I../../gst-utils/gstnvdscustomhelper \
	 -I../gst-nvmsgbroker \
	 -I../gst-nvmsgconv \
	 -I../gst-nvdsanalytics \
	 -I../../utils/nvds_analytics \
	 -I../../utils/nvmsgconv \
	 -I../../utils/nvds_rest_server \
	 -I../../utils/nvds_ymlparser \
	 -I/usr/local/cuda-$(CUDA_VER)/include

PKGS:= gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0 gstreamer-rtsp-server-1.0
CFLAGS+= $(shell pkg-config --cflags $(PKGS))

SRCS:= gstdeepstreambins.cpp \
	gstdsnvinferbin.cpp \
	gstdsnvinferserverbin.cpp \
	gstdsnvosdbin.cpp \
	gstdsnvdewarperbin.cpp \
	gstdsnvtilerbin.cpp \
	gstdsnvtrackerbin.cpp \
	gstdsnvanalyticsbin.cpp \
	gstdsnvmsgbrokerbin.cpp \
	gstdsnvrtspoutbin.cpp \
	gstdsnvvideoencfilesinkbin.cpp \
	gstdsnvvideorendererbin.cpp \
	gstdsnvcamerasrcbin.cpp \
	gstdsnvbuffersyncbin.cpp

LIB:= libnvdsgst_deepstream_bins.so

LIBS:= -shared -Wl,-no-undefined -lstdc++ \
	-L/usr/local/cuda-$(CUDA_VER)/lib64/ -lcudart \
	-L$(GST_INSTALL_DIR) -L$(LIB_INSTALL_DIR) \
	-lnvdsgst_helper -lnvdsgst_customhelper -lnvdsgst_smartrecord \
	-lnvdsgst_msgbroker -lnvdsgst_msgconv -lnvdsgst_osd \
	-lnvdsgst_dewarper -lnvdsgst_infer -lnvdsgst_tracker \
	-lnvdsgst_dsanalytics -lnvdsgst_multistreamtiler -lgstnvvideoconvert \
	-lnvdsgst_meta -lnvds_rest_server -lnvds_meta \
	-lnvbufsurface -lnvbufsurftransform \
	-Wl,-rpath,$(LIB_INSTALL_DIR) -ldl
LIBS+= $(shell pkg-config --libs $(PKGS))

OBJS:= $(SRCS:.cpp=.o)

all: $(LIB)

%.o: %.cpp Makefile
	$(CXX) -c -o $@ $(CFLAGS) $<

$(LIB): $(OBJS) Makefile
	$(CXX) -o $@ $(OBJS) $(LIBS)

install: $(LIB)
	cp -rv $(LIB) $(GST_INSTALL_DIR)

clean:
	rm -rf $(OBJS) $(LIB)
