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
SRCS:= gstdsnvmultiurisrcbin.cpp \
        gstdsnvmultiurisrcbininit.cpp

NVDS_UTILS_SRC:= ../../utils/nvds_msgapi/common_src/nvds_utils.cpp

INCS:= $(wildcard *.h)

LIB:=libnvdsgst_nvmultiurisrcbin.so
LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream/lib/
GST_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream/lib/gst-plugins/

CFLAGS+= -fPIC -DDS_VERSION=\"9.1.0\" \
     -I../../../includes \
     -I../../apps/common/includes/ \
     -I../../utils/nvds_rest_server \
 	 -I/usr/local/cuda-$(CUDA_VER)/include \
	 -I/usr/include/jsoncpp

LIBS := -shared -Wl,-no-undefined \
 	-L/usr/local/cuda-$(CUDA_VER)/lib64/ -lcudart
 
LIBS+= -L$(LIB_INSTALL_DIR) -L/opt/nvidia/deepstream/deepstream/lib/ \
        -lnvdsgst_helper -lnvdsgst_customhelper -lnvds_rest_server -ldl\
        -lnvds_stats -Wl,-rpath,$(LIB_INSTALL_DIR) -lnvds_logger -lcrypto

ifeq ($(TARGET_DEVICE),aarch64)
   LIBS+= -L/usr/lib/aarch64-linux-gnu/tegra/ -lgstnvcustomhelper -ljsoncpp -lcurl
else
   LIBS+= -L$(LIB_INSTALL_DIR) -lgstnvcustomhelper -ljsoncpp -lcurl
endif

OBJS:= $(SRCS:.cpp=.o) nvds_utils.o

PKGS:= gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0 gstreamer-audio-1.0
CFLAGS+= $(shell pkg-config --cflags $(PKGS))
LIBS+= $(shell pkg-config --libs $(PKGS))

all: $(LIB)

%.o: %.cpp $(INCS) Makefile
	@echo $(CFLAGS)
	$(CXX) -c -o $@ $(CFLAGS) $<

nvds_utils.o: $(NVDS_UTILS_SRC) Makefile
	$(CXX) -c -o $@ $(CFLAGS) $<

$(LIB): $(OBJS) $(DEP) Makefile
	$(CXX) -o $@ $(OBJS) $(LIBS)

install: $(LIB)
	cp -rv $(LIB) $(GST_INSTALL_DIR)

clean:
	rm -rf $(OBJS) nvds_utils.o $(LIB)

