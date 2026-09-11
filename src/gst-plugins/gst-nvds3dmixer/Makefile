# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

CXX:= g++

NVDS_VERSION:=9.1

GST_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/gst-plugins/
LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/

CFLAGS:= -fPIC -DHAVE_CONFIG_H -std=c++17 -Wall -DDS_VERSION=\"9.1.1\"

CFLAGS+= -I../../../includes \
	 -I../../utils

PKGS:= gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0
CFLAGS+= $(shell pkg-config --cflags $(PKGS))

SRCS:= gstnvds3dmixer.cpp
LIB:= libnvdsgst_3dmixer.so

LIBS:= -shared -Wl,-no-undefined \
	-L$(LIB_INSTALL_DIR) -lnvdsgst_3d_gst -lnvdsgst_helper \
	-lnvds_3d_common -lnvdsgst_meta -lnvds_meta \
	-lyaml-cpp -ldl -lpthread \
	-Wl,-rpath,$(LIB_INSTALL_DIR)
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
