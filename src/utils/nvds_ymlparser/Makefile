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

CC:= g++

PKGS:= glib-2.0 gobject-2.0 gstreamer-1.0

NVDS_VERSION:=9.1

LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/

CFLAGS:= -fPIC -DHAVE_CONFIG_H -std=c++17 -Wall

CFLAGS+= -I../../../includes \
	 -I../nvds_rest_server \
	 -I../../gst-utils/gstnvdscustomhelper \
	 -I/usr/include/jsoncpp \
	 -I/usr/local/cuda-$(CUDA_VER)/include
CFLAGS+= $(shell pkg-config --cflags $(PKGS))

SRCFILES:= nvds_common_parser.cpp \
	nvds_streammux_parse.cpp \
	nvds_msgconv_parse.cpp \
	nvds_tiler_parse.cpp \
	nvds_osd_parse.cpp \
	nvds_tracker_parse.cpp \
	nvds_sink_parse.cpp \
	nvds_source_list_parse.cpp \
	nvds_gie_parse.cpp \
	nvds_source_parse.cpp \
	nvds_msgbroker_parse.cpp \
	nvds_encoder_parse.cpp \
	nvds_multiurisrcbin_parse.cpp \
	nvds_server_appctx_parse.cpp \
	nvds_nvxfer_parse.cpp \
	nvds_nvdsanalytics_parse.cpp

TARGET_LIB:= libnvds_yml_parser.so

LIBS:= -shared \
	-L$(LIB_INSTALL_DIR) -lnvbufsurface -lnvbufsurftransform \
	-lnvds_rest_server -lnvdsgst_customhelper -lnvds_meta \
	-Wl,-rpath,$(LIB_INSTALL_DIR) \
	-L/usr/local/cuda-$(CUDA_VER)/lib64/ -lcudart \
	-ldl -lpthread -lyaml-cpp -ljsoncpp
LIBS+= $(shell pkg-config --libs $(PKGS))

all: $(TARGET_LIB)

$(TARGET_LIB) : $(SRCFILES)
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)

install: $(TARGET_LIB)
	cp -rv $(TARGET_LIB) $(LIB_INSTALL_DIR)

clean:
	rm -rf $(TARGET_LIB)
