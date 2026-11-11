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

CC:= g++

PKGS:= glib-2.0 gobject-2.0 json-glib-1.0 uuid gstreamer-1.0

NVDS_VERSION:=9.2

LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/

# Keep generated message code, headers and runtime on one protobuf provider.
TRITON_GRPC_CLIENT ?= /opt/tritonclient
PROTOBUF_BIN_DIR ?= /opt/proto/bin
NVDS_PROTOBUF_LEGACY_CXX_STANDARD := 11
include ../../../../includes/nvds_protobuf.mk
PROTOBUF_CXX_STANDARD ?= $(NVDS_PROTOBUF_MIN_CXX_STANDARD)
ifneq ($(MAKECMDGOALS),clean)
ifeq ($(filter $(PROTOBUF_CXX_STANDARD),11 14 17 20 23),)
$(error PROTOBUF_CXX_STANDARD must be 11, 14, 17, 20 or 23)
endif
ifneq ($(shell test "$(PROTOBUF_CXX_STANDARD)" -ge "$(NVDS_PROTOBUF_MIN_CXX_STANDARD)" && printf yes),yes)
$(error Selected protobuf provider requires C++$(NVDS_PROTOBUF_MIN_CXX_STANDARD) or newer)
endif
endif
MESSAGE_PROTOBUF_LIB_DIR ?= $(NVDS_PROTOBUF_LIB_DIR)
NVDS_PROTOBUF_SELECTED_CXX_STANDARD = $(PROTOBUF_CXX_STANDARD)
NVDS_PROTOBUF_SELECTED_LIB_DIR = $(MESSAGE_PROTOBUF_LIB_DIR)
MESSAGE_PROTOBUF_SUPPORT_LIBS := $(call nvds_protobuf_support_libraries,$(MESSAGE_PROTOBUF_LIB_DIR))
MESSAGE_PROTOBUF_LIBS := -L"$(MESSAGE_PROTOBUF_LIB_DIR)" -Wl,--start-group -l:libprotobuf.a $(MESSAGE_PROTOBUF_SUPPORT_LIBS) -Wl,--end-group -pthread

CFLAGS:= -Wall -std=c++$(PROTOBUF_CXX_STANDARD) -shared -fPIC

CFLAGS+= -fPIC -DDS_VERSION=\"9.2.1\" \
	 -I /usr/local/cuda-$(CUDA_VER)/include \
	 -I ../../../../includes

CFLAGS+= -I"$(TRITON_GRPC_CLIENT)/include"

CFLAGS+= $(shell pkg-config --cflags $(PKGS))
LIBS:= $(shell pkg-config --libs $(PKGS))

LIBS+= -shared -Wl,-no-undefined \
	-L$(LIB_INSTALL_DIR) -lnvdsgst_helper -lnvdsgst_meta -lnvds_meta \
	-Wl,-rpath,$(LIB_INSTALL_DIR)

LIBS+= $(MESSAGE_PROTOBUF_LIBS)

PROTOBUF_BUILD_DIR:= $(CURDIR)

SRCFILES:= video_metadata_serialization.cpp  \
  $(PROTOBUF_BUILD_DIR)/ds_meta.pb.cc

TARGET_LIB:= libnvds_video_metadata_serialization.so

SCHEMA_PROTOS:=$(wildcard $(CURDIR)/*.proto)
SCHEMA_GENERATED:=$(SCHEMA_PROTOS:.proto=.pb.cc) $(SCHEMA_PROTOS:.proto=.pb.h)

all: $(TARGET_LIB)

$(TARGET_LIB) : $(SRCFILES) $(SCHEMA_GENERATED) $(NVDS_PROTOBUF_PROVIDER_STAMP)
	$(CC) -o $@ $(SRCFILES) $(CFLAGS) $(LIBS)

install: $(TARGET_LIB)
	cp -rv $(TARGET_LIB) $(LIB_INSTALL_DIR)

clean:
	rm -rf $(TARGET_LIB) $(PROTOBUF_BUILD_DIR)/*.pb.*
	rm -f "$(NVDS_PROTOBUF_PROVIDER_STAMP)"

$(SCHEMA_GENERATED) &: $(SCHEMA_PROTOS) Makefile $(NVDS_PROTOBUF_MAKEFILE) $(NVDS_PROTOBUF_PROVIDER_STAMP)
	"$(PROTOBUF_BIN_DIR)/protoc" -I$(CURDIR) --cpp_out=$(PROTOBUF_BUILD_DIR)/ $(SCHEMA_PROTOS)
