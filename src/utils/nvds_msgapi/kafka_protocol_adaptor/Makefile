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

CXX:=g++

PKGS:= glib-2.0

DS_LIB:=/opt/nvidia/deepstream/deepstream/lib/

PROTOBUF_BUILD_DIR:= .

SRCS:=  nvds_kafka_proto.cpp kafka_client.cpp json_helper.cpp ../common_src/nvds_utils.cpp $(PROTOBUF_BUILD_DIR)/schema.pb.cc
TARGET_LIB:= libnvds_kafka_proto.so

# Keep generated message code, headers and runtime on one protobuf provider.
TRITON_GRPC_CLIENT ?= /opt/tritonclient
PROTOBUF_BIN_DIR ?= /opt/proto/bin
NVDS_PROTOBUF_LEGACY_CXX_STANDARD := 17
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

CFLAGS:= -fPIC -Wall -std=c++$(PROTOBUF_CXX_STANDARD)

CFLAGS+= $(shell pkg-config --cflags $(PKGS))

LIBS:= $(shell pkg-config --libs $(PKGS))
LDFLAGS:= -shared

DS_INC:= ../../../../includes
RDKAFKA_INC:=/usr/include/librdkafka

INC_PATHS:= -I $(DS_INC) -I $(RDKAFKA_INC) -I"$(TRITON_GRPC_CLIENT)/include"
CFLAGS+= $(INC_PATHS)

LIBS+= -L/$(DS_LIB) $(MESSAGE_PROTOBUF_LIBS) -lrdkafka -ljansson -lnvds_logger -lpthread -lcrypto
LDFLAGS+= -Wl,-rpath=$(DS_LIB)

SCHEMA_PROTOS:=../../nvmsgconv/deepstream_schema/schema.proto
SCHEMA_GENERATED:=$(PROTOBUF_BUILD_DIR)/schema.pb.cc $(PROTOBUF_BUILD_DIR)/schema.pb.h

all: $(TARGET_LIB)

$(TARGET_LIB) : $(SRCS) $(SCHEMA_GENERATED) $(NVDS_PROTOBUF_PROVIDER_STAMP)
	$(CXX) -o $@ $(SRCS) $(CFLAGS) $(LIBS) $(LDFLAGS)

install: $(TARGET_LIB)
	cp -rv $(TARGET_LIB) $(DS_LIB)

clean:
	rm -rf $(TARGET_LIB)
	rm -f "$(NVDS_PROTOBUF_PROVIDER_STAMP)"

$(SCHEMA_GENERATED) &: $(SCHEMA_PROTOS) Makefile $(NVDS_PROTOBUF_MAKEFILE) $(NVDS_PROTOBUF_PROVIDER_STAMP)
	"$(PROTOBUF_BIN_DIR)/protoc" -I../../nvmsgconv/deepstream_schema --cpp_out=$(PROTOBUF_BUILD_DIR)/ $<
