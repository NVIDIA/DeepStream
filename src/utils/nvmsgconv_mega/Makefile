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

PKGS:= glib-2.0 gobject-2.0 json-glib-1.0 uuid

NVDS_VERSION:=9.1

LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/

CFLAGS:= -Wall -std=c++14 -shared -fPIC

CFLAGS+= -I/opt/tritonclient/include -I../../../includes -I./deepstream_schema -I/usr/local/cuda/include

CFLAGS+= $(shell pkg-config --cflags $(PKGS))
LIBS:= $(shell pkg-config --libs $(PKGS))

LIBS+= -L/opt/tritonclient/lib -lyaml-cpp -lprotobuf \
  -L/usr/local/cuda/lib64 -lcudart

PROTOBUF_BUILD_DIR:= ./deepstream_schema
PROTOBUF_BIN_DIR ?= /opt/proto/bin

SRCFILES:= nvmsgconv_mega.cpp  \
  deepstream_schema/eventmsg_payload.cpp \
  deepstream_schema/dsmeta_payload.cpp \
  deepstream_schema/deepstream_schema.cpp \
  $(PROTOBUF_BUILD_DIR)/schema.pb.cc \
  $(PROTOBUF_BUILD_DIR)/lidar_schema.pb.cc

TARGET_LIB:= libnvds_msgconv_mega.so

all: $(TARGET_LIB)

$(TARGET_LIB) : $(SRCFILES)
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)

install: $(TARGET_LIB)
	cp -rv $(TARGET_LIB) $(LIB_INSTALL_DIR)

clean:
	rm -rf $(TARGET_LIB) $(PROTOBUF_BUILD_DIR)/*.pb.*

SCHEMA_PROTOS:=$(wildcard deepstream_schema/*.proto)
$(PROTOBUF_BUILD_DIR)/%.pb.cc: $(SCHEMA_PROTOS) Makefile
	$(PROTOBUF_BIN_DIR)/protoc -I./deepstream_schema --cpp_out=$(PROTOBUF_BUILD_DIR)/ $(SCHEMA_PROTOS)
