#######################################################################################################
#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
#######################################################################################################

CC:= g++

PKGS:= glib-2.0 gobject-2.0 json-glib-1.0 uuid

NVDS_VERSION?=9.0

LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/

CFLAGS:= -Wall -std=c++14 -shared -fPIC

CFLAGS+= -I/opt/tritonclient/include -I../../../includes -I./deepstream_schema -I/usr/local/cuda/include

CFLAGS+= $(shell pkg-config --cflags $(PKGS))
LIBS:= $(shell pkg-config --libs $(PKGS))

LIBS+= -L/opt/tritonclient/lib -lyaml-cpp -lprotobuf \
  -L/usr/local/cuda/lib64 -lcudart

PROTOBUF_BUILD_DIR:= ./deepstream_schema
PROTOBUF_BIN_DIR ?= /opt/proto/bin

SRCFILES:= nvmsgconv.cpp  \
  deepstream_schema/eventmsg_payload.cpp \
  deepstream_schema/dsmeta_payload.cpp \
  deepstream_schema/deepstream_schema.cpp \
  $(PROTOBUF_BUILD_DIR)/schema.pb.cc \
  $(PROTOBUF_BUILD_DIR)/lidar_schema.pb.cc

TARGET_LIB:= libnvds_msgconv_mega2d.so

UTILS_LIB_DIR?= /opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib

all: $(UTILS_LIB_DIR)/$(TARGET_LIB)

$(UTILS_LIB_DIR)/$(TARGET_LIB): $(SRCFILES)
	@mkdir -p $(UTILS_LIB_DIR)
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)

install: $(UTILS_LIB_DIR)/$(TARGET_LIB)
	ldconfig

clean:
	rm -rf $(UTILS_LIB_DIR)/$(TARGET_LIB) $(PROTOBUF_BUILD_DIR)/*.pb.*

SCHEMA_PROTOS:=$(wildcard deepstream_schema/*.proto)
$(PROTOBUF_BUILD_DIR)/%.pb.cc: $(SCHEMA_PROTOS) Makefile
	$(PROTOBUF_BIN_DIR)/protoc -I./deepstream_schema --cpp_out=$(PROTOBUF_BUILD_DIR)/ $(SCHEMA_PROTOS)
