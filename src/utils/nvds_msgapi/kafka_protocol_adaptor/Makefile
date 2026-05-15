################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2018-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
################################################################################

# this Makefile is to be used to build the kafka_proto protocol adaptor .so

CXX:=g++

PKGS:= glib-2.0

DS_LIB:=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib

PROTOBUF_BUILD_DIR:= .
PROTOBUF_BIN_DIR ?= /opt/proto/bin

SRCS:=  nvds_kafka_proto.cpp kafka_client.cpp json_helper.cpp ../common_src/nvds_utils.cpp $(PROTOBUF_BUILD_DIR)/schema.pb.cc
TARGET_LIB:= libnvds_kafka_proto.so

UTILS_LIB_DIR?= /opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib

CFLAGS:= -fPIC -Wall

CFLAGS+= $(shell pkg-config --cflags $(PKGS))

LIBS:= $(shell pkg-config --libs $(PKGS))
LDFLAGS:= -shared

DS_INC:= ../../../../includes
RDKAFKA_INC:=/usr/include/librdkafka

INC_PATHS:= -I $(DS_INC) -I $(RDKAFKA_INC) -I/opt/tritonclient/include
CFLAGS+= $(INC_PATHS)

LIBS+= -L$(DS_LIB) -L/opt/tritonclient/lib -lrdkafka -ljansson -lnvds_logger -lpthread -lcrypto -lprotobuf
LDFLAGS+= -Wl,-rpath=$(DS_LIB)

all: $(UTILS_LIB_DIR)/$(TARGET_LIB)

$(UTILS_LIB_DIR)/$(TARGET_LIB): $(SRCS)
	@mkdir -p $(UTILS_LIB_DIR)
	$(CXX) -o $@ $^ $(CFLAGS) $(LIBS) $(LDFLAGS)

install: $(UTILS_LIB_DIR)/$(TARGET_LIB)
	ldconfig

clean:
	rm -rf $(UTILS_LIB_DIR)/$(TARGET_LIB)
SCHEMA_PROTOS:=../../nvmsgconv/deepstream_schema/schema.proto
$(PROTOBUF_BUILD_DIR)/%.pb.cc: $(SCHEMA_PROTOS) Makefile
	$(PROTOBUF_BIN_DIR)/protoc -I../../nvmsgconv/deepstream_schema --cpp_out=$(PROTOBUF_BUILD_DIR)/ $<
