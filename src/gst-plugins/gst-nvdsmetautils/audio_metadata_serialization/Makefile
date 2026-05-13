# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

CC:= g++

PKGS:= glib-2.0 gobject-2.0 json-glib-1.0 uuid gstreamer-1.0

NVDS_VERSION?=9.0

LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/

CFLAGS:= -Wall -std=c++11 -shared -fPIC

CFLAGS+= -fPIC -DDS_VERSION=\"9.0\" \
	 -I /usr/local/cuda-$(CUDA_VER)/include \
	 -I ../../../includes

CFLAGS+= -I/opt/tritonclient/include

CFLAGS+= $(shell pkg-config --cflags $(PKGS))
LIBS:= $(shell pkg-config --libs $(PKGS))

LIBS+= -shared -Wl,-no-undefined \
	-L$(LIB_INSTALL_DIR) -lnvdsgst_helper -lnvdsgst_meta -lnvds_meta \
	-Wl,-rpath,$(LIB_INSTALL_DIR)

LIBS+= -L/opt/tritonclient/lib -lprotobuf
PROTOBUF_BUILD_DIR:= $(CURDIR)

SRCFILES:= audio_metadata_serialization.cpp  \
  $(PROTOBUF_BUILD_DIR)/ds_meta.pb.cc

TARGET_LIB:= libnvds_audio_metadata_serialization.so

all: $(TARGET_LIB)

$(TARGET_LIB) : $(SRCFILES)
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)

install: $(TARGET_LIB)
	ldconfig

clean:
	rm -rf $(TARGET_LIB) $(PROTOBUF_BUILD_DIR)/*.pb.*

SCHEMA_PROTOS:=$(wildcard $(CURDIR)/*.proto)
$(PROTOBUF_BUILD_DIR)/%.pb.cc: $(SCHEMA_PROTOS) Makefile
	protoc -I$(CURDIR) --cpp_out=$(PROTOBUF_BUILD_DIR)/ $<
