################################################################################
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
################################################################################

# External dependencies, update as required.
TRITON_GRPC_CLIENT ?= /opt/tritonclient
PROTOBUF_BIN_DIR ?= /opt/proto/bin

LIB := libnvdsgst_inferserver.so
include ../Rules.mk
NVDS_VERSION := 9.0

CUDA_VER ?=
ifeq ($(CUDA_VER),)
  $(error "CUDA_VER is not set")
endif

CXX := g++

SRCS := \
    gstnvinferserver_meta_utils.cpp \
    gstnvinferserver.cpp \
    gstnvinferserver_impl.cpp \
    $(NULL)



INCS += $(wildcard *.h)

CFLAGS+= \
    -O3 \
    -DNDEBUG \
    -fPIC \
    -std=c++17 \
    -DDS_VERSION=\"9.0.0\" \
    -DFOR_PRIVATE \
    $(NULL)

INC_PATHS := \
    $(TRITON_GRPC_CLIENT)/include \
    /usr/local/cuda-$(CUDA_VER)/include \
    ../../../includes \
    ../../../includes/nvdsinferserver \
    ../../utils/nvdsinferserver \
    ../gst-nvdspreprocess/include \
    . \
    $(NULL)

CFLAGS += $(addprefix -I, $(INC_PATHS))

GST_INSTALL_DIR ?= /opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/gst-plugins/
LIB_INSTALL_DIR ?= /opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/

LDFLAGS := \
    -shared \
    -Wl,-no-undefined \
    $(NULL)

LIBS := \
    -L$(TRITON_GRPC_CLIENT)/lib \
    -l:libprotobuf.a \
    -L/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib -L/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib -L/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib -Wl,-rpath-link,/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib -L/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib -L/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib -L/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib -Wl,-rpath-link,/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib \
    -lnvdsgst_helper \
    -lnvdsgst_meta \
    -lnvds_meta \
    -lnvds_infer_server \
    -lnvbufsurface \
    -lnvbufsurftransform \
    -L/usr/local/cuda-$(CUDA_VER)/lib64/ \
    -lcudart \
    -ldl \
    -lpthread \
    -Wl,-rpath,$(LIB_INSTALL_DIR) \
    -lnvdsgst_customhelper \
    $(NULL)


OBJS := $(SRCS:.cpp=.o)
OBJS := $(OBJS:.cc=.o)
OBJS := $(addprefix $(OBJ_DIR)/,$(OBJS))

PKGS := gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0
CFLAGS += $(shell pkg-config --cflags $(PKGS))
LIBS += $(shell pkg-config --libs $(PKGS))

INFERSERVER_PROTOS := $(wildcard ../../../includes/*.proto)

ALL_PROTOS := $(INFERSERVER_PROTOS)
ALL_PROTOS_PBCC := $(notdir $(ALL_PROTOS:.proto=.pb.cc))
ALL_PROTOS_PBH := $(notdir $(ALL_PROTOS:.proto=.pb.h))

.PRECIOUS:  $(ALL_PROTOS_PBCC)
.PRECIOUS:  $(ALL_PROTOS_PBH)


INCS += \
    $(INFERSERVER_PROTOS) \
    $(ALL_PROTOS_PBCC) \
    $(ALL_PROTOS_PBH) \
    $(NULL)


all: $(LIB_DIR)/$(LIB)

$(OBJ_DIR)/%.o: %.cc $(INCS) Makefile
	@mkdir -p $(OBJ_DIR)
	$(CXX) -c -o $@ $(CFLAGS) $<

$(OBJ_DIR)/%.o: %.cpp $(INCS) Makefile
	@mkdir -p $(OBJ_DIR)
	$(CXX) -c -o $@ $(CFLAGS) $<

$(LIB_DIR)/$(LIB): $(OBJS) $(INCS) $(DEP) Makefile
	@mkdir -p $(LIB_DIR)
	$(CXX) -o $(OBJ_DIR)/$(LIB) $(LDFLAGS) $(OBJS) $(LIBS)
	cp $(OBJ_DIR)/$(LIB) $@

install: $(LIB_DIR)/$(LIB)
	ldconfig

clean:
	rm -rf $(OBJ_DIR) $(LIB_DIR)/$(LIB) $(ALL_PROTOS_PBH)  $(ALL_PROTOS_PBCC)


%.pb.h %.pb.cc: ../../../includes/%.proto $(INFERSERVER_PROTOS) Makefile
	@mkdir -p $(dir $@)
	$(PROTOBUF_BIN_DIR)/protoc --proto_path=../../../includes --cpp_out=. $<
