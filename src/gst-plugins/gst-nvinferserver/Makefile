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

TRITON_GRPC_CLIENT ?= /opt/tritonclient
PROTOBUF_BIN_DIR ?= /opt/proto/bin
NVDS_PROTOBUF_LEGACY_CXX_STANDARD := 17
include ../../../includes/nvds_protobuf.mk

LIB := libnvdsgst_inferserver.so
NVDS_VERSION := 9.2

CUDA_VER?=
# Require CUDA_VER for every goal except a clean-only invocation.
# A bare `make` has empty MAKECMDGOALS and still checks.
ifneq ($(filter-out clean,$(MAKECMDGOALS)),)
  _DS_NEED_CUDA_VER := 1
else ifeq ($(MAKECMDGOALS),)
  _DS_NEED_CUDA_VER := 1
endif
ifeq ($(_DS_NEED_CUDA_VER),1)
ifeq ($(CUDA_VER),)
  $(error "CUDA_VER is not set")
endif
endif

CXX := g++

SRCS := \
    gstnvinferserver_meta_utils.cpp \
    gstnvinferserver.cpp \
    gstnvinferserver_impl.cpp \
    $(NULL)

ifeq ($(shell uname -m), aarch64)
SRCS +=   \
     ../../utils/nvdsinferserver/infer_utils.cpp \
     nvdsinferserver_config.pb.cc \
     nvdsinferserver_plugin.pb.cc \
     nvdsinferserver_common.pb.cc
endif


INCS += $(wildcard *.h)

# The selected client SDK's Abseil configuration can require C++20. Keep the
# older C++17 baseline and reject explicit standards below the SDK requirement.
_TRITON_CLIENT_MIN_CXX_STANDARD := $(NVDS_PROTOBUF_MIN_CXX_STANDARD)
TRITON_CLIENT_CXX_STANDARD ?= $(_TRITON_CLIENT_MIN_CXX_STANDARD)
ifeq ($(_DS_NEED_CUDA_VER),1)
ifeq ($(filter $(TRITON_CLIENT_CXX_STANDARD),17 20 23),)
$(error TRITON_CLIENT_CXX_STANDARD must be 17, 20 or 23)
endif
ifeq ($(_TRITON_CLIENT_MIN_CXX_STANDARD),20)
ifeq ($(TRITON_CLIENT_CXX_STANDARD),17)
$(error Selected Triton client SDK requires C++20 or newer; TRITON_CLIENT_CXX_STANDARD=17 is incompatible)
endif
endif
endif

CFLAGS+= \
    -O3 \
    -DNDEBUG \
    -fPIC \
    -std=c++$(TRITON_CLIENT_CXX_STANDARD) \
    -DDS_VERSION=\"9.2.1\" \
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

# Protobuf's static transitive dependencies must come from the same SDK, rather
# than relying on which Abseil symbols another DeepStream DSO happens to export.
TRITON_CLIENT_LIB_DIR ?= $(NVDS_PROTOBUF_LIB_DIR)
NVDS_PROTOBUF_SELECTED_CXX_STANDARD = $(TRITON_CLIENT_CXX_STANDARD)
NVDS_PROTOBUF_SELECTED_LIB_DIR = $(TRITON_CLIENT_LIB_DIR)
TRITON_ABSL_LIBS := $(addprefix -l:,$(notdir $(sort $(wildcard $(TRITON_CLIENT_LIB_DIR)/libabsl*.a))))
TRITON_PROTOBUF_UTF8_LIBS := $(addprefix -l:,$(notdir $(sort $(wildcard $(TRITON_CLIENT_LIB_DIR)/libutf8_range.a $(TRITON_CLIENT_LIB_DIR)/libutf8_validity.a))))

LIBS := \
    -L$(TRITON_CLIENT_LIB_DIR) \
    -Wl,--start-group \
    -l:libprotobuf.a \
    $(TRITON_ABSL_LIBS) \
    $(TRITON_PROTOBUF_UTF8_LIBS) \
    -Wl,--end-group \
    -L$(LIB_INSTALL_DIR) \
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

ifeq ($(shell uname -m), aarch64)
# Preserve ARM surface compatibility; SBSA additionally enables the server-ARM
# gRPC CUDA-sharing branch. Explicit caller selections override auto-detection.
DEEPSTREAM_VARIANT ?= $(if $(shell test -r /proc/device-tree/model && tr -d '\000' </proc/device-tree/model | grep -F IGX),igx,$(if $(wildcard /etc/nv_tegra_release),tegra,sbsa))
AARCH64_IS_SBSA ?= $(if $(filter sbsa,$(DEEPSTREAM_VARIANT)),1,0)
ifeq ($(_DS_NEED_CUDA_VER),1)
ifneq ($(AARCH64_IS_SBSA),0)
ifneq ($(AARCH64_IS_SBSA),1)
$(error AARCH64_IS_SBSA must be 0 or 1)
endif
endif
endif
CFLAGS += -DIS_TEGRA
ifeq ($(AARCH64_IS_SBSA),1)
CFLAGS += -DAARCH64_IS_SBSA=1
endif
LIBS += -L/usr/lib/aarch64-linux-gnu/nvidia -lcuda
endif

OBJS := $(SRCS:.cpp=.o)
OBJS := $(OBJS:.cc=.o)

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
    $(NVDS_PROTOBUF_MAKEFILE) \
    $(NVDS_PROTOBUF_PROVIDER_STAMP) \
    $(NULL)


all: $(LIB)

%.o: %.cc $(INCS) Makefile
	$(CXX) -c -o $@ $(CFLAGS) $<

%.o: %.cpp $(INCS) Makefile
	$(CXX) -c -o $@ $(CFLAGS) $<

$(LIB): $(OBJS) $(INCS) $(DEP) Makefile
	$(CXX) -o $@ $(LDFLAGS) $(OBJS) $(LIBS)

install: $(LIB)
	cp -rv $(LIB) $(GST_INSTALL_DIR)

clean:
	rm -rf $(OBJS) $(LIB) $(ALL_PROTOS_PBH)  $(ALL_PROTOS_PBCC)
	rm -f "$(NVDS_PROTOBUF_PROVIDER_STAMP)"


%.pb.h %.pb.cc &: ../../../includes/%.proto $(INFERSERVER_PROTOS) Makefile $(NVDS_PROTOBUF_MAKEFILE) $(NVDS_PROTOBUF_PROVIDER_STAMP)
	@mkdir -p $(dir $@)
	$(PROTOBUF_BIN_DIR)/protoc --proto_path=../../../includes --cpp_out=. $<
