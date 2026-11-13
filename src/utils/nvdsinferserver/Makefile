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

LIB:=libnvds_infer_server.so
NVDS_VERSION := 9.2
LIB_INSTALL_DIR ?= /opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/
TRITON_SERVER_STUB ?= build/nvdsinferserver

# External dependencies, update as required.
PROTOBUF_BIN_DIR ?= /opt/proto/bin
TRITON_GRPC_CLIENT ?= /opt/tritonclient
NVDS_PROTOBUF_LEGACY_CXX_STANDARD := 17
include ../../../includes/nvds_protobuf.mk
TRITON_GRPC_CLIENT_LIB_DIR ?= $(NVDS_PROTOBUF_LIB_DIR)
HALF_PRECISION_LIB ?= /opt/half

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

WITH_OPENCV ?= 0

NVCC := /usr/local/cuda-$(CUDA_VER)/bin/nvcc
CXX := g++

SRCS :=   \
     nvdsinferserver_config.pb.cc \
     nvdsinferserver_plugin.pb.cc \
     nvdsinferserver_common.pb.cc \
     infer_base_backend.cpp \
     infer_base_context.cpp \
     infer_proto_utils.cpp \
     infer_utils.cpp \
     infer_surface_bufs.cpp \
     infer_preprocess.cpp \
     infer_postprocess.cpp \
     infer_cuda_utils.cpp \
     infer_cuda_context.cpp \
     infer_trtis_backend.cpp \
     infer_trtis_context.cpp \
     infer_trtis_server.cpp \
     infer_trtis_utils.cpp \
     infer_lstm.cpp \
     infer_preprocess_kernel.cu \
     infer_simple_context.cpp \
     infer_simple_runtime.cpp \
     infer_extra_processor.cpp \
     infer_stream_manager.cpp \
     infer_grpc_context.cpp \
     infer_grpc_client.cpp \
     infer_grpc_backend.cpp \
     $(NULL)


# model_config.pb.h is selected below; an old generated local copy must not
# shadow or become a dependency of the selected client SDK's own header.
INCS := $(filter-out model_config.pb.h,$(wildcard *.h))

# The October client SDK configures Abseil to use C++20 ordering types. Older
# providers retain C++17 unless the caller explicitly selects a newer standard.
_TRITON_CLIENT_MIN_CXX_STANDARD := $(NVDS_PROTOBUF_MIN_CXX_STANDARD)
TRITON_CLIENT_CXX_STANDARD ?= $(_TRITON_CLIENT_MIN_CXX_STANDARD)
NVDS_PROTOBUF_SELECTED_CXX_STANDARD = $(TRITON_CLIENT_CXX_STANDARD)
NVDS_PROTOBUF_SELECTED_LIB_DIR = $(TRITON_GRPC_CLIENT_LIB_DIR)
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

CFLAGS := \
    -O3 \
    -s \
    -fPIC \
    -std=c++$(TRITON_CLIENT_CXX_STANDARD)  \
    -DNDEBUG  \
    -Wall \
    -Werror \
    -Wno-deprecated-declarations \
    -DTRTIS_ENABLE_GPU  \
    -DHAVE_CONFIG_H \
    -DTRITON_ENABLE_GPU \
    -DFOR_PRIVATE \
    $(NULL)

INC_PATHS += \
    $(TRITON_GRPC_CLIENT)/include \
    ../../../includes \
    /usr/local/cuda-$(CUDA_VER)/include \
    $(HALF_PRECISION_LIB)/include/ \
    tritonserver_include \
    ../../../includes/nvdsinferserver \
    .
    $(NULL)

CFLAGS += $(addprefix -I, $(INC_PATHS))

ifeq ($(WITH_OPENCV),1)
CFLAGS += -DWITH_OPENCV
ifeq ($(shell uname -m), aarch64)
CFLAGS +=-I/usr/include/opencv4
endif
endif

# Use the selected SDK's own Abseil archive set. Releases change archive names
# and transitive dependencies; combining names from another SDK is unsafe.
# Use that same complete provider for linking, static dependencies and RPATH.
TRITON_ABSL_LIBS := $(addprefix -l:,$(notdir $(sort $(wildcard $(TRITON_GRPC_CLIENT_LIB_DIR)/libabsl*.a))))
TRITON_PROTOBUF_UTF8_LIBS := $(addprefix -l:,$(notdir $(sort $(wildcard $(TRITON_GRPC_CLIENT_LIB_DIR)/libutf8_range.a $(TRITON_GRPC_CLIENT_LIB_DIR)/libutf8_validity.a))))
TRITON_CORE_VISIBILITY_MAP := .build/triton-client-visibility.map

# Modern client SDK archives contain private protobuf/Abseil state. Keep the
# client's generated ModelConfig and DeepStream's generated messages in this
# core's single runtime, rather than passing them between two private runtimes.
# The SDK's TritonClient::grpcclient_static target links gRPC::grpc++,
# gRPC::grpc and protobuf::libprotobuf. Their matching support archives are
# supplied by that same SDK; archive groups pull only the required objects.
_TRITON_CORE_PRIVATE_STATIC_CLIENT := $(shell test "$(NVDS_PROTOBUF_HEADER_VERSION)" -ge 6000000 2>/dev/null && echo 1)
ifeq ($(_TRITON_CORE_PRIVATE_STATIC_CLIENT),1)
TRITON_CORE_REQUIRED_ARCHIVES := $(addprefix $(TRITON_GRPC_CLIENT_LIB_DIR)/, \
    libgrpcclient_static.a libgrpc++.a libgrpc.a libgpr.a libprotobuf.a \
    libcares.a libre2.a libaddress_sorting.a libutf8_range.a libutf8_validity.a)
TRITON_CORE_ABSL_ARCHIVES := $(sort $(wildcard $(TRITON_GRPC_CLIENT_LIB_DIR)/libabsl*.a))
TRITON_CORE_UPB_ARCHIVES := $(sort $(wildcard $(TRITON_GRPC_CLIENT_LIB_DIR)/libupb*.a))
TRITON_CORE_UTF8_ARCHIVES := $(sort $(wildcard $(TRITON_GRPC_CLIENT_LIB_DIR)/libutf8*.a))
TRITON_CORE_CLIENT_ARCHIVES := $(sort $(TRITON_CORE_REQUIRED_ARCHIVES) \
    $(TRITON_CORE_ABSL_ARCHIVES) $(TRITON_CORE_UPB_ARCHIVES) $(TRITON_CORE_UTF8_ARCHIVES))
ifeq ($(_DS_NEED_CUDA_VER),1)
TRITON_CORE_MISSING_ARCHIVES := $(filter-out $(wildcard $(TRITON_CORE_REQUIRED_ARCHIVES)),$(TRITON_CORE_REQUIRED_ARCHIVES))
ifneq ($(TRITON_CORE_MISSING_ARCHIVES),)
$(error Modern Triton client core requires matching static SDK archives: $(TRITON_CORE_MISSING_ARCHIVES))
endif
ifeq ($(TRITON_CORE_ABSL_ARCHIVES),)
$(error Modern Triton client core requires the selected SDK's Abseil archives)
endif
ifeq ($(TRITON_CORE_UPB_ARCHIVES),)
$(error Modern Triton client core requires the selected SDK's upb archives)
endif
endif
TRITON_CORE_CLIENT_LIBS := $(TRITON_CORE_CLIENT_ARCHIVES)
TRITON_CORE_SUPPORT_LIBS :=
TRITON_CORE_VISIBILITY_DEP := $(TRITON_CORE_VISIBILITY_MAP)
# Hide archive-owned ModelConfig/protobuf/gRPC/Abseil definitions, not the
# source-object DeepStream protobuf types and typed reflection entry points.
# GNU ld also keeps absent optional TLS-init weak hooks non-dynamic, so another
# DSO cannot supply an initializer for this private runtime's trivial TLS data.
LDFLAGS += -Wl,--exclude-libs,ALL \
    -Wl,-z,nodynamic-undefined-weak \
    -Wl,--defsym,nvds_inferserver_static_protobuf_version=$(NVDS_PROTOBUF_HEADER_VERSION) \
    -Wl,--version-script=$(TRITON_CORE_VISIBILITY_MAP)
else
# Preserve the older SDK's shared-client link and its existing archive support.
TRITON_CORE_CLIENT_LIBS := -lgrpcclient -l:libprotobuf.a -l:libcares.a
TRITON_CORE_SUPPORT_LIBS := -Wl,--start-group $(TRITON_ABSL_LIBS) $(TRITON_PROTOBUF_UTF8_LIBS) -Wl,--end-group -l:libre2.a
endif

LIBS := \
    -L$(TRITON_SERVER_STUB) \
    -L$(TRITON_GRPC_CLIENT_LIB_DIR) \
    $(TRITON_CORE_CLIENT_LIBS) \
    -ltritonserver \
    -lnvbufsurftransform \
    -lnvbufsurface \
    -L$(LIB_INSTALL_DIR) \
    -lnvdsgst_helper \
    -lnvdsgst_meta \
    -lnvds_meta \
    -lnvds_inferlogger \
    -lnvds_inferutils \
    $(TRITON_CORE_SUPPORT_LIBS) \
    -ldl \
    -pthread \
    -lz \
    -lcrypto \
    -lssl \
    -L/usr/local/cuda-$(CUDA_VER)/lib64/ \
    -lcudart \
    -L$(LIB_INSTALL_DIR) \
     $(NULL)


ifeq ($(WITH_OPENCV),1)
LIBS += \
    -lopencv_objdetect \
    -lopencv_imgproc \
    -lopencv_core \
    $(NULL)
endif

ifeq ($(shell uname -m), aarch64)
# Keep the existing ARM surface ABI while distinguishing server ARM for gRPC
# CUDA sharing. Explicit caller selections override standalone detection.
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
LIBS += -lcuda
endif

LDFLAGS += \
    -Wl,--no-undefined \
    -Wl,--as-needed \
    -Wl,--allow-shlib-undefined \
    -Wl,-rpath,$(LIB_INSTALL_DIR) \
    -Wl,-rpath,$(TRITON_GRPC_CLIENT_LIB_DIR) \
    -Wl,-rpath,/usr/local/cuda-$(CUDA_VER)/lib64 \
    -shared \
    -s \
    $(NULL)


TRITONSERVER_HEADERS := $(wildcard tritonserver_include/*.h)
TRITONSERVER_PROTOS := $(wildcard tritonserver_include/*.proto)
TRITON_MODEL_CONFIG_HEADER := $(wildcard $(TRITON_GRPC_CLIENT)/include/model_config.pb.h)
ifneq ($(TRITON_MODEL_CONFIG_HEADER),)
# Newer client SDKs supply their exact schema/header alongside libgrpcclient.
# Retain local generation only for older providers that do not ship the header.
TRITONSERVER_PROTOS := $(filter-out tritonserver_include/model_config.proto,$(TRITONSERVER_PROTOS))
endif

INFERSERVER_PROTOS := $(wildcard ../../../includes/*.proto)

ALL_PROTOS := $(INFERSERVER_PROTOS) $(TRITONSERVER_PROTOS)
ALL_PROTOS_PBCC := $(notdir $(ALL_PROTOS:.proto=.pb.cc))
ALL_PROTOS_PBH := $(notdir $(ALL_PROTOS:.proto=.pb.h))

DEPS := $(TRITON_SERVER_STUB)

INCS += \
    $(INFERSERVER_PROTOS) \
    $(TRITONSERVER_PROTOS) \
    $(TRITONSERVER_HEADERS) \
    $(TRITON_MODEL_CONFIG_HEADER) \
    $(ALL_PROTOS_PBCC) \
    $(ALL_PROTOS_PBH) \
    $(DEPS)/libtritonserver.so \
    $(NVDS_PROTOBUF_MAKEFILE) \
    $(NVDS_PROTOBUF_PROVIDER_STAMP) \
    $(TRITON_CORE_CLIENT_ARCHIVES) \
    $(NULL) \

.PRECIOUS:  $(ALL_PROTOS_PBCC)
.PRECIOUS:  $(ALL_PROTOS_PBH)

OBJS:= $(SRCS:.cpp=.o)
OBJS:= $(OBJS:.cc=.o)
OBJS:= $(OBJS:.cu=.o)

all: $(LIB)

%.o: %.cc $(INCS) Makefile
	$(CXX) -c -o $@ $(CFLAGS) $<

%.o: %.cpp $(INCS) Makefile
	$(CXX) -c -o $@ $(CFLAGS) $<

%.o: %.cu $(INCS) Makefile
	@echo $(CFLAGS)
	$(NVCC) -c -o $@ --compiler-options '-fPIC -I../../../includes/nvdsinferserver -I../../../includes'  $<

$(LIB): $(OBJS) $(INCS) $(DEP) Makefile $(TRITON_CORE_VISIBILITY_DEP)
	$(CXX) -o $@ $(LDFLAGS) $(OBJS) -Wl,--start-group $(LIBS) -Wl,--end-group

install: $(LIB)
	cp -rv $(LIB) $(LIB_INSTALL_DIR)

clean:
	rm -rf $(OBJS) $(LIB)
	rm -rf $(DEPS)/libtritonserver.so $(ALL_PROTOS_PBCC) $(ALL_PROTOS_PBH)
	rm -f "$(NVDS_PROTOBUF_PROVIDER_STAMP)"
	rm -f "$(TRITON_CORE_VISIBILITY_MAP)"

ifeq ($(_TRITON_CORE_PRIVATE_STATIC_CLIENT),1)
# --exclude-libs hides archive definitions, but upstream inline/template
# definitions can also be emitted in ordinary core objects. Localize only
# their mangled ownership prefixes; do not match protobuf names in arguments
# of DeepStream APIs. Enumerate CV/ref qualifiers instead of a glob between
# the nested-name prefix and namespace token, which could overmatch arguments.
$(TRITON_CORE_VISIBILITY_MAP): Makefile $(NVDS_PROTOBUF_PROVIDER_STAMP)
	@set -eu; \
	  mkdir -p '$(@D)'; \
	  temporary='$@.'$$$$'.tmp'; \
	  trap 'rm -f "$$temporary"' EXIT HUP INT TERM; \
	  { \
	    printf '{\n  global: *;\n  local:\n'; \
	    for owner in 9inference 6triton6client 6google8protobuf 4absl 4grpc 9grpc_core 9grpc_impl 17grpc_event_engine 3upb 3re2; do \
	      for cv in '' K V VK r rK rV rVK; do \
	        for ref in '' R O; do \
	          for prefix in _ZN _ZZN _ZGVN _ZGVZN _ZTHN _ZTWN; do \
	            printf '    %s%s%s%s*;\n' "$$prefix" "$$cv" "$$ref" "$$owner"; \
	          done; \
	        done; \
	        printf '    _ZTI%sN%s*;\n    _ZTS%sN%s*;\n' "$$cv" "$$owner" "$$cv" "$$owner"; \
	      done; \
	      for prefix in _ZTVN _ZTTN _ZTCN; do \
	        printf '    %s%s*;\n' "$$prefix" "$$owner"; \
	      done; \
	    done; \
	    for prefix in grpc_ gpr_ upb_ _upb_ ares_ address_sorting_ utf8_range_ utf8_validity_; do \
	      printf '    %s*;\n' "$$prefix"; \
	    done; \
	    printf '};\n'; \
	  } > "$$temporary"; \
	  if ! cmp -s "$$temporary" '$@'; then mv -f "$$temporary" '$@'; fi
endif

$(DEPS)/libtritonserver.so: tritonserver_stub/libtritonserver_dummy.export Makefile
	@mkdir -p $(dir $@)
	awk '{ print "void",$$1,"() {}" }' $< | $(CC) -x c -O0 -fPIC -shared -Wl,-soname=libtritonserver.so -Wl,--strip-all -o $@ -


%.pb.h %.pb.cc &: ../../../includes/%.proto $(INFERSERVER_PROTOS) Makefile $(NVDS_PROTOBUF_MAKEFILE) $(NVDS_PROTOBUF_PROVIDER_STAMP)
	@mkdir -p $(dir $@)
	$(PROTOBUF_BIN_DIR)/protoc --proto_path=../../../includes --cpp_out=. $<

%.pb.h %.pb.cc &: tritonserver_include/%.proto $(TRITONSERVER_PROTOS) Makefile $(NVDS_PROTOBUF_MAKEFILE) $(NVDS_PROTOBUF_PROVIDER_STAMP)
	@mkdir -p $(dir $@)
	$(PROTOBUF_BIN_DIR)/protoc --proto_path=./tritonserver_include --cpp_out=. $<
