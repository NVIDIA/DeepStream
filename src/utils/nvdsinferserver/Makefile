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

LIB:=libnvds_infer_server.so
LIB_DIR = /opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib
OBJ_DIR := .build
include ../../Rules.mk
NVDS_VERSION := 9.0
LIB_INSTALL_DIR ?= /opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/
TRITON_SERVER_STUB ?= build/nvdsinferserver

# External dependencies, update as required.
PROTOBUF_BIN_DIR ?= /opt/proto/bin
TRITON_GRPC_CLIENT ?= /opt/tritonclient
HALF_PRECISION_LIB ?= /opt/half

CUDA_VER ?=
ifeq ($(CUDA_VER),)
  $(error "CUDA_VER is not set")
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


INCS := $(wildcard *.h)

CFLAGS := \
    -O3 \
    -s \
    -fPIC \
    -std=c++17  \
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

LIBS := \
    -L$(TRITON_SERVER_STUB) \
    -L$(TRITON_GRPC_CLIENT)/lib \
    -l:libgrpcclient_static.a \
    -l:libprotobuf.a \
    -l:libgrpc++.a \
    -l:libgrpc.a \
    -l:libgpr.a \
    -l:libcares.a \
    -l:libaddress_sorting.a  \
    -ltritonserver \
    -lnvbufsurftransform \
    -lnvbufsurface \
    -L/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib -L/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib -L/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib -Wl,-rpath-link,/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib -L/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib -L/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib -L/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib -Wl,-rpath-link,/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib \
    -lnvdsgst_helper \
    -lnvdsgst_meta \
    -lnvds_meta \
    -lnvds_inferlogger \
    -lnvds_inferutils \
    -l:libabsl_bad_optional_access.a \
    -l:libabsl_bad_variant_access.a \
    -l:libabsl_cordz_info.a \
    -l:libabsl_cord.a \
    -l:libabsl_cordz_functions.a \
    -l:libabsl_cordz_handle.a \
    -l:libabsl_cordz_sample_token.a \
    -l:libabsl_cord_internal.a \
    -l:libabsl_exponential_biased.a \
    -l:libabsl_hash.a \
    -l:libabsl_city.a \
    -l:libabsl_low_level_hash.a \
    -l:libabsl_random_distributions.a \
    -l:libabsl_random_internal_randen.a \
    -l:libabsl_random_internal_randen_hwaes_impl.a \
    -l:libabsl_random_internal_randen_hwaes.a \
    -l:libabsl_random_internal_platform.a \
    -l:libabsl_random_internal_pool_urbg.a \
    -l:libabsl_random_internal_randen_slow.a \
    -l:libabsl_random_internal_seed_material.a \
    -l:libabsl_random_seed_gen_exception.a \
    -l:libabsl_random_seed_sequences.a \
    -l:libabsl_raw_hash_set.a \
    -l:libabsl_raw_logging_internal.a \
    -l:libabsl_scoped_set_env.a \
    -l:libabsl_spinlock_wait.a \
    -l:libabsl_stacktrace.a \
    -l:libabsl_status.a \
    -l:libabsl_statusor.a \
    -l:libabsl_strerror.a \
    -l:libabsl_str_format_internal.a \
    -l:libabsl_strings.a \
    -l:libabsl_strings_internal.a \
    -l:libabsl_synchronization.a \
    -l:libabsl_symbolize.a \
    -l:libabsl_throw_delegate.a \
    -l:libabsl_time.a \
    -l:libabsl_time_zone.a \
    -l:libabsl_malloc_internal.a \
    -l:libabsl_int128.a \
    -l:libabsl_demangle_internal.a \
    -l:libabsl_debugging_internal.a \
    -l:libabsl_base.a \
    -l:libabsl_crc_cord_state.a \
    -l:libabsl_log_flags.a \
    -l:libabsl_flags_reflection.a \
    -l:libabsl_flags_config.a \
    -l:libabsl_flags_program_name.a \
    -l:libabsl_flags_private_handle_accessor.a \
    -l:libabsl_flags_internal.a \
    -l:libabsl_crc32c.a \
    -l:libabsl_crc_internal.a \
    -l:libabsl_flags_commandlineflag.a \
    -l:libabsl_flags_parse.a \
    -l:libabsl_flags_marshalling.a \
    -l:libabsl_flags_commandlineflag_internal.a \
    -l:libupb.a \
    -l:libre2.a \
    -ldl \
    -pthread \
    -lz \
    -lcrypto \
    -lssl \
    -L/usr/local/cuda-$(CUDA_VER)/lib64/ \
    -lcudart \
    -L/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib -L/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib -L/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib -Wl,-rpath-link,/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib -L/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib -L/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib -L/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib -Wl,-rpath-link,/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib \
     $(NULL)


ifeq ($(WITH_OPENCV),1)
LIBS += \
    -lopencv_objdetect \
    -lopencv_imgproc \
    -lopencv_core \
    $(NULL)
endif

ifeq ($(shell uname -m), aarch64)
CFLAGS += -DIS_TEGRA
LIBS += -lcuda
endif

LDFLAGS += \
    -Wl,--no-undefined \
    -Wl,--as-needed \
    -Wl,--allow-shlib-undefined \
    -Wl,-rpath,$(LIB_INSTALL_DIR) \
    -Wl,-rpath,/usr/local/$(CUDA_VER)/lib64 \
    -shared \
    -s \
    $(NULL)


TRITONSERVER_HEADERS := $(wildcard tritonserver_include/*.h)
TRITONSERVER_PROTOS := $(wildcard tritonserver_include/*.proto)

INFERSERVER_PROTOS := $(wildcard ../../../includes/*.proto)

ALL_PROTOS := $(INFERSERVER_PROTOS) $(TRITONSERVER_PROTOS)
ALL_PROTOS_PBCC := $(notdir $(ALL_PROTOS:.proto=.pb.cc))
ALL_PROTOS_PBH := $(notdir $(ALL_PROTOS:.proto=.pb.h))

DEPS := $(TRITON_SERVER_STUB)

INCS += \
    $(INFERSERVER_PROTOS) \
    $(TRITONSERVER_PROTOS) \
    $(TRITONSERVER_HEADERS) \
    $(ALL_PROTOS_PBCC) \
    $(ALL_PROTOS_PBH) \
    $(DEPS)/libtritonserver.so \
    $(NULL) \

.PRECIOUS:  $(ALL_PROTOS_PBCC)
.PRECIOUS:  $(ALL_PROTOS_PBH)

OBJS:= $(SRCS:.cpp=.o)
OBJS:= $(OBJS:.cc=.o)
OBJS:= $(OBJS:.cu=.o)
OBJS := $(addprefix $(OBJ_DIR)/,$(OBJS))

all: $(LIB_DIR)/$(LIB)

$(OBJ_DIR)/%.o: %.cc $(INCS) Makefile
	@mkdir -p $(OBJ_DIR)
	$(CXX) -c -o $@ $(CFLAGS) $<

$(OBJ_DIR)/%.o: %.cpp $(INCS) Makefile
	@mkdir -p $(OBJ_DIR)
	$(CXX) -c -o $@ $(CFLAGS) $<

$(OBJ_DIR)/%.o: %.cu $(INCS) Makefile
	@mkdir -p $(OBJ_DIR)
	@echo $(CFLAGS)
	$(NVCC) -c -o $@ --compiler-options '-fPIC -I../../../includes/nvdsinferserver -I../../../includes'  $<

$(LIB_DIR)/$(LIB): $(OBJS) $(INCS) $(DEP) Makefile
	@mkdir -p $(LIB_DIR)
	$(CXX) -o $@ $(LDFLAGS) $(OBJS)  $(LIBS)

install: $(LIB_DIR)/$(LIB)
	ldconfig

clean:
	rm -rf $(OBJ_DIR) $(LIB_DIR)/$(LIB)
	rm -rf $(DEPS)/libtritonserver.so $(ALL_PROTOS_PBCC) $(ALL_PROTOS_PBH)

$(DEPS)/libtritonserver.so: tritonserver_stub/libtritonserver_dummy.export Makefile
	@mkdir -p $(dir $@)
	awk '{ print "void",$$1,"() {}" }' $< | $(CC) -x c -O0 -fPIC -shared -Wl,-soname=libtritonserver.so -Wl,--strip-all -o $@ -


%.pb.h %.pb.cc: ../../../includes/%.proto $(INFERSERVER_PROTOS) Makefile
	@mkdir -p $(dir $@)
	$(PROTOBUF_BIN_DIR)/protoc --proto_path=../../../includes --cpp_out=. $<

%.pb.h %.pb.cc: tritonserver_include/%.proto $(TRITONSERVER_PROTOS) Makefile
	@mkdir -p $(dir $@)
	$(PROTOBUF_BIN_DIR)/protoc --proto_path=./tritonserver_include --cpp_out=. $<
