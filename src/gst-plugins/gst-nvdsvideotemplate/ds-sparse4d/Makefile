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

_SPARSE4D_PUBLIC_MAKEFILE := $(lastword $(MAKEFILE_LIST))
CUDA_VER?=
.DEFAULT_GOAL := all
# Keep clean-only usable without provisioning CUDA or the dependency stack.
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

TARGET_DEVICE = $(shell $(CXX) -dumpmachine | cut -f1 -d -)
CXX:= g++

SRCS:= customlib_impl.cpp \
       utils/config_parser.cpp utils/dump_utils.cpp utils/tensor_init.cpp \
       preprocess/calibration_parser.cpp preprocess/preprocessor.cpp \
       infer/engine_builder.cpp infer/tensor_rt_engine.cpp \
       postprocess/postprocessor.cpp postprocess/decoder.cpp postprocess/bbox3d_utils.cpp postprocess/display_bbox3d.cpp \
       postprocess/instance_bank.cpp postprocess/sparse_box_3d_generator.cpp

INCS:= $(wildcard *.h) \
       $(wildcard utils/*.h) \
       $(wildcard preprocess/*.h) \
       $(wildcard infer/*.h) \
       $(wildcard postprocess/*.h)

# Keep the linked output with its objects too: a newer root-level library from
# another stack must not suppress relinking when switching back to this one.
LIB = $(BUILD_DIR)/libnvdsgst_sparse4d.so

NVDS_VERSION:=9.2

# LibTorch paths
TORCH_PATH ?= /usr/local/lib/python3/dist-packages/torch
TORCH_INCLUDE_DIRS := $(TORCH_PATH)/include
TORCH_INCLUDE_DIRS2 := $(TORCH_PATH)/include/torch/csrc/api/include
TORCH_LIBRARY_DIRS := $(TORCH_PATH)/lib
_SPARSE4D_TORCH_DETECTED_ABI := $(shell value=$$(sed -n 's/.*_GLIBCXX_USE_CXX11_ABI=\([01]\).*/\1/p' "$(TORCH_PATH)/share/cmake/Torch/TorchConfig.cmake" 2>/dev/null | head -1); if test -n "$$value"; then printf '%s' "$$value"; else nm -D --defined-only "$(TORCH_LIBRARY_DIRS)/libc10.so" 2>/dev/null | awk '/torchCheckFail.*__cxx11/ {print 1; exit} /torchCheckFail.*RKSs/ {print 0; exit}'; fi)
TORCH_CXX11_ABI ?= $(_SPARSE4D_TORCH_DETECTED_ABI)
# Recent LibTorch providers publish their minimum standard as a CMake feature.
# Preserve the C++17 baseline for older providers and explicit caller overrides.
_SPARSE4D_TORCH_DETECTED_CXX_STANDARD := $(shell sed -n 's/.*cxx_std_\([0-9][0-9]*\).*/\1/p' "$(TORCH_PATH)/share/cmake/Torch/TorchConfig.cmake" 2>/dev/null | sort -nu | tail -1)
_SPARSE4D_TORCH_MIN_CXX_STANDARD := $(if $(_SPARSE4D_TORCH_DETECTED_CXX_STANDARD),$(_SPARSE4D_TORCH_DETECTED_CXX_STANDARD),17)
TORCH_CXX_STANDARD ?= $(_SPARSE4D_TORCH_MIN_CXX_STANDARD)
EIGEN_INCLUDE_DIR ?= /usr/include/eigen3
JSON_INCLUDE_DIR ?= /usr/include
TRT_INCLUDE_DIR ?= /usr/include/$(shell $(CXX) -dumpmachine)
TRT_LIB_DIR ?= /usr/lib/$(shell $(CXX) -dumpmachine)
_sparse4d_empty :=
_sparse4d_space := $(_sparse4d_empty) $(_sparse4d_empty)
TRT_HEADER_VERSION := $(subst $(_sparse4d_space),.,$(strip $(shell version=$$(printf '%s\n' 'NV_TENSORRT_MAJOR NV_TENSORRT_MINOR NV_TENSORRT_PATCH NV_TENSORRT_BUILD' | $(CXX) -E -P -x c++ -include "$(TRT_INCLUDE_DIR)/NvInferVersion.h" - 2>/dev/null) && printf '%s\n' "$$version")))
ifeq ($(TARGET_DEVICE),aarch64)
# Architecture alone cannot distinguish the JetPack and generic-SBSA stacks.
# Containers without the Jetson host marker and cross-builds must set this.
DEEPSTREAM_VARIANT ?= $(if $(shell test -r /proc/device-tree/model && tr -d '\000' </proc/device-tree/model | grep -F IGX),igx,$(if $(wildcard /etc/nv_tegra_release),tegra,sbsa))
ifeq ($(DEEPSTREAM_VARIANT),igx)
# IGX keeps its installed legacy stack. Inspect its selected real headers;
# do not introduce a TensorRT11 pin or change the TensorRT10 source path.
ifeq ($(strip $(TRT_VER)),)
override TRT_VER := $(TRT_HEADER_VERSION)
endif
else ifeq ($(DEEPSTREAM_VARIANT),tegra)
TRT_VER ?= 11.4.1.31
else ifeq ($(DEEPSTREAM_VARIANT),sbsa)
TRT_VER ?= 11.4.0.106
else
$(error DEEPSTREAM_VARIANT must be tegra, sbsa or igx for an AArch64 Sparse4D build)
endif
else
TRT_VER ?= 11.4.0.106
endif
BUILD_DIR ?= .build/$(TARGET_DEVICE)-cuda$(CUDA_VER)-trt$(TRT_VER)

# Repository paths are rewritten to sources/libs by build/package.sh.
CFLAGS+= -O3 -fPIC -DDS_VERSION=\"9.2.1\" -std=c++$(TORCH_CXX_STANDARD) -Wall -Werror -Wno-error=deprecated-declarations \
         -DHAVE_CONFIG_H -fexceptions \
         -I. -I../includes -I../../../../includes \
         -I../../../utils/nvdsinfer -I../../../utils/nvdsinferutils/include \
         -I../../gst-nvdspreprocess -I../../gst-nvdspreprocess/include \
         -I/usr/local/cuda-$(CUDA_VER)/include \
         -I$(TORCH_INCLUDE_DIRS) -I$(TORCH_INCLUDE_DIRS2) \
         -I$(EIGEN_INCLUDE_DIR) -I$(JSON_INCLUDE_DIR) -I$(TRT_INCLUDE_DIR) \
         -Iutils -Ipreprocess -Iinfer -Ipostprocess
CFLAGS += $(if $(TORCH_CXX11_ABI),-D_GLIBCXX_USE_CXX11_ABI=$(TORCH_CXX11_ABI))

GST_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/gst-plugins/
LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/

ifeq ($(TARGET_DEVICE),aarch64)
	LIBS := -L/usr/lib/aarch64-linux-gnu/tegra
endif

# LibTorch libraries
# CUDA backend registration can be constructor-only. Retain those providers
# while restoring the linker's previous as-needed state for unrelated libraries.
TORCH_LIBS := -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda

LIBS += -shared -Wl,-no-undefined \
	-L/usr/local/cuda-$(CUDA_VER)/lib64/ -lcudart -ldl -lpthread \
	-L$(LIB_INSTALL_DIR) -lnvdsgst_helper -lnvdsgst_meta -lnvds_meta -lnvds_infer \
	-lnvbufsurface -lnvbufsurftransform -lnvdsbufferpool \
	-L$(GST_INSTALL_DIR) -lnvdsgst_preprocess \
	-L$(TORCH_LIBRARY_DIRS) -Wl,--push-state,--no-as-needed $(TORCH_LIBS) -Wl,--pop-state \
	-lyaml-cpp -L$(TRT_LIB_DIR) -lnvinfer -lnvonnxparser \
	-Wl,-rpath,$(LIB_INSTALL_DIR) \
	-Wl,-rpath,$(GST_INSTALL_DIR) \
	-Wl,-rpath,$(TORCH_LIBRARY_DIRS)

OBJS:= $(addprefix $(BUILD_DIR)/,$(SRCS:.cpp=.o))
PKGS:= gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0

CFLAGS+=$(shell pkg-config --cflags $(PKGS))
LIBS+=$(shell pkg-config --libs $(PKGS))

all: validate_sparse4d_binary

$(BUILD_DIR)/%.o: %.cpp $(INCS) $(_SPARSE4D_PUBLIC_MAKEFILE)
	@mkdir -p $(dir $@)
	$(CXX) -MMD -MP -c -o $@ $(CFLAGS) $<

$(LIB): $(OBJS) $(_SPARSE4D_PUBLIC_MAKEFILE)
	@echo $(CFLAGS)
	$(CXX) -o $@ $(OBJS) $(LIBS)

.PHONY: check_sparse4d_dependencies validate_sparse4d_binary
check_sparse4d_dependencies:
	@set -e; \
	  case "$(TORCH_CXX_STANDARD)" in 11|14|17|20|23|26) ;; *) echo 'TORCH_CXX_STANDARD must name a numeric C++ standard such as 17 or 20.' >&2; exit 1 ;; esac; \
	  test "$(TORCH_CXX_STANDARD)" -ge "$(_SPARSE4D_TORCH_MIN_CXX_STANDARD)" || { echo 'Selected TORCH_CXX_STANDARD is lower than the LibTorch provider requires.' >&2; exit 1; }; \
	  test -f "$(TORCH_INCLUDE_DIRS2)/torch/torch.h" || { echo 'Set TORCH_PATH to a matching CUDA LibTorch distribution.' >&2; exit 1; }; \
	  test "$(TORCH_CXX11_ABI)" = 1 && test "$(_SPARSE4D_TORCH_DETECTED_ABI)" = 1 || { echo 'Sparse4D requires verified ABI1 LibTorch to match the SDK Property/std::string interface; ABI0 and unknown providers are unsupported.' >&2; exit 1; }; \
	  test "$(TRT_HEADER_VERSION)" = "$(TRT_VER)" && test -n "$(TRT_VER)" || { echo 'Selected TensorRT headers do not match TRT_VER.' >&2; exit 1; }; \
	  test -f "$(EIGEN_INCLUDE_DIR)/Eigen/Dense" && test -f "$(JSON_INCLUDE_DIR)/nlohmann/json.hpp" || { echo 'Set EIGEN_INCLUDE_DIR and JSON_INCLUDE_DIR to genuine dependency headers.' >&2; exit 1; }; \
	  for name in torch torch_cpu torch_cuda c10 c10_cuda; do \
	    test -f "$(TORCH_LIBRARY_DIRS)/lib$$name.so" || { echo "Missing CUDA LibTorch library: lib$$name.so" >&2; exit 1; }; \
	    case '$(TARGET_DEVICE)' in x86_64) machine='Advanced Micro Devices X86-64' ;; aarch64) machine=AArch64 ;; *) echo 'Unsupported Sparse4D target architecture.' >&2; exit 1 ;; esac; \
	    LC_ALL=C readelf -h "$(TORCH_LIBRARY_DIRS)/lib$$name.so" | grep -F "$$machine" >/dev/null || { echo "Selected lib$$name does not match the target architecture." >&2; exit 1; }; \
	  done; \
	  for name in nvinfer nvonnxparser; do \
	    LC_ALL=C readelf -h "$(TRT_LIB_DIR)/lib$$name.so" | grep -F "$$machine" >/dev/null || { echo "Selected lib$$name does not match the target architecture." >&2; exit 1; }; \
	    LC_ALL=C readelf -d "$(TRT_LIB_DIR)/lib$$name.so" | grep -F "Library soname: [lib$$name.so.$(firstword $(subst ., ,$(TRT_VER)))]" >/dev/null || { echo "Selected lib$$name does not match TensorRT ABI." >&2; exit 1; }; \
	  done

$(OBJS) $(LIB): | check_sparse4d_dependencies
validate_sparse4d_binary: $(LIB)
	@set -e; dynamic=$$(LC_ALL=C readelf -d "$(LIB)"); \
	  printf '%s\n' "$$dynamic" | grep -F 'Shared library: [libnvinfer.so.$(firstword $(subst ., ,$(TRT_VER)))]' >/dev/null; \
	  printf '%s\n' "$$dynamic" | grep -F 'Shared library: [libnvonnxparser.so.$(firstword $(subst ., ,$(TRT_VER)))]' >/dev/null; \
	  printf '%s\n' "$$dynamic" | grep -F 'Shared library: [libcudart.so.$(firstword $(subst ., ,$(CUDA_VER)))]' >/dev/null; \
	  needed=$$(printf '%s\n' "$$dynamic" | sed -n 's/.*Shared library: \[\(.*\)\].*/\1/p'); \
	  for name in $$needed; do \
	    case "$$name" in \
	      libnvinfer*.so.*|libnvonnxparser.so.*) test "$$name" = "$${name%.so.*}.so.$(firstword $(subst ., ,$(TRT_VER)))" || { echo "Sparse4D output links a mismatched TensorRT dependency: $$name" >&2; exit 1; } ;; \
	      libcudart.so.*) test "$$name" = 'libcudart.so.$(firstword $(subst ., ,$(CUDA_VER)))' || { echo "Sparse4D output links a mismatched CUDA dependency: $$name" >&2; exit 1; } ;; \
	    esac; \
	  done; \
	  case '$(TARGET_DEVICE)' in x86_64) machine='Advanced Micro Devices X86-64' ;; aarch64) machine=AArch64 ;; *) exit 1 ;; esac; \
	  LC_ALL=C readelf -h "$(LIB)" | grep -F "$$machine" >/dev/null || { echo 'Sparse4D output does not match the target architecture.' >&2; exit 1; }

install: validate_sparse4d_binary
	cp -rv $(LIB) $(LIB_INSTALL_DIR)

-include $(OBJS:.o=.d)

clean:
	rm -f $(OBJS) $(OBJS:.o=.d) $(LIB)
	rm -f libnvdsgst_sparse4d.so utils/*.o preprocess/*.o infer/*.o postprocess/*.o
