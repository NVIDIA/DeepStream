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

CXX:= g++
TARGET_DEVICE = $(shell gcc -dumpmachine | cut -f1 -d -)

NVDS_VERSION:=9.2

GST_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/gst-plugins/
LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/

# PyTorch (see configs/install_dependencies.sh). Override TORCH_PATH if needed.
TORCH_PATH?=/usr/local/lib/python3/dist-packages/torch
TORCH_CXX11_ABI?=$(shell sed -n 's/.*_GLIBCXX_USE_CXX11_ABI=\([01]\).*/\1/p' "$(TORCH_PATH)/share/cmake/Torch/TorchConfig.cmake" 2>/dev/null | head -n 1)

CFLAGS:= -fPIC -std=c++17 -Wall -Werror -Wno-deprecated-declarations \
	-DHAVE_CONFIG_H -fexceptions

CFLAGS+= -I. \
	 -I../includes \
	 -I../../../../includes \
	 -I../../../utils/nvdsinfer \
	 -I../../../utils/nvdsinferutils/include \
	 -I../../gst-nvdspreprocess \
	 -I../../gst-nvdspreprocess/include \
	 -I/usr/local/cuda-$(CUDA_VER)/include \
	 -I$(TORCH_PATH)/include \
	 -I$(TORCH_PATH)/include/torch/csrc/api/include

CFLAGS+= $(shell pkg-config --cflags eigen3)
CFLAGS+= $(if $(TORCH_CXX11_ABI),-D_GLIBCXX_USE_CXX11_ABI=$(TORCH_CXX11_ABI))

PKGS:= gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0
CFLAGS+= $(shell pkg-config --cflags $(PKGS))

SRCS:= customlib_impl.cpp \
	utils/config_parser.cpp utils/dump_utils.cpp utils/tensor_init.cpp \
	preprocess/calibration_parser.cpp preprocess/preprocessor.cpp \
	infer/engine_builder.cpp infer/tensor_rt_engine.cpp \
	postprocess/postprocessor.cpp postprocess/decoder.cpp \
	postprocess/bbox3d_utils.cpp postprocess/display_bbox3d.cpp \
	postprocess/instance_bank.cpp postprocess/sparse_box_3d_generator.cpp

OBJS:= $(SRCS:.cpp=.o)
LIB:= libnvdsgst_sparse4d.so

LIBS:= -shared -Wl,-no-undefined -Wl,--no-as-needed \
	-L/usr/local/cuda-$(CUDA_VER)/lib64/ -lcudart \
	-L$(LIB_INSTALL_DIR) -lnvdsgst_helper -lnvdsgst_meta -lnvds_meta \
	-lnvbufsurface -lnvbufsurftransform -lnvdsbufferpool \
	-L$(GST_INSTALL_DIR) -lnvdsgst_preprocess \
	-L$(TORCH_PATH)/lib -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda \
	-lyaml-cpp -lnvds_infer -lnvinfer -lnvonnxparser \
	-ldl -lpthread \
	-Wl,-rpath,$(LIB_INSTALL_DIR) \
	-Wl,-rpath,$(GST_INSTALL_DIR) \
	-Wl,-rpath,$(TORCH_PATH)/lib
LIBS+= $(shell pkg-config --libs $(PKGS))

ifeq ($(TARGET_DEVICE),aarch64)
LIBS+= -L/usr/lib/aarch64-linux-gnu/tegra
endif

all: $(LIB)

%.o: %.cpp Makefile
	$(CXX) -c -o $@ $(CFLAGS) $<

$(LIB): $(OBJS) Makefile
	$(CXX) -o $@ $(OBJS) $(LIBS)

install: $(LIB)
	cp -rv $(LIB) $(LIB_INSTALL_DIR)

clean:
	rm -rf $(OBJS) $(LIB)
