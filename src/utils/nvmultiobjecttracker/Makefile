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

# Builds libnvds_nvmultiobjecttracker.so, the low-level multi-object tracker
# that gst-nvtracker loads through its "ll-lib-file" property.
#
#   make CUDA_VER=13.2            build the library into $(OBJ_DIR)
#   make CUDA_VER=13.2 install    copy it to $(LIB_INSTALL_DIR)
#   make clean                    remove the build tree
#
# Only the seven NvMOT_* entry points listed in EXPORTED_SYMS are visible in
# the final .so; all other symbols are hidden via a GNU ld version script.
#
# Build prerequisites (see build/BUILD.md):
#   - CUDA toolkit (nvcc, cudart, cufft, cublas) and TensorRT 10.x development files
#   - libeigen3-dev, nlohmann-json3-dev, libyaml-cpp-dev, libmosquitto-dev, libcurl4-openssl-dev
#   - vpi4-dev, which provides the VPI headers under /opt/nvidia/vpi4/include
#   - libnvds_inferlogger, libnvds_inferutils, libnvbufsurface, libnvbufsurftransform,
#     libnvds_mqtt_proto and libnvvpi in $(LIB_INSTALL_DIR)

CUDA_VER?=
ifeq ($(CUDA_VER),)
  $(error "CUDA_VER is not set")
endif

CXX:= g++
NVCC:= /usr/local/cuda-$(CUDA_VER)/bin/nvcc

NVDS_VERSION:=9.2
NVDS_VERSION_MAJOR:=9
NVDS_VERSION_MINOR:=2
NVDS_VERSION_MICRO:=0

LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/
VPI_DIR?=/opt/nvidia/vpi4
VPI_LIB_DIR?=$(VPI_DIR)/lib/$(shell uname -m)-linux-gnu
OBJ_DIR?=$(CURDIR)/.build

ifeq ($(wildcard $(VPI_DIR)/include/vpi/algo/DCFTracker.h),)
  $(error "VPI headers not found in $(VPI_DIR)/include. Install the vpi4-dev package or set VPI_DIR")
endif

TARGET_NAME:= libnvds_nvmultiobjecttracker.so
TARGET:= $(OBJ_DIR)/$(TARGET_NAME)
VERSION_SCRIPT:= $(OBJ_DIR)/libnvds_nvmultiobjecttracker.map

EXPORTED_SYMS := \
    NvMOT_Init \
    NvMOT_Process \
    NvMOT_RetrieveMiscData \
    NvMOT_RemoveStreams \
    NvMOT_DeInit \
    NvMOT_Query \
    NvMOT_UpdateParams

MODULES:= \
	NvMultiObjectTracker \
	TrajectoryManager \
	Estimatorv2 \
	DataAssociator \
	VisualTracker \
	FrameTransform \
	utility \
	cuDCFv2 \
	ReID \
	Segmenter \
	ObjectModelProjector \
	MultiViewAssociator \
	PoseEstimator \
	ReIDService

CPP_SRCS:= $(foreach m,$(MODULES),$(wildcard src/modules/$(m)/*.cpp))
CU_SRCS:= $(foreach m,$(MODULES),$(wildcard src/modules/$(m)/*.cu))
OBJS:= $(addprefix $(OBJ_DIR)/,$(CPP_SRCS:.cpp=.o) $(CU_SRCS:.cu=.o))

INC_PATHS:= \
	include \
	../../../includes \
	../nvdsinferutils/include \
	../nvds_msgapi/mqtt_protocol_adaptor \
	../nvds_msgapi/common_src \
	$(VPI_DIR)/include \
	/usr/local/cuda-$(CUDA_VER)/include

PKGS:= eigen3 nlohmann_json glib-2.0
INC_FLAGS:= $(addprefix -I,$(INC_PATHS)) $(shell pkg-config --cflags $(PKGS))

DEFINES:= -DMODE_NVDGPU -DDEEPSTREAM_SDK \
	-DDS_VERSION=\"$(NVDS_VERSION).$(NVDS_VERSION_MICRO)\" \
	-DDS_VERSION_MAJOR=$(NVDS_VERSION_MAJOR) \
	-DDS_VERSION_MINOR=$(NVDS_VERSION_MINOR) \
	-DDS_VERSION_MICRO=$(NVDS_VERSION_MICRO)

CXXFLAGS:= -std=c++14 -O3 -fPIC -DNDEBUG -Wno-deprecated-declarations \
	$(DEFINES) $(INC_FLAGS)

# GPU architectures: SASS for each entry, plus PTX for the highest one listed
# in CU_PTX_ARCH so that newer GPUs can JIT the kernels.
CU_SASS_ARCHS:= 75 80 89 90 100 103 110 120 121
CU_PTX_ARCH:= 90
CU_GENCODES:= $(foreach sm,$(CU_SASS_ARCHS),-gencode arch=compute_$(sm),code=sm_$(sm)) \
	-gencode arch=compute_$(CU_PTX_ARCH),code=compute_$(CU_PTX_ARCH)
CUFLAGS:= -std=c++14 -DDEEPSTREAM_SDK $(CU_GENCODES) --compiler-options '-fPIC' \
	$(addprefix -I,$(INC_PATHS)) $(shell pkg-config --cflags $(PKGS))

LDFLAGS:= -shared -Wl,--no-undefined -Wl,--as-needed -Wl,--allow-shlib-undefined \
	-Wl,--version-script=$(VERSION_SCRIPT) \
	-Wl,-rpath,$(LIB_INSTALL_DIR) -Wl,-rpath,/usr/local/cuda-$(CUDA_VER)/lib64

LIBS:= \
	-L$(LIB_INSTALL_DIR) \
	-L/usr/local/cuda-$(CUDA_VER)/lib64 \
	-L/usr/local/cuda-$(CUDA_VER)/lib64/stubs \
	-L$(VPI_LIB_DIR) \
	-lnvinfer -lnvinfer_plugin -lnvonnxparser \
	-lcudart -lcuda -lcufft -lcublas \
	-lnvds_inferlogger -lnvds_inferutils \
	-lnvbufsurface -lnvbufsurftransform \
	-lnvds_mqtt_proto -l:libnvvpi.so.4 \
	-lyaml-cpp -lcurl -lmosquitto \
	-lpthread -lstdc++fs

.PHONY: all install clean

all: $(TARGET)

$(TARGET): $(OBJS) $(VERSION_SCRIPT)
	$(CXX) -o $@ $(OBJS) $(LDFLAGS) $(LIBS)

$(VERSION_SCRIPT): Makefile
	@mkdir -p $(dir $@)
	@{ echo '{'; echo '  global:'; \
	   $(foreach s,$(EXPORTED_SYMS),echo '    $(s);'; ) \
	   echo '  local:'; echo '    *;'; echo '};'; } > $@

$(OBJ_DIR)/%.o: %.cpp Makefile
	@mkdir -p $(dir $@)
	$(CXX) -c -o $@ $(CXXFLAGS) -MMD -MP $<

$(OBJ_DIR)/%.o: %.cu Makefile
	@mkdir -p $(dir $@)
	$(NVCC) -c -o $@ $(CUFLAGS) -MMD -MP $<

install: $(TARGET)
	mkdir -p $(LIB_INSTALL_DIR)
	cp -v $(TARGET) $(LIB_INSTALL_DIR)

clean:
	rm -rf $(OBJ_DIR)

-include $(OBJS:.o=.d)
