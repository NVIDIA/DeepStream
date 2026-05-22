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

TARGET_DEVICE = $(shell gcc -dumpmachine | cut -f1 -d -)
CC:= g++
NVCC:=/usr/local/cuda/bin/nvcc

CFLAGS:= -Wall -std=c++11 -shared -fPIC -Wno-error=deprecated-declarations
CFLAGS+= -I/opt/nvidia/deepstream/deepstream/sources/includes/ -I/usr/local/cuda/include

CUFLAGS:= -std=c++17 -shared 
CUFLAGS+= -I/opt/nvidia/deepstream/deepstream/sources/includes/ -I/usr/local/cuda/include
LIBS:= -lnvinfer_plugin -lnvinfer  -L/usr/local/cuda/lib64 -lcudart -lcublas -lstdc++fs
LFLAGS:= -shared -Wl,--start-group $(LIBS) -Wl,--end-group

INCS:= $(wildcard *.h)
SRCFILES:= nvdsparsebbox_Yolo.cpp\
			nvdsparsebbox_Yolo_cuda.cu

TARGET_LIB:= libnvdsinfer_custom_impl_Yolo.so

# Gencode arguments
SMS ?= 75 80 86 89 90 100 120
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))
ifeq ($(TARGET_DEVICE),aarch64)
	GENCODE_FLAGS := 
else
	GENCODE_FLAGS := $(GENCODE_FLAGS)
endif


TARGET_OBJS:= $(SRCFILES:.cpp=.o)
TARGET_OBJS:= $(TARGET_OBJS:.cu=.o)

all: $(TARGET_LIB)

%.o: %.cpp $(INCS) Makefile
	$(CC) -c -o $@ $(CFLAGS) $<

%.o: %.cu $(INCS) Makefile
	$(NVCC) -c -o $@ $(GENCODE_FLAGS) --compiler-options '-fPIC' $(CUFLAGS)  $<

$(TARGET_LIB) : $(TARGET_OBJS)
	$(CC) -o $@  $(TARGET_OBJS) $(LFLAGS)

clean:
	rm -rf $(TARGET_LIB) *.o
