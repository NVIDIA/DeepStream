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

CC:= g++
NVCC:=/usr/local/cuda/bin/nvcc


LIB_INSTALL_DIR?= /opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/

# Change to your deepstream SDK includes
CFLAGS+= -I../../../../../includes \
         -I/usr/local/cuda/include \
         -I../../../../../includes/nvdsinferserver \
         -I../../../../../includes/ds3d/common/helper

CFLAGS+= -Wall -std=c++14 -shared -fPIC

CUFLAGS:= -I../../includes -I/usr/local/cuda/include

LIBS+= -L/usr/local/cuda/lib64 -lcudart -lcublas -lyaml-cpp

TARGET_LIB:= libnvds_3d_v2x_infer_custom_postprocess.so
TARGET_LIB_SRCS:= nvinferserver_custom_postprocess.cpp \
                  decoder.cu

TARGET_LIB_INCS:= $(wildcard ./*.h ./*.hpp)
TARGET_LIB_OBJS:= $(TARGET_LIB_SRCS:.cpp=.o)
TARGET_LIB_OBJS:= $(TARGET_LIB_OBJS:.cu=.o)

LDFLAGS:= -shared -Wl,--no-undefined -Wl,--start-group $(LIBS) -Wl,--end-group

all: $(TARGET_LIB)

$(TARGET_LIB): $(TARGET_LIB_OBJS) Makefile
	$(CC) -o $(TARGET_LIB) $(TARGET_LIB_OBJS) $(LDFLAGS)

%.o: %.cpp $(TARGET_LIB_INCS) Makefile
	$(CC) -c -o $@ $(CFLAGS) $<

%.o: %.cu $(INCS) Makefile
	$(NVCC) -c -o $@ --compiler-options '-fPIC -I../../../../../includes/ds3d/common/helper' $(CUFLAGS) $<

install: $(TARGET_LIB)
	ldconfig

clean:
	rm -rf $(TARGET_LIB_OBJS) $(TARGET_LIB)

