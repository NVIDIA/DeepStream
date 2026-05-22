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

LIB_INSTALL_DIR ?= /opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/
CC:= g++

# Change to your deepstream SDK includes
CFLAGS+= -I../../../../../includes \
	  -I/usr/local/cuda/include \
	  -I../../../../../includes/nvdsinferserver

CFLAGS+= -Wall -std=c++14 -shared -fPIC

LIBS+= -L/usr/local/cuda/lib64 -lcudart -lcublas

LFLAGS:= -Wl,--start-group $(LIBS) -Wl,--end-group

SRCFILES:= custom_postprocess.cpp lidar_postprocess.cpp
TARGET_LIB:= libnvds_lidar_custom_postprocess_impl.so

all: $(TARGET_LIB)

$(TARGET_LIB) : $(SRCFILES)
	$(CC) -o $@ $^ $(CFLAGS) $(LFLAGS)

install: $(TARGET_LIB)
	ldconfig

clean:
	rm -rf $(TARGET_LIB)

