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
#
#######################################################################################################
# this Makefile is to be used to build the Azure device client protocol adaptor .so

CXX:=g++

PKGS:= glib-2.0 

DS_LIB:=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib

SRCS:=  azure_device_client.cpp ../../common_src/nvds_utils.cpp
TARGET_LIB:= libnvds_azure_proto.so

UTILS_LIB_DIR?= /opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib

CFLAGS:= -fPIC -Wall

CFLAGS+= $(shell pkg-config --cflags $(PKGS))

LIBS:= $(shell pkg-config --libs $(PKGS))
LDFLAGS:= -shared

DS_INC:= ../../../../../includes
AZURE_INC:=/usr/local/include/azureiot \
		   /usr/local/include/azure_c_shared_utility \
		   /usr/local/include/azure_macro_utils \
		   /usr/local/include/umock_c

INC_PATHS:= -I $(DS_INC) $(foreach d, $(AZURE_INC), -I$d)
CFLAGS+= $(INC_PATHS)

LIBS+= -L/$(DS_LIB) -lnvds_logger -liothub_client -lpthread -lm -lssl -lcurl -lcrypto -lglib-2.0
LDFLAGS+= -Wl,-rpath=$(DS_LIB)

all: $(UTILS_LIB_DIR)/$(TARGET_LIB)

$(UTILS_LIB_DIR)/$(TARGET_LIB): $(SRCS)
	@mkdir -p $(UTILS_LIB_DIR)
	$(CXX) -o $@ $^ $(CFLAGS) $(LIBS) $(LDFLAGS)

install: $(UTILS_LIB_DIR)/$(TARGET_LIB)
	ldconfig

clean:
	rm -rf $(UTILS_LIB_DIR)/$(TARGET_LIB)