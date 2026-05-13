#######################################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
#######################################################################################################
# this Makefile is to be used to build the Azure device client protocol adaptor .so

CXX:=g++

PKGS:= glib-2.0 

DS_LIB:=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib

SRCS:=  azure_module_client.cpp
TARGET_LIB:= libnvds_azure_edge_proto.so

UTILS_LIB_DIR?= /opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib

CFLAGS:= -fPIC -Wall

CFLAGS+= $(shell pkg-config --cflags $(PKGS))

CFLAGS+= -DUSE_EDGE_MODULES -DUSE_PROV_MODULE

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