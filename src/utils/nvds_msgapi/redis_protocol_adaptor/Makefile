#######################################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

#######################################################################################################
# this Makefile is to be used to build the Redis protocol adaptor .so


CXX:=g++

PKGS:= glib-2.0 

DS_LIB := /opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/

SRCS:=  nvds_redis_client.cpp ../common_src/nvds_utils.cpp
TARGET_LIB:= libnvds_redis_proto.so

UTILS_LIB_DIR?= /opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib

CFLAGS:= -fPIC -Wall

CFLAGS+= $(shell pkg-config --cflags $(PKGS))

LIBS:= $(shell pkg-config --libs $(PKGS))
LDFLAGS:= -shared

DS_INC:= ../../../../includes
REDIS_INC:=/usr/local/include/hiredis

INC_PATHS:= -I $(DS_INC) -I $(REDIS_INC)
CFLAGS+= $(INC_PATHS)

LIBS+= -L$(DS_LIB) -lnvds_logger -lhiredis -lpthread -lglib-2.0 -lcrypto
LDFLAGS+= -Wl,-rpath=$(DS_LIB)

all: $(UTILS_LIB_DIR)/$(TARGET_LIB)

$(UTILS_LIB_DIR)/$(TARGET_LIB): $(SRCS)
	@mkdir -p $(UTILS_LIB_DIR)
	$(CXX) -o $@ $^ $(CFLAGS) $(LIBS) $(LDFLAGS)

install: $(UTILS_LIB_DIR)/$(TARGET_LIB)
	ldconfig

clean:
	rm -rf $(UTILS_LIB_DIR)/$(TARGET_LIB)