##################################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
##################################################################################################

# this Makefile is to be used to build the msgbroker library nvmsgbroker.so

CXX:=g++

PKGS:= glib-2.0

NVDS_VERSION?=9.0
DS_LIB:=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib

SRCS:=  nvmsgbroker.cpp
TARGET_LIB:= libnvds_msgbroker.so

UTILS_LIB_DIR?= /opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib

CFLAGS:= -fPIC -g -Wall -std=c++11
CFLAGS+= $(shell pkg-config --cflags $(PKGS))

LIBS:= $(shell pkg-config --libs $(PKGS))
LDFLAGS:= -shared

DS_INC:= ../../../includes

INC_PATHS:= -I$(DS_INC)
CFLAGS+= $(INC_PATHS)

LIBS+= -L$(DS_LIB) -lnvds_logger -lpthread -lglib-2.0

LDFLAGS+= -Wl,-rpath=$(DS_LIB)

all: $(UTILS_LIB_DIR)/$(TARGET_LIB)

$(UTILS_LIB_DIR)/$(TARGET_LIB): $(SRCS)
	@mkdir -p $(UTILS_LIB_DIR)
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS) $(LDFLAGS)

install: $(UTILS_LIB_DIR)/$(TARGET_LIB)
	ldconfig

clean:
	rm -rf $(UTILS_LIB_DIR)/$(TARGET_LIB)