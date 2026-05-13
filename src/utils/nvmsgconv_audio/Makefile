###############################################################################
# SPDX-FileCopyrightText: Copyright (c) 2020-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
###############################################################################

CC:= g++

PKGS:= glib-2.0 gobject-2.0 json-glib-1.0 uuid

NVDS_VERSION?=9.0

LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/

CFLAGS:= -Wall -std=c++11 -shared -fPIC

CFLAGS+= -I../../../includes

CFLAGS+= $(shell pkg-config --cflags $(PKGS))
LIBS:= $(shell pkg-config --libs $(PKGS))

SRCFILES:= nvmsgconv.cpp
TARGET_LIB:= libnvds_msgconv_audio.so

UTILS_LIB_DIR?= /opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib

all: $(UTILS_LIB_DIR)/$(TARGET_LIB)

$(UTILS_LIB_DIR)/$(TARGET_LIB): $(SRCFILES)
	@mkdir -p $(UTILS_LIB_DIR)
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)

install: $(UTILS_LIB_DIR)/$(TARGET_LIB)
	ldconfig

clean:
	rm -rf $(UTILS_LIB_DIR)/$(TARGET_LIB)
