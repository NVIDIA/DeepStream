################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
################################################################################

CC:= g++

CFLAGS+=  -shared -fPIC  -DHAVE_CONFIG_H -std=c++11 -Wall

CFLAGS+= -I../../includes 

LIBS:= -lm
LFLAGS:= -Wl,--start-group $(LIBS) -Wl,--end-group

SRCFILES:= nvds_analytics.cpp

TARGET_LIB:= libnvds_dsanalytics.so

UTILS_LIB_DIR?= /opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib

all: $(UTILS_LIB_DIR)/$(TARGET_LIB)

$(UTILS_LIB_DIR)/$(TARGET_LIB): $(SRCFILES)
	@mkdir -p $(UTILS_LIB_DIR)
	$(CC) -o $@ $^ $(CFLAGS) $(LFLAGS)

install: $(UTILS_LIB_DIR)/$(TARGET_LIB)
	ldconfig

clean:
	rm -rf $(UTILS_LIB_DIR)/$(TARGET_LIB)

