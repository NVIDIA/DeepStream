################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2018-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

CFLAGS:= -Wall -std=c++11

CFLAGS+= -shared -fPIC

CFLAGS+= -I../../../includes \
         -I /usr/local/cuda-$(CUDA_VER)/include

LIBS:= -lnvinfer
LFLAGS:= -Wl,--start-group $(LIBS) -Wl,--end-group

SRCFILES:= nvdsinfer_custombboxparser.cpp \
	nvdsinfer_customclassifierparser.cpp \
	nvdsinfer_customsegmentationparser.cpp

TARGET_LIB:= libnvds_infercustomparser.so

UTILS_LIB_DIR?= /opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib

all: $(UTILS_LIB_DIR)/$(TARGET_LIB)

$(UTILS_LIB_DIR)/$(TARGET_LIB): $(SRCFILES)
	@mkdir -p $(UTILS_LIB_DIR)
	$(CC) -o $@ $^ $(CFLAGS) $(LFLAGS)

install: $(UTILS_LIB_DIR)/$(TARGET_LIB)
	ldconfig

clean:
	rm -rf $(UTILS_LIB_DIR)/$(TARGET_LIB)
