################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
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

TARGET_LIB:= libnvds_3d_infer_postprocess_lidar_detection.so
TARGET_LIB_SRCS:= ds3d_infer_postprocess_lidar_detection.cpp
TARGET_LIB_INCS:= $(wildcard ./*.h)
TARGET_LIB_OBJS:= $(TARGET_LIB_SRCS:.cpp=.o)

LFLAGS:= -shared -Wl,--start-group $(LIBS) -Wl,--end-group


all: $(TARGET_LIB)

$(TARGET_LIB): $(TARGET_LIB_OBJS) Makefile
	$(CC) -o $(TARGET_LIB) $(TARGET_LIB_OBJS) $(LFLAGS)


%.o: %.cpp $(TARGET_LIB_INCS) Makefile
	$(CC) -c -o $@ $(CFLAGS) $<

install: $(TARGET_LIB)
	ldconfig

clean:
	rm -rf $(TARGET_LIB_OBJS) $(TARGET_LIB)

