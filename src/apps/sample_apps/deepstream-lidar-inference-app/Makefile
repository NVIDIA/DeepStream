################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
################################################################################

NVDS_VERSION?=9.0

LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/
APP_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/bin/

APP:= deepstream-lidar-inference-app
include ../Rules.mk

CC = g++
CFLAGS:= -fvisibility=hidden -Wall -Werror
debug: CFLAGS+= -g -O0


APP_SRCS:= $(wildcard *.cpp)
APP_INCS:= $(wildcard *.hpp)

PKGS:= gstreamer-1.0 yaml-cpp

CFLAGS+= -I../../../../includes \
         -I../../../../libs \
         -I../../../../includes/nvdsinferserver \
         -fPIC -std=c++14

CFLAGS+= $(shell pkg-config --cflags $(PKGS))

LIBS:= $(shell pkg-config --libs $(PKGS))

LIBS+=  -lnvdsgst_helper -lm \
-L/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib -Wl,-rpath-link,/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib -lnvdsgst_3d_gst \
-L/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib -Wl,-rpath-link,/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib -lnvdsgst_meta -lnvds_meta \
		-lcuda -Wl,-rpath,$(LIB_INSTALL_DIR) \
		-lgstapp-1.0 -ldl
APP_OBJS:= $(APP_SRCS:.cpp=.o)
APP_OBJS := $(addprefix $(OBJ_DIR)/,$(APP_OBJS))

SUBFOLDERS:=custom_postprocess_impl custom_preprocess_impl

all: $(BIN_DIR)/$(APP) $(SUBFOLDERS)
$(OBJ_DIR)/%.o: %.cpp $(APP_INCS) Makefile
	@mkdir -p $(OBJ_DIR)
	$(CC) -c -o $@ $(CFLAGS) $<

$(BIN_DIR)/$(APP): $(APP_OBJS) Makefile
	@mkdir -p $(BIN_DIR)
	$(CC) -o $(BIN_DIR)/$(APP) $(APP_OBJS) $(LIBS)

$(SUBFOLDERS):
	$(MAKE) -C $@ $(MAKECMDGOALS)

install: $(BIN_DIR)/$(APP) $(SUBFOLDERS)
	cp -rv custom_postprocess_impl/libnvds_lidar_custom_postprocess_impl.so $(LIB_INSTALL_DIR)
	cp -rv custom_preprocess_impl/libnvds_lidar_custom_preprocess_impl.so $(LIB_INSTALL_DIR)

clean: $(SUBFOLDERS)
	rm -rf $(OBJ_DIR) $(BIN_DIR)/$(APP)

.PHONY: all $(SUBFOLDERS)
