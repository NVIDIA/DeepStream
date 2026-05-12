################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
################################################################################

CUDA_VER?=
ifeq ($(CUDA_VER),)
  $(error "CUDA_VER is not set")
endif

CXX=g++

APP:= deepstream-transfer-learning-app
include ../Rules.mk


NVDS_VERSION?=9.0

LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/
APP_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/bin/



SRCS:= deepstream_transfer_learning_app_main.cpp image_meta_consumer.cpp image_meta_producer.cpp capture_time_rules.cpp
SRCS+= ../deepstream-app/deepstream_app.c ../deepstream-app/deepstream_app_config_parser.c \
       ../../../../src/utils/nvds_msgapi/common_src/nvds_utils.cpp
SRCS+= $(wildcard ../../common/src/*.c)

INCS:= $(wildcard *.h)

PKGS:= gstreamer-1.0 gstreamer-video-1.0 x11 json-glib-1.0

VPATH := $(sort $(dir $(SRCS)))
OBJS:= $(patsubst %.c,%.o, $(patsubst %.cpp,%.o, $(SRCS)))
OBJS := $(addprefix $(OBJ_DIR)/,$(notdir $(OBJS)))

CFLAGS+= -I../../common/includes \
		 -I../../../../includes \
		 -I../deepstream-app/ -DDS_VERSION_MINOR=0 -DDS_VERSION_MAJOR=5 \
		 -I /usr/local/cuda-$(CUDA_VER)/include

LIBS:= -L/usr/local/cuda-$(CUDA_VER)/lib64/ -lcudart

LIBS+= -L/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib -Wl,-rpath-link,/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib -lnvdsgst_meta -lnvds_meta -lnvds_msgbroker -lnvds_batch_jpegenc -lnvdsgst_helper -lnvdsgst_customhelper -lnvdsgst_smartrecord -lnvds_utils -lm \
       -lcuda -lgstrtspserver-1.0 -lpthread -ldl -Wl,-rpath,$(LIB_INSTALL_DIR) \
       -lnvds_logger -lcrypto -lnvbufsurface

CFLAGS+= $(shell pkg-config --cflags $(PKGS))

LIBS+= $(shell pkg-config --libs $(PKGS))

all: $(BIN_DIR)/$(APP)

$(OBJ_DIR)/%.o: %.c $(INCS) Makefile
	@mkdir -p $(OBJ_DIR)
	$(CC) -c -o $@ $(CFLAGS) $<

$(OBJ_DIR)/%.o: %.cpp $(INCS) Makefile
	@mkdir -p $(OBJ_DIR)
	$(CXX) -c -o $@ $(CFLAGS) $<

$(BIN_DIR)/$(APP): $(OBJS) Makefile
	@mkdir -p $(BIN_DIR)
	$(CXX) -o $(BIN_DIR)/$(APP) $(OBJS) $(LIBS)

install: $(BIN_DIR)/$(APP)

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)/$(APP)


