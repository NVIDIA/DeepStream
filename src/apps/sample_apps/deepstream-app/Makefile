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

APP:= deepstream-app
include ../Rules.mk


NVDS_VERSION?=9.0

LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/
APP_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/bin/


SRCS:= $(wildcard *.c) $(wildcard *.cpp)
SRCS+= $(wildcard ../../common/src/*.c)
SRCS+= $(wildcard ../../common/src/deepstream-yaml/*.cpp)
SRCS+= ../../../../src/utils/nvds_msgapi/common_src/nvds_utils.cpp

INCS:= $(wildcard *.h)

PKGS:= gstreamer-1.0 gstreamer-video-1.0 x11 json-glib-1.0

VPATH := $(sort $(dir $(SRCS)))
OBJS:= $(SRCS:.c=.o)
OBJS:= $(OBJS:.cpp=.o)
OBJS := $(addprefix $(OBJ_DIR)/,$(notdir $(OBJS)))

CFLAGS+= -I./ -I../../common/includes \
		 -I../../../../includes -DDS_VERSION_MINOR=1 -DDS_VERSION_MAJOR=5 \
		 -I../../../../src/utils/nvds_msgapi/inc \
		 -I /usr/local/cuda-$(CUDA_VER)/include

LIBS:= -L/usr/local/cuda-$(CUDA_VER)/lib64/ -lcudart

LIBS+= -L/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib -Wl,-rpath-link,/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib -lnvdsgst_meta -lnvds_meta -lnvdsgst_helper -lnvdsgst_customhelper -lnvds_logger -lcrypto \
	  -lnvdsgst_smartrecord -lnvds_utils -lnvds_msgbroker -lm -lyaml-cpp -lnvbufsurface\
    -lcuda -lgstrtspserver-1.0 -ldl -Wl,-rpath,$(LIB_INSTALL_DIR)

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
