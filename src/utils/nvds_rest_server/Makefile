################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


CXX:= g++
SRCS:= nvds_dec_parse.cpp nvds_infer_parse.cpp \
       nvds_rest_server.cpp nvds_roi_parse.cpp nvds_stream_parse.cpp \
       nvds_mux_parse.cpp nvds_inferserver_parse.cpp nvds_conv_parse.cpp nvds_enc_parse.cpp \
       nvds_app_instance_parse.cpp nvds_osd_parse.cpp nvds_analytics_parse.cpp nvds_nvtracker_parse.cpp \
	   nvds_rest_metrics.cpp prometheus_client.cpp opentelemetry_client.cpp

INCS:= $(wildcard *.h)
LIB:=libnvds_rest_server.so
LIB_DIR = /opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib
OBJ_DIR := .build
include ../../Rules.mk
LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/
NVDS_VERSION?=9.0

CFLAGS+= -fPIC -DDS_VERSION=\"9.0.0\" \
 	 -I../../../includes \
 	 -I/usr/local/cuda-$(CUDA_VER)/include \
	 -I/usr/include/jsoncpp

LIBS := -shared \
 	-L/usr/local/cuda-$(CUDA_VER)/lib64/ -lcudart

LIBS+= -L/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib -L/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib -Wl,-rpath-link,/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib -lnvdsgst_customhelper -ldl\
        -Wl,-rpath,$(LIB_INSTALL_DIR) -lcivetweb-cpp -lcivetweb \
		-lprometheus-cpp-core -lnvds_stats \
		-lopentelemetry_metrics -lopentelemetry_exporter_otlp_http_metric -lopentelemetry_common \
    -lopentelemetry_exporter_otlp_http_client -lopentelemetry_resources -lopentelemetry_exporter_ostream_metrics

LIBS+= -L/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib -ljsoncpp

OBJS:= $(SRCS:.cpp=.o)
OBJS := $(addprefix $(OBJ_DIR)/,$(OBJS))

PKGS:= gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0
CFLAGS+= $(shell pkg-config --cflags $(PKGS))
LIBS+= $(shell pkg-config --libs $(PKGS))

all: $(LIB_DIR)/$(LIB)

$(OBJ_DIR)/%.o: %.cpp $(INCS) Makefile
	@mkdir -p $(OBJ_DIR)
	@echo $(CFLAGS)
	$(CXX) -c -o $@ $(CFLAGS) $<

$(LIB_DIR)/$(LIB): $(OBJS) $(DEP) Makefile
	@mkdir -p $(LIB_DIR)
	$(CXX) -o $@ $(OBJS) $(LIBS)

install: $(LIB_DIR)/$(LIB)
	ldconfig

clean:
	rm -rf $(OBJ_DIR) $(LIB_DIR)/$(LIB)
