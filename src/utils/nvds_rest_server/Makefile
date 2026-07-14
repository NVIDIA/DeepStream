# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

CUDA_VER?=
ifeq ($(CUDA_VER),)
  $(error "CUDA_VER is not set")
endif

TARGET_DEVICE = $(shell g++ -dumpmachine | cut -f1 -d -)

CXX:= g++
SRCS:= nvds_dec_parse.cpp nvds_infer_parse.cpp \
       nvds_rest_server.cpp nvds_roi_parse.cpp nvds_stream_parse.cpp \
       nvds_mux_parse.cpp nvds_inferserver_parse.cpp nvds_conv_parse.cpp nvds_enc_parse.cpp \
       nvds_app_instance_parse.cpp nvds_osd_parse.cpp nvds_analytics_parse.cpp nvds_nvtracker_parse.cpp \
	   nvds_rest_metrics.cpp prometheus_client.cpp opentelemetry_client.cpp

INCS:= $(wildcard *.h)
LIB:=libnvds_rest_server.so
LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/
NVDS_VERSION:=9.1

CFLAGS+= -fPIC -DDS_VERSION=\"9.1.0\" \
 	 -I../../../includes \
 	 -I/usr/local/cuda-$(CUDA_VER)/include \
	 -I/usr/include/jsoncpp

LIBS := -shared -Wl,-no-undefined \
 	-L/usr/local/cuda-$(CUDA_VER)/lib64/ -lcudart

LIBS+= -L$(LIB_INSTALL_DIR) -lnvdsgst_customhelper -ldl\
        -Wl,-rpath,$(LIB_INSTALL_DIR) -lcivetweb \
		-lprometheus-cpp-core -lnvds_stats \
		-lopentelemetry_metrics -lopentelemetry_exporter_otlp_http_metric -lopentelemetry_common \
    -lopentelemetry_exporter_otlp_http_client -lopentelemetry_resources -lopentelemetry_exporter_ostream_metrics

ifeq ($(TARGET_DEVICE),aarch64)
   LIBS+= -L/usr/lib/aarch64-linux-gnu/tegra/ -ljsoncpp
else
   LIBS+= -L$(LIB_INSTALL_DIR) -ljsoncpp
endif

OBJS:= $(SRCS:.cpp=.o)

PKGS:= gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0
CFLAGS+= $(shell pkg-config --cflags $(PKGS))
LIBS+= $(shell pkg-config --libs $(PKGS))

all: $(LIB)

%.o: %.cpp $(INCS) Makefile
	@echo $(CFLAGS)
	$(CXX) -c -o $@ $(CFLAGS) $<

$(LIB): $(OBJS) $(DEP) Makefile
	$(CXX) -o $@ $(OBJS) $(LIBS)

install: $(LIB)
	cp -rv $(LIB) $(LIB_INSTALL_DIR)

clean:
	rm -rf $(OBJS) $(LIB)
