################################################################################
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
################################################################################

CUDA_VER?=
ifeq ($(CUDA_VER),)
  $(error "CUDA_VER is not set")
endif

CXX=g++ -std=c++14

APP:= deepstream-pose-classification-app


CUDA_HOME:= /usr/local/cuda-$(CUDA_VER)
DEEPSTREAM_HOME:= /opt/nvidia/deepstream/deepstream

LIB_INSTALL_DIR?=$(DEEPSTREAM_HOME)/lib/
APP_INSTALL_DIR?=$(DEEPSTREAM_HOME)/bin/

SRCS:= deepstream_pose_classification_app.cpp    ../common/ds_yml_parse.cpp

INCS:= $(wildcard *.h)

PKGS:= gstreamer-1.0 gstreamer-video-1.0 x11 json-glib-1.0

OBJS:= $(patsubst %.c,%.o, $(patsubst %.cpp,%.o, $(SRCS)))

CFLAGS+= \
	-I$(CUDA_HOME)/include \
	-I$(DEEPSTREAM_HOME)/sources/includes \
	-I../../../../common/includes \
	-I../../../../sample_apps/deepstream-app \
	-I../common

LIBS+= \
	-L$(CUDA_HOME)/lib64 -lcudart -lcuda -lnvds_yml_parser -lyaml-cpp\
	-L$(LIB_INSTALL_DIR) -lnvdsgst_meta -lnvds_meta -lnvds_utils -lgstapp-1.0 -lgstrtspserver-1.0 \
	-lpthread -lm -ldl -Wl,-rpath,$(LIB_INSTALL_DIR)

CFLAGS+= $(shell pkg-config --cflags $(PKGS))

LIBS+= $(shell pkg-config --libs $(PKGS))

all: $(APP)
	make -C nvdsinfer_custom_impl_BodyPose3DNet
	make -C nvdspreprocess_lib
	make -C infer_pose_classification_parser
	make -C postprocesslib_impl

debug: CXXFLAGS += -DDEBUG -ggdb
debug: CCFLAGS += -DDEBUG -ggdb
debug: CFLAGS += -DDEBUG -ggdb
debug: $(APP)

%.o: %.c $(INCS) Makefile
	$(CC) -c -o $@ $(CFLAGS) $<

%.o: %.cpp $(INCS) Makefile
	$(CXX) -c -o $@ $(CFLAGS) $<

$(APP): $(OBJS) Makefile
	$(CXX) -o $(APP) $(OBJS) $(LIBS)

install: $(APP)
	cp -rv $(APP) $(APP_INSTALL_DIR)

clean:
	rm -rf $(OBJS) $(APP)
	make clean -C nvdsinfer_custom_impl_BodyPose3DNet
	make clean -C nvdspreprocess_lib
	make clean -C infer_pose_classification_parser
	make clean -C postprocesslib_impl


