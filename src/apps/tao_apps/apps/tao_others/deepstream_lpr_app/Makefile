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

APP:= deepstream-lpr-app

CXX=g++


LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream/lib/
APP_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream/bin/

SRCS:= deepstream_lpr_app.cpp \
       deepstream_nvdsanalytics_meta.cpp \
       ../common/ds_yml_parse.cpp

OBJS:= $(SRCS:.cpp=.o)

PKGS:= gstreamer-1.0
CFLAGS:=$(shell pkg-config --cflags $(PKGS))
CFLAGS+=-Wall -Werror -Wno-unused-function \
        -I/opt/nvidia/deepstream/deepstream/sources/includes \
        -I/usr/local/cuda/include \
        -I../common/

LIBS:= $(shell pkg-config --libs $(PKGS))
LIBS+= -L$(LIB_INSTALL_DIR) -lnvdsgst_meta -lnvds_meta -lm -lstdc++\
       -lnvds_yml_parser -lyaml-cpp -lgstrtspserver-1.0 -Wl,-rpath,$(LIB_INSTALL_DIR) \
       -L/usr/local/cuda/lib64/ -lcudart

all: $(APP)

%.o: %.cpp
	$(CXX) -c -o $@ $(CFLAGS) $<

$(APP): $(OBJS)
	@$(CC) -o $(APP) $(OBJS) $(LIBS)
	@make -C nvinfer_custom_lpr_parser

install: $(APP)
	cp -rv $(APP) $(APP_INSTALL_DIR)
	@make install -C nvinfer_custom_lpr_parser

clean:
	rm -rf $(OBJS) $(APP)
	@make -C nvinfer_custom_lpr_parser clean