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

APP:= deepstream-nvocdr-app

CXX=g++

LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream/lib/
APP_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream/bin/

SRCS:= deepstream_nvocdr_app.cpp \
       ../common/ds_yml_parse.cpp

OBJS:= $(SRCS:.cpp=.o)

PKGS:= gstreamer-1.0

CFLAGS:= $(shell pkg-config --cflags $(PKGS))
CFLAGS+= -I../../../../../../includes/ \
         -I/usr/local/cuda/include \
         -I../common

LIBS:= $(shell pkg-config --libs $(PKGS))
LIBS+= -L$(LIB_INSTALL_DIR) -lnvdsgst_meta -lnvds_meta \
       -lm -lstdc++ -lnvds_yml_parser -lgstrtspserver-1.0\
       -L/usr/local/cuda/lib64/ -lcudart -lcuda -lyaml-cpp \
       -Wl,-rpath,$(LIB_INSTALL_DIR)

all: $(APP)

%.o: %.cpp $(INCS) Makefile
	$(CXX) -c -o $@ $(CFLAGS) $<

$(APP): $(OBJS) Makefile
	make -C nvocdr
	make -C nvocdr/deepstream
	$(CXX) -o $(APP) $(OBJS) $(LIBS)

install: $(APP)
	cp -rv $(APP) $(APP_INSTALL_DIR)
	cp nvocdr/libnvocdr.so $(LIB_INSTALL_DIR)
	cp nvocdr/deepstream/libnvocdr_impl.so $(LIB_INSTALL_DIR)

clean:
	rm -rf *.o $(OBJS) $(APP)
