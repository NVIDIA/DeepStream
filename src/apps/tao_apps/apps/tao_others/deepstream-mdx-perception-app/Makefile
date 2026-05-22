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

APP:= deepstream-mdx-perception-app

CXX=g++


LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream/lib/
APP_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream/bin/

SRCS:= deepstream_mdx_perception_app.c

SRCS+= ../../../../../apps/sample_apps/deepstream-app/deepstream_app.c
SRCS+= ../../../../../apps/sample_apps/deepstream-app/deepstream_app_config_parser.c
SRCS+= ../../../../../apps/sample_apps/deepstream-app/deepstream_app_config_parser_yaml.cpp
SRCS+= $(wildcard ../../../../../apps/common/src/*.c)
SRCS+= $(wildcard ../../../../../apps/common/src/deepstream-yaml/*.cpp)
SRCS+= $(wildcard ../../../../../utils/nvds_msgapi/common_src/*.cpp)
INCS:= $(wildcard *.h)

PKGS:= gstreamer-1.0 gstreamer-video-1.0 x11 json-glib-1.0

OBJS:= $(SRCS:.c=.o)
OBJS:= $(OBJS:.cpp=.o)

CFLAGS+= -I../../../../../apps/apps-common/includes \
         -I../../../../../apps/sample_apps/deepstream-app/ \
         -I../../../../../../includes/ \
         -I../../../../common/includes/ \
         -I/usr/local/cuda/include
#		 -DDS_VERSION_MINOR=0 -DDS_VERSION_MAJOR=5 \

LIBS:= -L/usr/local/cuda/lib64/ -lcudart
LIBS+= -L$(LIB_INSTALL_DIR) -lnvdsgst_meta -lnvds_meta -lnvdsgst_helper -lnvdsgst_customhelper \
	   -lnvdsgst_smartrecord -lnvds_utils -lnvds_msgbroker -lm -lnvds_logger -lcrypto \
       -lyaml-cpp -lcuda -lgstrtspserver-1.0 -ldl -Wl,-rpath,$(LIB_INSTALL_DIR)

CFLAGS+= $(shell pkg-config --cflags $(PKGS))

LIBS+= $(shell pkg-config --libs $(PKGS))

all: $(APP)

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

