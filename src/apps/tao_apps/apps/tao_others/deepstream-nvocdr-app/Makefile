################################################################################
# Copyright (c) 2023-2025, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
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
