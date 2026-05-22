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

CC:= gcc

SRCS:= latency_serialization.c
LIB:=liblatency_serialization.so

CFLAGS+= -fPIC

LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream/lib/

LIBS += -shared -Wl,-no-undefined \
	-L$(LIB_INSTALL_DIR) \
	-Wl,-rpath,$(LIB_INSTALL_DIR)

OBJS:= $(SRCS:.c=.o)
PKGS:= gstreamer-1.0

CFLAGS+=$(shell pkg-config --cflags $(PKGS))
LIBS+=$(shell pkg-config --libs $(PKGS))

all: $(LIB)

%.o: %.c $(INCS)
	@echo $(CFLAGS)
	$(CC) -c -o $@ $(CFLAGS) $<

$(LIB): $(OBJS)
	@echo $(CFLAGS)
	$(CC) -o $@ $(OBJS) $(LIBS)

install: $(LIB)
	cp -rv $(LIB) $(LIB_INSTALL_DIR)

clean:
	rm -rf $(OBJS) $(LIB)
