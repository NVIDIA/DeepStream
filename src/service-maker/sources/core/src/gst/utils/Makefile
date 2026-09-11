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

CC:= gcc

SRCS:= plugin_stub.c custom_factory.c

INCS:= $(wildcard *.h)
LIB:=libnvds_service_maker_utils.a
LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/
NVDS_VERSION:=9.1

CFLAGS+= -fPIC -DHAVE_CONFIG_H -DHAVE_X11 -Wall -Wno-deprecated-declarations \
	 -I../../../../../includes \
	 -I/usr/local/cuda-$(CUDA_VER)/include

OBJS:= $(SRCS:.c=.o)

PKGS:= gstreamer-1.0 glib-2.0
CFLAGS+= $(shell pkg-config --cflags $(PKGS))

all: $(LIB)

%.o: %.c $(INCS) Makefile
	$(CC) -c -o $@ $(CFLAGS) $<

$(LIB): $(OBJS) Makefile
	$(AR) rcs $@ $(OBJS)

install: $(LIB)
	cp -rv $(LIB) $(LIB_INSTALL_DIR)

clean:
	rm -rf $(OBJS) $(LIB)
