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

CC:= gcc

PKGS:= glib-2.0 gobject-2.0 gstreamer-1.0

NVDS_VERSION:=9.1

LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/

CFLAGS:= -fPIC -ansi -std=c99 -DHAVE_CONFIG_H \
	 -Wunused-function -Wunused-variable -Wall

CFLAGS+= -I../../../includes
CFLAGS+= $(shell pkg-config --cflags $(PKGS))

SRCFILES:= gstnvdsmeta.c gstnvdsmeta_latency_measurement.c

TARGET_LIB:= libnvdsgst_meta.so

LIBS:= -shared \
	-L$(LIB_INSTALL_DIR) -lnvds_meta \
	-Wl,-rpath,$(LIB_INSTALL_DIR)
LIBS+= $(shell pkg-config --libs $(PKGS))

all: $(TARGET_LIB)

$(TARGET_LIB) : $(SRCFILES)
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)

install: $(TARGET_LIB)
	cp -rv $(TARGET_LIB) $(LIB_INSTALL_DIR)

clean:
	rm -rf $(TARGET_LIB)
