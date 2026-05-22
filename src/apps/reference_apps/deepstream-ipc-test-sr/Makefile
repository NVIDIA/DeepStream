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

APP:= deepstream-ipc-test-app

LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream/lib/
APP_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream/bin/

SRCS:= $(wildcard *.c)

PKGS:= gstreamer-1.0

OBJS:= $(SRCS:.c=.o)

CFLAGS+= -I/opt/nvidia/deepstream/deepstream/sources/includes \
        -I /usr/local/cuda-$(CUDA_VER)/include

CFLAGS+= $(shell pkg-config --cflags $(PKGS))

LIBS:= $(shell pkg-config --libs $(PKGS))

LIBS+= -L/usr/local/cuda-$(CUDA_VER)/lib64/ -lcudart -lnvdsgst_helper -lm \
		-L$(LIB_INSTALL_DIR) -lnvdsgst_meta -lnvds_meta -lnvds_yml_parser -lgstapp-1.0 \
		-lcuda -Wl,-rpath,$(LIB_INSTALL_DIR)

all: $(APP)

%.o: %.c $(INCS)
	$(CC) -c -o $@ $(CFLAGS) $<

$(APP): $(OBJS)
	$(CC) -o $(APP) $(OBJS) $(LIBS)
	make -C video_template_impl
	make -C latency_serialization
	@echo "Build complete."

install: $(APP)
	cp -rv $(APP) $(APP_INSTALL_DIR)
	make -C video_template_impl install
	make -C latency_serialization install
	@echo "Install complete."

clean:
	make -C video_template_impl clean
	make -C latency_serialization clean
	rm -rf $(OBJS) $(APP)
	@echo "Clean complete."

.PHONY: all
