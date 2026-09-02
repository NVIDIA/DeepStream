# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Build the deployed-engine image-decode eval app (ds_image_eval).
# Mirrors the DeepStream reference-app Makefile layout used by the import
# skill's custom parser, so it builds in the same DeepStream container.
#
#   make            # build ./ds_image_eval
#   make clean

APP := ds_image_eval

DS_ROOT ?= /opt/nvidia/deepstream/deepstream
NVDS_INC := $(DS_ROOT)/sources/includes
NVDS_LIB := $(DS_ROOT)/lib

CC := gcc

PKGS := gstreamer-1.0 glib-2.0
CFLAGS := -Wall -O2 $(shell pkg-config --cflags $(PKGS)) -I$(NVDS_INC)
LDFLAGS := $(shell pkg-config --libs $(PKGS)) \
           -L$(NVDS_LIB) -lnvdsgst_meta -lnvds_meta \
           -Wl,-rpath,$(NVDS_LIB)

SRCS := ds_image_eval.c

$(APP): $(SRCS)
	$(CC) $(CFLAGS) -o $(APP) $(SRCS) $(LDFLAGS)

clean:
	rm -f $(APP)

.PHONY: clean
