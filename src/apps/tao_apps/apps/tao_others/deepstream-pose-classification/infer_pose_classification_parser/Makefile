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

CC:= g++

LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream/lib/

CFLAGS:= -Wall -Werror -std=c++11 -shared -fPIC -Wno-error=deprecated-declarations

CFLAGS+= -I/opt/nvidia/deepstream/deepstream/sources/includes

LIBS:= -lnvinfer
LFLAGS:= -Wl,--start-group $(LIBS) -Wl,--end-group

SRCFILES:= infer_pose_classification_parser.cpp
TARGET_LIB:= libnvdsinfer_pose_classfication_parser.so

all: $(TARGET_LIB)

$(TARGET_LIB) : $(SRCFILES)
	$(CC) -o $@ $^ $(CFLAGS) $(LFLAGS)

install: $(TARGET_LIB)
	cp -rv $(TARGET_LIB) $(LIB_INSTALL_DIR)

clean:
	rm -rf $(TARGET_LIB)
