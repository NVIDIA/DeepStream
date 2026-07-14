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

_RULES_MK_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
_REPO_ROOT     := $(realpath $(_RULES_MK_DIR)/../../..)

NVDS_VERSION  ?= 9.1

BIN_DIR ?= /opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/bin
OBJ_DIR ?= $(CURDIR)/.build

# SOURCE_DIR points to repo root — used by apps that reference $(SOURCE_DIR)/includes etc.
SOURCE_DIR ?= $(_REPO_ROOT)
