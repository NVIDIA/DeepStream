# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# Absolute path to this Rules.mk, regardless of where make is invoked from
_RULES_MK_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
_REPO_ROOT     := $(realpath $(_RULES_MK_DIR)/..)

NVDS_VERSION  ?= 9.0

# Components build and install directly into the system DeepStream lib directory
LIB_DIR ?= /opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib
OBJ_DIR ?= $(CURDIR)/.build

# SOURCE_DIR points to repo root
SOURCE_DIR ?= $(_REPO_ROOT)
