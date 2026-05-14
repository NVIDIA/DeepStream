################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
################################################################################

# Delegates to the nvdsinferserver custom-process library so the mono-repo
# build.sh sample_apps loop picks this sample up. The Triton C++ custom
# backend under nvds_triton_cpp_custom_backend/ is built separately via cmake
# (see README) and is not wired into build.sh.

SUBDIRS := nvdsinferserver_custom_impl_ensemble

.PHONY: all clean $(SUBDIRS)

all: $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@

clean:
	$(MAKE) -C nvdsinferserver_custom_impl_ensemble clean
