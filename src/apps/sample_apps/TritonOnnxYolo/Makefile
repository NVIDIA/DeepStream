################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
################################################################################

# Delegates to the nvdsinferserver custom-process library so the mono-repo
# build.sh sample_apps loop picks this sample up.

SUBDIRS := nvdsinferserver_custom_impl_yolo

.PHONY: all clean $(SUBDIRS)

all: $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@

clean:
	$(MAKE) -C nvdsinferserver_custom_impl_yolo clean
