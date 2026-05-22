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

.PHONY: all install clean

all:
	$(MAKE) -C tao_detection
	$(MAKE) -C tao_segmentation
	$(MAKE) -C tao_classifier
	$(MAKE) -C tao_others

install:
	$(MAKE) -C tao_detection install
	$(MAKE) -C tao_segmentation install

clean:
	$(MAKE) -C tao_detection clean
	$(MAKE) -C tao_segmentation clean
	$(MAKE) -C tao_classifier clean
	$(MAKE) -C tao_others clean

%:
	$(MAKE) -C tao_detection $@
	$(MAKE) -C tao_segmentation $@
	$(MAKE) -C tao_classifier $@
	$(MAKE) -C tao_others $@
