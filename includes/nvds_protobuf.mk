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

# Shared provider rules for DeepStream's protobuf consumers. The caller keeps
# its public compiler/library overrides and sets the selected values below.
NVDS_PROTOBUF_MAKEFILE := $(lastword $(MAKEFILE_LIST))
TRITON_GRPC_CLIENT ?= /opt/tritonclient
PROTOBUF_BIN_DIR ?= /opt/proto/bin
NVDS_PROTOBUF_LEGACY_CXX_STANDARD ?= 17
NVDS_PROTOBUF_HEADER_VERSION := $(shell awk '$$2 == "GOOGLE_PROTOBUF_VERSION" && $$3 ~ /^[0-9]+$$/ { print $$3; exit }' "$(TRITON_GRPC_CLIENT)/include/google/protobuf/stubs/common.h" 2>/dev/null)
_NVDS_PROTOBUF_HEADER_MIN_CXX_STANDARD := $(shell awk 'BEGIN { v = "$(NVDS_PROTOBUF_HEADER_VERSION)" + 0; print (v >= 6030000 ? 17 : (v >= 4022000 ? 14 : 11)) }')
NVDS_PROTOBUF_MIN_CXX_STANDARD := $(shell printf '%s\n' "$(NVDS_PROTOBUF_LEGACY_CXX_STANDARD)" "$(_NVDS_PROTOBUF_HEADER_MIN_CXX_STANDARD)" | sort -n | tail -n 1)
ifneq ($(shell grep -E '^[[:space:]]*\#define[[:space:]]+ABSL_OPTION_USE_STD_ORDERING[[:space:]]+1([[:space:]]|$$)' "$(TRITON_GRPC_CLIENT)/include/absl/base/options.h" 2>/dev/null),)
NVDS_PROTOBUF_MIN_CXX_STANDARD := 20
endif

# A partial lib/ must not hide a complete lib64/. Shared clients contain their
# gRPC implementation; static-only providers still need the legacy archives.
nvds_protobuf_library_dir_is_complete = $(and \
    $(wildcard $(1)/libprotobuf.a),$(wildcard $(1)/libcares.a),\
    $(or $(wildcard $(1)/libgrpcclient.so),\
        $(and $(wildcard $(1)/libgrpcclient.a),\
            $(wildcard $(1)/libgrpcclient_static.a),$(wildcard $(1)/libgrpc++.a))))
NVDS_PROTOBUF_LIB_DIR ?= $(strip $(or $(firstword \
    $(foreach dir,$(TRITON_GRPC_CLIENT)/lib $(TRITON_GRPC_CLIENT)/lib64,\
        $(if $(call nvds_protobuf_library_dir_is_complete,$(dir)),$(dir)))),\
    $(TRITON_GRPC_CLIENT)/lib))
nvds_protobuf_support_libraries = $(addprefix -l:,$(notdir $(sort $(wildcard $(1)/libabsl*.a $(1)/libutf8_range.a $(1)/libutf8_validity.a))))

# A fresh checkout has no stamp, so checked-in code is regenerated with the
# selected provider. Include platform selections so switching between Jetson
# and SBSA cannot reuse objects compiled with different CUDA-sharing flags.
# Compare contents to preserve timestamps for unchanged builds. .build/ is
# ignored under src/.
NVDS_PROTOBUF_PROVIDER_STAMP := .build/protobuf-provider
NVDS_PROTOBUF_SELECTED_CXX_STANDARD ?= $(NVDS_PROTOBUF_MIN_CXX_STANDARD)
NVDS_PROTOBUF_SELECTED_LIB_DIR ?= $(NVDS_PROTOBUF_LIB_DIR)
ifeq ($(.DEFAULT_GOAL),)
.DEFAULT_GOAL := all
endif
.PHONY: nvds-protobuf-provider-force
nvds-protobuf-provider-force:

$(NVDS_PROTOBUF_PROVIDER_STAMP): nvds-protobuf-provider-force
	@set -eu; \
	  version="$$('$(PROTOBUF_BIN_DIR)/protoc' --version)"; \
	  code='$(NVDS_PROTOBUF_HEADER_VERSION)'; \
	  case "$$code" in ''|*[!0-9]*) echo 'Cannot identify protobuf headers in TRITON_GRPC_CLIENT.' >&2; exit 1 ;; esac; \
	  major=$$((code / 1000000)); minor=$$((code / 1000 % 1000)); patch=$$((code % 1000)); \
	  if test "$$major" -ge 4; then expected="$$minor.$$patch"; else expected="$$major.$$minor.$$patch"; fi; \
	  if test "$$version" != "libprotoc $$expected"; then \
	    echo "Selected SDK protobuf headers require protoc $$expected; found $$version. Set PROTOBUF_BIN_DIR to the matching compiler." >&2; exit 1; \
	  fi; \
	  mkdir -p '$(@D)'; \
	  temporary='$@.'$$$$'.tmp'; \
	  trap 'rm -f "$$temporary"' EXIT HUP INT TERM; \
	  printf '%s\n' '$(TRITON_GRPC_CLIENT)' '$(PROTOBUF_BIN_DIR)' "$$version" "$$code" \
	    '$(NVDS_PROTOBUF_SELECTED_CXX_STANDARD)' '$(NVDS_PROTOBUF_SELECTED_LIB_DIR)' \
	    '$(shell uname -m)' '$(CUDA_VER)' '$(TRT_VER)' \
	    '$(DEEPSTREAM_VARIANT)' '$(AARCH64_IS_SBSA)' > "$$temporary"; \
	  if ! cmp -s "$$temporary" '$@'; then mv -f "$$temporary" '$@'; fi
