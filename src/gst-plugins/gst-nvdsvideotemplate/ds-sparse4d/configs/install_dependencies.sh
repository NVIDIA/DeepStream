#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Sparse4D build dependencies: OpenCV, nlohmann-json, Eigen, and LibTorch.
# An explicit TORCH_PATH is a preprovisioned CUDA ABI1 provider, not a pip target.
# When unset, preserve the existing PyTorch 2.9.0 (cu130) default installation.
# Invoked from scripts/install_opensource_deps.sh.

set -e
export DEBIAN_FRONTEND=noninteractive

# Reject an invalid explicit provider before APT, removal, or pip. These are the
# same structural ABI/architecture requirements as the Sparse4D Makefile; they
# do not establish vendor provenance or target runtime compatibility.
if [[ -v TORCH_PATH ]]; then
  if [[ -z "$TORCH_PATH" || ! -r "$TORCH_PATH/include/torch/csrc/api/include/torch/torch.h" ]]; then
    echo "error: explicit TORCH_PATH must contain a CUDA LibTorch distribution." >&2
    exit 1
  fi
  TORCH_DETECTED_ABI=$(sed -n 's/.*_GLIBCXX_USE_CXX11_ABI=\([01]\).*/\1/p' \
    "$TORCH_PATH/share/cmake/Torch/TorchConfig.cmake" 2>/dev/null | head -1)
  if [[ -z "$TORCH_DETECTED_ABI" ]]; then
    TORCH_DETECTED_ABI=$(nm -D --defined-only "$TORCH_PATH/lib/libc10.so" 2>/dev/null |
      awk '/torchCheckFail.*__cxx11/ {print 1; exit} /torchCheckFail.*RKSs/ {print 0; exit}')
  fi
  if [[ "$TORCH_DETECTED_ABI" != 1 ]]; then
    echo "error: explicit TORCH_PATH requires independently verified ABI1 LibTorch; ABI0 or unknown is unsupported." >&2
    exit 1
  fi
  case "$(uname -m)" in
    x86_64) TORCH_ELF_MACHINE='Advanced Micro Devices X86-64' ;;
    aarch64) TORCH_ELF_MACHINE=AArch64 ;;
    *) echo 'error: unsupported LibTorch provider architecture.' >&2; exit 1 ;;
  esac
  for name in torch torch_cpu torch_cuda c10 c10_cuda; do
    if ! TORCH_ELF_HEADER=$(LC_ALL=C readelf -h "$TORCH_PATH/lib/lib$name.so" 2>/dev/null) ||
        ! grep -Fq "Machine:" <<< "$TORCH_ELF_HEADER" ||
        ! grep -Fq "$TORCH_ELF_MACHINE" <<< "$TORCH_ELF_HEADER"; then
      echo "error: explicit TORCH_PATH needs target-matching CUDA library lib$name.so." >&2
      exit 1
    fi
  done
  echo "Using explicit CUDA LibTorch provider at $TORCH_PATH (no pip or Torch removal)."
fi

echo "Installing Sparse4D apt dependencies"
apt-get update
apt-get install -y libopencv-dev libeigen3-dev nlohmann-json3-dev python3-pip
dpkg --configure -a

[[ ! -v TORCH_PATH ]] || exit 0

# Makefile requires: /usr/local/lib/python3/dist-packages/torch
TORCH_INSTALL_PATH="/usr/local/lib/python3/dist-packages"
TORCH_VER_FILE="${TORCH_INSTALL_PATH}/torch/version.py"

if [[ -f "$TORCH_VER_FILE" ]] && grep -q '2\.9\.0' "$TORCH_VER_FILE"; then
  echo "PyTorch 2.9.0 already present at ${TORCH_INSTALL_PATH}"
else
  # --target does not overwrite an existing tree; drop only torch, not nvidia-*
  echo "Removing existing PyTorch under ${TORCH_INSTALL_PATH}..."
  rm -rf "${TORCH_INSTALL_PATH}/torch" "${TORCH_INSTALL_PATH}"/torch-*

  echo "Installing PyTorch 2.9.0 to ${TORCH_INSTALL_PATH}..."
  python3 -m pip install torch==2.9.0 \
      --index-url https://download.pytorch.org/whl/cu130 \
      --target="${TORCH_INSTALL_PATH}"
fi

echo "Verifying PyTorch version..."
grep __version__ "${TORCH_VER_FILE}"
