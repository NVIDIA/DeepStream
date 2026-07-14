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

make

## enable NVSTREAMMUX_ADAPTIVE_BATCHING
if [[ "$NVSTREAMMUX_ADAPTIVE_BATCHING" != "yes" ]]; then
  export NVSTREAMMUX_ADAPTIVE_BATCHING=yes
  rm -rf ~/.cache/gstreamer-1.0/
  echo "export NVSTREAMMUX_ADAPTIVE_BATCHING=yes"
fi

## dict.txt is label file for LPR model
if [[ ! -f dict.txt ]]; then
  wget 'https://api.ngc.nvidia.com/v2/models/nvidia/tao/lprnet/versions/deployable_v1.0/files/us_lp_characters.txt' -O dict.txt
fi
