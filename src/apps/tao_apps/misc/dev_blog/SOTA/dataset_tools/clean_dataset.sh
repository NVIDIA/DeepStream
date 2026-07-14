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

#!/bin/bash
for img in $1/*.jpg
do
kitti_name=$2/$(basename $img .jpg).txt
if [[ ! -f "$kitti_name" ]]; then
echo "$kitti_name not exist, will remove the pair."
rm $img
fi
done

for kit in $2/*.txt
do
img_name=$1/$(basename $kit .txt).jpg
if [[ ! -f "$img_name" ]]; then
echo "$img_name not exist, will remove the pair."
rm $kit
fi
done

echo "$1: $(ls $1|wc -l)"
echo "$2: $(ls $2|wc -l)"
echo Done!

