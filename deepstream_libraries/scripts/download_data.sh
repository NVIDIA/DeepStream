#!/bin/bash -e

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Move to parent directory
if [ ! -d "assets" ]; then
    echo "Moving to parent directory."
    cd ..
fi
## CVCUDA Image Samples
wget -P assets/images/ https://github.com/CVCUDA/CV-CUDA/raw/main/samples/assets/images/Weimaraner.jpg
wget -P assets/images/ https://github.com/CVCUDA/CV-CUDA/raw/main/samples/assets/images/peoplenet.jpg
wget -P assets/images/ https://github.com/CVCUDA/CV-CUDA/raw/main/samples/assets/images/tabby_tiger_cat.jpg

## CVCUDA Video Samples
wget -P assets/videos/ https://github.com/CVCUDA/CV-CUDA/raw/main/samples/assets/videos/pexels-chiel-slotman-4423925-1920x1080-25fps.mp4
wget -P assets/videos/ https://github.com/CVCUDA/CV-CUDA/raw/main/samples/assets/videos/pexels-ilimdar-avgezer-7081456.mp4
