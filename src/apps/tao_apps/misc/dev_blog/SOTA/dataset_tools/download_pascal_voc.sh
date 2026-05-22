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

# download
cd ~/tao-experiments/data
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
tar xvf VOCtrainval_06-Nov-2007.tar
tar xvf VOCtrainval_11-May-2012.tar
tar xvf VOCtest_06-Nov-2007.tar
# Splitting datasets
mkdir -p ~/tao-experiments/data/voc0712trainval/images
mkdir -p ~/tao-experiments/data/voc0712trainval/labels
mkdir -p ~/tao-experiments/data/voc07test/images
mkdir -p ~/tao-experiments/data/voc07test/labels
cat ~/tao-experiments/data/VOCdevkit/VOC2007/ImageSets/Main/test.txt | xargs -I'{}' mv -t ~/tao-experiments/data/voc07test/images ~/tao-experiments/data/VOCdevkit/VOC2007/JPEGImages/'{}'.jpg
cat ~/tao-experiments/data/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt | xargs -I'{}' mv -t ~/tao-experiments/data/voc0712trainval/images ~/tao-experiments/data/VOCdevkit/VOC2007/JPEGImages/'{}'.jpg
cat ~/tao-experiments/data/VOCdevkit/VOC2012/ImageSets/Main/trainval.txt | xargs -I'{}' mv -t ~/tao-experiments/data/voc0712trainval/images ~/tao-experiments/data/VOCdevkit/VOC2012/JPEGImages/'{}'.jpg
cat ~/tao-experiments/data/VOCdevkit/VOC2007/ImageSets/Main/test.txt | xargs -I'{}' mv -t ~/tao-experiments/data/voc07test/labels ~/tao-experiments/data/VOCdevkit/VOC2007/Annotations/'{}'.xml
cat ~/tao-experiments/data/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt | xargs -I'{}' mv -t ~/tao-experiments/data/voc0712trainval/labels ~/tao-experiments/data/VOCdevkit/VOC2007/Annotations/'{}'.xml
cat ~/tao-experiments/data/VOCdevkit/VOC2012/ImageSets/Main/trainval.txt | xargs -I'{}' mv -t ~/tao-experiments/data/voc0712trainval/labels ~/tao-experiments/data/VOCdevkit/VOC2012/Annotations/'{}'.xml

